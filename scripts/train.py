# train_spad_lora_distill.py

import os
import math
import argparse
from itertools import cycle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from einops import rearrange
from tqdm import tqdm

from spad.utils import load_model_from_config
from spad.spad import SPAD

from spad.dataloader import ObjaverseMultiViewDataset


# ------------------------------------------------------------
# 1. 一些小工具函数
# ------------------------------------------------------------

def freeze_base_weights(model):
    """
    冻结所有非 LoRA 参数。
    假设 LoRA 参数在名字里带 "lora_" / "lora_down" / "lora_up" 之类。
    """
    for name, p in model.named_parameters():
        if any(k in name for k in ["lora_", "lora_down", "lora_up"]):
            p.requires_grad = True
        else:
            p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] trainable params: {trainable/1e6:.2f}M / total {total/1e6:.2f}M")


def get_lora_params(model):
    params = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            params.append(p)
    return params


# ------------------------------------------------------------
# 2. 蒸馏训练 step：用 teacher 的 epsilon 监督 student
# ------------------------------------------------------------

def distill_step(student: SPAD,
                 teacher: SPAD,
                 batch,
                 device,
                 noise_schedule="uniform"):
    """
    一个 training step:
      1) 用 student.get_input 得到 z, cond
      2) 采 t, eps, 加噪得到 z_t
      3) teacher / student 分别预测 eps
      4) MSE loss
    """

    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # student / teacher 用同一个 get_input（结构完全一致）
    # 你可以像 inference 那样要更多输出，这里只要 z 和 cond
    z, cond = student.get_input(batch, return_first_stage_outputs=False, return_uc=False)

    # z: [B, V, C, H, W]  or  [B*V, C, H, W] 取决于 SPAD 实现
    if z.dim() == 5:
        B, V, C, H, W = z.shape
        z = rearrange(z, "b v c h w -> (b v) c h w")
    else:
        Bv, C, H, W = z.shape
        B = Bv   # 这里只是名字，真正 batch_size 看需要

    # 获取 diffusion 的时间步数等信息（来自 LatentDiffusion）
    num_timesteps = student.num_timesteps

    # 采 t（这里简单 uniform，你之后可以改成 LCM / DMD 用到的特定采样）
    t = torch.randint(
        low=0,
        high=num_timesteps,
        size=(z.shape[0],),
        device=device,
        dtype=torch.long,
    )

    # 采高斯噪声
    noise = torch.randn_like(z)

    # 用 LatentDiffusion 自带的 q_sample 把 z0 加噪得到 z_t
    # 注意：SPAD 继承自 LatentDiffusion，所以应当有这个函数
    z_t = student.q_sample(z, t, noise)

    # teacher / student 都在 z_t, t, cond 上跑 UNet
    # apply_model 是 latent diffusion 的 “UNet 前向”
    eps_teacher = teacher.apply_model(z_t, t, cond)
    eps_student = student.apply_model(z_t, t, cond)

    # 蒸馏 loss: 让 student 拟合 teacher 的 epsilon
    loss = F.mse_loss(eps_student, eps_teacher)

    return loss


# ------------------------------------------------------------
# 3. 训练主函数
# ------------------------------------------------------------

def train(args):
    seed_everything(args.seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # ---------- 3.1 加载 config & teacher / student ----------
    cfg = OmegaConf.load(args.config)

    print("[Load teacher]")
    teacher = load_model_from_config(cfg, args.teacher_ckpt, verbose=True, inference_run=False)
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print("[Load student (LoRA)]")
    # student 初始可以先加载同一个 ckpt，再在外面插 LoRA 模块（你已经改过 mv_unet / attention / GEGLU）
    student = load_model_from_config(cfg, args.teacher_ckpt, verbose=True, inference_run=False)
    student = student.to(device).train()

    # 冻结 base 权重，只保留 LoRA
    freeze_base_weights(student)

    # ---------- 3.2 准备数据 ----------
    dataset = ObjaverseMultiViewDataset(
        root=args.data_root,
        num_views=args.num_views,
        image_size=args.image_size,
        training=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # 循环 dataloader（如果想无限循环训练 steps，可以用 cycle）
    data_iter = cycle(dataloader)

    # ---------- 3.3 优化器 ----------
    lora_params = get_lora_params(student)
    optimizer = torch.optim.AdamW(
        lora_params, lr=args.lr, weight_decay=args.weight_decay
    )

    # 简单的线性 warmup + cosine decay 你可以之后再加；现在先固定 lr
    global_step = 0

    # ---------- 3.4 训练 loop ----------
    os.makedirs(args.output_dir, exist_ok=True)

    for step in tqdm(range(args.train_steps), desc="training"):
        batch = next(data_iter)

        optimizer.zero_grad()

        loss = distill_step(student, teacher, batch, device)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)

        optimizer.step()

        global_step += 1

        if step % args.log_every == 0:
            print(f"[step {step}] loss = {loss.item():.6f}")

        # 定期保存 LoRA 权重
        if step % args.save_every == 0 and step > 0:
            ckpt_path = os.path.join(args.output_dir, f"spad_lora_step{step}.pt")
            # 只保存 LoRA 参数比较小
            lora_state = {
                k: v.cpu()
                for k, v in student.state_dict().items()
                if v.requires_grad
            }
            torch.save(lora_state, ckpt_path)
            print(f"saved LoRA checkpoint to {ckpt_path}")


# ------------------------------------------------------------
# 4. argument parser
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 模型 / 数据
    parser.add_argument("--config", type=str, default="configs/spad_two_views.yaml")
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)

    # 训练超参
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--train_steps", type=int, default=100000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=2000)

    parser.add_argument("--output_dir", type=str, default="checkpoints/spad_lora_distill")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train(args)
