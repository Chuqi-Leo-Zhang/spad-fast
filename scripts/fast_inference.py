import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "8"

import torch
import torch.nn as nn
import numpy as np
import math
import imageio
import time
import argparse

from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from einops import rearrange
from itertools import chain, cycle
from tqdm import tqdm

from spad.utils import load_model_from_config, slugify
from spad.geometry import get_batch_from_spherical
from spad.lora import LoRALinear


# ----------------- camera & gaussian utils (same as original) -----------------


def generate_batch(elevations=[45, 45, 45, 45],
                   azimuths=[0, 90, 180, 270],
                   use_abs=False):
    elevations = [math.radians(e) for e in elevations]
    azimuths = [math.radians(a) for a in azimuths]

    batch = get_batch_from_spherical(elevations, azimuths)

    abs_cams = []
    for theta, azimuth in zip(elevations, azimuths):
        abs_cams.append(torch.tensor([theta, azimuth, 3.5]))

    debug_cams = [[] for _ in range(len(azimuths))]
    for i, icam in enumerate(abs_cams):
        for j, jcam in enumerate(abs_cams):
            if use_abs:
                dcam = torch.tensor([
                    icam[0],
                    math.sin(icam[1]),
                    math.cos(icam[1]),
                    icam[2],
                ])
            else:
                dcam = icam - jcam
                dcam = torch.tensor([
                    dcam[0].item(),
                    math.sin(dcam[1].item()),
                    math.cos(dcam[1].item()),
                    dcam[2].item(),
                ])
            debug_cams[i].append(dcam)

    batch["cam"] = torch.stack([torch.stack(dc) for dc in debug_cams])  # [V,V,4]

    # intrinsics
    focal = 1 / np.tan(0.702769935131073 / 2)
    intrinsics = np.diag(np.array([focal, focal, 1])).astype(np.float32)
    intrinsics = torch.from_numpy(intrinsics).unsqueeze(0).float()
    intrinsics = intrinsics.repeat(batch["cam"].shape[0], 1, 1)
    batch["render_intrinsics_flat"] = intrinsics[:, [0, 1, 0, 1], [0, 1, -1, -1]]

    return batch


def get_gaussian_image(blob_width=256, blob_height=256, sigma=0.5):
    X = np.linspace(-1, 1, blob_width)[None, :]
    Y = np.linspace(-1, 1, blob_height)[:, None]
    inv_dev = 1 / sigma ** 2
    gaussian_blob = np.exp(-0.5 * (X ** 2) * inv_dev) * np.exp(-0.5 * (Y ** 2) * inv_dev)
    if gaussian_blob.max() > 0:
        gaussian_blob = 255.0 * (gaussian_blob - gaussian_blob.min()) / gaussian_blob.max()
    gaussian_blob = 255.0 - gaussian_blob

    gaussian_blob = (gaussian_blob / 255.0) * 2.0 - 1.0
    gaussian_blob = np.expand_dims(gaussian_blob, axis=-1).repeat(3, -1)
    gaussian_blob = torch.from_numpy(gaussian_blob)
    return gaussian_blob


def load_captions(path="data/1k_captions_viz.npy"):
    captions = np.load(path, allow_pickle=True).tolist()
    captions = ["[tdv] " + c if "[tdv]" not in c else c for c in captions]
    return captions


# ----------------- LoRA injection helpers (same as training) -----------------


def _is_class(module: nn.Module, name: str) -> bool:
    return module.__class__.__name__ == name


def insert_lora_layers(
    unet: nn.Module,
    r: int = 4,
    alpha: float = 1.0,
    enable_attn: bool = True,
    enable_mlp: bool = True,
):
    """
    Add LoRA to SPAD UNet in-place.

    - SPADAttention: to_q, to_k, to_v (Linear), to_out[0] (Linear)
    - FeedForward:   net = Sequential(GEGLU(proj=Linear), Dropout, Linear)
    """
    for module in unet.modules():
        # 1) Attention: q, k, v, out
        if enable_attn and _is_class(module, "SPADAttention"):
            if isinstance(module.to_q, nn.Linear) and not isinstance(module.to_q, LoRALinear):
                module.to_q = LoRALinear(module.to_q, r=r, alpha=alpha)
            if isinstance(module.to_k, nn.Linear) and not isinstance(module.to_k, LoRALinear):
                module.to_k = LoRALinear(module.to_k, r=r, alpha=alpha)
            if isinstance(module.to_v, nn.Linear) and not isinstance(module.to_v, LoRALinear):
                module.to_v = LoRALinear(module.to_v, r=r, alpha=alpha)

            if isinstance(module.to_out, nn.Sequential):
                if (
                    len(module.to_out) > 0
                    and isinstance(module.to_out[0], nn.Linear)
                    and not isinstance(module.to_out[0], LoRALinear)
                ):
                    module.to_out[0] = LoRALinear(module.to_out[0], r=r, alpha=alpha)

        # 2) FeedForward MLP
        if enable_mlp and _is_class(module, "FeedForward"):
            net = module.net
            if len(net) > 0 and hasattr(net[0], "proj"):
                proj = net[0].proj
                if isinstance(proj, nn.Linear) and not isinstance(proj, LoRALinear):
                    net[0].proj = LoRALinear(proj, r=r, alpha=alpha)
            if len(net) > 2 and isinstance(net[2], nn.Linear) and not isinstance(net[2], LoRALinear):
                net[2] = LoRALinear(net[2], r=r, alpha=alpha)


# ----------------- schedule helpers (same as training) -----------------


def build_train_time_schedule(model, num_train_timesteps: int, device: str):
    """
    Build reduced grid (indices -> ᾱ_t, σ_t) from model's DDPM schedule.
    """
    assert hasattr(model, "num_timesteps")
    assert hasattr(model, "alphas_cumprod")

    num_ddpm = int(model.num_timesteps)
    ddpm_indices = torch.linspace(
        0, num_ddpm - 1, num_train_timesteps, dtype=torch.long, device=device
    )

    alphas_cumprod = model.alphas_cumprod.to(device)
    alpha_bar_train = alphas_cumprod[ddpm_indices]

    sqrt_alpha_bar_train = torch.sqrt(alpha_bar_train)
    sqrt_one_minus_alpha_bar_train = torch.sqrt(1.0 - alpha_bar_train)

    return ddpm_indices, sqrt_alpha_bar_train, sqrt_one_minus_alpha_bar_train


def _expand_scalar_for(x_like: torch.Tensor, scalar: torch.Tensor):
    while scalar.dim() < x_like.dim():
        scalar = scalar.view(*scalar.shape, 1)
    return scalar


# ----------------- load SPAD + EMA LoRA student -----------------


def load_spad_student_from_ema(
    config_path: str,
    teacher_ckpt: str,
    ema_ckpt_path: str,
    device: str = "cuda",
    lora_rank: int = 16,
    lora_alpha: float = 16.0,
    train_timesteps: int = 1000,
):
    """
    Load SPAD base model, inject LoRA, then load EMA LoRA weights.
    Returns:
        model, ddpm_indices, alphas_train, sigmas_train
    """
    model_config = OmegaConf.load(config_path)
    model = load_model_from_config(
        model_config,
        teacher_ckpt,
        verbose=True,
        inference_run=True,
    )
    model.to(device)

    # inject LoRA into the UNet (same as training)
    insert_lora_layers(
        model.model.diffusion_model,
        r=lora_rank,
        alpha=lora_alpha,
    )

    # load EMA LoRA weights
    ckpt = torch.load(ema_ckpt_path, map_location="cpu")
    ema_lora = ckpt["ema_lora"]

    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in ema_lora:
                p.copy_(ema_lora[name].to(p.device))

    model.eval()

    # schedule for LCM sampling
    ddpm_indices, alphas_train, sigmas_train = build_train_time_schedule(
        model,
        num_train_timesteps=train_timesteps,
        device=device,
    )

    return model, ddpm_indices, alphas_train, sigmas_train


# ----------------- LCM 4-step sampler -----------------


@torch.no_grad()
def lcm_spad_sample(
    model,
    ddpm_indices: torch.Tensor,
    alphas_train: torch.Tensor,
    sigmas_train: torch.Tensor,
    shape,
    cond,
    uc,
    num_sampling_steps: int = 4,
    omega: float = 7.5,
    device: str = "cuda",
):
    """
    LCM-style multi-step sampling:
    - start from z_T ~ N(0, I)
    - step t_T -> ... -> t_0

    shape: latent shape, e.g. [B,V,C,H,W] or [B,C,H,W]
    """
    device = torch.device(device)
    ddpm_indices = ddpm_indices.to(device)
    alphas_train = alphas_train.to(device)
    sigmas_train = sigmas_train.to(device)

    num_train = len(alphas_train)
    z = torch.randn(shape, device=device)

    # time indices from large -> small on training grid
    t_indices = torch.linspace(
        num_train - 1,
        0,
        steps=num_sampling_steps,
        dtype=torch.long,
        device=device,
    )

    for i in range(num_sampling_steps - 1):
        t_idx = int(t_indices[i].item())
        t_next_idx = int(t_indices[i + 1].item())

        ddpm_t = int(ddpm_indices[t_idx].item())

        # build t_batch to match z shape
        if z.dim() == 5:
            n, v = z.shape[:2]
            t_batch = torch.full((n, v), ddpm_t, device=device, dtype=torch.long)
        else:
            b = z.shape[0]
            t_batch = torch.full((b,), ddpm_t, device=device, dtype=torch.long)

        # eps_cond / eps_uncond (same as training)
        eps_c = model.apply_model(z, t_batch, cond)
        eps_u = model.apply_model(z, t_batch, uc)
        eps = (1.0 + omega) * eps_c - omega * eps_u

        alpha_t = _expand_scalar_for(z, alphas_train[t_idx])
        sigma_t = _expand_scalar_for(z, sigmas_train[t_idx])

        x0_pred = (z - sigma_t * eps) / alpha_t

        alpha_next = _expand_scalar_for(z, alphas_train[t_next_idx])
        sigma_next = _expand_scalar_for(z, sigmas_train[t_next_idx])

        z = alpha_next * x0_pred + sigma_next * eps

    return z  # approximately clean z_0 at t=0


# ----------------- denoise wrapper using LCM student -----------------


def denoise_lcm(
    batch,
    model,
    device,
    idx,
    total_views,
    outpath,
    blob_sigma,
    ddpm_indices,
    alphas_train,
    sigmas_train,
    num_sampling_steps,
    omega,
):
    # move batch tensors to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # gaussian blob init (only used to build cond/uc via get_input)
    blob = get_gaussian_image(sigma=blob_sigma)
    batch["img"][:, :] = blob
    print("using gaussian initialization for input images")

    # get latent shape + cond / uc as in training (LCM distillation)
    z0, cond, uc = model.get_input(
        batch,
        return_first_stage_outputs=False,
        return_original_cond=False,
        return_uc=True,
    )
    z0 = z0.to(device)
    latent_shape = z0.shape

    # sampling with LCM 4-step student
    samples = lcm_spad_sample(
        model,
        ddpm_indices,
        alphas_train,
        sigmas_train,
        shape=latent_shape,
        cond=cond,
        uc=uc,
        num_sampling_steps=num_sampling_steps,
        omega=omega,
        device=device,
    )

    x_samples_cfg = model.decode_first_stage(samples)
    x_samples_cfg = torch.clamp(x_samples_cfg, -1., 1.)

    # flatten images and captions (same as original)
    x_samples = rearrange(x_samples_cfg, "b v c h w -> (b v) c h w")
    x_samples = ((x_samples + 1.0) / 2.0)
    xtxt = np.array(batch["txt"]).T.tolist()
    xtxt = list(chain(*xtxt))
    x_samples = rearrange(x_samples, "(n v) c h w -> n h (v w) c", v=total_views)
    x_samples = (x_samples * 255.0).cpu().float().numpy().astype(np.uint8)

    os.makedirs(outpath, exist_ok=True)
    for _idx, (image, caption) in enumerate(zip(x_samples, xtxt)):
        caption = slugify(caption)
        save_path = f"{outpath}/{caption}.png"
        imageio.imsave(save_path, image)
        print(f"saved image: {save_path}")

    return {}


# ----------------- main -----------------


def main(
    config_path,
    teacher_ckpt,
    ema_ckpt,
    captions,
    cfg_scale=7.5,
    blob_sigma=0.5,
    batch_size=1,
    total_views=8,
    num_sampling_steps=4,
    train_timesteps=1000,
):
    """
    cfg_scale here is used as ω (CFG scale) for LCM.
    """
    seed_everything(42 + 69)

    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    # breakpoint()

    # load SPAD + EMA LoRA student
    model, ddpm_indices, alphas_train, sigmas_train = load_spad_student_from_ema(
        config_path,
        teacher_ckpt,
        ema_ckpt,
        device=device,
        lora_rank=16,      # must match distillation
        lora_alpha=16.0,   # must match distillation
        train_timesteps=train_timesteps,
    )

    # logging dirs
    visuals_dir = "data/visuals/"
    ts = str(round(time.time()))
    outdir = os.path.join(visuals_dir, "inference_lcm", ts)
    os.makedirs(outdir, exist_ok=True)

    dataloader = cycle([{
        "img": torch.zeros(batch_size, total_views, 256, 256, 3)
    }])

    terminate = False
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="LCM sampling")):
            if batch_size * (idx + 1) >= len(captions):
                batch_size = len(captions) - batch_size * idx
                terminate = True

            elevations = [45 for _ in range(total_views)]
            azimuths = [az for az in np.linspace(0, 360 * ((total_views - 1) / total_views), total_views)]
            print(f"using elevations: {elevations}, azimuths: {azimuths}")

            batch_cams = generate_batch(elevations, azimuths, use_abs=getattr(model, "use_abs_extrinsics", False))
            batch_cams = {k: v[None].repeat_interleave(batch_size, dim=0).to(device)
                          for k, v in batch_cams.items()}
            batch.update(batch_cams)

            # captions for this mini-batch
            batch["txt"] = [captions[batch_size * idx: batch_size * (idx + 1)]] * total_views

            # no EMA scope here; model already uses EMA LoRA weights
            denoise_lcm(
                batch,
                model,
                device,
                idx,
                total_views,
                outdir,
                blob_sigma,
                ddpm_indices,
                alphas_train,
                sigmas_train,
                num_sampling_steps=num_sampling_steps,
                omega=cfg_scale,
            )

            if terminate:
                break


if __name__ == "__main__":
    # same model zoo style as original
    model_zoo = {
        "spad_four_views": ("configs/spad_four_views.yaml", "data/checkpoints/spad_four_views.ckpt"),
        "spad_two_views": ("configs/spad_two_views.yaml", "data/checkpoints/spad_two_views.ckpt"),
    }

    parser = argparse.ArgumentParser("LCM-LoRA 4-step inference for SPAD")
    parser.add_argument("--captions", type=str, default=None,
                        help="caption string or list; if None, uses default captions_eval.npy")
    parser.add_argument("--model", type=str, default="spad_two_views")
    parser.add_argument("--ema_ckpt", type=str, required=True,
                        help="Path to EMA LoRA checkpoint (e.g., logs/spad_lcm_lora/last.pt)")
    parser.add_argument("--cfg_scale", type=float, default=7.5,
                        help="LCM CFG scale ω (should be within [w_min, w_max] used in distillation)")
    parser.add_argument("--blob_sigma", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--total_views", type=int, default=8)
    parser.add_argument("--num_sampling_steps", type=int, default=4,
                        help="Number of LCM steps (target of distillation)")
    parser.add_argument("--train_timesteps", type=int, default=1000,
                        help="Reduced training grid T used in distillation (must match training script)")
    args = parser.parse_args()

    if args.captions is not None:
        captions = eval(f'"{args.captions}"')
        captions = [captions] if isinstance(captions, str) else captions
        captions = ["[tdv] " + c if "[tdv]" not in c else c for c in captions]
    else:
        captions = load_captions("data/captions_eval.npy")

    print(f"num of captions: {len(captions)}, batch_size: {args.batch_size}")

    config_path, teacher_ckpt = model_zoo[args.model]
    main(
        config_path=config_path,
        teacher_ckpt=teacher_ckpt,
        ema_ckpt=args.ema_ckpt,
        captions=captions,
        cfg_scale=args.cfg_scale,
        blob_sigma=args.blob_sigma,
        batch_size=args.batch_size,
        total_views=args.total_views,
        num_sampling_steps=args.num_sampling_steps,
        train_timesteps=args.train_timesteps,
    )
