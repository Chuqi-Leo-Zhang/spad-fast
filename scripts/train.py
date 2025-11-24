import os
import math
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from omegaconf import OmegaConf

from spad.utils import load_model_from_config
from spad.spad import SPAD  # adjust if your SPAD class is in a different module
from spad.dataloader import ObjaverseMultiViewDataset  # your dataset file


# ------------------------- Camera helpers ------------------------- #

SPAD_FOV = 0.702769935131073  # radians
CAMERA_RADIUS = 3.5           # distance to origin


def build_cam_and_intrinsics_from_radians(elevations, azimuths,
                                          use_abs_extrinsics=False):
    """
    Build 'cam' and 'render_intrinsics_flat' tensors for a single object,
    given per-view spherical coordinates in radians.

    Inputs:
        elevations: list/1D tensor of length V, radians
        azimuths:   list/1D tensor of length V, radians
    Returns:
        cam:                    [V, V, 4]
        render_intrinsics_flat: [V, 4] = [fx, fy, cx, cy]
    """
    elevations = [float(e) for e in elevations]
    azimuths = [float(a) for a in azimuths]
    num_views = len(elevations)

    abs_cams = []
    for theta, az in zip(elevations, azimuths):
        abs_cams.append(torch.tensor([theta, az, CAMERA_RADIUS],
                                     dtype=torch.float32))

    debug_cams = [[] for _ in range(num_views)]
    for i, icam in enumerate(abs_cams):
        for j, jcam in enumerate(abs_cams):
            if use_abs_extrinsics:
                # Absolute encoding: [theta, sin(phi), cos(phi), radius]
                dcam = torch.tensor(
                    [icam[0],
                     math.sin(icam[1]),
                     math.cos(icam[1]),
                     icam[2]],
                    dtype=torch.float32,
                )
            else:
                # Relative encoding: [Δtheta, sin(Δphi), cos(Δphi), Δr]
                diff = icam - jcam
                dcam = torch.tensor(
                    [
                        diff[0].item(),
                        math.sin(diff[1].item()),
                        math.cos(diff[1].item()),
                        diff[2].item(),
                    ],
                    dtype=torch.float32,
                )
            debug_cams[i].append(dcam)

    cam = torch.stack([torch.stack(dc) for dc in debug_cams], dim=0)  # [V,V,4]

    # Build intrinsics K from FOV, then flatten to [fx, fy, cx, cy]
    focal = 1.0 / np.tan(SPAD_FOV / 2.0)
    intrinsics = np.diag(np.array([focal, focal, 1.0], dtype=np.float32))  # [3,3]
    intrinsics = torch.from_numpy(intrinsics).unsqueeze(0).float()         # [1,3,3]
    intrinsics = intrinsics.repeat(num_views, 1, 1)                        # [V,3,3]

    # [V, 4] = [K[0,0], K[1,1], K[0,2], K[1,2]] == [f, f, 0, 0]
    render_intrinsics_flat = intrinsics[:, [0, 1, 0, 1], [0, 1, 2, 2]]

    return cam, render_intrinsics_flat


# ------------------------- LR warmup helper ------------------------- #


def set_lr_with_warmup(optimizer, base_lr, global_step, warmup_steps):
    """
    Linear warmup:
        - step <= warmup_steps: lr = base_lr * (step / warmup_steps)
        - step >  warmup_steps: lr = base_lr  (no decay afterwards)
    """
    if warmup_steps <= 0:
        lr = base_lr
    else:
        if global_step >= warmup_steps:
            lr = base_lr
        else:
            warmup_frac = float(global_step) / float(warmup_steps)
            lr = base_lr * warmup_frac

    for pg in optimizer.param_groups:
        pg["lr"] = lr

    return lr


# ------------------------- Training loop ------------------------- #


def save_checkpoint(model, optimizer, step, outdir, prefix="spad"):
    os.makedirs(outdir, exist_ok=True)
    ckpt_path = os.path.join(outdir, f"{prefix}_step_{step:07d}.ckpt")
    state = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
    }
    torch.save(state, ckpt_path)
    print(f"[checkpoint] saved to {ckpt_path}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----------------- Load config ----------------- #
    config = OmegaConf.load(args.config)

    # Base learning rate from config unless overridden
    if args.lr is None:
        base_lr = float(config.model.base_learning_rate)
        print(f"Using base_learning_rate from config: {base_lr}")
    else:
        base_lr = args.lr
        print(f"Using overridden learning rate: {base_lr}")

    warmup_steps = args.warmup_steps
    print(f"LR warmup steps: {warmup_steps} (then constant LR, no decay)")

    # ----------------- Dataset & DataLoader ----------------- #
    train_dataset = ObjaverseMultiViewDataset(
        root=args.data_root,
        num_views=args.num_views,
        image_size=args.image_size,
        training=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ----------------- Model ----------------- #
    # Note: inference_run=False so model is fully configurable for training
    model: SPAD = load_model_from_config(
        config,
        ckpt=args.ckpt if args.ckpt is not None else None,
        verbose=True,
        inference_run=False,
    )
    model = model.to(device)
    model.train()

    # Initialize EMA if enabled (SPAD uses EMA for better sampling quality)
    if getattr(model, "use_ema", False):
        model.reinit_ema()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=base_lr,                 # this will be immediately scaled by warmup
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    optimizer.zero_grad(set_to_none=True)
    scaler = GradScaler(enabled=args.fp16)

    global_step = 0
    micro_step = 0
    outdir = args.output_dir

    for epoch in range(args.num_epochs):
        print(f"\n===== Epoch {epoch+1}/{args.num_epochs} =====")

        for batch_idx, batch in enumerate(train_loader):
            micro_step += 1

            # Move geometry-related tensors to device
            batch["epi_constraint_masks"] = batch["epi_constraint_masks"].to(device)
            batch["plucker_embeds"] = batch["plucker_embeds"].to(device)

            # Build cam + intrinsics for this batch
            B, V = batch["img"].shape[:2]
            cam_list = []
            intr_list = []

            for b in range(B):
                elev = batch["elevations"][b].tolist()  # radians
                azim = batch["azimuths"][b].tolist()    # radians
                cam_b, intr_b = build_cam_and_intrinsics_from_radians(
                    elev,
                    azim,
                    use_abs_extrinsics=getattr(model, "use_abs_extrinsics", False),
                )
                cam_list.append(cam_b)    # [V, V, 4]
                intr_list.append(intr_b)  # [V, 4]

            batch["cam"] = torch.stack(cam_list, dim=0).to(device)                  # [B,V,V,4]
            batch["render_intrinsics_flat"] = torch.stack(intr_list, dim=0).to(device)  # [B,V,4]
            batch["txt"] = [batch["txt"]] * V  # replicate captions for all views

            with autocast(enabled=args.fp16):
                # get_input returns [z, cond, ...]
                z, cond = model.get_input(batch)[:2]
                # LatentDiffusion.forward returns (loss, loss_dict)
                loss, loss_dict = model(z, cond)

            loss_accum = loss / args.accumulate_steps
            scaler.scale(loss_accum).backward()
            
            if micro_step % args.accumulate_steps == 0:
                global_step += 1
                # Apply LR warmup schedule (after incrementing global_step)
                current_lr = set_lr_with_warmup(
                    optimizer,
                    base_lr=base_lr,
                    global_step=global_step,
                    warmup_steps=warmup_steps,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # Update EMA after optimizer
                if getattr(model, "use_ema", False):
                    model.model_ema(model.model)

                # Logging
                if global_step and global_step % args.log_every == 0:
                    loss_val = float(loss.detach().cpu())
                    log_str = f"[step {global_step}] loss: {loss_val:.4f}, lr: {current_lr:.6e}"
                    if isinstance(loss_dict, dict):
                        parts = []
                        for k, v in loss_dict.items():
                            try:
                                parts.append(f"{k}: {float(v):.4f}")
                            except Exception:
                                continue
                        if parts:
                            log_str += " | " + " ".join(parts)
                    print(log_str)

                # Checkpointing
                if global_step % args.ckpt_every == 0:
                    save_checkpoint(model, optimizer, global_step, outdir)

                if args.max_steps is not None and global_step >= args.max_steps:
                    print(f"Reached max_steps={args.max_steps}, stopping training.")
                    break

        if args.max_steps is not None and global_step >= args.max_steps:
            break

    # Final checkpoint
    if args.save_last:
        save_checkpoint(model, optimizer, global_step, outdir, prefix="spad_last")


# ------------------------- Argument parsing ------------------------- #


def parse_args():
    parser = argparse.ArgumentParser(description="Train SPAD on Objaverse")

    # Data
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root folder of Objaverse multi-view renderings.")
    parser.add_argument("--num_views", type=int, default=2,
                        help="Number of views per object used during training.")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Rendered image resolution (assumed square).")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulate_steps", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)

    # Model / config
    parser.add_argument("--config", type=str, required=True,
                        help="Path to SPAD config .yaml (LatentDiffusion-style).")
    parser.add_argument("--ckpt", type=str, default="data/checkpoints/spad_two_views.ckpt",
                        help="Optional path to initial checkpoint to resume/fine-tune from.")

    # Training hyperparams
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Base learning rate. If None, use config.model.base_learning_rate (1e-4 in SPAD).",
    )
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Linear learning rate warmup steps.")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay (paper uses None).")
    parser.add_argument("--fp16", action="store_true",
                        help="Enable mixed-precision training with autocast + GradScaler.")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Optional max global steps to stop early.")

    # Logging / checkpoints
    parser.add_argument("--output_dir", type=str, default="checkpoints_spad",
                        help="Directory to save checkpoints.")
    parser.add_argument("--log_every", type=int, default=100,
                        help="Log losses every N steps.")
    parser.add_argument("--ckpt_every", type=int, default=10000,
                        help="Save checkpoint every N steps.")
    parser.add_argument("--save_last", action="store_true",
                        help="Save a final 'last' checkpoint at the end of training.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
