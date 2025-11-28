# distill_lcm_lora_spad.py
#
# LCM-LoRA style consistency distillation for SPAD
# Target: 4-step sampler
#
# Algorithm 1 is adapted from LCM / LCM-LoRA:
# z_{t_{n+k}} ~ N(α(t_{n+k}) z, σ^2(t_{n+k}) I)
# z_hat_{Ψ, ω, t_n} = z_{t_{n+k}} + (1+ω) Ψ(z_{t_{n+k}}, t_{n+k}, t_n, c)
#                                   - ω Ψ(z_{t_{n+k}}, t_{n+k}, t_n, ∅)
# L = || f_θ(z_{t_{n+k}}, ω, c, t_{n+k})
#       - f_θ(z_hat_{Ψ, ω, t_n}, ω, c, t_n) ||^2
#
# Where f_θ is the student (SPAD + LoRA) and Ψ is an ODE step
# implemented here via a deterministic DDIM update with the teacher.

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "8"
import math
import copy
import argparse
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from einops import rearrange

# ---- your repo specific imports ----
from spad.utils import load_model_from_config  # you already use this in inference.py
from spad.dataloader import ObjaverseMultiViewDataset  # adapt this import if name differs
from spad.lora import LoRALinear, LoRAConv2d  # assuming you have LoRA implementations here 


# ----------------- helper: device & batch -----------------
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


def to_device_batch(batch, device):
    """Move a nested batch dict (np / torch / python) to device."""
    
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
            use_abs_extrinsics=False,
        )
        cam_list.append(cam_b)    # [V, V, 4]
        intr_list.append(intr_b)  # [V, 4]

    batch["cam"] = torch.stack(cam_list, dim=0).to(device)                  # [B,V,V,4]
    batch["render_intrinsics_flat"] = torch.stack(intr_list, dim=0).to(device)  # [B,V,4]
    batch["txt"] = [batch["txt"]] * V  # replicate captions for all views

    return batch


# ----------------- EMA -----------------


@torch.no_grad()
def ema_update(ema_model: nn.Module, model: nn.Module, decay: float):
    """θ⁻ ← μ θ⁻ + (1-μ) θ  (in-place)."""
    ema_params = dict(ema_model.named_parameters())
    for name, param in model.named_parameters():
        if name in ema_params:
            ema_params[name].data.mul_(decay).add_(param.data, alpha=1.0 - decay)


# ----------------- LoRA param selection -----------------


def get_lora_parameters(model: nn.Module):
    """Collect LoRA parameters by name.

    Assumes all LoRA params contain 'lora' in their name (case-insensitive),
    and non-LoRA weights are frozen.
    """
    lora_params = []
    for name, p in model.named_parameters():
        if "lora" in name.lower():
            p.requires_grad = True
            lora_params.append(p)
        else:
            p.requires_grad = False
    if len(lora_params) == 0:
        print(
            "[WARN] No parameters with 'lora' in name were found. "
            "Falling back to training all parameters."
        )
        for p in model.parameters():
            p.requires_grad = True
        lora_params = list(model.parameters())
    return lora_params


def get_lora_named_params(model: nn.Module):
    """
    Return list of (full_name, param) for all LoRA params.
    Assumes 'lora_' appears in the parameter name (lora_A, lora_B, etc.).
    """
    out = []
    for name, p in model.named_parameters():
        if "lora" in name.lower():
            out.append((name, p))
    if not out:
        print("[WARN] No LoRA parameters found (name containing 'lora').")
    return out


# ----------------- DDPM schedule helpers -----------------


def build_train_time_schedule(teacher, num_train_timesteps: int, device: str):
    """
    Build a reduced time grid for consistency distillation.

    - teacher.num_timesteps: original DDPM steps (e.g. 1000)
    - We pick num_train_timesteps indices linearly from [0, num_timesteps-1]
      and precompute:
        * sqrt_alpha_bar_t      = sqrt(ᾱ_t)
        * sqrt_one_minus_alpha  = sqrt(1 - ᾱ_t)
    """
    assert hasattr(teacher, "num_timesteps")
    assert hasattr(teacher, "alphas_cumprod")

    num_ddpm = int(teacher.num_timesteps)  # e.g. 1000
    ddpm_indices = torch.linspace(
        0, num_ddpm - 1, num_train_timesteps, dtype=torch.long, device=device
    )

    alphas_cumprod = teacher.alphas_cumprod.to(device)                  # ᾱ_t (1D, length num_ddpm)
    alpha_bar_train = alphas_cumprod[ddpm_indices]                      # ᾱ for training grid

    sqrt_alpha_bar_train = torch.sqrt(alpha_bar_train)                  # √ᾱ_t
    sqrt_one_minus_alpha_bar_train = torch.sqrt(1.0 - alpha_bar_train)  # √(1 - ᾱ_t)

    return ddpm_indices, sqrt_alpha_bar_train, sqrt_one_minus_alpha_bar_train


def _expand_scalar_for(x_like: torch.Tensor, scalar: torch.Tensor):
    """Expand scalar to match rank of x_like (for broadcasting)."""
    while scalar.dim() < x_like.dim():
        scalar = scalar.view(*scalar.shape, 1)
    return scalar


# ----------------- teacher ODE step Ψ via DDIM -----------------


@torch.no_grad()
def psi_ddim_step(
    teacher,
    z_t,  # [B,C,H,W] or [N,V,C,H,W]
    cond,
    t_from_idx: int,
    t_to_idx: int,
    ddpm_indices: torch.Tensor,
    sqrt_alpha_bar_train: torch.Tensor,
    sqrt_one_minus_alpha_bar_train: torch.Tensor,
):
    """
    One deterministic DDIM-style step Ψ(z_t, t_from, t_to, c), returning:
        delta = z_{t_to} - z_t
    using the teacher.

    This mirrors your ManyViewDDIMSampler.p_sample_ddim with eta=0,
    but we allow arbitrary (t_from -> t_to) instead of only prev.
    """
    device = z_t.device

    # Map training index -> original DDPM time index (0..num_ddpm-1)
    ddpm_t_from = int(ddpm_indices[t_from_idx].item())

    # Build time tensor with correct shape:
    #   - if z_t is [N,V,C,H,W], t should be [N,V]
    #   - if z_t is [B,C,H,W],  t should be [B]
    if z_t.dim() == 5:
        n, v = z_t.shape[:2]
        t_batch = torch.full(
            (n, v),
            ddpm_t_from,
            device=device,
            dtype=torch.long,
        )
    else:  # 4D latent
        b = z_t.shape[0]
        t_batch = torch.full(
            (b,),
            ddpm_t_from,
            device=device,
            dtype=torch.long,
        )

    # ε_θ(z_t, t_from, c)
    eps = teacher.apply_model(z_t, t_batch, cond)

    # DDPM / DDIM parameters on the reduced training grid
    alpha_from = _expand_scalar_for(z_t, sqrt_alpha_bar_train[t_from_idx].to(device))
    sigma_from = _expand_scalar_for(z_t, sqrt_one_minus_alpha_bar_train[t_from_idx].to(device))
    alpha_to   = _expand_scalar_for(z_t, sqrt_alpha_bar_train[t_to_idx].to(device))
    sigma_to   = _expand_scalar_for(z_t, sqrt_one_minus_alpha_bar_train[t_to_idx].to(device))

    # Reconstruct clean latent x_0 at t_from
    #   pred_x0 = (x_t - sqrt(1-ᾱ_t)*ε) / sqrt(ᾱ_t)
    x0_pred = (z_t - sigma_from * eps) / alpha_from

    # Deterministic DDIM step to t_to (keep ε fixed)
    #   z_{t_to} = sqrt(ᾱ_{t_to}) * x0_pred + sqrt(1-ᾱ_{t_to}) * ε
    z_to = alpha_to * x0_pred + sigma_to * eps

    # Return the increment Ψ = z_to - z_t
    return z_to - z_t


# ----------------- student mapping f_θ (LCM-style) -----------------


def lcm_forward(
    student,
    z_t,
    cond,
    ucond,
    omega: float,
    t_idx: int,
    ddpm_indices: torch.Tensor,
    sqrt_alpha_bar_train: torch.Tensor,
    sqrt_one_minus_alpha_bar_train: torch.Tensor,
):
    """
    f_θ(z_t, ω, c, t) for SPAD + LoRA.

    1. Compute ε_θ(z_t, t, c) and ε_θ(z_t, t, ∅)
    2. Combine via CFG: ε̃ = (1+ω) ε_c - ω ε_∅
    3. Map to x_0 via DDPM formula.

    Returns predicted clean latent x_0 at time t_idx.
    """
    device = z_t.device
    ddpm_t = int(ddpm_indices[t_idx].item())

    # time tensor with same logic as psi_ddim_step
    if z_t.dim() == 5:
        n, v = z_t.shape[:2]
        t_batch = torch.full((n, v), ddpm_t, device=device, dtype=torch.long)
    else:
        b = z_t.shape[0]
        t_batch = torch.full((b,), ddpm_t, device=device, dtype=torch.long)

    # ε_c and ε_∅
    eps_cond   = student.apply_model(z_t, t_batch, cond)
    eps_uncond = student.apply_model(z_t, t_batch, ucond)
    eps = (1.0 + omega) * eps_cond - omega * eps_uncond

    alpha_t = _expand_scalar_for(z_t, sqrt_alpha_bar_train[t_idx].to(device))
    sigma_t = _expand_scalar_for(z_t, sqrt_one_minus_alpha_bar_train[t_idx].to(device))

    x0_pred = (z_t - sigma_t * eps) / alpha_t
    return x0_pred


# ----------------- main trainer -----------------


class LCMSpadDistiller:
    def __init__(self, args):
        self.args = args
        os.makedirs(args.output_dir, exist_ok=True)

        self.device = torch.device(args.device)

        # ---- 1. Data ----
        self.train_dataset = ObjaverseMultiViewDataset(
            root=args.data_root,
            num_views=args.num_views,
            image_size=args.image_size,
            training=True,
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        # ---- 2. Teacher / Student ----
        config = OmegaConf.load(args.config)

        print(f"[INFO] Loading teacher from {args.teacher_ckpt}")
        self.teacher = load_model_from_config(config, args.teacher_ckpt)
        self.teacher.to(self.device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        if args.student_ckpt is None:
            print("[INFO] Initializing student from teacher weights")
            self.student = load_model_from_config(config, args.teacher_ckpt)
        else:
            print(f"[INFO] Loading student from {args.student_ckpt}")
            self.student = load_model_from_config(config, args.student_ckpt)

        self.student.to(self.device)
        self.student.train()
        for p in self.student.parameters():
            p.requires_grad = False
        self.insert_lora_layers(
            self.student.model.diffusion_model,
            r=args.lora_rank,
            alpha=args.lora_alpha,
            )
        self.student.to(self.device)

        self.lora_named_params = get_lora_named_params(self.student)  # [(name, param), ...]

        # simple dict: name -> tensor (same shape as param)
        self.ema_lora = {
            name: p.detach().clone()
            for name, p in self.lora_named_params
        }
        for t in self.ema_lora.values():
            t.requires_grad = False

        # ---- 3. LoRA params & optimizer ----
        lora_params = [p for _, p in self.lora_named_params]
        self.optimizer = torch.optim.AdamW(
            lora_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        # # ---- 4. EMA student θ⁻ ----
        # self.ema_student = copy.deepcopy(self.student).to(self.device)
        # self.ema_student.eval()
        # for p in self.ema_student.parameters():
        #     p.requires_grad = False

        # ---- 5. Time schedule ----
        (
            self.ddpm_indices,
            self.alphas_train,
            self.sigmas_train,
        ) = build_train_time_schedule(
            self.teacher,
            num_train_timesteps=args.train_timesteps,
            device=self.device,
        )

        assert (
            args.train_timesteps > args.num_sampling_steps
        ), "train_timesteps must be > num_sampling_steps"

        self.skip = args.train_timesteps // args.num_sampling_steps
        print(
            f"[INFO] train_timesteps={args.train_timesteps}, "
            f"num_sampling_steps={args.num_sampling_steps}, skip={self.skip}"
        )

        self.micro_step = 0
        self.global_step = 0

    def _is_class(self, module: nn.Module, name: str) -> bool:
        return module.__class__.__name__ == name


    def insert_lora_layers(
        self,
        unet: nn.Module,
        r: int = 4,
        alpha: float = 1.0,
        enable_attn: bool = True,
        enable_mlp: bool = True,
    ):
        """
        Add LoRA to SPAD UNet in-place.

        Expected structure (from your printout):
        - SPADAttention: to_q, to_k, to_v: nn.Linear
                        to_out: nn.Sequential(Linear, Dropout)
        - FeedForward:   net = Sequential(
                            GEGLU(proj=Linear),
                            Dropout,
                            Linear
                        )

        Usage:
            spad_ldm = load_model_from_config(...)
            unet = spad_ldm.model.diffusion_model  # DiffusionWrapper.diffusion_model
            inject_lora_into_spad_unet(unet, r=8, alpha=16)
        """
        for module in unet.modules():
            # ---- 1) Attention: q, k, v, out ----
            if enable_attn and self._is_class(module, "SPADAttention"):
                # to_q, to_k, to_v are Linear
                if isinstance(module.to_q, nn.Linear) and not isinstance(module.to_q, LoRALinear):
                    module.to_q = LoRALinear(module.to_q, r=r, alpha=alpha)
                if isinstance(module.to_k, nn.Linear) and not isinstance(module.to_k, LoRALinear):
                    module.to_k = LoRALinear(module.to_k, r=r, alpha=alpha)
                if isinstance(module.to_v, nn.Linear) and not isinstance(module.to_v, LoRALinear):
                    module.to_v = LoRALinear(module.to_v, r=r, alpha=alpha)

                # to_out is Sequential(Linear, Dropout)
                if isinstance(module.to_out, nn.Sequential):
                    if (
                        len(module.to_out) > 0
                        and isinstance(module.to_out[0], nn.Linear)
                        and not isinstance(module.to_out[0], LoRALinear)
                    ):
                        module.to_out[0] = LoRALinear(module.to_out[0], r=r, alpha=alpha)

            # ---- 2) FeedForward MLP: first & last Linear ----
            if enable_mlp and self._is_class(module, "FeedForward"):
                # module.net: Sequential(GEGLU(proj=Linear), Dropout, Linear)
                net = module.net

                # 2.1 first Linear inside GEGLU
                if len(net) > 0 and hasattr(net[0], "proj"):
                    proj = net[0].proj
                    if isinstance(proj, nn.Linear) and not isinstance(proj, LoRALinear):
                        net[0].proj = LoRALinear(proj, r=r, alpha=alpha)

                # 2.2 final Linear (net[2])
                if len(net) > 2 and isinstance(net[2], nn.Linear) and not isinstance(net[2], LoRALinear):
                    net[2] = LoRALinear(net[2], r=r, alpha=alpha)

    def save_checkpoint(self, name: str):
        ckpt_path = os.path.join(self.args.output_dir, name)
        save_dict = {
            # "student_lora": {name: p.detach().cpu() for name, p in self.lora_named_params},
            "ema_lora": {name: t.detach().cpu() for name, t in self.ema_lora.items()},
            # "optimizer": self.optimizer.state_dict(),
            # "global_step": self.global_step,
            # "teacher_ckpt": self.args.teacher_ckpt,  # for convenience
        }
        torch.save(save_dict, ckpt_path)
        print(f"[INFO] Saved checkpoint to {ckpt_path}")


    def train(self):
        args = self.args
        pbar = tqdm(total=args.max_steps, desc="LCM-LoRA distillation", dynamic_ncols=True)

        data_iter = iter(self.train_loader)

        while self.global_step < args.max_steps:
            self.micro_step += 1
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            batch = to_device_batch(batch, self.device)

            # ---- 1. Encode to latent space with SPAD get_input ----
            z_0, cond, ucond = self.teacher.get_input(
                batch,
                return_first_stage_outputs=False,
                return_original_cond=False,
                return_uc=True,
            )

            # z_0 shape can be [B,V,C,H,W] or [B,C,H,W], we treat it as generic tensor
            z_0 = z_0.to(self.device)

            # ---- 2. Sample n, ω, ε  ----
            # n ~ Uniform{0,...,T_skip-1}
            max_n = args.train_timesteps - self.skip - 1
            n_idx = torch.randint(
                low=0,
                high=max_n + 1,
                size=(1,),
                device=self.device,
            ).item()

            t_from_idx = n_idx + self.skip  # t_{n+k}
            t_to_idx = n_idx               # t_n

            # guidance scale ω ∈ [w_min, w_max]
            omega = float(
                torch.empty(1, device=self.device).uniform_(args.w_min, args.w_max).item()
            )

            # ε ~ N(0,I)
            eps = torch.randn_like(z_0)

            alpha_from = _expand_scalar_for(z_0, self.alphas_train[t_from_idx].to(self.device))
            sigma_from = _expand_scalar_for(z_0, self.sigmas_train[t_from_idx].to(self.device))

            # z_{t_{n+k}} = α(t_{n+k}) z_0 + σ(t_{n+k}) ε
            z_tnk = alpha_from * z_0 + sigma_from * eps

            # ---- 3. Teacher ODE target z_hat_{Ψ, ω, t_n} ----
            delta_c = psi_ddim_step(
                self.teacher,
                z_tnk,
                cond,
                t_from_idx,
                t_to_idx,
                self.ddpm_indices,
                self.alphas_train,
                self.sigmas_train,
            )
            delta_u = psi_ddim_step(
                self.teacher,
                z_tnk,
                ucond,
                t_from_idx,
                t_to_idx,
                self.ddpm_indices,
                self.alphas_train,
                self.sigmas_train,
            )

            z_hat_tn = z_tnk + (1.0 + omega) * delta_c - omega * delta_u

            # ---- 4. Student consistency outputs f_θ ----
            x0_tnk = lcm_forward(
                self.student,
                z_tnk,
                cond,
                ucond,
                omega,
                t_from_idx,
                self.ddpm_indices,
                self.alphas_train,
                self.sigmas_train,
            )
            x0_tn = lcm_forward(
                self.student,
                z_hat_tn,
                cond,
                ucond,
                omega,
                t_to_idx,
                self.ddpm_indices,
                self.alphas_train,
                self.sigmas_train,
            )

            # ---- 5. Consistency loss ----
            loss = torch.mean((x0_tnk - x0_tn) ** 2) / args.accumulate_steps
            loss.backward()

            if self.micro_step % args.accumulate_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                
                # ---- 6. EMA update ----
                self.ema_update_lora(self.ema_lora, self.lora_named_params, decay=args.ema_decay)
                
                self.micro_step = 0
                self.global_step += 1

            # ---- 7. Logging & checkpoint ----
            if self.global_step % args.log_every == 0:
                print(
                    f"[step {self.global_step}] "
                    f"loss={loss.item():.6f}, omega={omega:.3f}, "
                    f"n_idx={n_idx}, t_from_idx={t_from_idx}, t_to_idx={t_to_idx}"
                )

            if self.global_step % args.ckpt_every == 0 and self.global_step > 0:
                self.save_checkpoint(f"step_{self.global_step:07d}.pt")

            if self.micro_step % args.accumulate_steps == 0:
                pbar.update(1)

        # final save
        self.save_checkpoint("last.pt")
        pbar.close()

    @torch.no_grad()
    def ema_update_lora(self, ema_dict, named_params, decay):
        """
        ema_dict: dict[name] -> tensor
        named_params: list of (name, param)
        """
        for name, p in named_params:
            if name not in ema_dict:
                continue
            ema_param = ema_dict[name]
            ema_param.mul_(decay).add_(p.data, alpha=1.0 - decay)

    def load_spad_with_ema_lora(self, config_path, teacher_ckpt, ema_ckpt_path, device="cuda"):
        # 1. build base SPAD model (teacher weights)
        config = OmegaConf.load(config_path)
        model = load_model_from_config(config, teacher_ckpt)
        model.to(device)

        # 2. inject LoRA into the UNet (same function as before)
        self.insert_lora_layers(model.model.diffusion_model, r=4, alpha=1.0)

        # 3. load EMA LoRA weights
        ckpt = torch.load(ema_ckpt_path, map_location="cpu")
        ema_lora = ckpt["ema_lora"]

        # 4. copy into matching params
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in ema_lora:
                    p.copy_(ema_lora[name].to(p.device))

        model.eval()
        return model

# ----------------- argparse entry -----------------


def parse_args():
    parser = argparse.ArgumentParser("LCM-LoRA distillation for SPAD")

    parser.add_argument("--config", type=str, required=True,
                        help="SPAD config .yaml (same as training/inference)")
    parser.add_argument("--teacher_ckpt", type=str, required=True,
                        help="Path to full SPAD checkpoint (teacher)")
    parser.add_argument("--student_ckpt", type=str, default=None,
                        help="Optional initial student checkpoint, "
                             "default: copy from teacher_ckpt")

    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of Objaverse / GSO data")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accumulate_steps", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--train_timesteps", type=int, default=1000,
                        help="Number of training timesteps on reduced grid (e.g. 50)")
    parser.add_argument("--num_sampling_steps", type=int, default=4,
                        help="Target inference steps for LCM sampler (e.g. 4)")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)

    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--ckpt_every", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="logs/spad_lcm_lora")

    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank r")
    parser.add_argument("--lora_alpha", type=float, default=16,
                        help="LoRA scaling factor α")

    parser.add_argument("--w_min", type=float, default=3.0,
                        help="Min CFG scale ω used in distillation (LCM-LoRA typically 0~2)")
    parser.add_argument("--w_max", type=float, default=9.0,
                        help="Max CFG scale ω used in distillation")
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def main():
    args = parse_args()
    print(args)
    distiller = LCMSpadDistiller(args)
    distiller.train()


if __name__ == "__main__":
    main()
