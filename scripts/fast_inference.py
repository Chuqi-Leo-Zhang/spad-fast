import os
import math
import time
import argparse
from itertools import chain, cycle

import numpy as np
import torch
import torch.nn as nn
import imageio
from omegaconf import OmegaConf
from einops import rearrange
from pytorch_lightning import seed_everything
from tqdm import tqdm

from spad.utils import load_model_from_config, slugify
from spad.geometry import get_batch_from_spherical
from spad.lora import LoRALinear  # your LoRA module
from ldm.models.diffusion.ddim import ManyViewDDIMSampler


SPAD_FOV = 0.702769935131073  # radians
CAMERA_RADIUS = 3.5


# -----------------------------------------------------------------------------
# Camera + intrinsics (same as original script)
# -----------------------------------------------------------------------------
def generate_batch(elevations, azimuths, use_abs=False):
    elevations = [math.radians(e) for e in elevations]
    azimuths = [math.radians(a) for a in azimuths]

    batch = get_batch_from_spherical(elevations, azimuths)

    abs_cams = []
    for theta, az in zip(elevations, azimuths):
        abs_cams.append(torch.tensor([theta, az, CAMERA_RADIUS]))

    debug_cams = [[] for _ in range(len(azimuths))]
    for i, icam in enumerate(abs_cams):
        for j, jcam in enumerate(abs_cams):
            if use_abs:
                dcam = torch.tensor([icam[0],
                                     math.sin(icam[1]),
                                     math.cos(icam[1]),
                                     icam[2]])
            else:
                diff = icam - jcam
                dcam = torch.tensor([
                    diff[0].item(),
                    math.sin(diff[1].item()),
                    math.cos(diff[1].item()),
                    diff[2].item(),
                ])
            debug_cams[i].append(dcam)

    batch["cam"] = torch.stack([torch.stack(dc) for dc in debug_cams])

    focal = 1.0 / np.tan(SPAD_FOV / 2.0)
    intrinsics = np.diag(np.array([focal, focal, 1.0], dtype=np.float32))
    intrinsics = torch.from_numpy(intrinsics).unsqueeze(0).float()
    intrinsics = intrinsics.repeat(batch["cam"].shape[0], 1, 1)
    batch["render_intrinsics_flat"] = intrinsics[:, [0, 1, 0, 1], [0, 1, 2, 2]]

    return batch


def get_gaussian_image(blob_width=256, blob_height=256, sigma=0.5):
    X = np.linspace(-1, 1, blob_width)[None, :]
    Y = np.linspace(-1, 1, blob_height)[:, None]
    inv_dev = 1.0 / (sigma ** 2)
    gaussian_blob = np.exp(-0.5 * (X ** 2) * inv_dev) * np.exp(-0.5 * (Y ** 2) * inv_dev)

    if gaussian_blob.max() > 0:
        gaussian_blob = 255.0 * (gaussian_blob - gaussian_blob.min()) / gaussian_blob.max()
    gaussian_blob = 255.0 - gaussian_blob

    gaussian_blob = (gaussian_blob / 255.0) * 2.0 - 1.0
    gaussian_blob = np.expand_dims(gaussian_blob, axis=-1).repeat(3, axis=-1)
    return torch.from_numpy(gaussian_blob.astype(np.float32))


# -----------------------------------------------------------------------------
# LoRA insertion + EMA-LoRA loading (mirrors your training code)
# -----------------------------------------------------------------------------
def _is_class(module: nn.Module, name: str) -> bool:
    return module.__class__.__name__ == name


def insert_lora_layers(unet: nn.Module,
                       r: int = 16,
                       alpha: float = 16.0,
                       enable_attn: bool = True,
                       enable_mlp: bool = True):
    # SPADAttention + FeedForward
    for module in unet.modules():
        if enable_attn and _is_class(module, "SPADAttention"):
            if isinstance(module.to_q, nn.Linear) and not isinstance(module.to_q, LoRALinear):
                module.to_q = LoRALinear(module.to_q, r=r, alpha=alpha)
            if isinstance(module.to_k, nn.Linear) and not isinstance(module.to_k, LoRALinear):
                module.to_k = LoRALinear(module.to_k, r=r, alpha=alpha)
            if isinstance(module.to_v, nn.Linear) and not isinstance(module.to_v, LoRALinear):
                module.to_v = LoRALinear(module.to_v, r=r, alpha=alpha)

            if isinstance(module.to_out, nn.Sequential):
                if len(module.to_out) > 0 and isinstance(module.to_out[0], nn.Linear) \
                        and not isinstance(module.to_out[0], LoRALinear):
                    module.to_out[0] = LoRALinear(module.to_out[0], r=r, alpha=alpha)

        if enable_mlp and _is_class(module, "FeedForward"):
            net = module.net
            if len(net) > 0 and hasattr(net[0], "proj"):
                proj = net[0].proj
                if isinstance(proj, nn.Linear) and not isinstance(proj, LoRALinear):
                    net[0].proj = LoRALinear(proj, r=r, alpha=alpha)

            if len(net) > 2 and isinstance(net[2], nn.Linear) and not isinstance(net[2], LoRALinear):
                net[2] = LoRALinear(net[2], r=r, alpha=alpha)


def load_spad_with_ema_lora(config_path,
                            teacher_ckpt,
                            ema_lora_ckpt,
                            lora_rank=16,
                            lora_alpha=16.0,
                            device="cuda"):
    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, teacher_ckpt, verbose=True, inference_run=True)
    model.to(device)

    # Disable internal EMA – it does not know about LoRA parameter names
    if hasattr(model, "use_ema"):
        model.use_ema = False

    # Inject LoRA into UNet
    insert_lora_layers(model.model.diffusion_model, r=lora_rank, alpha=lora_alpha)

    # Load EMA-LoRA weights
    ckpt = torch.load(ema_lora_ckpt, map_location="cpu")
    ema_lora = ckpt["ema_lora"]
    # ema_lora = ckpt["student_lora"]

    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in ema_lora:
                p.copy_(ema_lora[name].to(p.device))

    model.to(device)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# DDIM sampling (original SPAD sampler, but with ddim_steps=4 etc.)
# -----------------------------------------------------------------------------
def denoise(batch, model, device, idx, total_views, outpath, blob_sigma, ddim_steps):
    sampler = ManyViewDDIMSampler(model)

    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}

    # Gaussian blob init
    blob = get_gaussian_image(sigma=blob_sigma)
    batch["img"][:, :] = blob
    print("using gaussian initialization")

    # SPAD get_input
    z, c, x, xrec, xc, uc = model.get_input(
        batch,
        return_first_stage_outputs=True,
        force_c_encode=True,
        return_original_cond=True,
        return_uc=True,
    )

    shape = (model.channels, model.image_size, model.image_size)
    batch_size = (len(x), total_views)

    kwargs = dict(
        unconditional_conditioning=uc,
        x0=z,   # blob latents
    )

    samples, _ = sampler.sample(
        ddim_steps,
        batch_size,
        shape,
        c,
        verbose=False,
        **kwargs,
    )

    x_samples = model.decode_first_stage(samples)
    x_samples = torch.clamp(x_samples, -1., 1.)

    # flatten & tile views
    x_samples = rearrange(x_samples, "b v c h w -> (b v) c h w")
    x_samples = (x_samples + 1.0) / 2.0
    xtxt = np.array(batch["txt"]).T.tolist()
    xtxt = list(chain(*xtxt))

    x_samples = rearrange(x_samples, "(n v) c h w -> n h (v w) c", v=total_views)
    x_samples = (x_samples * 255.0).cpu().numpy().astype(np.uint8)

    os.makedirs(outpath, exist_ok=True)
    for _idx, (image, caption) in enumerate(zip(x_samples, xtxt)):
        caption = slugify(caption)
        save_path = os.path.join(outpath, f"{caption}.png")
        imageio.imsave(save_path, image)
        print(f"saved image: {save_path}")


# -----------------------------------------------------------------------------
# Caption utils
# -----------------------------------------------------------------------------
def load_captions(path):
    caps = np.load(path, allow_pickle=True).tolist()
    caps = ["[tdv] " + c if "[tdv]" not in c else c for c in caps]
    return caps


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main(args):
    seed_everything(42 + 69)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_spad_with_ema_lora(
        args.config,
        args.teacher_ckpt,
        args.ema_lora_ckpt,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        device=device,
    )

    # CFG for SPAD
    model.cfg_conds = ["txt"]
    model.cfg_scales = [args.cfg_scale]

    visuals_dir = "data/visuals/"
    ts = str(round(time.time()))
    outdir = os.path.join(visuals_dir, "inference_lcm_lora", ts)
    os.makedirs(outdir, exist_ok=True)

    if args.captions is not None:
        caps = eval(f'"{args.captions}"')
        caps = [caps] if isinstance(caps, str) else caps
        caps = ["[tdv] " + c if "[tdv]" not in c else c for c in caps]
    else:
        caps = load_captions(args.captions_file)

    print(f"num of captions: {len(caps)}, batch_size: {args.batch_size}")

    # add opt params
    dataloader = cycle([{
        "img": torch.zeros(args.batch_size, args.total_views, 256, 256, 3)
    }])

    terminate = False
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="sampling")):
            if args.batch_size * (idx + 1) >= len(caps):
                bs = len(caps) - args.batch_size * idx
                terminate = True
            else:
                bs = args.batch_size

            elevations = [45 for _ in range(args.total_views)]
            azimuths = list(np.linspace(0,
                                        360 * ((args.total_views - 1) / args.total_views),
                                        args.total_views))
            print(f"using elevations: {elevations}, azimuths: {azimuths}")

            batch_cams = generate_batch(elevations, azimuths, use_abs=model.use_abs_extrinsics)
            batch_cams = {k: v[None].repeat_interleave(bs, dim=0).to(device)
                          for k, v in batch_cams.items()}
            batch.update(batch_cams)

            batch["txt"] = [caps[args.batch_size * idx: args.batch_size * idx + bs]] * args.total_views

            denoise(batch, model, device, idx, args.total_views, outdir,
                    args.blob_sigma, args.ddim_steps)

            if terminate:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser("4-step LCM-LoRA SPAD inference")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--ema_lora_ckpt", type=str, required=True,
                        help="LCM-LoRA distillation checkpoint (contains ema_lora)")
    parser.add_argument("--captions", type=str, default=None)
    parser.add_argument("--captions_file", type=str, default="data/captions_eval.npy")
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--blob_sigma", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--total_views", type=int, default=8)
    parser.add_argument("--ddim_steps", type=int, default=4,
                        help="Number of DDIM steps for fast sampler")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=16.0)

    args = parser.parse_args()
    main(args)
