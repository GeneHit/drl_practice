from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def augment_image(
    image: Tensor,
    *,
    random_shift_pad: int = 4,
    jitter_strength: float = 0.10,
    noise_std: float = 0.00,
    enable_rotate: bool = False,
    max_rotate_deg: float = 2.0,
    enable_hflip: bool = False,
    hflip_prob: float = 0.10,
    interpolation: str = "bilinear",
    padding_mode: str = "border",
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """
    Augment pixel observations for image-based RL.

    Args:
        image: Input tensor [N, C, H, W]. dtype can be uint8 ([0,255]) or float ([0,1]).
        random_shift_pad: Max pixel shift each axis via grid-sample (approx. DrQ random shift).
        jitter_strength: Brightness/contrast jitter amplitude. 0 disables jitter.
        noise_std: Std of additive Gaussian noise in [0,1] scale. 0 disables noise.
        enable_rotate: Enable tiny random rotation (±max_rotate_deg).
        max_rotate_deg: Max absolute rotation angle in degrees (small values only).
        enable_hflip: Enable random horizontal flip with probability hflip_prob.
        hflip_prob: Probability for hflip per sample.
        interpolation: Interpolation mode for grid_sample ("nearest"/"bilinear").
        padding_mode: Padding mode for grid_sample ("zeros"/"border"/"reflection").
        generator: Optional torch.Generator for deterministic sampling.

    Returns:
        Augmented tensor with the same shape and dtype/scale as input.
    """
    assert image.ndim == 4, f"image must be [N, C, H, W], got {image.shape}"
    N, _, H, W = image.shape
    dev = image.device
    orig_dtype = image.dtype

    # Normalize to float32 in [0,1] for augmentation
    if image.dtype == torch.uint8 or (image.max() > 1.5):
        x = image.float() / 255.0
        scaled255 = True
    else:
        x = image.float()
        scaled255 = False

    # ---- (Optional) tiny rotation ----
    if enable_rotate and max_rotate_deg > 0:
        angles = (torch.rand(N, device=dev, generator=generator) * 2 - 1) * (
            max_rotate_deg * 3.14159265 / 180.0
        )
        cos, sin = torch.cos(angles), torch.sin(angles)
        theta = torch.zeros(N, 2, 3, device=dev, dtype=x.dtype)
        theta[:, 0, 0] = cos
        theta[:, 0, 1] = -sin
        theta[:, 1, 0] = sin
        theta[:, 1, 1] = cos
        grid = F.affine_grid(theta, size=list(x.size()), align_corners=False)
        x = F.grid_sample(
            x, grid, mode=interpolation, padding_mode=padding_mode, align_corners=False
        )

    # ---- DrQ-style random shift via pure translation in normalized coords ----
    if random_shift_pad > 0:
        # Shift in integer pixels in [-pad, pad]
        dx = torch.randint(
            -random_shift_pad, random_shift_pad + 1, (N,), device=dev, generator=generator
        )
        dy = torch.randint(
            -random_shift_pad, random_shift_pad + 1, (N,), device=dev, generator=generator
        )
        # Convert pixel shift to normalized translation (align_corners=False)
        tx = (2.0 * dx.float()) / float(W)
        ty = (2.0 * dy.float()) / float(H)
        theta = torch.zeros(N, 2, 3, device=dev, dtype=x.dtype)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty
        grid = F.affine_grid(theta, size=list(x.size()), align_corners=False)
        x = F.grid_sample(
            x, grid, mode=interpolation, padding_mode=padding_mode, align_corners=False
        )

    # ---- Brightness / contrast jitter ----
    j = float(jitter_strength)
    if j > 0:
        # brightness factor in [1-j, 1+j]
        b = (1.0 - j) + (2.0 * j) * torch.rand(N, 1, 1, 1, device=dev, generator=generator)
        x = x * b
        # contrast around per-sample per-channel mean
        mean = x.mean(dim=(2, 3), keepdim=True)
        c = (1.0 - j) + (2.0 * j) * torch.rand(N, 1, 1, 1, device=dev, generator=generator)
        x = (x - mean) * c + mean

    # ---- Additive Gaussian noise ----
    if noise_std > 0:
        x = x + float(noise_std) * torch.randn(x.shape, device=dev, generator=generator)

    # ---- Random horizontal flip (use with care for Lander) ----
    if enable_hflip and hflip_prob > 0:
        mask = torch.rand(N, device=dev, generator=generator) < float(hflip_prob)
        if mask.any():
            x[mask] = torch.flip(x[mask], dims=[3])  # flip width

    # Clamp back to [0,1] and restore dtype/scale
    x = x.clamp(0.0, 1.0)
    if scaled255:
        x = (x * 255.0).round().to(orig_dtype)
    else:
        x = x.to(orig_dtype)
    return x
