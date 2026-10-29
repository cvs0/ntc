"""Factorized Laplace rate for latent codes.

One scale per channel, the mean absolute deviation of the centered code.
The bits are the continuous Laplace negative log-likelihood, which is the
rate a memoryless coder spends on those codes. The gradient flows into the
codes. The scale estimate is detached.
"""

from __future__ import annotations

import math

import torch

_LN2 = math.log(2.0)


def laplace_bits(values: torch.Tensor, log_b: torch.Tensor) -> torch.Tensor:
    """Bits of the rounded code under a zero-mean discrete Laplace.

    The value is the mass of the unit bin around the rounded code, so it is
    in the same units as a coder. The gradient is the continuous Laplace
    likelihood, which is defined for a noisy code.
    """
    log_b = log_b.clamp(-8.0, 6.0)
    scale = log_b.exp().clamp(min=1e-3)

    def _cdf(x: torch.Tensor) -> torch.Tensor:
        return torch.where(x < 0, 0.5 * torch.exp(x / scale), 1.0 - 0.5 * torch.exp(-x / scale))

    rounded = values.detach().round()
    prob = (_cdf(rounded + 0.5) - _cdf(rounded - 0.5)).clamp(min=1e-12)
    discrete = -torch.log2(prob)
    continuous = (1.0 + log_b / _LN2) + values.abs() / (scale * _LN2)
    return discrete + continuous - continuous.detach()


def file_rate_bpp(model_bpp: torch.Tensor, latent_rate_scale: float | torch.Tensor) -> torch.Tensor:
    """Laplace bits times the stored/plain ratio, in file bpp.

    The label is a file rate. A 4-bit grid's Laplace rate is often near 4
    raw bpp while the range coder stores about half of that. Comparing the
    raw rate to the label penalizes a file that is already under the label.
    Tiles135D b2.5 on 5a71784 sat at 31.7 dB from step 3000 with a file at
    2.21 bpp. A scale outside (0, 1) leaves the model bpp unchanged.
    A tensor scale is the running measurement. Values outside (0, 1) still
    leave the model bpp unchanged, including a zero probe that has not run.
    """
    if torch.is_tensor(latent_rate_scale):
        scale = latent_rate_scale.to(dtype=model_bpp.dtype, device=model_bpp.device).reshape(())
        use = (scale > 0) & (scale < 1)
        scale = torch.where(use, scale, torch.ones_like(scale))
        return model_bpp * scale
    scale = float(latent_rate_scale)
    if not (0.0 < scale < 1.0):
        scale = 1.0
    return model_bpp * scale


def blend_file_rate_scale(
    current: float,
    pca_scale: float,
    measured_bpp: float,
    model_bpp: float,
) -> float:
    """Move the penalty ratio toward the coder ratio of the current codes.

    The PCA probe on MetalPlates was about 0.40, so ``0.40 * Laplace`` stayed
    under the 2.5 label while the trained file landed at 2.79 bpp. The probe
    is a floor: training does not get more compressible than that init. One
    update moves halfway, so a single noisy measure cannot slam the loss.
    A scale outside (0, 1) is left alone. Tiles at 0.511 with a file under
    the label does not move.
    """
    if not (0.0 < float(pca_scale) < 1.0):
        return float(current)
    if not math.isfinite(float(measured_bpp)) or float(model_bpp) <= 1e-8:
        return float(current)
    if not math.isfinite(float(current)):
        current = float(pca_scale)
    ratio = float(measured_bpp) / float(model_bpp)
    target = min(0.95, max(float(pca_scale), ratio))
    updated = 0.5 * float(current) + 0.5 * target
    return min(0.95, max(float(pca_scale), updated))


def latent_rate_bits(latents: list[torch.Tensor], grid_bits: list[int]) -> torch.Tensor:
    """Bits for every latent code under a per-channel Laplace.

    ``latents`` are ``[S,C,H,W]`` or ``[C,H,W]`` in code units.
    """
    total = None
    for lat, bits in zip(latents, grid_bits, strict=True):
        mid = float((1 << int(bits)) - 1) / 2.0
        centered = lat - mid
        channel_axis = 1 if centered.ndim == 4 else 0
        dims = [axis for axis in range(centered.ndim) if axis != channel_axis]
        deviation = centered.detach().abs().mean(dim=dims).clamp(min=1e-3)
        log_b = deviation.log()
        view = [1] * centered.ndim
        view[channel_axis] = -1
        term = laplace_bits(centered, log_b.view(*view)).sum()
        total = term if total is None else total + term
    if total is None:
        raise ValueError("no latents")
    return total
