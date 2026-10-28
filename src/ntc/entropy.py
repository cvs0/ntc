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


def file_rate_bpp(model_bpp: torch.Tensor, latent_rate_scale: float) -> torch.Tensor:
    """Laplace bits times the stored/plain ratio, in file bpp.

    The label is a file rate. A 4-bit grid's Laplace rate is often near 4
    raw bpp while the range coder stores about half of that. Comparing the
    raw rate to the label penalizes a file that is already under the label.
    Tiles135D b2.5 on 5a71784 sat at 31.7 dB from step 3000 with a file at
    2.21 bpp. A scale outside (0, 1) leaves the model bpp unchanged.
    """
    scale = float(latent_rate_scale)
    if not (0.0 < scale < 1.0):
        scale = 1.0
    return model_bpp * scale


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
