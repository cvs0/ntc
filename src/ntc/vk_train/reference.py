"""CPU oracle for the Vulkan training kernels.

Adam is the library update, including the bias correction. The gather is
the four bilinear taps used by ``_bilinear_forward``. Both stay in fp32.
"""

from __future__ import annotations

import torch


def adam_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    moment: torch.Tensor,
    second: torch.Tensor,
    step: int,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One Adam step. ``step`` is 1-based, matching ``torch.optim.Adam``."""
    moment = beta1 * moment + (1.0 - beta1) * grad
    second = beta2 * second + (1.0 - beta2) * grad * grad
    mhat = moment / (1.0 - beta1**int(step))
    vhat = second / (1.0 - beta2**int(step))
    updated = param - float(lr) * mhat / (torch.sqrt(vhat) + float(eps))
    return updated, moment, second


def gather_taps(codes: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """``codes`` is ``[S,C,H,W]``. ``uv`` is ``[N,2]``. Result ``[S,N,C]``."""
    _s, _c, height, width = codes.shape
    x = uv[:, 0] * width - 0.5
    y = uv[:, 1] * height - 0.5
    x0 = torch.floor(x)
    y0 = torch.floor(y)
    tx = (x - x0).view(1, -1, 1)
    ty = (y - y0).view(1, -1, 1)
    x0i = x0.to(torch.long)
    y0i = y0.to(torch.long)
    x0w = torch.remainder(x0i, width)
    x1w = torch.remainder(x0i + 1, width)
    y0w = torch.remainder(y0i, height)
    y1w = torch.remainder(y0i + 1, height)
    plane = codes.reshape(codes.shape[0], codes.shape[1], height * width)

    def take(yy: torch.Tensor, xx: torch.Tensor) -> torch.Tensor:
        return plane[:, :, yy * width + xx].permute(0, 2, 1)

    c00 = take(y0w, x0w)
    c01 = take(y0w, x1w)
    c10 = take(y1w, x0w)
    c11 = take(y1w, x1w)
    return c00 + (c01 - c00) * tx + ((c10 + (c11 - c10) * tx) - (c00 + (c01 - c00) * tx)) * ty
