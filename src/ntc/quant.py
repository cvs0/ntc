"""Symmetric integer quantization for latents and MLP weights.

Latent codes are unsigned. A code ``q`` in ``0 .. 2^bits-1`` decodes as

    value = (q - mid) * scale
    mid   = (2^bits - 1) / 2

Weight codes are signed, in ``-(2^(bits-1)-1) .. +(2^(bits-1)-1)``
(the extreme negative value is unused). A group is a contiguous span of
the input axis of one output row. ``group_size <= 0`` means one scale
for the whole row.

This is the same grouping scheme described by cvs0/weight-quantizer
(per-group scale, round to nearest, clamp). That repository documents
the method; the implementation used here lives in this module and in
``cpp/include/ntc/group_quant.hpp``.
"""

from __future__ import annotations

import math

import torch

QMAX = {8: 127, 4: 7, 2: 1}


def latent_mid(bits: int) -> float:
    return float((1 << bits) - 1) / 2.0


def quantize_latent(x: torch.Tensor, scale: torch.Tensor, bits: int):
    """Straight-through uniform quantizer.

    ``x`` is ``[C,H,W]``, ``scale`` is ``[C,1,1]`` (positive).
    Returns dequantized values with a straight-through gradient and
    integer codes as int64 (detached).
    """
    mid = latent_mid(bits)
    qmax = (1 << bits) - 1
    # scale is broadcast over H,W. Keep the division in fp32.
    code = torch.round(x / scale + mid)
    code = code.clamp(0, qmax)
    deq = (code - mid) * scale
    ste = x + (deq - x).detach()
    return ste, code.detach().to(torch.int64)


def quantize_latent_noise(x: torch.Tensor, scale: torch.Tensor, bits: int):
    """Uniform-noise proxy for the same quantizer. Training only.

    Adds ``U(-scale/2, scale/2)`` and clamps to the representable range.
    """
    mid = latent_mid(bits)
    span = mid * scale
    noise = (torch.rand_like(x) - 0.5) * scale
    y = (x + noise).clamp(-span, span)
    return y, None


def quantize_weight(
    weight: torch.Tensor, bits: int, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize ``weight`` of shape ``[out, in]``.

    Returns ``(dequant_ste, codes_int, scales)`` where ``codes_int`` is
    int64 ``[out, in]`` and ``scales`` is fp32 ``[out, groups]``.
    Scales are stored the way the file stores them: rounded to fp16,
    then promoted back, so training sees the exported scale.
    """
    if bits not in QMAX:
        raise ValueError(f"unsupported weight bits {bits}")
    qmax = QMAX[bits]
    out, inn = weight.shape
    gs = inn if group_size <= 0 or group_size >= inn else group_size
    pad = (gs - (inn % gs)) % gs
    if pad:
        padded = torch.nn.functional.pad(weight, (0, pad))
    else:
        padded = weight
    groups = padded.view(out, -1, gs)
    amax = groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / float(qmax)
    scale = scale.half().float()
    q = torch.round(groups / scale).clamp(-qmax, qmax)
    deq = (q * scale).reshape(out, -1)[:, :inn]
    ste = weight + (deq - weight).detach()
    codes = q.reshape(out, -1)[:, :inn].detach().to(torch.int64)
    return ste, codes, scale.squeeze(-1).detach()


def inv_exp_scale(scale: torch.Tensor) -> torch.Tensor:
    """``log(scale)`` for a positive scale parameter."""
    return torch.log(scale.clamp(min=1e-8))


def fp16_round(t: torch.Tensor) -> torch.Tensor:
    return t.half().float()


def weight_group_count(in_features: int, group_size: int) -> int:
    gs = in_features if group_size <= 0 or group_size >= in_features else group_size
    return int(math.ceil(in_features / gs))
