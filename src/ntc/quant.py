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

# fp16 finite range. A value outside this becomes inf, and inf times a
# zero code is NaN in the dequant. The smallest positive normal is 2^-14.
FP16_MAX = 65504.0
FP16_MIN_POS = 2.0**-14


def _autocast_off(device_type: str):
    """fp32 scope. A compiled bf16 region must not rerun the scale math."""
    kind = device_type if device_type in ("cuda", "cpu", "mps") else "cpu"
    return torch.autocast(device_type=kind, enabled=False)


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
    # The old floor was amax >= 1e-8, so the pre-round scale was about
    # 1e-10. fp16 flushes that to 0. Every zero tap in the group is then
    # 0/0, and that NaN is what the bf16 GEMM trains. A scale above 65504
    # is stored as inf, and inf times a zero code is NaN in the decoder.
    # Clamp into the fp16 normal range in fp32, then round.
    kind = weight.device.type if weight.device.type in ("cuda", "cpu", "mps") else "cpu"
    with _autocast_off(kind):
        weight = weight.float()
        out, inn = weight.shape
        gs = inn if group_size <= 0 or group_size >= inn else group_size
        pad = (gs - (inn % gs)) % gs
        if pad:
            padded = torch.nn.functional.pad(weight, (0, pad))
        else:
            padded = weight
        groups = padded.view(out, -1, gs)
        amax = groups.abs().amax(dim=-1, keepdim=True)
        scale = (amax / float(qmax)).clamp(min=FP16_MIN_POS, max=FP16_MAX)
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
    """Round to fp16 without producing inf.

    ``.half()`` of a finite value past 65504 is inf. The decoder then
    evaluates ``inf * 0`` on a zero code and the texel is NaN.
    """
    kind = t.device.type if t.device.type in ("cuda", "cpu", "mps") else "cpu"
    with _autocast_off(kind):
        return t.float().clamp(-FP16_MAX, FP16_MAX).half().float()


def weight_group_count(in_features: int, group_size: int) -> int:
    gs = in_features if group_size <= 0 or group_size >= in_features else group_size
    return int(math.ceil(in_features / gs))
