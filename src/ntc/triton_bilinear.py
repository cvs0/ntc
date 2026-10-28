"""Optional Triton bilinear gather. The eager path stays the default on CPU.

On CUDA, the first call compares a Triton forward to ``bilinear_batched``'s
eager implementation. A mismatch disables Triton for the rest of the process.
The backward is the eager scatter of the four taps, so a bad Triton forward
cannot ship a silent wrong gradient: the check rejects it first.
"""

from __future__ import annotations

import os

# Triton 3.9 warns when this is unset and then defaults to fusion on.
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

import torch

_DISABLED = False
_CHECKED = False


def try_triton_bilinear(codes: torch.Tensor, uv: torch.Tensor) -> torch.Tensor | None:
    global _DISABLED, _CHECKED
    if _DISABLED or not codes.is_cuda or uv.numel() == 0:
        return None
    try:
        import triton
        import triton.language as tl
    except Exception:
        _DISABLED = True
        return None
    if not _CHECKED:
        _CHECKED = True
        eager = _eager(codes.detach(), uv.detach())
        try:
            trial = _forward(codes.detach(), uv.detach(), triton, tl)
        except Exception:
            _DISABLED = True
            return None
        if trial.shape != eager.shape or not torch.allclose(trial, eager, atol=1e-4, rtol=1e-4):
            _DISABLED = True
            return None
    try:
        return _Bilinear.apply(codes, uv, triton, tl)
    except Exception:
        _DISABLED = True
        return None


def _eager(codes: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    from ntc.train_fast import bilinear_batched as eager

    # Avoid recursion: call the reference body by temporarily disabling.
    global _DISABLED
    previous = _DISABLED
    _DISABLED = True
    try:
        return eager(codes, uv)
    finally:
        _DISABLED = previous


class _Bilinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, codes, uv, triton, tl):
        ctx.save_for_backward(codes, uv)
        ctx.triton = triton
        ctx.tl = tl
        return _forward(codes, uv, triton, tl)

    @staticmethod
    def backward(ctx, grad_out):
        codes, uv = ctx.saved_tensors
        grad = _eager_backward(codes, uv, grad_out)
        return grad, None, None, None


def _forward(codes, uv, triton, tl):
    seeds, channels, height, width = codes.shape
    count = uv.shape[0]
    out = torch.empty(seeds, count, channels, device=codes.device, dtype=codes.dtype)
    grid = (seeds * count,)

    @triton.jit
    def kernel(codes_ptr, uv_ptr, out_ptr, channels, height, width, count, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        s = pid // count
        n = pid % count
        u = tl.load(uv_ptr + n * 2)
        v = tl.load(uv_ptr + n * 2 + 1)
        x = u * width - 0.5
        y = v * height - 0.5
        x0 = tl.math.floor(x)
        y0 = tl.math.floor(y)
        tx = x - x0
        ty = y - y0
        x0i = tl.math.mod(x0, width)
        x1i = tl.math.mod(x0 + 1, width)
        y0i = tl.math.mod(y0, height)
        y1i = tl.math.mod(y0 + 1, height)
        # Negative modulo matches torch.remainder for the values we sample.
        offs = tl.arange(0, BLOCK)
        mask = offs < channels
        base = s * channels * height * width
        i00 = base + offs * height * width + y0i * width + x0i
        i01 = base + offs * height * width + y0i * width + x1i
        i10 = base + offs * height * width + y1i * width + x0i
        i11 = base + offs * height * width + y1i * width + x1i
        c00 = tl.load(codes_ptr + i00, mask=mask, other=0)
        c01 = tl.load(codes_ptr + i01, mask=mask, other=0)
        c10 = tl.load(codes_ptr + i10, mask=mask, other=0)
        c11 = tl.load(codes_ptr + i11, mask=mask, other=0)
        a = c00 + (c01 - c00) * tx
        b = c10 + (c11 - c10) * tx
        val = a + (b - a) * ty
        tl.store(out_ptr + (s * count + n) * channels + offs, val, mask=mask)

    block = 16 if channels <= 16 else 32
    kernel[grid](codes, uv, out, channels, height, width, count, BLOCK=block)
    return out


def _eager_backward(codes: torch.Tensor, uv: torch.Tensor, grad_out: torch.Tensor) -> torch.Tensor:
    """Scatter the four bilinear taps. ``grad_out`` is ``[S, N, C]``."""
    _s, _c, height, width = codes.shape
    x = uv[:, 0] * width - 0.5
    y = uv[:, 1] * height - 0.5
    x0 = torch.floor(x)
    y0 = torch.floor(y)
    tx = (x - x0).view(1, -1, 1)
    ty = (y - y0).view(1, -1, 1)
    x0i = torch.remainder(x0.to(torch.long), width)
    x1i = torch.remainder(x0i + 1, width)
    y0i = torch.remainder(y0.to(torch.long), height)
    y1i = torch.remainder(y0i + 1, height)
    grad = torch.zeros_like(codes)
    w00 = (1 - tx) * (1 - ty)
    w01 = tx * (1 - ty)
    w10 = (1 - tx) * ty
    w11 = tx * ty

    def splat(yy, xx, weight):
        # Sorted segment sum. Atomic scatter-add is not repeatable.
        from ntc.train_fast import _segment_add

        values = (grad_out * weight).permute(0, 2, 1).contiguous()
        _segment_add(grad.view(grad.shape[0], grad.shape[1], -1), yy * width + xx, values)

    splat(y0i, x0i, w00)
    splat(y0i, x1i, w01)
    splat(y1i, x0i, w10)
    splat(y1i, x1i, w11)
    return grad
