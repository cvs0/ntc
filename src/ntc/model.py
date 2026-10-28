"""Training network.

The forward pass uses the same quantizer, the same bilinear code lerp,
and the same MLP formula as ``decode_ref``. Grid mips used in the loss
are the integer box of the quantized codes. Gradients flow through the
straight-through base quantizer and through a matching average pool.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ntc.container import (
    ACT_HGELU,
    ACT_IDENTITY,
    ACT_RELU,
    PE_FOURIER,
    PE_TRIANGLE,
    pe_feature_count,
)
from ntc.quant import fp16_round, quantize_weight


def bilinear_codes_torch(codes: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """``codes`` float ``[C,H,W]`` (integer-valued), ``uv`` ``[N,2]``."""
    c, h, w = codes.shape
    x = uv[:, 0] * w - 0.5
    y = uv[:, 1] * h - 0.5
    x0 = torch.floor(x)
    y0 = torch.floor(y)
    tx = (x - x0).unsqueeze(1)
    ty = (y - y0).unsqueeze(1)
    x0i = x0.to(torch.long)
    y0i = y0.to(torch.long)
    x0w = torch.remainder(x0i, w)
    x1w = torch.remainder(x0i + 1, w)
    y0w = torch.remainder(y0i, h)
    y1w = torch.remainder(y0i + 1, h)
    plane = codes.reshape(c, h * w)

    def gather(yy, xx):
        idx = yy * w + xx
        return plane[:, idx].T

    c00 = gather(y0w, x0w)
    c01 = gather(y0w, x1w)
    c10 = gather(y1w, x0w)
    c11 = gather(y1w, x1w)
    a = c00 + (c01 - c00) * tx
    b = c10 + (c11 - c10) * tx
    return a + (b - a) * ty


def bilinear_tiled_torch(
    codes: torch.Tensor,
    uv: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    tile: int,
    mid: float,
    mip: int,
) -> torch.Tensor:
    """Dequantize each tap with the tile endpoint, then lerp. ``codes`` is ``[C,H,W]``."""
    c, h, w = codes.shape
    nty, ntx = scale.shape[1], scale.shape[2]
    x = uv[:, 0] * w - 0.5
    y = uv[:, 1] * h - 0.5
    x0 = torch.floor(x)
    y0 = torch.floor(y)
    tx = (x - x0).unsqueeze(1)
    ty = (y - y0).unsqueeze(1)
    x0i = x0.to(torch.long)
    y0i = y0.to(torch.long)
    x0w = torch.remainder(x0i, w)
    x1w = torch.remainder(x0i + 1, w)
    y0w = torch.remainder(y0i, h)
    y1w = torch.remainder(y0i + 1, h)
    plane = codes.reshape(c, h * w)
    shift = int(mip)

    def deq(yy, xx):
        idx = yy * w + xx
        code = plane[:, idx].T
        ox = xx.to(torch.long) << shift
        oy = yy.to(torch.long) << shift
        sx = torch.clamp(torch.div(ox, tile, rounding_mode="floor"), max=ntx - 1)
        sy = torch.clamp(torch.div(oy, tile, rounding_mode="floor"), max=nty - 1)
        s = scale[:, sy, sx].T
        o = offset[:, sy, sx].T
        return (code - mid) * s + o

    c00 = deq(y0w, x0w)
    c01 = deq(y0w, x1w)
    c10 = deq(y1w, x0w)
    c11 = deq(y1w, x1w)
    a = c00 + (c01 - c00) * tx
    b = c10 + (c11 - c10) * tx
    return a + (b - a) * ty


def _soft_box(values: torch.Tensor) -> torch.Tensor | None:
    """Differentiable 2x2 mean with the same crop as ``integer_box``."""
    c, h, w = values.shape
    if h == 1 and w == 1:
        return None
    h2 = max(1, h // 2)
    w2 = max(1, w // 2)
    if h >= 2 and w >= 2:
        src = values[:, : h2 * 2, : w2 * 2]
        src = src.reshape(c, h2, 2, w2, 2)
        return src.mean(dim=(2, 4))
    if h >= 2:
        src = values[:, : h2 * 2, :]
        src = src.reshape(c, h2, 2, w)
        return src.mean(dim=2)
    if w >= 2:
        src = values[:, :, : w2 * 2]
        src = src.reshape(c, h, w2, 2)
        return src.mean(dim=3)
    return None


def integer_box(codes: torch.Tensor) -> torch.Tensor | None:
    c, h, w = codes.shape
    if h == 1 and w == 1:
        return None
    h2 = max(1, h // 2)
    w2 = max(1, w // 2)
    src = codes
    if h >= 2 and w >= 2:
        src = src[:, : h2 * 2, : w2 * 2]
        src = src.reshape(c, h2, 2, w2, 2)
        acc = src.sum(dim=(2, 4))
        return (acc + 2) // 4
    if h >= 2:
        src = src[:, : h2 * 2, :]
        src = src.reshape(c, h2, 2, w)
        return (src.sum(dim=2) + 1) // 2
    if w >= 2:
        src = src[:, :, : w2 * 2]
        src = src.reshape(c, h, w2, 2)
        return (src.sum(dim=3) + 1) // 2
    return None


def mip_chain_from_codes(codes: torch.Tensor, limit: int) -> list[torch.Tensor]:
    chain = [codes]
    cur = codes
    while len(chain) < limit:
        nxt = integer_box(cur)
        if nxt is None:
            break
        if nxt.shape[-2] == cur.shape[-2] and nxt.shape[-1] == cur.shape[-1]:
            break
        chain.append(nxt)
        cur = nxt
        if cur.shape[-2] == 1 and cur.shape[-1] == 1:
            break
    return chain


class NtcNet(nn.Module):
    def __init__(
        self,
        *,
        image_channels: int,
        grids: list[tuple[int, int, int, int]],
        hidden: list[int],
        activation: int,
        pe_mode: int,
        pe_octaves: int,
        weight_bits: int,
        group_size: int,
        quant_proxy: str,
        activation_requant: bool,
        qat_latents: bool,
        qat_weights: bool,
        tile_size: int = 0,
        layer_bits: tuple = (),
    ):
        super().__init__()
        self.activation = activation
        self.pe_mode = pe_mode
        self.pe_octaves = pe_octaves
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.tile_size = int(tile_size)
        self.layer_bits = tuple(layer_bits)
        self.quant_proxy = quant_proxy
        self.activation_requant = activation_requant
        self.qat_latents = qat_latents
        self.qat_weights = qat_weights
        self.image_channels = image_channels
        self.grid_bits = []
        self.grid_shapes = []
        latents = []
        log_scales = []
        tile_logs = []
        tile_offs = []
        self._tile_param_of = []
        for gw, gh, ch, bits in grids:
            latents.append(nn.Parameter(torch.zeros(ch, gh, gw)))
            log_scales.append(nn.Parameter(torch.full((ch,), math.log(0.1))))
            self.grid_bits.append(bits)
            self.grid_shapes.append((gw, gh, ch, bits))
            nty = math.ceil(gh / self.tile_size) if self.tile_size >= 2 else 0
            ntx = math.ceil(gw / self.tile_size) if self.tile_size >= 2 else 0
            if self.tile_size >= 2 and (gw > self.tile_size or gh > self.tile_size):
                tile_logs.append(nn.Parameter(torch.full((ch, nty, ntx), math.log(0.1))))
                tile_offs.append(nn.Parameter(torch.zeros(ch, nty, ntx)))
                self._tile_param_of.append(len(tile_logs) - 1)
            else:
                self._tile_param_of.append(-1)
        self.latents = nn.ParameterList(latents)
        self.log_scales = nn.ParameterList(log_scales)
        self.tile_log_scales = nn.ParameterList(tile_logs)
        self.tile_offsets = nn.ParameterList(tile_offs)
        in_f = sum(g[2] for g in grids) + pe_feature_count(pe_mode, pe_octaves)
        dims = [in_f, *hidden, image_channels]
        self.linears = nn.ModuleList(
            nn.Linear(a, b, bias=True) for a, b in zip(dims[:-1], dims[1:])
        )
        self._init_linears()

    def _init_linears(self):
        for i, lin in enumerate(self.linears):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            if i != len(self.linears) - 1 and lin.weight.shape[0] == lin.weight.shape[1]:
                # near-identity hidden maps keep a PCA-sized latent usable
                eye = torch.eye(lin.weight.shape[0])
                lin.weight.data.copy_(0.5 * lin.weight.data + 0.5 * eye)

    def scales(self, index: int) -> torch.Tensor:
        s = torch.exp(self.log_scales[index]).clamp(1e-4, 5.0)
        return fp16_round(s).view(-1, 1, 1)

    def quant_grid(self, index: int):
        """Latent parameters are in code units, not reconstructed values.

        An Adam step of size 0.1 moves a code by a tenth of a bin. The
        reconstructed value is ``(code - mid) * scale``.
        """
        bits = self.grid_bits[index]
        scale = self.scales(index)
        z = self.latents[index]
        mid = float((1 << bits) - 1) / 2.0
        qmax = (1 << bits) - 1
        if not self.qat_latents:
            return (z - mid) * scale, None, scale
        if self.training and self.quant_proxy == "noise":
            return (z + (torch.rand_like(z) - 0.5) - mid) * scale, None, scale
        q = torch.round(z).clamp(0, qmax)
        q_ste = z + (q - z).detach()
        deq = (q_ste - mid) * scale
        return deq, q.detach().to(torch.int64), scale

    def layer_weight_bits(self, index: int) -> int:
        if self.layer_bits:
            return int(self.layer_bits[index])
        return self.weight_bits

    def _weight(self, linear: nn.Linear, index: int):
        w = linear.weight
        b = fp16_round(linear.bias)
        if not self.qat_weights:
            return w, b
        ste, _codes, _scales = quantize_weight(w, self.layer_weight_bits(index), self.group_size)
        return ste, b

    def positional(self, uv: torch.Tensor, mip: int, max_mip: int) -> torch.Tensor:
        n = uv.shape[0]
        if self.pe_mode == 0 or self.pe_octaves <= 0:
            return uv.new_zeros((n, 0))
        cols = []
        if self.pe_mode == PE_FOURIER:
            p = uv
            for o in range(self.pe_octaves):
                a = p * float(1 << o) * (2.0 * math.pi)
                cols.extend([torch.sin(a[:, 0]), torch.cos(a[:, 0]), torch.sin(a[:, 1]), torch.cos(a[:, 1])])
        elif self.pe_mode == PE_TRIANGLE:
            p = uv.clone()
            for _o in range(self.pe_octaves):
                fx = p[:, 0] - torch.floor(p[:, 0])
                fy = p[:, 1] - torch.floor(p[:, 1])
                fx2 = p[:, 0] + 0.25
                fy2 = p[:, 1] + 0.25
                fx2 = fx2 - torch.floor(fx2)
                fy2 = fy2 - torch.floor(fy2)
                cols.extend([fx * 2 - 1, fy * 2 - 1, fx2 * 2 - 1, fy2 * 2 - 1])
                p = p * 2
            lod = uv.new_full((n,), float(mip) / float(max(max_mip, 1)))
            cols.extend([lod, lod])
        return torch.stack(cols, dim=1)

    def _sample_tiled(self, index: int, ti: int, uv: torch.Tensor, mip: int) -> torch.Tensor:
        bits = self.grid_bits[index]
        z = self.latents[index]
        mid = float((1 << bits) - 1) / 2.0
        qmax = (1 << bits) - 1
        scale = fp16_round(torch.exp(self.tile_log_scales[ti]).clamp(1e-4, 5.0))
        offset = fp16_round(self.tile_offsets[ti].clamp(-2.0, 2.0))
        q = torch.round(z).clamp(0, qmax)
        if not self.qat_latents:
            q_ste = z
        elif self.training and self.quant_proxy == "noise":
            q_ste = z + (torch.rand_like(z) - 0.5)
        else:
            q_ste = z + (q - z).detach()
        _c, h, w = q_ste.shape
        ys = torch.arange(h, device=z.device)
        xs = torch.arange(w, device=z.device)
        sy = torch.clamp(torch.div(ys, self.tile_size, rounding_mode="floor"), max=scale.shape[1] - 1)
        sx = torch.clamp(torch.div(xs, self.tile_size, rounding_mode="floor"), max=scale.shape[2] - 1)
        value = (q_ste - mid) * scale[:, sy[:, None], sx[None, :]] + offset[:, sy[:, None], sx[None, :]]
        if mip <= 0:
            return bilinear_codes_torch(value, uv)
        chain = mip_chain_from_codes(q.detach(), mip + 1)
        k = min(mip, len(chain) - 1)
        fwd = bilinear_tiled_torch(chain[k].float(), uv, scale, offset, self.tile_size, mid, k)
        soft = value
        for _ in range(k):
            nxt = _soft_box(soft)
            if nxt is None:
                break
            soft = nxt
        soft_s = bilinear_codes_torch(soft, uv)
        return fwd.detach() + soft_s - soft_s.detach()

    def features(self, uv: torch.Tensor, mip: int, max_mip: int) -> torch.Tensor:
        parts = []
        for i in range(len(self.latents)):
            ti = self._tile_param_of[i]
            if ti >= 0:
                parts.append(self._sample_tiled(i, ti, uv, mip))
                continue
            deq, code, scale = self.quant_grid(i)
            bits = self.grid_bits[i]
            mid = float((1 << bits) - 1) / 2.0
            if code is None or mip == 0:
                field = deq
                if mip > 0 and code is None:
                    for _ in range(mip):
                        field = _soft_box(field)
                        if field is None:
                            break
                    if field is None:
                        field = deq
                sampled = bilinear_codes_torch(field, uv)
            else:
                chain = mip_chain_from_codes(code, max(mip + 1, 1))
                k = min(mip, len(chain) - 1)
                sampled_codes = bilinear_codes_torch(chain[k].float(), uv)
                fwd = (sampled_codes - mid) * scale.view(1, -1)
                # Gradient matches a box average of the straight-through field.
                soft = deq
                for _ in range(k):
                    nxt = _soft_box(soft)
                    if nxt is None:
                        break
                    soft = nxt
                soft_s = bilinear_codes_torch(soft, uv)
                sampled = fwd.detach() + soft_s - soft_s.detach()
            parts.append(sampled)
        feat = torch.cat(parts, dim=1) if parts else uv.new_zeros((uv.shape[0], 0))
        pe = self.positional(uv, mip, max_mip)
        if pe.shape[1]:
            feat = torch.cat([feat, pe], dim=1)
        return feat

    def forward_mlp(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        last = len(self.linears) - 1
        for i, lin in enumerate(self.linears):
            w, b = self._weight(lin, i)
            y = F.linear(h, w, b)
            if i != last:
                y = _activate(y, self.activation, self.activation_requant)
            h = y
        return h

    def forward(self, uv: torch.Tensor, mip: int = 0, max_mip: int = 1) -> torch.Tensor:
        return self.forward_mlp(self.features(uv, mip, max_mip))


def _activate(y: torch.Tensor, kind: int, requant: bool) -> torch.Tensor:
    if kind == ACT_IDENTITY:
        z = y
    elif kind == ACT_RELU:
        z = F.relu(y)
    elif kind == ACT_HGELU:
        z = torch.minimum(y, torch.tensor(3.0, device=y.device, dtype=y.dtype)) * torch.clamp(
            y / 3.0 + 0.5, 0.0, 1.0
        )
    else:
        raise ValueError(kind)
    if requant:
        q = torch.round(z * 80.0 - 113.0).clamp(-128, 127)
        # straight-through onto z
        return z + (q - z).detach()
    return z
