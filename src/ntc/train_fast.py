"""Batched training path. Off unless ``TrainConfig.fast`` is set.

``train_max`` does not set that flag. Its checkpoints and json files stay
on the eager loop in ``encode.py``.

Seeds that share a grid shape are one stacked module: latent grids are
``[seed, C, H, W]`` and each MLP matrix is a ``bmm``. One step launches the
gather and the GEMM once for the whole pool. Bad seeds are dropped at
``halve_steps`` (default 250) by successive halving. The MLP can run under
bf16 or fp16 autocast while latents, master weights, and the loss stay fp32.
``torch.compile`` and a CUDA graph are attempted on CUDA and skipped, with
the error recorded, when they fail. Triton is not required. A fused gather
is recommended only when a CUDA profile says the GPU is busy inside the
bilinear gather.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ntc.encode import (
    _ACT,
    _PE,
    TrainConfig,
    _color_mip,
    _loss,
    _mip_weight,
    _pca_init,
    export_network,
)
from ntc.metrics import group_metrics, psnr_u8
from ntc.model import NtcNet, _soft_box, integer_box
from ntc.quant import fp16_round, quantize_weight


def scaled_lr(lr: float, batch_pixels: int, ref_pixels: int) -> float:
    """Linear batch scale. ``ref_pixels`` of 0 leaves ``lr`` unchanged."""
    if ref_pixels <= 0 or batch_pixels <= 0:
        return float(lr)
    return float(lr) * (float(batch_pixels) / float(ref_pixels))


# A run that finishes within this many dB of its init did not train the latent.
NEAR_INIT_DB = 4.0


def fullres_batch_floor(
    height: int,
    width: int,
    grids: list,
    batch_pixels: int,
    batch_mip: str,
) -> tuple[int, str]:
    """Keep a dense mip0 sample when a latent is as large as the image.

    ``262144`` area samples on a 2048 grid touch 1/16 of the texels each step.
    That schedule left MetalPlates013 b5 at 11.87 dB (init 10.02) and b10 at
    12.45 dB (init 9.43). The same b5 shape at a flat ``1048576`` batch had
    reached 28.84 dB. A 1024 grid still trains at ``262144``. ``rd_lambda``
    is 0 on this path, so the loss has no rate term.
    """
    if not grids:
        return int(batch_pixels), str(batch_mip)
    finest = max(int(g[0]) * int(g[1]) for g in grids)
    image = int(height) * int(width)
    if finest < max(image // 2, 1):
        return int(batch_pixels), str(batch_mip)
    if int(batch_pixels) <= 0:
        return int(batch_pixels), str(batch_mip)
    floor = max(1, image // 4)
    if int(batch_pixels) >= floor and str(batch_mip) != "area":
        return int(batch_pixels), str(batch_mip)
    return max(int(batch_pixels), floor), "flat"


def mip_sample_count(batch_pixels: int, mip: int, count: int, mode: str = "flat") -> int:
    """Pixels to draw at this mip. 0 means every pixel.

    ``flat`` uses ``batch_pixels`` whenever the mip is larger. ``area`` uses
    ``batch_pixels >> (2 * mip)``, so a coarse mip is not a second full frame.
    """
    if batch_pixels <= 0 or count <= 0:
        return 0
    if mode == "area":
        budget = max(1, int(batch_pixels) >> (2 * int(mip)))
    elif mode == "flat":
        budget = int(batch_pixels)
    else:
        raise ValueError(mode)
    if budget >= count:
        return 0
    return budget


def _state_matches_grids(state: dict, grids: list) -> bool:
    """True when a checkpoint's latent tensors have the selected shapes."""
    for index, (gw, gh, ch, _bits) in enumerate(grids):
        tensor = state.get(f"latents.{index}")
        if tensor is None or tuple(tensor.shape) != (ch, gh, gw):
            return False
    n = sum(1 for key in state if key.startswith("latents.") and key.count(".") == 1)
    return n == len(grids)


def _mesh_uv(height: int, width: int, device: torch.device) -> torch.Tensor:
    ys = torch.arange(height, device=device)
    xs = torch.arange(width, device=device)
    yg, xg = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack(
        ((xg.reshape(-1).float() + 0.5) / width, (yg.reshape(-1).float() + 0.5) / height),
        dim=1,
    )


def bilinear_batched(codes: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """``codes`` ``[S,C,H,W]``, ``uv`` ``[N,2]`` in ``[0,1]``. Returns ``[S,N,C]``.

    Same taps as ``bilinear_codes_torch``: pixel centers at integer + 0.5,
    wrapped with remainder. CUDA may use a Triton kernel after a numeric check.
    """
    if codes.is_cuda:
        from ntc.triton_bilinear import try_triton_bilinear

        fused = try_triton_bilinear(codes, uv)
        if fused is not None:
            return fused
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

    def gather(yy: torch.Tensor, xx: torch.Tensor) -> torch.Tensor:
        idx = yy * width + xx
        return plane[:, :, idx].permute(0, 2, 1)

    c00 = gather(y0w, x0w)
    c01 = gather(y0w, x1w)
    c10 = gather(y1w, x0w)
    c11 = gather(y1w, x1w)
    a = c00 + (c01 - c00) * tx
    b = c10 + (c11 - c10) * tx
    return a + (b - a) * ty


def _leading_box(values: torch.Tensor, fn) -> torch.Tensor | None:
    seeds, channels, height, width = values.shape
    flat = values.reshape(seeds * channels, height, width)
    out = fn(flat)
    if out is None:
        return None
    return out.reshape(seeds, channels, out.shape[-2], out.shape[-1])


def quantize_weight_batched(weight: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    """``weight`` is ``[S, out, in]``. Per-row scales match ``quantize_weight``."""
    seeds, out, inn = weight.shape
    flat = weight.reshape(seeds * out, inn)
    ste, _codes, _scales = quantize_weight(flat, bits, group_size)
    return ste.reshape(seeds, out, inn)


class StackedTrain(nn.Module):
    """One module for ``S`` independent networks of the same shape."""

    def __init__(self, nets: list[NtcNet]):
        super().__init__()
        if not nets:
            raise ValueError("empty seed stack")
        ref = nets[0]
        self.grid_bits = list(ref.grid_bits)
        self.activation = ref.activation
        self.weight_bits = ref.weight_bits
        self.group_size = ref.group_size
        self.layer_bits = tuple(ref.layer_bits)
        self.n_seeds = len(nets)
        self.latents = nn.ParameterList(
            [
                nn.Parameter(torch.stack([n.latents[i].detach() for n in nets], 0))
                for i in range(len(ref.latents))
            ]
        )
        self.log_scales = nn.ParameterList(
            [
                nn.Parameter(torch.stack([n.log_scales[i].detach() for n in nets], 0))
                for i in range(len(ref.latents))
            ]
        )
        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.stack([n.linears[i].weight.detach() for n in nets], 0))
                for i in range(len(ref.linears))
            ]
        )
        self.biases = nn.ParameterList(
            [
                nn.Parameter(torch.stack([n.linears[i].bias.detach() for n in nets], 0))
                for i in range(len(ref.linears))
            ]
        )
        requires = [p.requires_grad for p in ref.linears.parameters()]
        # linears are weight, bias, weight, bias, ...
        flat = []
        for weight, bias in zip(self.weights, self.biases, strict=True):
            flat.append(weight)
            flat.append(bias)
        for param, flag in zip(flat, requires, strict=True):
            param.requires_grad_(flag)
        self.quantize = True

    def layer_bits_of(self, index: int) -> int:
        if self.layer_bits:
            return int(self.layer_bits[index])
        return int(self.weight_bits)

    def features(self, uv: torch.Tensor, mip: int) -> torch.Tensor:
        """Latent samples in fp32, ``[S, N, features]``."""
        feats = []
        for i, lat in enumerate(self.latents):
            bits = self.grid_bits[i]
            mid = float((1 << bits) - 1) / 2.0
            qmax = (1 << bits) - 1
            scale = fp16_round(torch.exp(self.log_scales[i]).clamp(1e-4, 5.0))
            if self.quantize:
                q = torch.round(lat).clamp(0, qmax)
                q_ste = lat + (q - lat).detach()
                deq = (q_ste - mid) * scale[:, :, None, None]
            else:
                q = None
                deq = (lat - mid) * scale[:, :, None, None]
            if mip <= 0 or q is None:
                sampled = bilinear_batched(deq, uv)
            else:
                code = q.detach().to(torch.int64)
                chain = [code]
                cur = code
                while len(chain) <= mip:
                    nxt = _leading_box(cur, integer_box)
                    if nxt is None:
                        break
                    if nxt.shape[-2] == cur.shape[-2] and nxt.shape[-1] == cur.shape[-1]:
                        break
                    chain.append(nxt)
                    cur = nxt
                k = min(mip, len(chain) - 1)
                sampled_codes = bilinear_batched(chain[k].float(), uv)
                fwd = (sampled_codes - mid) * scale[:, None, :]
                soft = deq
                for _ in range(k):
                    nxt = _leading_box(soft, _soft_box)
                    if nxt is None:
                        break
                    soft = nxt
                soft_s = bilinear_batched(soft, uv)
                sampled = fwd.detach() + soft_s - soft_s.detach()
            feats.append(sampled)
        if not feats:
            return uv.new_zeros((self.n_seeds, uv.shape[0], 0))
        return torch.cat(feats, dim=-1)

    def mlp(self, hidden: torch.Tensor, amp_dtype: torch.dtype | None, device_type: str) -> torch.Tensor:
        """fp32 master weights. ``amp_dtype`` autocasts the GEMM only."""
        last = len(self.weights) - 1
        for li, (weight, bias) in enumerate(zip(self.weights, self.biases, strict=True)):
            if self.quantize:
                ste = quantize_weight_batched(weight, self.layer_bits_of(li), self.group_size)
            else:
                ste = weight
            bb = fp16_round(bias)
            if amp_dtype is None:
                hidden = torch.bmm(hidden, ste.transpose(1, 2)) + bb[:, None, :]
                if li != last and self.activation == 1:
                    hidden = F.relu(hidden)
            else:
                with torch.autocast(device_type=device_type, dtype=amp_dtype):
                    hidden = torch.bmm(hidden, ste.transpose(1, 2)) + bb[:, None, :]
                    if li != last and self.activation == 1:
                        hidden = F.relu(hidden)
            if li != last and self.activation not in (0, 1):
                raise ValueError("stacked path supports identity and relu")
        return hidden

    def predict(
        self,
        uv: torch.Tensor,
        mip: int,
        max_mip: int,
        amp_dtype: torch.dtype | None = None,
        device_type: str = "cpu",
    ) -> torch.Tensor:
        """``[S, N, channels]``. ``max_mip`` is accepted so call sites match ``NtcNet``."""
        del max_mip
        return self.mlp(self.features(uv, mip), amp_dtype, device_type)

    def slice_seeds(self, index: torch.Tensor) -> StackedTrain:
        """New module with the selected seeds. ``index`` is a long tensor."""
        out = self._blank_like(int(index.shape[0]))
        with torch.no_grad():
            for dst, src in zip(out.latents, self.latents, strict=True):
                dst.copy_(src.index_select(0, index))
            for dst, src in zip(out.log_scales, self.log_scales, strict=True):
                dst.copy_(src.index_select(0, index))
            for dst, src in zip(out.weights, self.weights, strict=True):
                dst.copy_(src.index_select(0, index))
            for dst, src in zip(out.biases, self.biases, strict=True):
                dst.copy_(src.index_select(0, index))
        return out

    def _blank_like(self, seeds: int) -> StackedTrain:
        ref_nets = self._dummy_nets(seeds)
        return StackedTrain(ref_nets)

    def _dummy_nets(self, seeds: int) -> list[NtcNet]:
        device = self.latents[0].device
        grids = []
        for i, lat in enumerate(self.latents):
            _s, channels, height, width = lat.shape
            grids.append((width, height, channels, self.grid_bits[i]))
        hidden = []
        for weight in list(self.weights)[:-1]:
            hidden.append(int(weight.shape[1]))
        image_channels = int(self.weights[-1].shape[1])
        nets = []
        for _ in range(seeds):
            net = NtcNet(
                image_channels=image_channels,
                grids=grids,
                hidden=hidden,
                activation=self.activation,
                pe_mode=0,
                pe_octaves=0,
                weight_bits=self.weight_bits,
                group_size=self.group_size,
                quant_proxy="ste",
                activation_requant=False,
                qat_latents=True,
                qat_weights=True,
                layer_bits=self.layer_bits,
            ).to(device)
            nets.append(net)
        return nets

    def write_seed(self, index: int, net: NtcNet) -> None:
        with torch.no_grad():
            for i, lat in enumerate(net.latents):
                lat.copy_(self.latents[i][index])
                net.log_scales[i].copy_(self.log_scales[i][index])
            for i, lin in enumerate(net.linears):
                lin.weight.copy_(self.weights[i][index])
                lin.bias.copy_(self.biases[i][index])


def _can_stack(cfg: TrainConfig) -> bool:
    return (
        cfg.tile_size == 0
        and not cfg.act_int8
        and cfg.pe == "none"
        and cfg.activation in ("relu", "identity")
        and not cfg.activation_requant
        and cfg.rd_lambda == 0.0
    )


def _build_nets(
    images: torch.Tensor,
    cfg: TrainConfig,
    grids: list,
    seeds: list[int],
    device: torch.device,
) -> list[NtcNet]:
    channels = int(images.shape[0]) if images.ndim == 3 else int(images.shape[1])
    target = images if images.ndim == 3 else images[0]
    nets = []
    for seed in seeds:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        net = NtcNet(
            image_channels=channels,
            grids=grids,
            hidden=list(cfg.hidden),
            activation=_ACT[cfg.activation],
            pe_mode=_PE[cfg.pe],
            pe_octaves=cfg.pe_octaves,
            weight_bits=cfg.weight_bits,
            group_size=cfg.group_size,
            quant_proxy=cfg.quant_proxy,
            activation_requant=cfg.activation_requant,
            qat_latents=cfg.qat_latents,
            qat_weights=cfg.qat_weights,
            tile_size=cfg.tile_size,
            layer_bits=cfg.layer_bits,
            act_int8=cfg.act_int8,
        ).to(device)
        if cfg.pca_init or cfg.latent_init == "analytic":
            _pca_init(net, target)
        if cfg.warm_start:
            from ntc.train_steps import apply_warm_start

            apply_warm_start(net, cfg.warm_start, cfg.freeze_decoder)
        if cfg.latent_init == "analytic":
            from ntc.train_steps import analytic_project

            analytic_project(net, target)
        nets.append(net)
    return nets


def _prep_mips(
    images: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
) -> list[dict]:
    """``images`` is ``[S,C,H,W]`` or ``[C,H,W]`` (broadcast to one seed)."""
    if images.ndim == 3:
        images = images.unsqueeze(0)
    out = []
    level = images
    for mip in range(cfg.mip_levels):
        _s, _c, height, width = level.shape
        uv = _mesh_uv(height, width, device)
        tgt = level.permute(0, 2, 3, 1).reshape(level.shape[0], height * width, level.shape[1])
        out.append(
            {
                "uv": uv,
                "tgt": tgt.contiguous(),
                "weight": _mip_weight(cfg, mip),
                "height": height,
                "width": width,
            }
        )
        nxt = _leading_box(level, _soft_box)
        if nxt is None:
            break
        level = nxt
    return out


def _clip_no_sync(params, max_norm: float) -> None:
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    norms = torch.stack([g.detach().float().norm(2) for g in grads])
    total = norms.norm(2)
    coef = torch.clamp(total.new_tensor(max_norm) / (total + 1e-6), max=1.0)
    for grad in grads:
        grad.mul_(coef)


def _adam_on(stacked: StackedTrain, cfg: TrainConfig, capturable: bool):
    latent = [p for p in list(stacked.latents.parameters()) + list(stacked.log_scales.parameters()) if p.requires_grad]
    mlp = [p for p in list(stacked.weights.parameters()) + list(stacked.biases.parameters()) if p.requires_grad]
    groups = []
    if latent:
        groups.append({"params": latent, "lr": cfg.lr * cfg.latent_lr_mult, "name": "latent"})
    if mlp:
        groups.append({"params": mlp, "lr": cfg.lr, "name": "mlp"})
    if not groups:
        raise ValueError("no trainable parameters")
    if cfg.optimizer == "lion":
        from ntc.train_steps import Lion

        # Lion's stable step size is smaller than Adam's. The caller sets lr.
        return Lion(groups, lr=groups[0]["lr"])
    kwargs = {"capturable": capturable} if capturable else {}
    return torch.optim.Adam(groups, **kwargs)


def _copy_adam_slice(src: torch.optim.Optimizer, dst: torch.optim.Optimizer, index: torch.Tensor) -> None:
    src_params = [p for group in src.param_groups for p in group["params"]]
    dst_params = [p for group in dst.param_groups for p in group["params"]]
    for old, new in zip(src_params, dst_params, strict=True):
        state = src.state.get(old)
        if not state:
            continue
        copied = {}
        for key, value in state.items():
            if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == old.shape[0]:
                copied[key] = value.index_select(0, index.to(value.device)).clone()
            else:
                copied[key] = value
        dst.state[new] = copied


def _clamp_latents(stacked: StackedTrain) -> None:
    """Keep code-unit latents inside the representable bins.

    The straight-through estimator passes gradient outside the bin range, so a
    long run at a flat learning rate can push every code to 0 or to the max.
    The exported file is then a smooth field: low PSNR and a very small rate.
    """
    with torch.no_grad():
        for lat, bits in zip(stacked.latents, stacked.grid_bits, strict=True):
            lat.clamp_(0, (1 << int(bits)) - 1)


def _step_lr(cfg: TrainConfig, step: int) -> float:
    if cfg.schedule == "onecycle":
        peak = cfg.lr * max(float(cfg.onecycle_peak), 1.0)
        t = step / max(cfg.steps - 1, 1)
        if t < 0.3:
            return peak * (t / 0.3) if t > 0 else peak * 0.05
        u = (t - 0.3) / 0.7
        cos = 0.5 * (1.0 + math.cos(math.pi * u))
        return max(peak * cos, cfg.lr * 0.05)
    if cfg.warmup_steps <= 0 and cfg.schedule == "cosine":
        cos = 0.5 * (1.0 + math.cos(math.pi * step / max(cfg.steps, 1)))
        return max(cfg.lr * cos, cfg.lr * 0.05)
    warm = max(cfg.warmup_steps, 0)
    if step < warm:
        return cfg.lr * float(step + 1) / float(max(warm, 1))
    span = max(cfg.steps - warm, 1)
    t = min(1.0, float(step - warm) / float(span))
    if cfg.schedule == "step":
        lr = cfg.lr
        if t >= 0.5:
            lr *= 0.5
        if t >= 0.75:
            lr *= 0.5
        return lr
    cos = 0.5 * (1.0 + math.cos(math.pi * t))
    return max(cfg.lr * cos, cfg.lr * 0.05)


def _loss_from_mips(
    stacked: StackedTrain,
    mips: list[dict],
    cfg: TrainConfig,
    semantics: list[int],
    amp_dtype: torch.dtype | None,
    batch_pixels: int,
    step: int = 0,
) -> torch.Tensor:
    total = None
    chosen = list(enumerate(mips))
    if cfg.coarse_to_fine and step < int(cfg.steps * 0.4) and len(chosen) > 1:
        chosen = chosen[-1:]
    for mip, slot in chosen:
        uv = slot["uv"]
        tgt = slot["tgt"]
        count = uv.shape[0]
        budget = mip_sample_count(batch_pixels, mip, count, cfg.batch_mip)
        if budget:
            if cfg.importance and mip == 0:
                from ntc.train_steps import importance_index

                cache = getattr(stacked, "_importance", None)
                if cache is None:
                    cache = {}
                    stacked._importance = cache
                idx = importance_index(stacked, uv, tgt, budget, step, cache)
            else:
                idx = torch.randint(0, count, (budget,), device=uv.device)
            uv = uv.index_select(0, idx)
            tgt = tgt.index_select(1, idx)
        pred = stacked.predict(
            uv,
            mip,
            max(cfg.mip_levels - 1, 1),
            amp_dtype,
            stacked.latents[0].device.type,
        )
        # per-seed mse, then the historical channel weights via _loss on a
        # flattened batch would mix seeds. Apply _loss per seed only when
        # channel weights are on. The default path is a plain mean.
        if cfg.channel_weighting or cfg.normal_cosine:
            pieces = []
            for s in range(pred.shape[0]):
                pieces.append(_loss(pred[s].float(), tgt[s], semantics, cfg))
            per = torch.stack(pieces)
        else:
            per = (pred.float() - tgt).square().mean(dim=(1, 2))
        term = per * float(slot["weight"])
        total = term if total is None else total + term
    if total is None:
        return stacked.latents[0].new_zeros((stacked.n_seeds,))
    return total


def _run_loop(
    stacked: StackedTrain,
    opt: torch.optim.Optimizer,
    mips: list[dict],
    cfg: TrainConfig,
    semantics: list[int],
    steps: int,
    start_step: int,
    amp_dtype: torch.dtype | None,
    device_type: str,
    use_graph: bool,
) -> dict:
    """Train ``steps`` Adam steps starting at ``start_step``. No ``.item()``."""
    batch = int(cfg.batch_pixels or 0)
    lr_scale = 1.0
    if cfg.lr_batch_ref and batch:
        lr_scale = scaled_lr(1.0, batch, cfg.lr_batch_ref)
    params = [p for group in opt.param_groups for p in group["params"]]
    graph = None
    graph_error = None
    static_loss = None

    def one_step(step: int, replay: bool) -> torch.Tensor:
        lr = _step_lr(cfg, step) * lr_scale
        for group in opt.param_groups:
            if group.get("name") == "latent":
                group["lr"] = lr * cfg.latent_lr_mult
            else:
                group["lr"] = lr
        if cfg.qat_late:
            stacked.quantize = step >= int(cfg.steps * 0.7)
        if cfg.coarse_to_fine:
            from ntc.train_steps import coarse_lowpass

            state = getattr(stacked, "_coarse", None)
            if state is None:
                state = {}
                stacked._coarse = state
            coarse_lowpass(stacked, step, cfg.steps, state)
        opt.zero_grad(set_to_none=not replay)
        per = _loss_from_mips(stacked, mips, cfg, semantics, amp_dtype, batch, step)
        loss = per.sum()
        if amp_dtype == torch.float16 and device_type == "cuda":
            # GradScaler is constructed by the caller and stashed on cfg via a
            # list so this closure can see it. fp16 only.
            scaler = getattr(one_step, "scaler", None)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                _clip_no_sync(params, 1.0)
                scaler.step(opt)
                scaler.update()
                _clamp_latents(stacked)
                return loss
        loss.backward()
        if cfg.optimizer == "shampoo":
            from ntc.train_steps import shampoo_precondition

            shampoo_precondition(params)
        _clip_no_sync(params, 1.0)
        opt.step()
        _clamp_latents(stacked)
        return loss

    if amp_dtype == torch.float16 and device_type == "cuda":
        one_step.scaler = torch.amp.GradScaler("cuda")

    compiled = False
    compile_error = None
    loop_graph_from_compile = False
    import sys as _sys

    linux_cuda = device_type == "cuda" and _sys.platform.startswith("linux")
    compile_mode = "reduce-overhead" if linux_cuda else "default"
    want_compile = cfg.compile_step == "on" or (cfg.compile_step == "auto" and device_type == "cuda")
    if want_compile and not use_graph:
        try:
            compiled_loss = torch.compile(_loss_from_mips, dynamic=False, mode=compile_mode)
            # Touch it once outside the timed region of the caller. A failure
            # here is recorded and the eager function stays in use.
            probe = compiled_loss(stacked, mips, cfg, semantics, amp_dtype, batch)
            probe.sum().backward()
            opt.zero_grad(set_to_none=True)
            compiled = True
            if compile_mode == "reduce-overhead":
                # Inductor reduce-overhead replays the step as a CUDA/HIP graph.
                loop_graph_from_compile = True
            # The compiled function replaces the fp32 loss path only.
            def one_step(step: int, replay: bool) -> torch.Tensor:  # noqa: F811
                lr = _step_lr(cfg, step) * lr_scale
                for group in opt.param_groups:
                    if group.get("name") == "latent":
                        group["lr"] = lr * cfg.latent_lr_mult
                    else:
                        group["lr"] = lr
                opt.zero_grad(set_to_none=True)
                per = compiled_loss(stacked, mips, cfg, semantics, amp_dtype, batch, step)
                loss = per.sum()
                loss.backward()
                if cfg.optimizer == "shampoo":
                    from ntc.train_steps import shampoo_precondition

                    shampoo_precondition(params)
                _clip_no_sync(params, 1.0)
                opt.step()
                _clamp_latents(stacked)
                return loss
        except Exception as exc:
            compile_error = str(exc)
            compiled = False
            opt.zero_grad(set_to_none=True)

    try_graph = device_type == "cuda" and cfg.cuda_graph != "off" and (use_graph or (linux_cuda and not compiled))
    if try_graph:
        try:
            for warm in range(3):
                one_step(start_step + warm, replay=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_loss = one_step(start_step, replay=True)
        except Exception as exc:
            graph_error = str(exc)
            graph = None

    t0 = time.perf_counter()
    if graph is not None:
        for step in range(start_step, start_step + steps):
            # The captured step bakes the learning rate. Replay is the fast
            # path for a fixed lr. Cosine schedules fall back when the caller
            # passes use_graph False. See encode_arrays_fast.
            graph.replay()
        _ = static_loss
    else:
        for step in range(start_step, start_step + steps):
            one_step(step, replay=False)
    if device_type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "seconds": elapsed,
        "steps": steps,
        "compiled": compiled,
        "compile_error": compile_error,
        "cuda_graph": graph is not None or loop_graph_from_compile,
        "compile_mode": compile_mode if compiled else None,
        "cuda_graph_error": graph_error,
        "steps_per_s": (steps / elapsed) if elapsed > 0 else None,
    }


@torch.no_grad()
def _psnr_per_seed(
    stacked: StackedTrain,
    images: torch.Tensor,
    device: torch.device,
) -> list[float]:
    """Mip 0 8-bit PSNR for each seed. One host sync for the whole vector."""
    if images.ndim == 3:
        images = images.unsqueeze(0).expand(stacked.n_seeds, -1, -1, -1)
    _s, _c, height, width = images.shape
    uv = _mesh_uv(height, width, device)
    pred = stacked.predict(uv, 0, 1).float()
    tgt = images.permute(0, 2, 3, 1).reshape(images.shape[0], height * width, images.shape[1])
    q = torch.round(pred.clamp(0, 1) * 255.0) / 255.0
    tq = torch.round(tgt.clamp(0, 1) * 255.0) / 255.0
    mse = (q - tq).square().mean(dim=(1, 2))
    mse_cpu = mse.detach().float().cpu()
    out = []
    for value in mse_cpu.tolist():
        if value <= 0.0:
            out.append(100.0)
        else:
            out.append(float(min(100.0, 10.0 * math.log10(1.0 / value))))
    return out


def _halving_rung(cfg: TrainConfig) -> int:
    if cfg.halve_steps > 0:
        return int(cfg.halve_steps)
    if cfg.seed_pool > 1:
        return 250
    return 0


def _single_adam(net: NtcNet, cfg: TrainConfig) -> torch.optim.Adam:
    latent = list(net.latents.parameters()) + list(net.log_scales.parameters())
    latent += list(net.tile_log_scales.parameters()) + list(net.tile_offsets.parameters())
    return torch.optim.Adam(
        [
            {"params": latent, "lr": cfg.lr * cfg.latent_lr_mult},
            {"params": net.linears.parameters(), "lr": cfg.lr},
        ]
    )


def _transfer_moments(src_param: nn.Parameter, src_opt: torch.optim.Optimizer, dst_param: nn.Parameter, dst_opt: torch.optim.Optimizer, squeeze: bool) -> None:
    state = src_opt.state.get(src_param)
    if not state:
        return
    copied = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            copied[key] = value
            continue
        if squeeze and value.ndim == dst_param.ndim + 1 and value.shape[0] == 1:
            copied[key] = value.squeeze(0).clone()
        elif (not squeeze) and value.shape == dst_param.shape[1:]:
            copied[key] = value.unsqueeze(0).clone()
        else:
            copied[key] = value.clone()
    dst_opt.state[dst_param] = copied


def _save_resume_ckpt(stacked: StackedTrain, opt: torch.optim.Optimizer, cfg: TrainConfig, step: int) -> None:
    if not cfg.ckpt or stacked.n_seeds != 1:
        return
    net = stacked._dummy_nets(1)[0]
    stacked.write_seed(0, net)
    single = _single_adam(net, cfg)
    pairs = []
    for i, lat in enumerate(net.latents):
        pairs.append((stacked.latents[i], lat))
        pairs.append((stacked.log_scales[i], net.log_scales[i]))
    for i, lin in enumerate(net.linears):
        pairs.append((stacked.weights[i], lin.weight))
        pairs.append((stacked.biases[i], lin.bias))
    for src, dst in pairs:
        _transfer_moments(src, opt, dst, single, squeeze=True)
    torch.save({"model": net.state_dict(), "opt": single.state_dict(), "step": step}, cfg.ckpt)


def _load_resume_into_stack(stacked: StackedTrain, opt: torch.optim.Optimizer, ckpt: dict, cfg: TrainConfig, net: NtcNet) -> torch.optim.Optimizer:
    if not ckpt.get("opt"):
        return opt
    single = _single_adam(net, cfg)
    single.load_state_dict(ckpt["opt"])
    pairs = []
    for i, lat in enumerate(net.latents):
        pairs.append((lat, stacked.latents[i]))
        pairs.append((net.log_scales[i], stacked.log_scales[i]))
    for i, lin in enumerate(net.linears):
        pairs.append((lin.weight, stacked.weights[i]))
        pairs.append((lin.bias, stacked.biases[i]))
    for src, dst in pairs:
        _transfer_moments(src, single, dst, opt, squeeze=False)
    return opt


def encode_arrays_fast(
    images: np.ndarray,
    semantics: list[int],
    cfg: TrainConfig,
) -> tuple[bytes, dict]:
    """Same return contract as ``encode_arrays`` for the keys ``train_max`` reads."""
    from ntc.container import parse_texture_set, rate_report
    from ntc.device import device_report, pick_device
    from ntc.grids import select_grids

    if not _can_stack(cfg):
        slow = replace(cfg, fast=False)
        from ntc.encode import encode_arrays

        blob, stats = encode_arrays(images, semantics, slow)
        stats = dict(stats)
        stats["fast"] = False
        stats["fast_fallback"] = "tile, positional encoding, activation requant, act_int8, or rd_lambda"
        return blob, stats

    device = pick_device(cfg.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    channels, height, width = images.shape
    hidden = list(cfg.hidden)
    pe_mode = _PE[cfg.pe]
    if cfg.grids is None:
        select_bpp = float(cfg.target_bpp)
        if cfg.rate == "latent" and 0 < float(cfg.latent_rate_scale) < 1:
            select_bpp = select_bpp / float(cfg.latent_rate_scale)
        choices = select_grids(
            width,
            height,
            select_bpp,
            hidden,
            channels,
            pe_mode,
            cfg.pe_octaves,
            cfg.weight_bits,
            cfg.group_size,
            adaptive_pool=max(1, cfg.adaptive_pool),
            rate=cfg.rate,
            min_bits=int(cfg.min_latent_bits),
        )
        grids = choices[0]
        if cfg.adaptive_pool > 1 and cfg.steps > 0 and len(choices) > 1:
            from ntc.encode import encode_arrays

            ranked = []
            for candidate in choices:
                probe = replace(
                    cfg,
                    fast=False,
                    grids=candidate,
                    steps=0,
                    adaptive_pool=1,
                    seed_pool=1,
                    with_ssim=False,
                    log_every=10**9,
                )
                _blob, probe_stats = encode_arrays(images, semantics, probe)
                ranked.append((probe_stats["metrics"]["psnr_u8"], candidate))
            ranked.sort(key=lambda item: -item[0])
            grids = ranked[0][1]
    else:
        grids = cfg.grids

    batch_pixels, batch_mip = fullres_batch_floor(
        height, width, grids, int(cfg.batch_pixels or 0), str(cfg.batch_mip)
    )
    if batch_pixels != int(cfg.batch_pixels or 0) or batch_mip != str(cfg.batch_mip):
        print(
            f"fullres batch {cfg.batch_pixels} {cfg.batch_mip} -> {batch_pixels} {batch_mip}",
            flush=True,
        )
        cfg = replace(cfg, batch_pixels=int(batch_pixels), batch_mip=str(batch_mip))

    if cfg.resume:
        preview = torch.load(cfg.resume, map_location="cpu", weights_only=False)
        if not _state_matches_grids(preview.get("model", {}), grids):
            print("resume grid mismatch, training from scratch", cfg.resume, flush=True)
            cfg = replace(cfg, resume="")
    seeds = [cfg.seed + i for i in range(max(1, cfg.seed_pool if cfg.steps > 0 else 1))]
    if cfg.resume:
        seeds = [cfg.seed]
    target = torch.from_numpy(np.ascontiguousarray(images)).to(device)
    nets = _build_nets(target, cfg, grids, seeds, device)
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device, weights_only=False)
        nets[0].load_state_dict(ckpt["model"])
    init_states = [{k: v.detach().clone() for k, v in net.state_dict().items()} for net in nets]
    stacked = StackedTrain(nets)
    amp_dtype = None
    if cfg.precision == "bf16":
        amp_dtype = torch.bfloat16
    elif cfg.precision == "fp16":
        amp_dtype = torch.float16
    if device.type not in ("cuda", "cpu"):
        amp_dtype = None
    mips = _prep_mips(target, cfg, device)
    # Broadcast the single image across seeds inside the loss by expanding tgt.
    for slot in mips:
        if slot["tgt"].shape[0] == 1 and stacked.n_seeds > 1:
            slot["tgt"] = slot["tgt"].expand(stacked.n_seeds, -1, -1).contiguous()
    capturable = False
    use_graph = False
    if device.type == "cuda" and cfg.cuda_graph == "on":
        capturable = True
        use_graph = cfg.schedule == "constant" or (cfg.warmup_steps <= 0 and cfg.steps <= 1)
    elif device.type == "cuda" and cfg.cuda_graph == "auto":
        # A cosine schedule changes the learning rate every step. A captured
        # graph would freeze the first rate. Leave the graph off unless the
        # schedule is constant.
        if cfg.schedule == "constant":
            capturable = True
            use_graph = True
    opt = _adam_on(stacked, cfg, capturable=capturable)
    start_step = 0
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device, weights_only=False)
        start_step = int(ckpt.get("step", -1)) + 1
        if stacked.n_seeds == 1:
            opt = _load_resume_into_stack(stacked, opt, ckpt, cfg, nets[0])

    rung = _halving_rung(cfg)
    trace = [{"step": start_step, "alive": list(seeds)}]
    done_steps = start_step
    loop_info = {
        "seconds": 0.0,
        "compiled": False,
        "compile_error": None,
        "cuda_graph": False,
        "cuda_graph_error": None,
        "hit_target": False,
    }
    stopped_early = False
    best_val = -1e9
    images_bank = target.unsqueeze(0).expand(stacked.n_seeds, -1, -1, -1).contiguous()
    while done_steps < cfg.steps and stacked.n_seeds > 1 and rung > 0:
        span = min(rung, cfg.steps - done_steps)
        info = _run_loop(
            stacked,
            opt,
            mips,
            cfg,
            semantics,
            span,
            done_steps,
            amp_dtype,
            device.type,
            use_graph=False,
        )
        loop_info = _merge_info(loop_info, info)
        done_steps += span
        scores = _psnr_per_seed(stacked, images_bank, device)
        order = sorted(range(stacked.n_seeds), key=lambda i: -scores[i])
        keep_n = max(1, stacked.n_seeds // 2)
        keep = order[:keep_n]
        if keep_n == stacked.n_seeds:
            trace.append({"step": done_steps, "alive": [seeds[i] for i in keep], "psnr_u8": scores})
            break
        index = torch.tensor(keep, device=device, dtype=torch.long)
        new_stacked = stacked.slice_seeds(index)
        new_opt = _adam_on(new_stacked, cfg, capturable=False)
        _copy_adam_slice(opt, new_opt, index.cpu())
        seeds = [seeds[i] for i in keep]
        init_states = [init_states[i] for i in keep]
        stacked = new_stacked
        opt = new_opt
        images_bank = images_bank.index_select(0, index)
        for slot in mips:
            if slot["tgt"].shape[0] != stacked.n_seeds:
                slot["tgt"] = slot["tgt"].index_select(0, index)
        trace.append(
            {
                "step": done_steps,
                "alive": list(seeds),
                "psnr_u8": [scores[i] for i in keep],
            }
        )
    best_state = None
    if done_steps < cfg.steps and cfg.eval_every and cfg.early_stop and stacked.n_seeds == 1:
        stall = 0
        init_score = _psnr_per_seed(stacked, images_bank, device)[0]
        while done_steps < cfg.steps:
            span = min(int(cfg.eval_every), cfg.steps - done_steps)
            info = _run_loop(
                stacked, opt, mips, cfg, semantics, span, done_steps, amp_dtype, device.type,
                use_graph=False,
            )
            loop_info = _merge_info(loop_info, info)
            done_steps += span
            score = _psnr_per_seed(stacked, images_bank, device)[0]
            if score > best_val + 0.01:
                best_val = score
                stall = 0
                best_state = [p.detach().clone() for p in stacked.parameters()]
            elif score < init_score + NEAR_INIT_DB:
                # Still within a few dB of the init. A sparse batch climbs
                # slower than 0.01 dB per eval, and patience would keep the
                # near-init weights. Do not stop yet.
                stall = 0
            else:
                stall += 1
                if stall >= cfg.early_stop:
                    stopped_early = True
                    break
        if best_state is not None:
            # Same rule as the eager loop: the export is the best eval, including
            # when the run used every step and the last eval was not the best.
            with torch.no_grad():
                for p, src in zip(stacked.parameters(), best_state, strict=True):
                    p.copy_(src)
    elif done_steps < cfg.steps and (cfg.stop_psnr > 0 or cfg.plateau_stop):
        hit_target = False
        plateau_hits = 0
        prev_score = -1e9
        check = max(1, int(cfg.check_every or 4))
        while done_steps < cfg.steps:
            span = min(check, cfg.steps - done_steps)
            info = _run_loop(
                stacked, opt, mips, cfg, semantics, span, done_steps, amp_dtype, device.type,
                use_graph=False,
            )
            loop_info = _merge_info(loop_info, info)
            done_steps += span
            was = stacked.quantize
            stacked.quantize = True
            score = max(_psnr_per_seed(stacked, images_bank, device))
            stacked.quantize = was
            best_val = max(best_val, score)
            if cfg.stop_psnr > 0 and score >= cfg.stop_psnr:
                hit_target = True
                break
            if cfg.plateau_stop and score < prev_score + 0.02:
                plateau_hits += 1
                if plateau_hits >= 3:
                    stopped_early = True
                    break
            else:
                plateau_hits = 0
            prev_score = score
        loop_info["hit_target"] = hit_target
    elif done_steps < cfg.steps:
        info = _run_loop(
            stacked,
            opt,
            mips,
            cfg,
            semantics,
            cfg.steps - done_steps,
            done_steps,
            amp_dtype,
            device.type,
            use_graph=use_graph and stacked.n_seeds == 1,
        )
        loop_info = _merge_info(loop_info, info)
        done_steps = cfg.steps

    scores = _psnr_per_seed(stacked, images_bank if images_bank.shape[0] == stacked.n_seeds else target, device)
    winner = int(max(range(stacked.n_seeds), key=lambda i: scores[i]))
    # Export through the ordinary writer so the file matches the slow path.
    export_net = nets[0]
    # nets[0] may not be the winner after slicing. Build a fresh shell.
    export_net = _build_nets(target, replace(cfg, pca_init=False), grids, [seeds[winner]], device)[0]
    stacked.write_seed(winner, export_net)
    num_mips = min(8, int(math.floor(math.log2(max(width, height)))) + 1)
    coder = cfg.latent_coder if cfg.latent_deflate else "raw"
    blob = export_network(
        export_net,
        semantics,
        width,
        height,
        num_mips,
        latent_deflate=cfg.latent_deflate,
        latent_coder=coder,
    )
    ts = parse_texture_set(blob)
    from ntc.decode_ref import decode_mip as _decode

    recon = _decode(ts, 0)
    ref = np.transpose(images, (1, 2, 0))
    trained_psnr = psnr_u8(recon, ref)
    kept = "trained"
    init_psnr = None
    # A resumed run keeps the continued weights. A fresh run keeps the
    # initialization when its 8-bit PSNR is higher, same rule as encode_arrays.
    if not cfg.resume:
        init_net = _build_nets(target, replace(cfg, pca_init=False), grids, [seeds[winner]], device)[0]
        init_net.load_state_dict(init_states[winner] if winner < len(init_states) else init_states[0])
        blob_i = export_network(
            init_net, semantics, width, height, num_mips, latent_deflate=cfg.latent_deflate, latent_coder=coder
        )
        recon_i = _decode(parse_texture_set(blob_i), 0)
        init_psnr = psnr_u8(recon_i, ref)
        if init_psnr > trained_psnr:
            blob, recon, kept = blob_i, recon_i, "init"
            ts = parse_texture_set(blob)
    metrics = group_metrics(recon, ref, semantics) if cfg.with_ssim else {
        "psnr_u8": psnr_u8(recon, ref),
    }
    rates = rate_report(ts)
    stats = {
        "fast": True,
        "batched_seeds": True,
        "halving": trace,
        "winner_seed": seeds[winner],
        "seed": seeds[winner],
        "seed_pool": len(trace[0]["alive"]) if trace else 1,
        "precision": cfg.precision if amp_dtype is not None else "fp32",
        "amp_dtype": None if amp_dtype is None else str(amp_dtype).replace("torch.", ""),
        "compiled": loop_info["compiled"],
        "compile_error": loop_info["compile_error"],
        "cuda_graph": loop_info["cuda_graph"],
        "cuda_graph_error": loop_info["cuda_graph_error"],
        "steps_per_s": (done_steps / loop_info["seconds"]) if loop_info["seconds"] else None,
        "seconds_train": loop_info["seconds"],
        "steps_ran": done_steps,
        "hit_target": bool(loop_info.get("hit_target")),
        "latent_init": cfg.latent_init,
        "optimizer": cfg.optimizer,
        "warm_start": cfg.warm_start,
        "freeze_decoder": cfg.freeze_decoder,
        "coarse_to_fine": cfg.coarse_to_fine,
        "qat_late": cfg.qat_late,
        "importance": cfg.importance,
        "plateau_stop": cfg.plateau_stop,
        "host_syncs_per_step": 0,
        "lr_scale": scaled_lr(1.0, int(cfg.batch_pixels or 0), int(cfg.lr_batch_ref or 0))
        if cfg.lr_batch_ref and cfg.batch_pixels
        else 1.0,
        "batch_pixels": int(cfg.batch_pixels or 0),
        "batch_mip": str(cfg.batch_mip),
        "device": device_report(device),
        "steps": cfg.steps,
        "grids": [list(g) for g in grids],
        "hidden": hidden,
        "checkpoint": kept,
        "psnr_u8_init": init_psnr,
        "psnr_u8_trained": trained_psnr,
        "stopped_early": stopped_early,
        "best_val_psnr": None if best_val < -1e8 else best_val,
        "latent_coder": coder,
        "latent_stored_bytes": ts.latent_stored_bytes,
        "latent_plain_bytes": ts.latent_plain_bytes,
        "metrics": metrics,
        "fused_kernel": "not_used",
        "fused_kernel_reason": "enabled only when a CUDA profile names the bilinear gather as the busy op",
        **rates,
    }
    if cfg.ckpt and stacked.n_seeds == 1:
        _save_resume_ckpt(stacked, opt, cfg, done_steps - 1)
    return blob, stats


def _merge_info(acc: dict, info: dict) -> dict:
    acc["seconds"] = float(acc["seconds"]) + float(info["seconds"])
    acc["compiled"] = bool(acc["compiled"] or info["compiled"])
    acc["compile_error"] = info["compile_error"] or acc["compile_error"]
    acc["cuda_graph"] = bool(acc["cuda_graph"] or info["cuda_graph"])
    acc["cuda_graph_error"] = info["cuda_graph_error"] or acc["cuda_graph_error"]
    return acc


def profile_historical_step(
    images: np.ndarray,
    semantics: list[int],
    cfg: TrainConfig,
    steps: int,
) -> dict:
    """Profile the eager per-step loop used by ``encode_arrays``.

    ``gpu_idle_fraction`` is ``1 - cuda_busy / wall`` when CUDA events exist.
    On a CPU build it is null. Launch overhead is the share of self CPU time
    whose key contains ``launch`` or ``cudaLaunch``.
    """
    from torch.profiler import ProfilerActivity, profile

    from ntc.device import device_report, pick_device
    from ntc.encode import encode_arrays

    device = pick_device(cfg.device)
    # Build one network with the historical initializer, then profile only
    # the step loop by running a short eager encode is too coarse (it includes
    # PCA). The replica below calls the same net.forward and the same loss.
    probe = replace(
        cfg,
        fast=False,
        steps=0,
        seed_pool=1,
        adaptive_pool=1,
        with_ssim=False,
        log_every=10**9,
        eval_every=0,
        early_stop=0,
    )
    _blob, prep = encode_arrays(images, semantics, probe)
    grids = [tuple(g) for g in prep["grids"]]
    target_np = torch.from_numpy(np.ascontiguousarray(images))
    torch.manual_seed(cfg.seed)
    net = _build_nets(target_np.to(device), cfg, grids, [cfg.seed], device)[0]
    # _build_nets already PCA-inits. Match the profiled weights to a fresh
    # eager net rather than the steps-0 export.
    target = target_np.to(device)
    latent_params = list(net.latents.parameters()) + list(net.log_scales.parameters())
    opt = torch.optim.Adam(
        [
            {"params": latent_params, "lr": cfg.lr * cfg.latent_lr_mult},
            {"params": net.linears.parameters(), "lr": cfg.lr},
        ]
    )
    net.train()
    max_mip = max(cfg.mip_levels - 1, 1)
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    def body() -> None:
        opt.zero_grad(set_to_none=True)
        loss = target.new_zeros(())
        for mip in range(cfg.mip_levels):
            tm = _color_mip(target, mip)
            hm, wm = tm.shape[-2], tm.shape[-1]
            ys = torch.arange(hm, device=device)
            xs = torch.arange(wm, device=device)
            yg, xg = torch.meshgrid(ys, xs, indexing="ij")
            uv = torch.stack(
                ((xg.reshape(-1).float() + 0.5) / wm, (yg.reshape(-1).float() + 0.5) / hm),
                dim=1,
            )
            tgt = tm[:, yg.reshape(-1), xg.reshape(-1)].T
            pred = net(uv, mip=mip, max_mip=max_mip)
            loss = loss + _mip_weight(cfg, mip) * _loss(pred.float(), tgt, semantics, cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

    # Warmup outside the profile so autotune does not dominate a short run.
    warmup = 1 if device.type == "cpu" else 3
    for _ in range(warmup):
        body()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with profile(activities=activities, record_shapes=False) as prof:
        for _ in range(max(1, steps)):
            body()
        if device.type == "cuda":
            torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    events = prof.key_averages()
    cpu_us = 0.0
    cuda_us = 0.0
    launch_us = 0.0
    rows = []
    for event in events:
        cpu_us += float(event.self_cpu_time_total)
        device_us = float(getattr(event, "self_device_time_total", 0) or getattr(event, "self_cuda_time_total", 0) or 0)
        cuda_us += device_us
        key = event.key or ""
        if "launch" in key.lower() or "cudaLaunch" in key:
            launch_us += float(event.self_cpu_time_total)
        rows.append(
            {
                "name": key,
                "self_cpu_us": float(event.self_cpu_time_total),
                "self_device_us": device_us,
                "count": int(event.count),
            }
        )
    rows.sort(key=lambda r: -(r["self_device_us"] or r["self_cpu_us"]))
    cuda_s = cuda_us / 1e6
    idle = None
    idle_reason = None
    if device.type != "cuda":
        idle_reason = "no cuda device. GPU idle fraction is measured by this function on the RX 9070."
    elif wall <= 0:
        idle_reason = "zero wall time"
    else:
        idle = max(0.0, 1.0 - (cuda_s / wall))
    top_device = [r for r in rows if r["self_device_us"] > 0][:8]
    recommend = False
    if top_device:
        name = top_device[0]["name"].lower()
        recommend = "gather" in name or "index" in name or "grid_sampler" in name
    return {
        "device": device_report(device),
        "steps": int(steps),
        "warmup_steps": warmup,
        "wall_s": wall,
        "steps_per_s": (steps / wall) if wall > 0 else None,
        "self_cpu_s": cpu_us / 1e6,
        "self_cuda_s": cuda_s if device.type == "cuda" else None,
        "gpu_idle_fraction": idle,
        "gpu_idle_reason": idle_reason,
        "launch_cpu_fraction": (launch_us / cpu_us) if cpu_us else None,
        "top_ops": rows[:12],
        "recommend_fused_gather": bool(recommend) if device.type == "cuda" else None,
        "recommend_fused_gather_reason": (
            "top CUDA op looks like the bilinear gather"
            if recommend
            else "fused gather stays off until a CUDA profile names that op"
        ),
        "triton": _triton_status(),
        "historical_host_syncs": [
            "clip_grad_norm_ reads the global norm on the host every step",
            "uv and each mip target are rebuilt every step",
            "seeds in a pool are separate encode_arrays calls",
        ],
    }


def _triton_status() -> str:
    try:
        import triton  # noqa: F401
    except Exception:
        return "not_installed"
    return "installed"


def encode_bank(
    images: list[np.ndarray],
    semantics: list[int],
    cfg: TrainConfig,
    seeds_per_image: list[int] | None = None,
) -> list[tuple[bytes, dict]]:
    """Train several materials of the same shape in one stacked module.

    Row ``i`` of the batch is material ``i``. ``cfg.seed_pool`` seeds are built
    for every material. Halving is off: dropping a row would drop a material.
    Grids must already be chosen.
    """
    from ntc.container import parse_texture_set, rate_report
    from ntc.decode_ref import decode_mip
    from ntc.device import device_report, pick_device
    from ntc.metrics import group_metrics, psnr_u8

    if not images:
        return []
    if cfg.grids is None:
        raise ValueError("encode_bank needs explicit grids")
    device = pick_device(cfg.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    shape = images[0].shape
    if any(im.shape != shape for im in images):
        raise ValueError("packed materials must share C,H,W")
    grids = [tuple(g) for g in cfg.grids]
    per = max(1, cfg.seed_pool if cfg.steps > 0 else 1)
    bases = seeds_per_image or [cfg.seed] * len(images)
    nets = []
    owners: list[int] = []
    seed_ids: list[int] = []
    targets = []
    for i, image in enumerate(images):
        target = torch.from_numpy(np.ascontiguousarray(image)).to(device)
        these = [int(bases[i]) + s for s in range(per)]
        built = _build_nets(target, cfg, grids, these, device)
        nets.extend(built)
        owners.extend([i] * len(built))
        seed_ids.extend(these)
        targets.extend([target] * len(built))
    stacked = StackedTrain(nets)
    bank = torch.stack(targets, 0)
    mips = _prep_mips(bank, cfg, device)
    opt = _adam_on(stacked, cfg, capturable=False)
    info = _run_loop(
        stacked, opt, mips, cfg, semantics, cfg.steps, 0, None, device.type, use_graph=False,
    )
    scores = _psnr_per_seed(stacked, bank, device)
    channels, height, width = shape
    num_mips = min(8, int(math.floor(math.log2(max(width, height)))) + 1)
    coder = cfg.latent_coder if cfg.latent_deflate else "raw"
    out = []
    for i, image in enumerate(images):
        idxs = [k for k, owner in enumerate(owners) if owner == i]
        winner = max(idxs, key=lambda k: scores[k])
        net = _build_nets(targets[winner], replace(cfg, pca_init=False), grids, [seed_ids[winner]], device)[0]
        stacked.write_seed(winner, net)
        blob = export_network(
            net, semantics, width, height, num_mips,
            latent_deflate=cfg.latent_deflate, latent_coder=coder,
        )
        ts = parse_texture_set(blob)
        recon = decode_mip(ts, 0)
        ref = np.transpose(image, (1, 2, 0))
        file_psnr = psnr_u8(recon, ref)
        metrics = group_metrics(recon, ref, semantics) if cfg.with_ssim else {"psnr_u8": file_psnr}
        stats = {
            "fast": True,
            "packed_jobs": len(images),
            "packed_rows": stacked.n_seeds,
            "checkpoint": "trained",
            "seed": seed_ids[winner],
            "steps_ran": cfg.steps,
            "seconds_train": info["seconds"],
            "steps_per_s": (cfg.steps / info["seconds"]) if info.get("seconds") else None,
            "compiled": info.get("compiled"),
            "cuda_graph": info.get("cuda_graph"),
            "compile_error": info.get("compile_error"),
            "cuda_graph_error": info.get("cuda_graph_error"),
            "stopped_early": False,
            "psnr_u8_init": None,
            "psnr_u8_trained": float(scores[winner]),
            "best_val_psnr": None,
            "grids": [list(g) for g in grids],
            "latent_coder": coder,
            "latent_stored_bytes": ts.latent_stored_bytes,
            "latent_plain_bytes": ts.latent_plain_bytes,
            "metrics": metrics,
            "device": device_report(device),
            **rate_report(ts),
        }
        out.append((blob, stats))
    return out


def choose_precision(
    images: np.ndarray,
    semantics: list[int],
    cfg: TrainConfig,
    probe_steps: int,
) -> dict:
    """Pick fp32 or bf16 by PSNR at equal wall time.

    The historical ``train_max`` probe runs 400 steps of each and keeps the
    higher PSNR. On the RX 9070 at c33648b that chose fp32. bf16 finished
    those 400 steps in 12.0 s and fp32 in 17.0 s, so bf16 was doing 1.42x
    more steps per second and the equal-step rule never gave it the extra
    steps. This probe trains bf16 until its wall time matches the fp32 run,
    then compares PSNR.
    """
    from ntc.encode import encode_arrays

    base = replace(
        cfg,
        fast=False,
        steps=int(probe_steps),
        seed_pool=1,
        adaptive_pool=1,
        with_ssim=False,
        log_every=10**9,
        eval_every=0,
        early_stop=0,
        precision="fp32",
    )
    t0 = time.perf_counter()
    _blob, fp = encode_arrays(images, semantics, base)
    fp_wall = time.perf_counter() - t0
    fp_psnr = float(fp["metrics"]["psnr_u8"])
    bf_error = None
    bf_psnr_equal_steps = None
    bf_wall_equal_steps = None
    bf_psnr_equal_wall = None
    bf_steps_equal_wall = None
    try:
        t1 = time.perf_counter()
        _blob, bf = encode_arrays(images, semantics, replace(base, precision="bf16"))
        bf_wall_equal_steps = time.perf_counter() - t1
        bf_psnr_equal_steps = float(bf["metrics"]["psnr_u8"])
        if bf_wall_equal_steps > 0 and fp_wall > bf_wall_equal_steps + 1e-6:
            extra = int(round(probe_steps * (fp_wall / bf_wall_equal_steps)))
            extra = max(extra, probe_steps)
            bf_steps_equal_wall = extra
            if extra > probe_steps:
                _blob, bf2 = encode_arrays(
                    images, semantics, replace(base, precision="bf16", steps=extra)
                )
                bf_psnr_equal_wall = float(bf2["metrics"]["psnr_u8"])
            else:
                bf_psnr_equal_wall = bf_psnr_equal_steps
        else:
            bf_steps_equal_wall = probe_steps
            bf_psnr_equal_wall = bf_psnr_equal_steps
    except Exception as exc:
        bf_error = str(exc)
    historical = "fp32"
    if bf_psnr_equal_steps is not None and bf_psnr_equal_steps > fp_psnr:
        historical = "bf16"
    chosen = "fp32"
    if bf_psnr_equal_wall is not None and bf_psnr_equal_wall >= fp_psnr:
        chosen = "bf16"
    return {
        "probe_steps": int(probe_steps),
        "fp32_wall_s": fp_wall,
        "fp32_psnr_u8": fp_psnr,
        "bf16_equal_steps_wall_s": bf_wall_equal_steps,
        "bf16_equal_steps_psnr_u8": bf_psnr_equal_steps,
        "bf16_equal_wall_steps": bf_steps_equal_wall,
        "bf16_equal_wall_psnr_u8": bf_psnr_equal_wall,
        "bf16_error": bf_error,
        "historical_equal_steps_choice": historical,
        "equal_wall_choice": chosen,
        "rx9070_c33648b": {
            "fp32_400_s": 17.0,
            "bf16_400_s": 12.0,
            "historical_choice": "fp32",
            "note": (
                "train_max keeps the higher PSNR after 400 steps. "
                "bf16 was 12.0 s and fp32 was 17.0 s, so the equal-step rule "
                "discarded a 1.42x step-rate advantage."
            ),
        },
    }
