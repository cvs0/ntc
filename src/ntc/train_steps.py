"""Step-reduction options for the fast trainer.

Each option is off unless the matching ``TrainConfig`` field is set.
``encode_arrays`` does not read these fields.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ntc.model import NtcNet, _soft_box


def apply_warm_start(net: NtcNet, path: str, freeze: bool) -> None:
    """Copy a saved MLP into ``net``. Overlapping output rows and input columns.

    The file is a dict with ``weights`` and ``biases``, each a list of tensors
    in layer order, plus ``hidden``, ``latent_channels``, and ``out_channels``.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    weights = blob["weights"]
    biases = blob["biases"]
    with torch.no_grad():
        for i, lin in enumerate(net.linears):
            if i >= len(weights):
                break
            src_w = weights[i].to(device=lin.weight.device, dtype=lin.weight.dtype)
            src_b = biases[i].to(device=lin.bias.device, dtype=lin.bias.dtype)
            rows = min(lin.weight.shape[0], src_w.shape[0])
            cols = min(lin.weight.shape[1], src_w.shape[1])
            lin.weight.zero_()
            lin.bias.zero_()
            lin.weight[:rows, :cols].copy_(src_w[:rows, :cols])
            take = min(lin.bias.shape[0], src_b.shape[0])
            lin.bias[:take].copy_(src_b[:take])
    if freeze:
        for param in net.linears.parameters():
            param.requires_grad_(False)


def analytic_project(net: NtcNet, target: torch.Tensor) -> None:
    """Least-squares latents for the linearized decoder, averaged over mips.

    ``target`` is ``[C,H,W]``. ReLU is ignored, so ``y ≈ W f + b`` with ``W``
    the product of the affine layers. Each mip is solved on its own grid,
    upsampled to full resolution, and averaged. Values are written as code
    units using the current per-channel scales.
    """
    weight, bias = _linearized_map(net)
    if weight is None:
        return
    levels = []
    level: torch.Tensor | None = target
    while level is not None:
        levels.append(level)
        nxt = _soft_box(level)
        level = nxt
    feat_sum = None
    for tm in levels:
        _c, height, width = tm.shape
        pixels = tm.permute(1, 2, 0).reshape(-1, tm.shape[0]).float()
        rhs = (pixels - bias.view(1, -1)).T.contiguous()
        try:
            solved = torch.linalg.lstsq(weight, rhs).solution
        except RuntimeError:
            return
        feat = solved.reshape(solved.shape[0], height, width)
        up = F.interpolate(
            feat.unsqueeze(0), size=target.shape[-2:], mode="bilinear", align_corners=False
        )[0]
        feat_sum = up if feat_sum is None else feat_sum + up
    if feat_sum is None:
        return
    feat_sum = feat_sum / float(len(levels))
    offset = 0
    with torch.no_grad():
        for i, lat in enumerate(net.latents):
            channels, gh, gw = lat.shape
            part = feat_sum[offset : offset + channels]
            offset += channels
            if part.shape[0] < channels:
                break
            pooled = F.interpolate(part.unsqueeze(0), size=(gh, gw), mode="area")[0]
            bits = net.grid_bits[i]
            mid = float((1 << bits) - 1) / 2.0
            scale = torch.exp(net.log_scales[i]).clamp(1e-4, 5.0).view(-1, 1, 1)
            codes = (pooled / scale + mid).clamp(0, (1 << bits) - 1)
            lat.copy_(codes)


def _linearized_map(net: NtcNet) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    mapped = None
    bias = None
    for lin in net.linears:
        weight = lin.weight.detach().float()
        row = lin.bias.detach().float()
        if mapped is None:
            mapped = weight
            bias = row
        else:
            bias = weight @ bias + row
            mapped = weight @ mapped
    return mapped, bias


def coarse_lowpass(stacked, step: int, steps: int, state: dict) -> None:
    """At 40% of the budget, replace each latent with its 2x box upsample."""
    if state.get("done"):
        return
    if step < int(steps * 0.4):
        return
    with torch.no_grad():
        for lat in stacked.latents:
            seeds, channels, height, width = lat.shape
            if height < 4 or width < 4:
                continue
            flat = lat.reshape(seeds * channels, 1, height, width)
            pooled = F.avg_pool2d(flat, kernel_size=2, stride=2)
            up = F.interpolate(pooled, size=(height, width), mode="bilinear", align_corners=False)
            lat.copy_(up.reshape(seeds, channels, height, width))
    state["done"] = True


def importance_index(stacked, uv: torch.Tensor, tgt: torch.Tensor, batch: int, step: int, cache: dict) -> torch.Tensor:
    """Sample ``batch`` pixels with probability proportional to mip-0 squared error.

    The distribution is refreshed every 8 steps. No host sync.
    """
    cached = cache.get("idx")
    if cached is not None and cached.shape[0] == batch and step % 8 != 0:
        return cached
    with torch.no_grad():
        pred = stacked.predict(uv, 0, 1)
        err = (pred.float() - tgt).square().mean(dim=(0, 2)).clamp(min=0)
        prob = err + 1e-3
        prob = prob / prob.sum()
        index = torch.multinomial(prob, batch, replacement=True)
    cache["idx"] = index
    return index


def shampoo_precondition(params) -> None:
    """Diagonal shampoo: scale each gradient by the inverse sqrt of its row and column energy.

    State is kept on the parameter as ``_shampoo``. MLP-sized tensors only.
    """
    for param in params:
        grad = param.grad
        if grad is None or grad.ndim < 2:
            continue
        state = getattr(param, "_shampoo", None)
        sq = grad.detach().square()
        row = sq.mean(dim=-1, keepdim=True)
        col = sq.mean(dim=-2, keepdim=True)
        if state is None:
            state = {"row": row, "col": col}
            param._shampoo = state
        else:
            state["row"].mul_(0.95).add_(row, alpha=0.05)
            state["col"].mul_(0.95).add_(col, alpha=0.05)
        scale = state["row"].sqrt() * state["col"].sqrt() + 1e-6
        grad.div_(scale)


class Lion(torch.optim.Optimizer):
    """Sign-update optimizer. Moments stay on device. No host sync."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99)):
        super().__init__(params, {"lr": lr, "betas": betas})

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                state = self.state[param]
                moment = state.get("m")
                if moment is None:
                    moment = torch.zeros_like(param)
                update = beta1 * moment + (1.0 - beta1) * grad
                param.add_(update.sign(), alpha=-lr)
                state["m"] = beta2 * moment + (1.0 - beta2) * grad
