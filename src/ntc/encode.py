"""Encode one material into an NTC1 blob."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn.functional as F

from ntc.container import (
    ACT_HGELU,
    ACT_IDENTITY,
    ACT_RELU,
    FLAG_ACTIVATION_REQUANT,
    PE_FOURIER,
    PE_NONE,
    PE_TRIANGLE,
    ChannelRec,
    build_texture_set,
    parse_texture_set,
    payload_bytes,
    total_bpp,
)
from ntc.decode_ref import decode_mip
from ntc.device import device_report, pick_device
from ntc.grids import select_grids, shape_bpp
from ntc.metrics import group_metrics
from ntc.model import NtcNet, _activate, _soft_box
from ntc.quant import quantize_weight

_ACT = {"identity": ACT_IDENTITY, "relu": ACT_RELU, "hgelu": ACT_HGELU}
_PE = {"none": PE_NONE, "fourier": PE_FOURIER, "triangle": PE_TRIANGLE}

_BASE_WEIGHT = {
    0: 1.0,
    1: 1.0,
    2: 1.0,
    3: 2.0,
    4: 2.0,
    5: 2.0,
    6: 1.0,
    7: 0.5,
    8: 0.5,
    9: 0.25,
    10: 1.0,
    11: 1.0,
}


@dataclass
class TrainConfig:
    steps: int = 40
    lr: float = 5e-3
    latent_lr_mult: float = 2.0
    hidden: tuple = (32,)
    activation: str = "relu"
    pe: str = "none"
    pe_octaves: int = 1
    weight_bits: int = 8
    group_size: int = 0
    qat_latents: bool = True
    qat_weights: bool = True
    quant_proxy: str = "ste"
    mip_levels: int = 1
    channel_weighting: bool = False
    normal_cosine: float = 0.0
    pca_init: bool = True
    activation_requant: bool = False
    seed: int = 0
    batch_pixels: int = 0
    device: str = "auto"
    grids: list | None = None
    target_bpp: float = 4.0
    log_every: int = 20
    adaptive_pool: int = 8
    seed_pool: int = 4
    with_ssim: bool = True


def _color_mip(target: torch.Tensor, mip: int) -> torch.Tensor:
    t = target
    for _ in range(mip):
        nxt = _soft_box(t)
        if nxt is None:
            break
        t = nxt
    return t


def _pca_init(net: NtcNet, target: torch.Tensor) -> None:
    """Coarse-to-fine PCA of the residual into each latent grid."""
    recon = torch.zeros_like(target)
    order = sorted(
        range(len(net.latents)),
        key=lambda i: net.latents[i].shape[-1] * net.latents[i].shape[-2],
    )
    channels = target.shape[0]
    with torch.no_grad():
        for i in order:
            lat = net.latents[i]
            ch, gh, gw = lat.shape
            residual = target - recon
            ds = F.interpolate(residual.unsqueeze(0), size=(gh, gw), mode="area")[0]
            flat = ds.reshape(channels, -1)
            # Economy SVD. C is small (the material channel count).
            try:
                u, s, vh = torch.linalg.svd(flat, full_matrices=False)
            except RuntimeError:
                lat.normal_(0, 0.1)
                continue
            k = min(ch, int(s.shape[0]))
            coeff = vh[:k] * s[:k].unsqueeze(1)
            if k < ch:
                coeff = torch.cat([coeff, torch.zeros(ch - k, coeff.shape[1], device=coeff.device)], 0)
            values = coeff.reshape(ch, gh, gw)
            amax = values.abs().amax(dim=(1, 2)).clamp(min=1e-3)
            bits = net.grid_bits[i]
            mid = ((1 << bits) - 1) / 2.0
            scale = (amax / mid).clamp(1e-4, 5.0)
            net.log_scales[i].copy_(torch.log(scale))
            # Store the latent in code units so the optimizer steps in bins.
            lat.copy_(values / scale.view(-1, 1, 1) + mid)
            basis = u[:, :k]
            rec_ds = (basis @ coeff[:k]).reshape(channels, gh, gw)
            rec = F.interpolate(
                rec_ds.unsqueeze(0), size=target.shape[-2:], mode="bilinear", align_corners=False
            )[0]
            recon = recon + rec
        net.linears[-1].bias.copy_(target.mean(dim=(1, 2)))
        _fit_readout(net, target)


def _fit_readout(net: NtcNet, target: torch.Tensor) -> None:
    """Least-squares fit of the last layer to the current features.

    Hidden layers stay at their initializer. The fit is unregularized and
    runs with quantization on, so the exported integer weights start near
    a solution of the linear readout.
    """
    was = net.training
    net.eval()
    c, h, w = target.shape
    # subsample to keep the solve small
    step = max(1, int((h * w / 8192) ** 0.5))
    ys = torch.arange(0, h, step, device=target.device)
    xs = torch.arange(0, w, step, device=target.device)
    yg, xg = torch.meshgrid(ys, xs, indexing="ij")
    uv = torch.stack(((xg.reshape(-1).float() + 0.5) / w, (yg.reshape(-1).float() + 0.5) / h), dim=1)
    feat = net.features(uv, 0, 1).detach()
    h = feat
    with torch.no_grad():
        for lin in list(net.linears)[:-1]:
            w, b = net._weight(lin)
            h = F.linear(h, w, b)
            h = _activate(h, net.activation, net.activation_requant)
    tgt = target[:, yg.reshape(-1), xg.reshape(-1)].T.detach()
    # Solve h @ A ≈ tgt. A is [hidden, C], and the last weight is A.T.
    try:
        sol = torch.linalg.lstsq(h, tgt).solution
    except RuntimeError:
        if was:
            net.train()
        return
    with torch.no_grad():
        out_f, in_f = net.linears[-1].weight.shape
        net.linears[-1].weight.copy_(sol.T[:out_f, :in_f])
        net.linears[-1].bias.zero_()
    if was:
        net.train()


def _loss(pred, tgt, sems, cfg: TrainConfig) -> torch.Tensor:
    if cfg.channel_weighting:
        w = pred.new_tensor([_BASE_WEIGHT.get(s, 1.0) for s in sems])
    else:
        w = pred.new_ones(len(sems))
    mse = ((pred - tgt) ** 2 * w.view(1, -1)).mean()
    loss = mse
    cosine = cfg.normal_cosine if cfg.channel_weighting else 0.0
    if cosine and all(s in sems for s in (3, 4, 5)):
        ix = [sems.index(3), sems.index(4), sems.index(5)]
        p = pred[:, ix] * 2 - 1
        t = tgt[:, ix] * 2 - 1
        p = p / p.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        t = t / t.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        loss = loss + cosine * (1.0 - (p * t).sum(-1)).mean()
    return loss


def _subset_psnr(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    d = (pred.detach() - tgt).float()
    mse = torch.mean(d * d).clamp(min=1e-20)
    return float((10.0 * torch.log10(1.0 / mse)).item())


@torch.no_grad()
def export_network(net: NtcNet, sems: list[int], width: int, height: int, num_mips: int) -> bytes:
    net.eval()
    grids = []
    for i in range(len(net.latents)):
        scale = net.scales(i).view(-1).detach().float().cpu()
        bits = net.grid_bits[i]
        z = net.latents[i].detach().float().cpu()
        code = torch.round(z).clamp(0, (1 << bits) - 1)
        grids.append(
            {
                "codes": code.to(torch.uint8).numpy(),
                "bits": bits,
                "scale": scale.numpy(),
            }
        )
    layers = []
    for lin in net.linears:
        _ste, codes, scales = quantize_weight(
            lin.weight.detach().float().cpu(), net.weight_bits, net.group_size
        )
        bias = lin.bias.detach().float().cpu()
        gs = net.group_size
        inn = codes.shape[1]
        if gs <= 0 or gs >= inn:
            gs_store = 0
        else:
            gs_store = gs
        layers.append(
            {
                "codes": codes.numpy(),
                "bias": bias.numpy(),
                "scale": scales.numpy(),
                "group_size": gs_store,
            }
        )
    flags = FLAG_ACTIVATION_REQUANT if net.activation_requant else 0
    return build_texture_set(
        width=width,
        height=height,
        num_mips=num_mips,
        activation=net.activation,
        pe_mode=net.pe_mode,
        pe_octaves=net.pe_octaves,
        flags=flags,
        weight_bits=net.weight_bits,
        group_size=net.group_size if net.group_size > 0 else 0,
        channels=[ChannelRec(int(s), 0) for s in sems],
        grids=grids,
        layers=layers,
    )


def encode_arrays(
    images: np.ndarray,
    semantics: list[int],
    cfg: TrainConfig,
) -> tuple[bytes, dict]:
    """``images`` is float32 ``[C,H,W]`` in ``[0,1]``."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = pick_device(cfg.device)
    c, h, w = images.shape
    hidden = list(cfg.hidden)
    if any(v > 64 for v in hidden) or len(hidden) > 3:
        raise ValueError("hidden width must be <= 64 and depth <= 3")
    pe_mode = _PE[cfg.pe]
    probe_table = []
    if cfg.grids is None:
        choices = select_grids(
            w,
            h,
            cfg.target_bpp,
            hidden,
            c,
            pe_mode,
            cfg.pe_octaves,
            cfg.weight_bits,
            cfg.group_size,
            adaptive_pool=max(1, cfg.adaptive_pool),
        )
        if cfg.adaptive_pool > 1 and cfg.steps > 0 and len(choices) > 1:
            for candidate in choices:
                probe_cfg = replace(
                    cfg,
                    grids=candidate,
                    steps=0,
                    adaptive_pool=1,
                    seed_pool=1,
                    with_ssim=False,
                    log_every=10**9,
                )
                _blob, probe_stats = encode_arrays(images, semantics, probe_cfg)
                probe_table.append((probe_stats["metrics"]["psnr_u8"], candidate))
            probe_table.sort(key=lambda item: -item[0])
            grids = probe_table[0][1]
        else:
            grids = choices[0]
    else:
        grids = cfg.grids
    # The hidden-layer initializer is the only random parameter PCA does not
    # overwrite. Train ``seed_pool`` seeds on the chosen grids and keep the
    # export with the higher 8-bit PSNR. Ranking by the step-0 score alone
    # regressed the 8 bpp metal_plate point, so each seed is trained.
    # Grid probes above stay on ``cfg.seed``.
    seed_probe: list[tuple[float, int]] = []
    if cfg.seed_pool > 1 and cfg.steps > 0 and hidden:
        ranked: list[tuple[float, int, bytes, dict]] = []
        for s in range(cfg.seed, cfg.seed + cfg.seed_pool):
            probe_cfg = replace(
                cfg,
                seed=s,
                grids=grids,
                adaptive_pool=1,
                seed_pool=1,
            )
            blob_s, stats_s = encode_arrays(images, semantics, probe_cfg)
            ranked.append((stats_s["metrics"]["psnr_u8"], s, blob_s, stats_s))
        ranked.sort(key=lambda item: -item[0])
        _best_psnr, _best_seed, best_blob, best_stats = ranked[0]
        best_stats = dict(best_stats)
        best_stats["seed_pool"] = cfg.seed_pool
        best_stats["seed_probe"] = [
            {
                "psnr_u8": p,
                "seed": s,
                "ssim": st["metrics"].get("ssim"),
                "checkpoint": st.get("checkpoint"),
            }
            for p, s, _b, st in ranked
        ]
        best_stats["seconds_seed_pool"] = float(sum(st["seconds_train"] for _p, _s, _b, st in ranked))
        return best_blob, best_stats
    # Probes are separate encode calls and each reseeds. Reset again so the
    # network built below does not depend on how many probes ran.
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    nominal = shape_bpp(
        w,
        h,
        grids,
        hidden,
        c,
        pe_mode,
        cfg.pe_octaves,
        cfg.weight_bits,
        cfg.group_size,
    )
    net = NtcNet(
        image_channels=c,
        grids=grids,
        hidden=hidden,
        activation=_ACT[cfg.activation],
        pe_mode=pe_mode,
        pe_octaves=cfg.pe_octaves,
        weight_bits=cfg.weight_bits,
        group_size=cfg.group_size,
        quant_proxy=cfg.quant_proxy,
        activation_requant=cfg.activation_requant,
        qat_latents=cfg.qat_latents,
        qat_weights=cfg.qat_weights,
    ).to(device)
    target = torch.from_numpy(np.ascontiguousarray(images)).to(device)
    if cfg.pca_init:
        _pca_init(net, target)
    else:
        for i, lat in enumerate(net.latents):
            mid = ((1 << net.grid_bits[i]) - 1) / 2.0
            lat.data.normal_(mid, 0.5)
        _fit_readout(net, target)
    # Adam on a linear map, and on some hidden widths, finishes below the
    # PCA initialization. The export keeps whichever of the two has the
    # higher 8-bit PSNR.
    init_state = None
    if cfg.steps > 0:
        init_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    opt = torch.optim.Adam(
        [
            {
                "params": list(net.latents.parameters()) + list(net.log_scales.parameters()),
                "lr": cfg.lr * cfg.latent_lr_mult,
            },
            {"params": net.linears.parameters(), "lr": cfg.lr},
        ]
    )
    curve = []
    t0 = time.perf_counter()
    net.train()
    max_mip = max(cfg.mip_levels - 1, 1)
    for step in range(cfg.steps):
        cos = 0.5 * (1.0 + math.cos(math.pi * step / max(cfg.steps, 1)))
        lr = max(cfg.lr * cos, cfg.lr * 0.05)
        opt.param_groups[0]["lr"] = lr * cfg.latent_lr_mult
        opt.param_groups[1]["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss = target.new_zeros(())
        monitor_psnr = None
        for m in range(cfg.mip_levels):
            tm = _color_mip(target, m)
            hm, wm = tm.shape[-2], tm.shape[-1]
            n = hm * wm
            if cfg.batch_pixels and cfg.batch_pixels < n:
                idx = torch.randint(0, n, (cfg.batch_pixels,), device=device)
            else:
                idx = None
            if idx is None:
                ys = torch.arange(hm, device=device)
                xs = torch.arange(wm, device=device)
                yg, xg = torch.meshgrid(ys, xs, indexing="ij")
                ys = yg.reshape(-1)
                xs = xg.reshape(-1)
            else:
                ys = torch.div(idx, wm, rounding_mode="floor")
                xs = idx % wm
            uv = torch.stack(((xs.float() + 0.5) / wm, (ys.float() + 0.5) / hm), dim=1)
            tgt = tm[:, ys, xs].T
            pred = net(uv, mip=m, max_mip=max_mip)
            loss = loss + (0.5 ** m) * _loss(pred, tgt, semantics, cfg)
            if m == 0 and (step % cfg.log_every == 0 or step == cfg.steps - 1):
                monitor_psnr = _subset_psnr(pred, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if monitor_psnr is not None:
            curve.append({"step": step, "psnr_train": monitor_psnr, "loss": float(loss.detach())})
    train_s = time.perf_counter() - t0
    num_mips = min(8, int(math.floor(math.log2(max(w, h)))) + 1)
    ref = np.transpose(images, (1, 2, 0))
    from ntc.metrics import psnr_u8

    def _export_current():
        t_export = time.perf_counter()
        blob_i = export_network(net, semantics, w, h, num_mips)
        export_dt = time.perf_counter() - t_export
        t_dec = time.perf_counter()
        ts_i = parse_texture_set(blob_i)
        recon_i = decode_mip(ts_i, 0)
        dec_dt = time.perf_counter() - t_dec
        return blob_i, ts_i, recon_i, export_dt, dec_dt

    blob, ts, recon, export_s, decode_s = _export_current()
    psnr_here = psnr_u8(recon, ref)
    psnr_trained = psnr_here if cfg.steps > 0 else None
    psnr_init = None if cfg.steps > 0 else psnr_here
    kept = "init" if cfg.steps == 0 else "trained"
    if init_state is not None:
        trained_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(init_state)
        blob_i, ts_i, recon_i, export_i, decode_i = _export_current()
        psnr_init = psnr_u8(recon_i, ref)
        export_s += export_i
        decode_s += decode_i
        if psnr_init > psnr_trained:
            blob, ts, recon = blob_i, ts_i, recon_i
            kept = "init"
        else:
            net.load_state_dict(trained_state)
    if cfg.with_ssim:
        metrics = group_metrics(recon, ref, semantics)
    else:
        from ntc.metrics import psnr_f32, psnr_u8

        metrics = {"psnr_u8": psnr_u8(recon, ref), "psnr_f32": psnr_f32(recon, ref)}
    mip_metrics = {}
    for m in (1, 2):
        if m >= num_mips:
            break
        rec_m = decode_mip(ts, m)
        ref_m = _color_mip(target, m).detach().float().cpu().numpy()
        ref_m = np.transpose(ref_m, (1, 2, 0))
        if rec_m.shape[:2] != ref_m.shape[:2]:
            continue
        mip_metrics[f"m{m}"] = group_metrics(rec_m, ref_m, semantics)
    stats = {
        "device": device_report(device),
        "seconds_train": train_s,
        "seconds_export": export_s,
        "seconds_decode_ref": decode_s,
        "steps": cfg.steps,
        "nominal_bpp": nominal,
        "total_bpp": total_bpp(ts),
        "payload_bytes": payload_bytes(ts),
        "file_bytes": len(blob),
        "grids": [list(g) for g in grids],
        "hidden": hidden,
        "activation": cfg.activation,
        "pe": cfg.pe,
        "pe_octaves": cfg.pe_octaves,
        "weight_bits": cfg.weight_bits,
        "group_size": cfg.group_size,
        "qat_latents": cfg.qat_latents,
        "qat_weights": cfg.qat_weights,
        "quant_proxy": cfg.quant_proxy,
        "mip_levels": cfg.mip_levels,
        "channel_weighting": cfg.channel_weighting,
        "normal_cosine": cfg.normal_cosine,
        "pca_init": cfg.pca_init,
        "adaptive_pool": cfg.adaptive_pool,
        "checkpoint": kept,
        "psnr_u8_init": psnr_init,
        "psnr_u8_trained": psnr_trained,
        "seed": cfg.seed,
        "seed_pool": cfg.seed_pool,
        "seed_probe": [{"psnr_u8": p, "seed": s} for p, s in seed_probe],
        "seconds_seed_pool": None,
        "probe": [
            {"psnr_u8": p, "grids": [list(g) for g in grids_i]} for p, grids_i in probe_table
        ],
        "activation_requant": cfg.activation_requant,
        "curve": curve,
        "metrics": metrics,
        "mip_metrics": mip_metrics,
        "code_entropy": _entropies(ts),
    }
    return blob, stats


def _entropies(ts) -> list[float]:
    out = []
    for g in ts.grids:
        hist = np.bincount(g.codes.reshape(-1).astype(np.int64), minlength=1 << g.bits)
        p = hist.astype(np.float64)
        s = p.sum()
        p = p[p > 0] / s
        out.append(float(-(p * np.log2(p)).sum()))
    return out
