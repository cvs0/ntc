"""Locked grid and PCA-allocation scan.

Does not change the encoder defaults. Prints one row at a time and
rewrites the JSON after each row.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

import ntc.encode as enc
from ntc.container import parse_texture_set
from ntc.cost import sample_cost, workgroup_latent_bytes
from ntc.encode import TrainConfig, _fit_readout, encode_arrays
from ntc.grids import shape_bpp
from ntc.materials import load_material


def _pca(net, target, sems, mode: str) -> None:
    recon = torch.zeros_like(target)
    order = sorted(
        range(len(net.latents)),
        key=lambda i: net.latents[i].shape[-1] * net.latents[i].shape[-2],
    )
    finest = max(order, key=lambda i: net.latents[i].shape[-1] * net.latents[i].shape[-2])
    normal_ix = [i for i, s in enumerate(sems) if s in (3, 4, 5)]
    other_ix = [i for i in range(target.shape[0]) if i not in normal_ix]
    channels = target.shape[0]
    with torch.no_grad():
        for i in order:
            lat = net.latents[i]
            ch, gh, gw = lat.shape
            residual = target - recon
            use = list(range(channels))
            if mode == "normals_fine" and i == finest and normal_ix:
                use = normal_ix
            elif mode == "split" and i == finest and normal_ix:
                use = normal_ix
            elif mode == "split" and i != finest and other_ix:
                use = other_ix
            ds = F.interpolate(residual.unsqueeze(0), size=(gh, gw), mode="area")[0]
            src = ds[use]
            flat = src.reshape(src.shape[0], -1)
            try:
                u, s, vh = torch.linalg.svd(flat, full_matrices=False)
            except RuntimeError:
                lat.normal_(0, 0.1)
                continue
            k = min(ch, int(s.shape[0]))
            coeff = vh[:k] * s[:k].unsqueeze(1)
            if k < ch:
                coeff = torch.cat(
                    [coeff, torch.zeros(ch - k, coeff.shape[1], device=coeff.device)], 0
                )
            values = coeff.reshape(ch, gh, gw)
            amax = values.abs().amax(dim=(1, 2)).clamp(min=1e-3)
            bits = net.grid_bits[i]
            mid = ((1 << bits) - 1) / 2.0
            scale = (amax / mid).clamp(1e-4, 5.0)
            net.log_scales[i].copy_(torch.log(scale))
            lat.copy_(values / scale.view(-1, 1, 1) + mid)
            basis = u[:, :k]
            rec_small = torch.zeros(channels, gh, gw, device=target.device)
            rec_src = (basis @ coeff[:k]).reshape(len(use), gh, gw)
            for row, channel in enumerate(use):
                rec_small[channel] = rec_src[row]
            rec = F.interpolate(
                rec_small.unsqueeze(0), size=target.shape[-2:], mode="bilinear", align_corners=False
            )[0]
            recon = recon + rec
        net.linears[-1].bias.copy_(target.mean(dim=(1, 2)))
        _fit_readout(net, target)


def main() -> None:
    images, sems, _info = load_material("/tmp/mat/metal_plate", 256)
    out = Path("results/shape_scan_256.json")
    rows = []
    shapes = [
        [[256, 256, 2, 4], [128, 128, 4, 8]],
        [[256, 256, 4, 2], [128, 128, 4, 8]],
        [[256, 256, 4, 2], [128, 128, 8, 4]],
        [[256, 256, 2, 4], [128, 128, 8, 4]],
        [[256, 256, 8, 2]],
        [[256, 256, 4, 4]],
        [[256, 256, 4, 2], [128, 128, 2, 8], [64, 64, 4, 8]],
        [[256, 256, 2, 4], [128, 128, 2, 8], [64, 64, 8, 8]],
        [[256, 256, 4, 2], [64, 64, 8, 8]],
        [[256, 256, 2, 4], [64, 64, 4, 8]],
        [[128, 128, 4, 8], [128, 128, 8, 4]],
        [[128, 128, 8, 8], [64, 64, 4, 8]],
        [[256, 256, 2, 2], [128, 128, 4, 8]],
        [[256, 256, 4, 4], [64, 64, 8, 4]],
        [[256, 256, 8, 2], [64, 64, 4, 4]],
        [[256, 256, 2, 8], [64, 64, 4, 4]],
    ]
    jobs = []
    for shape in shapes:
        for mode in ("joint", "normals_fine", "split"):
            jobs.append((shape, mode, (), 0, False))
    # Training and a hidden layer, only on the shapes nearest 16 bpp and 10 bpp.
    focus = [
        [[256, 256, 2, 4], [128, 128, 4, 8]],
        [[256, 256, 4, 2], [128, 128, 4, 8]],
        [[256, 256, 4, 2], [64, 64, 8, 8]],
        [[256, 256, 2, 4], [64, 64, 4, 8]],
        [[128, 128, 4, 8], [128, 128, 8, 4]],
    ]
    for shape in focus:
        for mode in ("joint", "normals_fine", "split"):
            jobs.append((shape, mode, (), 40, False))
            jobs.append((shape, mode, (32,), 40, False))

    orig = enc._pca_init
    for shape, mode, hidden, steps, weight in jobs:
        def patched(net, target, mode=mode):
            return _pca(net, target, sems, mode)

        enc._pca_init = patched
        cfg = TrainConfig(
            steps=steps,
            target_bpp=16.0,
            adaptive_pool=1,
            seed_pool=1,
            hidden=hidden,
            channel_weighting=weight,
            normal_cosine=0.0,
            grids=[tuple(g) for g in shape],
            log_every=10**9,
            with_ssim=True,
            pca_init=True,
        )
        _blob, stats = encode_arrays(images, sems, cfg)
        ts = parse_texture_set(_blob)
        m = stats["metrics"]
        cost = sample_cost(ts, 1)
        tile = workgroup_latent_bytes(ts, 8, 0)
        nominal = shape_bpp(256, 256, [tuple(g) for g in shape], list(hidden), images.shape[0], 0, 1, 8, 0)
        rec = {
            "grids": shape,
            "pca": mode,
            "hidden": list(hidden),
            "steps": steps,
            "nominal_bpp": nominal,
            "total_bpp": stats["total_bpp"],
            "psnr_u8": m["psnr_u8"],
            "ssim": m.get("ssim"),
            "psnr_u8_albedo": m.get("psnr_u8_albedo"),
            "psnr_u8_normal": m.get("psnr_u8_normal"),
            "psnr_u8_roughness": m.get("psnr_u8_roughness"),
            "normal_angle_deg": m.get("normal_angle_deg"),
            "channel_psnr_u8": m.get("channel_psnr_u8"),
            "macs": cost["macs"],
            "flops": cost["flops"],
            "latent_bytes_upper": cost["latent_bytes_upper"],
            "weight_bytes_cold": cost["weight_bytes_cold"],
            "tile_unique_bytes_per_texel": tile["bytes_per_texel_unique"],
            "seconds_train": stats["seconds_train"],
        }
        rows.append(rec)
        out.write_text(json.dumps({"rows": rows}, indent=2))
        print(
            f"pca={mode} h={list(hidden)} steps={steps} bpp={rec['total_bpp']:.3f} "
            f"psnr={rec['psnr_u8']:.3f} ssim={rec['ssim']:.4f} nrm={rec['psnr_u8_normal']:.2f} "
            f"grids={shape}",
            flush=True,
        )
    enc._pca_init = orig


if __name__ == "__main__":
    main()
