"""Isolated encoder ablations. One factor changes per run.

Each record is one on/off (or one architecture) measurement on the CPU
reference decoder. PSNR is uint8-quantized. Train-curve PSNR is not the
headline number.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ntc.cost import sample_cost, workgroup_latent_bytes
from ntc.device import device_report, pick_device
from ntc.encode import TrainConfig, encode_arrays
from ntc.materials import load_material


def _hidden(text: str) -> tuple:
    text = text.strip()
    if text in ("", "()", "none", "linear"):
        return ()
    return tuple(int(x) for x in text.split(",") if x.strip())


def _base(steps: int, bpp: float, hidden: tuple) -> TrainConfig:
    return TrainConfig(
        steps=steps,
        hidden=hidden,
        target_bpp=bpp,
        adaptive_pool=1,
        seed_pool=1,
        channel_weighting=True,
        normal_cosine=0.5,
        log_every=max(steps, 1),
        with_ssim=True,
    )


def variants(steps: int, bpp: float) -> list[tuple[str, TrainConfig]]:
    """Each entry differs from ``baseline`` by one factor, except the arch rows."""
    b = _base(steps, bpp, (32,))
    rows = [
        ("baseline", b),
        ("pca_off", replace(b, pca_init=False)),
        ("qat_latents_off", replace(b, qat_latents=False)),
        ("noise_proxy", replace(b, quant_proxy="noise")),
        ("weighting_off", replace(b, channel_weighting=False, normal_cosine=0.0)),
        ("pe_fourier1", replace(b, pe="fourier", pe_octaves=1)),
        ("pe_triangle3", replace(b, pe="triangle", pe_octaves=3)),
        ("hidden_linear", replace(b, hidden=())),
        ("hidden_16", replace(b, hidden=(16,))),
        ("hidden_48", replace(b, hidden=(48,))),
        ("hidden_64_32", replace(b, hidden=(64, 32))),
        ("act_hgelu", replace(b, activation="hgelu")),
        ("act_hgelu_requant", replace(b, activation="hgelu", activation_requant=True)),
        ("weight_int4_g16", replace(b, weight_bits=4, group_size=16)),
        ("weight_int8_g16", replace(b, weight_bits=8, group_size=16)),
        ("mip_levels_4", replace(b, mip_levels=4)),
        ("adaptive_pool_4", replace(b, adaptive_pool=4)),
        ("steps_0", replace(b, steps=0)),
        (
            "rtxntc_arch",
            replace(
                b,
                hidden=(64, 48, 32),
                activation="hgelu",
                activation_requant=True,
                pe="triangle",
                pe_octaves=3,
                channel_weighting=False,
                normal_cosine=0.0,
            ),
        ),
    ]
    return rows


def _record(name: str, stats: dict, images: np.ndarray) -> dict:
    m = stats["metrics"]
    return {
        "name": name,
        "psnr_u8": m.get("psnr_u8"),
        "psnr_f32": m.get("psnr_f32"),
        "ssim": m.get("ssim"),
        "psnr_u8_albedo": m.get("psnr_u8_albedo"),
        "psnr_u8_normal": m.get("psnr_u8_normal"),
        "psnr_u8_roughness": m.get("psnr_u8_roughness"),
        "psnr_u8_metalness": m.get("psnr_u8_metalness"),
        "psnr_u8_ao": m.get("psnr_u8_ao"),
        "ssim_albedo": m.get("ssim_albedo"),
        "ssim_normal": m.get("ssim_normal"),
        "normal_angle_deg": m.get("normal_angle_deg"),
        "channel_psnr_u8": m.get("channel_psnr_u8"),
        "channel_mse_u8": m.get("channel_mse_u8"),
        "total_bpp": stats["total_bpp"],
        "nominal_bpp": stats["nominal_bpp"],
        "seconds_train": stats["seconds_train"],
        "seconds_decode_ref": stats["seconds_decode_ref"],
        "grids": stats["grids"],
        "hidden": stats["hidden"],
        "steps": stats["steps"],
        "mip_metrics": stats.get("mip_metrics"),
        "code_entropy": stats.get("code_entropy"),
        "probe": stats.get("probe"),
        "width": int(images.shape[2]),
        "height": int(images.shape[1]),
        "channels": int(images.shape[0]),
    }


def _parse_grids(text: str):
    # "64x64x4x2+16x16x12x8"
    grids = []
    for part in text.split("+"):
        w, h, c, b = (int(x) for x in part.lower().split("x"))
        grids.append((w, h, c, b))
    return grids


def run_matrix(
    material: Path,
    size: int,
    steps: int,
    bpp: float,
    out: Path,
    names: set[str] | None,
    grids: list | None = None,
):
    images, sems, info = load_material(material, size)
    device = device_report(pick_device("auto"))
    rows = []
    blobs = {}
    payload = {
        "material": str(material),
        "info": {k: info[k] for k in ("height", "width", "semantics") if k in info},
        "source": info.get("roles"),
        "size": size,
        "steps": steps,
        "target_bpp": bpp,
        "locked_grids": grids,
        "device": device,
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "runs": rows,
    }
    for name, cfg in variants(steps, bpp):
        if names and name not in names:
            continue
        if grids is not None:
            if name == "adaptive_pool_4":
                continue
            cfg = replace(cfg, grids=grids, adaptive_pool=1)
        t0 = time.perf_counter()
        blob, stats = encode_arrays(images, sems, cfg)
        rec = _record(name, stats, images)
        rec["wall_s"] = time.perf_counter() - t0
        # cost of the exported network, one feature-bilinear tap
        from ntc.container import parse_texture_set

        ts = parse_texture_set(blob)
        rec["cost_1tap"] = sample_cost(ts, taps=1)
        rec["cost_color_bilinear"] = sample_cost(ts, taps=4)
        rec["workgroup_latent_bytes"] = workgroup_latent_bytes(ts, group=8)
        rows.append(rec)
        blobs[name] = len(blob)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(
            f"{name} psnr_u8 {rec['psnr_u8']:.3f} ssim {rec['ssim']:.4f} "
            f"bpp {rec['total_bpp']:.3f} train_s {rec['seconds_train']:.2f} "
            f"macs {rec['cost_1tap']['macs']}",
            flush=True,
        )
    return payload


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ntc-ablate")
    p.add_argument("material", type=Path)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--bpp", type=float, default=4.0)
    p.add_argument("--out", type=Path, default=Path("results/ablation.json"))
    p.add_argument("--only", default="", help="comma-separated variant names")
    p.add_argument("--grids", default="", help="lock WxHxCxB+... so factors do not retarget the latent")
    args = p.parse_args(argv)
    names = {s for s in args.only.split(",") if s} or None
    grids = _parse_grids(args.grids) if args.grids else None
    run_matrix(args.material, args.size, args.steps, args.bpp, args.out, names, grids)


if __name__ == "__main__":
    main()
