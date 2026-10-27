"""Rate-distortion sweep. Writes one JSON row at a time."""

from __future__ import annotations

import json
from pathlib import Path

from ntc.container import parse_texture_set
from ntc.cost import sample_cost
from ntc.encode import TrainConfig, encode_arrays
from ntc.materials import load_material


def main() -> None:
    images, sems, _info = load_material("/tmp/mat/metal_plate", 256)
    out = Path("results/rd_fullres_256.json")
    rows = []
    jobs = [
        ((), False, 4),
        ((), False, 8),
        ((), False, 12),
        ((), False, 16),
        ((32,), False, 8),
        ((32,), False, 16),
        ((32,), True, 8),
    ]
    for hidden, weight, bpp in jobs:
        cfg = TrainConfig(
            steps=40,
            target_bpp=float(bpp),
            adaptive_pool=4,
            seed_pool=1,
            hidden=hidden,
            channel_weighting=weight,
            normal_cosine=0.5 if weight else 0.0,
            log_every=10**9,
        )
        _blob, stats = encode_arrays(images, sems, cfg)
        ts = parse_texture_set(_blob)
        m = stats["metrics"]
        cost = sample_cost(ts, 1)
        rec = {
            "hidden": list(hidden),
            "weighting": weight,
            "bpp_target": bpp,
            "total_bpp": stats["total_bpp"],
            "psnr_u8": m["psnr_u8"],
            "ssim": m["ssim"],
            "psnr_u8_albedo": m.get("psnr_u8_albedo"),
            "psnr_u8_normal": m.get("psnr_u8_normal"),
            "psnr_u8_roughness": m.get("psnr_u8_roughness"),
            "psnr_u8_metalness": m.get("psnr_u8_metalness"),
            "psnr_u8_ao": m.get("psnr_u8_ao"),
            "normal_angle_deg": m.get("normal_angle_deg"),
            "channel_psnr_u8": m.get("channel_psnr_u8"),
            "grids": stats["grids"],
            "seconds_train": stats["seconds_train"],
            "macs": cost["macs"],
            "flops": cost["flops"],
            "latent_bytes_upper": cost["latent_bytes_upper"],
            "weight_bytes_cold": cost["weight_bytes_cold"],
            "probe": stats["probe"],
            "mip2_psnr": (stats.get("mip_metrics") or {}).get("m2", {}).get("psnr_u8"),
        }
        rows.append(rec)
        out.write_text(
            json.dumps(
                {
                    "size": 256,
                    "steps": 40,
                    "note": "shape family includes full-resolution grids",
                    "rows": rows,
                },
                indent=2,
            )
        )
        print(
            f"h={list(hidden)} w={weight} tgt={bpp} bpp={rec['total_bpp']:.3f} "
            f"psnr={rec['psnr_u8']:.3f} grids={rec['grids']} macs={rec['macs']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
