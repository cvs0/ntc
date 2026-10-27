"""Full metrics for the steps-0 leader, plus a best-checkpoint train."""

from __future__ import annotations

import json
from pathlib import Path

from ntc.container import parse_texture_set
from ntc.cost import sample_cost, workgroup_latent_bytes
from ntc.encode import TrainConfig, encode_arrays
from ntc.materials import load_material


def main() -> None:
    images, sems, _info = load_material("/tmp/mat/metal_plate", 256)
    grids = [(256, 256, 4, 4), (8, 8, 2, 2)]
    out = Path("results/best_grid_256.json")
    rows = []
    jobs = [
        ((), 0),
        ((32,), 0),
        ((16,), 0),
        ((32,), 40),
        ((16,), 40),
        ((), 40),
    ]
    for hidden, steps in jobs:
        cfg = TrainConfig(
            steps=steps,
            hidden=hidden,
            grids=grids,
            adaptive_pool=1,
            seed_pool=1,
            channel_weighting=False,
            normal_cosine=0.0,
            with_ssim=True,
            log_every=10**9,
        )
        blob, stats = encode_arrays(images, sems, cfg)
        ts = parse_texture_set(blob)
        m = stats["metrics"]
        cost = sample_cost(ts, 1)
        tile = workgroup_latent_bytes(ts, 8, 0)
        rec = {
            "hidden": list(hidden),
            "steps": steps,
            "grids": [list(g) for g in grids],
            "total_bpp": stats["total_bpp"],
            "psnr_u8": m["psnr_u8"],
            "ssim": m.get("ssim"),
            "psnr_u8_albedo": m.get("psnr_u8_albedo"),
            "psnr_u8_normal": m.get("psnr_u8_normal"),
            "psnr_u8_roughness": m.get("psnr_u8_roughness"),
            "psnr_u8_metalness": m.get("psnr_u8_metalness"),
            "psnr_u8_ao": m.get("psnr_u8_ao"),
            "normal_angle_deg": m.get("normal_angle_deg"),
            "channel_psnr_u8": m.get("channel_psnr_u8"),
            "macs": cost["macs"],
            "flops": cost["flops"],
            "latent_bytes_upper": cost["latent_bytes_upper"],
            "weight_bytes_cold": cost["weight_bytes_cold"],
            "tile_unique": tile["bytes_per_texel_unique"],
            "seconds_train": stats["seconds_train"],
        }
        rows.append(rec)
        out.write_text(json.dumps({"rows": rows}, indent=2))
        print(
            f"h={list(hidden)} steps={steps} psnr={rec['psnr_u8']:.3f} ssim={rec['ssim']:.4f} "
            f"nrm={rec['psnr_u8_normal']:.2f} alb={rec['psnr_u8_albedo']:.2f} macs={rec['macs']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
