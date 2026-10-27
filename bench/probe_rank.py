"""Steps-0 and short-train ranking of the rate pool, hidden 32, target 16."""

from __future__ import annotations

import json
from pathlib import Path

from ntc.encode import TrainConfig, encode_arrays
from ntc.grids import select_grids
from ntc.materials import load_material


def main() -> None:
    images, sems, _info = load_material("/tmp/mat/metal_plate", 256)
    picks = select_grids(256, 256, 16.0, [32], 9, 0, 1, 8, 0, adaptive_pool=8)
    out = Path("results/probe_rank_256.json")
    rows = []
    for steps in (0, 10):
        for grids in picks:
            cfg = TrainConfig(
                steps=steps,
                hidden=(32,),
                grids=[tuple(g) for g in grids],
                adaptive_pool=1,
                seed_pool=1,
                channel_weighting=False,
                normal_cosine=0.0,
                with_ssim=steps > 0,
                log_every=10**9,
            )
            _blob, stats = encode_arrays(images, sems, cfg)
            m = stats["metrics"]
            rec = {
                "steps": steps,
                "grids": [list(g) for g in grids],
                "total_bpp": stats["total_bpp"],
                "psnr_u8": m["psnr_u8"],
                "ssim": m.get("ssim"),
                "psnr_u8_normal": m.get("psnr_u8_normal"),
                "seconds_train": stats["seconds_train"],
            }
            rows.append(rec)
            out.write_text(json.dumps({"rows": rows}, indent=2))
            print(
                f"steps={steps} psnr={rec['psnr_u8']:.3f} bpp={rec['total_bpp']:.3f} grids={rec['grids']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
