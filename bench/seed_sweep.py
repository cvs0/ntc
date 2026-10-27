"""Seeds on the 16 bpp grid. One factor: the torch seed."""

from __future__ import annotations

import json
from pathlib import Path

from ntc.encode import TrainConfig, encode_arrays
from ntc.materials import load_material


def main() -> None:
    images, sems, _info = load_material("/tmp/mat/metal_plate", 256)
    grids = [(256, 256, 4, 4), (8, 8, 2, 2)]
    out = Path("results/seed_sweep_256.json")
    rows = []
    for seed in (0, 1, 2, 3):
        cfg = TrainConfig(
            steps=40,
            seed=seed,
            hidden=(32,),
            grids=grids,
            adaptive_pool=1,
            seed_pool=1,
            target_bpp=16.0,
            log_every=10**9,
        )
        _blob, stats = encode_arrays(images, sems, cfg)
        m = stats["metrics"]
        rec = {
            "seed": seed,
            "psnr_u8": m["psnr_u8"],
            "ssim": m["ssim"],
            "psnr_u8_init": stats["psnr_u8_init"],
            "psnr_u8_trained": stats["psnr_u8_trained"],
            "checkpoint": stats["checkpoint"],
            "psnr_u8_normal": m["psnr_u8_normal"],
            "psnr_u8_albedo": m["psnr_u8_albedo"],
            "normal_angle_deg": m["normal_angle_deg"],
            "seconds_train": stats["seconds_train"],
            "total_bpp": stats["total_bpp"],
        }
        rows.append(rec)
        out.write_text(json.dumps({"rows": rows}, indent=2))
        print(
            f"seed={seed} psnr={rec['psnr_u8']:.3f} ssim={rec['ssim']:.4f} "
            f"init={rec['psnr_u8_init']:.3f} trained={rec['psnr_u8_trained']:.3f} kept={rec['checkpoint']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
