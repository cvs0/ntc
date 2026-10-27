"""Wall time of one shipped 16 bpp encode."""

from __future__ import annotations

import json
import time

from ntc.encode import TrainConfig, encode_arrays
from ntc.materials import load_material


def main() -> None:
    images, sems, _info = load_material("/tmp/mat/metal_plate", 256)
    t = time.perf_counter()
    _blob, stats = encode_arrays(images, sems, TrainConfig(target_bpp=16.0, log_every=10**9))
    wall = time.perf_counter() - t
    rec = {
        "wall_s": wall,
        "seconds_train_winner": stats["seconds_train"],
        "seconds_seed_pool": stats.get("seconds_seed_pool"),
        "seed": stats.get("seed"),
        "psnr_u8": stats["metrics"]["psnr_u8"],
        "ssim": stats["metrics"].get("ssim"),
        "total_bpp": stats["total_bpp"],
        "grids": stats["grids"],
        "checkpoint": stats.get("checkpoint"),
    }
    path = "results/encode_time_metal16.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
