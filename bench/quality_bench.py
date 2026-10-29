"""Reduced-resolution cloudfast ladder on CPU.

Runs the same job list, chain order, warm starts, rate control and export
as ``train_max_v2 --profile cloudfast`` with a smaller image, step budget
and batch, so a training change can be compared before a cloud run.

    python bench/quality_bench.py --materials data/materials --out /tmp/qb \
        --size 512 --steps-scale 0.2 --batch 131072 --only Bark001
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_max_v2 as tm  # noqa: E402


def scaled_jobs(materials: Path, size: int, steps_scale: float, batch: int, only: list[str]) -> list[dict]:
    jobs = []
    for job in tm.build_cloud_jobs(materials, "cloudfast"):
        if only and not any(f"_{name}_" in job["name"] for name in only):
            continue
        job = dict(job)
        job["size"] = int(size)
        job["steps"] = max(50, int(round(job["steps"] * steps_scale)))
        job["warmup"] = max(10, int(round(job["warmup"] * steps_scale)))
        if job.get("full_steps"):
            job["full_steps"] = max(50, int(round(job["full_steps"] * steps_scale)))
            job["full_warmup"] = max(10, int(round(job["full_warmup"] * steps_scale)))
        job["eval_every"] = max(25, int(round(job["eval_every"] * steps_scale)))
        job["halve_steps"] = max(25, int(round(job["halve_steps"] * steps_scale)))
        job["batch_pixels"] = int(batch)
        jobs.append(job)
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--materials", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps-scale", type=float, default=0.2)
    ap.add_argument("--batch", type=int, default=131072)
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--labels", nargs="*", type=float, default=[])
    ap.add_argument("--seed-pool", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = scaled_jobs(args.materials, args.size, args.steps_scale, args.batch, args.only)
    if args.labels:
        jobs = [j for j in jobs if any(abs(j["bpp"] - b) < 1e-6 for b in args.labels)]
    if args.seed_pool:
        for j in jobs:
            j["seed_pool"] = args.seed_pool
    rows = []
    for job in jobs:
        row = tm._run(job, args.out, "fp32", "bench", 86400.0)
        rows.append({
            "name": job["name"],
            "target": job["bpp"],
            "bpp": row.get("total_bpp"),
            "psnr": row.get("psnr_u8"),
            "grids": row.get("grids"),
        })
        print("QB", json.dumps(rows[-1]), flush=True)
    (args.out / "qb_summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
