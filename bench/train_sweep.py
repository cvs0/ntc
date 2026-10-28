"""Longer-training sweep. One process, one output folder.

The GPU tier is the command to run on a machine with a ROCm, CUDA, or
DirectML torch. It does not emit per-tile endpoint files: the C++ reader
rejects ``tile_size != 0``.

CPU scaling, measured in ``results/round2_vm.json`` on this 4-thread VM with
torch 2.14.1+cpu, locked grids ``256x256x4@4`` and ``8x8x2@2``, seed 3:

- 40 steps, mip 0 only: 0.74 s train, 35.749 dB, deflate 9.756 bpp
- 800 steps, mip 0 only: 10.71 s train, 37.049 dB, deflate 10.517 bpp
- 40 steps, four mips, footprint weights: 1.34 s train, 35.892 dB

A 4000-step footprint job is about ``4000/40 * 1.34`` s of train on that CPU,
near 130 s, before export. The six long jobs plus the 800-step cross-check,
the 10 bpp pair, and the seed pool are on the order of 15 minutes of train on
that CPU. A GPU time is not measured here. The script prints the backend and
exits before the jobs if you pass ``--dry-run``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ntc.device import device_report, pick_device
from ntc.encode import TrainConfig, encode_arrays
from ntc.materials import load_material

G16 = [(256, 256, 4, 4), (8, 8, 2, 2)]
G10 = [(256, 256, 2, 4), (64, 64, 4, 8)]


def _jobs(metal: str, paving: str) -> list[dict]:
    common = dict(latent_deflate=True, log_every=10**9, channel_weighting=False)
    return [
        dict(name="metal_g16_s800_mip0", material=metal, steps=800, bpp=16, grids=G16, seed=3,
             adaptive_pool=1, seed_pool=1, mip_levels=1, mip_loss="geom", **common),
        dict(name="metal_g16_s2000_foot", material=metal, steps=2000, bpp=16, grids=G16, seed=3,
             adaptive_pool=1, seed_pool=1, mip_levels=4, mip_loss="footprint", **common),
        dict(name="metal_g16_s4000_foot", material=metal, steps=4000, bpp=16, grids=G16, seed=3,
             adaptive_pool=1, seed_pool=1, mip_levels=4, mip_loss="footprint", **common),
        dict(name="paving_g16_s2000_foot", material=paving, steps=2000, bpp=16, grids=G16, seed=3,
             adaptive_pool=1, seed_pool=1, mip_levels=4, mip_loss="footprint", **common),
        dict(name="paving_g16_s4000_foot", material=paving, steps=4000, bpp=16, grids=G16, seed=3,
             adaptive_pool=1, seed_pool=1, mip_levels=4, mip_loss="footprint", **common),
        dict(name="metal_g10_s2000_foot", material=metal, steps=2000, bpp=10, grids=G10, seed=1,
             adaptive_pool=1, seed_pool=1, mip_levels=4, mip_loss="footprint", **common),
        dict(name="paving_g10_s2000_foot", material=paving, steps=2000, bpp=10, grids=G10, seed=1,
             adaptive_pool=1, seed_pool=1, mip_levels=4, mip_loss="footprint", **common),
        dict(name="metal_pool8_s80_foot", material=metal, steps=80, bpp=16, grids=None, seed=0,
             adaptive_pool=8, seed_pool=8, mip_levels=4, mip_loss="footprint", **common),
    ]


def _smoke(metal: str) -> list[dict]:
    return [
        dict(name="smoke_metal", material=metal, steps=2, bpp=16, grids=G16, seed=3,
             adaptive_pool=1, seed_pool=1, mip_levels=1, mip_loss="geom",
             latent_deflate=True, log_every=10**9, channel_weighting=False),
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("results/train_sweep"))
    p.add_argument("--metal", type=Path, default=Path("data/materials/metal_plate"))
    p.add_argument("--paving", type=Path, default=Path("data/materials/PavingStones070"))
    p.add_argument("--tier", choices=("gpu", "smoke"), default="gpu")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = pick_device("auto")
    info = device_report(device)
    (args.out / "device.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)
    jobs = _jobs(str(args.metal), str(args.paving)) if args.tier == "gpu" else _smoke(str(args.metal))
    manifest = []
    if args.dry_run:
        (args.out / "manifest.json").write_text(json.dumps({"device": info, "jobs": [j["name"] for j in jobs]}, indent=2))
        for j in jobs:
            print("job", j["name"], "steps", j["steps"], flush=True)
        return
    for job in jobs:
        dest_json = args.out / f"{job['name']}.json"
        dest_ntc = args.out / f"{job['name']}.ntc"
        if dest_json.is_file() and dest_ntc.is_file():
            print("skip", job["name"], flush=True)
            manifest.append({"name": job["name"], "skipped": True})
            continue
        images, sems, _info = load_material(Path(job["material"]), 256)
        cfg = TrainConfig(
            steps=job["steps"],
            target_bpp=float(job["bpp"]),
            grids=job["grids"],
            seed=job["seed"],
            adaptive_pool=job["adaptive_pool"],
            seed_pool=job["seed_pool"],
            mip_levels=job["mip_levels"],
            mip_loss=job["mip_loss"],
            latent_deflate=job["latent_deflate"],
            log_every=job["log_every"],
            channel_weighting=job["channel_weighting"],
            device="auto",
        )
        t0 = time.perf_counter()
        blob, stats = encode_arrays(images, sems, cfg)
        wall = time.perf_counter() - t0
        dest_ntc.write_bytes(blob)
        row = {
            "name": job["name"],
            "wall_s": wall,
            "psnr_u8": stats["metrics"]["psnr_u8"],
            "ssim": stats["metrics"].get("ssim"),
            "total_bpp": stats["total_bpp"],
            "latent_stored_bytes": stats.get("latent_stored_bytes"),
            "latent_plain_bytes": stats.get("latent_plain_bytes"),
            "checkpoint": stats.get("checkpoint"),
            "seed": stats.get("seed"),
            "grids": stats.get("grids"),
            "mip_metrics": stats.get("mip_metrics"),
            "device": stats.get("device"),
            "seconds_train": stats.get("seconds_train"),
            "steps": job["steps"],
            "mip_levels": job["mip_levels"],
            "mip_loss": job["mip_loss"],
        }
        dest_json.write_text(json.dumps(row, indent=2))
        manifest.append(row)
        (args.out / "manifest.json").write_text(json.dumps({"device": info, "jobs": manifest}, indent=2))
        print(
            f"{job['name']} psnr {row['psnr_u8']:.4f} bpp {row['total_bpp']:.4f} wall {wall:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
