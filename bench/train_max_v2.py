"""Fast training sweep. Writes ``results/train_max_v2`` and does not open ``results/train_max``.

The job list matches ``bench/train_max.py``. Each seed pool is one stacked run
with successive halving at 250 steps, then the surviving seed continues to
the same step count. Precision is chosen by PSNR at equal wall time, not by
PSNR at equal steps. ``TrainConfig.fast`` stays off in ``train_max.py``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from bench.train_max import G16
from ntc.device import device_report, pick_device
from ntc.encode import TrainConfig
from ntc.materials import load_material
from ntc.train_fast import choose_precision, encode_arrays_fast


def _run(job: dict, out: Path, precision: str) -> dict:
    dest = out / f"{job['name']}.json"
    ntc_path = out / f"{job['name']}.ntc"
    ckpt = out / f"{job['name']}.pt"
    if dest.is_file():
        prev = json.loads(dest.read_text())
        if prev.get("done"):
            print("skip", job["name"], flush=True)
            return prev
    mat = Path(job["material"])
    if not (mat / "manifest.json").is_file():
        row = {"name": job["name"], "done": False, "error": f"missing {mat}"}
        dest.write_text(json.dumps(row, indent=2))
        print("missing", job["name"], mat, flush=True)
        return row
    images, sems, _info = load_material(mat, int(job["size"]))
    cfg = TrainConfig(
        steps=int(job["steps"]),
        target_bpp=float(job["bpp"]),
        grids=None if job.get("grids") is None else [tuple(g) for g in job["grids"]],
        seed=int(job["seed"]),
        adaptive_pool=int(job.get("adaptive_pool", 1)),
        seed_pool=int(job.get("seed_pool", 1)),
        latent_deflate=True,
        latent_coder="best",
        rate=str(job.get("rate", "payload")),
        mip_levels=int(job.get("mip_levels", 4)),
        mip_loss="footprint",
        warmup_steps=int(job.get("warmup", 200)),
        schedule=str(job.get("schedule", "cosine")),
        precision=precision,
        eval_every=int(job.get("eval_every", 500)),
        early_stop=int(job.get("early_stop", 8)),
        batch_pixels=int(job.get("batch_pixels", 0)),
        resume=str(ckpt) if ckpt.is_file() else "",
        ckpt=str(ckpt),
        log_every=10**9,
        device="auto",
        with_ssim=False,
        fast=True,
        halve_steps=int(job.get("halve_steps", 250 if int(job.get("seed_pool", 1)) > 1 else 0)),
        compile_step="auto",
        cuda_graph="auto",
    )
    t0 = time.perf_counter()
    blob, stats = encode_arrays_fast(images, sems, cfg)
    wall = time.perf_counter() - t0
    ntc_path.write_bytes(blob)
    row = {
        "name": job["name"],
        "done": True,
        "fast": True,
        "wall_s": wall,
        "psnr_u8": stats["metrics"]["psnr_u8"],
        "total_bpp": stats["total_bpp"],
        "latent_bpp_mip0": stats.get("latent_bpp_mip0"),
        "file_bpp_chain": stats.get("file_bpp_chain"),
        "bppc": stats.get("bppc"),
        "latent_stored_bytes": stats.get("latent_stored_bytes"),
        "latent_plain_bytes": stats.get("latent_plain_bytes"),
        "flags_coder": stats.get("latent_coder"),
        "checkpoint": stats.get("checkpoint"),
        "seed": stats.get("seed"),
        "grids": stats.get("grids"),
        "stopped_early": stats.get("stopped_early"),
        "best_val_psnr": stats.get("best_val_psnr"),
        "precision": stats.get("precision", precision),
        "steps": job["steps"],
        "size": job["size"],
        "seconds_train": stats.get("seconds_train"),
        "steps_per_s": stats.get("steps_per_s"),
        "compiled": stats.get("compiled"),
        "cuda_graph": stats.get("cuda_graph"),
        "halving": stats.get("halving"),
        "device": stats.get("device"),
    }
    dest.write_text(json.dumps(row, indent=2))
    print(
        f"{job['name']} psnr {row['psnr_u8']:.4f} total_bpp {row['total_bpp']:.4f} "
        f"steps/s {row['steps_per_s']} wall {wall:.1f}",
        flush=True,
    )
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("results/train_max_v2"))
    p.add_argument("--materials", type=Path, default=Path("data/materials"))
    p.add_argument("--probe-steps", type=int, default=400)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = pick_device("auto")
    info = device_report(device)
    (args.out / "device.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)
    if args.dry_run:
        print("dry-run", args.out)
        return
    root = args.materials
    metal = root / "metal_plate"
    paving = root / "PavingStones070"
    if not (metal / "manifest.json").is_file():
        print("missing", metal)
        return
    images, sems, _info = load_material(metal, 256)
    probe_cfg = TrainConfig(
        steps=args.probe_steps,
        grids=G16,
        hidden=(32,),
        seed=3,
        adaptive_pool=1,
        seed_pool=1,
        with_ssim=False,
        mip_levels=4,
        mip_loss="footprint",
        device="auto",
    )
    precision_report = choose_precision(images, sems, probe_cfg, args.probe_steps)
    (args.out / "precision.json").write_text(json.dumps(precision_report, indent=2))
    precision = precision_report["equal_wall_choice"]
    print("precision", precision, "historical", precision_report["historical_equal_steps_choice"], flush=True)

    def base(name, material, steps, bpp, seed, **kw):
        row = {
            "name": name,
            "material": str(material),
            "size": kw.pop("size", 256),
            "steps": steps,
            "bpp": bpp,
            "seed": seed,
            "warmup": kw.pop("warmup", 200),
            "schedule": kw.pop("schedule", "cosine"),
            "eval_every": kw.pop("eval_every", 500),
            "early_stop": kw.pop("early_stop", 8),
            "batch_pixels": kw.pop("batch_pixels", 0),
            "mip_levels": kw.pop("mip_levels", 4),
            "adaptive_pool": kw.pop("adaptive_pool", 1),
            "grids": kw.pop("grids", G16),
            "rate": kw.pop("rate", "payload"),
            "seed_pool": kw.pop("seed_pool", 1),
            "halve_steps": kw.pop("halve_steps", 0),
        }
        row.update(kw)
        return row

    def seeds_then(tag, material, final_steps, bpp, nseeds, grids, size=256, batch=0, adaptive=1):
        # One stacked run. Halving at 250 drops seeds. The survivor keeps
        # training through ``final_steps`` inside that same call.
        return _run(
            base(
                f"{tag}_s{final_steps}",
                material,
                final_steps,
                bpp,
                0,
                grids=grids,
                size=size,
                batch_pixels=batch,
                adaptive_pool=adaptive,
                seed_pool=nseeds,
                halve_steps=250,
            ),
            args.out,
            precision,
        )

    seeds_then("metal_g16", metal, 20000, 16, 8, G16)
    seeds_then("paving_g16", paving, 20000, 16, 8, G16)
    for bpp in (4, 6, 8, 10, 12):
        seeds_then(f"metal_b{bpp}", metal, 10000, bpp, 4, None, adaptive=4)
        seeds_then(f"paving_b{bpp}", paving, 10000, bpp, 4, None, adaptive=4)
    _run(base("metal_g16_step_s10000", metal, 10000, 16, 3, schedule="step"), args.out, precision)
    for name in ("wood_floor", "denim_fabric", "aerial_rocks_01", "plastered_wall"):
        seeds_then(f"{name}_b16", root / name, 10000, 16, 4, None, adaptive=4)
    seeds_then("wood_floor_1k_b16", root / "wood_floor", 4000, 16, 2, None, size=1024, adaptive=4)
    seeds_then(
        "wood_floor_2k_b16", root / "wood_floor_2k", 1000, 16, 2, None,
        size=2048, batch=1024 * 1024, adaptive=2,
    )
    seeds_then(
        "aerial_rocks_2k_b16", root / "aerial_rocks_01_2k", 1000, 16, 2, None,
        size=2048, batch=1024 * 1024, adaptive=2,
    )
    for name in ("MetalPlates013_2K", "PavingStones070_2K", "Tiles135D_2K"):
        for bpp in (2.5, 5, 10, 15, 20):
            _run(
                base(
                    f"sdk_{name}_b{bpp}",
                    root / name,
                    4000,
                    bpp,
                    0,
                    size=2048,
                    batch_pixels=1024 * 1024,
                    mip_levels=12,
                    rate="latent",
                    adaptive_pool=1,
                    grids=None,
                    warmup=400,
                    schedule="cosine",
                    eval_every=500,
                    early_stop=6,
                ),
                args.out,
                precision,
            )
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
