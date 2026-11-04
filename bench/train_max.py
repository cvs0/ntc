"""Max-quality training sweep for a 16 GB GPU.

Resumable. A finished job is ``results/train_max/<name>.json`` with ``done``
true. A ``.pt`` next to it is the best Adam state. Re-running the script
continues a ``.pt`` that has no finished json.

The step-time used for the printed estimate is the measured
``metal_g16_s4000_foot`` wall on the RX 9070: 118.1 s for 4000 steps.
A 1024 or 2048 job with ``batch_pixels`` 0 costs ``(size/256)**2`` times
that. A 2048 job in this file uses ``batch_pixels = 1024*1024``, so the
multiplier is 16, not 64. Early stop can finish sooner. The estimate is
an upper bound, not a measurement of this run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from ntc.device import device_report, pick_device
from ntc.encode import TrainConfig, encode_arrays
from ntc.materials import load_material

G16 = [(256, 256, 4, 4), (8, 8, 2, 2)]
# metal_g16_s4000_foot wall on RX 9070, footprint, 256^2.
SEC_PER_STEP_256 = 118.1 / 4000.0


def _scale(size: int, batch_pixels: int) -> float:
    n = size * size
    if batch_pixels and 0 < batch_pixels < n:
        n = batch_pixels
    return n / float(256 * 256)


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
        seed_pool=1,
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
        with_ssim=True,
    )
    t0 = time.perf_counter()
    blob, stats = encode_arrays(images, sems, cfg)
    wall = time.perf_counter() - t0
    ntc_path.write_bytes(blob)
    row = {
        "name": job["name"],
        "done": True,
        "wall_s": wall,
        "psnr_u8": stats["metrics"]["psnr_u8"],
        "ssim": stats["metrics"].get("ssim"),
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
        "precision": precision,
        "steps": job["steps"],
        "size": job["size"],
        "seconds_train": stats.get("seconds_train"),
        "device": stats.get("device"),
    }
    dest.write_text(json.dumps(row, indent=2))
    print(
        f"{job['name']} psnr {row['psnr_u8']:.4f} total_bpp {row['total_bpp']:.4f} "
        f"latent_bpp_mip0 {row['latent_bpp_mip0']:.4f} file_bpp_chain {row['file_bpp_chain']:.4f} "
        f"bppc {row['bppc']:.4f} wall {wall:.1f}",
        flush=True,
    )
    return row


def _continue(out: Path, src_name: str, dst: dict, precision: str) -> dict:
    src = out / f"{src_name}.pt"
    ckpt = out / f"{dst['name']}.pt"
    if src.is_file() and not ckpt.is_file() and not (out / f"{dst['name']}.json").is_file():
        shutil.copy(src, ckpt)
    return _run(dst, out, precision)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("results/train_max"))
    p.add_argument("--materials", type=Path, default=Path("data/materials"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = pick_device("auto")
    info = device_report(device)
    (args.out / "device.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)
    # Upper bound at the measured 256-step cost. See the module docstring.
    plan = [
        ("probe", 800, 1.0),
        ("metal seeds + 20k", 8000 + 19000, 1.0),
        ("paving seeds + 20k", 8000 + 19000, 1.0),
        ("metal rates x5", 5 * 13000, 1.0),
        ("paving rates x5", 5 * 13000, 1.0),
        ("step decay 10k", 10000, 1.0),
        ("four extra materials, 4 seeds x 1k then 10k", 4 * 13000, 1.0),
        ("wood_floor 1k, 2 seeds x 1k then 4k", 5000, 16.0),
        ("two 2k materials, 2 seeds x 1k, batch 1024^2", 2000, 16.0),
        ("sdk 2k x3 x5 latent rates, 4k cosine, batch 1024^2", 3 * 5 * 4000, 16.0),
    ]
    est = sum(steps * scale * SEC_PER_STEP_256 for _n, steps, scale in plan)
    print(f"estimate_s {est:.0f} from 118.1 s / 4000 steps, before early stop", flush=True)
    if args.dry_run:
        for name, steps, scale in plan:
            print(f"plan {name} steps {steps} scale {scale}", flush=True)
        return
    root = args.materials
    metal = root / "metal_plate"
    paving = root / "PavingStones070"

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
        }
        row.update(kw)
        return row

    fp = _run(base("probe_fp32_400", metal, 400, 16, 3, eval_every=0, early_stop=0, warmup=0), args.out, "fp32")
    precision = "fp32"
    try:
        bf = _run(
            base("probe_bf16_400", metal, 400, 16, 3, eval_every=0, early_stop=0, warmup=0),
            args.out,
            "bf16",
        )
        if bf.get("done") and fp.get("done") and bf["psnr_u8"] > fp["psnr_u8"]:
            precision = "bf16"
    except Exception as exc:
        (args.out / "probe_bf16_error.txt").write_text(str(exc))
        print("bf16 probe failed", exc, flush=True)
    (args.out / "precision.json").write_text(json.dumps({"precision": precision}, indent=2))
    print("precision", precision, flush=True)

    def seeds_then(tag, material, final_steps, bpp, nseeds, grids, size=256, batch=0, adaptive=1):
        rows = []
        for s in range(nseeds):
            rows.append(
                _run(
                    base(
                        f"{tag}_seed{s}_s1000",
                        material,
                        1000,
                        bpp,
                        s,
                        grids=grids,
                        size=size,
                        batch_pixels=batch,
                        adaptive_pool=adaptive,
                        eval_every=250,
                        early_stop=0,
                    ),
                    args.out,
                    precision,
                )
            )
        done = [r for r in rows if r.get("done")]
        if not done:
            return None
        best = max(done, key=lambda r: r["psnr_u8"])
        return _continue(
            args.out,
            best["name"],
            base(
                f"{tag}_s{final_steps}",
                material,
                final_steps,
                bpp,
                int(best["seed"]),
                grids=[tuple(g) for g in best["grids"]] if best.get("grids") else grids,
                size=size,
                batch_pixels=batch,
                adaptive_pool=1,
            ),
            precision,
        )

    seeds_then("metal_g16", metal, 20000, 16, 8, G16)
    seeds_then("paving_g16", paving, 20000, 16, 8, G16)
    for bpp in (4, 6, 8, 10, 12):
        seeds_then(f"metal_b{bpp}", metal, 10000, bpp, 4, None, adaptive=4)
        seeds_then(f"paving_b{bpp}", paving, 10000, bpp, 4, None, adaptive=4)
    _run(
        base("metal_g16_step_s10000", metal, 10000, 16, 3, schedule="step"),
        args.out,
        precision,
    )
    for name in ("wood_floor", "denim_fabric", "aerial_rocks_01", "plastered_wall"):
        seeds_then(f"{name}_b16", root / name, 10000, 16, 4, None, adaptive=4)
    seeds_then("wood_floor_1k_b16", root / "wood_floor", 4000, 16, 2, None, size=1024, adaptive=4)
    seeds_then(
        "wood_floor_2k_b16",
        root / "wood_floor_2k",
        1000,
        16,
        2,
        None,
        size=2048,
        batch=1024 * 1024,
        adaptive=2,
    )
    seeds_then(
        "aerial_rocks_2k_b16",
        root / "aerial_rocks_01_2k",
        1000,
        16,
        2,
        None,
        size=2048,
        batch=1024 * 1024,
        adaptive=2,
    )
    # SDK materials at 2048. The target is latent_bpp_mip0, the same kind of
    # number as ntc-cli --bitsPerPixel. Download with
    # bench/download_materials.py --preset rtxntc. A missing directory is
    # recorded and skipped.
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
