"""Fast training sweep. Writes ``results/train_max_v2`` and does not open ``results/train_max``.

The job list matches ``bench/train_max.py``. Each seed pool is one stacked run
with successive halving at 250 steps, then the surviving seed continues to
the same step count. Precision is chosen by PSNR at equal wall time, not by
PSNR at equal steps. ``TrainConfig.fast`` stays off in ``train_max.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from bench.train_max import G16
from ntc.device import device_report, pick_device
from ntc.encode import TrainConfig
from ntc.materials import load_material
from ntc.train_fast import choose_precision, encode_arrays_fast


def parse_shard(text: str) -> tuple[int, int]:
    """``I/N`` with ``0 <= I < N``."""
    shard_s, n_s = text.split("/")
    shard, n = int(shard_s), int(n_s)
    if n < 1 or shard < 0 or shard >= n:
        raise SystemExit(f"bad shard {text}")
    return shard, n


def owns_index(index: int, shard: int | None, n: int | None) -> bool:
    if shard is None or n is None:
        return True
    return index % n == shard


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _lock_stale(lock: Path, stale_s: float) -> bool:
    try:
        payload = json.loads(lock.read_text())
        pid = int(payload.get("pid", 0))
    except (OSError, ValueError, json.JSONDecodeError):
        return True
    age = time.time() - lock.stat().st_mtime
    if age > stale_s:
        return True
    return not _pid_alive(pid)


def claim_job(out: Path, name: str, stale_s: float) -> bool:
    """Atomic create of ``locks/<name>.lock``. False if a live worker holds it."""
    lock = out / "locks" / f"{name}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        if not _lock_stale(lock, stale_s):
            return False
        try:
            lock.unlink()
        except OSError:
            return False
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(str(lock), flags)
    except FileExistsError:
        return False
    try:
        os.write(fd, json.dumps({"pid": os.getpid(), "name": name, "time": time.time()}).encode())
    finally:
        os.close(fd)
    return True


def release_job(out: Path, name: str) -> None:
    lock = out / "locks" / f"{name}.lock"
    try:
        lock.unlink()
    except OSError:
        pass


def _run(job: dict, out: Path, precision: str, shard_label: str, stale_s: float) -> dict:
    dest = out / f"{job['name']}.json"
    ntc_path = out / f"{job['name']}.ntc"
    ckpt = out / f"{job['name']}.pt"
    if dest.is_file():
        prev = json.loads(dest.read_text())
        if prev.get("done"):
            print("skip", job["name"], flush=True)
            return prev
    if not claim_job(out, job["name"], stale_s):
        print("locked", job["name"], flush=True)
        return {"name": job["name"], "done": False, "locked": True}
    try:
        return _run_claimed(job, out, precision, shard_label, dest, ntc_path, ckpt)
    finally:
        release_job(out, job["name"])


def _run_claimed(job: dict, out: Path, precision: str, shard_label: str, dest: Path, ntc_path: Path, ckpt: Path) -> dict:
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
        lr_batch_ref=int(job.get("lr_batch_ref", 0)),
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
        "shard": shard_label,
        "worker_pid": os.getpid(),
        "batch_pixels": int(job.get("batch_pixels", 0)),
        "lr_batch_ref": int(job.get("lr_batch_ref", 0)),
        "compiled": stats.get("compiled"),
        "cuda_graph": stats.get("cuda_graph"),
        "halving": stats.get("halving"),
        "device": stats.get("device"),
    }
    dest.write_text(json.dumps(row, indent=2))
    sps = row["steps_per_s"]
    sps_txt = f"{sps:.2f}" if isinstance(sps, (int, float)) else "na"
    print(
        f"worker {shard_label or '0/1'} pid {os.getpid()} {job['name']} "
        f"steps/s {sps_txt} psnr {row['psnr_u8']:.4f} total_bpp {row['total_bpp']:.4f} "
        f"wall {wall:.1f}",
        flush=True,
    )
    return row


def build_jobs(root: Path, batch_pixels: int | None) -> list[dict]:
    """Same names and order as the original sweep. ``batch_pixels`` overrides 256 jobs only."""
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
            "seed_pool": kw.pop("seed_pool", 1),
            "halve_steps": kw.pop("halve_steps", 0),
            "lr_batch_ref": 0,
        }
        row.update(kw)
        if batch_pixels is not None and row["size"] == 256 and int(row["batch_pixels"]) == 0:
            row["batch_pixels"] = int(batch_pixels)
            row["lr_batch_ref"] = 256 * 256
        return row

    jobs = [
        base("metal_g16_s20000", metal, 20000, 16, 0, grids=G16, seed_pool=8, halve_steps=250),
        base("paving_g16_s20000", paving, 20000, 16, 0, grids=G16, seed_pool=8, halve_steps=250),
    ]
    for bpp in (4, 6, 8, 10, 12):
        jobs.append(base(f"metal_b{bpp}_s10000", metal, 10000, bpp, 0, grids=None, adaptive_pool=4, seed_pool=4, halve_steps=250))
        jobs.append(base(f"paving_b{bpp}_s10000", paving, 10000, bpp, 0, grids=None, adaptive_pool=4, seed_pool=4, halve_steps=250))
    jobs.append(base("metal_g16_step_s10000", metal, 10000, 16, 3, schedule="step"))
    for name in ("wood_floor", "denim_fabric", "aerial_rocks_01", "plastered_wall"):
        jobs.append(base(f"{name}_b16_s10000", root / name, 10000, 16, 0, grids=None, adaptive_pool=4, seed_pool=4, halve_steps=250))
    jobs.append(base("wood_floor_1k_b16_s4000", root / "wood_floor", 4000, 16, 0, grids=None, size=1024, adaptive_pool=4, seed_pool=2, halve_steps=250))
    jobs.append(base("wood_floor_2k_b16_s1000", root / "wood_floor_2k", 1000, 16, 0, grids=None, size=2048, batch_pixels=1024 * 1024, adaptive_pool=2, seed_pool=2, halve_steps=250))
    jobs.append(base("aerial_rocks_2k_b16_s1000", root / "aerial_rocks_01_2k", 1000, 16, 0, grids=None, size=2048, batch_pixels=1024 * 1024, adaptive_pool=2, seed_pool=2, halve_steps=250))
    for name in ("MetalPlates013_2K", "PavingStones070_2K", "Tiles135D_2K"):
        for bpp in (2.5, 5, 10, 15, 20):
            jobs.append(base(
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
            ))
    return jobs


def _ensure_precision(args: argparse.Namespace) -> str:
    path = args.out / "precision.json"
    if path.is_file():
        report = json.loads(path.read_text())
        choice = report.get("equal_wall_choice", "fp32")
        print("precision", choice, "from", path, flush=True)
        return choice
    metal = args.materials / "metal_plate"
    if not (metal / "manifest.json").is_file():
        print("missing", metal)
        return "fp32"
    images, _sems, _info = load_material(metal, 256)
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
    # One lock so shard workers do not each train the probe.
    if not claim_job(args.out, "precision", stale_s=args.stale_lock_s):
        for _ in range(60):
            time.sleep(1.0)
            if path.is_file():
                return json.loads(path.read_text()).get("equal_wall_choice", "fp32")
        print("precision lock held, using fp32", flush=True)
        return "fp32"
    try:
        if path.is_file():
            return json.loads(path.read_text()).get("equal_wall_choice", "fp32")
        precision_report = choose_precision(images, _sems, probe_cfg, args.probe_steps)
        path.write_text(json.dumps(precision_report, indent=2))
        precision = precision_report["equal_wall_choice"]
        print("precision", precision, "historical", precision_report["historical_equal_steps_choice"], flush=True)
        return precision
    finally:
        release_job(args.out, "precision")


def _spawn_workers(args: argparse.Namespace) -> int:
    script = str(Path(__file__).resolve())
    procs = []
    for i in range(args.workers):
        cmd = [
            sys.executable,
            script,
            "--out", str(args.out),
            "--materials", str(args.materials),
            "--probe-steps", str(args.probe_steps),
            "--stale-lock-s", str(args.stale_lock_s),
            "--shard", f"{i}/{args.workers}",
        ]
        if args.batch_pixels is not None:
            cmd.extend(["--batch-pixels", str(args.batch_pixels)])
        if args.dry_run:
            cmd.append("--dry-run")
        procs.append(subprocess.Popen(cmd))
    rc = 0
    for proc in procs:
        code = proc.wait()
        if code:
            rc = code
    return rc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("results/train_max_v2"))
    p.add_argument("--materials", type=Path, default=Path("data/materials"))
    p.add_argument("--probe-steps", type=int, default=400)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--shard", default="", help="I/N. This process runs jobs where index %% N == I.")
    p.add_argument("--workers", type=int, default=0, help="Spawn N shard subprocesses and wait.")
    p.add_argument("--batch-pixels", type=int, default=None, help="Override the pixel batch on 256 jobs. LR scales by batch/(256*256).")
    p.add_argument("--stale-lock-s", type=float, default=86400.0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.workers and args.shard:
        raise SystemExit("pass --workers or --shard, not both")
    if args.workers:
        if args.workers < 1:
            raise SystemExit("--workers must be >= 1")
        # Probe once in the parent so the shards do not each train it.
        if not args.dry_run:
            device = pick_device("auto")
            (args.out / "device.json").write_text(json.dumps(device_report(device), indent=2))
            _ensure_precision(args)
        raise SystemExit(_spawn_workers(args))
    shard_i, shard_n = (None, None)
    shard_label = ""
    if args.shard:
        shard_i, shard_n = parse_shard(args.shard)
        shard_label = args.shard
    device = pick_device("auto")
    info = device_report(device)
    (args.out / "device.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({"shard": shard_label or "0/1", "pid": os.getpid(), **info}), flush=True)
    jobs = build_jobs(args.materials, args.batch_pixels)
    if args.dry_run:
        for index, job in enumerate(jobs):
            if owns_index(index, shard_i, shard_n):
                print(f"plan {index} {job['name']} batch {job['batch_pixels']} lr_ref {job['lr_batch_ref']}", flush=True)
        return
    precision = _ensure_precision(args)
    for index, job in enumerate(jobs):
        if not owns_index(index, shard_i, shard_n):
            continue
        _run(job, args.out, precision, shard_label, args.stale_lock_s)
    print("wrote", args.out, "shard", shard_label or "0/1", flush=True)


if __name__ == "__main__":
    main()
