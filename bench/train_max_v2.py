"""Fast training sweep. Writes ``results/train_max_v2`` and does not open ``results/train_max``.

The job list matches ``bench/train_max.py``. Each seed pool is one stacked run
with successive halving at 250 steps, then the surviving seed continues to
the same step count. Precision is chosen by PSNR at equal wall time, not by
PSNR at equal steps. ``TrainConfig.fast`` stays off in ``train_max.py``.

``--workers N`` starts N processes on one shared lock pool. Each process
claims the next undone job (highest pixels * steps first) and
an idle process takes whatever is still unlocked. ``--shard I/N`` is that
same cost order, split longest-job-first, and does not steal.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
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


def job_cost(job: dict) -> float:
    """Estimated work: pixels per step times steps.

    A ``batch_pixels`` below the frame counts as the batch. A batch of 0, or a
    batch at least as large as the frame, counts as the full frame. The 256
    sampler ignores a batch larger than the frame. A seed pool is one stacked
    step, so it is not a second factor.
    """
    size = int(job["size"])
    full = size * size
    batch = int(job.get("batch_pixels") or 0)
    pixels = batch if 0 < batch < full else full
    return float(pixels) * float(job["steps"])


def assign_shards(jobs: list[dict], n: int) -> list[int]:
    """Longest-processing-time-first. One shard id per job, in job order."""
    if n < 1:
        raise SystemExit("shard count must be >= 1")
    order = sorted(range(len(jobs)), key=lambda i: job_cost(jobs[i]), reverse=True)
    load = [0.0] * n
    assign = [0] * len(jobs)
    for index in order:
        shard = min(range(n), key=lambda s: (load[s], s))
        assign[index] = shard
        load[shard] += job_cost(jobs[index])
    return assign


_SDK_DIRS = ("MetalPlates013_2K", "PavingStones070_2K", "Tiles135D_2K")


def _sdk_missing(root: Path) -> list[Path]:
    return [root / name for name in _SDK_DIRS if not (root / name / "manifest.json").is_file()]


def ensure_sdk_materials(root: Path) -> None:
    """Download the RTXNTC 2K set when a manifest is missing, then stop if it still is.

    This is ``download_materials --preset rtxntc``. Workers are not started
    while one of the three directories is absent, so a process cannot exit
    early on a job list that is only those materials. A lock file keeps two
    processes from unzipping the same archive.
    """
    if not _sdk_missing(root):
        return
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".rtxntc_download.lock"
    while _sdk_missing(root):
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if _lock_stale(lock, 3600):
                try:
                    lock.unlink()
                except OSError:
                    pass
            else:
                time.sleep(1.0)
            continue
        try:
            os.write(fd, json.dumps({"pid": os.getpid(), "name": "rtxntc"}).encode())
        finally:
            os.close(fd)
        try:
            gap = _sdk_missing(root)
            if not gap:
                return
            print("download rtxntc", " ".join(path.name for path in gap), flush=True)
            try:
                from bench.download_materials import sdk_materials

                sdk_materials(root)
            except Exception as exc:
                raise SystemExit(f"download_materials --preset rtxntc failed: {exc}") from exc
            gap = _sdk_missing(root)
            if gap:
                raise SystemExit("missing SDK materials after download: " + ", ".join(str(path) for path in gap))
        finally:
            try:
                lock.unlink()
            except OSError:
                pass
        return


def _job_blocked(out: Path, job: dict) -> bool:
    """Skip a finished json. Skip an error json only while its material is still absent.

    A previous run records ``error`` for a missing directory. Once that
    directory has a manifest, the next process trains it. Inside one process
    the caller also passes a skip set so a crash is not claimed again.
    """
    dest = out / f"{job['name']}.json"
    if not dest.is_file():
        return False
    try:
        prev = json.loads(dest.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if prev.get("done"):
        return True
    if prev.get("error"):
        mat = Path(job["material"])
        return not (mat / "manifest.json").is_file()
    return False


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
        "steps_ran": stats.get("steps_ran"),
        "psnr_u8_init": stats.get("psnr_u8_init"),
        "psnr_u8_trained": stats.get("psnr_u8_trained"),
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
        f"steps/s {sps_txt} steps_ran {row.get('steps_ran')} ckpt {row.get('checkpoint')} "
        f"psnr {row['psnr_u8']:.4f} init {row.get('psnr_u8_init')} trained {row.get('psnr_u8_trained')} "
        f"total_bpp {row['total_bpp']:.4f} stopped {row.get('stopped_early')} wall {wall:.1f}",
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
    """N processes share the lock pool. They do not get a fixed ``--shard``."""
    script = str(Path(__file__).resolve())
    procs = []
    for _i in range(args.workers):
        cmd = [
            sys.executable,
            script,
            "--out", str(args.out),
            "--materials", str(args.materials),
            "--probe-steps", str(args.probe_steps),
            "--stale-lock-s", str(args.stale_lock_s),
            "--queue",
        ]
        if args.batch_pixels is not None:
            cmd.extend(["--batch-pixels", str(args.batch_pixels)])
        if not args.pack:
            cmd.append("--no-pack")
        if args.dry_run:
            cmd.append("--dry-run")
        procs.append(subprocess.Popen(cmd))
    rc = 0
    for proc in procs:
        code = proc.wait()
        if code:
            rc = code
    return rc


def work_units(jobs: list[dict], pack: bool) -> list[list[dict]]:
    """One unit per job, or one unit per pack group when ``pack`` is set."""
    if not pack:
        return [[job] for job in jobs]
    groups: dict[tuple, list[dict]] = {}
    units: list[list[dict]] = []
    for job in jobs:
        key = _pack_key(job)
        if key is None:
            units.append([job])
            continue
        bucket = groups.get(key)
        if bucket is None:
            bucket = []
            groups[key] = bucket
            units.append(bucket)
        bucket.append(job)
    return units


def claim_next(jobs: list[dict], out: Path, stale_s: float, pack: bool, skip: set[str] | None = None) -> list[dict]:
    """Claim the heaviest unlocked unit. A live lock is left for its owner.

    Returns the jobs this process now holds. Empty when every remaining job
    is finished, already erred with a missing material, or locked by a live pid.
    """
    blocked = skip or set()
    units = work_units(jobs, pack)
    units.sort(key=lambda unit: sum(job_cost(job) for job in unit), reverse=True)
    for unit in units:
        claimed: list[dict] = []
        try:
            for job in unit:
                if job["name"] in blocked or _job_blocked(out, job):
                    continue
                if not claim_job(out, job["name"], stale_s):
                    continue
                if _job_blocked(out, job):
                    release_job(out, job["name"])
                    continue
                claimed.append(job)
        except Exception:
            for job in claimed:
                release_job(out, job["name"])
            raise
        if claimed:
            return claimed
    return []


def _queue_loop(jobs: list[dict], out: Path, precision: str, stale_s: float, pack: bool) -> None:
    """Steal from the lock pool until nothing is left to claim."""
    failed: set[str] = set()
    label = "queue"
    while True:
        claimed = claim_next(jobs, out, stale_s, pack, skip=failed)
        if not claimed:
            return
        try:
            if len(claimed) == 1:
                job = claimed[0]
                _run_claimed(
                    job,
                    out,
                    precision,
                    label,
                    out / f"{job['name']}.json",
                    out / f"{job['name']}.ntc",
                    out / f"{job['name']}.pt",
                )
            else:
                _train_claimed_pack(claimed, out, precision, label)
        except Exception as exc:
            traceback.print_exc()
            for job in claimed:
                failed.add(job["name"])
                dest = out / f"{job['name']}.json"
                keep = False
                if dest.is_file():
                    try:
                        keep = bool(json.loads(dest.read_text()).get("done"))
                    except (OSError, ValueError, json.JSONDecodeError):
                        keep = False
                if not keep:
                    dest.write_text(json.dumps({"name": job["name"], "done": False, "error": str(exc)}, indent=2))
            print("fail", " ".join(job["name"] for job in claimed), exc, flush=True)
        finally:
            for job in claimed:
                release_job(out, job["name"])


def _pack_key(job: dict) -> tuple | None:
    grids = job.get("grids")
    if not grids:
        return None
    return (
        int(job["size"]),
        int(job["steps"]),
        str(job["schedule"]),
        int(job["mip_levels"]),
        int(job["seed_pool"]),
        int(job["warmup"]),
        tuple(tuple(int(x) for x in g) for g in grids),
    )


def _train_claimed_pack(claimed: list[dict], out: Path, precision: str, shard_label: str) -> None:
    """Train jobs whose locks this process already holds. Does not release them."""
    from ntc.train_fast import encode_bank

    buckets: dict[tuple, list] = {}
    for job in claimed:
        mat = Path(job["material"])
        if not (mat / "manifest.json").is_file():
            row = {"name": job["name"], "done": False, "error": f"missing {mat}"}
            (out / f"{job['name']}.json").write_text(json.dumps(row, indent=2))
            print("missing", job["name"], mat, flush=True)
            continue
        images, sems, _info = load_material(mat, int(job["size"]))
        buckets.setdefault((images.shape[0], tuple(sems)), []).append((job, images, sems))
    for (_channels, sems), items in buckets.items():
        if len(items) == 1:
            job = items[0][0]
            _run_claimed(
                job,
                out,
                precision,
                shard_label,
                out / f"{job['name']}.json",
                out / f"{job['name']}.ntc",
                out / f"{job['name']}.pt",
            )
            continue
        first = items[0][0]
        cfg = TrainConfig(
            steps=int(first["steps"]),
            target_bpp=float(first["bpp"]),
            grids=[tuple(g) for g in first["grids"]],
            seed=int(first["seed"]),
            adaptive_pool=1,
            seed_pool=int(first["seed_pool"]),
            latent_deflate=True,
            latent_coder="best",
            rate=str(first.get("rate", "payload")),
            mip_levels=int(first["mip_levels"]),
            mip_loss="footprint",
            warmup_steps=int(first["warmup"]),
            schedule=str(first["schedule"]),
            precision=precision,
            eval_every=0,
            early_stop=0,
            batch_pixels=int(first.get("batch_pixels", 0)),
            lr_batch_ref=int(first.get("lr_batch_ref", 0)),
            log_every=10**9,
            device="auto",
            with_ssim=False,
            fast=True,
            halve_steps=0,
            compile_step="auto",
            cuda_graph="auto",
        )
        packed = encode_bank(
            [image for _job, image, _sems in items],
            list(sems),
            cfg,
            seeds_per_image=[int(job["seed"]) for job, _image, _sems in items],
        )
        for (job, _image, _sems), (blob, stats) in zip(items, packed, strict=True):
            (out / f"{job['name']}.ntc").write_bytes(blob)
            sps = stats.get("steps_per_s")
            row = {
                "name": job["name"],
                "done": True,
                "fast": True,
                "packed_jobs": stats.get("packed_jobs"),
                "wall_s": stats.get("seconds_train"),
                "psnr_u8": stats["metrics"]["psnr_u8"],
                "total_bpp": stats["total_bpp"],
                "latent_bpp_mip0": stats.get("latent_bpp_mip0"),
                "file_bpp_chain": stats.get("file_bpp_chain"),
                "bppc": stats.get("bppc"),
                "latent_stored_bytes": stats.get("latent_stored_bytes"),
                "latent_plain_bytes": stats.get("latent_plain_bytes"),
                "flags_coder": stats.get("latent_coder"),
                "checkpoint": stats.get("checkpoint"),
                "steps_ran": stats.get("steps_ran"),
                "psnr_u8_init": stats.get("psnr_u8_init"),
                "psnr_u8_trained": stats.get("psnr_u8_trained"),
                "seed": stats.get("seed"),
                "grids": stats.get("grids"),
                "stopped_early": False,
                "precision": precision,
                "steps": job["steps"],
                "size": job["size"],
                "seconds_train": stats.get("seconds_train"),
                "steps_per_s": sps,
                "compiled": stats.get("compiled"),
                "cuda_graph": stats.get("cuda_graph"),
                "shard": shard_label,
                "worker_pid": os.getpid(),
                "batch_pixels": int(job.get("batch_pixels", 0)),
                "lr_batch_ref": int(job.get("lr_batch_ref", 0)),
            }
            (out / f"{job['name']}.json").write_text(json.dumps(row, indent=2))
            sps_txt = f"{sps:.2f}" if isinstance(sps, (int, float)) else "na"
            print(
                f"worker {shard_label or '0/1'} pid {os.getpid()} {job['name']} "
                f"steps/s {sps_txt} packed {stats.get('packed_jobs')} "
                f"psnr {row['psnr_u8']:.4f} total_bpp {row['total_bpp']:.4f}",
                flush=True,
            )


def _run_group(group: list[dict], out: Path, precision: str, shard_label: str, stale_s: float) -> None:
    live = []
    for job in group:
        dest = out / f"{job['name']}.json"
        if dest.is_file() and json.loads(dest.read_text()).get("done"):
            print("skip", job["name"], flush=True)
            continue
        live.append(job)
    if len(live) <= 1:
        for job in live:
            _run(job, out, precision, shard_label, stale_s)
        return
    claimed: list[dict] = []
    try:
        for job in live:
            if claim_job(out, job["name"], stale_s):
                claimed.append(job)
            else:
                print("locked", job["name"], flush=True)
        if len(claimed) <= 1:
            for job in claimed:
                release_job(out, job["name"])
            held = claimed
            claimed = []
            for job in held:
                _run(job, out, precision, shard_label, stale_s)
            return
        _train_claimed_pack(claimed, out, precision, shard_label)
    finally:
        for job in claimed:
            release_job(out, job["name"])


def _run_packed(jobs: list[dict], out: Path, precision: str, shard_label: str, stale_s: float) -> None:
    groups: dict[tuple, list[dict]] = {}
    order: list[tuple | None] = []
    for job in jobs:
        key = _pack_key(job)
        if key is None:
            order.append(None)
            groups.setdefault(None, []).append(job)
        else:
            if key not in groups:
                order.append(key)
            groups.setdefault(key, []).append(job)
    seen = set()
    for job in jobs:
        key = _pack_key(job)
        token = ("single", job["name"]) if key is None else ("group", key)
        if token in seen:
            continue
        seen.add(token)
        if key is None:
            _run(job, out, precision, shard_label, stale_s)
        else:
            _run_group(groups[key], out, precision, shard_label, stale_s)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("results/train_max_v2"))
    p.add_argument("--materials", type=Path, default=Path("data/materials"))
    p.add_argument("--probe-steps", type=int, default=400)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--shard", default="", help="I/N. Split by pixels*steps, longest job first. Does not steal.")
    p.add_argument("--workers", type=int, default=0, help="Spawn N processes on one shared lock queue and wait.")
    p.add_argument("--queue", action="store_true", help="Claim the next undone job from the lock pool until none remain.")
    p.add_argument("--batch-pixels", type=int, default=None, help="Override the pixel batch on 256 jobs. LR scales by batch/(256*256).")
    p.add_argument("--pack", action=argparse.BooleanOptionalAction, default=True, help="Stack jobs that share grid shape, size, and step count.")
    p.add_argument("--stale-lock-s", type=float, default=86400.0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.workers and args.shard:
        raise SystemExit("pass --workers or --shard, not both")
    if args.workers and args.queue:
        raise SystemExit("pass --workers alone; it starts the queue")
    if args.queue and args.shard:
        raise SystemExit("pass --queue or --shard, not both")
    if args.workers:
        if args.workers < 1:
            raise SystemExit("--workers must be >= 1")
        if args.dry_run:
            jobs = build_jobs(args.materials, args.batch_pixels)
            for job in sorted(jobs, key=job_cost, reverse=True):
                print(
                    f"plan queue cost {job_cost(job):.0f} {job['name']} batch {job['batch_pixels']} lr_ref {job['lr_batch_ref']}",
                    flush=True,
                )
            print(f"would spawn {args.workers} queue workers", flush=True)
            return
        # Download before the probe so a missing 2K set stops the parent.
        ensure_sdk_materials(args.materials)
        device = pick_device("auto")
        (args.out / "device.json").write_text(json.dumps(device_report(device), indent=2))
        _ensure_precision(args)
        raise SystemExit(_spawn_workers(args))
    shard_i, shard_n = (None, None)
    shard_label = "queue" if args.queue else ""
    if args.shard:
        shard_i, shard_n = parse_shard(args.shard)
        shard_label = args.shard
    if not args.dry_run:
        ensure_sdk_materials(args.materials)
    device = pick_device("auto")
    info = device_report(device)
    (args.out / "device.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({"shard": shard_label or "0/1", "pid": os.getpid(), **info}), flush=True)
    jobs = build_jobs(args.materials, args.batch_pixels)
    if args.dry_run:
        if args.queue:
            for job in sorted(jobs, key=job_cost, reverse=True):
                print(
                    f"plan queue cost {job_cost(job):.0f} {job['name']} batch {job['batch_pixels']} lr_ref {job['lr_batch_ref']}",
                    flush=True,
                )
            return
        assignment = assign_shards(jobs, shard_n) if shard_n else [0] * len(jobs)
        width = shard_n or 1
        for index, job in enumerate(jobs):
            if shard_n is None or assignment[index] == shard_i:
                print(
                    f"plan {index} shard {assignment[index]}/{width} cost {job_cost(job):.0f} "
                    f"{job['name']} batch {job['batch_pixels']} lr_ref {job['lr_batch_ref']}",
                    flush=True,
                )
        return
    precision = _ensure_precision(args)
    if args.queue:
        _queue_loop(jobs, args.out, precision, args.stale_lock_s, args.pack)
        print("wrote", args.out, "queue", flush=True)
        return
    if shard_n:
        assignment = assign_shards(jobs, shard_n)
        selected = [job for index, job in enumerate(jobs) if assignment[index] == shard_i]
    else:
        selected = list(jobs)
    if args.pack:
        _run_packed(selected, args.out, precision, shard_label, args.stale_lock_s)
    else:
        for job in selected:
            _run(job, args.out, precision, shard_label, args.stale_lock_s)
    print("wrote", args.out, "shard", shard_label or "0/1", flush=True)


if __name__ == "__main__":
    main()
