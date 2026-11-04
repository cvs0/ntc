"""Shared job queue for two machines.

Layout under ``--out`` (a directory both machines can see):

- ``pending/<id>.json`` jobs waiting for a worker
- ``running/<worker>/<id>.json`` claimed jobs
- ``done/<id>.json`` finished rows
- ``failed/<id>.json`` errors

Claiming is an atomic rename. A worker that dies leaves a file in ``running``.
``--steal-after-s`` moves those back to ``pending`` when their mtime is older
than that many seconds. ``train_max`` is not read or written.

Windows, from the repo root, with a directory both machines can write:

    python bench/train_queue.py enqueue --out results/train_queue --demo
    python bench/train_queue.py work --worker amd-desktop --out //share/train_queue --once
    python bench/train_queue.py work --worker nvidia-laptop --out //share/train_queue --once
    python bench/train_queue.py merge --out //share/train_queue
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path


def _dirs(root: Path) -> dict[str, Path]:
    paths = {
        "pending": root / "pending",
        "running": root / "running",
        "done": root / "done",
        "failed": root / "failed",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def enqueue_demo(root: Path) -> None:
    paths = _dirs(root)
    job = {
        "id": "demo-synthetic",
        "material": "synthetic",
        "size": 16,
        "steps": 2,
        "hidden": [8],
        "grids": [[8, 8, 2, 4]],
        "seed": 0,
        "fast": True,
    }
    dest = paths["pending"] / f"{job['id']}.json"
    if not dest.is_file() and not (paths["done"] / f"{job['id']}.json").is_file():
        dest.write_text(json.dumps(job, indent=2))
    print("pending", dest)


def _claim(root: Path, worker: str) -> Path | None:
    paths = _dirs(root)
    worker_dir = paths["running"] / worker
    worker_dir.mkdir(parents=True, exist_ok=True)
    for job in sorted(paths["pending"].glob("*.json")):
        dest = worker_dir / job.name
        try:
            os.replace(job, dest)
        except OSError:
            continue
        return dest
    return None


def _steal(root: Path, older_than_s: float) -> int:
    if older_than_s <= 0:
        return 0
    paths = _dirs(root)
    now = time.time()
    moved = 0
    for job in paths["running"].glob("*/*.json"):
        if now - job.stat().st_mtime < older_than_s:
            continue
        dest = paths["pending"] / job.name
        if dest.exists():
            continue
        os.replace(job, dest)
        moved += 1
    return moved


def _run_job(job: dict) -> dict:
    import numpy as np

    from ntc.encode import TrainConfig
    from ntc.train_fast import encode_arrays_fast

    if job.get("material") == "synthetic":
        size = int(job.get("size", 16))
        image = np.linspace(0, 1, 3 * size * size, dtype=np.float32).reshape(3, size, size)
        sems = [0, 1, 2]
    else:
        from ntc.materials import load_material

        image, sems, _info = load_material(job["material"], int(job["size"]))
    grids = None if job.get("grids") is None else [tuple(g) for g in job["grids"]]
    cfg = TrainConfig(
        steps=int(job.get("steps", 4)),
        hidden=tuple(job.get("hidden", [8])),
        grids=grids,
        seed=int(job.get("seed", 0)),
        seed_pool=int(job.get("seed_pool", 1)),
        adaptive_pool=1,
        with_ssim=False,
        log_every=10**9,
        fast=bool(job.get("fast", True)),
        compile_step="off",
        cuda_graph="off",
        mip_levels=int(job.get("mip_levels", 2)),
        halve_steps=int(job.get("halve_steps", 0)),
        stop_psnr=float(job.get("stop_psnr", 0.0)),
        check_every=int(job.get("check_every", 0)),
        device=str(job.get("device", "auto")),
    )
    _blob, stats = encode_arrays_fast(image, sems, cfg)
    return {
        "id": job["id"],
        "psnr_u8": stats["metrics"]["psnr_u8"],
        "seconds_train": stats.get("seconds_train"),
        "steps_ran": stats.get("steps_ran"),
        "total_bpp": stats.get("total_bpp"),
        "hit_target": stats.get("hit_target"),
    }


def work(root: Path, worker: str, once: bool, steal_after_s: float) -> None:
    paths = _dirs(root)
    while True:
        _steal(root, steal_after_s)
        claimed = _claim(root, worker)
        if claimed is None:
            if once:
                print("idle", worker)
                return
            time.sleep(2.0)
            continue
        job = json.loads(claimed.read_text())
        try:
            row = _run_job(job)
            row["worker"] = worker
            row["host"] = socket.gethostname()
            row["done"] = True
            (paths["done"] / claimed.name).write_text(json.dumps(row, indent=2))
            claimed.unlink(missing_ok=True)
            print("done", job["id"], "psnr", row["psnr_u8"])
        except Exception as exc:
            fail = {"id": job.get("id"), "error": str(exc), "worker": worker, "done": False}
            (paths["failed"] / claimed.name).write_text(json.dumps(fail, indent=2))
            claimed.unlink(missing_ok=True)
            print("failed", job.get("id"), exc)
        if once:
            return


def merge(root: Path) -> None:
    paths = _dirs(root)
    rows = []
    for path in sorted(paths["done"].glob("*.json")):
        rows.append(json.loads(path.read_text()))
    failed = [json.loads(path.read_text()) for path in sorted(paths["failed"].glob("*.json"))]
    report = {"done": rows, "failed": failed, "pending": [p.name for p in paths["pending"].glob("*.json")]}
    dest = root / "merged.json"
    dest.write_text(json.dumps(report, indent=2))
    print("wrote", dest, "done", len(rows), "failed", len(failed))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("enqueue", "work", "merge"))
    p.add_argument("--out", type=Path, default=Path("results/train_queue"))
    p.add_argument("--worker", default=socket.gethostname())
    p.add_argument("--once", action="store_true")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--steal-after-s", type=float, default=0.0)
    args = p.parse_args()
    if args.command == "enqueue":
        if not args.demo:
            raise SystemExit("enqueue currently ships the synthetic demo job. Pass --demo.")
        enqueue_demo(args.out)
    elif args.command == "work":
        work(args.out, args.worker, args.once, args.steal_after_s)
    else:
        merge(args.out)


if __name__ == "__main__":
    main()
