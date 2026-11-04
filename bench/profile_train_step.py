"""Profile one full training step: sample, forward, backward, optimizer.

Writes JSON to stdout. CPU, MPS, and CUDA all run. Compile and CUDA graphs
follow the same switches as training and stay off where the device cannot
use them. No network and no remote GPU.

    python bench/profile_train_step.py
    python bench/profile_train_step.py --device mps --steps 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_ROOT), str(_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from ntc.encode import TrainConfig  # noqa: E402
from ntc.train_fast import (  # noqa: E402
    StackedTrain,
    _adam_on,
    _build_nets,
    _prep_mips,
    _run_loop,
    overhaul_plan,
    set_fused_gather,
)


def _bucket(name: str) -> str:
    text = name.lower()
    if "optimizer" in text or "adam" in text:
        return "optimizer"
    if "backward" in text:
        return "backward"
    if any(part in text for part in ("randint", "index_select", "index_add", "crop", "sample")):
        return "sampling"
    if any(part in text for part in ("bmm", "addmm", "mm", "linear", "bilinear", "matmul", "convolution")):
        return "forward"
    return "other"


def _events(prof) -> list[dict]:
    rows = []
    for event in prof.key_averages():
        rows.append(
            {
                "name": event.key,
                "self_cpu_us": float(event.self_cpu_time_total),
                "cpu_us": float(getattr(event, "cpu_time_total", event.self_cpu_time_total)),
                "calls": int(event.count),
                "bucket": _bucket(event.key),
            }
        )
    rows.sort(key=lambda row: -row["self_cpu_us"])
    return rows


def _sum_buckets(rows: list[dict], steps: int) -> dict:
    totals = {"sampling": 0.0, "forward": 0.0, "backward": 0.0, "optimizer": 0.0, "other": 0.0}
    marked = 0.0
    for row in rows:
        totals[row["bucket"]] += row["self_cpu_us"]
        if row["name"] == "ntc_train_step":
            marked += row["cpu_us"]
    scale = 1e-3 / max(int(steps), 1)
    out = {key: value * scale for key, value in totals.items()}
    out["ntc_train_step_ms"] = marked * scale
    return out


def _cfg(device: str, steps: int, **flags) -> TrainConfig:
    base = dict(
        steps=steps,
        hidden=(8,),
        grids=[(16, 16, 4, 4)],
        fast=True,
        device=device,
        batch_pixels=512,
        seed_pool=1,
        pca_init=False,
        log_every=10**9,
        eval_every=0,
        early_stop=0,
        mip_levels=2,
        halve_steps=0,
        compile_step="off",
        cuda_graph="off",
        fused_gather="off",
        fused_mlp="off",
        device_pipeline=False,
        schedule="cosine",
        warmup_steps=0,
        lr_scale_rule="off",
        auto_batch="off",
        target_bpp=4.0,
        with_ssim=False,
    )
    base.update(flags)
    return TrainConfig(**base)


def profile_case(name: str, device: str, steps: int, warmup: int, **flags) -> dict:
    import ntc.train_fast as train_fast
    from ntc.device import pick_device

    cfg = _cfg(device, steps, **flags)
    dev = pick_device(cfg.device)
    rng = np.random.default_rng(0)
    images = rng.random((3, 32, 32), dtype=np.float32)
    target = torch.from_numpy(np.ascontiguousarray(images)).to(dev)
    set_fused_gather(cfg.fused_gather)
    net = _build_nets(target, cfg, list(cfg.grids), [0], dev)[0]
    stacked = StackedTrain([net])
    mips = _prep_mips(target, cfg, dev)
    opt = _adam_on(stacked, cfg, capturable=False)
    if warmup:
        _run_loop(
            stacked, opt, mips, cfg, [0, 1, 2], warmup, 0, None, dev.type, use_graph=False,
        )
    activities = [ProfilerActivity.CPU]
    if dev.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    train_fast._STEP_PROFILE = True
    started = time.perf_counter()
    try:
        with profile(activities=activities, record_shapes=False) as prof:
            info = _run_loop(
                stacked, opt, mips, cfg, [0, 1, 2], steps, warmup, None, dev.type, use_graph=False,
            )
    finally:
        train_fast._STEP_PROFILE = False
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    rows = _events(prof)
    plan = overhaul_plan(dev.type, cfg)
    steady = info.get("steps_per_s")
    return {
        "name": name,
        "device": dev.type,
        "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else dev.type,
        "steps": steps,
        "warmup": warmup,
        "wall_s": elapsed,
        "steady_steps_per_s": steady,
        "steady_step_ms": (1000.0 / steady) if steady else None,
        "profiler_step_ms": _sum_buckets(rows, steps),
        "plan": plan,
        "compiled": bool(info.get("compiled")),
        "compile_mode": info.get("compile_mode"),
        "cuda_graph": bool(info.get("cuda_graph")),
        "cuda_graph_error": info.get("cuda_graph_error"),
        "top": rows[:12],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    before = profile_case("before", args.device, args.steps, args.warmup)
    after = profile_case(
        "after",
        args.device,
        args.steps,
        args.warmup,
        compile_step="auto",
        cuda_graph="auto",
        fused_gather="auto",
        fused_mlp="auto",
        device_pipeline=True,
        schedule="warmup_cosine",
        warmup_steps=1,
        lr_scale_rule="sqrt",
        lr_batch_ref=128,
        auto_batch="off",
    )
    before_rate = before.get("steady_steps_per_s") or 0.0
    after_rate = after.get("steady_steps_per_s") or 0.0
    ratio = (after_rate / before_rate) if before_rate else None
    report = {"before": before, "after": after, "steady_ratio": ratio}
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
