"""Timed steps for one dense 2K job. Does not touch results/train_max_v2.

The default job is sdk_MetalPlates013_2K_b10: grid
[(2048, 2048, 4, 4), (1024, 1024, 4, 4)], flat batch 1048576, the shape that
reached 33.07 dB at 4.855 bpp. Each mode is warmed up, then timed. A failure
in one mode is recorded and the next mode runs.

Modes: eager, torch.compile default, torch.compile reduce-overhead, a HIP
graph (torch.cuda.CUDAGraph on ROCm), and the Triton bilinear gather.
Eager mode also records the top 10 profiler ops.

```powershell
$env:PYTHONPATH = "src;."
.venv-rocm/Scripts/python.exe bench/train_2k_speed.py --out results/train_speed/train_2k_speed.json
```

```bash
source /root/venv-ntc/bin/activate
export HSA_ENABLE_DXG_DETECTION=1
export PYTHONPATH=src:.
python bench/train_2k_speed.py --out results/train_speed/train_2k_speed.json
```
"""

from __future__ import annotations

import os

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ntc.device import device_report, pick_device
from ntc.encode import TrainConfig
from ntc.train_fast import (
    StackedTrain,
    _adam_on,
    _build_nets,
    _prep_mips,
    _run_loop,
)

# Packed size matches latent target 20, which is label 10 / 0.5.
B10_GRID = [(2048, 2048, 4, 4), (1024, 1024, 4, 4)]
DENSE_BATCH = 1024 * 1024
MODES = (
    "eager",
    "compile_default",
    "compile_reduce_overhead",
    "hip_graph",
    "triton_gather",
)


def _forbid_v2(path: Path) -> None:
    if "train_max_v2" in path.parts:
        raise SystemExit("refusing to write into results/train_max_v2")


def _load_image(materials: Path, name: str, size: int) -> tuple[np.ndarray, list[int], str]:
    folder = materials / "MetalPlates013_2K"
    if size == 2048 and (folder / "manifest.json").is_file():
        from ntc.materials import load_material

        images, sems, _info = load_material(folder, size)
        return images, sems, str(folder)
    rng = np.random.default_rng(0)
    channels = 9
    image = rng.random((channels, size, size), dtype=np.float32)
    return image, [1] * channels, "synthetic"


def _grid_for(size: int) -> list[tuple[int, int, int, int]]:
    if size == 2048:
        return list(B10_GRID)
    half = max(4, size // 2)
    return [(size, size, 4, 4), (half, half, 4, 4)]


def _batch_for(size: int) -> int:
    if size == 2048:
        return DENSE_BATCH
    return max(1, (size * size) // 4)


def _cfg(size: int, precision: str, steps: int) -> TrainConfig:
    return TrainConfig(
        steps=steps,
        hidden=(32,),
        grids=_grid_for(size),
        seed=0,
        seed_pool=1,
        adaptive_pool=1,
        with_ssim=False,
        mip_levels=min(12, int(np.log2(size)) + 1),
        mip_loss="footprint",
        warmup_steps=0,
        schedule="constant",
        precision=precision,
        eval_every=0,
        early_stop=0,
        fast=True,
        device="auto",
        pca_init=True,
        compile_step="off",
        cuda_graph="off",
        log_every=10**9,
        halve_steps=0,
        batch_pixels=_batch_for(size),
        batch_mip="flat",
        qat_weights=True,
        qat_latents=True,
    )


def _set_triton(enabled: bool) -> None:
    import ntc.triton_bilinear as triton_bilinear

    triton_bilinear._DISABLED = not enabled
    triton_bilinear._CHECKED = False


def _prepare(cfg: TrainConfig, images: np.ndarray, device: torch.device):
    target = torch.from_numpy(np.ascontiguousarray(images)).to(device)
    nets = _build_nets(target, cfg, list(cfg.grids), [cfg.seed], device)
    stacked = StackedTrain(nets)
    mips = _prep_mips(target, cfg, device)
    init = {key: value.detach().cpu().clone() for key, value in stacked.state_dict().items()}
    return stacked, mips, init


def _reset(stacked: StackedTrain, init: dict, cfg: TrainConfig, device: torch.device, capturable: bool = False):
    stacked.load_state_dict({key: value.to(device) for key, value in init.items()})
    return _adam_on(stacked, cfg, capturable=capturable)


def _amp(cfg: TrainConfig, device: torch.device):
    if cfg.precision == "bf16" and device.type == "cuda":
        return torch.bfloat16
    if cfg.precision == "fp16" and device.type == "cuda":
        return torch.float16
    return None


def _mode_call(name: str, device: torch.device) -> dict:
    """Flags for _run_loop. cpu-only modes that need cuda return an error."""
    if name == "eager":
        return {"compile_step": "off", "cuda_graph": "off", "use_graph": False, "compile_mode": None, "triton": False}
    if name == "compile_default":
        if device.type != "cuda":
            return {"error": "torch.compile needs a cuda device"}
        return {"compile_step": "on", "cuda_graph": "off", "use_graph": False, "compile_mode": "default", "triton": False}
    if name == "compile_reduce_overhead":
        if device.type != "cuda":
            return {"error": "torch.compile needs a cuda device"}
        return {
            "compile_step": "on",
            "cuda_graph": "off",
            "use_graph": False,
            "compile_mode": "reduce-overhead",
            "triton": False,
        }
    if name == "hip_graph":
        if device.type != "cuda":
            return {"error": "HIP graph needs a cuda device"}
        return {"compile_step": "off", "cuda_graph": "on", "use_graph": True, "compile_mode": None, "triton": False}
    if name == "triton_gather":
        if device.type != "cuda":
            return {"error": "Triton gather runs on cuda"}
        return {"compile_step": "off", "cuda_graph": "off", "use_graph": False, "compile_mode": None, "triton": True}
    return {"error": f"unknown mode {name}"}


def _time_mode(
    name: str,
    stacked: StackedTrain,
    mips: list,
    init: dict,
    cfg: TrainConfig,
    sems: list[int],
    device: torch.device,
    warmup: int,
    timed: int,
) -> dict:
    spec = _mode_call(name, device)
    if spec.get("error"):
        return {"name": name, "error": spec["error"], "steps_per_s": None}
    try:
        _set_triton(bool(spec["triton"]))
        opt = _reset(stacked, init, cfg, device, capturable=bool(spec["use_graph"]))
        run_cfg = TrainConfig(
            **{
                **cfg.__dict__,
                "compile_step": spec["compile_step"],
                "cuda_graph": spec["cuda_graph"],
                "schedule": "constant",
            }
        )
        info = _run_loop(
            stacked,
            opt,
            mips,
            run_cfg,
            sems,
            timed,
            0,
            _amp(run_cfg, device),
            device.type,
            use_graph=bool(spec["use_graph"]),
            warmup=warmup,
            compile_mode_override=spec["compile_mode"],
        )
    except Exception as exc:
        return {"name": name, "error": str(exc), "steps_per_s": None}
    row = {
        "name": name,
        "seconds": info.get("seconds"),
        "steps": info.get("steps"),
        "steps_per_s": info.get("steps_per_s"),
        "compiled": info.get("compiled"),
        "cuda_graph": info.get("cuda_graph"),
        "error": None,
    }
    if name.startswith("compile") and not info.get("compiled"):
        row["error"] = info.get("compile_error") or "compile failed"
        row["steps_per_s"] = None
    elif name == "hip_graph" and not info.get("cuda_graph"):
        row["error"] = info.get("cuda_graph_error") or "graph was not captured"
        row["steps_per_s"] = None
    elif name == "triton_gather":
        import ntc.triton_bilinear as triton_bilinear

        if triton_bilinear._DISABLED:
            row["error"] = "triton gather disabled itself after the numeric check or an import failure"
            row["steps_per_s"] = None
    return row


def _profile_eager(
    stacked: StackedTrain,
    mips: list,
    init: dict,
    cfg: TrainConfig,
    sems: list[int],
    device: torch.device,
    steps: int,
) -> list[dict]:
    from torch.profiler import ProfilerActivity, profile

    _set_triton(False)
    opt = _reset(stacked, init, cfg, device)
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    run_cfg = TrainConfig(**{**cfg.__dict__, "compile_step": "off", "cuda_graph": "off"})
    for _ in range(2):
        _run_loop(
            stacked, opt, mips, run_cfg, sems, 1, 0, _amp(run_cfg, device), device.type, use_graph=False,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    with profile(activities=activities, record_shapes=False) as prof:
        _run_loop(
            stacked,
            opt,
            mips,
            run_cfg,
            sems,
            max(1, steps),
            0,
            _amp(run_cfg, device),
            device.type,
            use_graph=False,
        )
    rows = []
    for event in prof.key_averages():
        device_us = float(
            getattr(event, "self_device_time_total", 0) or getattr(event, "self_cuda_time_total", 0) or 0
        )
        rows.append(
            {
                "name": event.key or "",
                "self_cpu_us": float(event.self_cpu_time_total),
                "self_device_us": device_us,
                "count": int(event.count),
            }
        )
    rows.sort(key=lambda row: -(row["self_device_us"] or row["self_cpu_us"]))
    return rows[:10]


def _child_env() -> dict:
    env = os.environ.copy()
    root = Path(__file__).resolve().parents[1]
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([str(root), str(root / "src"), prev])
    env.setdefault("TRITON_DEFAULT_FP_FUSION", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def _run_isolated(args: argparse.Namespace) -> dict:
    """One subprocess per mode so a failed CUDA graph does not poison the next."""
    out = Path(args.out)
    _forbid_v2(out)
    modes = [name.strip() for name in args.modes.split(",") if name.strip()]
    script = str(Path(__file__).resolve())
    results = []
    for name in modes:
        fragment = out.with_name(out.stem + f".{name}.partial.json")
        cmd = [
            sys.executable,
            script,
            "--worker-mode",
            name,
            "--job",
            args.job,
            "--materials",
            str(args.materials),
            "--size",
            str(args.size),
            "--warmup",
            str(args.warmup),
            "--timed-steps",
            str(args.timed_steps),
            "--profile-steps",
            "0",
            "--precision",
            args.precision,
            "--modes",
            name,
            "--out",
            str(fragment),
        ]
        proc = subprocess.run(cmd, env=_child_env(), capture_output=True, text=True)
        if proc.stdout:
            print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n", flush=True)
        if proc.returncode != 0 or not fragment.is_file():
            err = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()
            err = err[-800:]
            print(f"{name:24} error {err}", flush=True)
            results.append({"name": name, "error": err, "steps_per_s": None, "wall_s": None})
        else:
            results.append(json.loads(fragment.read_text()))
            fragment.unlink()
    kernels = []
    eager_ok = any(row["name"] == "eager" and not row.get("error") for row in results)
    if eager_ok and args.profile_steps > 0:
        try:
            device = pick_device("auto")
            precision = args.precision
            if precision == "auto":
                precision = "bf16" if device.type == "cuda" else "fp32"
            images, sems, _source = _load_image(Path(args.materials), args.job, args.size)
            cfg = _cfg(args.size, precision if device.type == "cuda" else "fp32", args.profile_steps)
            stacked, mips, init = _prepare(cfg, images, device)
            kernels = _profile_eager(stacked, mips, init, cfg, sems, device, args.profile_steps)
        except Exception as exc:
            kernels = [{"name": "profile_failed", "error": str(exc)}]
        for row in kernels:
            us = row.get("self_device_us") or row.get("self_cpu_us") or 0
            print(f"kernel {us:12.1f} us  {row.get('name')}", flush=True)
    device = pick_device("auto")
    info = device_report(device)
    precision = args.precision
    if precision == "auto":
        precision = "bf16" if device.type == "cuda" else "fp32"
    _images, _sems, source = _load_image(Path(args.materials), args.job, args.size)
    cfg = _cfg(args.size, precision if device.type == "cuda" else "fp32", args.warmup + args.timed_steps)
    report = {
        "platform": platform.platform(),
        "system": sys.platform,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "device": info,
        "job": args.job,
        "image_source": source,
        "size": args.size,
        "grid": [list(g) for g in cfg.grids],
        "batch_pixels": int(cfg.batch_pixels),
        "batch_mip": cfg.batch_mip,
        "precision": cfg.precision,
        "warmup": args.warmup,
        "timed_steps": args.timed_steps,
        "modes": results,
        "eager_top_kernels": kernels,
        "workers_note": (
            "One dense 2K job fills this GPU. Three processes made each job about 3x slower. "
            "Packing three tiny CPU steps raised throughput, and that does not carry to a 2.35 steps/s 2K step. "
            "On an RTX 5070 Laptop, compile reduce-overhead was 19.99 steps/s, compile default 19.57, eager 3.94."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("wrote", out, flush=True)
    return report


def run(args: argparse.Namespace) -> dict:
    if not args.worker_mode:
        return _run_isolated(args)
    out = Path(args.out)
    _forbid_v2(out)
    device = pick_device("auto")
    info = device_report(device)
    precision = args.precision
    if precision == "auto":
        precision = "bf16" if device.type == "cuda" else "fp32"
    images, sems, source = _load_image(Path(args.materials), args.job, args.size)
    cfg = _cfg(args.size, precision if device.type == "cuda" else "fp32", args.warmup + args.timed_steps)
    stacked, mips, init = _prepare(cfg, images, device)
    name = args.worker_mode
    t0 = time.perf_counter()
    row = _time_mode(name, stacked, mips, init, cfg, sems, device, args.warmup, args.timed_steps)
    row["wall_s"] = time.perf_counter() - t0
    err = row.get("error")
    if err:
        print(f"{name:24} error {err}", flush=True)
    else:
        print(f"{name:24} steps/s {row.get('steps_per_s'):.2f}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=2))
    return {"modes": [row], "device": info, "image_source": source}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--job", default="sdk_MetalPlates013_2K_b10")
    p.add_argument("--materials", type=Path, default=Path("data/materials"))
    p.add_argument("--size", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--timed-steps", type=int, default=200)
    p.add_argument("--profile-steps", type=int, default=4)
    p.add_argument("--precision", default="auto", choices=("auto", "fp32", "bf16", "fp16"))
    p.add_argument("--modes", default=",".join(MODES))
    p.add_argument("--worker-mode", default="")
    p.add_argument("--out", type=Path, default=Path("results/train_speed/train_2k_speed.json"))
    args = p.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
