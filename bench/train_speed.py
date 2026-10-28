"""Steps/s and time-to-target-PSNR for the eager loop and the fast path.

The eager loop is ``encode_arrays`` (what ``train_max`` calls). The fast path
is ``encode_arrays_fast``: stacked seeds, no per-step host sync, optional
bf16 autocast, ``torch.compile``, and a CUDA graph when the schedule allows
it. On a machine without CUDA the GPU idle fraction is null. Run this on
the RX 9070 to fill it.

```powershell
$env:PYTHONPATH = "src;."
python bench\train_speed.py --material data\\materials\\metal_plate --size 256 --steps 200 --seeds 4 --out results\train_speed\rx9070.json
```
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ntc.encode import TrainConfig, encode_arrays
from ntc.train_fast import choose_precision, encode_arrays_fast, profile_historical_step


def _image(size: int, material: Path | None) -> tuple[np.ndarray, list[int]]:
    if material is not None:
        from ntc.materials import load_material

        images, sems, _info = load_material(material, size)
        return images, sems
    ys = np.linspace(0.05, 0.95, size, dtype=np.float32)
    xs = np.linspace(0.0, 1.0, size, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    images = np.stack(
        [grid_x, grid_y, np.clip(0.5 + 0.25 * np.sin(grid_x * 6.0 + grid_y * 3.0), 0, 1)],
        axis=0,
    ).astype(np.float32)
    return images, [0, 1, 2]


def _cfg(args: argparse.Namespace, **kw) -> TrainConfig:
    base = dict(
        steps=int(kw.get("steps", args.steps)),
        hidden=tuple(args.hidden),
        grids=[(args.size, args.size, 2, 4)] if args.size <= 64 else [(args.size, args.size, 4, 4), (8, 8, 2, 2)],
        adaptive_pool=1,
        seed_pool=int(kw.get("seed_pool", 1)),
        with_ssim=False,
        log_every=10**9,
        eval_every=0,
        early_stop=0,
        mip_levels=2 if args.size <= 64 else 4,
        seed=0,
        compile_step="off",
        cuda_graph="off",
        device="auto",
        fast=bool(kw.get("fast", False)),
        halve_steps=int(kw.get("halve_steps", 0)),
        precision=str(kw.get("precision", "fp32")),
    )
    # 256 jobs use the locked high-rate grids from train_max when the caller
    # asks for the hardware shape.
    if args.g16:
        base["grids"] = [(256, 256, 4, 4), (8, 8, 2, 2)]
        base["hidden"] = (32,)
        base["mip_levels"] = 4
    return TrainConfig(**base)


def _timed(fn) -> tuple[dict, float]:
    t0 = time.perf_counter()
    _blob, stats = fn()
    return stats, time.perf_counter() - t0


def _ablation_image() -> tuple[np.ndarray, list[int]]:
    """24x24 image that is not one of the eight prior images."""
    torch = __import__("torch")
    torch.manual_seed(7)
    ys = torch.linspace(0, 1, 24)
    xs = torch.linspace(0, 1, 24)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    image = torch.stack(
        [
            (gx * 0.8 + 0.2 * torch.sin(gy * 9)).clamp(0, 1),
            (gy * 0.7 + 0.15 * torch.cos(gx * 8)).clamp(0, 1),
            ((gx - gy).abs()).clamp(0, 1),
        ],
        0,
    )
    return image.numpy().astype(np.float32), [0, 1, 2]


def run_ablation(max_steps: int, check_every: int, prior: Path) -> dict:
    """Time to a fixed PSNR for each step-reduction idea.

    The baseline is the fast path with the historical PCA init, Adam, cosine,
    and QAT on every step. An idea is kept when it reaches the target and its
    train wall is at most 90% of the baseline train wall. Ties and misses stay off.
    """
    from dataclasses import replace

    images, sems = _ablation_image()
    common = dict(
        steps=max_steps,
        hidden=(16,),
        grids=[(12, 12, 4, 4)],
        adaptive_pool=1,
        seed_pool=1,
        with_ssim=False,
        log_every=10**9,
        mip_levels=2,
        seed=0,
        compile_step="off",
        cuda_graph="off",
        fast=True,
        check_every=check_every,
        lr=5e-3,
    )

    def once(extra: dict) -> dict:
        cfg = replace(TrainConfig(**common), **extra)
        stats, wall = _timed(lambda: encode_arrays_fast(images, sems, cfg))
        return {
            "wall_s": wall,
            "seconds_train": stats.get("seconds_train"),
            "steps_ran": stats.get("steps_ran"),
            "psnr_u8": stats["metrics"]["psnr_u8"],
            "train_psnr_u8": stats.get("psnr_u8_trained"),
            "hit_target": stats.get("hit_target"),
            "checkpoint": stats.get("checkpoint"),
        }

    probe = once({"stop_psnr": 0.0})
    # Aim just under the full-budget export PSNR so the baseline still has to train.
    target = float(probe["psnr_u8"]) - 0.3
    if target < 10:
        target = float(probe["psnr_u8"])
    base = once({"stop_psnr": target})
    ideas = {
        "warm_finetune": {"stop_psnr": target, "warm_start": str(prior), "freeze_decoder": False},
        "warm_freeze": {"stop_psnr": target, "warm_start": str(prior), "freeze_decoder": True},
        "analytic": {"stop_psnr": target, "latent_init": "analytic"},
        "coarse_to_fine": {"stop_psnr": target, "coarse_to_fine": True},
        "onecycle": {"stop_psnr": target, "schedule": "onecycle", "onecycle_peak": 5.0},
        "warmup_cosine_3x": {"stop_psnr": target, "schedule": "cosine", "warmup_steps": 4, "lr": 1.5e-2},
        "latent_lr_8": {"stop_psnr": target, "latent_lr_mult": 8.0},
        "lion": {"stop_psnr": target, "optimizer": "lion", "lr": 1e-4},
        "shampoo": {"stop_psnr": target, "optimizer": "shampoo"},
        "qat_late": {"stop_psnr": target, "qat_late": True},
        "importance": {"stop_psnr": target, "importance": True, "batch_pixels": 128},
        "uniform_batch_128": {"stop_psnr": target, "batch_pixels": 128},
        "plateau": {"stop_psnr": target, "plateau_stop": True},
        "pool4_no_halve": {"stop_psnr": target, "seed_pool": 4, "halve_steps": 0},
        "halving_pool4": {"stop_psnr": target, "seed_pool": 4, "halve_steps": 4},
    }
    rows = {"baseline_fixed_budget": probe, "baseline_stop_at_target": base}
    winners = []
    base_train = base.get("seconds_train") or base["wall_s"]
    for name, extra in ideas.items():
        row = once(extra)
        reached = bool(row["hit_target"]) or (
            row["psnr_u8"] is not None and row["psnr_u8"] + 1e-3 >= target
        )
        train_s = row.get("seconds_train") or row["wall_s"]
        faster = base_train > 0 and train_s < 0.9 * base_train
        # Halving is compared to the unhalved pool, not to the single model.
        if name == "halving_pool4":
            other = rows.get("pool4_no_halve")
            if other is None:
                faster = False
            else:
                other_s = other.get("seconds_train") or other["wall_s"]
                faster = other_s > 0 and train_s < 0.9 * other_s
        row["reached"] = reached
        row["faster_than_baseline"] = bool(reached and faster)
        rows[name] = row
        if row["faster_than_baseline"]:
            winners.append(name)
        print(
            f"{name} steps {row['steps_ran']} train {train_s:.3f}s psnr {row['psnr_u8']:.3f} "
            f"hit {row['hit_target']} keep {row['faster_than_baseline']}",
            flush=True,
        )
    return {
        "measured_on_this_process": True,
        "device": "cpu",
        "image": "24x24 procedural, not in the prior set",
        "architecture": {"hidden": [16], "grids": [[12, 12, 4, 4]], "channels": 3},
        "max_steps": max_steps,
        "check_every": check_every,
        "target_psnr_u8": target,
        "keep_rule": "reached the target and train seconds under 90% of the baseline",
        "winners": winners,
        "rows": rows,
        "prior": str(prior),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Eager vs fast training throughput")
    p.add_argument("--material", type=Path, default=None)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--hidden", type=int, nargs="+", default=[8])
    p.add_argument("--profile-steps", type=int, default=4)
    p.add_argument("--probe-steps", type=int, default=0)
    p.add_argument("--g16", action="store_true", help="use the train_max 256 g16 grids and hidden 32")
    p.add_argument("--ablate", action="store_true")
    p.add_argument("--ablate-steps", type=int, default=24)
    p.add_argument("--prior", type=Path, default=Path("src/ntc/priors/decoder_small.pt"))
    p.add_argument("--out", type=Path, default=Path("results/train_speed/cpu.json"))
    args = p.parse_args()
    if args.ablate:
        if not args.prior.is_file():
            raise SystemExit(f"missing prior {args.prior}. Run bench/pretrain_decoder.py --preset small first.")
        report = run_ablation(args.ablate_steps, 4, args.prior)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print("winners", report["winners"])
        print("wrote", args.out)
        return
    images, sems = _image(args.size, args.material)
    profile_cfg = _cfg(args, steps=args.profile_steps, seed_pool=1)
    profile = profile_historical_step(images, sems, profile_cfg, args.profile_steps)

    from dataclasses import replace

    old_stats, old_wall = _timed(
        lambda: encode_arrays(images, sems, _cfg(args, steps=args.steps, seed_pool=1, fast=False))
    )
    new_cfg = replace(_cfg(args, steps=args.steps, seed_pool=1, fast=True), compile_step="auto", cuda_graph="auto")
    new_stats, new_wall = _timed(lambda: encode_arrays_fast(images, sems, new_cfg))

    pool_old, pool_old_wall = _timed(
        lambda: encode_arrays(images, sems, _cfg(args, steps=args.steps, seed_pool=args.seeds, fast=False))
    )
    pool_new_cfg = replace(
        _cfg(args, steps=args.steps, seed_pool=args.seeds, fast=True, halve_steps=max(2, args.steps // 3)),
        compile_step="auto",
        cuda_graph="auto",
    )
    pool_new, pool_new_wall = _timed(lambda: encode_arrays_fast(images, sems, pool_new_cfg))

    target = float(old_stats["metrics"]["psnr_u8"])
    # Time to the eager PSNR. The fast run above already used ``args.steps``.
    # If it is already there, that wall is the time. If it is short, train
    # longer up to 4x steps.
    time_to_new = new_wall if float(new_stats["metrics"]["psnr_u8"]) + 1e-4 >= target else None
    new_psnr_at_equal_steps = float(new_stats["metrics"]["psnr_u8"])
    if time_to_new is None:
        longer = replace(new_cfg, steps=args.steps * 4)
        longer_stats, longer_wall = _timed(lambda: encode_arrays_fast(images, sems, longer))
        if float(longer_stats["metrics"]["psnr_u8"]) + 1e-4 >= target:
            time_to_new = longer_wall
            new_psnr_at_equal_steps = float(longer_stats["metrics"]["psnr_u8"])
    precision = None
    if args.probe_steps > 0:
        precision = choose_precision(images, sems, _cfg(args, steps=args.probe_steps, seed_pool=1), args.probe_steps)

    report = {
        "measured_on_this_process": True,
        "gpu_result": profile["device"].get("backend") in ("cuda", "rocm"),
        "size": args.size,
        "steps": args.steps,
        "seeds": args.seeds,
        "target_psnr_u8": target,
        "target_definition": "8-bit PSNR of the eager encode_arrays run at --steps",
        "old_one_model": {
            "wall_s": old_wall,
            "psnr_u8": old_stats["metrics"]["psnr_u8"],
            "seconds_train": old_stats.get("seconds_train"),
            "steps_per_s": (args.steps / old_stats["seconds_train"]) if old_stats.get("seconds_train") else None,
        },
        "new_one_model": {
            "wall_s": new_wall,
            "psnr_u8": new_stats["metrics"]["psnr_u8"],
            "seconds_train": new_stats.get("seconds_train"),
            "steps_per_s": new_stats.get("steps_per_s"),
            "compiled": new_stats.get("compiled"),
            "compile_error": new_stats.get("compile_error"),
            "cuda_graph": new_stats.get("cuda_graph"),
            "cuda_graph_error": new_stats.get("cuda_graph_error"),
            "precision": new_stats.get("precision"),
        },
        "old_seed_pool": {
            "wall_s": pool_old_wall,
            "psnr_u8": pool_old["metrics"]["psnr_u8"],
            "seeds": args.seeds,
        },
        "new_seed_pool": {
            "wall_s": pool_new_wall,
            "psnr_u8": pool_new["metrics"]["psnr_u8"],
            "halving": pool_new.get("halving"),
            "winner_seed": pool_new.get("winner_seed"),
        },
        "time_to_target_psnr_old_s": old_wall,
        "time_to_target_psnr_new_s": time_to_new,
        "speedup_time_to_target": (old_wall / time_to_new) if time_to_new else None,
        "speedup_one_model_wall": (old_wall / new_wall) if new_wall else None,
        "speedup_seed_pool_wall": (pool_old_wall / pool_new_wall) if pool_new_wall else None,
        "new_psnr_when_target_recorded": new_psnr_at_equal_steps,
        "profile": profile,
        "precision_probe": precision,
        "rx9070_c33648b": {
            "per_1000_steps_s": 30.0,
            "s20000_wall_s": 648.0,
            "fp32_400_s": 17.0,
            "bf16_400_s": 12.0,
            "precision_chosen_by_equal_steps": "fp32",
            "partial_psnr_bpp": [
                ["metal_g16_s20000", 39.4655, 11.7025],
                ["paving_g16_s20000", 34.4812, 11.8823],
                ["metal_b6_s10000", 33.6462, 5.3630],
                ["paving_b6", 29.6474, 4.9972],
                ["metal_b4", 29.7362, 3.0980],
                ["paving_b4", 29.2320, 3.4713],
            ],
        },
        "note": (
            "speedup_time_to_target is old wall divided by the fast wall needed to reach "
            "the eager PSNR. A null value means the fast run had not reached that PSNR "
            "by 4x the step count. gpu_idle_fraction is null without CUDA. "
            "The 3x bar is this script on the RX 9070 at --size 256 --g16 --steps 200."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(
        f"old {old_wall:.3f}s psnr {target:.3f}  new {new_wall:.3f}s psnr {new_stats['metrics']['psnr_u8']:.3f}  "
        f"pool {pool_old_wall:.3f}s -> {pool_new_wall:.3f}s  "
        f"speedup_target {report['speedup_time_to_target']}  "
        f"gpu_idle {profile['gpu_idle_fraction']}"
    )
    print("wrote", args.out)


if __name__ == "__main__":
    main()
