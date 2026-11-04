"""Step-reduction ablation at the 256 g16 shape.

Run this on the RX 9070 (Windows or WSL) and keep the json. It does not
write ``results/train_max`` or ``results/train_max_v2``. Each idea trains
until ``--target-psnr`` or ``--steps``, whichever comes first.

Windows:

    $env:PYTHONPATH = "src;."
    .venv-rocm/Scripts/python.exe bench/ablate_g16.py --material data/materials/metal_plate --steps 2000 --target-psnr 32 --out results/train_speed/ablation_g16.json

WSL:

    export PYTHONPATH=src:. HSA_ENABLE_DXG_DETECTION=1
    python bench/ablate_g16.py --material data/materials/metal_plate --steps 2000 --target-psnr 32 --out results/train_speed/ablation_g16.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ntc.encode import TrainConfig
from ntc.materials import load_material
from ntc.train_fast import encode_arrays_fast

G16 = [(256, 256, 4, 4), (8, 8, 2, 2)]


def _once(images, sems, steps, target, extra) -> dict:
    cfg = TrainConfig(
        steps=steps,
        hidden=(32,),
        grids=G16,
        adaptive_pool=1,
        seed_pool=1,
        with_ssim=False,
        log_every=10**9,
        mip_levels=4,
        seed=3,
        fast=True,
        compile_step="auto",
        cuda_graph="auto",
        stop_psnr=target,
        check_every=100,
        eval_every=0,
        early_stop=0,
        **extra,
    )
    t0 = time.perf_counter()
    _blob, stats = encode_arrays_fast(images, sems, cfg)
    wall = time.perf_counter() - t0
    return {
        "wall_s": wall,
        "seconds_train": stats.get("seconds_train"),
        "steps_ran": stats.get("steps_ran"),
        "psnr_u8": stats["metrics"]["psnr_u8"],
        "psnr_u8_init": stats.get("psnr_u8_init"),
        "psnr_u8_trained": stats.get("psnr_u8_trained"),
        "checkpoint": stats.get("checkpoint"),
        "total_bpp": stats.get("total_bpp"),
        "hit_target": stats.get("hit_target"),
        "compiled": stats.get("compiled"),
        "cuda_graph": stats.get("cuda_graph"),
        "compile_error": stats.get("compile_error"),
        "steps_per_s": stats.get("steps_per_s"),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--material", type=Path, default=Path("data/materials/metal_plate"))
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--target-psnr", type=float, default=32.0)
    p.add_argument("--prior", type=Path, default=Path("src/ntc/priors/decoder_g16.pt"))
    p.add_argument("--out", type=Path, default=Path("results/train_speed/ablation_g16.json"))
    args = p.parse_args()
    images, sems, _info = load_material(args.material, 256)
    ideas = {
        "baseline": {},
        "lion": {"optimizer": "lion", "lr": 1e-4},
        "warmup_lr3": {"warmup_steps": 200, "lr": 1.5e-2},
        "warmup_lr5": {"warmup_steps": 200, "lr": 2.5e-2},
        "onecycle_peak5": {"schedule": "onecycle", "onecycle_peak": 5.0},
        "coarse_to_fine": {"coarse_to_fine": True},
        "warm_finetune": {"warm_start": str(args.prior), "freeze_decoder": False},
        "warm_freeze": {"warm_start": str(args.prior), "freeze_decoder": True},
        "latent_lr_8": {"latent_lr_mult": 8.0},
    }
    rows = {}
    for name, extra in ideas.items():
        print("start", name, flush=True)
        row = _once(images, sems, args.steps, args.target_psnr, extra)
        rows[name] = row
        print(
            f"{name} steps {row['steps_ran']} psnr {row['psnr_u8']:.3f} "
            f"bpp {row['total_bpp']:.3f} ckpt {row['checkpoint']} compiled {row['compiled']} "
            f"graph {row['cuda_graph']}",
            flush=True,
        )
    base = rows["baseline"]
    base_s = base.get("seconds_train") or base["wall_s"]
    winners = []
    for name, row in rows.items():
        if name == "baseline":
            continue
        reached = bool(row["hit_target"]) or (row["psnr_u8"] + 1e-3 >= args.target_psnr)
        train_s = row.get("seconds_train") or row["wall_s"]
        row["reached"] = reached
        row["faster_than_baseline"] = bool(reached and base_s and train_s < 0.9 * base_s)
        if row["faster_than_baseline"]:
            winners.append(name)
    report = {
        "material": str(args.material),
        "size": 256,
        "grids": G16,
        "hidden": [32],
        "steps_cap": args.steps,
        "target_psnr": args.target_psnr,
        "keep_rule": "reached the target and train seconds under 90% of baseline",
        "winners": winners,
        "rows": rows,
        "note": (
            "metal_g16_step_s10000 at 18.32 dB and 2.43 bpp is the PCA init "
            "or a railed latent: a trained G16 file is near 11 bpp. The step "
            "schedule holds the peak learning rate until halfway, and early "
            "stop used to give up before the run beat that init. Latent codes "
            "are now clamped to the bin range, and patience does not increment "
            "while the eval PSNR is still under the init."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print("winners", winners)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
