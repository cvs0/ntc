"""Command line entry points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ntc.container import parse_texture_set, total_bpp
from ntc.decode_ref import decode_mip
from ntc.encode import TrainConfig, encode_arrays
from ntc.materials import load_material
from ntc.metrics import group_metrics


def encode_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ntc-encode")
    p.add_argument("material", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--bpp", type=float, default=4.0)
    p.add_argument("--hidden", default="32")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--adaptive-pool", type=int, default=8)
    p.add_argument("--seed-pool", type=int, default=4)
    p.add_argument("--stats", type=Path, default=None)
    args = p.parse_args(argv)
    images, sems, info = load_material(args.material, args.size)
    hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip() != "")
    cfg = TrainConfig(
        steps=args.steps,
        hidden=hidden,
        target_bpp=args.bpp,
        device=args.device,
        seed=args.seed,
        adaptive_pool=args.adaptive_pool,
        seed_pool=args.seed_pool,
    )
    blob, stats = encode_arrays(images, sems, cfg)
    args.output.write_bytes(blob)
    stats["material"] = info
    text = json.dumps(stats, indent=2)
    if args.stats:
        args.stats.write_text(text)
    print(text)


def decode_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ntc-decode")
    p.add_argument("ntc", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--mip", type=int, default=0)
    p.add_argument("--reference", type=Path, default=None)
    p.add_argument("--size", type=int, default=None)
    args = p.parse_args(argv)
    blob = args.ntc.read_bytes()
    ts = parse_texture_set(blob)
    img = decode_mip(ts, args.mip)
    # npz: image HWC float32
    np.savez(args.output, image=img.astype(np.float32), bpp=np.float32(total_bpp(ts)))
    print(f"decoded {img.shape} bpp {total_bpp(ts):.4f}")
    if args.reference:
        ref, sems, _info = load_material(args.reference, args.size)
        ref_hwc = np.transpose(ref, (1, 2, 0))
        if args.mip:
            print("reference metrics are for mip 0 only", file=sys.stderr)
        else:
            print(json.dumps(group_metrics(img, ref_hwc, sems), indent=2))


def ablate_main(argv: list[str] | None = None) -> None:
    from ntc.ablate import main as run

    run(argv)


def bench_main(argv: list[str] | None = None) -> None:
    from ntc.bench import main as run

    run(argv)
