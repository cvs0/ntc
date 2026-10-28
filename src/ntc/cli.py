"""Command line entry points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ntc.container import parse_texture_set, rate_report
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
    p.add_argument("--no-deflate", action="store_true")
    p.add_argument(
        "--latent-coder",
        default="best",
        choices=("deflate", "range", "best", "context", "best_all", "raw"),
    )
    p.add_argument("--target-psnr", type=float, default=None)
    p.add_argument("--rd-lambda", type=float, default=0.0)
    p.add_argument("--search-steps", type=int, default=0)
    p.add_argument("--mip-levels", type=int, default=4)
    p.add_argument("--mip-loss", default="footprint", choices=("geom", "footprint", "uniform"))
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
        latent_deflate=not args.no_deflate and args.latent_coder != "raw",
        latent_coder="raw" if args.latent_coder == "raw" else args.latent_coder,
        target_psnr=args.target_psnr,
        rd_lambda=args.rd_lambda,
        search_steps=args.search_steps,
        mip_levels=args.mip_levels,
        mip_loss=args.mip_loss,
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
    rates = rate_report(ts)
    np.savez(
        args.output,
        image=img.astype(np.float32),
        bpp=np.float32(rates["total_bpp"]),
        latent_bpp_mip0=np.float32(rates["latent_bpp_mip0"]),
        file_bpp_chain=np.float32(rates["file_bpp_chain"]),
        bppc=np.float32(rates["bppc"]),
    )
    print(
        f"decoded {img.shape} total_bpp {rates['total_bpp']:.4f} "
        f"latent_bpp_mip0 {rates['latent_bpp_mip0']:.4f} "
        f"file_bpp_chain {rates['file_bpp_chain']:.4f} bppc {rates['bppc']:.4f} "
        f"header_bytes {rates['header_bytes']} network_bytes {rates['network_bytes']}"
    )
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
