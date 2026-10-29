"""Write synthetic .ntc files for ntc_decbench.

Random codes and weights in shipped and 2K shapes. Decode cost depends only
on the shape, so these time the same as trained files of that shape, and they
let the bench run on a machine without trained 2K results.

    python bench/make_bench_ntc.py [--out DIR] [--presets a,b,...]

Presets (grids as (width, height, channels, bits), MLP widths):
    ship256   (256,256,4,4) (8,8,2,2)             6-32-9   the shipped file shape
    b10_2k    (2048,2048,4,4) (1024,1024,4,4)     8-32-9   sdk b10 grids
    b15_2k    (2048,2048,4,4) (1024,1024,12,4)    16-32-9  sdk b15 grids
    wide_2k   b15_2k grids                        16-64-64-9
    big_4k    (4096,4096,4,4) (2048,2048,4,4)     8-32-9   larger than L2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ntc.container import ACT_RELU, PE_NONE, ChannelRec, build_texture_set

PRESETS: dict[str, tuple[int, list[tuple[int, int, int, int]], list[int]]] = {
    "ship256": (256, [(256, 256, 4, 4), (8, 8, 2, 2)], [32]),
    "b10_2k": (2048, [(2048, 2048, 4, 4), (1024, 1024, 4, 4)], [32]),
    "b15_2k": (2048, [(2048, 2048, 4, 4), (1024, 1024, 12, 4)], [32]),
    "wide_2k": (2048, [(2048, 2048, 4, 4), (1024, 1024, 12, 4)], [64, 64]),
    "big_4k": (4096, [(4096, 4096, 4, 4), (2048, 2048, 4, 4)], [32]),
}


def make(size: int, grid_specs, hidden: list[int], channels: int = 9, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    grids = []
    for w, h, c, bits in grid_specs:
        codes = rng.integers(0, 1 << bits, size=(c, h, w), dtype=np.uint8)
        grids.append({"codes": codes, "bits": bits, "scale": np.full(c, 0.25, np.float32)})
    widths = [sum(g[2] for g in grid_specs), *hidden, channels]
    layers = []
    for inn, out in zip(widths[:-1], widths[1:]):
        layers.append({
            "codes": rng.integers(-127, 128, size=(out, inn)).astype(np.int32),
            "bias": rng.normal(0.0, 0.05, size=out).astype(np.float32),
            "scale": np.full((out, 1), 1.0 / (127.0 * np.sqrt(inn)), np.float32),
            "group_size": 0,
        })
    mips = int(np.log2(size)) + 1
    return build_texture_set(
        width=size, height=size, num_mips=mips, activation=ACT_RELU, pe_mode=PE_NONE,
        pe_octaves=0, flags=0, weight_bits=8, group_size=0,
        channels=[ChannelRec(i) for i in range(channels)], grids=grids, layers=layers,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="synthetic .ntc files for ntc_decbench")
    ap.add_argument("--out", default="build/decbench")
    ap.add_argument("--presets", default="ship256,b10_2k,b15_2k,wide_2k")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in [p for p in args.presets.split(",") if p]:
        size, grids, hidden = PRESETS[name]
        path = out / f"synth_{name}.ntc"
        if path.exists():
            print(f"keep {path}")
            continue
        path.write_bytes(make(size, grids, hidden))
        print(f"wrote {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
