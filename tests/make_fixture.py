"""Write tests/fixtures/tiny.ntc and the Python reference images."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ntc.container import ACT_RELU, PE_NONE, ChannelRec, build_texture_set
from ntc.decode_ref import decode_mip


def main() -> None:
    rng = np.random.default_rng(1)
    grids = []
    specs = [
        (8, 8, 4, 4),
        (8, 8, 2, 2),
        (4, 4, 2, 8),
    ]
    for w, h, c, bits in specs:
        qmax = (1 << bits) - 1
        codes = rng.integers(0, qmax + 1, size=(c, h, w), dtype=np.uint8)
        scale = rng.uniform(0.01, 0.05, size=(c,)).astype(np.float32)
        grids.append({"codes": codes, "bits": bits, "scale": scale})
    layers = []
    shapes = [(8, 4), (4, 3)]
    for inn, out in shapes:
        codes = rng.integers(-30, 31, size=(out, inn)).astype(np.int32)
        bias = rng.normal(0.0, 0.05, size=(out,)).astype(np.float32)
        scale = np.full((out, 1), 1.0 / 127.0, dtype=np.float32)
        layers.append({"codes": codes, "bias": bias, "scale": scale, "group_size": 0})
    blob = build_texture_set(
        width=8,
        height=8,
        num_mips=4,
        activation=ACT_RELU,
        pe_mode=PE_NONE,
        pe_octaves=0,
        flags=0,
        weight_bits=8,
        group_size=0,
        channels=[ChannelRec(i) for i in range(3)],
        grids=grids,
        layers=layers,
    )
    from ntc.container import parse_texture_set

    ts = parse_texture_set(blob)
    root = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
    root.mkdir(parents=True, exist_ok=True)
    (root / "tiny.ntc").write_bytes(blob)
    for mip in (0, 1):
        img = decode_mip(ts, mip)
        h, w, c = img.shape
        hdr = struct.pack("<IIII", w, h, c, mip)
        (root / f"tiny_mip{mip}.f32").write_bytes(hdr + img.astype(np.float32).tobytes())
        print(f"mip {mip} {w}x{h}x{c}")
    print(f"wrote {root} ({len(blob)} bytes)")


if __name__ == "__main__":
    main()
