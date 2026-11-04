"""Short CPU check that the saved file stays with the training PSNR.

One 16x16 stand-in, 40 steps, each run 3 flag on its own and all of them
together. Prints loop PSNR, saved psnr_u8, and the gap. Exits 1 when a
saved file is more than 0.5 dB under the end-of-training score.

    python bench/repro_export_gap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_ROOT), str(_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

from ntc.encode import TrainConfig  # noqa: E402
from ntc.train_fast import EXPORT_MATCH_DB, encode_arrays_fast  # noqa: E402


def stand_in(n: int = 16):
    yy, xx = np.mgrid[0:n, 0:n]
    img = np.stack(
        [
            0.45 + 0.4 * np.sin(xx / 3.0) * np.cos(yy / 4.0),
            0.35 + 0.25 * np.sin(xx / 2.1 + yy / 5.0),
            0.55 + 0.2 * np.cos(xx / 6.0),
            0.5 + 0.5 * np.sin(xx / 2.0),
            0.5 + 0.5 * np.cos(yy / 2.0),
            0.2 * np.ones((n, n)),
            0.4 + 0.1 * np.sin(xx / 8.0),
            0.3 * np.ones((n, n)),
            0.6 * np.ones((n, n)),
        ],
        0,
    ).astype(np.float32)
    img = np.clip(img, 0, 1)
    sems = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    return img, sems


def case_config(**kw) -> TrainConfig:
    base = dict(
        steps=40,
        hidden=(32,),
        grids=[(8, 8, 4, 4)],
        fast=True,
        adaptive_pool=1,
        seed_pool=1,
        with_ssim=False,
        log_every=10**9,
        mip_levels=2,
        seed=1,
        compile_step="off",
        cuda_graph="off",
        precision="fp32",
        batch_pixels=256,
        batch_mip="flat",
        target_bpp=4.0,
        device="cpu",
    )
    base.update(kw)
    return TrainConfig(**base)


# The destructive flags are on here on purpose. The export has to refuse
# them when they open a gap, which is what run 3 did not do.
CASES = [
    ("plain", {}),
    ("prune", {"prune_keep": 0.5}),
    ("vq", {"vq_latents": True}),
    ("shared", {"shared_decoder": True}),
    ("tile", {"tile_head": True}),
    ("residual", {"bc_residual": True}),
    ("matryoshka", {"matryoshka": True}),
    ("int8", {"mlp_int8_tail": 0.5}),
    ("ema", {"ema_decay": 0.5}),
    ("render", {"render_loss": True}),
    ("rd", {"rd_alloc": True}),
    (
        "run3",
        {
            "prune_keep": 0.5,
            "vq_latents": True,
            "shared_decoder": True,
            "tile_head": True,
            "bc_residual": True,
            "matryoshka": True,
            "mlp_int8_tail": 0.5,
            "ema_decay": 0.5,
            "render_loss": True,
            "rd_alloc": True,
        },
    ),
]


def measure(name: str, **kw) -> dict:
    img, sems = stand_in()
    _blob, stats = encode_arrays_fast(img, sems, case_config(**kw))
    loop = float(stats["psnr_u8_loop"])
    final = float(stats["metrics"]["psnr_u8"])
    return {
        "name": name,
        "loop": loop,
        "file": final,
        "gap": loop - final,
        "checkpoint": stats.get("checkpoint"),
        "ship_rejected": bool(stats.get("ship_rejected")),
    }


def main() -> int:
    failed = 0
    for name, kw in CASES:
        row = measure(name, **kw)
        flag = "FAIL" if row["gap"] > EXPORT_MATCH_DB else "ok"
        if flag == "FAIL":
            failed += 1
        print(
            f"{flag:4} {name:12} loop {row['loop']:7.2f} psnr_u8 {row['file']:7.2f} "
            f"gap {row['gap']:+6.2f} ckpt {row['checkpoint']} "
            f"ship_rejected {row['ship_rejected']}",
            flush=True,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
