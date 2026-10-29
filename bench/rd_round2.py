"""Rate curve after latent deflate and footprint mip loss.

Same selector, step count, and materials as ``bench/rd_auto.py``. ``total_bpp``
counts the deflated latent. PSNR is mip 0. Block-codec rows are not recomputed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rd_auto import _row

from ntc.encode import TrainConfig
from ntc.materials import load_material


def main() -> None:
    out = Path("results/rd_round2.json")
    rows: list[dict] = []
    protocol = "deflate on, mip_levels 4, mip_loss footprint, pool 8, seed pool 4, steps 40, weighting off"

    def add(material, images, sems, cfg, tag):
        rec = _row(images, sems, cfg)
        rec["tag"] = tag
        rec["material"] = material
        rows.append(rec)
        out.write_text(json.dumps({"protocol": protocol, "rows": rows}, indent=2))
        print(
            f"{tag} bpp={rec['total_bpp']:.3f} psnr={rec['psnr_u8']:.3f} "
            f"ssim={rec.get('ssim')} kept={rec['checkpoint']} grids={rec['grids']} "
            f"train={rec['seconds_train']:.2f}",
            flush=True,
        )

    images, sems, _info = load_material("/tmp/mat/metal_plate", 256)
    for bpp in (4, 8, 10, 12, 16):
        add(
            "metal_plate",
            images,
            sems,
            TrainConfig(
                steps=40,
                target_bpp=float(bpp),
                log_every=10**9,
                latent_deflate=True,
                mip_levels=4,
                mip_loss="footprint",
            ),
            f"auto_h32_s40_b{bpp}",
        )
    add(
        "metal_plate",
        images,
        sems,
        TrainConfig(
            steps=40,
            target_bpp=16.0,
            hidden=(),
            adaptive_pool=8,
            seed_pool=1,
            log_every=10**9,
            latent_deflate=True,
            mip_levels=4,
            mip_loss="footprint",
        ),
        "auto_linear_s40_b16",
    )
    pav, psem, _pinfo = load_material("/tmp/paving", 256)
    for bpp in (8, 10, 16):
        add(
            "PavingStones070",
            pav,
            psem,
            TrainConfig(
                steps=40,
                target_bpp=float(bpp),
                log_every=10**9,
                latent_deflate=True,
                mip_levels=4,
                mip_loss="footprint",
            ),
            f"paving_auto_h32_s40_b{bpp}",
        )


if __name__ == "__main__":
    main()
