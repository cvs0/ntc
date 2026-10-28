"""Automatic-selector rate curve. Uses TrainConfig defaults for the pool."""

from __future__ import annotations

import json
from pathlib import Path

from ntc.container import parse_texture_set
from ntc.cost import sample_cost, workgroup_latent_bytes
from ntc.encode import TrainConfig, encode_arrays
from ntc.materials import load_material


def derived_mip_bytes(ts) -> dict:
    """Bytes of the integer-box mip chain that are not stored in the file.

    Mip 0 of each grid is stored. Mips 1..num_mips-1 are derived at upload.
    """
    extra = 0
    stored = 0
    for g in ts.grids:
        stored += g.data_bytes
        w, h = g.width, g.height
        for mip in range(1, ts.num_mips):
            w = max(1, w // 2)
            h = max(1, h // 2)
            if w == g.width and h == g.height:
                break
            stride = (w * g.bits + 7) // 8
            extra += stride * h * g.channels
            if w == 1 and h == 1:
                break
    return {"stored_latent_bytes": stored, "derived_mip_bytes": extra}


def _row(images, sems, cfg) -> dict:
    _blob, stats = encode_arrays(images, sems, cfg)
    ts = parse_texture_set(_blob)
    m = stats["metrics"]
    cost = sample_cost(ts, 1)
    tile = workgroup_latent_bytes(ts, 8, 0)
    return {
        "hidden": stats["hidden"],
        "steps": stats["steps"],
        "weighting": stats["channel_weighting"],
        "adaptive_pool": stats["adaptive_pool"],
        "checkpoint": stats.get("checkpoint"),
        "psnr_u8_init": stats.get("psnr_u8_init"),
        "psnr_u8_trained": stats.get("psnr_u8_trained"),
        "seed": stats.get("seed"),
        "seed_probe": stats.get("seed_probe"),
        "total_bpp": stats["total_bpp"],
        "psnr_u8": m["psnr_u8"],
        "ssim": m.get("ssim"),
        "psnr_u8_albedo": m.get("psnr_u8_albedo"),
        "psnr_u8_normal": m.get("psnr_u8_normal"),
        "psnr_u8_roughness": m.get("psnr_u8_roughness"),
        "psnr_u8_metalness": m.get("psnr_u8_metalness"),
        "psnr_u8_ao": m.get("psnr_u8_ao"),
        "normal_angle_deg": m.get("normal_angle_deg"),
        "channel_psnr_u8": m.get("channel_psnr_u8"),
        "grids": stats["grids"],
        "macs": cost["macs"],
        "flops": cost["flops"],
        "latent_bytes_upper": cost["latent_bytes_upper"],
        "latent_bytes_shared_x": cost["latent_bytes_shared_x"],
        "weight_bytes_cold": cost["weight_bytes_cold"],
        "weight_bytes_shared_64": cost["weight_bytes_per_invocation_if_shared_64"],
        "tile_unique_bytes_per_texel": tile["bytes_per_texel_unique"],
        "tile_upper_bytes_per_texel": tile["bytes_per_texel_upper"],
        "seconds_train": stats["seconds_train"],
        "latent_deflate": stats.get("latent_deflate"),
        "latent_stored_bytes": stats.get("latent_stored_bytes"),
        "latent_plain_bytes": stats.get("latent_plain_bytes"),
        "mip_loss": stats.get("mip_loss"),
        "mip_levels": stats.get("mip_levels"),
        "mip0_psnr": m.get("psnr_u8"),
        "mip1_psnr": (stats.get("mip_metrics") or {}).get("m1", {}).get("psnr_u8"),
        "mip2_psnr": (stats.get("mip_metrics") or {}).get("m2", {}).get("psnr_u8"),
        "mip_bytes": derived_mip_bytes(ts),
        "probe": stats["probe"],
    }


def _frozen(**kwargs):
    """The curve in ``results/rd_auto_256.json`` predates deflate and footprint loss."""
    kwargs.setdefault("latent_deflate", False)
    kwargs.setdefault("mip_levels", 1)
    kwargs.setdefault("mip_loss", "geom")
    kwargs.setdefault("log_every", 10**9)
    return TrainConfig(**kwargs)


def main() -> None:
    out = Path("results/rd_auto_256.json")
    rows = []

    def add(material, images, sems, cfg, tag):
        rec = _row(images, sems, cfg)
        rec["tag"] = tag
        rec["material"] = material
        rows.append(rec)
        out.write_text(json.dumps({"rows": rows}, indent=2))
        print(
            f"{tag} bpp={rec['total_bpp']:.3f} psnr={rec['psnr_u8']:.3f} "
            f"ssim={rec['ssim']:.4f} kept={rec['checkpoint']} grids={rec['grids']} "
            f"macs={rec['macs']} train={rec['seconds_train']:.2f}",
            flush=True,
        )
        return rec

    images, sems, _info = load_material("/tmp/mat/metal_plate", 256)
    for bpp in (4, 8, 10, 12, 16):
        add(
            "metal_plate",
            images,
            sems,
            _frozen(steps=40, target_bpp=float(bpp), log_every=10**9),
            f"auto_h32_s40_b{bpp}",
        )
    winner = rows[-1]
    grids = [tuple(g) for g in winner["grids"]]
    add(
        "metal_plate",
        images,
        sems,
        _frozen(
            steps=80,
            target_bpp=16.0,
            grids=grids,
            adaptive_pool=1,
            seed_pool=1,
            log_every=10**9,
        ),
        "locked_h32_s80",
    )
    add(
        "metal_plate",
        images,
        sems,
        _frozen(
            steps=40,
            target_bpp=16.0,
            grids=grids,
            adaptive_pool=1,
            seed_pool=1,
            channel_weighting=True,
            normal_cosine=0.5,
            log_every=10**9,
        ),
        "locked_h32_s40_weight",
    )
    add(
        "metal_plate",
        images,
        sems,
        _frozen(
            steps=40,
            target_bpp=16.0,
            hidden=(),
            adaptive_pool=8,
            seed_pool=1,
            log_every=10**9,
        ),
        "auto_linear_s40_b16",
    )
    pav, psem, _pinfo = load_material("/tmp/paving", 256)
    for bpp in (8, 16):
        add(
            "PavingStones070",
            pav,
            psem,
            _frozen(steps=40, target_bpp=float(bpp), log_every=10**9),
            f"paving_auto_h32_s40_b{bpp}",
        )


if __name__ == "__main__":
    main()
