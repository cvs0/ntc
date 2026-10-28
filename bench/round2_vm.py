"""Locked-grid ablations for the rate and quality gaps.

One factor changes per row. The reference decoder is the headline. Each
row also recompresses the latent region with raw deflate and records that
rate. PSNR of the deflated file is checked against the plain file.

This is the CPU tier. Longer schedules live in bench/train_sweep.py.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np

from ntc.container import (
    ChannelRec,
    build_texture_set,
    parse_texture_set,
    total_bpp,
)
from ntc.decode_ref import decode_mip
from ntc.device import device_report, pick_device
from ntc.encode import TrainConfig, encode_arrays
from ntc.materials import load_material
from ntc.metrics import psnr_u8

G16 = [(256, 256, 4, 4), (8, 8, 2, 2)]
G10 = [(256, 256, 2, 4), (64, 64, 4, 8)]
G3 = [(256, 256, 4, 4), (64, 64, 2, 4), (8, 8, 2, 2)]


def _cfg(**kw) -> TrainConfig:
    base = dict(
        steps=40,
        hidden=(32,),
        grids=G16,
        adaptive_pool=1,
        seed_pool=1,
        channel_weighting=False,
        normal_cosine=0.0,
        seed=3,
        log_every=10**9,
        with_ssim=True,
        target_bpp=16.0,
        latent_deflate=False,
    )
    base.update(kw)
    return TrainConfig(**base)


def jobs() -> list[tuple[str, str, int, TrainConfig]]:
    """(name, material, size, cfg)."""
    metal = "/tmp/mat/metal_plate"
    paving = "/tmp/paving"
    rows = [
        ("m16_s40", metal, _cfg()),
        ("m16_s80", metal, _cfg(steps=80)),
        ("m16_s160", metal, _cfg(steps=160)),
        ("m16_s320", metal, _cfg(steps=320)),
        ("m16_s800", metal, _cfg(steps=800)),
        ("m16_int4_row", metal, _cfg(weight_bits=4, group_size=0)),
        ("m16_int4_g16", metal, _cfg(weight_bits=4, group_size=16)),
        ("m16_int4_row_s160", metal, _cfg(steps=160, weight_bits=4, group_size=0)),
        ("m16_h64_32_s40", metal, _cfg(hidden=(64, 32))),
        ("m16_h64_32_s160", metal, _cfg(steps=160, hidden=(64, 32))),
        ("m16_h64_32_mix_s40", metal, _cfg(hidden=(64, 32), layer_bits=(8, 4, 8), weight_bits=8)),
        ("m16_h64_32_mix_s160", metal, _cfg(steps=160, hidden=(64, 32), layer_bits=(8, 4, 8))),
        ("m16_pe_f_s40", metal, _cfg(pe="fourier", pe_octaves=1)),
        ("m16_pe_f_s160", metal, _cfg(steps=160, pe="fourier", pe_octaves=1)),
        ("m16_pe_t_s40", metal, _cfg(pe="triangle", pe_octaves=3)),
        ("m16_pe_t_s160", metal, _cfg(steps=160, pe="triangle", pe_octaves=3)),
        ("m16_hgelu_s40", metal, _cfg(activation="hgelu")),
        ("m16_hgelu_s160", metal, _cfg(steps=160, activation="hgelu")),
        ("m16_requant_s40", metal, _cfg(activation="hgelu", activation_requant=True)),
        ("m16_requant_s160", metal, _cfg(steps=160, activation="hgelu", activation_requant=True)),
        ("m16_mip4_geom", metal, _cfg(mip_levels=4, mip_loss="geom")),
        ("m16_mip4_foot", metal, _cfg(mip_levels=4, mip_loss="footprint")),
        ("m16_mip4_foot_cur", metal, _cfg(mip_levels=4, mip_loss="footprint", mip_curriculum=True)),
        ("m16_mip4_foot_s160", metal, _cfg(steps=160, mip_levels=4, mip_loss="footprint")),
        ("m16_tile8_s40", metal, _cfg(tile_size=8)),
        ("m16_tile16_s40", metal, _cfg(tile_size=16)),
        ("m16_tile32_s40", metal, _cfg(tile_size=32)),
        ("m16_tile16_s160", metal, _cfg(steps=160, tile_size=16)),
        ("m16_tile8_s160", metal, _cfg(steps=160, tile_size=8)),
        ("m16_grid3_s40", metal, _cfg(grids=G3)),
        ("m16_grid3_s160", metal, _cfg(steps=160, grids=G3)),
        ("m10_s40", metal, _cfg(grids=G10, seed=1, target_bpp=10)),
        ("m10_s160", metal, _cfg(steps=160, grids=G10, seed=1, target_bpp=10)),
        ("m10_tile16_s40", metal, _cfg(grids=G10, seed=1, tile_size=16, target_bpp=10)),
        ("m10_tile16_s160", metal, _cfg(steps=160, grids=G10, seed=1, tile_size=16, target_bpp=10)),
        ("p16_s40", paving, _cfg()),
        ("p16_s160", paving, _cfg(steps=160)),
        ("p16_tile16_s40", paving, _cfg(tile_size=16)),
        ("p16_tile16_s160", paving, _cfg(steps=160, tile_size=16)),
        ("p10_s40", paving, _cfg(grids=G10, seed=1, target_bpp=10)),
    ]
    return [(n, m, 256, c) for n, m, c in rows]


def _redeflate(blob: bytes) -> tuple[bytes, float]:
    ts = parse_texture_set(blob)
    grids = []
    for g in ts.grids:
        item = {"codes": g.codes, "bits": g.bits, "scale": g.scale.astype(np.float32)}
        if g.tile_scale is not None:
            item["tile_scale"] = g.tile_scale
            item["tile_offset"] = g.tile_offset
        grids.append(item)
    layers = []
    for layer in ts.layers:
        layers.append(
            {
                "codes": np.rint(layer.weight).astype(np.int16),
                "bias": layer.bias.astype(np.float32),
                "scale": layer.scale.astype(np.float32),
                "group_size": int(layer.group_size),
                "weight_bits": int(layer.weight_bits or ts.weight_bits),
            }
        )
    out = build_texture_set(
        width=ts.width,
        height=ts.height,
        num_mips=ts.num_mips,
        activation=ts.activation,
        pe_mode=ts.pe_mode,
        pe_octaves=ts.pe_octaves,
        flags=ts.flags & ~0x4,
        weight_bits=ts.weight_bits,
        group_size=ts.group_size,
        channels=[ChannelRec(c.semantic, c.color_space) for c in ts.channels],
        grids=grids,
        layers=layers,
        latent_deflate=True,
        tile_size=ts.tile_size,
    )
    return out, total_bpp(parse_texture_set(out))


def main() -> None:
    out = Path("results/round2_vm.json")
    payload = {
        "device": device_report(pick_device("auto")),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "protocol": "locked grids, seed_pool 1, channel weighting off, cosine LR to 0.05",
        "runs": [],
    }
    cache: dict[tuple[str, int], tuple] = {}
    for name, material, size, cfg in jobs():
        key = (material, size)
        if key not in cache:
            cache[key] = load_material(material, size)
        images, sems, _info = cache[key]
        t0 = time.perf_counter()
        blob, stats = encode_arrays(images, sems, cfg)
        wall = time.perf_counter() - t0
        ref = np.transpose(images, (1, 2, 0))
        dblob, dbpp = _redeflate(blob)
        drecon = decode_mip(parse_texture_set(dblob), 0)
        dpsnr = psnr_u8(drecon, ref)
        m = stats["metrics"]
        row = {
            "name": name,
            "material": material,
            "psnr_u8": m.get("psnr_u8"),
            "ssim": m.get("ssim"),
            "psnr_u8_albedo": m.get("psnr_u8_albedo"),
            "psnr_u8_normal": m.get("psnr_u8_normal"),
            "psnr_u8_roughness": m.get("psnr_u8_roughness"),
            "psnr_u8_metalness": m.get("psnr_u8_metalness"),
            "psnr_u8_ao": m.get("psnr_u8_ao"),
            "normal_angle_deg": m.get("normal_angle_deg"),
            "channel_psnr_u8": m.get("channel_psnr_u8"),
            "total_bpp": stats["total_bpp"],
            "deflate_bpp": dbpp,
            "deflate_psnr_u8": dpsnr,
            "latent_plain_bytes": stats["latent_plain_bytes"],
            "deflate_latent_bytes": parse_texture_set(dblob).latent_stored_bytes,
            "steps": cfg.steps,
            "hidden": list(cfg.hidden),
            "grids": stats["grids"],
            "weight_bits": cfg.weight_bits,
            "group_size": cfg.group_size,
            "layer_bits": list(cfg.layer_bits),
            "pe": cfg.pe,
            "activation": cfg.activation,
            "activation_requant": cfg.activation_requant,
            "mip_levels": cfg.mip_levels,
            "mip_loss": cfg.mip_loss,
            "mip_curriculum": cfg.mip_curriculum,
            "tile_size": cfg.tile_size,
            "seed": cfg.seed,
            "checkpoint": stats["checkpoint"],
            "psnr_u8_init": stats["psnr_u8_init"],
            "psnr_u8_trained": stats["psnr_u8_trained"],
            "mip_metrics": {
                k: {"psnr_u8": v.get("psnr_u8")} for k, v in (stats.get("mip_metrics") or {}).items()
            },
            "seconds_train": stats["seconds_train"],
            "wall_s": wall,
        }
        payload["runs"].append(row)
        out.write_text(json.dumps(payload, indent=2))
        print(
            f"{name} psnr {row['psnr_u8']:.4f} bpp {row['total_bpp']:.4f} "
            f"deflate_bpp {dbpp:.4f} dpsnr {dpsnr:.4f} train {row['seconds_train']:.2f} "
            f"ckpt {row['checkpoint']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
