"""Rate-distortion curves from 2 to 20 bpp, and BD-rate against block codecs.

Two bpp axes are plotted. ``latent_bpp_mip0`` is the SDK axis (stored latent
codes plus fp16 latent scales, per mip 0 texel). ``file_bpp_chain`` is the
whole file over the mip-chain texels. ``bppc`` is in the JSON and on a third
panel because Vaidyanathan et al. 2023 Table 3 is in bits/pixel/channel.

Block rows use the nominal codec rate (BC1 4, BC5 8, BC7 8, ASTC 128 bits
per block). Their PSNR is ``bench/score_mips.py`` on the same mip chain.
ASTC PSNR is filled when ``astcenc`` is on PATH. Otherwise the row is an
anchor: the bpp is known and ``psnr_u8`` is null.

Table 3 is drawn and labeled as a different 20-material set. BD-rate against
those points is omitted. Measured RTXNTC rows are copied from
``results/rtxntc_compare/*/report.json`` when that file exists.

This script does not edit ``bench/train_max.py``. A 2048 run is
``python bench/rd_curve.py --material <sdk dir> --size 2048 --steps 4000``.
``--demo`` is a 32x32 CPU check of the tooling.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

# Vaidyanathan et al. 2023, Table 3. 20-material average, bppc including mips
# and weights. A different material set from the SDK trio.
PAPER_TABLE3 = (
    (0.23, 32.71),
    (0.56, 36.12),
    (1.13, 39.92),
    (2.51, 45.30),
)
PAPER_NOTE = (
    "Vaidyanathan et al. 2023 Table 3, 20-material average, "
    "bits/pixel/channel including mips and weights. "
    "A different material set, not a matched comparison."
)
RATES = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0)
ASTC_BLOCKS = {
    "4x4": (4, 4, 128.0 / 16.0),
    "5x5": (5, 5, 128.0 / 25.0),
    "6x6": (6, 6, 128.0 / 36.0),
    "8x8": (8, 8, 128.0 / 64.0),
}
_ROLE = {
    0: ("albedo", 0),
    1: ("albedo", 1),
    2: ("albedo", 2),
    3: ("normal", 0),
    4: ("normal", 1),
    5: ("normal", 2),
    6: ("roughness", 0),
    7: ("metalness", 0),
    8: ("ao", 0),
    9: ("height", 0),
}


def bits_saved_percent(
    baseline: list[tuple[float, float]],
    candidate: list[tuple[float, float]],
) -> tuple[float | None, str]:
    """Percent bits the candidate saves at equal PSNR.

    Each curve is ``(psnr_db, rate)`` sorted by PSNR. Positive means the
    candidate uses fewer bits. Two or more overlapping points use a
    piecewise-linear integral of log(rate) over PSNR (Bjontegaard with a
    linear log-rate interpolant). One baseline point interpolates the
    candidate at that PSNR.
    """
    base = _sorted_curve(baseline)
    cand = _sorted_curve(candidate)
    if not base or not cand:
        return None, "empty"
    if len(base) == 1:
        q, rate = base[0]
        if cand[0][0] - 1e-6 <= q <= cand[-1][0] + 1e-6 and len(cand) >= 2:
            log_c = _log_at(cand, q)
            if log_c is None:
                return None, "psnr ranges do not overlap"
            return (1.0 - math.exp(log_c) / rate) * 100.0, "equal_psnr_interpolation"
        return None, "psnr ranges do not overlap"
    lo = max(base[0][0], cand[0][0])
    hi = min(base[-1][0], cand[-1][0])
    if hi <= lo + 1e-9:
        return None, "psnr ranges do not overlap"
    n = 32
    acc = 0.0
    prev = None
    prev_q = None
    for i in range(n):
        q = lo + (hi - lo) * i / (n - 1)
        delta = _log_at(cand, q) - _log_at(base, q)
        if prev is not None:
            acc += 0.5 * (prev + delta) * (q - prev_q)
        prev = delta
        prev_q = q
    classic = (math.exp(acc / (hi - lo)) - 1.0) * 100.0
    return -classic, "bjontegaard_linear_log_rate"


def _sorted_curve(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    clean = [(float(q), float(r)) for q, r in points if r is not None and q is not None and r > 0]
    clean.sort()
    out: list[tuple[float, float]] = []
    for q, r in clean:
        if out and abs(out[-1][0] - q) < 1e-6:
            if r < out[-1][1]:
                out[-1] = (q, r)
        else:
            out.append((q, r))
    return out


def _log_at(curve: list[tuple[float, float]], q: float) -> float | None:
    if q < curve[0][0] - 1e-9 or q > curve[-1][0] + 1e-9:
        return None
    if q <= curve[0][0]:
        return math.log(curve[0][1])
    for i in range(len(curve) - 1):
        q0, r0 = curve[i]
        q1, r1 = curve[i + 1]
        if q <= q1 or i == len(curve) - 2:
            if abs(q1 - q0) < 1e-12:
                return math.log(min(r0, r1))
            t = min(1.0, max(0.0, (q - q0) / (q1 - q0)))
            return (1.0 - t) * math.log(r0) + t * math.log(r1)
    return math.log(curve[-1][1])


def _groups_of(row: dict) -> tuple[str, ...]:
    groups = row.get("groups")
    if isinstance(groups, dict):
        return tuple(sorted(groups))
    if isinstance(groups, (list, tuple)):
        return tuple(sorted(str(g) for g in groups))
    return tuple()


def _curve_of(rows: list[dict], axis: str) -> list[tuple[float, float]]:
    return [(r["psnr_u8"], r[axis]) for r in rows if r.get("psnr_u8") is not None and r.get(axis)]


def bd_rate_rows(ours: list[dict], blocks: list[dict], rtx: list[dict]) -> list[dict]:
    """BD-rate of each ours series against block/ASTC baselines and RTXNTC.

    Table 3 is always present with a null number. Block comparisons require
    the same scored channel groups. RTXNTC uses ``latent_bpp_mip0`` and ``bppc``.
    """
    out = [
        {
            "baseline": "paper_table3",
            "axis": "bppc",
            "bits_saved_percent": None,
            "method": None,
            "reason": "different material set",
            "series": None,
        }
    ]
    by_series: dict[str, list[dict]] = {}
    for row in ours:
        if row.get("psnr_u8") is None:
            continue
        by_series.setdefault(str(row.get("series", "ours")), []).append(row)
    baselines: dict[tuple[str, tuple[str, ...]], list[dict]] = {}
    for row in blocks:
        if row.get("psnr_u8") is None:
            continue
        key = (str(row.get("name")), _groups_of(row))
        baselines.setdefault(key, []).append(row)
    for series, rows in by_series.items():
        groups = _groups_of(rows[0])
        for (name, bgroups), brows in baselines.items():
            if bgroups != groups:
                continue
            for axis in ("file_bpp_chain", "bppc"):
                saved, method = bits_saved_percent(_curve_of(brows, axis), _curve_of(rows, axis))
                out.append(
                    {
                        "baseline": name,
                        "axis": axis,
                        "bits_saved_percent": saved,
                        "method": method if saved is not None else None,
                        "reason": None if saved is not None else method,
                        "series": series,
                    }
                )
        rtx_ok = [r for r in rtx if r.get("psnr_u8") is not None]
        if rtx_ok:
            for axis in ("latent_bpp_mip0", "bppc", "file_bpp_chain"):
                saved, method = bits_saved_percent(_curve_of(rtx_ok, axis), _curve_of(rows, axis))
                out.append(
                    {
                        "baseline": "rtxntc_measured",
                        "axis": axis,
                        "bits_saved_percent": saved,
                        "method": method if saved is not None else None,
                        "reason": None if saved is not None else method,
                        "series": series,
                    }
                )
    return out


def _save_u8(arr_hwc: np.ndarray, path: Path) -> None:
    q = np.clip(np.round(np.clip(arr_hwc, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    if q.shape[2] == 1:
        Image.fromarray(q[:, :, 0], mode="L").save(path)
    else:
        Image.fromarray(q[:, :, :3], mode="RGB").save(path)


def write_chain(dest: Path, chw: np.ndarray, semantics: list[int]) -> None:
    from ntc.materials import box_down_chw

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    level = chw
    mip = 0
    while level is not None:
        hwc = np.transpose(level, (1, 2, 0))
        planes: dict[str, list] = {}
        for channel, sem in enumerate(semantics):
            role = _ROLE.get(int(sem))
            if role is None:
                continue
            name, slot = role
            planes.setdefault(name, [None, None, None])[slot] = hwc[:, :, channel]
        for name, slots in planes.items():
            used = [s for s in slots if s is not None]
            stack = np.stack(used, axis=2)
            if mip == 0:
                path = dest / f"{name}.png"
            else:
                path = dest / "mips" / f"{name}.{mip:02d}.png"
            _save_u8(stack, path)
        mip += 1
        level = box_down_chw(level)


def _pad4(image: np.ndarray) -> tuple[np.ndarray, int, int]:
    h, w = image.shape[:2]
    ph = (4 - h % 4) % 4
    pw = (4 - w % 4) % 4
    if ph == 0 and pw == 0:
        return image, h, w
    return np.pad(image, ((0, ph), (0, pw), (0, 0)), mode="edge"), h, w


def _reconstruct_normal_z(xy_u8: np.ndarray) -> np.ndarray:
    x = xy_u8[:, :, 0].astype(np.float32) / 255.0 * 2.0 - 1.0
    y = xy_u8[:, :, 1].astype(np.float32) / 255.0 * 2.0 - 1.0
    z = np.sqrt(np.clip(1.0 - x * x - y * y, 0.0, 1.0))
    z_u8 = np.clip(np.round((z + 1.0) * 0.5 * 255.0), 0, 255).astype(np.uint8)
    rgb = np.zeros((xy_u8.shape[0], xy_u8.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = xy_u8[:, :, 0]
    rgb[:, :, 1] = xy_u8[:, :, 1]
    rgb[:, :, 2] = z_u8
    return rgb


def _transcode_role(image: np.ndarray, codec: str) -> np.ndarray:
    """Return float HWC in ``[0, 1]`` after a nominal block round trip."""
    from bench.bc_pack import (
        image_u8,
        pack_image_bc1,
        pack_image_bc5,
        pack_image_bc7,
        unpack_bc1_block,
        unpack_bc4_block,
        unpack_bc7_mode6_block,
    )

    u8 = image_u8(image)
    if u8.shape[2] == 1:
        u8 = np.repeat(u8, 3, axis=2)
    padded, h, w = _pad4(u8)
    ph, pw = padded.shape[:2]
    if codec == "bc1":
        raw = pack_image_bc1(padded[:, :, :3])
        rec = np.zeros((ph, pw, 3), dtype=np.uint8)
        i = 0
        for y in range(0, ph, 4):
            for x in range(0, pw, 4):
                rec[y : y + 4, x : x + 4] = unpack_bc1_block(raw[i : i + 8])[:, :3].reshape(4, 4, 3)
                i += 8
        rec = rec[:h, :w]
    elif codec == "bc7":
        rgba = np.full((ph, pw, 4), 255, dtype=np.uint8)
        rgba[:, :, :3] = padded[:, :, :3]
        raw = pack_image_bc7(rgba)
        rec = np.zeros((ph, pw, 3), dtype=np.uint8)
        i = 0
        for y in range(0, ph, 4):
            for x in range(0, pw, 4):
                rec[y : y + 4, x : x + 4] = unpack_bc7_mode6_block(raw[i : i + 16])[:, :3].reshape(4, 4, 3)
                i += 16
        rec = rec[:h, :w]
    elif codec == "bc5":
        raw = pack_image_bc5(padded[:, :, :2])
        xy = np.zeros((ph, pw, 2), dtype=np.uint8)
        i = 0
        for y in range(0, ph, 4):
            for x in range(0, pw, 4):
                block = raw[i : i + 16]
                xy[y : y + 4, x : x + 4, 0] = unpack_bc4_block(block[:8]).reshape(4, 4)
                xy[y : y + 4, x : x + 4, 1] = unpack_bc4_block(block[8:]).reshape(4, 4)
                i += 16
        rec = _reconstruct_normal_z(xy[:h, :w])
    else:
        raise ValueError(codec)
    if image.shape[2] == 1:
        rec = rec[:, :, :1]
    return rec.astype(np.float32) / 255.0


def _load_tree(root: Path) -> dict[tuple[str, int], np.ndarray]:
    from bench.score_mips import _index, _load

    return {key: _load(path) for key, path in _index(root).items()}


def _write_role(dest: Path, role: str, mip: int, image: np.ndarray) -> None:
    if mip == 0:
        path = dest / f"{role}.png"
    else:
        path = dest / "mips" / f"{role}.{mip:02d}.png"
    _save_u8(image, path)


def _score_codec_tree(
    reference: Path,
    codec: str,
    roles: list[str] | None,
    name: str,
) -> dict:
    from bench.score_mips import score_tree

    tree = _load_tree(reference)
    present = sorted({role for role, _mip in tree})
    use = [r for r in present if roles is None or r in roles]
    if codec == "bc5":
        use = [r for r in use if r == "normal"]
    if not use:
        return {"name": name, "codec": codec, "psnr_u8": None, "reason": "missing roles"}
    bpp = {"bc1": 4.0, "bc5": 8.0, "bc7": 8.0}[codec]
    decoded_root = Path(tempfile.mkdtemp(prefix="ntc-rd-"))
    try:
        bits = 0.0
        channels = 0
        counted = set()
        for (role, mip), img in sorted(tree.items()):
            if role not in use:
                continue
            rec = _transcode_role(img, codec)
            _write_role(decoded_root, role, mip, rec)
            bits += bpp * img.shape[0] * img.shape[1]
            if role not in counted:
                channels += img.shape[2]
                counted.add(role)
        scored = score_tree(reference, decoded_root)
    finally:
        shutil.rmtree(decoded_root, ignore_errors=True)
    one = {}
    for (role, mip), img in tree.items():
        if role in use and mip not in one:
            one[mip] = img.shape[0] * img.shape[1]
    ntex = max(len(use), 1)
    return {
        "name": name,
        "codec": codec,
        "tool": "bc_pack.py",
        "groups": sorted(use),
        "psnr_u8": scored["psnr_u8"],
        "mips": scored["mips"],
        "codec_bpp": bpp,
        "textures": ntex,
        "file_bpp_chain": bpp * ntex,
        "bppc": (bpp * ntex) / float(max(channels, 1)),
        "latent_bpp_mip0": None,
        "note": "nominal codec bits per mip texel, summed over the textures in this row. Block codecs have no latent split.",
        "chain_texels": int(sum(one.values())),
        "payload_bits": bits,
    }


def _find_astcenc() -> str | None:
    for name in ("astcenc-avx2", "astcenc-sse2", "astcenc"):
        found = shutil.which(name)
        if found:
            return found
    root = Path("tools/astcenc")
    if root.is_dir():
        for path in root.rglob("astcenc*"):
            if path.is_file() and os.access(path, os.X_OK) and path.suffix != ".txt":
                return str(path)
    return None


def _score_astc(reference: Path) -> list[dict]:
    from bench.score_mips import score_tree

    exe = _find_astcenc()
    tree = _load_tree(reference)
    present = sorted({role for role, _mip in tree})
    rows = []
    for label, (_bw, _bh, bpp) in ASTC_BLOCKS.items():
        if exe is None:
            channels = 0
            counted = set()
            for (role, _mip), img in tree.items():
                if role not in counted:
                    channels += img.shape[2]
                    counted.add(role)
            rows.append(
                {
                    "name": f"astc_{label}",
                    "codec": "astc",
                    "block": label,
                    "groups": present,
                    "psnr_u8": None,
                    "codec_bpp": bpp,
                    "textures": len(present),
                    "file_bpp_chain": bpp * max(len(present), 1),
                    "bppc": (bpp * max(len(present), 1)) / float(max(channels, 1)),
                    "latent_bpp_mip0": None,
                    "note": "astcenc is not on PATH. bpp anchor only.",
                }
            )
            continue
        work = Path(tempfile.mkdtemp(prefix="ntc-astc-"))
        decoded = work / "dec"
        decoded.mkdir()
        try:
            ok = True
            for (role, mip), _img in sorted(tree.items()):
                if mip == 0:
                    src = reference / f"{role}.png"
                else:
                    src = reference / "mips" / f"{role}.{mip:02d}.png"
                astc = work / f"{role}_{mip}.astc"
                out_png = decoded / (f"{role}.png" if mip == 0 else f"mips/{role}.{mip:02d}.png")
                out_png.parent.mkdir(parents=True, exist_ok=True)
                enc = subprocess.run(
                    [exe, "-cl", str(src), str(astc), label, "-medium"],
                    capture_output=True,
                    text=True,
                )
                dec = subprocess.run(
                    [exe, "-dl", str(astc), str(out_png)],
                    capture_output=True,
                    text=True,
                )
                if enc.returncode != 0 or dec.returncode != 0 or not out_png.is_file():
                    ok = False
                    break
            channels = 0
            counted = set()
            for (role, _mip), img in tree.items():
                if role not in counted:
                    channels += img.shape[2]
                    counted.add(role)
            row = {
                "name": f"astc_{label}",
                "codec": "astc",
                "block": label,
                "tool": exe,
                "groups": present,
                "codec_bpp": bpp,
                "textures": len(present),
                "file_bpp_chain": bpp * max(len(present), 1),
                "bppc": (bpp * max(len(present), 1)) / float(max(channels, 1)),
                "latent_bpp_mip0": None,
            }
            if ok:
                scored = score_tree(reference, decoded)
                row["psnr_u8"] = scored["psnr_u8"]
                row["mips"] = scored["mips"]
            else:
                row["psnr_u8"] = None
                row["note"] = "astcenc failed. bpp anchor only."
            rows.append(row)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return rows


def block_rows(reference: Path) -> list[dict]:
    tree = _load_tree(reference)
    present = sorted({role for role, _mip in tree})
    rows = []
    if "albedo" in present:
        rows.append(_score_codec_tree(reference, "bc1", ["albedo"], "bc1_albedo"))
        rows.append(_score_codec_tree(reference, "bc7", ["albedo"], "bc7_albedo"))
    if "normal" in present:
        rows.append(_score_codec_tree(reference, "bc5", ["normal"], "bc5_normal"))
        rows.append(_score_codec_tree(reference, "bc7", ["normal"], "bc7_normal"))
    if all(r in present for r in ("roughness", "metalness", "ao")):
        rows.append(_score_codec_tree(reference, "bc1", ["roughness", "metalness", "ao"], "bc1_orm"))
        rows.append(_score_codec_tree(reference, "bc7", ["roughness", "metalness", "ao"], "bc7_orm"))
    if present:
        rows.append(_score_codec_tree(reference, "bc7", present, "bc7_textures"))
    rows.extend(_score_astc(reference))
    rows.extend(_bundle_16(reference, present))
    return rows


def _bundle_16(reference: Path, present: list[str]) -> list[dict]:
    """BC1 albedo + BC5 normal + one BC1 of roughness/metalness/ao, when all exist."""
    from bench.score_mips import score_tree

    need = ["albedo", "normal", "roughness", "metalness", "ao"]
    if any(r not in present for r in need):
        return []
    tree = _load_tree(reference)
    decoded = Path(tempfile.mkdtemp(prefix="ntc-bundle-"))
    try:
        for (role, mip), img in tree.items():
            if role == "albedo":
                rec = _transcode_role(img, "bc1")
            elif role == "normal":
                rec = _transcode_role(img, "bc5")
            elif role in ("roughness", "metalness", "ao"):
                continue
            else:
                rec = img
            if role in ("albedo", "normal"):
                _write_role(decoded, role, mip, rec)
        # Pack the three scalars as one RGB image per mip, then split back.
        mips = sorted({mip for role, mip in tree if role == "roughness"})
        for mip in mips:
            r = tree[("roughness", mip)][:, :, 0]
            m = tree[("metalness", mip)][:, :, 0]
            a = tree[("ao", mip)][:, :, 0]
            rgb = np.stack([r, m, a], axis=2)
            rec = _transcode_role(rgb, "bc1")
            _write_role(decoded, "roughness", mip, rec[:, :, 0:1])
            _write_role(decoded, "metalness", mip, rec[:, :, 1:2])
            _write_role(decoded, "ao", mip, rec[:, :, 2:3])
        scored = score_tree(reference, decoded)
    finally:
        shutil.rmtree(decoded, ignore_errors=True)
    channels = 3 + 3 + 1 + 1 + 1
    # 4 (albedo) + 8 (normal) + 4 (orm) = 16 bits per mip texel.
    return [
        {
            "name": "bc1_bc5_bc1",
            "codec": "bundle",
            "tool": "bc_pack.py",
            "groups": need,
            "psnr_u8": scored["psnr_u8"],
            "mips": scored["mips"],
            "codec_bpp": 16.0,
            "textures": 3,
            "file_bpp_chain": 16.0,
            "bppc": 16.0 / float(channels),
            "latent_bpp_mip0": None,
            "note": "BC1 albedo, BC5 normal with Z reconstructed from XY, BC1 of roughness/metalness/ao. Nominal 16 bits per mip texel.",
        }
    ]


def _ours_rows(
    chw: np.ndarray,
    semantics: list[int],
    reference: Path,
    rates: list[float],
    steps: int,
    rd_lambdas: list[float],
    latent_coder: str,
    rate_mode: str,
    hidden: tuple,
) -> list[dict]:
    from bench.score_mips import score_tree
    from ntc.encode import TrainConfig, encode_arrays

    rows = []
    for lam in rd_lambdas:
        for bpp in rates:
            cfg = TrainConfig(
                steps=int(steps),
                hidden=hidden,
                target_bpp=float(bpp),
                rate=rate_mode,
                adaptive_pool=1,
                seed_pool=1,
                with_ssim=False,
                latent_coder=latent_coder,
                latent_deflate=latent_coder != "raw",
                rd_lambda=float(lam),
                mip_levels=12,
                device="cpu",
                log_every=10**9,
            )
            blob, stats = encode_arrays(chw, semantics, cfg)
            dest = Path(tempfile.mkdtemp(prefix="ntc-ours-"))
            try:
                _write_ntc_chain(blob, semantics, dest)
                scored = score_tree(reference, dest)
            finally:
                shutil.rmtree(dest, ignore_errors=True)
            rows.append(
                {
                    "series": f"ours_lambda_{lam:g}",
                    "tool": "ours",
                    "requested_bpp": float(bpp),
                    "rate_mode": rate_mode,
                    "rd_lambda": float(lam),
                    "latent_coder": stats.get("latent_coder"),
                    "psnr_u8": scored["psnr_u8"],
                    "groups": sorted(scored["groups"]),
                    "mips": scored["mips"],
                    "latent_bpp_mip0": stats["latent_bpp_mip0"],
                    "file_bpp_chain": stats["file_bpp_chain"],
                    "bppc": stats["bppc"],
                    "total_bpp": stats["total_bpp"],
                    "header_bytes": stats["header_bytes"],
                    "network_bytes": stats["network_bytes"],
                    "file_bytes": stats["file_bytes"],
                    "grids": stats["grids"],
                }
            )
            print(
                f"ours bpp {bpp:g} lambda {lam:g} psnr {scored['psnr_u8']:.3f} "
                f"latent {stats['latent_bpp_mip0']:.3f} file {stats['file_bpp_chain']:.3f} "
                f"bppc {stats['bppc']:.3f} net {stats['network_bytes']}",
                flush=True,
            )
    return rows


def _write_ntc_chain(blob: bytes, semantics: list[int], dest: Path) -> None:
    from ntc.container import parse_texture_set
    from ntc.decode_ref import decode_mip

    ts = parse_texture_set(blob)
    for mip in range(int(ts.num_mips)):
        img = decode_mip(ts, mip)
        planes: dict[str, list] = {}
        for channel, sem in enumerate(semantics):
            role = _ROLE.get(int(sem))
            if role is None:
                continue
            name, slot = role
            planes.setdefault(name, [None, None, None])[slot] = img[:, :, channel]
        for name, slots in planes.items():
            used = [s for s in slots if s is not None]
            _write_role(dest, name, mip, np.stack(used, axis=2))


def _load_rtx(root: Path) -> list[dict]:
    rows = []
    if not root.is_dir():
        return rows
    for path in sorted(root.glob("*/report.json")):
        report = json.loads(path.read_text())
        for row in report.get("rtxntc") or []:
            if row.get("psnr_u8") is None or row.get("error"):
                continue
            item = {
                "tool": "rtxntc",
                "source": str(path),
                "material": row.get("material"),
                "psnr_u8": row.get("psnr_u8"),
                "latent_bpp_mip0": row.get("latent_bpp_mip0"),
                "file_bpp_chain": row.get("file_bpp_chain"),
                "bppc": row.get("bppc"),
                "groups": row.get("groups"),
                "note": "Measured by bench/rtxntc_compare. latent_bpp_mip0 is the requested --bitsPerPixel.",
            }
            rows.append(item)
    return rows


def _svg(path: Path, ours: list[dict], blocks: list[dict], rtx: list[dict]) -> None:
    def series_points(rows, key):
        pts = []
        for row in rows:
            if row.get(key) is None or row.get("psnr_u8") is None:
                continue
            pts.append((float(row[key]), float(row["psnr_u8"])))
        pts.sort()
        return pts

    def panel(x0, title, key, xmin, xmax, extra):
        groups = [("ours", "#1d4ed8", series_points(ours, key))]
        # One polyline per block name would crowd the chart. Plot each block
        # row as its own marker, and ASTC as one polyline when PSNR exists.
        astc = [r for r in blocks if str(r.get("name", "")).startswith("astc_") and r.get("psnr_u8") is not None]
        other = [r for r in blocks if not str(r.get("name", "")).startswith("astc_") and r.get(key) is not None and r.get("psnr_u8") is not None]
        groups.append(("astc", "#15803d", series_points(astc, key)))
        groups.append(("rtxntc", "#b45309", series_points(rtx, key)))
        xs = [p[0] for _n, _c, pts in groups for p in pts]
        ys = [p[1] for _n, _c, pts in groups for p in pts]
        xs += [p[0] for p in extra]
        ys += [p[1] for p in extra]
        for row in other:
            xs.append(float(row[key]))
            ys.append(float(row["psnr_u8"]))
        if xmin is None:
            xmin = min(xs) if xs else 0.0
        if xmax is None:
            xmax = max(xs) if xs else 1.0
        if xs:
            xmin = min(xmin, min(xs))
            xmax = max(xmax, max(xs))
        if not ys:
            ys = [30.0, 50.0]
        ymin, ymax = min(ys), max(ys)
        if xmax <= xmin:
            xmax = xmin + 1.0
        if ymax <= ymin:
            ymax = ymin + 1.0
        pad_y = (ymax - ymin) * 0.08
        ymin -= pad_y
        ymax += pad_y

        def xy(x, y):
            px = x0 + 48 + (x - xmin) / (xmax - xmin) * 300
            py = 36 + (1.0 - (y - ymin) / (ymax - ymin)) * 200
            return px, py

        parts = [f'<text x="{x0 + 48}" y="18" font-size="12">{title}</text>']
        for _name, color, pts in groups:
            if len(pts) >= 2:
                d = " ".join(f"{xy(x, y)[0]:.1f},{xy(x, y)[1]:.1f}" for x, y in pts)
                parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.4" points="{d}"/>')
            for x, y in pts:
                px, py = xy(x, y)
                parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.6" fill="{color}"/>')
        for row in other:
            px, py = xy(float(row[key]), float(row["psnr_u8"]))
            parts.append(f'<rect x="{px - 2.5:.1f}" y="{py - 2.5:.1f}" width="5" height="5" fill="#6b7280"/>')
        for x, y in extra:
            px, py = xy(x, y)
            parts.append(f'<rect x="{px - 3:.1f}" y="{py - 3:.1f}" width="6" height="6" fill="#111827"/>')
        parts.append(
            f'<text x="{x0 + 48}" y="252" font-size="10">x {xmin:.2f}..{xmax:.2f}  y {ymin + pad_y:.1f}..{ymax - pad_y:.1f} dB</text>'
        )
        return "".join(parts)

    body = panel(0, "latent_bpp_mip0", "latent_bpp_mip0", 2.0, 20.0, [])
    body += panel(360, "file_bpp_chain", "file_bpp_chain", 2.0, 20.0, [])
    body += panel(720, "bppc", "bppc", None, None, list(PAPER_TABLE3))
    note = (
        "blue ours, green ASTC when measured, orange RTXNTC measured, gray blocks, "
        "black squares Table 3 (different material set). "
        "BD-rate against Table 3 is omitted."
    )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="300" font-family="sans-serif">'
        f"{body}<text x=\"16\" y=\"286\" font-size=\"10\">{note}</text></svg>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg)


def _demo_image(size: int) -> tuple[np.ndarray, list[int]]:
    ys = np.linspace(0.05, 0.95, size, dtype=np.float32)
    xs = np.linspace(0.0, 1.0, size, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    albedo = np.stack(
        [
            grid_x,
            grid_y,
            0.5 * (np.sin(grid_x * 6.0) * 0.5 + 0.5 + grid_y) * 0.5,
        ],
        axis=0,
    ).astype(np.float32)
    return np.clip(albedo, 0.0, 1.0), [0, 1, 2]


def build_report(args: argparse.Namespace) -> dict:
    rtx = _load_rtx(args.rtx_dir)
    with tempfile.TemporaryDirectory(prefix="ntc-rd-ref-") as tmp:
        ref = Path(tmp) / "ref"
        if args.demo:
            chw, sems = _demo_image(args.size)
            write_chain(ref, chw, sems)
            scope = f"cpu_proof_{args.size}"
            sdk = False
        elif args.material:
            from ntc.materials import export_reference_mips, load_material

            export_reference_mips(args.material, ref, args.size)
            chw, sems, _info = load_material(args.material, args.size)
            scope = f"material_{args.size}"
            sdk = args.size == 2048
        else:
            chw, sems = None, None
            scope = "anchors_only"
            sdk = False
            # Paper, empty measured RTX, and no block PSNR without an image.
            ref = None
        blocks: list[dict] = []
        ours: list[dict] = []
        if ref is not None:
            blocks = block_rows(ref)
            if args.encode or args.demo:
                hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip())
                ours = _ours_rows(
                    chw,
                    sems,
                    ref,
                    list(args.rates),
                    args.steps,
                    list(args.rd_lambdas),
                    args.latent_coder,
                    args.rate,
                    hidden,
                )
    report = {
        "scope": scope,
        "sdk_2048_measured": sdk and bool(ours),
        "note": (
            "2048 SDK materials are measured only when this script is run with "
            "--material on those directories at --size 2048. A demo run is a "
            "32-wide CPU check of the scorer, the block codecs, and the encoder options."
        ),
        "definitions": {
            "latent_bpp_mip0": "stored latent codes + fp16 latent scales, per mip 0 texel. Block codecs leave this null.",
            "file_bpp_chain": "whole NTC file, or nominal block bits, per mip-chain texel",
            "bppc": "NTC payload bits, or nominal block bits, per mip-chain texel per channel",
            "header_bytes": 96,
            "network_bytes": "int8 or int4 weights plus fp16 bias and fp16 weight scales",
        },
        "paper": {"note": PAPER_NOTE, "table3_bppc_db": list(PAPER_TABLE3)},
        "bd_rate_method": (
            "bits_saved_percent is positive when the candidate uses fewer bits at equal PSNR. "
            "Overlapping curves use a piecewise-linear integral of ln(rate) over PSNR. "
            "A one-point baseline interpolates the candidate."
        ),
        "ours": ours,
        "blocks": blocks,
        "rtxntc_measured": rtx,
        "bd_rate": bd_rate_rows(ours, blocks, rtx),
    }
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="RD curves and BD-rate from 2 to 20 bpp")
    p.add_argument("--material", type=Path, default=None)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--rates", type=float, nargs="+", default=list(RATES))
    p.add_argument("--rd-lambdas", type=float, nargs="+", default=[0.0])
    p.add_argument("--latent-coder", default="best", choices=("deflate", "range", "best", "context", "best_all", "raw"))
    p.add_argument("--rate", default="latent", choices=("latent", "payload"))
    p.add_argument("--hidden", default="32")
    p.add_argument("--encode", action="store_true")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--rtx-dir", type=Path, default=Path("results/rtxntc_compare"))
    p.add_argument("--out", type=Path, default=Path("results/rd_curve.json"))
    args = p.parse_args()
    report = build_report(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    svg = args.out.with_suffix(".svg")
    _svg(svg, report["ours"], report["blocks"], report["rtxntc_measured"])
    print("wrote", args.out)
    print("wrote", svg)
    for row in report["bd_rate"]:
        if row["bits_saved_percent"] is None:
            print(f"bd {row['baseline']} {row['axis']}: {row['reason']}")
        else:
            print(
                f"bd {row['series']} vs {row['baseline']} {row['axis']}: "
                f"{row['bits_saved_percent']:.2f}% bits saved ({row['method']})"
            )


if __name__ == "__main__":
    main()
