"""BC1/BC5/BC7 and ASTC baselines scored with the same uint8 PSNR as NTC.

Whole-material bpp is the sum of the per-texture rates. A 3-texture BC7
bundle is 24 bpp. The tight bundle is BC1 albedo (4) + BC5 normal (8) +
BC1 roughness/metal/ao packed into one texture (4) = 16 bpp.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from ntc.materials import load_material
from ntc.metrics import group_metrics


def _png(path: Path, arr: np.ndarray) -> None:
    img = (np.clip(arr, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    if img.shape[2] == 1:
        Image.fromarray(img[:, :, 0], mode="L").save(path)
    else:
        Image.fromarray(img, mode="RGB").save(path)


def _load_png(path: Path, channels: int) -> np.ndarray:
    img = Image.open(path)
    if channels == 1:
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return arr[:, :, :channels]


def _run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-500:] or r.stdout[-500:] or " ".join(cmd))


def score_material(
    material: Path,
    size: int,
    bc7enc: Path,
    astcenc: Path,
    work: Path,
) -> dict:
    images, sems, info = load_material(material, size)
    hwc = np.transpose(images, (1, 2, 0))
    slices = info["slices"]
    work.mkdir(parents=True, exist_ok=True)
    roles = {}
    for role, (a, b) in slices.items():
        plane = hwc[:, :, a:b]
        if plane.shape[2] == 2:
            # BC tools want 3 channels for BC5 source; pad B with 0
            plane = np.concatenate([plane, np.zeros_like(plane[:, :, :1])], axis=2)
        path = work / f"{role}.png"
        _png(path, plane if plane.shape[2] != 1 else plane)
        roles[role] = {"path": str(path), "channels": b - a, "slice": [a, b]}

    # ORM pack: roughness, metalness (or 0), ao
    h, w, _c = hwc.shape
    orm = np.zeros((h, w, 3), np.float32)
    if "roughness" in slices:
        a, b = slices["roughness"]
        orm[:, :, 0] = hwc[:, :, a]
    if "metalness" in slices:
        a, b = slices["metalness"]
        orm[:, :, 1] = hwc[:, :, a]
    if "ao" in slices:
        a, b = slices["ao"]
        orm[:, :, 2] = hwc[:, :, a]
    _png(work / "orm.png", orm)

    results = {"size": size, "semantics": sems, "codecs": []}

    def add(name: str, bpp: float, recon: np.ndarray, note: str) -> None:
        m = group_metrics(recon, hwc, sems)
        m["name"] = name
        m["bpp"] = bpp
        m["note"] = note
        results["codecs"].append(m)

    def bc(flag: str, src: Path, tag: str, quality: str) -> Path:
        out = work / f"{tag}.png"
        cmd = [str(bc7enc), flag, quality, "-l", str(src), str(work / f"{tag}.dds"), str(out)]
        # -l is BC7 only. Drop it for BC1/4/5.
        if flag != "-1" and flag != "":
            # bc7enc rejects -l on BC1. The default BC7 call uses -l.
            pass
        if flag in ("-1", "-3", "-4", "-5"):
            cmd = [str(bc7enc), flag, quality, str(src), str(work / f"{tag}.dds"), str(out)]
        else:
            cmd = [str(bc7enc), "-l", quality, str(src), str(work / f"{tag}.dds"), str(out)]
        _run(cmd)
        return out

    # Per-texture BC7 (8 bpp each) and the bundle sum.
    recon = np.zeros_like(hwc)
    bpp = 0.0
    for role, meta in roles.items():
        ch = meta["channels"]
        src = Path(meta["path"])
        if ch == 1:
            src_img = _load_png(src, 1)
            # encode as RGB with the value in R so BC7 has a defined input
            rgb = np.repeat(src_img, 3, axis=2)
            rgb_path = work / f"{role}_rgb.png"
            _png(rgb_path, rgb)
            unpacked = bc("", rgb_path, f"{role}_bc7", "-u4")
            got = _load_png(unpacked, 3)[:, :, :1]
            a, b = meta["slice"]
            recon[:, :, a:b] = got
            bpp += 8.0
        elif ch == 3:
            unpacked = bc("", src, f"{role}_bc7", "-u4")
            got = _load_png(unpacked, 3)
            a, b = meta["slice"]
            recon[:, :, a:b] = got
            bpp += 8.0
    add("bc7_per_texture", bpp, recon, "bc7enc -l -u4, 8 bpp per texture, linear metrics")

    # Tight bundle: BC1 albedo, BC5 normal, BC1 ORM.
    bundle = np.zeros_like(hwc)
    bundle_bpp = 0.0
    if "albedo" in roles:
        unpacked = bc("-1", Path(roles["albedo"]["path"]), "albedo_bc1", "-u5")
        a, b = slices["albedo"]
        bundle[:, :, a:b] = _load_png(unpacked, 3)
        bundle_bpp += 4.0
    if "normal" in roles:
        unpacked = bc("-5", Path(roles["normal"]["path"]), "normal_bc5", "-u5")
        a, b = slices["normal"]
        got = _load_png(unpacked, 3)
        bundle[:, :, a : a + 2] = got[:, :, :2]
        # reconstruct z from xy so the third channel is not left at 0
        xy = bundle[:, :, a : a + 2] * 2.0 - 1.0
        z2 = np.clip(1.0 - xy[:, :, 0] ** 2 - xy[:, :, 1] ** 2, 0.0, 1.0)
        z = (np.sqrt(z2) + 1.0) * 0.5
        bundle[:, :, a + 2] = z
        bundle_bpp += 8.0
    if (work / "orm.png").is_file() and any(k in slices for k in ("roughness", "metalness", "ao")):
        unpacked = bc("-1", work / "orm.png", "orm_bc1", "-u5")
        got = _load_png(unpacked, 3)
        for i, role in enumerate(("roughness", "metalness", "ao")):
            if role in slices:
                a, _b = slices[role]
                bundle[:, :, a] = got[:, :, i]
        bundle_bpp += 4.0
    add("bc1_albedo_bc5_normal_bc1_orm", bundle_bpp, bundle, "bc7enc -1/-5 -u5, z reconstructed from xy")

    # ASTC per texture, 6x6 (3.555... bpp) and 4x4 (8 bpp).
    def astc_one(src: Path, block: str, tag: str) -> Path:
        enc = work / f"{tag}.astc"
        dec = work / f"{tag}_astc.png"
        _run([str(astcenc), "-cl", str(src), str(enc), block, "-medium"])
        _run([str(astcenc), "-dl", str(enc), str(dec)])
        return dec

    for block, rate in (("4x4", 8.0), ("6x6", 128.0 / (6 * 6)), ("8x8", 2.0)):
        recon = np.zeros_like(hwc)
        bpp = 0.0
        for role, meta in roles.items():
            ch = meta["channels"]
            src = Path(meta["path"])
            if ch == 1:
                rgb = np.repeat(_load_png(src, 1), 3, axis=2)
                rgb_path = work / f"{role}_rgb.png"
                _png(rgb_path, rgb)
                dec = astc_one(rgb_path, block, f"{role}_{block}")
                got = _load_png(dec, 3)[:, :, :1]
            else:
                dec = astc_one(src, block, f"{role}_{block}")
                got = _load_png(dec, ch)
            a, b = meta["slice"]
            recon[:, :, a:b] = got[:, :, : b - a]
            bpp += rate
        add(f"astc_{block}_per_texture", bpp, recon, "astcenc -cl -medium, LDR linear")

    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("material", type=Path)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--bc7enc", type=Path, default=Path("/tmp/bc7-build/bc7enc"))
    p.add_argument("--astcenc", type=Path, default=Path("/tmp/astc-build/Source/astcenc-native"))
    p.add_argument("--work", type=Path, default=Path("/tmp/ntc-bc"))
    p.add_argument("--out", type=Path, default=Path("results/blocks.json"))
    args = p.parse_args()
    res = score_material(args.material, args.size, args.bc7enc, args.astcenc, args.work)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    for row in res["codecs"]:
        print(f"{row['name']} bpp {row['bpp']:.3f} psnr_u8 {row['psnr_u8']:.3f} ssim {row['ssim']:.4f}")


if __name__ == "__main__":
    main()
