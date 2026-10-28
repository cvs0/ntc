"""Score a decoded mip chain against a reference mip chain.

Both trees use the same names. Mip 0 is ``<role>.png`` (or ``.jpg``,
``.tga``, ``.exr``). Mips 1 and up are ``mips/<role>.NN.ext`` with NN
starting at 01. That is the layout ``ntc-cli --loadMips`` and
``--saveMips`` use, and the layout ``export_reference_mips`` writes.

PSNR is one number from the pooled 8-bit MSE. Every scored texel of
every channel of every mip that exists on both sides goes into the same
sum. Per-channel-group PSNR is the same pool restricted to that group.
Groups are albedo, normal, roughness, metalness, ao, height. A missing
mip on one side is listed and omitted. It is not filled in.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from ntc.metrics import to_u8_unit

_EXTS = (".png", ".jpg", ".jpeg", ".tga", ".tif", ".tiff", ".exr")
_GROUPS = {
    "albedo": ("albedo", "color", "diffuse", "basecolor", "base_color"),
    "normal": ("normal", "normalgl", "nor", "normaldx"),
    "roughness": ("roughness", "rough"),
    "metalness": ("metalness", "metal", "metallic"),
    "ao": ("ao", "ambientocclusion", "occlusion"),
    "height": ("height", "displacement", "disp"),
}


def _group(stem: str) -> str | None:
    key = re.sub(r"[^a-z0-9]", "", stem.lower())
    for name, aliases in _GROUPS.items():
        if key in aliases:
            return name
    return None


def _load(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".exr":
        try:
            import imageio.v3 as iio

            arr = np.asarray(iio.imread(path), dtype=np.float32)
        except ImportError as exc:
            raise SystemExit("EXR needs imageio (pip install imageio)") from exc
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.shape[2] > 3:
            arr = arr[:, :, :3]
        return np.clip(arr, 0.0, 1.0)
    img = Image.open(path)
    if img.mode in ("L", "I", "F"):
        arr = np.asarray(img.convert("L"), dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return arr[:, :, None]
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return arr


def _index(root: Path) -> dict[tuple[str, int], Path]:
    found: dict[tuple[str, int], Path] = {}
    if not root.is_dir():
        return found
    for path in root.iterdir():
        if not path.is_file() or path.suffix.lower() not in _EXTS:
            continue
        group = _group(path.stem)
        if group is None:
            continue
        found[(group, 0)] = path
    mips = root / "mips"
    if mips.is_dir():
        for path in mips.iterdir():
            if not path.is_file() or path.suffix.lower() not in _EXTS:
                continue
            match = re.match(r"(.+)\.(\d+)$", path.stem)
            if not match:
                continue
            group = _group(match.group(1))
            if group is None:
                continue
            found[(group, int(match.group(2)))] = path
    return found


def _psnr_from_mse(mse: float) -> float:
    if mse <= 1e-20:
        return 100.0
    return float(min(100.0, 10.0 * np.log10(1.0 / mse)))


def score_tree(reference: Path, decoded: Path) -> dict:
    ref = _index(reference)
    dec = _index(decoded)
    keys = sorted(set(ref) & set(dec))
    missing = sorted(set(ref) - set(dec))
    extra = sorted(set(dec) - set(ref))
    grouped: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    all_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    mips = sorted({k[1] for k in keys})
    for group, mip in keys:
        pred = _load(dec[(group, mip)])
        truth = _load(ref[(group, mip)])
        if pred.shape != truth.shape:
            raise ValueError(f"{group} mip {mip}: {pred.shape} != {truth.shape}")
        grouped.setdefault(group, []).append((pred, truth))
        all_pairs.append((pred, truth))
    se = 0.0
    n = 0
    for pred, truth in all_pairs:
        d = to_u8_unit(pred) - to_u8_unit(truth)
        se += float(np.sum(np.square(d.astype(np.float64))))
        n += int(d.size)
    mse = se / float(n) if n else 0.0
    out = {
        "psnr_u8": None if n == 0 else _psnr_from_mse(mse),
        "mse_u8": None if n == 0 else mse,
        "texels": n,
        "mips": mips,
        "groups": {},
        "missing": [f"{g}.m{m:02d}" for g, m in missing],
        "extra": [f"{g}.m{m:02d}" for g, m in extra],
    }
    for group, pairs in grouped.items():
        gse = 0.0
        gn = 0
        for pred, truth in pairs:
            d = to_u8_unit(pred) - to_u8_unit(truth)
            gse += float(np.sum(np.square(d.astype(np.float64))))
            gn += int(d.size)
        gmse = gse / float(gn)
        out["groups"][group] = {"psnr_u8": _psnr_from_mse(gmse), "mse_u8": gmse, "texels": gn}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="PSNR of two mip-chain directories")
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--decoded", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    report = score_tree(args.reference, args.decoded)
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
