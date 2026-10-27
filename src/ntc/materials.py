"""Load a PBR material directory.

If ``manifest.json`` exists it maps semantics to files:

    {"albedo": "color.jpg", "normal": "normal.jpg",
     "roughness": "rough.jpg", "metalness": "metal.jpg", "ao": "ao.jpg"}

Otherwise the loader looks for ``albedo.png``, ``normal.png``,
``roughness.png``, ``metalness.png``, ``ao.png``, ``height.png``
with ``.png``, ``.jpg``, or ``.jpeg``.

Values are the file's 8-bit levels divided by 255. No gamma is applied.
Albedo is three channels. Normal is three channels, stored as authored
(OpenGL Y+ when the source is ``nor_gl`` or ``NormalGL``). Missing maps
are omitted. Odd sizes are cropped by one pixel so mips stay even.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

_CANON = [
    ("albedo", "albedo"),
    ("normal", "normal"),
    ("roughness", "roughness"),
    ("metalness", "metalness"),
    ("ao", "ao"),
    ("height", "height"),
]

_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _find(directory: Path, stem: str) -> Path | None:
    for ext in _EXTS:
        p = directory / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def _to_float(img: Image.Image, channels: int, size: int | None) -> np.ndarray:
    if channels == 1:
        img = img.convert("L")
    elif channels == 3:
        img = img.convert("RGB")
    else:
        raise ValueError(channels)
    if size is not None and (img.width != size or img.height != size):
        img = img.resize((size, size), Image.Resampling.BOX)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[:, :, None]
    h, w, _c = arr.shape
    if h % 2:
        arr = arr[:-1, :, :]
    if w % 2:
        arr = arr[:, :-1, :]
    return arr


def load_material(directory: str | Path, size: int | None = None) -> tuple[np.ndarray, list[int], dict]:
    """Returns ``images [C,H,W]``, semantic ids, and a small info dict."""
    from ntc.container import (
        SEM_ALBEDO_B,
        SEM_ALBEDO_G,
        SEM_ALBEDO_R,
        SEM_AO,
        SEM_HEIGHT,
        SEM_METALNESS,
        SEM_NORMAL_X,
        SEM_NORMAL_Y,
        SEM_NORMAL_Z,
        SEM_ROUGHNESS,
    )

    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    roles: dict[str, Path] = {}
    if manifest_path.is_file():
        spec = json.loads(manifest_path.read_text())
        for key in ("albedo", "normal", "roughness", "metalness", "ao", "height"):
            if key in spec and spec[key]:
                roles[key] = directory / spec[key]
    else:
        for key, stem in _CANON:
            found = _find(directory, stem)
            if found is not None:
                roles[key] = found
    if "albedo" not in roles:
        raise FileNotFoundError(f"no albedo in {directory}")

    planes = []
    sems: list[int] = []
    role_slices = {}
    ref_hw = None
    order = ["albedo", "normal", "roughness", "metalness", "ao", "height"]
    role_sem = {
        "albedo": (SEM_ALBEDO_R, SEM_ALBEDO_G, SEM_ALBEDO_B),
        "normal": (SEM_NORMAL_X, SEM_NORMAL_Y, SEM_NORMAL_Z),
        "roughness": (SEM_ROUGHNESS,),
        "metalness": (SEM_METALNESS,),
        "ao": (SEM_AO,),
        "height": (SEM_HEIGHT,),
    }
    role_ch = {k: len(v) for k, v in role_sem.items()}
    for key in order:
        if key not in roles:
            continue
        arr = _to_float(Image.open(roles[key]), role_ch[key], size)
        if ref_hw is None:
            ref_hw = arr.shape[:2]
        elif arr.shape[:2] != ref_hw:
            img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8).squeeze())
            # already resized; force exact match with box if manifest files differ
            if arr.shape[2] == 1:
                img = Image.fromarray((arr[:, :, 0] * 255).astype(np.uint8), mode="L")
            else:
                img = Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")
            img = img.resize((ref_hw[1], ref_hw[0]), Image.Resampling.BOX)
            arr = _to_float(img, role_ch[key], None)
        start = len(sems)
        for s in role_sem[key]:
            sems.append(s)
        planes.append(arr)
        role_slices[key] = [start, start + role_ch[key]]
    stacked = np.concatenate(planes, axis=2)
    chw = np.transpose(stacked, (2, 0, 1)).astype(np.float32)
    info = {
        "height": chw.shape[1],
        "width": chw.shape[2],
        "roles": {k: str(v) for k, v in roles.items()},
        "slices": role_slices,
        "semantics": sems,
    }
    return chw, sems, info
