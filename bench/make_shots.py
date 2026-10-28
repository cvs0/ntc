"""Assemble PNGs from GPU readbacks.

Reference panels are the source image. NTC, BC, and lit panels are the float
buffers written by ``ntc_vk`` / ``ntc_gl``. Missing buffers are skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _load_f32(path: Path, w: int, h: int, c: int) -> np.ndarray | None:
    if not path.is_file():
        return None
    raw = np.fromfile(path, dtype=np.float32)
    need = w * h * c
    if raw.size < need:
        return None
    return raw[:need].reshape(h, w, c)


def _u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.clip(x, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)


def _group(img: np.ndarray, idx: list[int]) -> np.ndarray:
    cols = []
    for i in idx:
        if i < img.shape[2]:
            cols.append(img[:, :, i])
        else:
            cols.append(np.zeros(img.shape[:2], dtype=np.float32))
    out = np.stack(cols, axis=2)
    if out.shape[2] == 1:
        out = np.repeat(out, 3, axis=2)
    if out.shape[2] == 2:
        z = np.zeros(out.shape[:2] + (1,), dtype=np.float32)
        out = np.concatenate([out, z], axis=2)
    return out[:, :, :3]


def _heat(err: np.ndarray) -> np.ndarray:
    t = np.clip(err / max(float(np.percentile(err, 99)) if err.size else 1.0, 1e-6), 0, 1)
    rgb = np.zeros(t.shape + (3,), dtype=np.float32)
    rgb[..., 0] = np.clip(1.5 * t, 0, 1)
    rgb[..., 1] = np.clip(1.5 * (1.0 - np.abs(t - 0.5) * 2.0), 0, 1) * 0.2
    rgb[..., 2] = np.clip(1.5 * (1.0 - t), 0, 1)
    return _u8(rgb)


def _crop(img: np.ndarray, scale: int = 4) -> np.ndarray:
    h, w = img.shape[:2]
    y0, x0 = h // 4, w // 4
    y1, x1 = min(h, y0 + max(16, h // 8)), min(w, x0 + max(16, w // 8))
    tile = img[y0:y1, x0:x1]
    return np.repeat(np.repeat(tile, scale, axis=0), scale, axis=1)


def channel_groups(semantics: list[int]) -> list[tuple[str, list[int]]]:
    sem = {s: i for i, s in enumerate(semantics)}
    groups = []
    if all(s in sem for s in (0, 1, 2)):
        groups.append(("albedo", [sem[0], sem[1], sem[2]]))
    if all(s in sem for s in (3, 4, 5)):
        groups.append(("normal", [sem[3], sem[4], sem[5]]))
    orm = [sem[s] for s in (8, 6, 7) if s in sem]
    if orm:
        groups.append(("orm", orm))
    return groups


def _hstack(rows: list[np.ndarray]) -> np.ndarray:
    h = max(im.shape[0] for im in rows)
    padded = []
    for im in rows:
        if im.shape[0] < h:
            pad = np.zeros((h - im.shape[0], im.shape[1], im.shape[2]), dtype=im.dtype)
            im = np.concatenate([im, pad], axis=0)
        padded.append(im)
    return np.concatenate(padded, axis=1)


def write_material(
    out_dir: Path,
    api: str,
    name: str,
    reference: np.ndarray,
    semantics: list[int],
    readback_dir: Path,
) -> list[str]:
    """``reference`` is HWC float in ``[0, 1]``."""
    h, w, c = reference.shape
    prefix = "vk" if api == "vulkan" else "gl"
    ntc = _load_f32(readback_dir / f"{prefix}_ntc_fp32.f32", w, h, c)
    written = []
    if ntc is None:
        return written
    groups = channel_groups(semantics)
    panels = []
    for label, idx in groups:
        ref_g = _u8(_group(reference, idx))
        ntc_g = _u8(_group(ntc, idx))
        err = np.mean(np.abs(ntc[:, :, idx].astype(np.float32) - reference[:, :, idx]), axis=2)
        row = _hstack([ref_g, ntc_g, _heat(err), _u8(_crop(_group(ntc, idx)))])
        Image.fromarray(row, mode="RGB").save(out_dir / f"{name}_{api}_{label}.png")
        panels.append(label)
        written.append(f"{name}_{api}_{label}.png")
    lit_name = f"{prefix}_lit_fp32.f32"
    lit = _load_f32(readback_dir / lit_name, w, h, 3)
    if lit is not None:
        Image.fromarray(_u8(lit), mode="RGB").save(out_dir / f"{name}_{api}_lit.png")
        written.append(f"{name}_{api}_lit.png")
    bc = _load_f32(readback_dir / "bc7.f32", w, h, 4)
    if bc is not None and groups:
        Image.fromarray(_u8(bc[:, :, :3]), mode="RGB").save(out_dir / f"{name}_{api}_bc7.png")
        written.append(f"{name}_{api}_bc7.png")
    (out_dir / f"{name}_{api}.json").write_text(json.dumps({"panels": panels, "written": written}, indent=2))
    return written


def write_astc_panels(out_dir: Path, name: str, reference: np.ndarray, semantics: list[int], exe: str, work: Path) -> list[str]:
    """ASTC 6x6 of each channel group via ``astcenc -cl -medium`` then ``-dl``.

    The panel is the decoded image, not a GPU sample. A failing encode is skipped.
    """
    import subprocess

    written = []
    work.mkdir(parents=True, exist_ok=True)
    for label, idx in channel_groups(semantics):
        src = work / f"{name}_{label}.png"
        enc = work / f"{name}_{label}.astc"
        dec = work / f"{name}_{label}_dec.png"
        Image.fromarray(_u8(_group(reference, idx)), mode="RGB").save(src)
        enc_run = subprocess.run([exe, "-cl", str(src), str(enc), "6x6", "-medium"], capture_output=True, text=True)
        if enc_run.returncode != 0 or not enc.is_file():
            continue
        dec_run = subprocess.run([exe, "-dl", str(enc), str(dec)], capture_output=True, text=True)
        if dec_run.returncode != 0 or not dec.is_file():
            continue
        dest = out_dir / f"{name}_astc6x6_{label}.png"
        dest.write_bytes(dec.read_bytes())
        written.append(dest.name)
    return written
