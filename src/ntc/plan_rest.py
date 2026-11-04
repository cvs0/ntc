"""Remaining decoder ideas from the BC7 plan, as CPU references.

Each function is the behavior behind a training flag or a decbench host
variant. GPU timings are not produced here.
"""

from __future__ import annotations

import numpy as np


def magnitude_prune(weight: np.ndarray, keep: float) -> np.ndarray:
    """Keep the largest ``keep`` fraction of ``|weight|``. The rest are zero."""
    w = np.asarray(weight, np.float32).copy()
    if keep >= 1:
        return w
    if keep <= 0:
        return np.zeros_like(w)
    flat = np.abs(w).ravel()
    k = max(1, int(round(float(keep) * flat.size)))
    if k >= flat.size:
        return w
    thresh = np.partition(flat, flat.size - k)[flat.size - k]
    return np.where(np.abs(w) >= thresh, w, np.float32(0))


def sparsify_2to4(weight: np.ndarray) -> np.ndarray:
    """Keep the two largest magnitudes in every group of four inputs."""
    w = np.asarray(weight, np.float32).copy()
    out, inn = w.shape
    pad = (4 - (inn % 4)) % 4
    if pad:
        w = np.pad(w, ((0, 0), (0, pad)))
    groups = w.reshape(out, -1, 4)
    order = np.argsort(np.abs(groups), axis=-1)
    drop = order[:, :, :2]
    mask = np.ones_like(groups, dtype=bool)
    rows = np.arange(out)[:, None, None]
    cols = np.arange(groups.shape[1])[None, :, None]
    mask[rows, cols, drop] = False
    groups = np.where(mask, groups, np.float32(0))
    return groups.reshape(out, -1)[:, :inn]


def density(weight: np.ndarray) -> float:
    w = np.asarray(weight)
    if w.size == 0:
        return 0.0
    return float(np.count_nonzero(w)) / float(w.size)


def int_lerp(a: np.ndarray, b: np.ndarray, numer: int, denom: int) -> np.ndarray:
    """Exact integer lerp. ``numer`` and ``denom`` are the fixed-point weight.

    Result is ``(a * (denom - numer) + b * numer + denom // 2) // denom``.
    """
    if denom <= 0 or numer < 0 or numer > denom:
        raise ValueError((numer, denom))
    aa = np.asarray(a, np.int32)
    bb = np.asarray(b, np.int32)
    return (aa * (denom - numer) + bb * numer + denom // 2) // denom


def layer0_int(codes: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Integer first dot. ``codes`` int ``[N,K]``, ``weight`` int ``[K,Out]``."""
    acc = np.asarray(codes, np.int32) @ np.asarray(weight, np.int32)
    return acc + np.asarray(bias, np.int32)


def prefix_macs(hidden: int, inn: int, out: int) -> int:
    return int(hidden) * int(inn) + int(hidden) * int(out)


def prefix_apply(features: np.ndarray, weight: np.ndarray, bias: np.ndarray, width: int) -> np.ndarray:
    """First ``width`` rows of a hidden layer. Tail rows are not read."""
    w = np.asarray(weight, np.float32)[: int(width)]
    b = np.asarray(bias, np.float32)[: int(width)]
    return features.astype(np.float32) @ w.T + b


def wave_unique(texel_ids: np.ndarray) -> int:
    return int(np.unique(np.asarray(texel_ids)).size)


def zoom_texel_ids(width: int, zoom: float, lanes: int = 32) -> np.ndarray:
    """Texel ids for ``lanes`` adjacent pixels at ``zoom`` (1 is 1:1)."""
    step = 1.0 / float(zoom)
    ids = []
    for lane in range(lanes):
        x = int(np.floor(lane * step)) % int(width)
        ids.append(x)
    return np.asarray(ids, np.int32)


def masked_macs(full_out: int, hidden: int, keep: int) -> tuple[int, int]:
    """Last-layer MACs with every row, then with ``keep`` rows."""
    full = int(full_out) * int(hidden)
    part = int(keep) * int(hidden)
    return full, part


def apply_channel_mask(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``values`` ``[N,C]``. Rows outside ``mask`` are zero."""
    out = np.zeros_like(values, dtype=np.float32)
    keep = np.asarray(mask, np.int32)
    out[:, keep] = np.asarray(values, np.float32)[:, keep]
    return out


def octahedral_encode(normal: np.ndarray) -> np.ndarray:
    """Unit xyz to a 2-vector in ``[-1,1]``. ``normal`` is ``[N,3]``."""
    n = np.asarray(normal, np.float32)
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
    l1 = np.sum(np.abs(n), axis=-1, keepdims=True)
    p = n[:, :2] / np.maximum(l1, 1e-8)
    flip = n[:, 2:3] < 0
    sign = np.sign(p)
    sign = np.where(sign == 0, 1, sign)
    folded = (1.0 - np.abs(p[:, ::-1])) * sign
    return np.where(flip, folded, p).astype(np.float32)


def octahedral_decode(xy: np.ndarray) -> np.ndarray:
    p = np.asarray(xy, np.float32)
    z = 1.0 - np.sum(np.abs(p), axis=-1, keepdims=True)
    x = p[:, :1]
    y = p[:, 1:2]
    neg = z < 0
    sign = np.sign(x)
    sign = np.where(sign == 0, 1, sign)
    sign_y = np.sign(y)
    sign_y = np.where(sign_y == 0, 1, sign_y)
    x2 = np.where(neg, (1.0 - np.abs(y)) * sign, x)
    y2 = np.where(neg, (1.0 - np.abs(x)) * sign_y, y)
    n = np.concatenate([x2, y2, z], axis=-1)
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
    return n.astype(np.float32)


def roughness_squared(roughness: np.ndarray) -> np.ndarray:
    r = np.clip(np.asarray(roughness, np.float32), 0.0, 1.0)
    return r * r


def int8_dot(a: np.ndarray, b: np.ndarray) -> np.int32:
    """int8 dot, int32 accumulator. The SIMD path uses the same product."""
    return np.int32(np.dot(np.asarray(a, np.int32), np.asarray(b, np.int32)))


def int8_dot_blocks(a: np.ndarray, b: np.ndarray, block: int = 16) -> np.int32:
    """Same dot, accumulated in blocks the way a SIMD kernel would."""
    aa = np.asarray(a, np.int32)
    bb = np.asarray(b, np.int32)
    acc = np.int32(0)
    for i in range(0, aa.size, block):
        acc = acc + np.int32(np.dot(aa[i : i + block], bb[i : i + block]))
    return acc


def tile_cache_bytes(tile: int, blocks: int, bc7: bool) -> int:
    """RGBA8 is 4 bytes per texel. BC7 is 16 bytes per 4x4 block."""
    if bc7:
        return int(blocks) * 16
    return int(tile) * int(tile) * 4


def quarter_base(images: np.ndarray) -> np.ndarray:
    """Box down by 4 and nearest back up. ``images`` is ``[C,H,W]``."""
    c, h, w = images.shape
    hs = max(1, h // 4)
    ws = max(1, w // 4)
    small = np.zeros((c, hs, ws), np.float32)
    for y in range(hs):
        for x in range(ws):
            small[:, y, x] = images[:, y * 4 : (y + 1) * 4, x * 4 : (x + 1) * 4].mean(axis=(1, 2))
    up = np.repeat(np.repeat(small, 4, axis=1), 4, axis=2)
    return up[:, :h, :w].astype(np.float32)


def residual_pair(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Base and a ``[0,1]`` encoding of ``images - base``."""
    base = quarter_base(images)
    resid = np.clip((images - base) * 0.5 + 0.5, 0.0, 1.0).astype(np.float32)
    return base, resid


def restore_residual(base: np.ndarray, resid_01: np.ndarray) -> np.ndarray:
    return np.clip(base + (resid_01 - 0.5) * 2.0, 0.0, 1.0).astype(np.float32)


def vq_codebook(vectors: np.ndarray, k: int, rounds: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """``vectors`` ``[N,C]``. Returns int32 indices and a float codebook."""
    v = np.asarray(vectors, np.float32)
    n = v.shape[0]
    k = max(1, min(int(k), n))
    choice = np.linspace(0, n - 1, k).astype(np.int32)
    book = v[choice].copy()
    idx = np.zeros(n, np.int32)
    for _ in range(rounds):
        dist = ((v[:, None, :] - book[None, :, :]) ** 2).sum(axis=-1)
        idx = dist.argmin(axis=1).astype(np.int32)
        for i in range(k):
            sel = v[idx == i]
            if sel.shape[0]:
                book[i] = sel.mean(axis=0)
    return idx, book


def vq_index_bpp(indices: np.ndarray, k: int, pixels: int) -> float:
    bits = int(np.ceil(np.log2(max(int(k), 2))))
    return float(indices.size * bits) / float(max(int(pixels), 1))


def tile_linear_fraction(
    reference: np.ndarray,
    linear: np.ndarray,
    full: np.ndarray,
    tile: int = 8,
    tol_db: float = 0.5,
) -> float:
    """Fraction of tiles whose linear head is within ``tol_db`` of the full head."""
    ref = np.asarray(reference, np.float32)
    h, w, _c = ref.shape
    close = 0
    total = 0
    for y in range(0, h - tile + 1, tile):
        for x in range(0, w - tile + 1, tile):
            sl = (slice(y, y + tile), slice(x, x + tile))
            r = ref[sl]
            lin_mse = float(np.mean((linear[sl] - r) ** 2))
            full_mse = float(np.mean((full[sl] - r) ** 2))
            def _db(mse: float) -> float:
                if mse <= 1e-12:
                    return 100.0
                return float(10.0 * np.log10(1.0 / mse))
            total += 1
            if _db(lin_mse) + 1e-6 >= _db(full_mse) - float(tol_db):
                close += 1
    if total == 0:
        return 0.0
    return close / float(total)


def ldr_flip_approx(pred: np.ndarray, ref: np.ndarray) -> float:
    """Mean absolute error in a rough luma and chroma split, in ``[0,1]``.

    This is not the NVIDIA FLIP package. It is the number reported when that
    package is not installed.
    """
    p = np.clip(np.asarray(pred, np.float32), 0, 1)
    r = np.clip(np.asarray(ref, np.float32), 0, 1)
    if p.ndim == 2:
        p = p[:, :, None]
        r = r[:, :, None]
    if p.shape[-1] == 1:
        p = np.repeat(p, 3, axis=-1)
        r = np.repeat(r, 3, axis=-1)
    luma_p = 0.299 * p[:, :, 0] + 0.587 * p[:, :, 1] + 0.114 * p[:, :, 2]
    luma_r = 0.299 * r[:, :, 0] + 0.587 * r[:, :, 1] + 0.114 * r[:, :, 2]
    chroma = np.mean(np.abs(p - r), axis=-1)
    return float(np.mean(0.7 * np.abs(luma_p - luma_r) + 0.3 * chroma))


def shade_lambert(color: np.ndarray, normal_01: np.ndarray, light: np.ndarray) -> np.ndarray:
    n = np.asarray(normal_01, np.float32) * 2.0 - 1.0
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
    ell = np.asarray(light, np.float32)
    ell = ell / max(float(np.linalg.norm(ell)), 1e-6)
    ndl = np.clip((n * ell.reshape(1, 1, 3)).sum(axis=-1, keepdims=True), 0.0, 1.0)
    return (np.asarray(color, np.float32) * ndl).astype(np.float32)


def shared_apply(weight: np.ndarray, bias: np.ndarray, features: list[np.ndarray]) -> list[np.ndarray]:
    """One MLP, one feature matrix per material."""
    w = np.asarray(weight, np.float32)
    b = np.asarray(bias, np.float32)
    return [feat.astype(np.float32) @ w.T + b for feat in features]


def progressive_psnr(full: np.ndarray, coarse: np.ndarray) -> float:
    err = np.asarray(full, np.float64) - np.asarray(coarse, np.float64)
    mse = float(np.mean(err * err))
    if mse <= 1e-20:
        return 100.0
    return float(min(100.0, 10.0 * np.log10(1.0 / mse)))
