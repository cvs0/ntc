"""Integer MLP reference.

Activations are int8. The dot is int32. A layer scale is a multiplier and
a right shift, which is what the file can store without a float in the
inner loop. Positional features use the triangle encoding, not sin/cos.
Old files stay on the fp32 decoder. This path is selected by the caller.
"""

from __future__ import annotations

import hashlib

import numpy as np


def triangle_features(uv: np.ndarray, octaves: int, mip: int, num_mips: int) -> np.ndarray:
    """``uv`` is ``[N,2]`` in ``[0,1]``. Values are in about ``[-1,1]``."""
    cols = []
    point = np.array(uv, dtype=np.float64, copy=True)
    for _ in range(int(octaves)):
        frac = point - np.floor(point)
        frac2 = (point + 0.25) - np.floor(point + 0.25)
        cols.append(frac * 2.0 - 1.0)
        cols.append(frac2 * 2.0 - 1.0)
        point *= 2.0
    denom = max(int(num_mips) - 1, 1)
    lod = float(int(mip)) / float(denom)
    cols.append(np.full((uv.shape[0], 2), lod, dtype=np.float64))
    if not cols:
        return np.zeros((uv.shape[0], 0), np.float64)
    return np.concatenate(cols, axis=1)


def quantize_int8(values: np.ndarray, scale: float) -> np.ndarray:
    step = max(float(scale), 1e-6)
    q = np.rint(values / step).astype(np.int32)
    return np.clip(q, -127, 127).astype(np.int8)


def requant_shift(acc: np.ndarray, multiplier: int, shift: int, bits: int = 8) -> np.ndarray:
    """``(acc * multiplier + round) >> shift``, then clamp to int8."""
    span = (1 << (int(bits) - 1)) - 1
    rounded = (acc.astype(np.int64) * np.int64(multiplier)) + np.int64(1 << max(int(shift) - 1, 0))
    shifted = rounded >> np.int64(int(shift))
    return np.clip(shifted, -span, span).astype(np.int32)


def int_layer(
    act: np.ndarray,
    weight: np.ndarray,
    multiplier: int,
    shift: int,
    bias_q: np.ndarray,
) -> np.ndarray:
    """``act`` is int8 ``[N,in]``. ``weight`` is int8 ``[out,in]``.

    The accumulator is int32. Bias is added after the shift, in the same
    units as the requantized activation.
    """
    acc = act.astype(np.int32) @ weight.astype(np.int32).T
    y = requant_shift(acc, multiplier, shift)
    return y + bias_q.astype(np.int32).reshape(1, -1)


def golden_sha256(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        blob = np.ascontiguousarray(array)
        digest.update(blob.shape.__repr__().encode())
        digest.update(blob.dtype.str.encode())
        digest.update(blob.tobytes())
    return digest.hexdigest()
