"""I8 x I8 to I32 tiles of 16.

The tile is the same product as a plain int32 dot. Cooperative-matrix
hardware on NVIDIA and on RDNA4 both expose this shape. Padding is zeros,
so a layer whose sides are not multiples of 16 matches ``int_layer``
before the requant shift.
"""

from __future__ import annotations

import numpy as np

TILE = 16


def i8_tile(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """``left`` is ``[16,16]`` int8. ``right`` is ``[16,16]`` int8.

    The product is int32 ``[16,16]``, row of ``left`` times column of ``right``.
    """
    if left.shape != (TILE, TILE) or right.shape != (TILE, TILE):
        raise ValueError((left.shape, right.shape))
    return left.astype(np.int32) @ right.astype(np.int32)


def i8_gemm(weight: np.ndarray, act: np.ndarray) -> np.ndarray:
    """``weight`` int8 ``[out, inn]``. ``act`` int8 ``[inn, pixels]``.

    Tiles cover both sides. The cropped int32 result is ``weight @ act``.
    """
    out, inn = (int(weight.shape[0]), int(weight.shape[1]))
    if int(act.shape[0]) != inn:
        raise ValueError(act.shape)
    pixels = int(act.shape[1])
    out_t = (out + TILE - 1) // TILE * TILE
    inn_t = (inn + TILE - 1) // TILE * TILE
    pix_t = (pixels + TILE - 1) // TILE * TILE
    left = np.zeros((out_t, inn_t), np.int8)
    right = np.zeros((inn_t, pix_t), np.int8)
    left[:out, :inn] = np.asarray(weight, dtype=np.int8)
    right[:inn, :pixels] = np.asarray(act, dtype=np.int8)
    acc = np.zeros((out_t, pix_t), np.int32)
    for o0 in range(0, out_t, TILE):
        for p0 in range(0, pix_t, TILE):
            tile = np.zeros((TILE, TILE), np.int32)
            for k0 in range(0, inn_t, TILE):
                tile += i8_tile(left[o0 : o0 + TILE, k0 : k0 + TILE], right[k0 : k0 + TILE, p0 : p0 + TILE])
            acc[o0 : o0 + TILE, p0 : p0 + TILE] = tile
    return acc[:out, :pixels]
