"""I8 cooperative-matrix tiles match a plain int32 product."""

import numpy as np

from ntc.coopmat_i8 import TILE, i8_gemm, i8_tile


def test_one_tile_is_an_int32_product():
    rng = np.random.default_rng(0)
    left = rng.integers(-127, 128, (TILE, TILE), dtype=np.int8)
    right = rng.integers(-127, 128, (TILE, TILE), dtype=np.int8)
    got = i8_tile(left, right)
    assert got.dtype == np.int32
    assert np.array_equal(got, left.astype(np.int32) @ right.astype(np.int32))


def test_padded_gemm_matches_the_integer_layer_dot():
    rng = np.random.default_rng(1)
    out, inn, pixels = 20, 18, 13
    weight = rng.integers(-40, 41, (out, inn), dtype=np.int8)
    act = rng.integers(-40, 41, (pixels, inn), dtype=np.int8)
    got = i8_gemm(weight, act.T)
    plain = act.astype(np.int32) @ weight.astype(np.int32).T
    assert np.array_equal(got.T, plain)
