"""BC block packers round-trip through the matching unpacker."""

import numpy as np

from bench.bc_pack import (
    pack_bc1_block,
    pack_bc5_block,
    pack_bc7_mode6_block,
    unpack_bc1_block,
    unpack_bc4_block,
    unpack_bc7_mode6_block,
)


def test_bc1_solid():
    px = np.zeros((16, 3), dtype=np.uint8)
    px[:] = (40, 80, 160)
    got = unpack_bc1_block(pack_bc1_block(px))
    assert np.max(np.abs(got[:, :3].astype(int) - px.astype(int))) <= 8


def test_bc1_two_colors():
    px = np.zeros((16, 3), dtype=np.uint8)
    px[:8] = (255, 0, 0)
    px[8:] = (0, 0, 255)
    got = unpack_bc1_block(pack_bc1_block(px))
    assert got[0, 0] > 200
    assert got[15, 2] > 200


def test_bc5_constant():
    px = np.full((16, 2), 128, dtype=np.uint8)
    raw = pack_bc5_block(px)
    x = unpack_bc4_block(raw[:8])
    y = unpack_bc4_block(raw[8:])
    assert np.max(np.abs(x.astype(int) - 128)) <= 1
    assert np.max(np.abs(y.astype(int) - 128)) <= 1


def test_bc7_mode6_solid_and_gradient():
    solid = np.zeros((16, 4), dtype=np.uint8)
    solid[:] = (20, 40, 60, 255)
    got = unpack_bc7_mode6_block(pack_bc7_mode6_block(solid))
    assert np.max(np.abs(got.astype(int) - solid.astype(int))) <= 2
    grad = np.zeros((16, 4), dtype=np.uint8)
    for i in range(16):
        t = i * 17
        grad[i] = (t, t // 2, t, 255)
    got = unpack_bc7_mode6_block(pack_bc7_mode6_block(grad))
    # One subset, endpoints at the channel min and max. Quantization is 7 bits.
    assert np.max(np.abs(got.astype(int) - grad.astype(int))) <= 16
