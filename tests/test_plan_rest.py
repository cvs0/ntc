"""CPU checks for decoder ideas 10, 11, and 13 to 24."""

import numpy as np

from ntc.plan_rest import (
    apply_channel_mask,
    density,
    int8_dot,
    int8_dot_blocks,
    int_lerp,
    layer0_int,
    ldr_flip_approx,
    magnitude_prune,
    masked_macs,
    octahedral_decode,
    octahedral_encode,
    prefix_apply,
    prefix_macs,
    progressive_psnr,
    residual_pair,
    restore_residual,
    roughness_squared,
    shade_lambert,
    shared_apply,
    sparsify_2to4,
    tile_cache_bytes,
    tile_linear_fraction,
    vq_codebook,
    vq_index_bpp,
    wave_unique,
    zoom_texel_ids,
)


def test_prune_sweep_and_two_of_four():
    rng = np.random.default_rng(0)
    weight = rng.normal(size=(8, 16)).astype(np.float32)
    for keep in (1.0, 0.75, 0.5, 0.25):
        pruned = magnitude_prune(weight, keep)
        assert abs(density(pruned) - keep) < 0.08 or keep == 1.0
    sparse = sparsify_2to4(weight)
    groups = sparse.reshape(8, 4, 4)
    assert np.all(np.count_nonzero(groups, axis=-1) == 2)


def test_integer_layer0_matches_its_rounding_rule():
    a = np.array([1, 4, 7], np.int32)
    b = np.array([9, 2, 6], np.int32)
    got = int_lerp(a, b, 3, 8)
    expect = (a * 5 + b * 3 + 4) // 8
    np.testing.assert_array_equal(got, expect)
    codes = np.array([[1, 2, 3, 4], [0, 1, 0, 1]], np.int32)
    weight = np.array([[1, 0, -1, 2], [2, 2, 1, 0]], np.int32).T
    bias = np.array([3, -1], np.int32)
    acc = layer0_int(codes, weight, bias)
    np.testing.assert_array_equal(acc, codes @ weight + bias)


def test_matryoshka_prefix_matches_when_the_tail_is_zero():
    feat = np.array([[0.2, -0.4, 0.5, 0.1]], np.float32)
    weight = np.zeros((8, 4), np.float32)
    weight[:4, :] = np.eye(4, dtype=np.float32)
    bias = np.zeros(8, np.float32)
    full = feat @ weight.T + bias
    prefix = prefix_apply(feat, weight, bias, 4)
    assert np.max(np.abs(full[:, :4] - prefix)) == 0
    assert prefix_macs(8, 16, 10) < prefix_macs(32, 16, 10)


def test_wave_dedupe_drops_under_magnification():
    near = zoom_texel_ids(64, zoom=4.0, lanes=32)
    far = zoom_texel_ids(64, zoom=0.5, lanes=32)
    assert wave_unique(near) < 32
    assert wave_unique(far) == 32


def test_channel_mask_and_shader_outputs():
    full, part = masked_macs(10, 32, 4)
    assert part < full
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    masked = apply_channel_mask(values, np.array([0, 2]))
    assert masked[0, 1] == 0 and masked[0, 0] == values[0, 0]
    n = np.array([[0.0, 0.0, 1.0], [0.3, -0.2, 0.9]], np.float32)
    back = octahedral_decode(octahedral_encode(n))
    cos = np.sum(back * (n / np.linalg.norm(n, axis=-1, keepdims=True)), axis=-1)
    assert np.min(cos) > 0.98
    assert float(roughness_squared(np.array(0.5))) == 0.25


def test_int8_blocks_match_the_scalar_dot():
    rng = np.random.default_rng(1)
    a = rng.integers(-127, 127, size=64, dtype=np.int8)
    b = rng.integers(-127, 127, size=64, dtype=np.int8)
    assert int(int8_dot(a, b)) == int(int8_dot_blocks(a, b))


def test_bc7_tile_is_four_times_smaller_and_coarse_grid_decodes():
    rgba = tile_cache_bytes(16, 0, bc7=False)
    bc7 = tile_cache_bytes(16, 16, bc7=True)
    assert rgba == 1024 and bc7 == 256 and rgba / bc7 == 4
    full = np.linspace(0, 1, 16, dtype=np.float32)
    coarse = np.full(16, 0.5, np.float32)
    score = progressive_psnr(full, coarse)
    assert 0 < score < 40


def test_residual_vq_tile_head_and_shared_decoder():
    image = np.zeros((3, 16, 16), np.float32)
    image[0] = np.linspace(0, 1, 16, dtype=np.float32)
    base, resid = residual_pair(image)
    back = restore_residual(np.transpose(base, (1, 2, 0)), np.transpose(resid, (1, 2, 0)))
    assert np.max(np.abs(np.transpose(back, (2, 0, 1)) - image)) < 1e-5
    tile = np.tile(np.array([[1.0, 0.0], [0.0, 1.0]], np.float32), (8, 8))
    vec = np.stack([tile, tile * 0.5], 0).reshape(2, -1).T
    idx, book = vq_codebook(vec, k=2, rounds=3)
    assert vq_index_bpp(idx, 2, pixels=tile.size) < 4.0
    ref = np.zeros((8, 8, 1), np.float32)
    ref[:] = 0.4
    frac = tile_linear_fraction(ref, ref, ref + 0.01, tile=8, tol_db=0.5)
    assert frac == 1.0
    outs = shared_apply(np.eye(2, dtype=np.float32), np.zeros(2), [np.ones((3, 2), np.float32), np.full((3, 2), 2.0)])
    assert outs[0].shape == (3, 2) and float(outs[1][0, 0]) == 2.0
    assert ldr_flip_approx(ref[:, :, 0], ref[:, :, 0]) == 0.0
    color = np.full((2, 2, 3), 0.5, np.float32)
    normal = np.full((2, 2, 3), 0.5, np.float32)
    normal[:, :, 2] = 1.0
    shade = shade_lambert(color, normal, np.array([0.0, 0.0, 1.0], np.float32))
    assert shade.shape == color.shape
