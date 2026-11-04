"""CPU checks for the decoder pieces that do not need a GPU."""

import numpy as np

from ntc.container import (
    FLAG_BC_LATENTS,
    ChannelRec,
    GridRec,
    LayerRec,
    TextureSet,
    build_texture_set,
)
from ntc.decode_ref import mlp
from ntc.decode_ship import (
    bc_latent_psnr,
    classify_tiles,
    decode_mip_hoisted,
    emit_immediate_mlp,
    hidden_knee,
    hint_bpp,
    hybrid_tail_bpp,
    immediate_mlp,
    lod_histogram,
    mode_hint_bytes,
    pack_latent_bc,
    reference_matches_hoist,
)


def _layer(weight, bias):
    weight = np.asarray(weight, np.float32)
    return LayerRec(
        in_features=int(weight.shape[1]),
        out_features=int(weight.shape[0]),
        group_size=0,
        num_groups=1,
        weight_offset=0,
        weight_bytes=0,
        bias_offset=0,
        scale_offset=0,
        weight=weight,
        bias=np.asarray(bias, np.float32),
        scale=np.ones((weight.shape[0], 1), np.float32),
    )


def _ts() -> TextureSet:
    codes = np.zeros((2, 8, 8), np.uint8)
    yy, xx = np.mgrid[0:8, 0:8]
    codes[0] = ((xx + yy) % 16).astype(np.uint8)
    codes[1] = ((xx * 2) % 16).astype(np.uint8)
    grid = GridRec(
        width=8,
        height=8,
        channels=2,
        bits=4,
        row_stride=0,
        data_offset=0,
        data_bytes=0,
        scale_offset=0,
        plane_bytes=0,
        scale=np.array([0.2, 0.15], np.float32),
        codes=codes,
    )
    return TextureSet(
        width=8,
        height=8,
        num_mips=1,
        activation=1,
        pe_mode=0,
        pe_octaves=0,
        flags=0,
        weight_bits=8,
        group_size=0,
        channels=[ChannelRec(0)],
        grids=[grid],
        layers=[
            _layer([[1, 0], [0, 1], [-1, 1], [2, -1]], [0.1, -0.2, 0.0, 0.05]),
            _layer([[1, -1, 0.5, 0]], [0.0]),
        ],
        raw=b"",
    )


def test_hoisted_layer0_matches_decode_mip():
    ts = _ts()
    err = reference_matches_hoist(ts, 0)
    assert err < 1e-4
    assert decode_mip_hoisted(ts, 0).shape == (8, 8, 1)


def test_bc_latent_roundtrip_on_a_ramp_and_the_flag():
    codes = np.zeros((4, 8, 8), np.uint8)
    yy, xx = np.mgrid[0:8, 0:8]
    codes[0] = xx.astype(np.uint8)
    codes[1] = yy.astype(np.uint8)
    codes[2] = ((xx + yy) // 2).astype(np.uint8)
    codes[3] = 7
    payload, restored = pack_latent_bc(codes, 4)
    assert len(payload) == (8 // 4) * (8 // 4) * 16
    # Mode 6 endpoints are the per-block min and max. On this ramp the
    # restored codes are within 2 bins and about 28 dB, which is not the
    # 0.5 dB training target. That target needs BC in the training loop.
    score = bc_latent_psnr(codes, restored, 4)
    assert score > 25.0
    assert int(np.max(np.abs(codes.astype(np.int16) - restored.astype(np.int16)))) <= 2
    blob = build_texture_set(
        width=4,
        height=4,
        num_mips=1,
        activation=1,
        pe_mode=0,
        pe_octaves=0,
        flags=FLAG_BC_LATENTS,
        weight_bits=8,
        group_size=0,
        channels=[ChannelRec(0)],
        grids=[{"codes": np.zeros((1, 4, 4), np.uint8), "bits": 4, "scale": np.ones(1, np.float32)}],
        layers=[
            {
                "codes": np.ones((1, 1), np.int16),
                "bias": np.zeros(1, np.float32),
                "scale": np.ones((1, 1), np.float32),
                "group_size": 0,
            }
        ],
    )
    from ntc.container import parse_texture_set

    ts = parse_texture_set(blob)
    assert ts.flags & FLAG_BC_LATENTS
    np.testing.assert_array_equal(ts.grids[0].codes, np.zeros((1, 4, 4), np.uint8))


def test_immediates_drop_zeros_and_match_the_mlp():
    ts = _ts()
    text = emit_immediate_mlp(ts.layers)
    assert "0.00000000e+00" not in text
    assert "// w0_3" not in text
    x = np.array([[0.2, -0.4, 0.1, 0.0], [1.0, 0.0, -1.0, 0.5]], np.float32)
    # Layer 1 only, so the zero column is in the product under test.
    got = immediate_mlp(x, [ts.layers[1]], ts.activation)
    ref = mlp(x, _one_layer(ts))
    assert np.max(np.abs(got - ref)) < 1e-5


def _one_layer(ts: TextureSet) -> TextureSet:
    return TextureSet(
        width=ts.width,
        height=ts.height,
        num_mips=1,
        activation=ts.activation,
        pe_mode=0,
        pe_octaves=0,
        flags=0,
        weight_bits=8,
        group_size=0,
        channels=ts.channels,
        grids=ts.grids,
        layers=[ts.layers[1]],
        raw=b"",
    )


def test_hybrid_tail_deferred_tiles_and_hints():
    assert abs(hybrid_tail_bpp() - 2.0) < 1e-6
    hist = lod_histogram(np.array([0.2, 1.8, 2.1, 2.9, 5.0]))
    assert hist[0] == 1 and hist[1] == 1 and hist[2] == 2 and hist[5] == 1
    ids = np.zeros((8, 16), np.int32)
    ids[:, 8:] = 1
    tiles = classify_tiles(ids, 8, 4)
    assert len(tiles) == 4
    assert all(not tile["mixed"] for tile in tiles)
    assert {tile["material"] for tile in tiles} == {0, 1}
    raw = mode_hint_bytes(8, 8, mode=6, partition=0)
    assert hint_bpp(8, 8) == 0.25
    assert raw[0] & 0x0F == 6
    assert len(raw) == 2


def test_hidden_knee_picks_the_smallest_close_width():
    assert hidden_knee([(0, 30.0), (8, 33.2), (16, 33.25), (32, 33.3)]) == 8
    assert hidden_knee([(0, 20.0), (8, 30.0), (16, 33.0)]) == 16
