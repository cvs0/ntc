import numpy as np

from ntc.container import (
    HEADER_BYTES,
    ChannelRec,
    build_texture_set,
    file_bpp,
    parse_texture_set,
    payload_bytes,
    rate_report,
    total_bpp,
    unpack_weight_matrix,
)


def _tiny_blob():
    codes = np.arange(4 * 4 * 4, dtype=np.uint8).reshape(4, 4, 4) % 16
    # 4-bit, width 4
    w = np.array([[1, -2, 3, 0], [0, 1, -1, 2]], dtype=np.int16)
    return build_texture_set(
        width=8,
        height=8,
        num_mips=4,
        activation=1,
        pe_mode=0,
        pe_octaves=0,
        flags=0,
        weight_bits=8,
        group_size=0,
        channels=[ChannelRec(i) for i in range(2)],
        grids=[{"codes": codes, "bits": 4, "scale": np.array([0.1, 0.2, 0.3, 0.4], np.float32)}],
        layers=[
            {
                "codes": w,
                "bias": np.array([0.25, -0.5], np.float32),
                "scale": np.array([[0.5], [0.25]], np.float32),
                "group_size": 0,
            }
        ],
    )


def test_header_size_and_roundtrip():
    blob = _tiny_blob()
    assert len(blob) > HEADER_BYTES
    ts = parse_texture_set(blob)
    assert ts.width == 8 and ts.height == 8
    assert ts.grids[0].codes.shape == (4, 4, 4)
    assert ts.grids[0].bits == 4
    np.testing.assert_array_equal(
        ts.grids[0].codes, np.arange(4 * 4 * 4, dtype=np.uint8).reshape(4, 4, 4) % 16
    )
    np.testing.assert_allclose(ts.grids[0].scale, [0.1, 0.2, 0.3, 0.4], atol=1e-3)
    np.testing.assert_array_equal(ts.layers[0].weight, [[1, -2, 3, 0], [0, 1, -1, 2]])
    np.testing.assert_allclose(ts.layers[0].bias, [0.25, -0.5], atol=2e-3)
    np.testing.assert_allclose(ts.layers[0].scale[:, 0], [0.5, 0.25], atol=2e-3)
    again = parse_texture_set(blob)
    assert again.raw == blob
    assert payload_bytes(ts) > 0
    assert total_bpp(ts) > 0
    assert file_bpp(blob, 8, 8) > total_bpp(ts)
    rates = rate_report(ts)
    assert rates["latent_bpp_mip0"] < rates["total_bpp"]
    assert rates["file_bpp_chain"] < file_bpp(blob, 8, 8)
    assert rates["bppc"] < rates["file_bpp_chain"]
    assert rates["mip_chain_texels"] == 8 * 8 + 4 * 4 + 2 * 2 + 1 * 1


def test_int4_weight_pack():
    codes = np.array([[7, -7, 1, -1], [0, 3, -3, 2]], dtype=np.int16)
    blob = build_texture_set(
        width=4,
        height=4,
        num_mips=2,
        activation=0,
        pe_mode=0,
        pe_octaves=0,
        flags=0,
        weight_bits=4,
        group_size=2,
        channels=[ChannelRec(0)],
        grids=[
            {
                "codes": np.zeros((1, 4, 4), np.uint8),
                "bits": 8,
                "scale": np.array([1.0], np.float32),
            }
        ],
        layers=[
            {
                "codes": codes,
                "bias": np.zeros(2, np.float32),
                "scale": np.ones((2, 2), np.float32),
                "group_size": 2,
            }
        ],
    )
    ts = parse_texture_set(blob)
    np.testing.assert_array_equal(ts.layers[0].weight, codes)
    # nibble packing: low nibble is the even input
    raw = unpack_weight_matrix(blob, ts.layers[0].weight_offset, 2, 4, 4)
    np.testing.assert_array_equal(raw, codes)


def test_latent_deflate_roundtrip_and_rate():
    from ntc.container import FLAG_LATENT_DEFLATE

    codes = np.zeros((4, 32, 32), np.uint8)
    codes[:, :4, :4] = 3
    common = dict(
        width=32,
        height=32,
        num_mips=2,
        activation=1,
        pe_mode=0,
        pe_octaves=0,
        flags=0,
        weight_bits=8,
        group_size=0,
        channels=[ChannelRec(0), ChannelRec(1)],
        grids=[{"codes": codes, "bits": 4, "scale": np.ones(4, np.float32) * 0.1}],
        layers=[
            {
                "codes": np.ones((2, 4), np.int16),
                "bias": np.zeros(2, np.float32),
                "scale": np.ones((2, 1), np.float32),
                "group_size": 0,
            }
        ],
    )
    plain = build_texture_set(**common)
    packed = build_texture_set(**common, latent_deflate=True)
    assert len(packed) < len(plain)
    ts = parse_texture_set(packed)
    assert ts.flags & FLAG_LATENT_DEFLATE
    np.testing.assert_array_equal(ts.grids[0].codes, codes)
    assert total_bpp(ts) < total_bpp(parse_texture_set(plain))
    assert ts.latent_stored_bytes < ts.latent_plain_bytes


def test_latent_range_roundtrip_matches_plain_codes():
    from ntc.container import FLAG_LATENT_RANGE
    from ntc.range_coder import range_decode, range_encode

    raw = bytes([0, 1, 2, 15, 15, 15, 0, 4, 8, 255, 7, 7, 7, 1])
    assert range_decode(range_encode(raw), len(raw)) == raw
    codes = np.zeros((2, 16, 16), np.uint8)
    codes[:, :, :] = 1
    codes[:, 2, 3] = 7
    common = dict(
        width=16,
        height=16,
        num_mips=1,
        activation=1,
        pe_mode=0,
        pe_octaves=0,
        flags=0,
        weight_bits=8,
        group_size=0,
        channels=[ChannelRec(0), ChannelRec(1)],
        grids=[{"codes": codes, "bits": 4, "scale": np.ones(2, np.float32)}],
        layers=[
            {
                "codes": np.ones((2, 2), np.int16),
                "bias": np.zeros(2, np.float32),
                "scale": np.ones((2, 1), np.float32),
                "group_size": 0,
            }
        ],
    )
    ranged = build_texture_set(**common, latent_coder="range")
    ts = parse_texture_set(ranged)
    assert ts.flags & FLAG_LATENT_RANGE
    np.testing.assert_array_equal(ts.grids[0].codes, codes)
    best = build_texture_set(**common, latent_coder="best")
    deflated = build_texture_set(**common, latent_deflate=True)
    assert len(best) <= len(deflated)


def test_layer_weight_bits_roundtrip():
    hidden = np.array([[1, -2, 3, 0], [7, -7, 1, -1]], dtype=np.int16)
    out = np.array([[4, -3], [0, 1], [2, -2]], dtype=np.int16)
    blob = build_texture_set(
        width=4,
        height=4,
        num_mips=1,
        activation=1,
        pe_mode=0,
        pe_octaves=0,
        flags=0,
        weight_bits=8,
        group_size=0,
        channels=[ChannelRec(0)],
        grids=[{"codes": np.zeros((1, 4, 4), np.uint8), "bits": 8, "scale": np.array([0.2], np.float32)}],
        layers=[
            {
                "codes": hidden,
                "bias": np.zeros(2, np.float32),
                "scale": np.ones((2, 1), np.float32),
                "group_size": 0,
                "weight_bits": 4,
            },
            {
                "codes": out,
                "bias": np.zeros(3, np.float32),
                "scale": np.ones((3, 1), np.float32),
                "group_size": 0,
                "weight_bits": 8,
            },
        ],
    )
    ts = parse_texture_set(blob)
    assert ts.layers[0].weight_bits == 4
    assert ts.layers[1].weight_bits == 8
    np.testing.assert_array_equal(ts.layers[0].weight, hidden)
    np.testing.assert_array_equal(ts.layers[1].weight, out)


def test_tile_endpoints_roundtrip():
    codes = (np.arange(2 * 8 * 8) % 16).astype(np.uint8).reshape(2, 8, 8)
    scale = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.05, 0.06], [0.07, 0.08]]], np.float32)
    offset = np.zeros_like(scale)
    offset[0, 0, 1] = 0.25
    blob = build_texture_set(
        width=8,
        height=8,
        num_mips=2,
        activation=1,
        pe_mode=0,
        pe_octaves=0,
        flags=0,
        weight_bits=8,
        group_size=0,
        tile_size=4,
        channels=[ChannelRec(0)],
        grids=[
            {
                "codes": codes,
                "bits": 4,
                "scale": np.array([0.1, 0.1], np.float32),
                "tile_scale": scale,
                "tile_offset": offset,
            }
        ],
        layers=[
            {
                "codes": np.ones((1, 2), np.int16),
                "bias": np.zeros(1, np.float32),
                "scale": np.ones((1, 1), np.float32),
                "group_size": 0,
            }
        ],
    )
    ts = parse_texture_set(blob)
    assert ts.tile_size == 4
    np.testing.assert_allclose(ts.grids[0].tile_scale, scale, atol=2e-3)
    np.testing.assert_allclose(ts.grids[0].tile_offset, offset, atol=2e-3)
    np.testing.assert_array_equal(ts.grids[0].codes, codes)


def _recrc(blob: bytearray) -> bytes:
    import struct
    import zlib

    crc = zlib.crc32(blob[HEADER_BYTES:]) & 0xFFFFFFFF
    struct.pack_into("<I", blob, 76, crc)
    return bytes(blob)


def test_truncated_bad_magic_and_version_are_value_errors():
    blob = _tiny_blob()
    for n in (0, 8, HEADER_BYTES - 1, HEADER_BYTES):
        try:
            parse_texture_set(blob[:n] if n else b"")
            assert n == HEADER_BYTES
        except ValueError:
            pass
    bad = bytearray(blob)
    bad[0] ^= 0xFF
    try:
        parse_texture_set(_recrc(bad))
        raised = False
    except ValueError as exc:
        raised = "magic" in str(exc)
    assert raised
    bad = bytearray(blob)
    bad[4] = 99
    try:
        parse_texture_set(_recrc(bad))
        raised = False
    except ValueError as exc:
        raised = "version" in str(exc)
    assert raised


def test_zero_size_and_wild_offset_are_value_errors():
    blob = bytearray(_tiny_blob())
    import struct

    struct.pack_into("<I", blob, 12, 0)
    try:
        parse_texture_set(_recrc(blob))
        raised = False
    except ValueError as exc:
        raised = "empty" in str(exc)
    assert raised
    blob = bytearray(_tiny_blob())
    struct.pack_into("<I", blob, 40, 0xFFFFFFF0)
    try:
        parse_texture_set(_recrc(blob))
        raised = False
    except ValueError as exc:
        raised = "channel table" in str(exc)
    assert raised


def test_range_coder_roundtrip_and_truncated():
    from ntc.range_coder import range_decode, range_decode_ctx, range_encode, range_encode_ctx

    rng = np.random.default_rng(1)
    for n in (0, 1, 2, 7, 15, 64, 200):
        raw = bytes(int(v) for v in rng.integers(0, 256, size=n))
        assert range_decode(range_encode(raw), n) == raw
        assert range_decode_ctx(range_encode_ctx(raw), n) == raw
    try:
        range_decode(b"\x00\x01", 4)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_encode_odd_size_single_channel_is_finite_and_deterministic():
    from ntc.decode_ref import decode_mip
    from ntc.encode import TrainConfig, encode_arrays

    img = np.linspace(0.0, 1.0, 3 * 5, dtype=np.float32).reshape(1, 3, 5)
    cfg = TrainConfig(
        steps=2,
        hidden=(4,),
        grids=[(4, 2, 1, 4)],
        seed=3,
        seed_pool=1,
        adaptive_pool=1,
        with_ssim=False,
        mip_levels=2,
        fast=False,
        device="cpu",
        pca_init=True,
        log_every=10**9,
    )
    a, _ = encode_arrays(img, [0], cfg)
    b, _ = encode_arrays(img, [0], cfg)
    assert a == b
    recon = decode_mip(parse_texture_set(a), 0)
    assert recon.shape == (3, 5, 1)
    assert np.isfinite(recon).all()


def test_mutated_bytes_do_not_escape_value_error():
    blob = _tiny_blob()
    rng = np.random.default_rng(0)
    for _ in range(80):
        trial = bytearray(blob)
        index = int(rng.integers(0, len(trial)))
        trial[index] ^= int(rng.integers(1, 256))
        try:
            parse_texture_set(bytes(trial))
        except ValueError:
            continue


def test_crc_detects_corruption():
    blob = bytearray(_tiny_blob())
    blob[-1] ^= 0xFF
    try:
        parse_texture_set(bytes(blob))
        raised = False
    except ValueError as exc:
        raised = "crc" in str(exc)
    assert raised
