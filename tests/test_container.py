import numpy as np

from ntc.container import (
    HEADER_BYTES,
    ChannelRec,
    build_texture_set,
    file_bpp,
    parse_texture_set,
    payload_bytes,
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


def test_crc_detects_corruption():
    blob = bytearray(_tiny_blob())
    blob[-1] ^= 0xFF
    try:
        parse_texture_set(bytes(blob))
        raised = False
    except ValueError as exc:
        raised = "crc" in str(exc)
    assert raised
