"""Golden bytes for the integer MLP reference."""

import numpy as np

from ntc.ref_int import golden_sha256, int_layer, quantize_int8, triangle_features

# SHA-256 of the int8 activation and the int32 layer output for the fixture.
_GOLDEN = "03465ba9b891480ad90041e5c72149b0a9282d61da74923aa534aea69e9645d8"


def _fixture() -> list[np.ndarray]:
    uv = np.array([[0.25, 0.5], [0.0, 1.0], [0.75, 0.125]], np.float64)
    feat = triangle_features(uv, octaves=2, mip=1, num_mips=4)
    act = quantize_int8(feat, scale=0.02)
    weight = np.array(
        [
            [1, -2, 3, -4, 5, 0, 1, -1, 2, 0],
            [0, 1, -1, 2, -2, 4, 0, 1, -3, 2],
        ],
        np.int8,
    )
    # triangle_features with 2 octaves is 2 * (2 xy + 2 xy2) + lod2 = 10. Good.
    y = int_layer(act, weight, multiplier=3, shift=4, bias_q=np.array([1, -2], np.int32))
    return [act, y]


def test_integer_mlp_golden_is_stable():
    arrays = _fixture()
    digest = golden_sha256(arrays)
    again = golden_sha256(_fixture())
    assert digest == again
    assert arrays[0].dtype == np.int8
    assert arrays[1].dtype == np.int32
    # The hash is the contract. Print once if it changes, then lock it.
    assert digest == _GOLDEN
