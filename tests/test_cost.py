import numpy as np

from ntc.container import ACT_RELU, ChannelRec, build_texture_set, parse_texture_set
from ntc.cost import macs, sample_cost, workgroup_latent_bytes


def test_macs_and_tile_reuse():
    codes = np.zeros((4, 8, 8), np.uint8)
    blob = build_texture_set(
        width=16,
        height=16,
        num_mips=4,
        activation=ACT_RELU,
        pe_mode=0,
        pe_octaves=0,
        flags=0,
        weight_bits=8,
        group_size=0,
        channels=[ChannelRec(0), ChannelRec(1)],
        grids=[{"codes": codes, "bits": 8, "scale": np.ones(4, np.float32)}],
        layers=[
            {
                "codes": np.zeros((2, 4), np.int32),
                "bias": np.zeros(2, np.float32),
                "scale": np.ones((2, 1), np.float32),
                "group_size": 0,
            }
        ],
    )
    ts = parse_texture_set(blob)
    assert macs(ts) == 8
    cost = sample_cost(ts, taps=1)
    assert cost["macs"] == 8
    assert cost["latent_bytes_upper"] == 16  # 4 taps * 4 channels
    tile = workgroup_latent_bytes(ts, group=8)
    assert tile["bytes_per_texel_unique"] < tile["bytes_per_texel_upper"]
