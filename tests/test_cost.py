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


def _shipped_like(group_size=0, act_scales=None):
    rng = np.random.default_rng(0)
    grids = [
        {"codes": rng.integers(0, 16, (4, 32, 32)).astype(np.uint8), "bits": 4,
         "scale": np.full(4, 0.1, np.float32)},
        {"codes": rng.integers(0, 4, (2, 8, 8)).astype(np.uint8), "bits": 2,
         "scale": np.full(2, 0.1, np.float32)},
    ]
    layers = []
    for inn, out in [(6, 32), (32, 9)]:
        ng = 1 if group_size == 0 else -(-inn // group_size)
        layers.append({"codes": rng.integers(-127, 128, (out, inn)).astype(np.int32),
                       "bias": np.zeros(out, np.float32),
                       "scale": np.full((out, ng), 0.01, np.float32),
                       "group_size": group_size})
    blob = build_texture_set(
        width=32, height=32, num_mips=6, activation=ACT_RELU, pe_mode=0, pe_octaves=0,
        flags=0, weight_bits=8, group_size=group_size,
        channels=[ChannelRec(i) for i in range(9)], grids=grids, layers=layers,
        act_scales=act_scales,
    )
    return parse_texture_set(blob)


def test_fast_planes_power_of_two_records():
    from ntc.cost import fast_planes

    assert fast_planes(12, 4) == [(0, 8, 32), (8, 4, 16)]
    assert fast_planes(6, 8) == [(0, 4, 32), (4, 2, 16)]
    assert fast_planes(2, 2) == [(0, 2, 4)]
    assert fast_planes(3, 4) == [(0, 2, 8), (2, 1, 4)]
    for c in range(1, 17):
        for b in (2, 4, 8):
            planes = fast_planes(c, b)
            assert sum(p[1] for p in planes) == c
            assert all(p[2] in (2, 4, 8, 16, 32) for p in planes)


def test_decode_ops_shipped_shape():
    # Same numbers ntc_decbench prints for results/train_sweep/metal_g16_*.ntc.
    from ntc.cost import decode_ops

    ops = decode_ops(_shipped_like())
    leg, fast = ops["legacy"], ops["fast"]
    assert leg["macs"] == fast["macs"] == 480
    assert (leg["latent_loads"], leg["weight_loads"], leg["scale_loads"]) == (24, 644, 12)
    assert leg["load_bytes"] == 2720 and leg["alu"] == 2209
    assert fast["loads"] == 8 and fast["load_bytes"] == 32
    assert fast["uniform_words"] == 576 and fast["alu"] == 243


def test_cost_cli(tmp_path, capsys):
    from ntc.container import parse_texture_set as _p  # noqa: F401
    from ntc.cost import main

    ts = _shipped_like()
    f = tmp_path / "s.ntc"
    f.write_bytes(ts.raw)
    assert main([str(f)]) == 0
    out = capsys.readouterr().out
    assert "legacy loads 680" in out and "fast   loads 8" in out
