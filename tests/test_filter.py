"""Filtering identities against the reference decoder."""

import numpy as np

from ntc.container import ACT_RELU, PE_NONE, ChannelRec, build_texture_set, parse_texture_set
from ntc.decode_ref import (
    anisotropic_offsets,
    anisotropic_uv,
    color_bilinear_uv,
    decode_uv,
    feature_trilinear_uv,
    stochastic_aniso_uv,
    stochastic_choice,
    stochastic_uv,
)


def _ts():
    rng = np.random.default_rng(2)
    codes = rng.integers(0, 16, size=(4, 8, 8), dtype=np.uint8)
    scale = np.full(4, 0.02, np.float32)
    w0 = rng.integers(-5, 6, size=(4, 4)).astype(np.int32)
    w1 = rng.integers(-5, 6, size=(3, 4)).astype(np.int32)
    blob = build_texture_set(
        width=8,
        height=8,
        num_mips=4,
        activation=ACT_RELU,
        pe_mode=PE_NONE,
        pe_octaves=0,
        flags=0,
        weight_bits=8,
        group_size=0,
        channels=[ChannelRec(i) for i in range(3)],
        grids=[{"codes": codes, "bits": 4, "scale": scale}],
        layers=[
            {"codes": w0, "bias": np.zeros(4, np.float32), "scale": np.full((4, 1), 0.02, np.float32), "group_size": 0},
            {"codes": w1, "bias": np.zeros(3, np.float32), "scale": np.full((3, 1), 0.05, np.float32), "group_size": 0},
        ],
    )
    return parse_texture_set(blob)


def test_stochastic_choice_partitions_the_square():
    rng = np.random.default_rng(0)
    tx = rng.random(1000)
    ty = rng.random(1000)
    # a uniform u should land in each corner near its weight
    u = rng.random(1000)
    c = stochastic_choice(tx, ty, u)
    # reconstruct weights and the empirical rate
    w = np.stack([(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty], axis=1)
    for k in range(4):
        rate = np.mean(c == k)
        expect = float(w[:, k].mean())
        assert abs(rate - expect) < 0.03, (k, rate, expect)


def test_trilinear_is_a_lerp():
    ts = _ts()
    uv = np.array([[0.3, 0.7], [0.5, 0.5]], np.float32)
    lod = 1.25
    got = feature_trilinear_uv(ts, uv, lod)
    a = decode_uv(ts, uv, 1)
    b = decode_uv(ts, uv, 2)
    want = (1 - 0.25) * a + 0.25 * b
    assert np.max(np.abs(got - want)) < 1e-6


def test_stochastic_mean_matches_color_bilinear():
    ts = _ts()
    uv = np.array([[0.37, 0.62]], np.float32)
    ref = color_bilinear_uv(ts, uv, 0)
    acc = np.zeros_like(ref)
    n = 400
    for frame in range(n):
        acc += stochastic_uv(ts, uv, 0, frame)
    mean = acc / n
    err = float(np.max(np.abs(mean - ref)))
    assert err < 0.05, err


def test_aniso_reference_and_stochastic_mean():
    ts = _ts()
    uv = np.array([[0.4, 0.55]], np.float32)
    dudx = np.array([0.08, 0.01], np.float64)
    dudy = np.array([0.0, 0.02], np.float64)
    offs = anisotropic_offsets(dudx, dudy, 8)
    manual = np.mean([decode_uv(ts, uv + off.reshape(1, 2), 0) for off in offs], axis=0)
    got = anisotropic_uv(ts, uv, 0, dudx, dudy, 8)
    assert np.max(np.abs(manual - got)) < 1e-6
    # expectation of the stochastic taps is color-bilinear at each offset
    color_ref = np.mean([color_bilinear_uv(ts, (uv + off.reshape(1, 2)).astype(np.float32), 0) for off in offs], axis=0)
    acc = np.zeros_like(color_ref)
    n = 300
    for frame in range(n):
        acc += stochastic_aniso_uv(ts, uv, 0, dudx, dudy, 8, frame)
    err = float(np.max(np.abs(acc / n - color_ref)))
    assert err < 0.08, err
