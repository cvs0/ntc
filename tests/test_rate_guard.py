"""The file must land at or under its rate label (b1 came out 3 to 6% over)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ntc.container import parse_texture_set, rate_report
from ntc.encode import TrainConfig, export_network
from ntc.train_fast import (
    RATE_GOAL_CEIL,
    RATE_MARGIN,
    RATE_MULT_MAX,
    _build_nets,
    encode_arrays_fast,
    rate_overhead_bpp,
    steer_rate_values,
    trim_latents_to_rate,
)


def test_steer_over_budget_lowers_goal_and_raises_weight():
    goal, mult = steer_rate_values(1.0, 1.0, 1.0, 1.05)
    assert goal < 1.0
    assert goal == pytest.approx(1.0 / 1.05)
    assert mult == 2.0
    for _ in range(20):
        goal, mult = steer_rate_values(goal, mult, 1.0, 1.5)
    assert mult == RATE_MULT_MAX
    assert goal >= 0.5


def test_steer_under_budget_relaxes_up_to_the_ceiling():
    goal, mult = steer_rate_values(0.8, 8.0, 1.0, 0.9)
    assert mult == 4.0
    assert 0.8 < goal <= 1.0
    goal, mult = steer_rate_values(1.0, 1.0, 1.0, 0.5)
    assert (goal, mult) == (pytest.approx(1.05), 1.0)
    for _ in range(20):
        goal, mult = steer_rate_values(goal, mult, 1.0, 0.5)
    assert goal == pytest.approx(RATE_GOAL_CEIL)
    # A measurement that failed leaves the controller alone.
    assert steer_rate_values(0.9, 2.0, 1.0, float("nan")) == (0.9, 2.0)


def _net(grids, seed=0):
    img = torch.rand(3, 64, 64, generator=torch.Generator().manual_seed(seed))
    cfg = TrainConfig(fast=True, device="cpu", pca_init=False, hidden=(16,))
    return _build_nets(img, cfg, grids, [seed], torch.device("cpu"))[0]


def test_overhead_matches_real_export():
    net = _net([(32, 32, 4, 4), (16, 16, 4, 4)])
    with torch.no_grad():
        for lat in net.latents:
            lat.uniform_(0, 15)
    blob = export_network(net, [0, 1, 2], 64, 64, 7, latent_deflate=True, latent_coder="best")
    rates = rate_report(parse_texture_set(blob))
    want = 8.0 * (rates["payload_bytes"] - rates["latent_code_bytes"]) / (64 * 64)
    got = rate_overhead_bpp(net, [0, 1, 2], 64, 64, 7, True, "best")
    assert got == pytest.approx(want)
    assert got > 0


def test_trim_fits_the_limit():
    net = _net([(32, 32, 4, 4)])
    with torch.no_grad():
        net.latents[0].copy_(torch.randn(net.latents[0].shape, generator=torch.Generator().manual_seed(3)) * 2 + 7)

    def _export(n):
        data = export_network(n, [0, 1, 2], 64, 64, 7, latent_deflate=True, latent_coder="best")
        return data, float(rate_report(parse_texture_set(data))["total_bpp"])

    _blob, start = _export(net)
    limit = 0.7 * start
    blob, bpp, snapped = trim_latents_to_rate(net, _export, limit)
    assert snapped is not None and snapped > 0
    assert bpp <= limit
    assert float(rate_report(parse_texture_set(blob))["total_bpp"]) == pytest.approx(bpp)
    # The count bisection lands close under the limit, not far below it.
    assert bpp >= (1 - 2 * RATE_MARGIN) * limit


def _cfg(**kw):
    base = dict(
        steps=100, hidden=(16,), grids=[(32, 32, 4, 4), (16, 16, 4, 4)], fast=True,
        adaptive_pool=1, seed_pool=1, with_ssim=False, log_every=10, mip_levels=2,
        seed=0, compile_step="off", cuda_graph="off", precision="fp32", device="cpu",
        batch_pixels=1024, batch_mip="flat", quant_schedule=True, entropy_rate=True,
        target_bpp=1.0, rate="latent", latent_rate_scale=0.5, eval_every=50,
        early_stop=16, job_name="rate", quant_freeze_at=0.9, latent_deflate=True,
        latent_coder="best",
    )
    base.update(kw)
    return TrainConfig(**base)


def test_total_bpp_lands_at_or_under_target(monkeypatch):
    monkeypatch.delenv("NTC_RATE_GUARD", raising=False)
    rng = np.random.default_rng(0)
    img = rng.random((3, 64, 64)).astype(np.float32)
    target = 1.0
    _blob, stats = encode_arrays_fast(img, [0, 1, 2], _cfg(target_bpp=target))
    assert stats["rate_guard"] is True
    assert stats["rate_overhead_bpp"] > 0
    assert stats["rate_budget_bpp"] == pytest.approx(
        target * (1 - RATE_MARGIN) - stats["rate_overhead_bpp"]
    )
    assert stats["total_bpp"] <= target
    # Noise codes do not fit 1 bpp on their own, so this run needs the trim.
    assert stats["rate_trim"] is not None
    assert stats["total_bpp"] >= (1 - 2 * RATE_MARGIN) * target


def test_guard_off_keeps_old_behaviour(monkeypatch):
    monkeypatch.setenv("NTC_RATE_GUARD", "0")
    img = np.random.default_rng(0).random((3, 64, 64)).astype(np.float32)
    _blob, stats = encode_arrays_fast(img, [0, 1, 2], _cfg(steps=20))
    assert stats["rate_guard"] is False
    assert stats["rate_trim"] is None
    assert stats["rate_budget_bpp"] is None
