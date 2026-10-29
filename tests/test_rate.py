"""Context coder, rate report, PSNR search, and the latent rate term."""

import numpy as np
import torch

from ntc.container import (
    FLAG_LATENT_CTX,
    FLAG_LATENT_RANGE,
    HEADER_BYTES,
    ChannelRec,
    build_texture_set,
    parse_texture_set,
    rate_report,
)
from ntc.encode import TrainConfig, _code_rate_per_texel, encode_arrays
from ntc.model import NtcNet
from ntc.range_coder import range_decode_ctx, range_encode, range_encode_ctx


def _net() -> NtcNet:
    return NtcNet(
        image_channels=1,
        grids=[(8, 8, 2, 4)],
        hidden=[],
        activation=1,
        pe_mode=0,
        pe_octaves=0,
        weight_bits=8,
        group_size=0,
        quant_proxy="ste",
        activation_requant=False,
        qat_latents=True,
        qat_weights=True,
    )


def test_context_roundtrip_and_beats_order1_on_a_ramp():
    raw = bytes([min(255, (i * 3) // 8) for i in range(800)])
    enc = range_encode_ctx(raw)
    assert range_decode_ctx(enc, len(raw)) == raw
    assert len(enc) < len(range_encode(raw))


def test_best_does_not_set_context_flag():
    codes = np.zeros((2, 16, 16), np.uint8)
    codes[:, :, :] = 4
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
    best = parse_texture_set(build_texture_set(**common, latent_coder="best"))
    assert not (best.flags & FLAG_LATENT_CTX)
    ctx = parse_texture_set(build_texture_set(**common, latent_coder="context"))
    assert ctx.flags & FLAG_LATENT_CTX
    assert not (ctx.flags & FLAG_LATENT_RANGE)
    np.testing.assert_array_equal(ctx.grids[0].codes, codes)
    rates = rate_report(ctx)
    assert rates["header_bytes"] == HEADER_BYTES == 96
    assert rates["network_bytes"] > 0
    assert rates["network_bytes"] < rates["payload_bytes"]


def test_constant_codes_have_zero_rate_and_scales_get_no_rate_grad():
    net = _net()
    with torch.no_grad():
        net.latents[0].fill_(3.0)
    assert float(_code_rate_per_texel(net, 64).detach()) == 0.0
    net.latents[0].data[0, 0, 0] = 0.0
    net.latents[0].data[0, 0, 1] = 8.0
    rate = _code_rate_per_texel(net, 64)
    assert float(rate.detach()) > 0.0
    rate.backward()
    assert net.latents[0].grad is not None
    assert float(net.latents[0].grad.abs().sum()) > 0.0
    assert net.log_scales[0].grad is None


def test_rd_lambda_trains_and_is_recorded():
    img = np.linspace(0.0, 1.0, 16 * 16, dtype=np.float32).reshape(1, 16, 16)
    cfg = TrainConfig(
        steps=2,
        hidden=(),
        grids=[(8, 8, 2, 4)],
        adaptive_pool=1,
        seed_pool=1,
        with_ssim=False,
        rd_lambda=1e-3,
        latent_coder="best",
    )
    blob, stats = encode_arrays(img, [0], cfg)
    assert stats["rd_lambda"] == 1e-3
    assert "rate_search" not in stats
    ts = parse_texture_set(blob)
    assert not (ts.flags & FLAG_LATENT_CTX)
    assert rate_report(ts)["header_bytes"] == 96


def test_target_psnr_picks_smallest_file():
    img = np.linspace(0.0, 1.0, 16 * 16, dtype=np.float32).reshape(1, 16, 16)
    cfg = TrainConfig(
        steps=0,
        hidden=(),
        adaptive_pool=1,
        seed_pool=1,
        with_ssim=False,
        target_psnr=0.0,
        search_steps=0,
        latent_coder="best",
    )
    blob, stats = encode_arrays(img, [0], cfg)
    assert len(blob) > 96
    meeting = [r for r in stats["rate_search"] if r["psnr_u8"] >= 0.0]
    picked = min(meeting, key=lambda r: (r["file_bytes"], r["target_bpp"]))
    assert stats["rate_search_picked_bpp"] == picked["target_bpp"]
    assert stats["target_psnr"] == 0.0
    assert len(stats["rate_search"]) >= 8
    assert parse_texture_set(blob).width == 16
