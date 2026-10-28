"""Fast trainer: same mip-0 math as NtcNet, halving, and the off-by-default flag."""

import numpy as np
import torch

from ntc.encode import TrainConfig, encode_arrays
from ntc.model import NtcNet, bilinear_codes_torch
from ntc.train_fast import (
    StackedTrain,
    _mesh_uv,
    bilinear_batched,
    encode_arrays_fast,
    scaled_lr,
)


def _net() -> NtcNet:
    torch.manual_seed(0)
    net = NtcNet(
        image_channels=3,
        grids=[(8, 8, 2, 4)],
        hidden=[8],
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
    with torch.no_grad():
        net.latents[0].normal_(7, 1.0)
    return net


def test_bilinear_batch_matches_reference():
    torch.manual_seed(1)
    codes = torch.randn(3, 5, 6)
    uv = _mesh_uv(8, 8, torch.device("cpu"))
    ref = bilinear_codes_torch(codes, uv)
    got = bilinear_batched(codes.unsqueeze(0), uv)[0]
    assert torch.allclose(ref, got)


def test_stacked_matches_net_including_mip1():
    net = _net()
    stacked = StackedTrain([net])
    uv0 = _mesh_uv(8, 8, torch.device("cpu"))
    uv1 = _mesh_uv(4, 4, torch.device("cpu"))
    assert torch.allclose(net(uv0, mip=0, max_mip=1), stacked.predict(uv0, 0, 1)[0])
    assert torch.allclose(net(uv1, mip=1, max_mip=1), stacked.predict(uv1, 1, 1)[0], atol=1e-5)


def test_fast_flag_off_is_the_historical_loop_and_on_matches_psnr():
    img = np.linspace(0.0, 1.0, 3 * 16 * 16, dtype=np.float32).reshape(3, 16, 16)
    common = dict(
        steps=3,
        hidden=(8,),
        grids=[(8, 8, 2, 4)],
        adaptive_pool=1,
        seed_pool=1,
        with_ssim=False,
        log_every=10**9,
        mip_levels=2,
        seed=1,
        compile_step="off",
        cuda_graph="off",
    )
    _b0, slow = encode_arrays(img, [0, 1, 2], TrainConfig(fast=False, **common))
    assert "fast" not in slow or slow.get("fast") is not True
    _b1, fast = encode_arrays_fast(img, [0, 1, 2], TrainConfig(fast=True, **common))
    assert fast["fast"] is True
    assert fast["host_syncs_per_step"] == 0
    assert abs(slow["metrics"]["psnr_u8"] - fast["metrics"]["psnr_u8"]) < 1e-4


def test_halving_drops_seeds():
    rng = np.random.default_rng(0)
    img = rng.random((3, 16, 16), dtype=np.float32)
    cfg = TrainConfig(
        steps=6,
        hidden=(8,),
        grids=[(8, 8, 2, 4)],
        adaptive_pool=1,
        seed_pool=4,
        with_ssim=False,
        fast=True,
        log_every=10**9,
        mip_levels=2,
        seed=0,
        compile_step="off",
        cuda_graph="off",
        halve_steps=2,
    )
    _blob, stats = encode_arrays_fast(img, [0, 1, 2], cfg)
    alive = [row["alive"] for row in stats["halving"]]
    assert alive[0] == [0, 1, 2, 3]
    assert len(alive[-1]) == 1
    assert stats["winner_seed"] in alive[-1]


def test_lr_scales_linearly_with_batch():
    assert scaled_lr(1e-3, 65536, 16384) == 4e-3
    assert scaled_lr(1e-3, 100, 0) == 1e-3
