"""Fast trainer: same mip-0 math as NtcNet, halving, and the off-by-default flag."""

import numpy as np
import torch

from ntc.encode import TrainConfig, encode_arrays
from ntc.model import NtcNet, bilinear_codes_torch
from ntc.train_fast import (
    NEAR_INIT_DB,
    StackedTrain,
    _mesh_uv,
    bilinear_batched,
    encode_arrays_fast,
    fullres_batch_floor,
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


def test_fullres_grid_raises_a_sparse_batch():
    image = 2048 * 2048
    batch, mode = fullres_batch_floor(2048, 2048, [(2048, 2048, 2, 4), (1024, 1024, 2, 4)], 256 * 1024, "area")
    assert batch == image // 4
    assert mode == "flat"
    kept, kept_mode = fullres_batch_floor(2048, 2048, [(1024, 1024, 4, 4), (512, 512, 4, 4)], 256 * 1024, "area")
    assert kept == 256 * 1024
    assert kept_mode == "area"
    full, full_mode = fullres_batch_floor(2048, 2048, [(2048, 2048, 2, 4)], 0, "flat")
    assert full == 0
    assert full_mode == "flat"


def test_fullres_sparse_batch_trains_above_init():
    """A 1:1 latent with a 1/16 area batch must not finish next to its init.

    The trainer raises that batch to a flat quarter of the frame. The image
    is 32 so the run stays on CPU. pca is off so the init is not already the
    answer.
    """
    rng = np.random.default_rng(4)
    n = 32
    yy, xx = np.mgrid[0:n, 0:n]
    img = np.stack(
        [
            0.5 + 0.4 * np.sin(xx / 1.7) * np.cos(yy / 2.1),
            0.2 + 0.6 * ((xx // 4 + yy // 3) % 2),
            np.clip(rng.random((n, n)) * 0.35 + (xx + yy) / (2 * n), 0, 1),
        ],
        0,
    ).astype(np.float32)
    cfg = TrainConfig(
        steps=80,
        hidden=(16,),
        grids=[(n, n, 2, 4), (n // 2, n // 2, 2, 4)],
        seed=1,
        seed_pool=1,
        adaptive_pool=1,
        with_ssim=False,
        mip_levels=4,
        mip_loss="footprint",
        warmup_steps=8,
        eval_every=0,
        early_stop=0,
        fast=True,
        device="cpu",
        pca_init=False,
        batch_pixels=max(1, (n * n) // 16),
        batch_mip="area",
        compile_step="off",
        cuda_graph="off",
        log_every=10**9,
    )
    _blob, stats = encode_arrays_fast(img, [1, 1, 1], cfg)
    assert stats["batch_pixels"] == (n * n) // 4
    assert stats["batch_mip"] == "flat"
    assert stats["psnr_u8_trained"] >= stats["psnr_u8_init"] + NEAR_INIT_DB
