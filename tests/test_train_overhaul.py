"""Switches for the training step. Each one can be turned off."""

import numpy as np
import torch

from ntc.encode import TrainConfig
from ntc.grids import grid_weights, rd_rebalance
from ntc.train_fast import (
    _step_lr,
    batch_lr_scale,
    beats_plateau,
    crop_indices,
    gather_backend,
    mlp_backend,
    overhaul_plan,
    pipeline_plan,
    plateau_limits,
    portable_step_plan,
    size_batch_for_vram,
)


def test_compile_and_triton_fall_back_off_cuda():
    assert gather_backend("cpu", "auto") == "eager"
    assert gather_backend("mps", "on") == "eager"
    assert gather_backend("cuda", "auto") == "triton"
    assert gather_backend("cuda", "off") == "eager"
    assert mlp_backend("cpu", "auto", "auto") == "eager"
    assert mlp_backend("mps", "auto", "auto") == "eager"
    assert mlp_backend("cuda", "auto", "auto") == "inductor"
    assert mlp_backend("cuda", "off", "auto") == "cublas"
    assert mlp_backend("cuda", "auto", "off") == "eager"


def test_cuda_graph_stays_off_without_cuda_and_on_when_compile_is_off():
    assert portable_step_plan("auto", "auto", "cosine", hip=False, win32=False) == ("auto", False)
    assert portable_step_plan("off", "auto", "warmup_cosine", hip=False, win32=False) == ("off", True)
    assert portable_step_plan("auto", "off", "cosine", hip=False, win32=False) == ("auto", False)
    cpu = overhaul_plan(
        "cpu",
        TrainConfig(compile_step="auto", cuda_graph="auto", schedule="warmup_cosine", fused_gather="auto", fused_mlp="auto"),
    )
    assert cpu["compile"] == "eager"
    assert cpu["cuda_graph"] is False
    cuda = overhaul_plan(
        "cuda",
        TrainConfig(compile_step="auto", cuda_graph="auto", schedule="warmup_cosine", fused_gather="auto", fused_mlp="auto"),
    )
    assert cuda["compile"] == "inductor"
    assert cuda["cuda_graph"] is True
    assert cuda["cuda_graph_kind"] == "inductor"
    assert cuda["gather"] == "triton"


def test_warmup_cosine_ramps_then_falls():
    cfg = TrainConfig(lr=1.0, steps=20, warmup_steps=4, schedule="warmup_cosine")
    assert abs(_step_lr(cfg, 0) - 0.25) < 1e-9
    assert abs(_step_lr(cfg, 3) - 1.0) < 1e-9
    assert _step_lr(cfg, 19) < 0.2
    bare = TrainConfig(lr=1.0, steps=20, warmup_steps=0, schedule="cosine")
    assert abs(_step_lr(bare, 0) - 1.0) < 1e-6


def test_sqrt_and_linear_batch_scales_and_vram_sizing():
    assert batch_lr_scale(8, 2, "linear") == 4.0
    assert abs(batch_lr_scale(4, 1, "sqrt") - 2.0) < 1e-9
    assert batch_lr_scale(8, 2, "off") == 1.0
    grids = [(64, 64, 4, 4)]
    small = size_batch_for_vram(1_000_000, grids, 8 * (1 << 30), 2048 * 2048)
    large = size_batch_for_vram(1_000_000, grids, 80 * (1 << 30), 2048 * 2048)
    assert small <= 1_000_000
    assert large >= 1_000_000
    assert size_batch_for_vram(123, grids, 0, 1000) == 123


def test_resident_sampling_does_not_copy_on_cpu_and_flags_a_gpu_roundtrip():
    idx = crop_indices(32, 32, 64, torch.device("cpu"), resident=True)
    assert idx.device.type == "cpu"
    assert idx.dtype == torch.long
    assert pipeline_plan("cuda", True)["host_roundtrip"] is False
    assert pipeline_plan("mps", True)["host_roundtrip"] is False
    assert pipeline_plan("cuda", False)["host_roundtrip"] is True
    assert pipeline_plan("cpu", False)["host_roundtrip"] is False


def test_plateau_patience_and_delta_are_configurable():
    assert plateau_limits(0, 0.0, 16) == (16, 0.01)
    assert plateau_limits(8, 0.05, 16) == (8, 0.05)
    assert beats_plateau(30.06, 30.0, 0.05) is True
    assert beats_plateau(30.04, 30.0, 0.05) is False
    assert beats_plateau(float("nan"), 30.0, 0.05) is False


def test_rd_allocation_moves_channels_toward_the_varying_plane():
    image = np.zeros((4, 16, 16), np.float32)
    image[0] = np.linspace(0.0, 1.0, 16, dtype=np.float32)
    image[1] = np.linspace(0.0, 1.0, 16, dtype=np.float32)[:, None]
    weights = grid_weights(image, 2)
    assert weights[0] > weights[1]
    grids = [(32, 32, 8, 4), (16, 16, 8, 4)]
    moved = rd_rebalance(grids, [0.01, 10.0], 64, 64, 8.0)
    assert moved[1][2] > grids[1][2]
    assert moved[0][2] < grids[0][2]
    assert moved[0][0] == 32 and moved[1][3] == 4
    same = rd_rebalance(grids, [1.0, 1.0], 64, 64, 8.0)
    assert [g[2] for g in same] == [8, 8]
