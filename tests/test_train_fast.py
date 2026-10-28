"""Fast trainer: same mip-0 math as NtcNet, halving, and the off-by-default flag."""

from pathlib import Path

import numpy as np
import pytest
import torch

from ntc.encode import (
    TrainConfig,
    _code_scale,
    _fit_readout,
    _ridge_lstsq,
    checkpoint_is_finite,
    discard_nonfinite_checkpoint,
    encode_arrays,
    export_network,
)
from ntc.model import NtcNet, bilinear_codes_torch
from ntc.train_fast import (
    LATENT_ADAM_EPS,
    NEAR_INIT_DB,
    StackedTrain,
    _adam_on,
    _assign_lr,
    _better_psnr,
    _cover_mip_recompiles,
    _mesh_uv,
    _psnr_from_mse,
    _psnr_per_seed,
    _reported_best_val,
    _require_finite_stats,
    _rewind_if_below_init,
    bilinear_batched,
    encode_arrays_fast,
    fit_latent_grids,
    fullres_batch_floor,
    latent_coder_ratio,
    portable_step_plan,
    preferred_compile_mode,
    rate_overshoots,
    refill_sample_indices,
    scaled_lr,
    static_sample_indices,
    wide_latent,
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


def test_one_step_builds_each_mip_chain_once(monkeypatch):
    import ntc.train_fast as tf

    calls = {"n": 0}
    real = tf.integer_box

    def wrapped(codes):
        calls["n"] += 1
        return real(codes)

    monkeypatch.setattr(tf, "integer_box", wrapped)
    img = np.zeros((3, 16, 16), dtype=np.float32)
    cfg = TrainConfig(
        steps=1,
        hidden=(8,),
        grids=[(16, 16, 2, 4)],
        seed=0,
        seed_pool=1,
        adaptive_pool=1,
        with_ssim=False,
        mip_levels=4,
        fast=True,
        device="cpu",
        pca_init=False,
        compile_step="off",
        cuda_graph="off",
        log_every=10**9,
        latent_deflate=False,
    )
    encode_arrays_fast(img, [0, 1, 2], cfg)
    # Four mips used to rebuild the pyramid (0+1+2+3 boxes). One prepare is three.
    assert calls["n"] <= 4


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


def test_assign_lr_fills_a_capturable_tensor():
    slot = torch.tensor(0.1)
    group = {"lr": slot}
    _assign_lr(group, 0.25)
    assert group["lr"] is slot
    assert float(slot) == 0.25
    plain = {"lr": 0.1}
    _assign_lr(plain, 0.3)
    assert plain["lr"] == 0.3


def test_latent_adam_step_matches_lr_at_2k_grad_scale():
    """A 2048 mean loss yields per-texel grads near 1e-10.

    Default eps 1e-8 keeps about 1% of the learning rate there. The latent
    group uses 1e-15, so the first step stays next to the learning rate.
    """
    torch.manual_seed(0)
    net = _net()
    stacked = StackedTrain([net])
    cfg = TrainConfig(steps=1, lr=1e-2, latent_lr_mult=1.0, fast=True, compile_step="off", cuda_graph="off")
    opt = _adam_on(stacked, cfg, capturable=False)
    latent = next(group for group in opt.param_groups if group.get("name") == "latent")
    mlp = next(group for group in opt.param_groups if group.get("name") == "mlp")
    assert latent["eps"] == LATENT_ADAM_EPS == 1e-15
    assert latent["betas"] == (0.9, 0.99)
    assert mlp["eps"] == 1e-8
    grad = 1e-10
    probe = torch.nn.Parameter(torch.zeros(4, 8, 8))
    step_opt = torch.optim.Adam([{"params": [probe], "lr": 1e-2, "eps": LATENT_ADAM_EPS, "betas": (0.9, 0.99)}])
    probe.grad = torch.full_like(probe, grad)
    before = probe.detach().clone()
    step_opt.step()
    delta = float((probe.detach() - before).abs().mean())
    assert delta > 0.5 * 1e-2
    default = torch.nn.Parameter(torch.zeros(4, 8, 8))
    slow = torch.optim.Adam([default], lr=1e-2)
    default.grad = torch.full_like(default, grad)
    base = default.detach().clone()
    slow.step()
    shrunk = float((default.detach() - base).abs().mean())
    assert shrunk < 0.05 * 1e-2


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
    deep, deep_mode = fullres_batch_floor(
        2048, 2048, [(1024, 1024, 12, 8), (512, 512, 12, 8)], 256 * 1024, "area"
    )
    assert deep == image // 4
    assert deep_mode == "flat"
    # b5 after the 4-bit ranking: 8 channels at 1024. The 256K area batch
    # stayed at 11.91 dB against an init of 10.58. 4 channels at 1024 did not.
    wide, wide_mode = fullres_batch_floor(
        2048, 2048, [(1024, 1024, 8, 4), (512, 512, 8, 4)], 256 * 1024, "area"
    )
    assert wide == image // 4
    assert wide_mode == "flat"
    narrow, narrow_mode = fullres_batch_floor(
        2048, 2048, [(1024, 1024, 4, 4), (512, 512, 4, 4)], 256 * 1024, "area"
    )
    assert narrow == 256 * 1024
    assert narrow_mode == "area"
    # MetalPlates b2.5 at scale 0.397: 8 channels on the coarse grid, 4 on
    # the finest. The dense batch is for an 8-channel finest lattice.
    coarse8, coarse8_mode = fullres_batch_floor(
        2048, 2048, [(1024, 1024, 4, 4), (512, 512, 8, 4)], 256 * 1024, "area"
    )
    assert coarse8 == 256 * 1024
    assert coarse8_mode == "area"
    # PavingStones b2.5: 12 channels is the finest lattice, so the flat
    # quarter frame stays.
    paving, paving_mode = fullres_batch_floor(
        2048, 2048, [(512, 512, 12, 4), (256, 256, 12, 4)], 256 * 1024, "area"
    )
    assert paving == image // 4
    assert paving_mode == "flat"


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


def _checker(n: int) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n]
    hi = ((xx + yy) & 1).astype(np.float32)
    lo = (0.5 + 0.4 * np.sin(xx / 7.0) * np.cos(yy / 11.0)).astype(np.float32)
    chans = []
    for k in range(9):
        if k < 3:
            base = (0.2 + 0.6 * hi) if k != 1 else (0.8 - 0.6 * hi)
            v = 0.7 * base + 0.3 * lo
        elif k < 6:
            sign = 1.0 if k % 2 == 0 else -1.0
            v = np.clip((0.5 + 0.5 * hi * sign) * 0.5 + 0.5, 0.0, 1.0)
        else:
            v = lo * (0.4 + 0.1 * k) + 0.05 * hi
        chans.append(np.clip(v, 0.0, 1.0).astype(np.float32))
    return np.stack(chans, 0)


def test_wide_latent_is_eight_channels_or_more():
    assert wide_latent([(2048, 2048, 4, 4), (1024, 1024, 12, 4)])
    assert wide_latent([(2048, 2048, 8, 4), (1024, 1024, 8, 4)])
    assert not wide_latent([(2048, 2048, 4, 4), (1024, 1024, 4, 4)])


def test_windows_rocm_uses_a_hip_graph_instead_of_compile():
    # 9070 eager 2.35 steps/s, 5070 reduce-overhead 19.99. Compile does not
    # run on Windows ROCm, so that platform captures a HIP graph.
    assert portable_step_plan("auto", "auto", "cosine", hip=True, win32=True) == ("off", True)
    assert portable_step_plan("auto", "auto", "cosine", hip=False, win32=False) == ("auto", False)
    assert portable_step_plan("auto", "auto", "constant", hip=False, win32=False) == ("auto", True)
    assert portable_step_plan("auto", "off", "cosine", hip=True, win32=True) == ("auto", False)


def test_cuda_compile_mode_is_reduce_overhead():
    assert preferred_compile_mode("cuda") == "reduce-overhead"
    assert preferred_compile_mode("cpu") == "default"


def test_static_indices_are_refilled_in_place():
    uv = torch.zeros(8, 2)
    mips = [{"uv": uv, "tgt": torch.zeros(1, 8, 1), "weight": 1.0}]
    slots = static_sample_indices(mips, 4, "flat", torch.device("cpu"))
    assert slots[0] is not None and slots[0].shape == (4,)
    ptr = slots[0].data_ptr()
    torch.manual_seed(1)
    refill_sample_indices(slots, mips, 4, "flat")
    assert slots[0].data_ptr() == ptr
    assert int(slots[0].min()) >= 0
    assert int(slots[0].max()) < 8


def test_wide_checker_stays_above_init():
    """A 4+12 grid's mip-1 term used to be 50x mip 0 and the file fell under the init.

    The coarse terms are capped at the mip-0 term. On this 32 stand-in the
    file then finishes above the init at the historical learning rate.
    """
    n = 32
    sems = list(range(9))
    common = dict(
        steps=600,
        hidden=(32,),
        seed=1,
        seed_pool=1,
        adaptive_pool=1,
        with_ssim=False,
        mip_levels=12,
        mip_loss="footprint",
        warmup_steps=400,
        schedule="cosine",
        eval_every=0,
        early_stop=0,
        fast=True,
        device="cpu",
        pca_init=True,
        batch_pixels=max(1, (n * n) // 4),
        batch_mip="flat",
        compile_step="off",
        cuda_graph="off",
        log_every=10**9,
        latent_deflate=False,
    )
    img = _checker(n)
    _blob, wide = encode_arrays_fast(
        img, sems, TrainConfig(grids=[(n, n, 4, 4), (n // 2, n // 2, 12, 4)], **common)
    )
    assert wide["psnr_u8_trained"] > wide["psnr_u8_init"]
    _blob, narrow = encode_arrays_fast(
        img,
        sems,
        TrainConfig(
            grids=[(n, n, 4, 4), (n // 2, n // 2, 4, 4)],
            **{**common, "steps": 1, "warmup_steps": 0},
        ),
    )
    assert narrow["fast"] is True


def test_nan_mse_is_not_a_perfect_psnr():
    with pytest.raises(RuntimeError, match="non-finite PSNR"):
        _psnr_from_mse(float("nan"))
    assert _psnr_from_mse(0.0) == 100.0
    # abs(100 - NaN) > 1 is false, which is how best_val_psnr stayed 100
    # while the file PSNR was NaN.
    assert not torch.tensor(_reported_best_val(100.0, float("nan"))).isfinite()
    assert _reported_best_val(12.67, 33.07) == 33.07
    assert _reported_best_val(float("nan"), 33.0) == 33.0
    assert _better_psnr(float("nan"), 10.0) is False
    assert _better_psnr(12.0, float("nan")) is True
    with pytest.raises(RuntimeError, match="non-finite"):
        _require_finite_stats({"psnr_u8_init": float("nan"), "metrics": {"psnr_u8": 30.0}})


def test_quiet_bf16_forward_is_finite():
    torch.manual_seed(0)
    net = _net()
    with torch.no_grad():
        net.linears[0].weight.mul_(0.0)
    stacked = StackedTrain([net])
    uv = torch.tensor([[0.25, 0.75]])
    pred = stacked.predict(uv, 0, 1, torch.bfloat16, "cpu")
    assert torch.isfinite(pred).all()


def test_export_rejects_a_nan_latent():
    net = _net()
    with torch.no_grad():
        net.latents[0].fill_(float("nan"))
    with pytest.raises(RuntimeError, match="non-finite latent"):
        export_network(net, [0, 1, 2], 8, 8, 2)


def test_non_finite_readout_does_not_write_the_layer(monkeypatch):
    net = _net()
    before = net.linears[-1].weight.detach().clone()

    def _nan_solve(*_args, **_kwargs):
        return torch.full((8, 3), float("nan"))

    monkeypatch.setattr("ntc.encode._ridge_lstsq", _nan_solve)
    with pytest.raises(RuntimeError, match="non-finite readout"):
        _fit_readout(net, torch.rand(3, 8, 8))
    assert torch.equal(net.linears[-1].weight, before)


def test_wide_fp32_step_keeps_init_and_trained_finite():
    """The 4+12 grid is the b15 shape that exported NaN at 989362d in fp32.

    One Adam step is enough. Init is the PCA readout, taken before the step.
    """
    rng = np.random.default_rng(1)
    img = rng.random((3, 32, 32), dtype=np.float32)
    img[0] += np.linspace(0.0, 0.2, 32, dtype=np.float32)[None, :]
    img = np.clip(img, 0.0, 1.0)
    _blob, stats = encode_arrays_fast(
        img,
        [0, 1, 2],
        TrainConfig(
            steps=2,
            hidden=(32,),
            grids=[(32, 32, 4, 4), (16, 16, 12, 4)],
            fast=True,
            adaptive_pool=1,
            seed_pool=1,
            with_ssim=False,
            log_every=10**9,
            mip_levels=3,
            seed=1,
            compile_step="off",
            cuda_graph="off",
            precision="fp32",
            batch_pixels=64,
            batch_mip="flat",
        ),
    )
    assert np.isfinite(stats["psnr_u8_init"])
    assert np.isfinite(stats["psnr_u8_trained"])
    assert np.isfinite(stats["metrics"]["psnr_u8"])
    assert np.isfinite(stats["best_val_psnr"])
    assert stats["steps_ran"] == 2


def test_nonfinite_checkpoint_is_not_resumed(tmp_path: Path):
    net = _net()
    with torch.no_grad():
        net.linears[-1].weight.fill_(float("nan"))
    path = tmp_path / "bad.pt"
    torch.save({"model": net.state_dict(), "opt": {"state": {"exp_avg": torch.tensor([float("nan")])}}, "step": 40}, path)
    assert checkpoint_is_finite(torch.load(path, weights_only=False)) is False
    img = np.linspace(0.0, 1.0, 3 * 16 * 16, dtype=np.float32).reshape(3, 16, 16)
    _blob, stats = encode_arrays_fast(
        img,
        [0, 1, 2],
        TrainConfig(
            steps=1,
            hidden=(8,),
            grids=[(8, 8, 2, 4)],
            fast=True,
            adaptive_pool=1,
            seed_pool=1,
            with_ssim=False,
            log_every=10**9,
            mip_levels=2,
            seed=1,
            compile_step="off",
            cuda_graph="off",
            precision="fp32",
            resume=str(path),
            ckpt=str(tmp_path / "next.pt"),
        ),
    )
    assert not path.is_file()
    assert discard_nonfinite_checkpoint("") is False
    assert stats["steps_ran"] == 1
    assert np.isfinite(stats["psnr_u8_init"])
    assert np.isfinite(stats["psnr_u8_trained"])
    assert np.isfinite(stats["metrics"]["psnr_u8"])


def test_pca_code_ratio_aims_each_material_at_its_label():
    """A smooth field and a noise field must not share the 0.5 scale.

    MetalPlates stored 1.42 bpp at the b2.5 label, Tiles 3.24, and
    PavingStones 3.89, because one scale sized every grid at 5 raw bpp.
    The measured stored/plain ratio is the scale, so label/ratio is the
    packed target and a higher label does not pick a smaller grid.
    """
    from ntc.grids import latent_payload_bits, select_grids

    n = 32
    smooth = np.zeros((3, n, n), np.float32)
    smooth[0] = np.linspace(0.0, 1.0, n, dtype=np.float32)[None, :]
    smooth[1] = np.linspace(1.0, 0.0, n, dtype=np.float32)[:, None]
    smooth[2] = 0.4
    noise = np.random.default_rng(0).random((3, n, n), dtype=np.float32)
    sems = [0, 1, 2]
    smooth_ratio = latent_coder_ratio(smooth, sems)
    noise_ratio = latent_coder_ratio(noise, sems)
    assert smooth_ratio < noise_ratio
    assert 0.15 <= smooth_ratio <= 0.95
    assert 0.15 <= noise_ratio <= 0.95
    label = 4.0

    def _picked(ratio: float, bpp: float):
        return select_grids(
            n, n, bpp / ratio, [32], 3, 0, 0, 8, 0, rate="latent", min_bits=4
        )[0]

    def _file_error(ratio: float, raw_target: float) -> float:
        grids = select_grids(
            n, n, raw_target, [32], 3, 0, 0, 8, 0, rate="latent", min_bits=4
        )[0]
        raw = latent_payload_bits(grids) / float(n * n)
        return abs(raw * ratio - label)

    for ratio in (smooth_ratio, noise_ratio):
        calibrated = _file_error(ratio, label / ratio)
        fixed = _file_error(ratio, label / 0.5)
        assert calibrated <= fixed + 1e-6
    low = _picked(smooth_ratio, 2.5)
    high = _picked(smooth_ratio, 10.0)
    assert latent_payload_bits(high) >= latent_payload_bits(low)
    _blob, stats = encode_arrays_fast(
        smooth,
        sems,
        TrainConfig(
            steps=0,
            target_bpp=label,
            grids=None,
            rate="latent",
            latent_rate_scale=0.0,
            latent_deflate=True,
            latent_coder="best",
            min_latent_bits=4,
            fast=True,
            compile_step="off",
            cuda_graph="off",
            log_every=10**9,
            mip_levels=1,
            seed=0,
            with_ssim=False,
            adaptive_pool=1,
            seed_pool=1,
        ),
    )
    # The reported scale is stored/plain on the grid that was kept, which
    # is not the half-resolution probe when the search picked a wider grid.
    from ntc.train_fast import grid_coder_ratio

    assert abs(float(stats["latent_rate_scale"]) - grid_coder_ratio(smooth, sems, stats["grids"])) < 1e-6
    kept_raw = latent_payload_bits(stats["grids"]) / float(n * n)
    assert kept_raw * float(stats["latent_rate_scale"]) <= label * 1.10 + 1e-6
    assert np.isfinite(stats["metrics"]["psnr_u8"])


def test_one_inf_gradient_is_skipped_and_the_run_stays_finite(monkeypatch):
    import ntc.train_fast as tf

    real = tf._clip_no_sync
    seen = {"n": 0}

    def _poison_once(params, max_norm, skips=None):
        seen["n"] += 1
        if seen["n"] == 1:
            for param in params:
                if param.grad is not None:
                    param.grad.fill_(float("inf"))
                    break
        real(params, max_norm, skips)

    monkeypatch.setattr(tf, "_clip_no_sync", _poison_once)
    img = np.linspace(0.0, 1.0, 3 * 16 * 16, dtype=np.float32).reshape(3, 16, 16)
    _blob, stats = encode_arrays_fast(
        img,
        [0, 1, 2],
        TrainConfig(
            steps=2,
            hidden=(8,),
            grids=[(8, 8, 2, 4)],
            fast=True,
            adaptive_pool=1,
            seed_pool=1,
            with_ssim=False,
            log_every=10**9,
            mip_levels=2,
            seed=1,
            compile_step="off",
            cuda_graph="off",
            precision="fp32",
        ),
    )
    assert stats["grad_skips"] == 1
    assert np.isfinite(stats["metrics"]["psnr_u8"])
    assert np.isfinite(stats["psnr_u8_trained"])


def test_finished_checkpoint_starts_fresh(tmp_path: Path):
    net = _net()
    path = tmp_path / "done.pt"
    torch.save({"model": net.state_dict(), "step": 9}, path)
    img = np.linspace(0.0, 1.0, 3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
    _blob, stats = encode_arrays_fast(
        img,
        [0, 1, 2],
        TrainConfig(
            steps=2,
            hidden=(8,),
            grids=[(8, 8, 2, 4)],
            fast=True,
            adaptive_pool=1,
            seed_pool=1,
            with_ssim=False,
            log_every=10**9,
            mip_levels=1,
            seed=1,
            compile_step="off",
            cuda_graph="off",
            precision="fp32",
            resume=str(path),
        ),
    )
    assert not path.is_file()
    assert stats["steps_ran"] == 2
    assert np.isfinite(stats["psnr_u8_init"])


def test_rewind_once_when_psnr_drops_below_init():
    net = _net()
    stacked = StackedTrain([net])
    images = torch.rand(1, 3, 8, 8)
    init = [p.detach().clone() for p in stacked.parameters()]
    score = _psnr_per_seed(stacked, images, torch.device("cpu"))[0]
    rewind = {
        "init_score": score + 5.0,
        "retries": 0,
        "latent_mult": 1.0,
        "last_good": [t.clone() for t in init],
        "images": images,
    }
    with torch.no_grad():
        stacked.latents[0].add_(1.0)
    _rewind_if_below_init(stacked, rewind, 2, 8, 1)
    assert rewind["retries"] == 1
    assert rewind["latent_mult"] == 0.5
    for param, src in zip(stacked.parameters(), init, strict=True):
        assert torch.allclose(param, src)
    with torch.no_grad():
        stacked.latents[0].add_(1.0)
    _rewind_if_below_init(stacked, rewind, 4, 8, 1)
    assert rewind["retries"] == 1


def test_w8_shaped_runs_do_not_ship_a_collapse():
    """Five seeds of an 8-bit wide grid. None may finish more than 1 dB under init."""
    rng = np.random.default_rng(2)
    n = 32
    img = rng.random((3, n, n), dtype=np.float32)
    for seed in range(5):
        _blob, stats = encode_arrays_fast(
            img,
            [0, 1, 2],
            TrainConfig(
                steps=8,
                hidden=(8,),
                grids=[(16, 16, 12, 8), (8, 8, 4, 4)],
                fast=True,
                adaptive_pool=1,
                seed_pool=1,
                with_ssim=False,
                log_every=10**9,
                mip_levels=2,
                seed=seed,
                compile_step="off",
                cuda_graph="off",
                precision="fp32",
                batch_pixels=64,
                batch_mip="flat",
            ),
        )
        assert stats["psnr_u8_trained"] >= stats["psnr_u8_init"] - 1.0
        assert isinstance(stats["lr_rewinds"], int)


def test_quant_schedule_file_stays_near_the_loop():
    rng = np.random.default_rng(5)
    img = rng.random((3, 16, 16), dtype=np.float32)
    _blob, stats = encode_arrays_fast(
        img,
        [0, 1, 2],
        TrainConfig(
            steps=40,
            hidden=(8,),
            grids=[(8, 8, 4, 4)],
            fast=True,
            adaptive_pool=1,
            seed_pool=1,
            with_ssim=False,
            log_every=10**9,
            mip_levels=1,
            seed=1,
            compile_step="off",
            cuda_graph="off",
            precision="fp32",
            quant_schedule=True,
            target_bpp=4.0,
        ),
    )
    assert abs(float(stats["metrics"]["psnr_u8"]) - float(stats["psnr_u8_loop"])) < 0.1


def test_same_seed_exports_the_same_bytes():
    import hashlib

    img = np.linspace(0.0, 1.0, 3 * 16 * 16, dtype=np.float32).reshape(3, 16, 16)
    cfg = TrainConfig(
        steps=2,
        hidden=(8,),
        grids=[(8, 8, 2, 4)],
        fast=True,
        adaptive_pool=1,
        seed_pool=1,
        with_ssim=False,
        log_every=10**9,
        mip_levels=2,
        seed=3,
        compile_step="off",
        cuda_graph="off",
        precision="fp32",
    )
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        first, _stats_a = encode_arrays_fast(img, [0, 1, 2], cfg)
        second, _stats_b = encode_arrays_fast(img, [0, 1, 2], cfg)
    finally:
        torch.set_num_threads(threads)
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()


def test_bilinear_backward_is_repeatable():
    codes = torch.randn(2, 3, 5, 5, requires_grad=True)
    uv = torch.rand(11, 2)
    first = bilinear_batched(codes, uv)
    first.sum().backward()
    grad_a = codes.grad.detach().clone()
    codes.grad = None
    second = bilinear_batched(codes, uv)
    second.sum().backward()
    assert torch.equal(grad_a, codes.grad)
    assert torch.allclose(first, second)


def test_recompile_limit_covers_every_mip_sample_count():
    import torch._dynamo.config as dyn

    previous = dyn.recompile_limit
    try:
        dyn.recompile_limit = 8
        assert _cover_mip_recompiles(12) >= 16
        assert dyn.recompile_limit >= 16
        # A higher limit already set for another run is left alone.
        dyn.recompile_limit = 32
        assert _cover_mip_recompiles(12) == 32
    finally:
        dyn.recompile_limit = previous


def test_clip_quantile_is_chosen_per_bit_depth():
    from ntc.encode import CLIP_QUANTILE_BY_BITS, choose_clip_quantile, clip_quantile

    rows = [
        (4, 0.99, 24.295, 2.750),
        (4, 0.995, 24.150, 2.747),
        (4, 0.999, 24.199, 2.732),
        (8, 0.99, 24.987, 4.354),
        (8, 0.995, 24.996, 4.354),
        (8, 0.999, 25.002, 4.362),
    ]
    picked = choose_clip_quantile(rows)
    assert picked[4] == 0.99
    assert picked[8] == 0.999
    assert clip_quantile(4) == CLIP_QUANTILE_BY_BITS[4]
    assert clip_quantile(8) == CLIP_QUANTILE_BY_BITS[8]
    values = torch.ones(1, 32, 32)
    values[0, 0, 0] = 40.0
    # 0.99 ignores more of the spike than 0.999, so the 4-bit level is smaller.
    four = _code_scale(values, 4)
    eight = _code_scale(values, 8)
    assert float(four[0]) < 40.0 / 7.5
    assert float(eight[0]) < 40.0 / 127.5
    assert torch.isfinite(four).all() and torch.isfinite(eight).all()


def test_code_scale_uses_a_percentile_not_one_spike():
    values = torch.ones(2, 32, 32)
    values[0, 0, 0] = 50.0
    scale = _code_scale(values, 4)
    # amax/mid would be 50/7.5, clamped to 5. The bulk is 1.
    assert float(scale[0]) < 1.0
    assert float(scale[1]) < 1.0
    assert float(scale[0]) > 0.05


def test_readout_ridge_bounds_a_weak_column():
    cols = 4
    rows = 32
    feat = torch.randn(rows, cols)
    feat[:, -1] *= 1e-4
    target = torch.randn(rows, 3)
    ridged = _ridge_lstsq(feat, target, 1e-2)
    plain = torch.linalg.lstsq(feat, target).solution
    assert torch.isfinite(ridged).all()
    assert float(ridged[:, -1].abs().max()) < float(plain[:, -1].abs().max())


def test_transplant_matches_the_smaller_grid(tmp_path: Path):
    from ntc.train_steps import apply_warm_start

    torch.manual_seed(0)
    src = NtcNet(
        image_channels=3,
        grids=[(4, 4, 2, 4)],
        hidden=[4],
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
        src.latents[0].fill_(3.0)
        src.linears[0].weight.normal_(0, 0.05)
        src.linears[1].weight.normal_(0, 0.05)
    dst = NtcNet(
        image_channels=3,
        grids=[(4, 4, 4, 4)],
        hidden=[4],
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
    path = tmp_path / "src.pt"
    torch.save({"model": src.state_dict(), "step": 0}, path)
    apply_warm_start(dst, str(path), False)
    uv = torch.rand(6, 2)
    assert torch.allclose(src(uv, mip=0, max_mip=1), dst(uv, mip=0, max_mip=1), atol=1e-5)


def test_metalplates_probe_does_not_keep_a_17_percent_overshoot():
    """Scale 0.397 selects 4+8 raw 6 bpp. That file stored 2.93, not 2.5.

    The extra coarse channels do not compress at the 4+4 probe ratio.
    Measuring them and selecting again at that ratio returns 4+4.
    PavingStones at 2.68 bpp is 7% over 2.5 and stays on its 512x12 grid.
    25.32 dB at 2.68 bpp versus 27.4 dB at 3.89 bpp is 1.7 dB per bpp.
    """
    from ntc.grids import latent_payload_bits, select_grids

    def _pick(raw: float):
        return select_grids(
            2048, 2048, raw, [32], 9, 0, 0, 8, 0, rate="latent", min_bits=4
        )[0]

    label = 2.5
    pixels = 2048 * 2048
    first = _pick(label / 0.397)
    assert [list(g) for g in first] == [[1024, 1024, 4, 4], [512, 512, 8, 4]]
    raw = latent_payload_bits(first) / pixels
    assert abs(raw - 6.0) < 1e-3
    realized = 2.9265 / raw
    assert rate_overshoots(raw * realized, label)
    second = _pick(label / realized)
    assert [list(g) for g in second] == [[1024, 1024, 4, 4], [512, 512, 4, 4]]
    paving = _pick(label / 0.643)
    assert [list(g) for g in paving] == [[512, 512, 12, 4], [256, 256, 12, 4]]
    assert rate_overshoots(2.6826, label) is False
    slope = (27.4 - 25.32) / (3.89 - 2.6826)
    assert 1.5 < slope < 2.0


def test_halfres_8ch_coarse_grid_trains_above_its_init():
    """The MetalPlates b2.5 grid walked a 30 dB init down when coarse matched mip 0.

    Same shape on a 48 stand-in: 4 channels at half resolution, 8 at quarter.
    The file has to finish above the init, and the run must not need a rewind.
    """
    n = 48
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    bases = [
        0.55 + 0.35 * np.sin(xx / 9.0) * np.cos(yy / 11.0),
        0.45 + 0.25 * np.sin((xx + yy) / 14.0),
        0.50 + 0.20 * np.cos(xx / 18.0),
        0.40 + 0.30 * np.sin(yy / 8.0),
        0.60 + 0.20 * np.cos((xx - yy) / 10.0),
        0.35 + 0.25 * np.sin(xx / 6.0) * np.sin(yy / 6.0),
        0.50 + 0.15 * np.cos(xx / 12.0),
        0.48 + 0.22 * np.sin((xx + 2 * yy) / 16.0),
        0.52 + 0.18 * np.cos(yy / 9.0),
    ]
    img = np.stack(bases, 0).astype(np.float32)
    _blob, stats = encode_arrays_fast(
        img,
        list(range(9)),
        TrainConfig(
            steps=120,
            hidden=(32,),
            grids=[(n // 2, n // 2, 4, 4), (n // 4, n // 4, 8, 4)],
            seed=0,
            fast=True,
            adaptive_pool=1,
            seed_pool=1,
            with_ssim=False,
            log_every=10**9,
            mip_levels=6,
            compile_step="off",
            cuda_graph="off",
            precision="fp32",
            batch_pixels=(n * n) // 4,
            batch_mip="flat",
            pca_init=True,
            rate="latent",
            target_bpp=2.5,
            latent_deflate=False,
            warmup_steps=12,
            schedule="cosine",
            entropy_rate=False,
            quant_schedule=False,
            eval_every=0,
        ),
    )
    assert stats["batch_mip"] == "flat"
    assert stats["lr_rewinds"] == 0
    assert stats["psnr_u8_trained"] >= stats["psnr_u8_init"]
    assert stats["checkpoint"] == "trained"


def test_uv_batch_dim_does_not_recompile_per_mip():
    """65536 versus 16384 used to be the guard that hit recompile_limit 16.

    The batch dim is marked on the mip buffers before compile. A second
    step with the same pyramid must not compile once per mip.
    """
    from ntc.train_fast import _build_nets, _loss_from_mips, _mark_sample_shapes, _prep_mips

    torch._dynamo.reset()
    n = 32
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    img = np.stack(
        [
            0.5 + 0.2 * np.sin(xx / 3.0),
            0.4 + 0.2 * np.cos(yy / 4.0),
            0.3 + 0.1 * np.sin((xx + yy) / 5.0),
        ],
        0,
    ).astype(np.float32)
    target = torch.from_numpy(img)
    grids = [(n // 2, n // 2, 4, 4), (n // 4, n // 4, 4, 4)]
    cfg = TrainConfig(
        steps=4,
        hidden=(8,),
        grids=grids,
        seed=0,
        mip_levels=6,
        batch_pixels=n * n // 4,
        batch_mip="area",
        pca_init=True,
        precision="fp32",
        compile_step="off",
        cuda_graph="off",
        log_every=10**9,
    )
    device = target.device
    nets = _build_nets(target, cfg, grids, [0], device)
    stacked = StackedTrain(nets)
    mips = _prep_mips(target, cfg, device)
    indices = static_sample_indices(mips, int(cfg.batch_pixels), "area", device)
    _mark_sample_shapes(mips, indices)
    compiled = torch.compile(_loss_from_mips, backend="eager", dynamic=False)
    torch._dynamo.utils.counters.clear()
    compiled(stacked, mips, cfg, [0, 1, 2], None, int(cfg.batch_pixels), 0, indices)
    refill_sample_indices(indices, mips, int(cfg.batch_pixels), "area")
    compiled(stacked, mips, cfg, [0, 1, 2], None, int(cfg.batch_pixels), 1, indices)
    recompiles = int(torch._dynamo.utils.counters["frames"]["recompile"])
    assert recompiles <= 2


def test_fit_latent_grids_measures_the_grid_it_keeps():
    n = 32
    smooth = np.zeros((3, n, n), np.float32)
    smooth[0] = np.linspace(0.0, 1.0, n, dtype=np.float32)[None, :]
    smooth[1] = np.linspace(1.0, 0.0, n, dtype=np.float32)[:, None]
    smooth[2] = 0.4
    grids, scale = fit_latent_grids(smooth, [0, 1, 2], n, n, 4.0, [32], 3, 4)
    from ntc.grids import latent_payload_bits
    from ntc.train_fast import grid_coder_ratio

    measured = grid_coder_ratio(smooth, [0, 1, 2], grids)
    assert abs(scale - measured) < 1e-6
    predicted = latent_payload_bits(grids) / float(n * n) * scale
    assert predicted <= 4.0 * 1.10 + 1e-6
