"""Fast trainer: same mip-0 math as NtcNet, halving, and the off-by-default flag."""

from pathlib import Path

import numpy as np
import torch

from ntc.encode import TrainConfig, encode_arrays
from ntc.model import NtcNet, bilinear_codes_torch
from ntc.train_fast import (
    LATENT_ADAM_EPS,
    NEAR_INIT_DB,
    StackedTrain,
    _adam_on,
    _assign_lr,
    _mesh_uv,
    bilinear_batched,
    encode_arrays_fast,
    fullres_batch_floor,
    preferred_compile_mode,
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
