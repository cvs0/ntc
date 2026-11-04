"""Compiled-step reuse across chain jobs, and the warm-start seed pool."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from ntc.encode import TrainConfig
from ntc.train_fast import (
    _build_nets,
    _segment_add,
    compiled_loss_fn,
    encode_arrays_fast,
    identical_seed_pool,
)


def _naive_segment_add(dest, index, values):
    for j in range(index.numel()):
        dest[:, :, index[j]] += values[:, :, j]


def test_segment_add_matches_loop():
    gen = torch.Generator().manual_seed(0)
    for _ in range(20):
        n = int(torch.randint(1, 40, (1,), generator=gen))
        size = int(torch.randint(1, 12, (1,), generator=gen))
        index = torch.randint(0, size, (n,), generator=gen)
        values = torch.randn(2, 3, n, generator=gen, dtype=torch.float64)
        want = torch.zeros(2, 3, size, dtype=torch.float64)
        got = torch.zeros_like(want)
        _naive_segment_add(want, index, values)
        _segment_add(got, index, values)
        assert torch.allclose(got, want, atol=1e-12)


def test_segment_add_empty_index():
    dest = torch.zeros(1, 1, 4)
    _segment_add(dest, torch.zeros(0, dtype=torch.long), torch.zeros(1, 1, 0))
    assert torch.equal(dest, torch.zeros(1, 1, 4))


def _small_cfg(**kw):
    base = dict(
        hidden=(8,), grids=[(16, 16, 2, 4)], fast=True, seed_pool=3, device="cpu", pca_init=True
    )
    base.update(kw)
    return TrainConfig(**base)


def test_warm_started_pool_is_identical(tmp_path, monkeypatch):
    monkeypatch.delenv("NTC_KEEP_IDENTICAL_POOL", raising=False)
    img = torch.rand(3, 32, 32, generator=torch.Generator().manual_seed(1))
    cpu = torch.device("cpu")
    cfg = _small_cfg()
    low = [(16, 16, 2, 4), (8, 8, 2, 4)]
    high = [(16, 16, 4, 4), (8, 8, 4, 4)]
    src = _build_nets(img, cfg, low, [0], cpu)[0]
    ckpt = tmp_path / "low.pt"
    torch.save({"model": src.state_dict()}, ckpt)
    assert not identical_seed_pool(_build_nets(img, cfg, low, [0, 1, 2], cpu))
    warm = replace(cfg, warm_start=str(ckpt))
    assert identical_seed_pool(_build_nets(img, warm, high, [0, 1, 2], cpu))
    monkeypatch.setenv("NTC_KEEP_IDENTICAL_POOL", "1")
    assert not identical_seed_pool(_build_nets(img, warm, high, [0, 1, 2], cpu))


def _have_cpu_compiler() -> bool:
    try:
        from torch._inductor.cpp_builder import get_cpp_compiler

        get_cpp_compiler()
    except Exception:
        return False
    return True


def _chain_cfg(**kw):
    base = dict(
        steps=20, hidden=(8,), grids=[(16, 16, 2, 4)], fast=True, adaptive_pool=1,
        seed_pool=1, with_ssim=False, log_every=10**9, mip_levels=2, seed=0,
        compile_step="on", cuda_graph="off", precision="fp32", device="cpu",
        batch_pixels=256, batch_mip="flat", quant_schedule=True, entropy_rate=True,
        target_bpp=1.0, rate="latent", latent_rate_scale=0.5, eval_every=10,
        early_stop=16, job_name="c1", quant_freeze_at=0.9,
    )
    base.update(kw)
    return TrainConfig(**base)


@pytest.mark.skipif(not _have_cpu_compiler(), reason="Inductor needs a C++ compiler on CPU")
def test_second_chain_job_same_shapes_does_not_recompile():
    from torch._dynamo.utils import counters

    img = np.random.default_rng(0).random((3, 32, 32)).astype(np.float32)
    _blob, stats = encode_arrays_fast(img, [0, 1, 2], _chain_cfg())
    assert stats["compiled"], stats["compile_error"]
    assert compiled_loss_fn("default") is compiled_loss_fn("default")
    before = counters["frames"]["total"]
    assert before > 0
    encode_arrays_fast(img, [0, 1, 2], _chain_cfg(job_name="c2"))
    assert counters["frames"]["total"] == before


def test_known_compile_noise_prints_once_and_errors_stay(monkeypatch):
    import logging

    import torch._dynamo  # noqa: F401  (sets up the torch loggers)

    from ntc.train_fast import CompileNoiseFilter, install_compile_log_filter

    monkeypatch.delenv("NTC_COMPILE_LOGS", raising=False)
    shared = install_compile_log_filter()
    handler = logging.getLogger("torch._inductor").handlers[0]
    assert shared in handler.filters
    assert install_compile_log_filter() is shared
    assert handler.filters.count(shared) == 1

    filt = CompileNoiseFilter()

    def _rec(msg, level=logging.ERROR):
        return logging.LogRecord("torch._inductor.select_algorithm", level, __file__, 1, msg, None, None)

    noisy = "Runtime error during autotuning: out of resource: shared memory. Ignoring this choice."
    first = _rec(noisy)
    assert filt.filter(first)
    assert "hidden" in first.getMessage()
    assert not filt.filter(_rec(noisy))
    assert not filt.filter(_rec(noisy, logging.WARNING))
    assert filt.counts["Runtime error during autotuning"] == 3
    real = _rec("CUDA error: an illegal memory access was encountered")
    assert filt.filter(real)
    assert filt.filter(_rec("CUDA error: an illegal memory access was encountered"))
