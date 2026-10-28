"""Triton bilinear gather: index wrap and kernel output versus the eager path."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _kernel_wrap(i: torch.Tensor, n: int) -> torch.Tensor:
    """The kernel's floor modulo: truncated % (C and Triton), then add n back."""
    r = torch.fmod(i, n)
    return torch.where(r < 0, r + n, r)


@pytest.mark.parametrize("n", [1, 2, 3, 7, 16, 64, 257, 1024, 2048])
def test_wrap_matches_torch_remainder(n: int) -> None:
    i = torch.arange(-5 * n - 3, 5 * n + 4, dtype=torch.int64)
    assert torch.equal(_kernel_wrap(i, n), torch.remainder(i, n))
    nxt = _kernel_wrap(_kernel_wrap(i, n) + 1, n)
    assert torch.equal(nxt, torch.remainder(torch.remainder(i, n) + 1, n))


def test_kernel_source_has_no_math_mod() -> None:
    src = (ROOT / "src" / "ntc" / "triton_bilinear.py").read_text()
    assert "tl.math.mod(" not in src


def test_cover_mip_recompiles_raises_cudagraph_warn_limit() -> None:
    import torch._inductor.config as inductor_config

    from ntc.train_fast import _cover_mip_recompiles

    before = inductor_config.triton.cudagraph_dynamic_shape_warn_limit
    try:
        inductor_config.triton.cudagraph_dynamic_shape_warn_limit = 8
        need = _cover_mip_recompiles(12)
        assert inductor_config.triton.cudagraph_dynamic_shape_warn_limit >= need
    finally:
        inductor_config.triton.cudagraph_dynamic_shape_warn_limit = before


_INTERP = textwrap.dedent(
    """
    import torch, triton, triton.language as tl
    import ntc.triton_bilinear as tb
    from ntc.train_fast import _bilinear_forward
    g = torch.Generator().manual_seed(3)
    for seeds, ch, h, w in [(1, 8, 16, 16), (2, 20, 9, 13), (1, 40, 32, 8), (1, 4, 1, 5)]:
        codes = torch.randn(seeds, ch, h, w, generator=g)
        uv = torch.rand(97, 2, generator=g) * 1.2 - 0.1
        uv[:4] = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [-0.03, 1.02]])
        got = tb._forward(codes, uv, triton, tl)
        ref = _bilinear_forward(codes, uv)
        assert got.shape == ref.shape, (got.shape, ref.shape)
        err = (got - ref).abs().max().item()
        assert err <= 1e-5, (seeds, ch, h, w, err)
    print("ok")
    """
)


def test_kernel_matches_eager_in_triton_interpreter() -> None:
    pytest.importorskip("triton")
    env = dict(os.environ)
    env["TRITON_INTERPRET"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")])
    proc = subprocess.run(
        [sys.executable, "-c", _INTERP], env=env, capture_output=True, text=True, timeout=600
    )
    if proc.returncode != 0 and "interpret" in (proc.stderr or "").lower() and "not supported" in proc.stderr.lower():
        pytest.skip("Triton interpreter unavailable")
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert proc.stdout.strip().endswith("ok")
