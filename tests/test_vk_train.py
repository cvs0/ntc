"""The Vulkan training reference matches PyTorch eager."""

import shutil
import subprocess
from pathlib import Path

import torch

from ntc.train_fast import bilinear_batched
from ntc.vk_train.reference import adam_step, gather_taps

ROOT = Path(__file__).resolve().parents[1]


def _rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    denom = ref.detach().abs().clamp(min=1e-8)
    return float(((got - ref).abs() / denom).max())


def test_adam_matches_torch_within_1e_5():
    torch.manual_seed(1)
    start = torch.nn.Parameter(torch.randn(6, 5))
    grad = torch.randn(6, 5)
    before = start.detach().clone()
    opt = torch.optim.Adam([start], lr=1e-2, betas=(0.9, 0.99), eps=1e-15)
    start.grad = grad.clone()
    opt.step()
    got, _m, _v = adam_step(before, grad, torch.zeros_like(before), torch.zeros_like(before), 1, 1e-2, 0.9, 0.99, 1e-15)
    assert _rel(got, start.detach()) < 1e-5


def test_gather_matches_the_pytorch_bilinear():
    torch.manual_seed(2)
    codes = torch.randn(1, 3, 5, 7)
    uv = torch.rand(11, 2)
    got = gather_taps(codes, uv)
    ref = bilinear_batched(codes, uv)
    assert _rel(got, ref) < 1e-5


def test_train_shaders_compile(tmp_path: Path):
    glslang = shutil.which("glslangValidator")
    if not glslang:
        return
    for name in ("ntc_train_adam.comp", "ntc_train_gather.comp"):
        subprocess.check_call(
            [glslang, "-V", "--target-env", "vulkan1.3", str(ROOT / "shaders" / name), "-o", str(tmp_path / f"{name}.spv")]
        )
