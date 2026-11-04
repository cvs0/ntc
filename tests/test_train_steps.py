"""Step-reduction helpers and the two-machine queue."""

import json
from pathlib import Path

import torch

from bench.train_queue import _claim, _steal, enqueue_demo, merge
from ntc.encode import TrainConfig
from ntc.model import NtcNet
from ntc.train_steps import analytic_project, apply_warm_start


def _net() -> NtcNet:
    return NtcNet(
        image_channels=3,
        grids=[(8, 8, 4, 4)],
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


def test_analytic_projection_writes_finite_codes():
    torch.manual_seed(0)
    net = _net()
    image = torch.rand(3, 16, 16)
    analytic_project(net, image)
    codes = net.latents[0].detach()
    assert torch.isfinite(codes).all()
    assert codes.min() >= -0.1
    assert codes.max() <= 16


def test_warm_start_copies_overlap_and_can_freeze(tmp_path: Path):
    torch.manual_seed(0)
    src = _net()
    path = tmp_path / "prior.pt"
    torch.save(
        {
            "weights": [lin.weight.detach().cpu() for lin in src.linears],
            "biases": [lin.bias.detach().cpu() for lin in src.linears],
        },
        path,
    )
    dst = _net()
    apply_warm_start(dst, str(path), freeze=True)
    assert torch.allclose(dst.linears[0].weight, src.linears[0].weight)
    assert not dst.linears[0].weight.requires_grad


def test_queue_claim_run_and_merge(tmp_path: Path):
    enqueue_demo(tmp_path)
    claimed = _claim(tmp_path, "amd-desktop")
    assert claimed is not None
    assert claimed.parent.name == "amd-desktop"
    job = json.loads(claimed.read_text())
    assert job["material"] == "synthetic"
    # A second claim finds nothing.
    assert _claim(tmp_path, "nvidia-laptop") is None
    # Stale running jobs return to pending.
    old = claimed.stat().st_mtime - 100
    os_utime = __import__("os").utime
    os_utime(claimed, (old, old))
    assert _steal(tmp_path, 10) == 1
    assert (tmp_path / "pending" / "demo-synthetic.json").is_file()
    (tmp_path / "done").mkdir(exist_ok=True)
    (tmp_path / "done" / "demo-synthetic.json").write_text(json.dumps({"id": "demo-synthetic", "psnr_u8": 1.0}))
    (tmp_path / "pending" / "demo-synthetic.json").unlink()
    merge(tmp_path)
    merged = json.loads((tmp_path / "merged.json").read_text())
    assert merged["done"][0]["id"] == "demo-synthetic"


def test_fast_defaults_leave_new_flags_off():
    cfg = TrainConfig()
    assert cfg.fast is False
    assert cfg.latent_init == "pca"
    assert cfg.optimizer == "adam"
    assert cfg.warm_start == ""
    assert cfg.stop_psnr == 0.0
    assert cfg.coarse_to_fine is False
