"""Device auto-select. MPS is preferred when CUDA is absent."""

from __future__ import annotations

import pytest
import torch

from ntc import device as dev
from ntc.device import device_report, pick_device


def test_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(dev, "_mps_available", lambda: True)
    assert pick_device("auto").type == "cuda"


def test_auto_uses_mps_before_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(dev, "_mps_available", lambda: True)
    monkeypatch.setattr(dev, "_try_directml", lambda: None)
    picked = pick_device("auto")
    assert picked.type == "mps"
    report = device_report(picked)
    assert report["backend"] == "mps"


def test_named_mps_requires_the_backend(monkeypatch):
    monkeypatch.setattr(dev, "_mps_available", lambda: False)
    with pytest.raises(RuntimeError, match="mps is not available"):
        pick_device("mps")


def test_auto_stays_on_cpu_without_mps(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(dev, "_mps_available", lambda: False)
    monkeypatch.setattr(dev, "_try_directml", lambda: None)
    assert pick_device("auto").type == "cpu"
