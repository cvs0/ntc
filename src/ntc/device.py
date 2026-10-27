"""Torch device selection. CUDA and ROCm both report ``torch.cuda``.

A ROCm build sets ``torch.version.hip`` and still uses the ``cuda`` device
API. There is no CUDA-only op in the encoder.
"""

from __future__ import annotations

import torch


def pick_device(name: str = "auto") -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_report(device: torch.device) -> dict:
    info = {"device": str(device), "torch": torch.__version__}
    if device.type == "cuda":
        hip = getattr(torch.version, "hip", None)
        info["backend"] = "rocm" if hip else "cuda"
        info["hip"] = hip
        info["cuda"] = getattr(torch.version, "cuda", None)
        info["name"] = torch.cuda.get_device_name(device)
    else:
        info["backend"] = "cpu"
        info["name"] = "cpu"
    return info
