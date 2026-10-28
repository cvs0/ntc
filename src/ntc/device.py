"""Torch device selection.

Order for ``auto``: CUDA, then ROCm, then DirectML, then CPU.

CUDA and ROCm both report ``torch.cuda``. A ROCm build sets
``torch.version.hip`` and still uses the ``cuda`` device API, including
AMD's Windows ROCm wheels (RDNA4 / RX 9070 is gfx1201 in the ROCm 7.2.1
Windows wheels). DirectML is the ``torch_directml`` package and shows up
as a ``privateuseone`` device. There is no CUDA-only op in the encoder.
"""

from __future__ import annotations

import torch


def _directml_device():
    import torch_directml

    return torch_directml.device()


def _try_directml():
    try:
        import torch_directml
    except ImportError:
        return None
    try:
        if hasattr(torch_directml, "is_available") and not torch_directml.is_available():
            return None
        return torch_directml.device()
    except Exception:
        return None


def pick_device(name: str = "auto") -> torch.device:
    if name and name != "auto":
        if name.lower() in ("dml", "directml"):
            return _directml_device()
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    dml = _try_directml()
    if dml is not None:
        return dml
    return torch.device("cpu")


def device_report(device: torch.device) -> dict:
    info = {
        "device": str(device),
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": getattr(torch.version, "cuda", None),
    }
    if device.type == "cuda":
        info["backend"] = "rocm" if info["hip"] else "cuda"
        try:
            info["name"] = torch.cuda.get_device_name(device)
        except Exception as exc:
            info["name"] = "cuda"
            info["name_error"] = str(exc)
    elif device.type == "privateuseone" or str(device).startswith("privateuseone"):
        info["backend"] = "directml"
        info["name"] = str(device)
        try:
            import torch_directml

            index = device.index if device.index is not None else 0
            info["name"] = torch_directml.device_name(index)
            info["directml"] = getattr(torch_directml, "__version__", None)
        except Exception as exc:
            info["directml_error"] = str(exc)
    else:
        info["backend"] = "cpu"
        info["name"] = "cpu"
    return info
