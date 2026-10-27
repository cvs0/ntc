"""PSNR and SSIM on values in ``[0, 1]``.

Headline PSNR quantizes both images to 8-bit unorm before the error,
which is the comparison used against BC and ASTC decoders.
``psnr_f32`` is the clamped float error, without that round trip.
SSIM is Wang 2004, 11x11 Gaussian, sigma 1.5, on the 8-bit-quantized
values. FLIP is optional and imported only when the package is present.
"""

from __future__ import annotations

import numpy as np

_WIN = None


def to_u8_unit(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.clip(x, 0.0, 1.0) * 255.0), 0, 255).astype(np.float32) / 255.0


def psnr(a: np.ndarray, b: np.ndarray, cap: float = 100.0) -> float:
    d = a.astype(np.float64) - b.astype(np.float64)
    mse = float(np.mean(d * d))
    if mse <= 1e-20:
        return cap
    value = 10.0 * np.log10(1.0 / mse)
    if value > cap:
        return cap
    return float(value)


def psnr_u8(pred: np.ndarray, ref: np.ndarray) -> float:
    return psnr(to_u8_unit(pred), to_u8_unit(ref))


def psnr_f32(pred: np.ndarray, ref: np.ndarray) -> float:
    return psnr(np.clip(pred, 0.0, 1.0), np.clip(ref, 0.0, 1.0))


def _window() -> np.ndarray:
    global _WIN
    if _WIN is None:
        size = 11
        sigma = 1.5
        coords = np.arange(size, dtype=np.float64) - size // 2
        g = np.exp(-(coords ** 2) / (2.0 * sigma * sigma))
        w = np.outer(g, g)
        w /= w.sum()
        _WIN = w.astype(np.float32)
    return _WIN


def _conv2(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    pad_y, pad_x = kh // 2, kw // 2
    padded = np.pad(img, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    # Direct convolution keeps the reflect pad exact.
    from numpy.lib.stride_tricks import sliding_window_view

    windows = sliding_window_view(padded, (kh, kw))
    return np.einsum("ij,xyij->xy", kernel, windows, optimize=True).astype(np.float32)


def ssim_channel(a: np.ndarray, b: np.ndarray) -> float:
    """``a`` and ``b`` are ``[H,W]`` in ``[0,1]``."""
    a = to_u8_unit(a)
    b = to_u8_unit(b)
    if a.shape != b.shape:
        raise ValueError("shape")
    if min(a.shape) < 11:
        # too small for the window: fall back to a global SSIM
        return _ssim_global(a, b)
    k = _window()
    c1 = np.float32((0.01) ** 2)
    c2 = np.float32((0.03) ** 2)
    mu_a = _conv2(a, k)
    mu_b = _conv2(b, k)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b
    sig_a = _conv2(a * a, k) - mu_a2
    sig_b = _conv2(b * b, k) - mu_b2
    sig_ab = _conv2(a * b, k) - mu_ab
    num = (2 * mu_ab + c1) * (2 * sig_ab + c2)
    den = (mu_a2 + mu_b2 + c1) * (sig_a + sig_b + c2)
    return float(np.mean(num / den))


def _ssim_global(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_a = a.mean()
    mu_b = b.mean()
    sig_a = a.var()
    sig_b = b.var()
    sig_ab = ((a - mu_a) * (b - mu_b)).mean()
    num = (2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (sig_a + sig_b + c2)
    return float(num / den)


def ssim_mean(pred: np.ndarray, ref: np.ndarray, axes: list[int] | None = None) -> float:
    """Mean SSIM over channels. ``pred`` is ``[H,W,C]``."""
    if axes is None:
        axes = list(range(pred.shape[2]))
    if not axes:
        return float("nan")
    return float(np.mean([ssim_channel(pred[:, :, i], ref[:, :, i]) for i in axes]))


def normal_angle_deg(pred: np.ndarray, ref: np.ndarray) -> float:
    """Mean angle in degrees between unorm normals stored in ``[0,1]``."""
    p = pred.astype(np.float64) * 2.0 - 1.0
    r = ref.astype(np.float64) * 2.0 - 1.0
    pn = np.linalg.norm(p, axis=-1, keepdims=True)
    rn = np.linalg.norm(r, axis=-1, keepdims=True)
    p = p / np.maximum(pn, 1e-8)
    r = r / np.maximum(rn, 1e-8)
    d = np.clip((p * r).sum(axis=-1), -1.0, 1.0)
    return float(np.degrees(np.arccos(d)).mean())


def try_flip(pred: np.ndarray, ref: np.ndarray) -> float | None:
    """Mean FLIP if a FLIP package is importable, else None.

    Accepts ``[H,W]`` or ``[H,W,3]`` in ``[0,1]``.
    """
    pred = np.clip(pred, 0.0, 1.0).astype(np.float32)
    ref = np.clip(ref, 0.0, 1.0).astype(np.float32)
    if pred.ndim == 2:
        pred = np.repeat(pred[:, :, None], 3, axis=2)
        ref = np.repeat(ref[:, :, None], 3, axis=2)
    try:
        import flip_evaluator  # type: ignore
    except Exception:
        return None
    # The NVIDIA package expects HxWx3 float HDR or LDR in [0,1].
    err = flip_evaluator.evaluate(ref, pred, "LDR")
    return float(np.mean(err))


GROUP_SEMANTICS = {
    "albedo": (0, 1, 2),
    "normal": (3, 4, 5),
    "roughness": (6,),
    "metalness": (7,),
    "ao": (8,),
    "height": (9,),
    "opacity": (10,),
}


def group_metrics(pred: np.ndarray, ref: np.ndarray, semantics: list[int]) -> dict:
    """``pred`` and ``ref`` are ``[H,W,C]`` in channel order."""
    out = {
        "psnr_u8": psnr_u8(pred, ref),
        "psnr_f32": psnr_f32(pred, ref),
        "ssim": ssim_mean(pred, ref),
        "channel_psnr_u8": [psnr_u8(pred[:, :, i], ref[:, :, i]) for i in range(pred.shape[2])],
        "channel_mse_u8": [
            float(np.mean((to_u8_unit(pred[:, :, i]) - to_u8_unit(ref[:, :, i])) ** 2))
            for i in range(pred.shape[2])
        ],
    }
    index = {sem: i for i, sem in enumerate(semantics)}
    for name, sems in GROUP_SEMANTICS.items():
        axes = [index[s] for s in sems if s in index]
        if not axes:
            continue
        sl = pred[:, :, axes]
        sr = ref[:, :, axes]
        out[f"psnr_u8_{name}"] = psnr_u8(sl, sr)
        out[f"ssim_{name}"] = ssim_mean(sl, sr)
        if name == "normal" and sl.shape[2] == 3:
            out["normal_angle_deg"] = normal_angle_deg(sl, sr)
    flip_axes = [index[s] for s in (0, 1, 2) if s in index]
    if len(flip_axes) == 3:
        flip = try_flip(pred[:, :, flip_axes], ref[:, :, flip_axes])
        if flip is not None:
            out["flip_albedo"] = flip
    return out
