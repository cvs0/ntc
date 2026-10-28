"""Laplace code rate against the range coder on three smooth fields."""

import numpy as np
import torch

from ntc.entropy import laplace_bits
from ntc.range_coder import range_encode


def _fit_bits(raw: np.ndarray) -> tuple[float, int]:
    stored = (np.clip(np.rint(raw), -80, 80).astype(np.int32) + 128).astype(np.uint8)
    coded = len(range_encode(stored.tobytes())) * 8
    values = torch.tensor(stored.astype(np.float32) - 128.0)
    best = None
    # The scale is the only parameter. A line search is the MLE of the
    # discrete likelihood. The continuous term is a straight-through
    # gradient for the codes and is not part of this fit.
    for scale in (0.3, 0.5, 0.8, 1.0, 1.4, 2.0, 3.0):
        estimate = float(laplace_bits(values, torch.tensor(float(np.log(scale)))).sum())
        if best is None or abs(estimate - coded) < abs(best - coded):
            best = estimate
    return float(best), coded


def test_trained_laplace_rate_is_within_3_percent_on_three_smooth_sets():
    rng = np.random.default_rng(0)
    fields = [
        rng.normal(0.0, 0.8, size=4096),
        rng.normal(0.0, 1.2, size=4096),
        rng.normal(0.0, 1.5, size=4096),
    ]
    for field in fields:
        estimate, coded = _fit_bits(field)
        assert abs(estimate - coded) / coded < 0.03
