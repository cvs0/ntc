"""Laplace code rate against the range coder on three smooth fields."""

import numpy as np
import pytest
import torch

from ntc.entropy import file_rate_bpp, laplace_bits
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


def test_file_rate_does_not_penalize_a_coder_that_is_under_the_label():
    """Tiles sat under 2.5 bpp while a raw Laplace near 4 bpp was still penalized.

    The file rate is the model times the stored/plain ratio. 4 raw bpp at
    the Tiles ratio is under the label, so the excess is zero. The same
    model with no ratio is over the label.
    """
    model = torch.tensor(4.0)
    assert float(file_rate_bpp(model, 0.55)) == pytest.approx(2.2)
    assert float(file_rate_bpp(model, 0.55)) < 2.5
    assert float(file_rate_bpp(model, 1.0)) > 2.5
    assert float(file_rate_bpp(model, 0.0)) == pytest.approx(4.0)


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
