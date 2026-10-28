"""Split conformal recalibration of upper quantile forecasts.

Scores are computed on the calibration (validation) window only, in site-normalised units
(y / scale - q / scale), so that one additive correction is shared across sites of different size.
For a target level tau the correction is the ceil((n + 1) * tau) / n empirical quantile of the scores
(standard finite-sample split conformal, one-sided upper bound).
"""
from __future__ import annotations

import math

import numpy as np


def conformal_offset(y_cal, q_cal, scale_cal, tau: float) -> float:
    """Additive offset (normalised units) so that P(y <= q + offset * scale) >= tau on exchangeable data."""
    y_cal, q_cal, scale_cal = (np.asarray(a, float) for a in (y_cal, q_cal, scale_cal))
    s = (y_cal - q_cal) / scale_cal
    n = len(s)
    if n == 0:
        raise ValueError("empty calibration set")
    k = math.ceil((n + 1) * tau)
    if k > n:
        return float(np.inf)
    return float(np.sort(s)[k - 1])


def apply_offset(q, scale, offset: float):
    return np.clip(np.asarray(q, float) + offset * np.asarray(scale, float), 0, None)


def calibrate_quantiles(cal: dict, target: dict, y_cal, scale_cal, scale_target, levels: dict) -> tuple[dict, dict]:
    """Recalibrate several quantile columns and keep them non-crossing.

    cal / target: name -> raw predictions on the calibration / target rows.
    levels: name -> nominal level (e.g. {"q90": 0.9}).
    Returns (recalibrated target predictions, offsets). Offsets depend only on calibration rows.
    """
    names = sorted(levels, key=lambda n: levels[n])
    offs = {n: conformal_offset(y_cal, cal[n], scale_cal, levels[n]) for n in names}
    out = {n: apply_offset(target[n], scale_target, offs[n]) for n in names}
    # enforce monotonicity across levels (running maximum in level order)
    for lo, hi in zip(names, names[1:]):
        out[hi] = np.maximum(out[hi], out[lo])
    return out, offs
