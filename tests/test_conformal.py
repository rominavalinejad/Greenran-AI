import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from greenran.conformal import apply_offset, calibrate_quantiles, conformal_offset  # noqa: E402
from greenran.data import EMAIL_ENV, resolve_guestbook  # noqa: E402


def test_conformal_reaches_target_coverage_on_synthetic_data():
    rng = np.random.default_rng(0)
    n = 20000
    scale = rng.uniform(1, 10, 2 * n)
    y = scale * (5 + rng.normal(0, 1, 2 * n))
    q = scale * 5.5  # biased "q90" that undercovers
    o = conformal_offset(y[:n], q[:n], scale[:n], 0.9)
    cov = np.mean(y[n:] <= apply_offset(q[n:], scale[n:], o))
    assert np.mean(y[n:] <= q[n:]) < 0.8
    assert abs(cov - 0.9) < 0.01


def test_recalibrated_quantiles_are_monotonic():
    rng = np.random.default_rng(1)
    n = 500
    s = np.ones(n)
    y = rng.normal(0, 1, n)
    cal = {"q50": np.zeros(n), "q80": np.full(n, 2.0), "q90": np.full(n, 0.1)}
    tgt = {k: v[:50].copy() for k, v in cal.items()}
    out, _ = calibrate_quantiles(cal, tgt, y + 5, s, s[:50], {"q50": 0.5, "q80": 0.8, "q90": 0.9})
    assert np.all(out["q50"] <= out["q80"]) and np.all(out["q80"] <= out["q90"])


def test_offsets_do_not_depend_on_test_rows():
    rng = np.random.default_rng(2)
    n = 300
    y, s = rng.normal(5, 1, n), np.ones(n)
    cal = {"q90": np.full(n, 5.0)}
    _, o1 = calibrate_quantiles(cal, {"q90": np.zeros(10)}, y, s, np.ones(10), {"q90": 0.9})
    _, o2 = calibrate_quantiles(cal, {"q90": np.full(10, 1e6)}, y, s, np.ones(10) * 7, {"q90": 0.9})
    assert o1 == o2


def test_guestbook_email_from_env_and_clear_error():
    gb = {"name": "x", "email": "<EMAIL_FROM_ENV>"}
    assert resolve_guestbook(gb, {EMAIL_ENV: "a@b.org"})["email"] == "a@b.org"
    assert gb["email"] == "<EMAIL_FROM_ENV>"
    with pytest.raises(RuntimeError, match=EMAIL_ENV):
        resolve_guestbook(gb, {})


def test_guestbook_rejects_placeholder_domain():
    import pytest
    from greenran.data import resolve_guestbook
    with pytest.raises(RuntimeError):
        resolve_guestbook({"email": "<EMAIL_FROM_ENV>"}, env={"GREENRAN_GUESTBOOK_EMAIL": "placeholder@example.com"})
