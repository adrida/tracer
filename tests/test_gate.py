"""Parity-gate honesty: held-out lower-bound gating and NaN robustness.

Regression tests for the in-sample-optimism fix. The gate must refuse to
certify when the accepted set is too small to support the target agreement,
and must not crash on NaN surrogate probabilities.
"""
import numpy as np

from tracer.fit.pipeline import (
    build_global, _split_buffer, _accept_features, _fit_acceptor, _cp_lower,
)


def _make_split(n_per, sep, d=8, seed=0):
    rng = np.random.RandomState(seed)
    Xa = rng.randn(n_per, d) + np.r_[sep, np.zeros(d - 1)]
    Xb = rng.randn(n_per, d) - np.r_[sep, np.zeros(d - 1)]
    X = np.vstack([Xa, Xb])
    y = np.r_[np.zeros(n_per), np.ones(n_per)].astype(int)
    return _split_buffer(X, y)


def test_cp_lower_below_point_and_monotone():
    assert _cp_lower(90, 100, 0.1) < 0.90
    assert _cp_lower(9, 10, 0.1) < _cp_lower(900, 1000, 0.1)
    assert _cp_lower(0, 0, 0.1) == 0.0


def test_low_n_refuses_accept_all():
    # Tiny calibration set: the point estimate can hit 1.0 by luck, but the
    # lower bound is well under target, so the gate must refuse.
    g = build_global(_make_split(20, 3.0), target_ta=0.9, alpha=0.1)
    assert g["summary"]["status"] == "below_target"
    assert g["summary"]["teacher_agreement_lower_cal_total"] < 0.9


def test_large_separable_certifies_with_lower_bound():
    g = build_global(_make_split(800, 3.0), target_ta=0.9, alpha=0.1)
    assert g["summary"]["status"] == "ok"
    # the reported certification is the honest lower bound, and it clears target
    assert g["summary"]["teacher_agreement_lower_cal_total"] >= 0.9


def test_nan_probs_do_not_crash_acceptor():
    probs = np.array([[0.7, 0.3], [np.nan, np.nan], [0.6, 0.4], [0.2, 0.8]])
    feats = _accept_features(probs)
    assert np.isfinite(feats).all()
    # must not raise
    _fit_acceptor(probs, np.array([0, 1, 0, 1]), np.array([0, 0, 0, 1]))
