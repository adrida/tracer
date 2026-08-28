"""Parity-gate honesty: held-out lower-bound gating and NaN robustness.

Regression tests for the in-sample-optimism fix. The gate must refuse to
certify when the accepted set is too small to support the target agreement,
and must not crash on NaN surrogate probabilities.
"""
import numpy as np

from tracer.fit.pipeline import (
    build_global, _split_buffer, _accept_features, _fit_acceptor, _cp_lower,
    _calibrate_threshold,
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


def _max_qualifying_coverage(scores, preds, y, target, alpha, min_accept):
    """Highest coverage among thresholds whose CP lower bound clears target."""
    for t in np.unique(np.sort(scores)):
        acc = scores >= t
        n_acc = int(acc.sum())
        if n_acc < min_accept:
            continue
        k_acc = int((preds[acc] == y[acc]).sum())
        if _cp_lower(k_acc, n_acc, alpha) >= target:
            return float(acc.mean())
    return None


def test_single_set_gate_deploys_max_coverage_threshold():
    # n < 40 -> single-set fallback branch. Among the thresholds that clear the
    # same CP bound, the gate must deploy the highest-coverage one, as the
    # holdout branch does. It previously kept the last (lowest-coverage) one.
    n = 36
    scores = np.linspace(0.50, 0.99, n)
    y = np.zeros(n, dtype=int)
    preds = y.copy()
    preds[:6] = 1  # the 6 lowest-scoring rows disagree with the teacher

    out = _calibrate_threshold(scores, preds, y, 0.80, alpha=0.1, min_accept=10)
    assert out is not None and out["holdout"] is False

    best = _max_qualifying_coverage(scores, preds, y, 0.80, 0.1, 10)
    assert best is not None
    assert out["coverage"] == best
    # the guarantee still holds at the deployed threshold
    assert out["teacher_agreement_lower"] >= 0.80


def test_single_set_gate_coverage_does_not_shrink_with_more_data():
    # On cleanly-separated calibration data the accepted set must grow with n,
    # not stay pinned at min_accept.
    covs = []
    for n in (20, 30, 39):
        scores = np.linspace(0.50, 0.99, n)
        y = np.zeros(n, dtype=int)
        out = _calibrate_threshold(scores, y.copy(), y, 0.80, alpha=0.1, min_accept=10)
        assert out is not None
        covs.append(out["coverage"])
    assert all(c == 1.0 for c in covs), covs


def test_nan_probs_do_not_crash_acceptor():
    probs = np.array([[0.7, 0.3], [np.nan, np.nan], [0.6, 0.4], [0.2, 0.8]])
    feats = _accept_features(probs)
    assert np.isfinite(feats).all()
    # must not raise
    _fit_acceptor(probs, np.array([0, 1, 0, 1]), np.array([0, 0, 0, 1]))


def test_routing_robust_to_extreme_ood_inputs():
    """Out-of-distribution / extreme embeddings must route without crashing and
    produce finite accept scores. (The OSS library defers via the acceptor
    threshold; it does not ship a separate distance-based OOD gate, which lives
    in the hosted gateway.)"""
    from tracer.fit.pipeline import build_l2d, apply_stage
    split = _make_split(800, 3.0)
    res = build_l2d(split, target_ta=0.9, alpha=0.1)
    if not res["stages"]:
        import pytest
        pytest.skip("no stage certified on this split")
    stage = res["stages"][0]
    d = split["X_cal"].shape[1]
    X = np.vstack([
        split["X_cal"][:5],                  # in-domain
        np.full((3, d), 1e6, dtype=float),   # extreme, far OOD
        np.full((2, d), -1e6, dtype=float),  # extreme, opposite direction
    ]).astype(np.float32)
    preds, accept, scores = apply_stage(stage, X)
    assert accept.shape == (X.shape[0],) and accept.dtype == bool
    assert np.isfinite(scores).all()         # no NaN scores leak through
