"""Tests for threshold calibration: the small-set fallback fix (#77) and
per-class calibration with small-class refusal (#38)."""

from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tracer.config import FitConfig  # noqa: E402
from tracer.fit.pipeline import (  # noqa: E402
    _apply_per_class_thresholds,
    _calibrate_threshold,
    _calibrate_threshold_per_class,
    apply_stage,
    build_global,
)
import pytest  # noqa: E402


# ── #77: small-set fallback must keep maximum coverage ────────────────────────

def test_small_set_fallback_keeps_max_coverage():
    # 36 rows (< 40 → single-set branch), perfect agreement everywhere: every
    # threshold with >= min_accept accepted rows clears the CP gate, so the
    # correct pick is the lowest passing threshold, which accepts all rows.
    # The pre-fix behavior kept the highest passing threshold instead
    # (coverage 10/36 ≈ 0.28).
    scores = np.linspace(0.1, 0.99, 36)
    preds = np.zeros(36, dtype=int)
    teacher = np.zeros(36, dtype=int)
    ti = _calibrate_threshold(scores, preds, teacher, target_ta=0.5, alpha=0.1)
    assert ti is not None
    assert ti["holdout"] is False
    assert ti["coverage"] == 1.0
    assert ti["threshold"] == float(scores.min())


def test_small_set_fallback_still_gates_on_lower_bound():
    # 30 rows at 50% agreement: no threshold can clear a 0.9 target.
    rng = np.random.RandomState(0)
    scores = rng.uniform(size=30)
    preds = np.zeros(30, dtype=int)
    teacher = (rng.uniform(size=30) < 0.5).astype(int)  # half disagree
    ti = _calibrate_threshold(scores, preds, teacher, target_ta=0.9, alpha=0.1)
    assert ti is None


# ── #38: per-class calibration ────────────────────────────────────────────────

def _imbalanced_case(seed=0, n_major=400, n_minor=60):
    """Majority class 0: high agreement, informative scores. Minority class 1:
    coin-flip agreement, uninformative scores. Pooled agreement clears 0.90;
    class-1 agreement cannot."""
    rng = np.random.RandomState(seed)
    preds = np.array([0] * n_major + [1] * n_minor)
    teacher = preds.copy()
    # 4 majority errors → 0.99 agreement; teacher differs on those rows.
    err_major = rng.choice(n_major, size=4, replace=False)
    teacher[err_major] = 1
    # Minority: 50% agreement, independent of score.
    err_minor = rng.choice(n_minor, size=n_minor // 2, replace=False) + n_major
    teacher[err_minor] = 0
    scores = np.empty(len(preds))
    # Majority scores informative: errors score low.
    scores[:n_major] = rng.uniform(0.6, 1.0, size=n_major)
    scores[err_major] = rng.uniform(0.0, 0.3, size=4)
    # Minority scores carry no signal.
    scores[n_major:] = rng.uniform(size=n_minor)
    return scores, preds, teacher


def test_global_threshold_masks_minority_class():
    # Documents the failure mode: pooled calibration certifies a threshold
    # that routes minority traffic at coin-flip agreement.
    scores, preds, teacher = _imbalanced_case()
    ti = _calibrate_threshold(scores, preds, teacher, target_ta=0.90, alpha=0.1)
    assert ti is not None
    accepted = scores >= ti["threshold"]
    minority_accepted = accepted & (preds == 1)
    assert minority_accepted.sum() > 0
    minority_agreement = (preds[minority_accepted] == teacher[minority_accepted]).mean()
    assert minority_agreement < 0.90  # the pooled gate hides this


def test_per_class_defers_uncertifiable_minority():
    scores, preds, teacher = _imbalanced_case()
    ti = _calibrate_threshold_per_class(scores, preds, teacher, target_ta=0.90,
                                        alpha=0.1)
    assert ti is not None
    # Majority certified, minority refused (no certifiable threshold).
    assert 0 in ti["thresholds"]
    assert 1 not in ti["thresholds"]
    assert ti["per_class"][1]["status"] == "no_certifiable_threshold"
    # Every accepted row now belongs to a certified class.
    accept = _apply_per_class_thresholds(scores, preds, ti["thresholds"])
    assert (preds[accept] == 0).all()
    accepted_agreement = (preds[accept] == teacher[accept]).mean()
    assert accepted_agreement >= 0.90
    # Majority coverage survives the change.
    assert accept[preds == 0].mean() > 0.5


def test_per_class_refuses_small_classes():
    scores, preds, teacher = _imbalanced_case(n_minor=60)
    # Add a 10-row class with perfect agreement: still refused, the evidence
    # cannot certify the target regardless of how clean it looks.
    scores = np.concatenate([scores, np.full(10, 0.99)])
    preds = np.concatenate([preds, np.full(10, 2, dtype=int)])
    teacher = np.concatenate([teacher, np.full(10, 2, dtype=int)])
    ti = _calibrate_threshold_per_class(scores, preds, teacher, target_ta=0.90,
                                        alpha=0.1, min_class_n=25)
    assert ti is not None
    assert 2 not in ti["thresholds"]
    assert ti["per_class"][2] == {"status": "insufficient_calibration",
                                  "n_cal": 10, "min_class_n": 25}


def test_per_class_returns_none_when_nothing_certifies():
    rng = np.random.RandomState(1)
    scores = rng.uniform(size=60)
    preds = np.array([0] * 30 + [1] * 30)
    teacher = 1 - preds  # zero agreement everywhere
    assert _calibrate_threshold_per_class(scores, preds, teacher, 0.9) is None


# ── apply_stage integration ──────────────────────────────────────────────────

class _FixedClf:
    """Deterministic stand-in classifier: argmax of stored probabilities."""

    def __init__(self, probs):
        self._probs = np.asarray(probs, dtype=float)

    def predict_proba(self, X):
        return self._probs[: len(X)]


def test_apply_stage_per_class_thresholds_route_and_defer():
    probs = np.array([
        [0.9, 0.1],   # class 0, score 0.9 → accepted (τ0 = 0.6)
        [0.65, 0.35], # class 0, score 0.65 → accepted
        [0.55, 0.45], # class 0, score 0.55 → below τ0, deferred
        [0.2, 0.8],   # class 1 → no threshold, always deferred
    ])
    stage = {"clf": _FixedClf(probs), "acceptor": None, "accept_all": False,
             "threshold": None, "per_class_thresholds": {0: 0.6}}
    X = np.zeros((4, 3))
    preds, accept, scores = apply_stage(stage, X)
    assert preds.tolist() == [0, 0, 0, 1]
    assert accept.tolist() == [True, True, False, False]


def test_apply_stage_global_threshold_unchanged():
    probs = np.array([[0.9, 0.1], [0.55, 0.45]])
    stage = {"clf": _FixedClf(probs), "acceptor": None, "accept_all": False,
             "threshold": 0.6}
    preds, accept, _ = apply_stage(stage, np.zeros((2, 3)))
    assert accept.tolist() == [True, False]


# ── build_global per-class gate ──────────────────────────────────────────────

def _clustered_split(seed=7, n_major=300, n_minor=40, minor_noise=0.5):
    """Separable two-cluster data; minority teacher labels are noisy at rate
    `minor_noise`, so the surrogate cannot agree with the teacher there."""
    rng = np.random.RandomState(seed)
    Xa = rng.randn(n_major, 8) + 4.0
    Xb = rng.randn(n_minor, 8) - 4.0
    X = np.vstack([Xa, Xb])
    y = np.array([0] * n_major + [1] * n_minor)
    flip = rng.uniform(size=n_minor) < minor_noise
    y[n_major:][flip] = 0  # teacher calls half the minority rows class 0
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    n_val = len(X) // 5
    n_cal = len(X) // 5
    return {"X_train": X[: -n_val - n_cal], "y_train": y[: -n_val - n_cal],
            "X_val": X[-n_val - n_cal: -n_cal], "y_val": y[-n_val - n_cal: -n_cal],
            "X_cal": X[-n_cal:], "y_cal": y[-n_cal:], "n_fit": len(X)}


def test_build_global_per_class_gate_blocks_masked_minority():
    split = _clustered_split()
    pooled = build_global(split, target_ta=0.85, alpha=0.1)
    per_cls = build_global(split, target_ta=0.85, alpha=0.1,
                           per_class=True, min_class_n=10)
    # Pooled gate certifies accept-all; the per-class gate refuses it because
    # the minority class cannot hold the target on its own.
    assert pooled["summary"]["status"] == "ok"
    assert per_cls["summary"]["status"] == "below_target_per_class"
    assert per_cls["summary"]["failing_class"] == 1
    assert per_cls["stages"] == []


# ── FitConfig plumbing ───────────────────────────────────────────────────────

def test_fit_config_validates_min_class_n():
    FitConfig(per_class_calibration=True, min_class_calibration_n=1)
    with pytest.raises(ValueError):
        FitConfig(min_class_calibration_n=0)
