"""Tests for the distance-based OOD gate (fit/ood.py).

Base deps only (numpy, scikit-learn). The gate must defer clearly off-distribution
inputs while leaving in-distribution traffic alone.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tracer.fit.ood import fit_ood_gate, ood_mask


def _train(n=500, dim=16, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(0, 1, (n, dim)).astype(np.float32)
    labels = ["a" if i % 2 else "b" for i in range(n)]
    return X, labels


def test_gate_calibrates_and_has_thresholds():
    X, labels = _train()
    gate = fit_ood_gate(X, labels)
    assert gate is not None
    assert gate["global_thr"] > 0
    assert set(gate["per_label_thr"]) <= {"a", "b"}


def test_far_points_are_flagged_in_distribution_passes():
    X, labels = _train()
    gate = fit_ood_gate(X, labels)
    rng = np.random.RandomState(1)
    X_in = rng.normal(0, 1, (40, 16)).astype(np.float32)        # same distribution
    X_far = rng.normal(0, 1, (40, 16)).astype(np.float32) + 50  # far away
    q = ["a"] * 40
    # almost all in-distribution queries pass (not flagged)
    assert ood_mask(X_in, X, q, gate).mean() < 0.2
    # every far query is flagged OOD
    assert ood_mask(X_far, X, q, gate).all()


def test_gate_none_on_tiny_data():
    X = np.zeros((5, 4), dtype=np.float32)
    assert fit_ood_gate(X, ["a"] * 5) is None


def test_ood_mask_no_gate_is_noop():
    X = np.zeros((3, 4), dtype=np.float32)
    out = ood_mask(X, X, ["a", "a", "a"], None)
    assert out.shape == (3,) and not out.any()


def test_ood_mask_with_prefitted_nn():
    X, labels = _train()
    gate = fit_ood_gate(X, labels)
    from sklearn.neighbors import NearestNeighbors
    k = gate["k"]
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    X_in = X[:10]
    q = ["a"] * 10
    # verify it runs and returns correct output when passing pre-fitted nn
    out1 = ood_mask(X_in, X, q, gate, nn=nn)
    out2 = ood_mask(X_in, X, q, gate)
    assert np.array_equal(out1, out2)
