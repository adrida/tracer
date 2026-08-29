"""Tests for EmbeddingIndex.build (embeddings/index.py).

Regression coverage for a real bug: build() used to normalize its input
array IN PLACE for the cosine metric, silently rescaling whatever array the
caller passed in (e.g. api.py reused the same `X` for OOD-gate calibration
right before building the index). It must not mutate the caller's array, and
`.embeddings` must match exactly what was indexed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tracer.embeddings.index import EmbeddingIndex


def _unnormalized(n=20, dim=8, seed=0):
    rng = np.random.RandomState(seed)
    return (rng.normal(0, 1, (n, dim)) * 5.0).astype(np.float32)


def test_build_does_not_mutate_caller_array():
    pytest.importorskip("faiss")
    X = _unnormalized()
    original = X.copy()
    EmbeddingIndex.build(X, metric="cosine")
    np.testing.assert_array_equal(X, original)


def test_build_embeddings_attr_matches_what_was_indexed():
    pytest.importorskip("faiss")
    X = _unnormalized()
    idx = EmbeddingIndex.build(X, metric="cosine")
    # .embeddings should already be L2-normalized (what the FAISS index holds),
    # not the raw un-normalized input -- so the two never silently diverge.
    norms = np.linalg.norm(idx.embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    # The nearest neighbour of a training point (searched via the same
    # normalization) should be itself.
    I, _D = idx.search(X[0], k=1)
    assert I[0] == 0


def test_build_no_faiss_falls_back_without_mutating(monkeypatch):
    monkeypatch.setitem(sys.modules, "faiss", None)  # forces `import faiss` to raise ImportError

    X = _unnormalized()
    original = X.copy()
    idx = EmbeddingIndex.build(X, metric="cosine")
    np.testing.assert_array_equal(X, original)
    np.testing.assert_array_equal(idx.embeddings, original)
