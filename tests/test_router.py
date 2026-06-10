"""Tests for the production Router (runtime/router.py).

Covers the Router interface directly — no HTTP server involved.
Complements test_serve.py (HTTP layer) and the routing smoke tests in
test_fit.py.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import tracer


# ── Synthetic data ───────────────────────────────────────────────────────────

def _make_traces(tmpdir, n=240, dim=16, n_classes=3, noise=0.05):
    """Well-separated synthetic traces so fit() deploys a surrogate."""
    rng = np.random.RandomState(0)
    centers = rng.randn(n_classes, dim) * 4
    labels_int = rng.randint(0, n_classes, size=n)
    X = centers[labels_int] + rng.randn(n, dim) * 0.6
    names = [f"cls_{i}" for i in range(n_classes)]
    teacher = [names[i] for i in labels_int]
    for i in range(n):
        if rng.random() < noise:
            teacher[i] = names[rng.randint(0, n_classes)]
    path = Path(tmpdir) / "traces.jsonl"
    with path.open("w") as f:
        for i in range(n):
            f.write(json.dumps({"input": f"text {i}", "teacher": teacher[i],
                                "id": str(i)}) + "\n")
    return path, X.astype(np.float32)


# ── Module-scoped fixture (one fit per file) ─────────────────────────────────

@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    """Fit once, reuse for every test in this module."""
    tmpdir = tmp_path_factory.mktemp("router")
    traces_path, X = _make_traces(tmpdir)
    artifact_dir = Path(tmpdir) / ".tracer"
    result = tracer.fit(traces_path, artifact_dir, embeddings=X,
                        config=tracer.FitConfig(verbose=False))
    assert result.manifest.selected_method is not None, \
        "fixture expected a deployable policy"
    return artifact_dir, X


@pytest.fixture(scope="module")
def router(fitted):
    artifact_dir, _ = fitted
    return tracer.load_router(artifact_dir)


# ── Single prediction ────────────────────────────────────────────────────────

def test_predict_single_returns_expected_keys(router, fitted):
    _, X = fitted
    out = router.predict(X[0])
    assert set(out.keys()) == {"label", "decision", "accept_score", "stage"}


def test_predict_single_decision_is_valid(router, fitted):
    _, X = fitted
    out = router.predict(X[0])
    assert out["decision"] in ("handled", "deferred")


def test_predict_single_score_and_stage_types(router, fitted):
    _, X = fitted
    out = router.predict(X[0])
    assert isinstance(out["accept_score"], float)
    assert isinstance(out["stage"], int)


# ── Batch prediction ─────────────────────────────────────────────────────────

def test_predict_batch_output_lengths(router, fitted):
    _, X = fitted
    batch = X[:10]
    out = router.predict_batch(batch)
    assert len(out["labels"]) == 10
    assert len(out["decisions"]) == 10
    assert len(out["handled"]) == 10
    assert len(out["preds"]) == 10
    assert len(out["stage_id"]) == 10


def test_predict_batch_decisions_are_valid(router, fitted):
    _, X = fitted
    out = router.predict_batch(X[:10])
    assert all(d in ("handled", "deferred") for d in out["decisions"])


# ── Dimension mismatch ────────────────────────────────────────────────────────

def test_predict_wrong_dim_raises(router):
    wrong = np.ones(99, dtype=np.float32)  # 99 != 16
    with pytest.raises(ValueError, match="(?i)dimension mismatch"):
        router.predict(wrong)


def test_predict_batch_wrong_dim_raises(router):
    wrong = np.ones((5, 99), dtype=np.float32)
    with pytest.raises(ValueError, match="(?i)dimension mismatch"):
        router.predict_batch(wrong)


# ── Text input without embedder ──────────────────────────────────────────────

def test_predict_text_without_embedder_raises(router):
    with pytest.raises(ValueError, match="(?i)no embedder"):
        router.predict("some text")


def test_predict_batch_text_without_embedder_raises(router):
    with pytest.raises(ValueError, match="(?i)no embedder"):
        router.predict_batch(["a", "b"])


# ── Text input with a fake embedder ──────────────────────────────────────────

class _FakeEmbedder:
    """Deterministic embedder that maps text to fixed-dim vectors."""

    def __init__(self, dim):
        self._dim = dim

    def embed_one(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], "little")
        rng = np.random.RandomState(seed)
        return rng.randn(self._dim).astype(np.float32)

    def embed(self, texts):
        return np.stack([self.embed_one(t) for t in texts])


def test_predict_text_with_embedder(fitted):
    artifact_dir, X = fitted
    dim = X.shape[1]
    router_e = tracer.load_router(artifact_dir, embedder=_FakeEmbedder(dim))
    out = router_e.predict("hello world")
    assert out["decision"] in ("handled", "deferred")
    assert "label" in out


def test_predict_batch_text_with_embedder(fitted):
    artifact_dir, X = fitted
    dim = X.shape[1]
    router_e = tracer.load_router(artifact_dir, embedder=_FakeEmbedder(dim))
    out = router_e.predict_batch(["hello", "goodbye"])
    assert len(out["labels"]) == 2
    assert len(out["decisions"]) == 2


# ── Fallback ─────────────────────────────────────────────────────────────────

def test_fallback_called_on_deferred(fitted):
    """If a prediction defers, the fallback should supply the label."""
    artifact_dir, X = fitted
    router_fb = tracer.load_router(artifact_dir)
    # Use extreme embeddings far from the training distribution to increase
    # the chance that the policy defers.
    rng = np.random.RandomState(99)
    deferred_found = False
    for _ in range(200):
        emb = rng.randn(X.shape[1]).astype(np.float32) * 100
        out = router_fb.predict(emb, fallback=lambda: "teacher_label")
        if out["decision"] == "deferred":
            assert out["label"] == "teacher_label"
            assert out["accept_score"] == 0.0
            assert out["stage"] == -1
            deferred_found = True
            break
    if not deferred_found:
        pytest.skip("no deferred prediction found — policy handled everything")
