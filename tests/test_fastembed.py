"""
Tests for the FastEmbed embedding backend (tracer.embeddings.embedder.Embedder.from_fastembed).

These tests are independent of the sentence-transformers backend and can be run
with:

    pip install tracer-llm[fastembed,dev]
    pytest tests/test_fastembed.py -v

No PyTorch or CUDA required.
"""

import numpy as np
import pytest

from tracer.embeddings.embedder import Embedder


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fastembed_model_name():
    """Return a small, fast model for testing. BGE-small is a good default."""
    return "BAAI/bge-small-en-v1.5"


@pytest.fixture(scope="module")
def embedder(fastembed_model_name):
    """
    Module-level fixture: load the FastEmbed model once and reuse it.
    Saves re-downloading the model for every test.
    """
    # Skip all tests in this module if fastembed is not installed.
    pytest.importorskip("fastembed")
    return Embedder.from_fastembed(fastembed_model_name)


# ── Basic shape / dtype / non-degeneracy tests ─────────────────────────────────

def test_embed_returns_2d_float32(embedder):
    """Embedding a list of strings should return a 2D float32 array."""
    texts = ["hello world", "this is a test"]
    result = embedder.embed(texts)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape[0] == len(texts)
    assert result.dtype == np.float32


def test_embed_one_returns_1d(embedder):
    """embed_one should return a 1D vector."""
    vec = embedder.embed_one("test sentence")
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert vec.dtype == np.float32


def test_embedding_dimension_consistency(embedder):
    """All embeddings from the same model must have the same dimension."""
    dim = embedder.embed_one("a").shape[0]
    batch = embedder.embed(["a", "b", "c"])
    assert batch.shape[1] == dim


def test_embed_empty_list_returns_empty(embedder):
    """Passing an empty list should return an empty 2D array (0 x dim)."""
    result = embedder.embed([])
    dim = embedder.embed_one("test").shape[0]
    print(f"\n  Empty result shape: {result.shape}, dim: {dim}")  # Add this
    assert result.shape == (0, dim)
    assert result.dtype == np.float32


def test_embeddings_are_finite_and_non_zero(embedder):
    """Embeddings should not contain NaNs, infinities, or be all-zero."""
    texts = ["quick brown fox", "jumps over", "the lazy dog"]
    vecs = embedder.embed(texts)
    assert np.all(np.isfinite(vecs))
    # At least one vector should have non-zero norm (not degenerate).
    norms = np.linalg.norm(vecs, axis=1)
    assert np.all(norms > 0.0)


# ── Normalisation tests (L2) ──────────────────────────────────────────────────

def test_normalized_embeddings_have_unit_l2_norm(embedder):
    """When normalize=True (default), every embedding vector should have L2 norm ≈ 1."""
    texts = ["one", "two words", "a much longer sentence to test normalization"]
    vecs = embedder.embed(texts)
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_embed_one_normalization(embedder):
    """Single text normalization should also produce unit vector."""
    vec = embedder.embed_one("single")
    norm = np.linalg.norm(vec)
    np.testing.assert_allclose(norm, 1.0, atol=1e-5)


def test_normalization_is_consistent_across_calls(embedder):
    """Repeated embeddings of the same text should yield the same normalized vector."""
    text = "consistency check"
    v1 = embedder.embed_one(text)
    v2 = embedder.embed_one(text)
    np.testing.assert_array_almost_equal(v1, v2, decimal=5)


def test_normalization_can_be_disabled():
    """When normalize=False, the call should succeed and produce valid vectors.
    We cannot guarantee they are *not* unit length because some models (like BGE)
    naturally output normalized embeddings."""
    pytest.importorskip("fastembed")
    embedder_no_norm = Embedder.from_fastembed(
        "BAAI/bge-small-en-v1.5", normalize=False
    )
    vec = embedder_no_norm.embed_one("hello")
    assert vec.ndim == 1
    assert vec.dtype == np.float32
    assert np.all(np.isfinite(vec))
    # No assertion about norm — the raw model output is accepted as-is.


# ── Batch vs single consistency ───────────────────────────────────────────────

def test_batch_consistent_with_single(embedder):
    """Embedding a batch should give the same result as embedding each text individually."""
    texts = ["sample A", "sample B", "sample C"]
    batch = embedder.embed(texts)
    singles = np.vstack([embedder.embed_one(t) for t in texts])
    # Allow for tiny numerical differences; they should be very close.
    np.testing.assert_allclose(batch, singles, rtol=1e-5, atol=1e-7)


# ── Reproducibility ───────────────────────────────────────────────────────────

def test_embeddings_are_deterministic(embedder):
    """Two calls with the same input must produce identical embeddings."""
    texts = ["determinism", "test", "again"]
    first = embedder.embed(texts)
    second = embedder.embed(texts)
    np.testing.assert_array_equal(first, second)


# ── Backend metadata ──────────────────────────────────────────────────────────

def test_repr_shows_fastembed(embedder):
    """repr() should identify the backend as 'fastembed'."""
    rep = repr(embedder)
    assert "fastembed" in rep
    assert "bge-small" in rep.lower()


# ── Edge case: very long text (no crash) ──────────────────────────────────────

def test_long_text_does_not_crash(embedder):
    """Very long input should not cause shape mismatches or exceptions."""
    long_text = "word " * 500
    vec = embedder.embed_one(long_text)
    assert vec.ndim == 1
    # Should still be normalized
    np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-5)


# ── Integration with Router (light touch) ─────────────────────────────────────

def test_embedder_can_be_used_with_router():
    """
    Verify that the embedder's output is accepted by the Router's internal
    conversion (the Router expects float32 arrays of correct dimension).
    We don't need a full .tracer artifact; just check that _to_embedding works.
    """
    pytest.importorskip("fastembed")
    from tracer.runtime.router import Router

    # Create a minimal mock manifest and dummy stages so Router can be instantiated.
    class MockManifest:
        embedding_dim = 384  # bge-small dimension

    embedder = Embedder.from_fastembed("BAAI/bge-small-en-v1.5")
    router = Router(stages=[], label_space=[], manifest=MockManifest(), embedder=embedder)

    # Calling _to_embedding directly (part of Router API) should work.
    vec = router._to_embedding("test")
    assert vec.shape == (384,)
    assert vec.dtype == np.float32

    # Batch version
    batch = router._to_embeddings(["a", "b"])
    assert batch.shape == (2, 384)

    # Passing raw embedding (non-text) should be left untouched
    raw = np.random.randn(384).astype(np.float32)
    assert np.array_equal(router._to_embedding(raw), raw)