"""Tests for high-throughput routing: batch fallback, async predict, and the
dynamic micro-batcher (runtime/router.py + runtime/batching.py).

Covers issues #28 (predict_batch fallback) and #27 (async / batch predict for
high-throughput pipelines). Async tests are driven with ``asyncio.run`` so the
suite needs no extra plugins beyond ``pytest``. All data is synthetic and the
artifact is built once per module in a temp dir.
"""

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import tracer
from tracer import AsyncBatcher
from tracer.runtime.router import Router, route_scored


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def router_pool(tmp_path_factory):
    """A fitted router plus a pool of samples split into handled / deferred.

    The pool deliberately mixes in-distribution points (handled) with ambiguous
    class midpoints and near-origin out-of-distribution points (deferred) so both
    routing paths are exercised.
    """
    tmpdir = tmp_path_factory.mktemp("batch_artifact")
    rng = np.random.RandomState(0)
    n, dim, n_classes = 300, 16, 3
    centers = rng.randn(n_classes, dim) * 4
    labels_int = rng.randint(0, n_classes, size=n)
    X = (centers[labels_int] + rng.randn(n, dim) * 0.6).astype(np.float32)
    names = [f"cls_{i}" for i in range(n_classes)]
    teacher = [names[i] for i in labels_int]
    # Inject teacher noise so the data is not perfectly separable; combined with a
    # strict target this yields a gated (non-accept-all) policy that genuinely
    # defers low-confidence inputs, so both routing paths can be tested.
    for i in range(n):
        if rng.random() < 0.10:
            teacher[i] = names[rng.randint(0, n_classes)]

    traces = Path(tmpdir) / "traces.jsonl"
    with traces.open("w") as f:
        for i in range(n):
            f.write(json.dumps({"input": f"text {i}", "teacher": teacher[i],
                                "id": str(i)}) + "\n")

    artifact = Path(tmpdir) / ".tracer"
    result = tracer.fit(traces, artifact, embeddings=X,
                        config=tracer.FitConfig(verbose=False,
                                                target_teacher_agreement=0.99))
    assert result.manifest.selected_method is not None, \
        "fixture expected a deployable policy"
    router = Router.load(artifact)

    # Points near the centroid of all class centers are ~equidistant from every
    # class, so the surrogate is maximally uncertain and the acceptor defers them.
    global_center = centers.mean(axis=0)
    ambiguous = (global_center + rng.randn(40, dim) * 0.3).astype(np.float32)
    pool = np.vstack([X[:30], ambiguous]).astype(np.float32)

    decisions = router.predict_batch(pool)["decisions"]
    handled = [i for i, d in enumerate(decisions) if d == "handled"]
    deferred = [i for i, d in enumerate(decisions) if d == "deferred"]
    return router, pool, handled, deferred


def test_fixture_has_both_paths(router_pool):
    """Sanity: the pool exercises both the handled and deferred paths."""
    _, _, handled, deferred = router_pool
    assert len(handled) > 0, "expected at least one handled sample"
    assert len(deferred) > 0, "expected at least one deferred sample"


# ── predict_batch fallback (issue #28) ────────────────────────────────────────

def test_predict_batch_fallback_fills_deferred(router_pool):
    router, pool, handled, deferred = router_pool
    out = router.predict_batch(pool, fallback=lambda original: "FALLBACK")
    for i in deferred:
        assert out["labels"][i] == "FALLBACK"
        assert out["decisions"][i] == "deferred"
    for i in handled:
        assert out["labels"][i] != "FALLBACK"
        assert out["labels"][i] is not None
        assert out["decisions"][i] == "handled"


def test_predict_batch_without_fallback_is_backward_compatible(router_pool):
    router, pool, handled, deferred = router_pool
    out = router.predict_batch(pool)
    for i in deferred:
        assert out["labels"][i] is None
    for i in handled:
        assert out["labels"][i] is not None


def test_predict_batch_exposes_accept_scores(router_pool):
    router, pool, _, _ = router_pool
    out = router.predict_batch(pool)
    assert "accept_scores" in out
    assert len(out["accept_scores"]) == len(pool)
    assert all(np.isfinite(s) for s in out["accept_scores"])


def test_predict_batch_fallback_receives_original_input(router_pool):
    router, pool, _, deferred = router_pool
    seen = {}

    def fallback(original):
        seen[id(original)] = original
        return "X"

    router.predict_batch(pool, fallback=fallback)
    # For an array input the fallback receives the embedding row itself.
    for i in deferred:
        # Find a captured original matching this row.
        assert any(np.allclose(np.asarray(v), pool[i]) for v in seen.values())


# ── async predict (issue #27) ─────────────────────────────────────────────────

def test_apredict_matches_predict_when_handled(router_pool):
    router, pool, handled, _ = router_pool
    i = handled[0]
    got = asyncio.run(router.apredict(pool[i]))
    assert got == router.predict(pool[i])


def test_apredict_sync_and_async_fallback(router_pool):
    router, pool, _, deferred = router_pool
    i = deferred[0]

    sync_out = asyncio.run(router.apredict(pool[i], fallback=lambda: "SYNC"))
    assert sync_out["label"] == "SYNC"
    assert sync_out["decision"] == "deferred"

    async def afb():
        await asyncio.sleep(0)
        return "ASYNC"

    async_out = asyncio.run(router.apredict(pool[i], fallback=afb))
    assert async_out["label"] == "ASYNC"
    assert async_out["decision"] == "deferred"


def test_apredict_batch_matches_predict_batch(router_pool):
    router, pool, _, _ = router_pool
    a = asyncio.run(router.apredict_batch(pool))
    b = router.predict_batch(pool)
    assert a["labels"] == b["labels"]
    assert a["decisions"] == b["decisions"]
    assert np.array_equal(a["handled"], b["handled"])


def test_apredict_batch_fallback_bounded_concurrency(router_pool):
    router, pool, _, deferred = router_pool
    # Build a batch with many deferred items to exercise concurrency.
    idx = (deferred * 5)[:12]
    batch = pool[idx]

    state = {"active": 0, "peak": 0}

    async def afb(original):
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(0.01)
        state["active"] -= 1
        return "TEACHER"

    out = asyncio.run(router.apredict_batch(batch, fallback=afb, max_concurrency=4))
    # Every item in this batch is deferred, so all get the teacher label.
    assert all(lbl == "TEACHER" for lbl in out["labels"])
    assert state["peak"] <= 4
    assert state["peak"] >= 2  # genuinely ran concurrently


# ── AsyncBatcher (dynamic micro-batching, issue #27) ──────────────────────────

def test_async_batcher_results_match_router(router_pool):
    router, pool, _, _ = router_pool

    async def run():
        async with AsyncBatcher(router, max_batch_size=64, max_wait=0.02) as b:
            outs = await asyncio.gather(*(b.predict(pool[i]) for i in range(len(pool))))
            return outs, b.stats

    outs, stats = asyncio.run(run())
    for i, out in enumerate(outs):
        ref = router.predict(pool[i])
        # Routing decision is identical; accept_score matches to float tolerance
        # (batched (k, dim) vs single (1, dim) inference differ at the ~1e-8 level).
        assert out["label"] == ref["label"], f"label mismatch at {i}"
        assert out["decision"] == ref["decision"], f"decision mismatch at {i}"
        assert out["stage"] == ref["stage"], f"stage mismatch at {i}"
        assert out["accept_score"] == pytest.approx(ref["accept_score"], abs=1e-6)
    # Coalescing actually happened: fewer batches than requests.
    assert stats["items_seen"] == len(pool)
    assert stats["batches_run"] < stats["items_seen"]
    assert stats["avg_batch_size"] > 1.0


def test_async_batcher_fallback(router_pool):
    router, pool, _, deferred = router_pool

    async def run():
        async with AsyncBatcher(router, max_batch_size=32, max_wait=0.02,
                                fallback=lambda original: "TEACHER") as b:
            return await asyncio.gather(*(b.predict(pool[i]) for i in deferred))

    outs = asyncio.run(run())
    assert all(o["label"] == "TEACHER" and o["decision"] == "deferred" for o in outs)


def test_async_batcher_closed_rejects(router_pool):
    router, _, _, _ = router_pool

    async def run():
        b = AsyncBatcher(router)
        await b.aclose()
        with pytest.raises(RuntimeError):
            await b.predict(np.zeros(router.manifest.embedding_dim, dtype=np.float32))

    asyncio.run(run())


def test_async_batcher_propagates_dim_mismatch(router_pool):
    router, _, _, _ = router_pool

    async def run():
        async with AsyncBatcher(router, max_wait=0.0) as b:
            with pytest.raises(ValueError):
                await b.predict(np.zeros(3, dtype=np.float32))

    asyncio.run(run())


def test_async_batcher_idempotent_close(router_pool):
    router, pool, _, _ = router_pool

    async def run():
        b = AsyncBatcher(router)
        await b.predict(pool[0])
        await b.aclose()
        await b.aclose()  # second close is a no-op, must not raise

    asyncio.run(run())


# ── route_scored helper ───────────────────────────────────────────────────────

def test_route_scored_matches_route_pipeline(router_pool):
    from tracer.fit.pipeline import route_pipeline

    router, pool, _, _ = router_pool
    preds, handled, stage_id, scores = route_scored(router._stages, pool)
    p2, h2, s2 = route_pipeline(router._stages, pool)
    assert np.array_equal(preds, p2)
    assert np.array_equal(handled, h2)
    assert np.array_equal(stage_id, s2)
    assert scores.shape == (len(pool),)
    assert np.all(np.isfinite(scores))
