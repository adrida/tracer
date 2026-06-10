"""Throughput benchmark for the dynamic micro-batcher (issue #27).

Measures how many embedding+surrogate forward passes are needed to serve a burst
of concurrent predictions, with and without :class:`tracer.AsyncBatcher`. Without
batching, N concurrent ``apredict`` calls each run their own ``(1, dim)`` pass.
With the batcher, requests that arrive within ``max_wait`` are fused into a single
``(k, dim)`` pass, so the surrogate stops being a per-request bottleneck.

The structural metric (number of forward passes) is exact and machine-independent.
A ``--work-ms`` knob optionally adds a fixed per-pass cost to illustrate the
wall-clock effect when each forward pass is latency-bound -- e.g. a real
sentence-transformers model or a remote embedding API -- which is the regime this
feature targets.

    python benchmarks/batching_throughput.py
    python benchmarks/batching_throughput.py --requests 2000 --max-batch 64 --work-ms 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path

import numpy as np

import tracer
from tracer import AsyncBatcher
from tracer.runtime.router import Router


def _build_router(tmpdir: Path, dim: int = 32, n: int = 600, n_classes: int = 5) -> Router:
    rng = np.random.RandomState(0)
    centers = rng.randn(n_classes, dim) * 4
    labels = rng.randint(0, n_classes, size=n)
    X = (centers[labels] + rng.randn(n, dim) * 0.6).astype(np.float32)
    names = [f"cls_{i}" for i in range(n_classes)]
    teacher = [names[i] for i in labels]
    traces = tmpdir / "traces.jsonl"
    with traces.open("w") as f:
        for i in range(n):
            f.write(json.dumps({"input": f"t{i}", "teacher": teacher[i], "id": str(i)}) + "\n")
    artifact = tmpdir / ".tracer"
    tracer.fit(traces, artifact, embeddings=X, config=tracer.FitConfig(verbose=False))
    router = Router.load(artifact)
    router._bench_X = X  # stash sample embeddings for the driver
    return router


class _Instrumented:
    """Count forward (embedding) passes on a router and add an optional cost.

    Both prediction paths funnel their embeddings through ``_to_embedding`` (single)
    or ``_to_embeddings`` (batch); wrapping both counts one pass per surrogate
    call and lets ``--work-ms`` model a latency-bound embedder/forward pass.
    """

    def __init__(self, router: Router, work_ms: float):
        self.router = router
        self.work = work_ms / 1000.0
        self.calls = 0
        self._one = router._to_embedding
        self._many = router._to_embeddings

    def __enter__(self) -> "_Instrumented":
        def one(x):
            self.calls += 1
            if self.work:
                time.sleep(self.work)
            return self._one(x)

        def many(xs):
            self.calls += 1
            if self.work:
                time.sleep(self.work)
            return self._many(xs)

        self.router._to_embedding = one
        self.router._to_embeddings = many
        return self

    def __exit__(self, *exc):
        self.router._to_embedding = self._one
        self.router._to_embeddings = self._many


async def _bench(router: Router, requests: int, max_batch: int,
                 max_wait: float, work_ms: float):
    X = router._bench_X
    inputs = [X[i % len(X)] for i in range(requests)]

    with _Instrumented(router, work_ms) as inst:
        # Without batching: one forward pass per request.
        inst.calls = 0
        t0 = time.perf_counter()
        for x in inputs:
            await router.apredict(x)
        seq_time = time.perf_counter() - t0
        seq_passes = inst.calls

        # With the batcher: concurrent requests fused into vectorized passes.
        inst.calls = 0
        t0 = time.perf_counter()
        async with AsyncBatcher(router, max_batch_size=max_batch, max_wait=max_wait) as b:
            await asyncio.gather(*(b.predict(x) for x in inputs))
            stats = b.stats
        bat_time = time.perf_counter() - t0
        bat_passes = inst.calls

    print(f"  requests           {requests}")
    print(f"  max_batch_size     {max_batch}   max_wait {max_wait * 1000:.1f} ms")
    print(f"  forward passes     sequential={seq_passes}   batched={bat_passes}"
          f"   ({seq_passes / max(bat_passes, 1):.1f}x fewer)")
    print(f"  mean batch size    {stats['avg_batch_size']:.1f}")
    if work_ms:
        print(f"  wall clock         sequential={seq_time * 1000:.0f} ms"
              f"   batched={bat_time * 1000:.0f} ms"
              f"   ({seq_time / max(bat_time, 1e-9):.1f}x faster)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--requests", type=int, default=1000)
    p.add_argument("--max-batch", type=int, default=64)
    p.add_argument("--max-wait", type=float, default=0.005)
    p.add_argument("--work-ms", type=float, default=0.0,
                   help="simulated per-forward-pass cost in milliseconds")
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as d:
        router = _build_router(Path(d))
        print("\n  TRACER  micro-batching throughput")
        asyncio.run(_bench(router, args.requests, args.max_batch,
                           args.max_wait, args.work_ms))
        print()


if __name__ == "__main__":
    main()
