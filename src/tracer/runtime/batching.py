"""Dynamic micro-batching in front of a :class:`~tracer.runtime.router.Router`.

In an async service, every concurrent ``await router.predict(x)`` triggers its
own ``(1, dim)`` embedding + surrogate forward pass. Under load those calls queue
behind one another and the cheap fast path -- the whole point of TRACER -- turns
into a per-request bottleneck (see issue #27).

:class:`AsyncBatcher` removes that bottleneck the same way high-throughput
inference servers do: it coalesces requests that arrive close together in time
into a *single* vectorized pass. Callers still ``await batcher.predict(x)`` one
input at a time; under the hood, up to ``max_batch_size`` requests (or whatever
arrived within ``max_wait`` seconds) are embedded and routed together.

Routing results are **identical** to :meth:`Router.predict` for the same input --
only throughput changes. Deferred items are handed to the configured
``fallback`` concurrently, off the collector path, so a slow teacher call never
stalls the surrogate fast path.

    from tracer import load_router, AsyncBatcher

    router = load_router(".tracer", embedder=embedder)
    async with AsyncBatcher(router, max_batch_size=64, max_wait=0.005) as batcher:
        outs = await asyncio.gather(*(batcher.predict(t) for t in texts))
        # outs[i] == router.predict(texts[i])
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, List, Optional

from tracer.runtime.router import Router, route_scored, _maybe_await


class _Pending:
    """One queued request: the raw input and the future awaiting its result."""

    __slots__ = ("input", "future")

    def __init__(self, input: Any, future: "asyncio.Future") -> None:
        self.input = input
        self.future = future


# Sentinel pushed onto the queue by aclose() to wake and stop the collector.
_CLOSE = object()


class AsyncBatcher:
    """Coalesce concurrent async predictions into vectorized surrogate passes.

    Parameters
    ----------
    router : a loaded :class:`~tracer.runtime.router.Router`.
    max_batch_size : maximum number of requests fused into one pass.
    max_wait : seconds to wait, after the first queued request, for more requests
        to arrive before flushing. Bounds added latency; ``0`` flushes whatever
        is already queued without waiting. Typical values: 1-10 ms.
    fallback : optional ``fallback(original_input) -> label`` (sync or async)
        invoked for deferred items. When omitted, deferred items resolve with
        ``label=None`` (same as :meth:`Router.predict` with no fallback).
    max_fallback_concurrency : ceiling on in-flight fallback calls across all
        batches (protects a downstream teacher/LLM from overload).
    """

    def __init__(
        self,
        router: Router,
        max_batch_size: int = 64,
        max_wait: float = 0.005,
        fallback: Optional[Callable[[Any], Any]] = None,
        max_fallback_concurrency: int = 16,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if max_wait < 0:
            raise ValueError("max_wait must be >= 0")
        self._router = router
        self._max_batch_size = int(max_batch_size)
        self._max_wait = float(max_wait)
        self._fallback = fallback
        self._fallback_sem = asyncio.Semaphore(max(1, int(max_fallback_concurrency)))
        self._queue: "asyncio.Queue" = asyncio.Queue()
        self._task: Optional["asyncio.Task"] = None
        self._pending: set = set()
        self._closed = False
        # Observability (used by tests and the throughput benchmark).
        self._items_seen = 0
        self._batches_run = 0

    # -- public API ----------------------------------------------------------

    async def predict(self, input: Any) -> dict:
        """Route a single input through the batcher.

        Equivalent in result to ``router.predict(input)`` (and ``router.predict``
        with the batcher's ``fallback``), but the surrogate pass may be shared
        with other concurrent calls. Returns the same dict shape as
        :meth:`Router.predict`.
        """
        if self._closed:
            raise RuntimeError("AsyncBatcher is closed")
        self._ensure_started()
        future = asyncio.get_running_loop().create_future()
        self._items_seen += 1
        await self._queue.put(_Pending(input, future))
        return await future

    async def aclose(self) -> None:
        """Flush queued requests, finish in-flight fallbacks, and stop."""
        if self._closed:
            return
        self._closed = True
        if self._task is not None:
            await self._queue.put(_CLOSE)
            await self._task
            self._task = None
        if self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)

    @property
    def stats(self) -> dict:
        """Throughput counters: requests seen, batches flushed, mean batch size."""
        avg = (self._items_seen / self._batches_run) if self._batches_run else 0.0
        return {
            "items_seen": self._items_seen,
            "batches_run": self._batches_run,
            "avg_batch_size": avg,
        }

    async def __aenter__(self) -> "AsyncBatcher":
        self._ensure_started()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()

    # -- internals -----------------------------------------------------------

    def _ensure_started(self) -> None:
        if self._task is None:
            self._task = asyncio.ensure_future(self._run())

    async def _run(self) -> None:
        """Collect requests into batches and flush them.

        The classic dynamic-batching collector: block for the first request,
        then gather more until the batch is full or ``max_wait`` elapses.
        """
        loop = asyncio.get_running_loop()
        while True:
            first = await self._queue.get()
            if first is _CLOSE:
                return
            batch: List[_Pending] = [first]
            if self._max_wait > 0:
                deadline = loop.time() + self._max_wait
                while len(batch) < self._max_batch_size:
                    timeout = deadline - loop.time()
                    if timeout <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(self._queue.get(), timeout)
                    except asyncio.TimeoutError:
                        break
                    if item is _CLOSE:
                        await self._flush(batch)
                        return
                    batch.append(item)
            else:
                while len(batch) < self._max_batch_size and not self._queue.empty():
                    item = self._queue.get_nowait()
                    if item is _CLOSE:
                        await self._flush(batch)
                        return
                    batch.append(item)
            await self._flush(batch)

    async def _flush(self, batch: List[_Pending]) -> None:
        self._batches_run += 1
        inputs = [p.input for p in batch]
        try:
            X = self._router._to_embeddings(inputs)
            self._router._check_dim(X)
            preds, handled, stage_id, scores = route_scored(self._router._stages, X)
        except Exception as exc:  # embedding / dim / routing error: fail the batch
            for p in batch:
                if not p.future.done():
                    p.future.set_exception(exc)
            return

        deferred: List[tuple] = []
        for i, p in enumerate(batch):
            if handled[i]:
                if not p.future.done():
                    p.future.set_result(
                        self._router._assemble(i, preds, handled, stage_id, scores)
                    )
            else:
                deferred.append((i, p))

        if not deferred:
            return
        if self._fallback is None:
            for _, p in deferred:
                if not p.future.done():
                    p.future.set_result(
                        {"label": None, "decision": "deferred",
                         "accept_score": 0.0, "stage": -1}
                    )
            return

        # Resolve deferred items via the (possibly slow / async) fallback in a
        # detached task so the collector can immediately assemble the next batch
        # instead of blocking the surrogate fast path behind teacher calls.
        task = asyncio.ensure_future(self._resolve_deferred(deferred, inputs))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _resolve_deferred(self, deferred: List[tuple], inputs: List[Any]) -> None:
        async def _one(i: int, p: _Pending) -> None:
            try:
                async with self._fallback_sem:
                    label = await _maybe_await(self._fallback(inputs[i]))
            except Exception as exc:
                if not p.future.done():
                    p.future.set_exception(exc)
                return
            if not p.future.done():
                p.future.set_result(
                    {"label": label, "decision": "deferred",
                     "accept_score": 0.0, "stage": -1}
                )

        await asyncio.gather(*(_one(i, p) for i, p in deferred))
