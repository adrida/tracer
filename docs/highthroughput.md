# High-throughput routing

TRACER's value is cost reduction *at scale*, so the prediction path is built to
stay cheap under concurrency. This page covers three additive tools, from
simplest to most powerful:

1. [`predict_batch(..., fallback=...)`](#batch-prediction-with-fallback) — route a
   whole batch in one vectorized pass and fill deferred items with the teacher.
2. [`apredict` / `apredict_batch`](#async-prediction) — `async` prediction with
   concurrent fallback, for `asyncio` apps.
3. [`AsyncBatcher`](#dynamic-micro-batching) — dynamic micro-batching that fuses
   independent concurrent calls into shared surrogate passes.

All three return the same per-item shape as
[`Router.predict()`](api.md#routerpredict) and make **identical routing
decisions** — only throughput changes.

---

## Batch prediction with fallback

`predict_batch` routes every input through the surrogate in a single vectorized
pass. Pass a `fallback` to resolve the deferred items in the same call instead of
getting `label=None` back:

```python
router = tracer.load_router(".tracer", embedder=embedder)

out = router.predict_batch(
    texts,
    fallback=lambda text: call_my_llm(text),   # only invoked for deferred items
)
# out["labels"][i] is the surrogate label when handled, else the teacher label.
```

`fallback(original_input)` receives the original text (when `inputs` is a list)
or the embedding row (when `inputs` is an array). The return value also includes
`accept_scores` (the per-item acceptor score) alongside `labels`, `decisions`,
`handled`, `preds`, and `stage_id`.

If your teacher supports batching, call it once over the deferred slice for
maximum efficiency:

```python
out = router.predict_batch(texts)
deferred = [i for i, d in enumerate(out["decisions"]) if d == "deferred"]
teacher_labels = call_my_llm_batch([texts[i] for i in deferred])   # one batched call
for i, label in zip(deferred, teacher_labels):
    out["labels"][i] = label
```

---

## Async prediction

For `asyncio` services, `apredict` / `apredict_batch` keep the event loop free
while deferred inputs wait on the (async) teacher. The surrogate pass itself is
synchronous and sub-millisecond, so it runs inline; only the `fallback` is
awaited.

```python
# Single input — fallback takes no args, may be sync or async.
out = await router.apredict("What is my balance?", fallback=call_llm_async)

# Batch — one vectorized surrogate pass, deferred items resolved concurrently.
out = await router.apredict_batch(
    texts,
    fallback=call_llm_async,     # fallback(text) -> label, sync or async
    max_concurrency=8,           # cap simultaneous teacher calls
)
```

`apredict_batch` runs the surrogate once over the whole batch and then fans the
deferred items out to `fallback` concurrently, bounded by `max_concurrency` so a
burst of deferrals can't overwhelm a downstream LLM API.

---

## Dynamic micro-batching

In an async server, each concurrent `await router.apredict(x)` triggers its own
`(1, dim)` embedding + surrogate pass. Under load those calls pile up and the
cheap fast path becomes a per-request bottleneck (see
[issue #27](https://github.com/adrida/tracer/issues/27)).

`AsyncBatcher` removes that bottleneck the way high-throughput inference servers
do: it **coalesces requests that arrive close together into a single vectorized
pass**. Callers still `await batcher.predict(x)` one input at a time.

```python
import asyncio
from tracer import load_router, AsyncBatcher

router = load_router(".tracer", embedder=embedder)

async with AsyncBatcher(router, max_batch_size=64, max_wait=0.005) as batcher:
    # 1,000 concurrent requests share ~16 surrogate passes instead of 1,000.
    results = await asyncio.gather(*(batcher.predict(t) for t in texts))
```

In a long-lived server, construct the batcher once and route each request through
it:

```python
batcher = AsyncBatcher(router, fallback=call_llm_async)

async def handle(request):
    return await batcher.predict(request.text)   # joins the current micro-batch

# on shutdown:
await batcher.aclose()
```

**Parameters**

| Name | Default | Meaning |
|------|---------|---------|
| `max_batch_size` | `64` | Most requests fused into one pass. |
| `max_wait` | `0.005` | Seconds to wait (after the first queued request) for more to arrive before flushing. Bounds added latency. `0` flushes whatever is already queued. |
| `fallback` | `None` | `fallback(original_input) -> label` (sync or async) for deferred items. |
| `max_fallback_concurrency` | `16` | Ceiling on in-flight teacher calls across all batches. |

Deferred items are resolved off the collector path, so a slow teacher call never
stalls the surrogate fast path. `batcher.stats` exposes `items_seen`,
`batches_run`, and `avg_batch_size` for monitoring.

### Throughput

`benchmarks/batching_throughput.py` measures the effect. With 1,000 concurrent
requests and `max_batch_size=64`:

```
forward passes     sequential=1000   batched=16   (62.5x fewer)
```

When each forward pass is latency-bound (e.g. a remote embedding API at ~2 ms),
that structural reduction becomes a wall-clock win:

```
wall clock         sequential=3112 ms   batched=60 ms   (51.5x faster)
```

```bash
python benchmarks/batching_throughput.py --requests 1000 --max-batch 64 --work-ms 2
```

---

## Concurrent serving

`tracer serve` uses a threaded HTTP server, so multiple requests are handled
concurrently rather than queued behind one another. Prediction is read-only on
the loaded model, so the shared router is safe across threads.

```bash
tracer serve .tracer --port 8000
```

See the [CLI reference](cli.md) for the server's endpoints.
