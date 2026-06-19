# `tracer.watch`: observability

Record every LLM call in your pipeline as an OpenTelemetry GenAI span. Local-first
(nothing leaves your machine by default), standards-native (the open `gen_ai.*`
OpenTelemetry GenAI schema), and the same object that feeds
`tracer.fit()` / `tracer.scan()`.

## Quick start

```python
import tracer

watch = tracer.watch("support_classifier", system="my-provider", model="my-model")

# (A) decorator: returning the provider's response object auto-captures the
#     model, token counts, finish reason, and any tool calls.
@watch
def classify(ticket: str):
    return llm.chat(model="my-model", messages=[{"role": "user", "content": ticket}],
                    temperature=0.2)

# (B) context manager: full control / non-standard clients.
with watch.span("how do I reset my PIN?", user_id="u_42", session_id="s_1",
                metadata={"plan": "pro"}) as s:
    resp = llm.chat(...)
    s.record(resp)                              # auto-extract from the response
    # …or set fields explicitly:
    s.set_output("here is how…")
    s.set_usage(prompt=120, completion=8, cost_usd=0.0003)
    s.set_params(temperature=0.2, top_p=1, max_tokens=64)
    s.add_tool_call("get_balance", {"acct": 1}, {"bal": 42})
```

Spans nest automatically: a watched call made inside another shares the parent's
trace and links to it, so multi-step pipelines form a trace tree.

By default each call is appended to `.tracer/watch/<name>.jsonl`. No key, no
account, no network.

## What gets captured

Each span follows the OpenTelemetry GenAI conventions (`gen_ai.*`):

- input + output messages (roles, multi-turn), and a flat input/output text
- request + response model, token counts (prompt / completion / total)
- request params (temperature, top_p, max_tokens, stop, seed, …)
- tool / function calls (name, arguments, result)
- finish reason, cost, latency (+ time-to-first-token for streams)
- status / error, timestamps
- trace id, span id, parent span id (nested trace tree)
- conversation / session id, user id, tags, arbitrary metadata

`record(response)` auto-extracts the model, tokens, finish reason, tool calls,
and output text from common provider response objects (object- or dict-shaped);
it never raises, so it is safe on the hot path.

## Stream to Tracer Cloud (free)

Tracer Cloud observability is free. Get a workspace ingest key and point the
watcher at it, your traffic appears in the dashboard within seconds.

Mint a key:

```bash
tracer cloud login            # one-time, opens your browser
tracer cloud ingest-keys create --name "my app"   # prints a trobs_... key
```

Turn it on with **one env var** (no code change):

```bash
export TRACER_CLOUD_KEY=trobs_...
```

…or per-watcher:

```python
watch = tracer.watch("support_classifier", cloud_key="trobs_...")
```

It's **prod-safe**: spans are batched on a background thread, so a slow or down
endpoint never adds latency to (or throws inside) your function. Set
`TRACER_WATCH_DEBUG=1` to surface export errors.

## Sinks

| Sink | What it does | Turn on with |
|------|--------------|--------------|
| `LocalFileSink` | JSONL to `.tracer/watch/` (default) | always on |
| `TracerCloudSink` | stream to Tracer Cloud (free) | `cloud_key=` / `TRACER_CLOUD_KEY` |
| `OTLPSink` | fan out to any OTLP/HTTP backend | `TRACER_WATCH_OTLP_ENDPOINT` [+ `_HEADERS`] |
| `MultiSink` | several at once (local + cloud + OTLP) | set more than one of the above |

Pass a custom `sink=` to `watch()` to fully control export.

## Environment variables

| Var | Effect |
|-----|--------|
| `TRACER_CLOUD_KEY` | stream watched spans to Tracer Cloud (free) |
| `TRACER_CLOUD_URL` | override the Cloud base URL (default `https://app.tracerml.ai`) |
| `TRACER_WATCH_DIR` | local JSONL directory (default `.tracer/watch`) |
| `TRACER_WATCH_OTLP_ENDPOINT` | also POST spans to this OTLP/HTTP endpoint |
| `TRACER_WATCH_OTLP_HEADERS` | comma-separated `k=v` headers for the OTLP endpoint |
| `TRACER_WATCH_DEBUG` | print export errors instead of swallowing them |

## JavaScript / TypeScript

The same watcher ships for JS/TS as `@tracer/watch` (zero dependencies, same
capture, same free Tracer Cloud streaming):

```ts
import { watch } from "@tracer/watch";

const w = watch("support-router");                 // local by default; cloud with one key

// wrap a function
const classify = w(async (ticket: string) =>
  await client.chat.completions.create({ model: "my-model", messages: [...] })
);

// TypeScript method decorator
class Support {
  @w.llm
  async answer(q: string) { return client.chat.completions.create({...}); }
}

// manual span
await w.span({ input, userId: "u_42" }, async (s) => {
  const resp = await client.chat.completions.create({...});
  s.record(resp);
});
```

Set `TRACER_CLOUD_KEY` (or `watch(name, { cloudKey })`) to stream to Tracer
Cloud. See the `@tracer/watch` package README for details.

## From watched traffic to a router

Watched spans map 1:1 to `tracer.types.TraceRecord`, so once you've collected
real traffic you can fit a routing policy straight from it, or let Tracer Cloud
auto-optimize one for you. See the [concepts guide](concepts.md).
