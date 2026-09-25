# `tracer.watch`: observability

Record every LLM call in your pipeline as an OpenTelemetry GenAI span. Local-first
(nothing leaves your machine by default), standards-native (the open `gen_ai.*`
OpenTelemetry GenAI schema), and the same object that feeds
`tracer.fit()` / `tracer.scan()`.

## Quick start

Decorators support both regular functions and `async def`; async spans finish
after the awaited call returns or raises. Recording failures do not replace the
application's result or exception. Set `TRACER_WATCH_DEBUG=1` to diagnose them.
Local watcher names must start with an ASCII letter or digit and contain only
letters, digits, dots, underscores, or hyphens (maximum 128 characters).

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

## Sinks

| Sink | What it does | Turn on with |
|------|--------------|--------------|
| `LocalFileSink` | JSONL to `.tracer/watch/` (default) | always on |
| `OTLPSink` | fan out to any OTLP/HTTP backend | `TRACER_WATCH_OTLP_ENDPOINT` [+ `_HEADERS`] |
| `MultiSink` | several at once (local + your own backend) | set more than one of the above |

Pass a custom `sink=` to `watch()` to fully control export.

## Environment variables

| Var | Effect |
|-----|--------|
| `TRACER_WATCH_DIR` | local JSONL directory (default `.tracer/watch`) |
| `TRACER_WATCH_OTLP_ENDPOINT` | also POST spans to this OTLP/HTTP endpoint |
| `TRACER_WATCH_OTLP_HEADERS` | comma-separated `k=v` headers for the OTLP endpoint |
| `TRACER_WATCH_DEBUG` | print export errors instead of swallowing them |

## JavaScript / TypeScript

The same watcher ships for JS/TS as `@tracer-llm/watch` (zero dependencies, same
capture, local files and opt-in generic export):

```ts
import { watch } from "@tracer-llm/watch";

const w = watch("support-router");                 // local by default

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

See the [`@tracer-llm/watch` package README](../js/README.md) for details.

## From watched traffic to a router

Convert saved spans to the fitting schema before passing them to `tracer.fit()`:

```python
import json
from pathlib import Path
from tracer.watch import GenAISpan

source = Path(".tracer/watch/support_classifier.jsonl")
with Path("traces.jsonl").open("w", encoding="utf-8") as output:
    for line in source.read_text(encoding="utf-8").splitlines():
        span = GenAISpan(**json.loads(line))
        record = span.to_trace_record()
        output.write(json.dumps({
            "input": record.input_text,
            "teacher": record.teacher_label,
        }) + "\n")
```

Use calls whose outputs are classification labels, then compute embeddings for
the same rows and call `tracer.fit("traces.jsonl", embeddings=X)`. See the
[concepts guide](concepts.md).
