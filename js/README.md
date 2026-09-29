# @tracer-llm/watch

Local-first, OpenTelemetry GenAI-aligned trace recording for JavaScript / TypeScript.
Classification traces can become training data for a TRACER student:
a small model that predicts fixed labels directly
and leaves other inputs to your teacher. This package records calls only;
fitting and routing use the [Python package or HTTP sidecar](../docs/javascript.md).

Watch any model call in your pipeline with a wrapper, a method decorator, or an
async span. No key, no account, nothing leaves your machine by default: traces
are appended to `./.tracer/watch/<name>.jsonl`.

- GenAI-aligned attributes (`gen_ai.*`) alongside the package's local span
  fields. Validate the payload contract when connecting a collector.
- Zero runtime dependencies. Node stdlib + the global `fetch` only.
- Recording failures are contained. Local writes use synchronous filesystem
  calls and add overhead; HTTP exports are asynchronous and best-effort. There
  is no zero-latency or durable-delivery guarantee.

## Install

```sh
npm install @tracer-llm/watch
```

## Usage

```ts
import { watch } from "@tracer-llm/watch";

const w = watch("support_classifier", { system: "provider-x", model: "model-x" });
```

### 1. Wrap a function

The watcher is callable: pass it a function and it returns a wrapped one. The
first argument is recorded as the input, the return value as the output. If the
return value looks like a provider response (see below) the model, tokens,
finish reason and tool calls are auto-captured.

```ts
const classify = w(async (ticket: string) => {
  const resp = await callModel(ticket); // any provider response object
  return resp;
});

await classify("how do I reset my PIN?");
```

### 2. Method decorator

```ts
class Support {
  @w.llm
  async answer(question: string): Promise<string> {
    return await callModel(question);
  }
}
```

(Requires `"experimentalDecorators": true` in your `tsconfig.json`.)

### 3. Async span

For full control, run a body inside a span and enrich it through the handle:

```ts
await w.span(
  { input: "how do I reset my PIN?", userId: "u1", sessionId: "s1", metadata: { plan: "pro" } },
  async (s) => {
    const resp = await callModel("...");
    s.record(resp);                                   // auto-extract from a response
    s.setOutput("...");                               // or set output text directly
    s.setUsage({ prompt: 5, completion: 2, costUsd: 0.001 });
    s.setParams({ temperature: 0.2, maxTokens: 64 });
    s.addToolCall("get_balance", { acct: 1 }, { bal: 42 });
    s.setAttribute("experiment", "A");
  },
);
```

Spans opened inside another span inherit the parent's `traceId` and point at it
via `parentSpanId`, forming a trace tree (tracked with `AsyncLocalStorage`).

## Sinks

| Sink              | What it does                                                             | Turned on by                                              |
| ----------------- | ----------------------------------------------------------------------- | -------------------------------------------------------- |
| `LocalFileSink`   | Append spans as JSONL to `<dir>/<name>.jsonl`. No network, no key.       | Default unless replaced by a custom sink.                 |
| `OTLPSink`        | POST the package's GenAI-aligned JSON payload to your endpoint.         | `TRACER_WATCH_OTLP_ENDPOINT` (+ `TRACER_WATCH_OTLP_HEADERS`). |
| `MultiSink`       | Fan-out to several sinks at once.                                        | Composed automatically when more than one is configured. |

`sinkFromEnv(name)` composes local files with the generic HTTP sink when
`TRACER_WATCH_OTLP_ENDPOINT` is explicitly configured. Pass `sink` to `watch()`
for a custom exporter. There is no default remote destination.
Despite its historical name, `OTLPSink` does not emit the standard OTLP
`resourceSpans` envelope. It sends one JSON object with `name`, trace/span IDs,
timestamps and `attributes`. Use a compatible endpoint or a translating sink;
arbitrary OTLP collectors are not guaranteed to accept it unchanged.

There is no automatic hosted account, model catalog, credit wallet or cloud
training integration. The local JSONL contains recorded inputs and outputs;
redaction and any sharing policy belong to the application.

## Provider response auto-extraction

`record(resp)` and the function wrapper recognise the two prevailing response
shapes without naming any provider, best-effort and never throwing:

1. `{ model, usage: { prompt_tokens, completion_tokens, total_tokens }, choices: [{ finish_reason, message: { content, tool_calls: [{ function: { name, arguments } }] } }] }`
2. `{ model, usage: { input_tokens, output_tokens }, stop_reason, content: [{ text }] }`

Both objects and plain dicts are handled.

## Environment variables

| Variable                      | Purpose                                                          | Default          |
| ----------------------------- | --------------------------------------------------------------- | ---------------- |
| `TRACER_WATCH_DIR`            | Directory for local JSONL files.                                | `.tracer/watch`  |
| `TRACER_WATCH_OTLP_ENDPOINT`  | Endpoint for the OTLP/HTTP sink.                                | (unset)          |
| `TRACER_WATCH_OTLP_HEADERS`   | Comma-separated `k=v` headers for the OTLP sink.                | (unset)          |
| `TRACER_WATCH_DEBUG`          | Print why a telemetry send was dropped (never affects the host).| (unset)          |

## From recording to a student

Select successful fixed-label classification calls, extract the validated label
from each output, and write `{"input":"...","teacher":"label"}` JSONL.
Provider response text is not automatically parsed into your label schema.
Compute matching embeddings and use Python `tracer.fit()`; inspect the final
teacher-agreement certificate before serving. Keep session metadata for separate
holdouts and include representative traffic, not just deferred requests. General
chat or tool trajectories do not automatically become classification examples.

Teacher agreement is distinct from ground-truth accuracy. No coverage, savings,
or per-request correctness guarantee comes from installing this recorder.

## License

Apache-2.0
