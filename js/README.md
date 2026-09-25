# @tracer-llm/watch

Local-first, OpenTelemetry-aligned LLM trace recording for JavaScript / TypeScript.

Watch any model call in your pipeline with a wrapper, a method decorator, or an
async span. No key, no account, nothing leaves your machine by default: traces
are appended to `./.tracer/watch/<name>.jsonl`.

- Standards-native schema: every watched call is recorded as a span following
  the OpenTelemetry GenAI semantic conventions (`gen_ai.*`), so a record is
  portable to any OTel-aware backend. No proprietary schema.
- Zero runtime dependencies. Node stdlib + the global `fetch` only.
- Prod-safe: telemetry never throws into, or adds latency to, the host call.
  Generic HTTP exports are sent asynchronously; export failures are swallowed.

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
| `LocalFileSink`   | Append spans as JSONL to `<dir>/<name>.jsonl`. No network, no key.       | Default, always on.                                       |
| `OTLPSink`        | POST an OTel GenAI-shaped payload to any OTLP/HTTP backend.              | `TRACER_WATCH_OTLP_ENDPOINT` (+ `TRACER_WATCH_OTLP_HEADERS`). |
| `MultiSink`       | Fan-out to several sinks at once.                                        | Composed automatically when more than one is configured. |

`sinkFromEnv(name)` composes local files with the generic HTTP sink when
`TRACER_WATCH_OTLP_ENDPOINT` is explicitly configured. Pass `sink` to `watch()`
for a custom exporter. There is no default remote destination.

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

## License

Apache-2.0
