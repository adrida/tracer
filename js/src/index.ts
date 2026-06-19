/**
 * tracer.watch (JS/TS) -- local-first, OpenTelemetry-aligned LLM trace recording.
 *
 * Watch any model call in your pipeline with a wrapper, a method decorator, or
 * an async span. No key, no account, nothing leaves your machine by default:
 * traces are appended to `./.tracer/watch/<name>.jsonl`.
 *
 * Design (standards-native):
 *   - Each watched call is recorded as a span following the OpenTelemetry GenAI
 *     semantic conventions (`gen_ai.*` attributes), an open standard, so a
 *     record is portable to any OTel-aware backend. We do NOT invent a
 *     proprietary schema.
 *   - The same span maps 1:1 to a TRACER trace record (input + the model's
 *     answer), so watched traffic feeds the optimizer directly.
 *   - Exporters (sinks) are pluggable. `LocalFileSink` (default, no key) writes
 *     JSONL. `TracerCloudSink` streams the same spans to the free Tracer Cloud
 *     observability with one key. `OTLPSink` fans the SAME spans out over
 *     OTLP/HTTP to any OTLP/HTTP backend, zero code change.
 *
 * Zero runtime dependencies (Node stdlib only). Prod-safe: telemetry never
 * throws into, or adds latency to, the host call.
 */

import { AsyncLocalStorage } from "node:async_hooks";
import * as fs from "node:fs";
import * as path from "node:path";

export const VERSION = "0.1.0";

// --------------------------------------------------------------------------- //
// The span: OpenTelemetry GenAI semantic-convention shape.
// --------------------------------------------------------------------------- //

export type SpanStatus = "ok" | "error";

/** A recorded tool / function call. */
export interface ToolCall {
  name: string;
  arguments?: unknown;
  result?: unknown;
}

/** GenAI request sampling params (subset of `gen_ai.request.*`). */
export interface RequestParams {
  temperature?: number;
  topP?: number;
  maxTokens?: number;
  frequencyPenalty?: number;
  presencePenalty?: number;
  stop?: string | string[];
  seed?: number;
  [k: string]: unknown;
}

/**
 * One observed GenAI call, in OTel GenAI-semconv terms.
 *
 * Field names mirror the `gen_ai.*` conventions so a record serialises cleanly
 * to OTLP attributes and is recognisable to any OTel-aware backend.
 */
export class GenAISpan {
  /** gen_ai.operation.name (chat | text_completion | embeddings | ...) */
  operationName = "chat";
  /** gen_ai.system / provider (a neutral provider identifier) */
  system?: string;
  /** gen_ai.request.model / gen_ai.response.model */
  requestModel?: string;
  responseModel?: string;
  /** Plain input/output text (convenience; also packed into messages below). */
  inputText = "";
  outputText = "";
  /** gen_ai.usage.input_tokens / output_tokens / total_tokens */
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
  /** gen_ai.response.finish_reasons */
  finishReasons?: string[];
  /** gen_ai.request.* sampling params */
  requestParams: RequestParams = {};
  /** Tool / function calls. */
  toolCalls: ToolCall[] = [];
  /** Span timing / identity. */
  latencyMs?: number;
  /** Streaming: time to first token (ms) when known. */
  ttftMs?: number;
  costUsd?: number;
  status: SpanStatus = "ok";
  error?: string;
  /** ISO-8601 UTC start time. */
  startTime = "";
  traceId = "";
  spanId = "";
  /** Set for nested spans (trace tree). */
  parentSpanId?: string;
  conversationId?: string;
  sessionId?: string;
  userId?: string;
  tags: string[] = [];
  attributes: Record<string, unknown> = {};

  constructor(init?: Partial<GenAISpan>) {
    if (init) Object.assign(this, init);
  }

  /** OTel convention: "<operation> <model>" e.g. "chat model-x". */
  spanName(): string {
    return `${this.operationName} ${this.requestModel || this.system || "llm"}`.trim();
  }

  /** Flatten to `gen_ai.*` OTLP span attributes. */
  toOtelAttributes(): Record<string, unknown> {
    const a: Record<string, unknown> = {
      "gen_ai.operation.name": this.operationName,
    };
    if (this.system) a["gen_ai.system"] = this.system;
    if (this.requestModel) a["gen_ai.request.model"] = this.requestModel;
    if (this.responseModel) a["gen_ai.response.model"] = this.responseModel;
    if (this.inputTokens != null) a["gen_ai.usage.input_tokens"] = this.inputTokens;
    if (this.outputTokens != null) a["gen_ai.usage.output_tokens"] = this.outputTokens;
    if (this.totalTokens != null) a["gen_ai.usage.total_tokens"] = this.totalTokens;
    if (this.finishReasons && this.finishReasons.length) {
      a["gen_ai.response.finish_reasons"] = [...this.finishReasons];
    }
    // gen_ai.request.* sampling params (temperature, top_p, max_tokens, ...).
    for (const [k, v] of Object.entries(this.requestParams)) {
      if (v != null) a[`gen_ai.request.${snake(k)}`] = v;
    }
    if (this.conversationId) a["gen_ai.conversation.id"] = this.conversationId;
    if (this.sessionId) a["session.id"] = this.sessionId;
    if (this.userId) a["enduser.id"] = this.userId;
    if (this.ttftMs != null) a["gen_ai.server.time_to_first_token"] = this.ttftMs;
    // Messages, OTel GenAI shape (role/content parts).
    a["gen_ai.input.messages"] = [
      { role: "user", parts: [{ type: "text", content: this.inputText }] },
    ];
    const parts: Array<Record<string, unknown>> = [
      { type: "text", content: this.outputText },
    ];
    for (const tc of this.toolCalls) {
      const part: Record<string, unknown> = {
        type: "tool_call",
        name: tc.name,
        arguments: tc.arguments,
      };
      if (tc.result !== undefined) part.result = tc.result;
      parts.push(part);
    }
    a["gen_ai.output.messages"] = [
      {
        role: "assistant",
        parts,
        finish_reason: (this.finishReasons && this.finishReasons[0]) || "stop",
      },
    ];
    Object.assign(a, this.attributes);
    return a;
  }

  /** Plain serialisable snapshot (snake_case keys, matching the JSONL schema). */
  toDict(): Record<string, unknown> {
    return {
      operation_name: this.operationName,
      system: this.system ?? null,
      request_model: this.requestModel ?? null,
      response_model: this.responseModel ?? null,
      input_text: this.inputText,
      output_text: this.outputText,
      input_tokens: this.inputTokens ?? null,
      output_tokens: this.outputTokens ?? null,
      total_tokens: this.totalTokens ?? null,
      finish_reasons: this.finishReasons ?? null,
      request_params: this.requestParams,
      tool_calls: this.toolCalls,
      latency_ms: this.latencyMs ?? null,
      ttft_ms: this.ttftMs ?? null,
      cost_usd: this.costUsd ?? null,
      status: this.status,
      error: this.error ?? null,
      start_time: this.startTime,
      trace_id: this.traceId,
      span_id: this.spanId,
      parent_span_id: this.parentSpanId ?? null,
      conversation_id: this.conversationId ?? null,
      session_id: this.sessionId ?? null,
      user_id: this.userId ?? null,
      tags: this.tags,
      attributes: this.attributes,
    };
  }

  /**
   * The watched call as a TRACER trace: input + the model's answer.
   *
   * `teacher_label` = the watched model's output (the thing a surrogate would
   * learn to reproduce). `ground_truth` is carried when known.
   */
  toTraceRecord(): Record<string, unknown> {
    const md: Record<string, unknown> = {
      system: this.system,
      request_model: this.requestModel,
      response_model: this.responseModel,
      latency_ms: this.latencyMs,
      cost_usd: this.costUsd,
      input_tokens: this.inputTokens,
      output_tokens: this.outputTokens,
      total_tokens: this.totalTokens,
      status: this.status,
      ts: this.startTime,
      span_id: this.spanId,
      parent_span_id: this.parentSpanId,
      session_id: this.sessionId,
      user_id: this.userId,
      tags: this.tags,
      ...this.attributes,
    };
    const metadata: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(md)) {
      if (v != null) metadata[k] = v;
    }
    return {
      input_text: this.inputText,
      teacher_label: this.outputText,
      trace_id: this.traceId || null,
      ground_truth: this.attributes["ground_truth"] ?? null,
      metadata,
    };
  }
}

// --------------------------------------------------------------------------- //
// Best-effort response extraction (never throws).
// --------------------------------------------------------------------------- //

/** Read a chain of attribute/key names across object- or dict-shaped values. */
function rd(obj: unknown, ...names: string[]): unknown {
  let cur: unknown = obj;
  for (const n of names) {
    if (cur == null || typeof cur !== "object") return undefined;
    cur = (cur as Record<string, unknown>)[n];
  }
  return cur;
}

function toInt(v: unknown): number | undefined {
  if (typeof v === "number" && Number.isFinite(v)) return Math.trunc(v);
  if (typeof v === "string" && v.trim() !== "" && !Number.isNaN(Number(v))) {
    return Math.trunc(Number(v));
  }
  return undefined;
}

/**
 * Auto-capture model / tokens / finish reason / tool calls / output text from a
 * provider response (object- or dict-shaped). Best-effort and exception-proof:
 * telemetry must never break the host call. Returns true if anything
 * recognisable was extracted.
 *
 * Handles the two prevailing response shapes without naming any provider:
 *   (1) {model, usage:{prompt_tokens, completion_tokens, total_tokens},
 *        choices:[{finish_reason, message:{content, tool_calls:[{function:{name,arguments}}]}}]}
 *   (2) {model, usage:{input_tokens, output_tokens}, stop_reason, content:[{text}]}
 */
export function extractResponse(span: GenAISpan, resp: unknown): boolean {
  try {
    if (resp == null || typeof resp !== "object") return false;
    let hit = false;

    const model = rd(resp, "model");
    if (model) {
      span.responseModel = String(model);
      span.requestModel = span.requestModel || String(model);
      hit = true;
    }

    const usage = rd(resp, "usage");
    if (usage != null && typeof usage === "object") {
      let pt = rd(usage, "prompt_tokens");
      if (pt == null) pt = rd(usage, "input_tokens");
      let ct = rd(usage, "completion_tokens");
      if (ct == null) ct = rd(usage, "output_tokens");
      const tt = rd(usage, "total_tokens");
      const ptN = toInt(pt);
      const ctN = toInt(ct);
      const ttN = toInt(tt);
      if (ptN != null) {
        span.inputTokens = ptN;
        hit = true;
      }
      if (ctN != null) {
        span.outputTokens = ctN;
        hit = true;
      }
      if (ttN != null) {
        span.totalTokens = ttN;
      } else if (ptN != null && ctN != null) {
        span.totalTokens = ptN + ctN;
      }
    }

    // Shape (1): choices[0].message
    const choices = rd(resp, "choices");
    if (Array.isArray(choices) && choices.length) {
      const ch0 = choices[0];
      const fr = rd(ch0, "finish_reason");
      if (fr) {
        span.finishReasons = [String(fr)];
        hit = true;
      }
      const msg = rd(ch0, "message");
      const content = rd(msg, "content");
      if (typeof content === "string" && content) {
        span.outputText = span.outputText || content;
        hit = true;
      }
      const toolCalls = rd(msg, "tool_calls");
      if (Array.isArray(toolCalls)) {
        for (const tc of toolCalls) {
          const fn = rd(tc, "function") ?? tc;
          const name = rd(fn, "name");
          if (name) {
            span.toolCalls.push({
              name: String(name),
              arguments: rd(fn, "arguments"),
            });
            hit = true;
          }
        }
      }
    }

    // Shape (2): top-level content list + stop_reason
    const sr = rd(resp, "stop_reason");
    if (sr) {
      if (!span.finishReasons) span.finishReasons = [String(sr)];
      hit = true;
    }
    const content2 = rd(resp, "content");
    if (Array.isArray(content2) && content2.length && !span.outputText) {
      const txt = rd(content2[0], "text");
      if (typeof txt === "string" && txt) {
        span.outputText = txt;
        hit = true;
      }
    }
    return hit;
  } catch {
    // never let extraction crash the host pipeline
    return false;
  }
}

// --------------------------------------------------------------------------- //
// Sinks (exporters). Default is local-only; OTLP fans out to any OTel backend.
// --------------------------------------------------------------------------- //

export interface Sink {
  emit(span: GenAISpan): void;
  close(): void | Promise<void>;
}

const DEBUG = (): boolean => !!getEnv("TRACER_WATCH_DEBUG");

function debugLog(msg: string): void {
  if (DEBUG()) {
    // eslint-disable-next-line no-console
    console.error(`[tracer.watch] ${msg}`);
  }
}

function getEnv(name: string): string | undefined {
  const g = globalThis as { process?: { env?: Record<string, string | undefined> } };
  return g.process?.env?.[name];
}

/** Append spans as JSONL to `<dir>/<name>.jsonl`. No network, no key. */
export class LocalFileSink implements Sink {
  readonly path: string;
  constructor(name: string, dir: string = ".tracer/watch") {
    this.path = path.join(dir, `${name}.jsonl`);
    try {
      fs.mkdirSync(dir, { recursive: true });
    } catch (e) {
      debugLog(`could not create dir ${dir}: ${String(e)}`);
    }
  }

  emit(span: GenAISpan): void {
    try {
      const line = JSON.stringify(span.toDict());
      fs.appendFileSync(this.path, line + "\n", { encoding: "utf-8" });
    } catch (e) {
      // never let telemetry crash the host pipeline
      debugLog(`local write failed: ${String(e)}`);
    }
  }

  close(): void {
    // nothing buffered
  }
}

/**
 * Fan spans out over OTLP/HTTP to any OTel-compatible backend.
 *
 * Works with any OTLP/HTTP backend -- pass the endpoint + headers for it. Posts
 * an OTel GenAI-shaped payload; batching is left to the caller for simplicity.
 */
export class OTLPSink implements Sink {
  readonly endpoint: string;
  readonly headers: Record<string, string>;
  readonly timeoutMs: number;

  constructor(endpoint: string, headers?: Record<string, string>, timeoutMs = 10000) {
    this.endpoint = endpoint.replace(/\/+$/, "");
    this.headers = { "Content-Type": "application/json", ...(headers || {}) };
    this.timeoutMs = timeoutMs;
  }

  emit(span: GenAISpan): void {
    const payload = {
      name: span.spanName(),
      trace_id: span.traceId,
      span_id: span.spanId,
      start_time: span.startTime,
      latency_ms: span.latencyMs,
      status: span.status,
      attributes: span.toOtelAttributes(),
    };
    // Fire and forget; never block or throw into the host call.
    void postJson(this.endpoint, payload, this.headers, this.timeoutMs).catch((e) =>
      debugLog(`OTLP export failed: ${String(e)}`),
    );
  }

  close(): void {
    // nothing buffered
  }
}

/** Fan-out to several sinks (e.g. local + cloud + your own OTLP backend). */
export class MultiSink implements Sink {
  readonly sinks: Sink[];
  constructor(sinks: Sink[]) {
    this.sinks = [...sinks];
  }

  emit(span: GenAISpan): void {
    for (const s of this.sinks) {
      try {
        s.emit(span);
      } catch (e) {
        debugLog(`sink emit failed: ${String(e)}`);
      }
    }
  }

  async close(): Promise<void> {
    for (const s of this.sinks) {
      try {
        await s.close();
      } catch (e) {
        debugLog(`sink close failed: ${String(e)}`);
      }
    }
  }
}

// Default Tracer Cloud endpoint. Override with TRACER_CLOUD_URL.
const DEFAULT_CLOUD_URL = "https://app.tracerml.ai";
const USER_AGENT = `tracer-watch-js/${VERSION}`;

export interface TracerCloudSinkOptions {
  baseUrl?: string;
  source?: string;
  batchSize?: number;
  flushIntervalMs?: number;
  timeoutMs?: number;
  maxQueue?: number;
}

/**
 * Stream observed spans to Tracer Cloud (free observability).
 *
 * Point it at a Tracer Cloud key and your watched traffic shows up in the
 * dashboard within seconds. No login, no SDK -- just a key (`cloudKey=...` or
 * `TRACER_CLOUD_KEY`).
 *
 * Routes by key type, matching the two product paths:
 *   - `trobs_*` (workspace ingest key) -> `/v1/observe` (tenant-wide)
 *   - otherwise (per-tracer gateway)   -> `/v1/ingest`  (bound to a tracer)
 *
 * Prod-safe: sends are batched on a timer, so a slow or down endpoint never
 * adds latency to (or crashes) the host function. Drops silently on overflow /
 * error; set TRACER_WATCH_DEBUG=1 to see why.
 */
export class TracerCloudSink implements Sink {
  readonly key: string;
  readonly baseUrl: string;
  readonly path: string;
  readonly source: string;
  readonly batchSize: number;
  readonly flushIntervalMs: number;
  readonly timeoutMs: number;
  readonly maxQueue: number;
  /** trobs_ = workspace ingest key -> /v1/observe; otherwise per-tracer -> /v1/ingest */
  readonly observe: boolean;

  private queue: Array<Record<string, unknown>> = [];
  private timer: ReturnType<typeof setInterval> | undefined;
  private closed = false;

  constructor(key: string, opts: TracerCloudSinkOptions = {}) {
    this.key = key;
    this.baseUrl = (opts.baseUrl || getEnv("TRACER_CLOUD_URL") || DEFAULT_CLOUD_URL).replace(
      /\/+$/,
      "",
    );
    this.observe = key.startsWith("trobs_");
    this.path = this.observe ? "/v1/observe" : "/v1/ingest";
    this.source = opts.source ?? "watch";
    this.batchSize = opts.batchSize ?? 25;
    this.flushIntervalMs = opts.flushIntervalMs ?? 2000;
    this.timeoutMs = opts.timeoutMs ?? 10000;
    this.maxQueue = opts.maxQueue ?? 10000;

    this.timer = setInterval(() => {
      void this.flush();
    }, this.flushIntervalMs);
    // Do not keep the event loop alive just for telemetry.
    if (this.timer && typeof (this.timer as { unref?: () => void }).unref === "function") {
      (this.timer as { unref?: () => void }).unref!();
    }
  }

  /** Internal event mapping (exposed for tests). */
  event(span: GenAISpan): Record<string, unknown> {
    if (this.observe) {
      return {
        ts: span.startTime || null,
        system: span.system ?? null,
        model: span.responseModel || span.requestModel || null,
        input: span.inputText,
        output: span.outputText,
        prompt_tokens: span.inputTokens ?? null,
        completion_tokens: span.outputTokens ?? null,
        latency_ms: span.latencyMs ?? null,
        cost_usd: span.costUsd ?? null,
        status: span.status,
        trace_id: span.traceId,
        tags: span.tags,
      };
    }
    // per-tracer /v1/ingest shape
    return {
      input: span.inputText,
      teacher: span.outputText || null,
      output: span.outputText || null,
      model: span.responseModel || span.requestModel || "observed",
      cost_usd: span.costUsd ?? null,
      ts: span.startTime || null,
    };
  }

  emit(span: GenAISpan): void {
    try {
      if (this.queue.length >= this.maxQueue) {
        debugLog("cloud queue full, dropping span");
        return;
      }
      this.queue.push(this.event(span));
    } catch (e) {
      debugLog(`cloud enqueue failed: ${String(e)}`);
    }
  }

  /** POST one batch. Exposed for tests. Never throws. */
  async post(events: Array<Record<string, unknown>>): Promise<void> {
    const body = this.observe ? { events } : { source: this.source, events };
    const headers = {
      "Content-Type": "application/json",
      Authorization: `Bearer ${this.key}`,
      // A real UA: a default UA can trip bot protection (403/1010) at the edge.
      "User-Agent": USER_AGENT,
    };
    try {
      await postJson(`${this.baseUrl}${this.path}`, body, headers, this.timeoutMs);
    } catch (e) {
      // never let telemetry crash or block the host
      debugLog(`cloud export failed: ${String(e)}`);
    }
  }

  /** Drain and POST everything currently queued, in batches. */
  async flush(): Promise<void> {
    while (this.queue.length) {
      const batch = this.queue.splice(0, this.batchSize);
      await this.post(batch);
    }
  }

  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = undefined;
    }
    await this.flush();
  }
}

/**
 * Compose the default sink chain from the environment:
 *   LocalFileSink (always)
 *   + TracerCloudSink (if cloudKey or TRACER_CLOUD_KEY)
 *   + OTLPSink (if TRACER_WATCH_OTLP_ENDPOINT [+ TRACER_WATCH_OTLP_HEADERS])
 */
export function sinkFromEnv(name: string, cloudKey?: string): Sink {
  const dir = getEnv("TRACER_WATCH_DIR") || ".tracer/watch";
  const local = new LocalFileSink(name, dir);
  const sinks: Sink[] = [local];

  const key = cloudKey || getEnv("TRACER_CLOUD_KEY");
  if (key) {
    sinks.push(new TracerCloudSink(key, { source: name }));
  }

  const endpoint = getEnv("TRACER_WATCH_OTLP_ENDPOINT");
  if (endpoint) {
    const headers: Record<string, string> = {};
    const raw = getEnv("TRACER_WATCH_OTLP_HEADERS") || "";
    for (const pair of raw.split(",")) {
      const idx = pair.indexOf("=");
      if (idx > 0) {
        headers[pair.slice(0, idx).trim()] = pair.slice(idx + 1).trim();
      }
    }
    sinks.push(new OTLPSink(endpoint, headers));
  }

  return sinks.length === 1 ? local : new MultiSink(sinks);
}

// --------------------------------------------------------------------------- //
// HTTP helper (uses the global fetch; no runtime deps).
// --------------------------------------------------------------------------- //

async function postJson(
  url: string,
  body: unknown,
  headers: Record<string, string>,
  timeoutMs: number,
): Promise<void> {
  const g = globalThis as { fetch?: typeof fetch; AbortController?: typeof AbortController };
  if (typeof g.fetch !== "function") {
    throw new Error("global fetch is not available");
  }
  let signal: AbortSignal | undefined;
  let timer: ReturnType<typeof setTimeout> | undefined;
  if (typeof g.AbortController === "function") {
    const ctrl = new g.AbortController();
    signal = ctrl.signal;
    timer = setTimeout(() => ctrl.abort(), timeoutMs);
  }
  try {
    const resp = await g.fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal,
    });
    // Drain/ignore the body; do not throw on non-2xx (telemetry is best-effort).
    if (resp && typeof (resp as Response).text === "function") {
      await (resp as Response).text().catch(() => undefined);
    }
  } finally {
    if (timer) clearTimeout(timer);
  }
}

// --------------------------------------------------------------------------- //
// Watcher: the wrapper / decorator / async span. The one object the cloud reuses.
// --------------------------------------------------------------------------- //

/** Tracks the currently-open span so nested calls form a trace tree. */
const ACTIVE = new AsyncLocalStorage<GenAISpan>();

function newId(n = 16): string {
  // Hex id without a crypto dependency; fine for trace/span identity.
  let s = "";
  while (s.length < n) {
    s += Math.floor(Math.random() * 0x100000000)
      .toString(16)
      .padStart(8, "0");
  }
  return s.slice(0, n);
}

function nowIso(): string {
  return new Date().toISOString();
}

function snake(k: string): string {
  return k.replace(/([a-z0-9])([A-Z])/g, "$1_$2").toLowerCase();
}

function defaultExtractInput(args: unknown[]): string {
  if (args.length) {
    const a0 = args[0];
    return typeof a0 === "string" ? a0 : safeJson(a0);
  }
  return "";
}

function defaultExtractOutput(result: unknown): string {
  if (typeof result === "string") return result;
  if (result && typeof result === "object") {
    for (const k of ["label", "intent", "output", "text", "content"]) {
      const v = (result as Record<string, unknown>)[k];
      if (typeof v === "string") return v;
    }
  }
  return safeJson(result);
}

function safeJson(v: unknown): string {
  try {
    return JSON.stringify(v) ?? String(v);
  } catch {
    return String(v);
  }
}

export interface WatchOptions {
  system?: string;
  model?: string;
  operation?: string;
  cloudKey?: string;
  sink?: Sink;
  defaultTags?: string[];
  baseUrl?: string;
  extractInput?: (args: unknown[]) => string;
  extractOutput?: (result: unknown) => string;
}

/** Options for `Watcher.span`. */
export interface SpanOptions {
  input: string;
  userId?: string;
  sessionId?: string;
  conversationId?: string;
  metadata?: Record<string, unknown>;
}

/** The live handle passed to a `Watcher.span` body. */
export interface SpanHandle {
  readonly span: GenAISpan;
  setOutput(output: unknown): SpanHandle;
  record(response: unknown): SpanHandle;
  setUsage(u: {
    prompt?: number;
    completion?: number;
    total?: number;
    costUsd?: number;
  }): SpanHandle;
  setParams(params: RequestParams): SpanHandle;
  addToolCall(name: string, args?: unknown, result?: unknown): SpanHandle;
  setAttribute(key: string, value: unknown): SpanHandle;
}

class SpanHandleImpl implements SpanHandle {
  constructor(
    readonly span: GenAISpan,
    private readonly watcher: WatcherClass,
  ) {}

  setOutput(output: unknown): SpanHandle {
    this.span.outputText = this.watcher.extractOutput(output);
    return this;
  }

  record(response: unknown): SpanHandle {
    extractResponse(this.span, response);
    return this;
  }

  setUsage(u: { prompt?: number; completion?: number; total?: number; costUsd?: number }): SpanHandle {
    if (u.prompt != null) this.span.inputTokens = u.prompt;
    if (u.completion != null) this.span.outputTokens = u.completion;
    if (u.total != null) this.span.totalTokens = u.total;
    else if (u.prompt != null && u.completion != null) {
      this.span.totalTokens = u.prompt + u.completion;
    }
    if (u.costUsd != null) this.span.costUsd = u.costUsd;
    return this;
  }

  setParams(params: RequestParams): SpanHandle {
    for (const [k, v] of Object.entries(params)) {
      if (v != null) this.span.requestParams[k] = v;
    }
    return this;
  }

  addToolCall(name: string, args?: unknown, result?: unknown): SpanHandle {
    const tc: ToolCall = { name, arguments: args };
    if (result !== undefined) tc.result = result;
    this.span.toolCalls.push(tc);
    return this;
  }

  setAttribute(key: string, value: unknown): SpanHandle {
    this.span.attributes[key] = value;
    return this;
  }
}

type AnyFn = (...args: any[]) => any;

/**
 * Records every watched call as a GenAI span.
 *
 * The instance is callable, so it doubles as a function wrapper:
 *   const f = w(async (input) => { ... });   // returns a wrapped fn
 *
 * It also exposes:
 *   - `w.llm`        a method decorator: `@w.llm async answer(q) { ... }`
 *   - `w.span(opts, fn)`  an async span around a body that receives a handle.
 */
class WatcherClass {
  readonly watcherName: string;
  readonly system?: string;
  readonly model?: string;
  readonly operation: string;
  readonly sink: Sink;
  readonly defaultTags: string[];
  readonly extractInput: (args: unknown[]) => string;
  readonly extractOutput: (result: unknown) => string;

  constructor(name: string, opts: WatchOptions = {}) {
    this.watcherName = name;
    this.system = opts.system;
    this.model = opts.model;
    this.operation = opts.operation ?? "chat";
    // Default: local-only. Free Tracer Cloud streaming turns on with a single
    // key (cloudKey or TRACER_CLOUD_KEY). OTLP fan-out via TRACER_WATCH_OTLP_*.
    this.sink = opts.sink ?? this.sinkFromEnvWithBase(name, opts.cloudKey, opts.baseUrl);
    this.defaultTags = [...(opts.defaultTags || [])];
    this.extractInput = opts.extractInput ?? defaultExtractInput;
    this.extractOutput = opts.extractOutput ?? defaultExtractOutput;
  }

  private sinkFromEnvWithBase(name: string, cloudKey?: string, baseUrl?: string): Sink {
    // sinkFromEnv handles the local + OTLP + cloud composition. When a baseUrl
    // is supplied we rebuild the cloud sink with it so opts.baseUrl is honored.
    const base = sinkFromEnv(name, cloudKey);
    if (!baseUrl) return base;
    const rebind = (s: Sink): Sink =>
      s instanceof TracerCloudSink ? new TracerCloudSink(s.key, { source: s.source, baseUrl }) : s;
    if (base instanceof MultiSink) return new MultiSink(base.sinks.map(rebind));
    return rebind(base);
  }

  /** Wrap a (sync or async) function so each call records a span. */
  wrap<F extends AnyFn>(fn: F): F {
    const self = this;
    const wrapped = function (this: unknown, ...args: unknown[]): unknown {
      const span = self.begin(self.extractInput(args));
      return ACTIVE.run(span, () => {
        let result: unknown;
        try {
          result = fn.apply(this, args as any[]);
        } catch (e) {
          self.failAndFinish(span, e);
          throw e;
        }
        if (isPromiseLike(result)) {
          return (result as Promise<unknown>).then(
            (v) => {
              self.capture(span, v);
              self.finish(span);
              return v;
            },
            (e) => {
              self.failAndFinish(span, e);
              throw e;
            },
          );
        }
        self.capture(span, result);
        self.finish(span);
        return result;
      });
    };
    return wrapped as F;
  }

  /**
   * Method decorator (TypeScript experimentalDecorators / "legacy" form):
   *   class C { @w.llm async answer(q: string) { ... } }
   */
  get llm(): MethodDecorator {
    const self = this;
    return ((_target: object, _key: string | symbol, descriptor: PropertyDescriptor) => {
      const original = descriptor.value as AnyFn;
      descriptor.value = self.wrap(original);
      return descriptor;
    }) as MethodDecorator;
  }

  /**
   * Run `fn` inside an async span; `fn` receives a live handle to enrich the
   * span (setOutput / record / setUsage / setParams / addToolCall / setAttribute).
   */
  async span<T>(opts: SpanOptions, fn: (s: SpanHandle) => Promise<T> | T): Promise<T> {
    const span = this.begin(opts.input, {
      userId: opts.userId,
      sessionId: opts.sessionId,
      conversationId: opts.conversationId,
    });
    if (opts.metadata) Object.assign(span.attributes, opts.metadata);
    const handle = new SpanHandleImpl(span, this);
    return ACTIVE.run(span, async () => {
      try {
        const out = await fn(handle);
        this.finish(span);
        return out;
      } catch (e) {
        this.failAndFinish(span, e);
        throw e;
      }
    });
  }

  private begin(
    inputText: string,
    extra?: { userId?: string; sessionId?: string; conversationId?: string },
  ): GenAISpan {
    // Nest under any currently-open span so watched calls form a trace tree.
    const parent = ACTIVE.getStore();
    const span = new GenAISpan({
      operationName: this.operation,
      system: this.system,
      requestModel: this.model,
      responseModel: this.model,
      inputText,
      startTime: nowIso(),
      traceId: parent ? parent.traceId : newId(32),
      spanId: newId(16),
      parentSpanId: parent ? parent.spanId : undefined,
      userId: extra?.userId,
      sessionId: extra?.sessionId,
      conversationId: extra?.conversationId,
      tags: [...this.defaultTags],
    });
    span.attributes["watcher"] = this.watcherName;
    (span as GenAISpan & { _t0?: number })._t0 = nowMs();
    return span;
  }

  /** Auto-capture from a returned provider response, else plain output text. */
  private capture(span: GenAISpan, result: unknown): void {
    extractResponse(span, result);
    if (!span.outputText) span.outputText = this.extractOutput(result);
  }

  private failAndFinish(span: GenAISpan, e: unknown): void {
    span.status = "error";
    span.error = e instanceof Error ? e.message : String(e);
    this.finish(span);
  }

  private finish(span: GenAISpan): void {
    const t0 = (span as GenAISpan & { _t0?: number })._t0;
    if (t0 != null) span.latencyMs = Math.round((nowMs() - t0) * 100) / 100;
    try {
      this.sink.emit(span);
    } catch (e) {
      debugLog(`finish/emit failed: ${String(e)}`);
    }
  }

  async close(): Promise<void> {
    await this.sink.close();
  }
}

/**
 * Build the callable hybrid: a function (so `w(fn)` wraps a function) whose
 * prototype is `WatcherClass`, so all instance methods + `instanceof Watcher`
 * work. We avoid `extends Function` because that makes the reserved `name`
 * property read-only.
 */
function makeWatcher(name: string, opts?: WatchOptions): Watcher {
  const inst = new WatcherClass(name, opts);
  const callable = function (this: unknown, fn: AnyFn): AnyFn {
    return inst.wrap(fn);
  } as unknown as Watcher;
  Object.setPrototypeOf(callable, WatcherClass.prototype);
  // Copy the instance's own (data) properties onto the callable.
  Object.assign(callable, inst);
  return callable;
}

/**
 * A Watcher: callable (so `w(fn)` wraps a function) and carrying the span API
 * (`wrap`, `span`, `llm`, `close`, ...).
 */
export type Watcher = WatcherClass & {
  <F extends AnyFn>(fn: F): F;
};

/**
 * The Watcher class/value. `watch()` returns instances; `x instanceof Watcher`
 * holds for them. `new Watcher(...)` also works (returns a callable hybrid).
 */
export const Watcher = function Watcher(this: unknown, name: string, opts?: WatchOptions): Watcher {
  return makeWatcher(name, opts);
} as unknown as {
  new (name: string, opts?: WatchOptions): Watcher;
  (name: string, opts?: WatchOptions): Watcher;
  prototype: WatcherClass;
};
Watcher.prototype = WatcherClass.prototype;

function nowMs(): number {
  const g = globalThis as { performance?: { now(): number } };
  return g.performance && typeof g.performance.now === "function"
    ? g.performance.now()
    : Date.now();
}

function isPromiseLike(v: unknown): v is PromiseLike<unknown> {
  return !!v && (typeof v === "object" || typeof v === "function") &&
    typeof (v as { then?: unknown }).then === "function";
}

/** Create a Watcher. Use as a function wrapper, a method decorator, or `.span`. */
export function watch(name: string, opts?: WatchOptions): Watcher {
  return makeWatcher(name, opts);
}

export default watch;
