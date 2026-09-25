/**
 * Tests for @tracer-llm/watch: spans, OTel mapping, response extraction (both
 * shapes), opt-in generic export, local-only defaults, nested
 * spans, the wrapper/decorator/span forms. All hermetic: no network, temp dirs,
 * fetch is mocked.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

import {
  GenAISpan,
  LocalFileSink,
  Watcher,
  extractResponse,
  sinkFromEnv,
  watch,
  type Sink,
} from "../src/index.js";

// A capturing sink for assertions.
class ListSink implements Sink {
  out: GenAISpan[] = [];
  emit(span: GenAISpan): void {
    this.out.push(span);
  }
  close(): void {}
}

function tmpDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "tracer-watch-"));
}

// ----- GenAISpan ------------------------------------------------------------- //
describe("GenAISpan", () => {
  it("maps to gen_ai.* OTel attributes", () => {
    const s = new GenAISpan({
      system: "provider-x",
      requestModel: "model-x",
      inputText: "hi",
      outputText: "yo",
      inputTokens: 3,
      outputTokens: 1,
    });
    const a = s.toOtelAttributes();
    expect(a["gen_ai.operation.name"]).toBe("chat");
    expect(a["gen_ai.system"]).toBe("provider-x");
    expect(a["gen_ai.request.model"]).toBe("model-x");
    expect(a["gen_ai.usage.input_tokens"]).toBe(3);
    expect((a["gen_ai.input.messages"] as any)[0].parts[0].content).toBe("hi");
    expect((a["gen_ai.output.messages"] as any)[0].parts[0].content).toBe("yo");
  });

  it("maps output to teacher_label in a trace record", () => {
    const s = new GenAISpan({
      inputText: "ticket",
      outputText: "billing",
      system: "acme",
      costUsd: 0.01,
    });
    const tr = s.toTraceRecord();
    expect(tr.input_text).toBe("ticket");
    expect(tr.teacher_label).toBe("billing");
    expect((tr.metadata as any).cost_usd).toBe(0.01);
  });
});

// ----- LocalFileSink --------------------------------------------------------- //
describe("LocalFileSink", () => {
  it("writes JSONL", () => {
    const dir = tmpDir();
    const sink = new LocalFileSink("unit", dir);
    sink.emit(new GenAISpan({ inputText: "a", outputText: "b" }));
    sink.emit(new GenAISpan({ inputText: "c", outputText: "d" }));
    const lines = fs.readFileSync(path.join(dir, "unit.jsonl"), "utf-8").trim().split("\n");
    expect(lines).toHaveLength(2);
    expect(JSON.parse(lines[0]).input_text).toBe("a");
  });
});

// ----- Watcher: wrapper, decorator, span ------------------------------------- //
describe("Watcher wrapper", () => {
  it("records input + output and latency", async () => {
    const cap = new ListSink();
    const w = new Watcher("dec", { sink: cap });
    const classify = w.wrap(async (t: string) => "label:" + t);
    const out = await classify("hello");
    expect(out).toBe("label:hello");
    expect(cap.out).toHaveLength(1);
    expect(cap.out[0].inputText).toBe("hello");
    expect(cap.out[0].outputText).toBe("label:hello");
    expect(cap.out[0].status).toBe("ok");
    expect(cap.out[0].latencyMs).not.toBeNull();
  });

  it("is callable to wrap: w(fn)", async () => {
    const cap = new ListSink();
    const w = watch("call", { sink: cap });
    const f = w(async (t: string) => "x:" + t);
    expect(await f("y")).toBe("x:y");
    expect(cap.out[0].outputText).toBe("x:y");
  });

  it("records error and rethrows (async)", async () => {
    const cap = new ListSink();
    const w = new Watcher("err", { sink: cap });
    const boom = w.wrap(async () => {
      throw new Error("kaboom");
    });
    await expect(boom()).rejects.toThrow("kaboom");
    expect(cap.out[0].status).toBe("error");
    expect(cap.out[0].error).toContain("kaboom");
  });

  it("records error and rethrows (sync)", () => {
    const cap = new ListSink();
    const w = new Watcher("errsync", { sink: cap });
    const boom = w.wrap(() => {
      throw new Error("sync-boom");
    });
    expect(() => boom()).toThrow("sync-boom");
    expect(cap.out[0].status).toBe("error");
    expect(cap.out[0].error).toContain("sync-boom");
  });

  it("auto-extracts a provider response (shape 1) when wrapped fn returns it", async () => {
    const cap = new ListSink();
    const w = new Watcher("p1", { sink: cap });
    const call = w.wrap(async () => ({
      model: "m-2",
      usage: { prompt_tokens: 10, completion_tokens: 3, total_tokens: 13 },
      choices: [
        {
          finish_reason: "stop",
          message: { content: "hi", tool_calls: [{ function: { name: "lookup", arguments: "{}" } }] },
        },
      ],
    }));
    await call();
    const s = cap.out[0];
    expect(s.outputText).toBe("hi");
    expect([s.inputTokens, s.outputTokens, s.totalTokens]).toEqual([10, 3, 13]);
    expect(s.finishReasons).toEqual(["stop"]);
    expect(s.responseModel).toBe("m-2");
    expect(s.toolCalls.map((t) => t.name)).toEqual(["lookup"]);
  });
});

describe("Watcher.llm decorator", () => {
  it("records a span and rethrows on error", async () => {
    const cap = new ListSink();
    const w = new Watcher("llm", { sink: cap });

    class Svc {
      @w.llm
      async answer(q: string): Promise<string> {
        return "A:" + q;
      }
      @w.llm
      async fail(_q: string): Promise<string> {
        throw new Error("decor-boom");
      }
    }
    const svc = new Svc();
    expect(await svc.answer("q")).toBe("A:q");
    expect(cap.out[0].outputText).toBe("A:q");
    expect(cap.out[0].status).toBe("ok");

    await expect(svc.fail("z")).rejects.toThrow("decor-boom");
    expect(cap.out[1].status).toBe("error");
    expect(cap.out[1].error).toContain("decor-boom");
  });
});

describe("Watcher.span", () => {
  it("setOutput + metadata", async () => {
    const cap = new ListSink();
    const w = new Watcher("ctx", { sink: cap });
    await w.span({ input: "q", metadata: { foo: "bar" } }, async (s) => {
      s.setOutput("answer");
    });
    expect(cap.out[0].outputText).toBe("answer");
    expect(cap.out[0].attributes.foo).toBe("bar");
  });

  it("records exception and rethrows", async () => {
    const cap = new ListSink();
    const w = new Watcher("ctxerr", { sink: cap });
    await expect(
      w.span({ input: "q" }, async () => {
        throw new Error("bad");
      }),
    ).rejects.toThrow("bad");
    expect(cap.out[0].status).toBe("error");
    expect(cap.out[0].error).toContain("bad");
  });

  it("setters surface in OTel attributes + tool calls", async () => {
    const cap = new ListSink();
    const w = new Watcher("set", { sink: cap });
    await w.span({ input: "q", metadata: { plan: "pro" } }, async (s) => {
      s.setParams({ temperature: 0.2, maxTokens: 64, topP: 1 });
      s.setUsage({ prompt: 5, completion: 2, costUsd: 0.001 });
      s.addToolCall("get_balance", { acct: 1 }, { bal: 42 });
    });
    const sp = cap.out[0];
    expect(sp.totalTokens).toBe(7);
    expect(sp.costUsd).toBe(0.001);
    expect(sp.attributes.plan).toBe("pro");
    const a = sp.toOtelAttributes();
    expect(a["gen_ai.request.temperature"]).toBe(0.2);
    expect(a["gen_ai.request.max_tokens"]).toBe(64);
    const parts = (a["gen_ai.output.messages"] as any)[0].parts;
    expect(parts.some((p: any) => p.type === "tool_call" && p.name === "get_balance")).toBe(true);
  });

  it("record() auto-extracts from a provider response", async () => {
    const cap = new ListSink();
    const w = new Watcher("rec", { sink: cap });
    await w.span({ input: "q" }, async (s) => {
      s.record({ model: "m", usage: { input_tokens: 7, output_tokens: 2 }, content: [{ text: "z" }] });
    });
    expect(cap.out[0].outputText).toBe("z");
    expect(cap.out[0].inputTokens).toBe(7);
  });
});

// ----- nested spans ---------------------------------------------------------- //
describe("nested spans", () => {
  it("inherit traceId and set parentSpanId", async () => {
    const cap = new ListSink();
    const w = new Watcher("nest", { sink: cap });
    await w.span({ input: "parent", userId: "u1", sessionId: "s1" }, async () => {
      await w.span({ input: "child" }, async () => {});
    });
    // child finishes (emits) first, then the parent.
    const child = cap.out[0];
    const parent = cap.out[1];
    expect(child.traceId).toBe(parent.traceId);
    expect(child.parentSpanId).toBe(parent.spanId);
    expect(parent.parentSpanId).toBeUndefined();
    expect(parent.userId).toBe("u1");
    expect(parent.sessionId).toBe("s1");
  });
});

// ----- extractResponse ------------------------------------------------------- //
describe("extractResponse", () => {
  it("shape 1: choices/message/tool_calls/usage", () => {
    const s = new GenAISpan();
    const ok = extractResponse(s, {
      model: "m-2",
      usage: { prompt_tokens: 10, completion_tokens: 3, total_tokens: 13 },
      choices: [
        {
          finish_reason: "stop",
          message: { content: "hi", tool_calls: [{ function: { name: "lookup", arguments: "{}" } }] },
        },
      ],
    });
    expect(ok).toBe(true);
    expect(s.outputText).toBe("hi");
    expect([s.inputTokens, s.outputTokens, s.totalTokens]).toEqual([10, 3, 13]);
    expect(s.finishReasons).toEqual(["stop"]);
    expect(s.toolCalls.map((t) => t.name)).toEqual(["lookup"]);
  });

  it("shape 2: content list + stop_reason + input/output tokens", () => {
    const s = new GenAISpan();
    const ok = extractResponse(s, {
      model: "m",
      usage: { input_tokens: 7, output_tokens: 2 },
      stop_reason: "end_turn",
      content: [{ text: "answer" }],
    });
    expect(ok).toBe(true);
    expect(s.inputTokens).toBe(7);
    expect(s.outputTokens).toBe(2);
    expect(s.totalTokens).toBe(9);
    expect(s.finishReasons).toEqual(["end_turn"]);
    expect(s.outputText).toBe("answer");
  });

  it("is exception-proof on junk inputs", () => {
    const s = new GenAISpan();
    expect(extractResponse(s, {})).toBe(false);
    expect(extractResponse(s, null)).toBe(false);
    expect(extractResponse(s, 12345)).toBe(false);
    expect(extractResponse(s, "str")).toBe(false);
  });
});

// ----- sinkFromEnv composition ----------------------------------------------- //
describe("sinkFromEnv", () => {
  const saved: Record<string, string | undefined> = {};
  const keys = ["TRACER_WATCH_DIR", "TRACER_CLOUD_KEY", "TRACER_CLOUD_URL", "TRACER_WATCH_OTLP_ENDPOINT"];
  beforeEach(() => {
    for (const k of keys) saved[k] = process.env[k];
  });
  afterEach(async () => {
    for (const k of keys) {
      if (saved[k] === undefined) delete process.env[k];
      else process.env[k] = saved[k];
    }
  });

  it("local only by default", () => {
    process.env.TRACER_WATCH_DIR = tmpDir();
    delete process.env.TRACER_CLOUD_KEY;
    delete process.env.TRACER_WATCH_OTLP_ENDPOINT;
    const sink = sinkFromEnv("x");
    expect(sink).toBeInstanceOf(LocalFileSink);
  });

  it.each(["trobs_obsolete", "trc_obsolete"])(
    "old app credentials (%s) never enable network export",
    async (key) => {
      const dir = tmpDir();
      process.env.TRACER_WATCH_DIR = dir;
      process.env.TRACER_CLOUD_KEY = key;
      process.env.TRACER_CLOUD_URL = "https://unused.invalid";
      delete process.env.TRACER_WATCH_OTLP_ENDPOINT;
      const fetchMock = vi.fn();
      vi.stubGlobal("fetch", fetchMock);
      try {
        const recorder = watch("local");
        await recorder.span({ input: "ticket" }, async (span) => {
          span.setOutput("billing");
        });
        await recorder.close();
        expect(fetchMock).not.toHaveBeenCalled();
        const saved = JSON.parse(fs.readFileSync(path.join(dir, "local.jsonl"), "utf8"));
        expect(saved.input_text).toBe("ticket");
        expect(saved.output_text).toBe("billing");
      } finally {
        vi.unstubAllGlobals();
      }
    },
  );

  it("exports only to the explicitly configured generic endpoint", async () => {
    const dir = tmpDir();
    process.env.TRACER_WATCH_DIR = dir;
    process.env.TRACER_CLOUD_KEY = "trobs_obsolete";
    process.env.TRACER_WATCH_OTLP_ENDPOINT = "http://collector/v1/traces";
    const fetchMock = vi.fn(async () => ({ text: async () => "{}" }));
    vi.stubGlobal("fetch", fetchMock);
    try {
      const recorder = watch("export");
      await recorder.span({ input: "ticket" }, async (span) => {
        span.setOutput("billing");
      });
      await recorder.close();
      expect(fetchMock).toHaveBeenCalledTimes(1);
      const [url, options] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
      expect(url).toBe("http://collector/v1/traces");
      expect(JSON.parse(options.body as string).attributes["gen_ai.operation.name"]).toBe("chat");
      expect(fs.existsSync(path.join(dir, "export.jsonl"))).toBe(true);
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

// ----- watch factory --------------------------------------------------------- //
describe("watch factory", () => {
  it("returns a Watcher", () => {
    process.env.TRACER_WATCH_DIR = tmpDir();
    delete process.env.TRACER_CLOUD_KEY;
    const w = watch("f", { system: "provider-x", model: "model-x" });
    expect(w).toBeInstanceOf(Watcher);
  });
});
