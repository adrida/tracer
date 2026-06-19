"""Unit tests for tracer.watch: spans, sinks, the decorator/context manager,
env-driven sink composition, and the Tracer Cloud sink (mapping + prod-safety).
All hermetic: no network, temp dirs only."""
import json
import os

import pytest

from tracer.watch import (
    GenAISpan,
    LocalFileSink,
    MultiSink,
    OTLPSink,
    TracerCloudSink,
    Watcher,
    watch,
)


# ----- GenAISpan ------------------------------------------------------------- #
def test_span_to_otel_attributes_genai_shape():
    s = GenAISpan(system="openai", request_model="gpt-4o", input_text="hi", output_text="yo",
                  input_tokens=3, output_tokens=1)
    a = s.to_otel_attributes()
    assert a["gen_ai.operation.name"] == "chat"
    assert a["gen_ai.system"] == "openai"
    assert a["gen_ai.request.model"] == "gpt-4o"
    assert a["gen_ai.usage.input_tokens"] == 3
    assert a["gen_ai.input.messages"][0]["parts"][0]["content"] == "hi"
    assert a["gen_ai.output.messages"][0]["parts"][0]["content"] == "yo"


def test_span_to_trace_record_maps_output_to_teacher():
    s = GenAISpan(input_text="ticket", output_text="billing", system="acme", cost_usd=0.01)
    tr = s.to_trace_record()
    assert tr.input_text == "ticket"
    assert tr.teacher_label == "billing"
    assert tr.metadata["cost_usd"] == 0.01


# ----- LocalFileSink --------------------------------------------------------- #
def test_local_file_sink_writes_jsonl(tmp_path):
    sink = LocalFileSink("unit", dir=str(tmp_path))
    sink.emit(GenAISpan(input_text="a", output_text="b"))
    sink.emit(GenAISpan(input_text="c", output_text="d"))
    lines = (tmp_path / "unit.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["input_text"] == "a"


# ----- Watcher decorator + context manager ----------------------------------- #
def test_decorator_records_input_output(tmp_path):
    captured = []
    w = Watcher("dec", sink=_ListSink(captured))
    @w
    def classify(t):
        return "label:" + t
    out = classify("hello")
    assert out == "label:hello"
    assert len(captured) == 1
    assert captured[0].input_text == "hello"
    assert captured[0].output_text == "label:hello"
    assert captured[0].status == "ok"
    assert captured[0].latency_ms is not None


def test_decorator_records_error_and_reraises(tmp_path):
    captured = []
    w = Watcher("err", sink=_ListSink(captured))
    @w
    def boom(_t):
        raise ValueError("kaboom")
    with pytest.raises(ValueError):
        boom("x")
    assert captured[0].status == "error"
    assert "kaboom" in captured[0].error


def test_context_manager_set_output(tmp_path):
    captured = []
    w = Watcher("ctx", sink=_ListSink(captured))
    with w.span("q", foo="bar") as s:
        s.set_output("answer")
    assert captured[0].output_text == "answer"
    assert captured[0].attributes["foo"] == "bar"


def test_context_manager_records_exception(tmp_path):
    captured = []
    w = Watcher("ctxerr", sink=_ListSink(captured))
    with pytest.raises(RuntimeError):
        with w.span("q"):
            raise RuntimeError("bad")
    assert captured[0].status == "error"
    assert "bad" in captured[0].error


# ----- _sink_from_env composition -------------------------------------------- #
def test_sink_from_env_local_only(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACER_WATCH_DIR", str(tmp_path))
    monkeypatch.delenv("TRACER_CLOUD_KEY", raising=False)
    monkeypatch.delenv("TRACER_WATCH_OTLP_ENDPOINT", raising=False)
    sink = Watcher._sink_from_env("x")
    assert isinstance(sink, LocalFileSink)


def test_sink_from_env_adds_cloud_when_key(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACER_WATCH_DIR", str(tmp_path))
    monkeypatch.setenv("TRACER_CLOUD_KEY", "trobs_abc")
    monkeypatch.delenv("TRACER_WATCH_OTLP_ENDPOINT", raising=False)
    sink = Watcher._sink_from_env("x")
    assert isinstance(sink, MultiSink)
    kinds = {type(s).__name__ for s in sink.sinks}
    assert "LocalFileSink" in kinds and "TracerCloudSink" in kinds
    for s in sink.sinks:
        if isinstance(s, TracerCloudSink):
            s.close()


def test_cloud_key_kwarg_beats_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACER_WATCH_DIR", str(tmp_path))
    monkeypatch.delenv("TRACER_CLOUD_KEY", raising=False)
    sink = Watcher._sink_from_env("x", cloud_key="trc_xyz")
    assert isinstance(sink, MultiSink)
    cloud = [s for s in sink.sinks if isinstance(s, TracerCloudSink)]
    assert cloud and cloud[0]._observe is False  # trc_ -> per-tracer /v1/ingest
    cloud[0].close()


def test_sink_from_env_adds_otlp(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACER_WATCH_DIR", str(tmp_path))
    monkeypatch.delenv("TRACER_CLOUD_KEY", raising=False)
    monkeypatch.setenv("TRACER_WATCH_OTLP_ENDPOINT", "http://collector/v1/traces")
    sink = Watcher._sink_from_env("x")
    assert any(isinstance(s, OTLPSink) for s in sink.sinks)


# ----- TracerCloudSink: routing + mapping + prod-safety ----------------------- #
def test_cloud_sink_routes_by_key_prefix():
    obs = TracerCloudSink("trobs_k", base_url="http://x"); obs.close()
    ing = TracerCloudSink("trc_k", base_url="http://x"); ing.close()
    assert obs.path == "/v1/observe" and obs._observe is True
    assert ing.path == "/v1/ingest" and ing._observe is False


def test_cloud_sink_observe_event_shape():
    s = TracerCloudSink("trobs_k", base_url="http://x"); s.close()
    span = GenAISpan(system="openai", request_model="gpt-4o", response_model="gpt-4o",
                     input_text="in", output_text="out", input_tokens=5, output_tokens=2,
                     cost_usd=0.001, status="ok", trace_id="t1")
    e = s._event(span)
    assert e["input"] == "in" and e["output"] == "out"
    assert e["model"] == "gpt-4o"
    assert e["prompt_tokens"] == 5 and e["completion_tokens"] == 2
    assert e["cost_usd"] == 0.001


def test_cloud_sink_ingest_event_shape():
    s = TracerCloudSink("trc_k", base_url="http://x"); s.close()
    span = GenAISpan(request_model="gpt-4o", input_text="in", output_text="out", cost_usd=0.002)
    e = s._event(span)
    assert e["input"] == "in"
    assert e["teacher"] == "out" and e["output"] == "out"
    assert e["model"] == "gpt-4o"


def test_cloud_sink_post_builds_request(monkeypatch):
    import urllib.request
    seen = {}
    class FakeResp:
        def read(self): return b"{}"
        def __enter__(self): return self
        def __exit__(self, *a): return False
    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["headers"] = {k.lower(): v for k, v in req.header_items()}
        seen["body"] = req.data
        return FakeResp()
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    s = TracerCloudSink("trobs_k", base_url="http://host", source="app"); s.close()
    s._post([{"input": "x"}])
    assert seen["url"] == "http://host/v1/observe"
    assert seen["headers"]["authorization"] == "Bearer trobs_k"
    assert "tracer-watch" in seen["headers"]["user-agent"].lower()
    assert json.loads(seen["body"]) == {"events": [{"input": "x"}]}


def test_cloud_sink_post_swallows_errors(monkeypatch):
    import urllib.request
    def boom(req, timeout=None):
        raise OSError("network down")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    s = TracerCloudSink("trobs_k", base_url="http://host"); s.close()
    # Must NOT raise, telemetry can never crash the host.
    s._post([{"input": "x"}])


def test_cloud_sink_emit_flush_via_close(monkeypatch):
    import urllib.request
    posted = []
    class FakeResp:
        def read(self): return b"{}"
    def fake_urlopen(req, timeout=None):
        posted.append(json.loads(req.data))
        return FakeResp()
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    s = TracerCloudSink("trobs_k", base_url="http://host", flush_interval=0.05)
    s.emit(GenAISpan(input_text="a", output_text="b"))
    s.emit(GenAISpan(input_text="c", output_text="d"))
    s.close()  # flushes the queue
    flat = [ev for batch in posted for ev in batch["events"]]
    assert len(flat) == 2
    assert {e["input"] for e in flat} == {"a", "c"}


def test_watch_factory_returns_watcher(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACER_WATCH_DIR", str(tmp_path))
    monkeypatch.delenv("TRACER_CLOUD_KEY", raising=False)
    w = watch("f", system="openai", model="gpt-4o")
    assert isinstance(w, Watcher)


# ----- full-parity capture --------------------------------------------------- #
from tracer.watch import extract_response


def test_decorator_auto_extracts_response_shape_1():
    cap = []
    w = Watcher("p1", sink=_ListSink(cap))
    @w
    def call(_q):
        return {"model": "m-2",
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
                "choices": [{"finish_reason": "stop",
                             "message": {"content": "hi", "tool_calls": [{"function": {"name": "lookup", "arguments": "{}"}}]}}]}
    call("q")
    s = cap[0]
    assert s.output_text == "hi"
    assert (s.input_tokens, s.output_tokens, s.total_tokens) == (10, 3, 13)
    assert s.finish_reasons == ["stop"]
    assert s.response_model == "m-2"
    assert [t["name"] for t in s.tool_calls] == ["lookup"]


def test_extract_response_shape_2():
    s = GenAISpan()
    ok = extract_response(s, {"model": "m", "usage": {"input_tokens": 7, "output_tokens": 2},
                              "stop_reason": "end_turn", "content": [{"text": "answer"}]})
    assert ok
    assert s.input_tokens == 7 and s.output_tokens == 2 and s.total_tokens == 9
    assert s.finish_reasons == ["end_turn"]
    assert s.output_text == "answer"


def test_extract_response_is_exception_proof():
    s = GenAISpan()
    assert extract_response(s, object()) is False
    assert extract_response(s, None) is False
    assert extract_response(s, 12345) is False  # never raises


def test_nested_spans_form_trace_tree():
    cap = []
    w = Watcher("nest", sink=_ListSink(cap))
    with w.span("parent", user_id="u1", session_id="s1"):
        with w.span("child"):
            pass
    child, parent = cap[0], cap[1]  # child finishes (emits) first
    assert child.trace_id == parent.trace_id
    assert child.parent_span_id == parent.span_id
    assert parent.parent_span_id is None
    assert parent.user_id == "u1" and parent.session_id == "s1"


def test_span_setters_and_otel_params():
    cap = []
    w = Watcher("set", sink=_ListSink(cap))
    with w.span("q", metadata={"plan": "pro"}) as s:
        s.set_params(temperature=0.2, max_tokens=64, top_p=1)
        s.set_usage(prompt=5, completion=2, cost_usd=0.001)
        s.add_tool_call("get_balance", {"acct": 1}, {"bal": 42})
    sp = cap[0]
    assert sp.total_tokens == 7 and sp.cost_usd == 0.001
    assert sp.attributes["plan"] == "pro"
    a = sp.to_otel_attributes()
    assert a["gen_ai.request.temperature"] == 0.2
    assert a["gen_ai.request.max_tokens"] == 64
    # tool call surfaced in the output messages
    parts = a["gen_ai.output.messages"][0]["parts"]
    assert any(p.get("type") == "tool_call" and p.get("name") == "get_balance" for p in parts)


class _ListSink:
    def __init__(self, out): self.out = out
    def emit(self, span): self.out.append(span)
    def close(self): pass
