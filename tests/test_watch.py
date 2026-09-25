"""Unit tests for tracer.watch: spans, sinks, the decorator/context manager,
and opt-in generic export with local-only defaults.
All hermetic: no network, temp dirs only."""
import json

import pytest

from tracer.watch import (
    GenAISpan,
    LocalFileSink,
    OTLPSink,
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


def test_sink_from_env_adds_otlp(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACER_WATCH_DIR", str(tmp_path))
    monkeypatch.delenv("TRACER_CLOUD_KEY", raising=False)
    monkeypatch.setenv("TRACER_WATCH_OTLP_ENDPOINT", "http://collector/v1/traces")
    sink = Watcher._sink_from_env("x")
    assert any(isinstance(s, OTLPSink) for s in sink.sinks)


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


@pytest.mark.parametrize("legacy_key", ["trobs_obsolete", "trc_obsolete"])
def test_legacy_app_configuration_cannot_export(tmp_path, monkeypatch, legacy_key):
    """Upgrades must not keep streaming data because old credentials remain set."""
    import urllib.request
    monkeypatch.setenv("TRACER_WATCH_DIR", str(tmp_path))
    monkeypatch.setenv("TRACER_CLOUD_KEY", legacy_key)
    monkeypatch.setenv("TRACER_CLOUD_URL", "https://unused.invalid")
    monkeypatch.delenv("TRACER_WATCH_OTLP_ENDPOINT", raising=False)
    calls = []
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: calls.append(a))
    recorder = watch("local")
    with recorder.span("ticket") as span:
        span.set_output("billing")
    recorder.sink.close()
    assert calls == []
    saved = json.loads((tmp_path / "local.jsonl").read_text())
    assert saved["input_text"] == "ticket"
    assert saved["output_text"] == "billing"


def test_explicit_otlp_export_still_works(tmp_path, monkeypatch):
    import urllib.request
    monkeypatch.setenv("TRACER_WATCH_DIR", str(tmp_path))
    monkeypatch.setenv("TRACER_WATCH_OTLP_ENDPOINT", "http://collector/v1/traces")
    monkeypatch.setenv("TRACER_WATCH_OTLP_HEADERS", "Authorization=Bearer test")
    monkeypatch.setenv("TRACER_CLOUD_KEY", "trobs_obsolete")
    seen = []
    class Response:
        def read(self): return b"{}"
    def capture(req, **kwargs):
        seen.append(req)
        return Response()
    monkeypatch.setattr(urllib.request, "urlopen", capture)
    recorder = watch("export")
    with recorder.span("ticket") as span:
        span.set_output("billing")
    recorder.sink.close()
    assert len(seen) == 1
    assert seen[0].full_url == "http://collector/v1/traces"
    assert seen[0].get_header("Authorization") == "Bearer test"
    assert json.loads(seen[0].data)["attributes"]["gen_ai.operation.name"] == "chat"
    assert (tmp_path / "export.jsonl").is_file()
