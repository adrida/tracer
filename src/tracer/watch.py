"""tracer.watch -- local-first, OpenTelemetry-aligned LLM trace recording.

Watch any LLM call in your pipeline with a decorator or context manager. No API
key, no account, nothing leaves your machine by default: traces are appended to
``./.tracer/watch/<name>.jsonl``.

Design (SOTA / standards-native):
  * Each watched call is recorded as a span following the OpenTelemetry GenAI
    semantic conventions (``gen_ai.*`` attributes), an open standard, so a
    record is portable to any OTel-aware backend. We do NOT invent a
    proprietary schema.
  * The same span maps 1:1 to :class:`tracer.types.TraceRecord`, so watched
    traffic feeds ``tracer.fit()`` / ``tracer.scan()`` directly -- one object
    end to end.
  * Exporters are pluggable. ``LocalFileSink`` (default, no key) writes JSONL.
    ``OTLPSink`` fans the SAME spans out over OTLP to anything else that speaks
    it (any OTLP/HTTP backend), zero code change.

Core has zero dependencies (stdlib only). Local recording works out of the box;
network export requires an explicitly configured endpoint or custom sink.
"""
from __future__ import annotations

import contextvars
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from tracer.types import TraceRecord

__all__ = [
    "GenAISpan",
    "Sink",
    "LocalFileSink",
    "OTLPSink",
    "MultiSink",
    "Watcher",
    "watch",
    "extract_response",
]


# --------------------------------------------------------------------------- #
# The span: OpenTelemetry GenAI semantic-convention shape.
# --------------------------------------------------------------------------- #
@dataclass
class GenAISpan:
    """One observed GenAI call, in OTel GenAI-semconv terms.

    Field names mirror the ``gen_ai.*`` conventions so a record serialises
    cleanly to OTLP attributes and is recognisable to any OTel-aware backend.
    """

    # gen_ai.operation.name (chat | text_completion | embeddings | ...)
    operation_name: str = "chat"
    # gen_ai.system / provider (free-form vendor/provider id)
    system: Optional[str] = None
    # gen_ai.request.model / gen_ai.response.model
    request_model: Optional[str] = None
    response_model: Optional[str] = None
    # Plain input/output text (convenience; also packed into messages below).
    input_text: str = ""
    output_text: str = ""
    # gen_ai.usage.input_tokens / output_tokens
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    # gen_ai.response.finish_reasons
    finish_reasons: Optional[List[str]] = None
    # span timing / identity
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None
    status: str = "ok"            # ok | error
    error: Optional[str] = None
    start_time: str = ""          # ISO-8601 UTC
    trace_id: str = ""
    span_id: str = ""
    parent_span_id: Optional[str] = None   # set for nested spans (trace tree)
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    # gen_ai.usage.total_tokens + gen_ai.request.* sampling params
    total_tokens: Optional[int] = None
    request_params: Dict[str, Any] = field(default_factory=dict)
    # Tool / function calls: [{"name", "arguments", "result"?}]
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    # Streaming: time to first token (ms) when known.
    ttft_ms: Optional[float] = None
    streaming: bool = False
    # gen_ai.* free-form + tags
    tags: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def span_name(self) -> str:
        # OTel convention: "<operation> <model>" e.g. "chat my-model".
        return f"{self.operation_name} {self.request_model or self.system or 'llm'}".strip()

    def to_otel_attributes(self) -> Dict[str, Any]:
        """Flatten to ``gen_ai.*`` OTLP span attributes."""
        a: Dict[str, Any] = {
            "gen_ai.operation.name": self.operation_name,
        }
        if self.system:
            a["gen_ai.system"] = self.system
        if self.request_model:
            a["gen_ai.request.model"] = self.request_model
        if self.response_model:
            a["gen_ai.response.model"] = self.response_model
        if self.input_tokens is not None:
            a["gen_ai.usage.input_tokens"] = self.input_tokens
        if self.output_tokens is not None:
            a["gen_ai.usage.output_tokens"] = self.output_tokens
        if self.total_tokens is not None:
            a["gen_ai.usage.total_tokens"] = self.total_tokens
        if self.finish_reasons:
            a["gen_ai.response.finish_reasons"] = list(self.finish_reasons)
        # gen_ai.request.* sampling params (temperature, top_p, max_tokens, ...).
        for k, v in self.request_params.items():
            if v is not None:
                a[f"gen_ai.request.{k}"] = v
        if self.conversation_id:
            a["gen_ai.conversation.id"] = self.conversation_id
        if self.session_id:
            a["session.id"] = self.session_id
        if self.user_id:
            a["enduser.id"] = self.user_id
        if self.ttft_ms is not None:
            a["gen_ai.server.time_to_first_token"] = self.ttft_ms
        # Messages, OTel GenAI shape (role/content parts).
        a["gen_ai.input.messages"] = [
            {"role": "user", "parts": [{"type": "text", "content": self.input_text}]}
        ]
        parts: List[Dict[str, Any]] = [{"type": "text", "content": self.output_text}]
        for tc in self.tool_calls:
            parts.append({
                "type": "tool_call",
                "name": tc.get("name"),
                "arguments": tc.get("arguments"),
                **({"result": tc["result"]} if "result" in tc else {}),
            })
        a["gen_ai.output.messages"] = [
            {
                "role": "assistant",
                "parts": parts,
                "finish_reason": (self.finish_reasons or ["stop"])[0],
            }
        ]
        a.update(self.attributes)
        return a

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_trace_record(self) -> TraceRecord:
        """The watched call as a TRACER trace: input + the model's answer.

        ``teacher_label`` = the watched model's output (the thing a surrogate
        would learn to reproduce). ``ground_truth`` is carried when known.
        """
        md: Dict[str, Any] = {
            "system": self.system,
            "request_model": self.request_model,
            "response_model": self.response_model,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "status": self.status,
            "ts": self.start_time,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "tags": list(self.tags),
            **self.attributes,
        }
        return TraceRecord(
            input_text=self.input_text,
            teacher_label=self.output_text,
            trace_id=self.trace_id or None,
            ground_truth=self.attributes.get("ground_truth"),
            metadata={k: v for k, v in md.items() if v is not None},
        )


# --------------------------------------------------------------------------- #
# Sinks (exporters). Default is local-only; OTLP fans out to any OTel backend.
# --------------------------------------------------------------------------- #
class Sink(Protocol):
    def emit(self, span: GenAISpan) -> None: ...
    def close(self) -> None: ...


class LocalFileSink:
    """Append spans as JSONL to ``<dir>/<name>.jsonl``. No network, no key."""

    def __init__(self, name: str, dir: str = ".tracer/watch") -> None:
        self.path = os.path.join(dir, f"{name}.jsonl")
        os.makedirs(dir, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, span: GenAISpan) -> None:
        line = json.dumps(span.to_dict(), ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def close(self) -> None:  # nothing buffered
        return None


class OTLPSink:
    """Fan spans out over OTLP/HTTP to any OTel-compatible backend.

    Works with any OTLP/HTTP backend -- pass the endpoint + headers for it. Posts an
    OTel GenAI-shaped payload; batching is left to the caller for simplicity.
    """

    def __init__(self, endpoint: str, headers: Optional[Dict[str, str]] = None,
                 timeout: float = 10.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.timeout = timeout

    def emit(self, span: GenAISpan) -> None:
        import urllib.request

        payload = {
            "name": span.span_name(),
            "trace_id": span.trace_id,
            "span_id": span.span_id,
            "start_time": span.start_time,
            "latency_ms": span.latency_ms,
            "status": span.status,
            "attributes": span.to_otel_attributes(),
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.endpoint, data=data, headers=self.headers, method="POST")
        try:
            urllib.request.urlopen(req, timeout=self.timeout).read()
        except Exception as e:  # never let telemetry crash the host pipeline
            if os.environ.get("TRACER_WATCH_DEBUG"):
                print(f"[tracer.watch] OTLP export failed: {e}")

    def close(self) -> None:
        return None


class MultiSink:
    """Fan-out to several sinks (e.g. local + your own OTLP backend)."""

    def __init__(self, sinks: Sequence[Sink]) -> None:
        self.sinks = list(sinks)

    def emit(self, span: GenAISpan) -> None:
        for s in self.sinks:
            s.emit(span)

    def close(self) -> None:
        for s in self.sinks:
            s.close()


# Tracks the currently-open span so nested watched calls form a trace tree
# (a child span inherits the parent's trace_id and points at it via
# parent_span_id). contextvars makes this correct under threads + async.
_ACTIVE: "contextvars.ContextVar[Optional[GenAISpan]]" = contextvars.ContextVar(
    "tracer_watch_active_span", default=None
)


def _g(obj: Any, *names: str) -> Any:
    """Best-effort attribute/key read across object- and dict-shaped responses."""
    for n in names:
        if obj is None:
            return None
        if isinstance(obj, dict):
            obj = obj.get(n)
        else:
            obj = getattr(obj, n, None)
    return obj


def extract_response(span: GenAISpan, resp: Any) -> bool:
    """Auto-capture model / tokens / finish reason / tool calls / output text
    from a provider response object (object- or dict-shaped). Best-effort and
    exception-proof: telemetry must never break the host call. Returns True if
    anything recognisable was extracted.

    Handles the two prevailing response shapes without naming any provider:
      (1) {model, usage:{prompt_tokens, completion_tokens, total_tokens},
           choices:[{finish_reason, message:{content, tool_calls:[...]}}]}
      (2) {model, usage:{input_tokens, output_tokens}, stop_reason,
           content:[{text}]}
    """
    try:
        hit = False
        model = _g(resp, "model")
        if model:
            span.response_model = str(model)
            span.request_model = span.request_model or str(model)
            hit = True

        usage = _g(resp, "usage")
        if usage is not None:
            pt = _g(usage, "prompt_tokens")
            if pt is None:
                pt = _g(usage, "input_tokens")
            ct = _g(usage, "completion_tokens")
            if ct is None:
                ct = _g(usage, "output_tokens")
            tt = _g(usage, "total_tokens")
            if pt is not None:
                span.input_tokens = int(pt); hit = True
            if ct is not None:
                span.output_tokens = int(ct); hit = True
            if tt is not None:
                span.total_tokens = int(tt)
            elif pt is not None and ct is not None:
                span.total_tokens = int(pt) + int(ct)

        # Shape (1): choices[0].message
        choices = _g(resp, "choices")
        if isinstance(choices, (list, tuple)) and choices:
            ch0 = choices[0]
            fr = _g(ch0, "finish_reason")
            if fr:
                span.finish_reasons = [str(fr)]; hit = True
            msg = _g(ch0, "message")
            content = _g(msg, "content")
            if isinstance(content, str) and content:
                span.output_text = span.output_text or content; hit = True
            for tc in (_g(msg, "tool_calls") or []):
                fn = _g(tc, "function") or tc
                name = _g(fn, "name")
                if name:
                    span.tool_calls.append({"name": str(name), "arguments": _g(fn, "arguments")})
                    hit = True

        # Shape (2): top-level content list + stop_reason
        sr = _g(resp, "stop_reason")
        if sr:
            span.finish_reasons = span.finish_reasons or [str(sr)]; hit = True
        content2 = _g(resp, "content")
        if isinstance(content2, (list, tuple)) and content2 and not span.output_text:
            txt = _g(content2[0], "text")
            if isinstance(txt, str) and txt:
                span.output_text = txt; hit = True
        return hit
    except Exception:  # never let extraction crash the host pipeline
        return False


# --------------------------------------------------------------------------- #
# Watcher: the decorator / context manager. Records calls to local or explicitly configured sinks.
# --------------------------------------------------------------------------- #
def _new_id(n: int = 16) -> str:
    return uuid.uuid4().hex[:n]


def _default_extract_input(args: tuple, kwargs: dict) -> str:
    if args:
        return args[0] if isinstance(args[0], str) else json.dumps(args[0], default=str)
    if kwargs:
        v = next(iter(kwargs.values()))
        return v if isinstance(v, str) else json.dumps(v, default=str)
    return ""


def _default_extract_output(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for k in ("label", "intent", "output", "text", "content"):
            if k in result and isinstance(result[k], str):
                return result[k]
    return json.dumps(result, default=str)


class Watcher:
    """Records every wrapped call as a GenAI span.

    Use as a decorator::

        watch = tracer.watch("support_classifier", system="acme", model="my-model")

        @watch
        def classify(ticket: str) -> str:
            return my_llm(ticket)

    or as a context manager::

        with watch.span("how do I reset my PIN?") as s:
            ans = my_llm(...)
            s.set_output(ans)
    """

    def __init__(
        self,
        name: str,
        *,
        system: Optional[str] = None,
        model: Optional[str] = None,
        operation: str = "chat",
        sink: Optional[Sink] = None,
        extract_input: Callable[[tuple, dict], str] = _default_extract_input,
        extract_output: Callable[[Any], str] = _default_extract_output,
        default_tags: Optional[List[str]] = None,
    ) -> None:
        self.name = name
        self.system = system
        self.model = model
        self.operation = operation
        # Default: local-only. Generic export is opt-in via an explicit sink
        # or TRACER_WATCH_OTLP_ENDPOINT [+ _HEADERS].
        self.sink: Sink = sink or self._sink_from_env(name)
        self.extract_input = extract_input
        self.extract_output = extract_output
        self.default_tags = list(default_tags or [])

    @staticmethod
    def _sink_from_env(name: str) -> Sink:
        local = LocalFileSink(name, dir=os.environ.get("TRACER_WATCH_DIR", ".tracer/watch"))
        sinks: List[Sink] = [local]

        # Generic OTLP fan-out (any OTLP/HTTP backend).
        endpoint = os.environ.get("TRACER_WATCH_OTLP_ENDPOINT")
        if endpoint:
            headers: Dict[str, str] = {}
            raw = os.environ.get("TRACER_WATCH_OTLP_HEADERS", "")
            for pair in raw.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    headers[k.strip()] = v.strip()
            sinks.append(OTLPSink(endpoint, headers))

        return local if len(sinks) == 1 else MultiSink(sinks)

    # -- decorator -------------------------------------------------------- #
    def __call__(self, fn: Callable) -> Callable:
        import functools

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            span = self._begin(self.extract_input(args, kwargs))
            try:
                result = fn(*args, **kwargs)
            except Exception as e:
                span.status = "error"
                span.error = str(e)
                self._finish(span)
                raise
            # Auto-capture model / tokens / finish_reason / tool calls if the
            # function returned a provider response object; otherwise fall back
            # to the plain output extractor.
            extract_response(span, result)
            if not span.output_text:
                span.output_text = self.extract_output(result)
            self._finish(span)
            return result

        return wrapper

    # -- context manager -------------------------------------------------- #
    def span(
        self,
        input_text: str,
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **attributes: Any,
    ) -> "_SpanCtx":
        return _SpanCtx(
            self, input_text, attributes,
            user_id=user_id, session_id=session_id, metadata=metadata,
        )

    # -- low level -------------------------------------------------------- #
    def _begin(
        self,
        input_text: str,
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> GenAISpan:
        # Nest under any currently-open span so watched calls form a trace tree.
        parent = _ACTIVE.get()
        s = GenAISpan(
            operation_name=self.operation,
            system=self.system,
            request_model=self.model,
            response_model=self.model,
            input_text=input_text,
            start_time=datetime.now(timezone.utc).isoformat(),
            trace_id=parent.trace_id if parent else _new_id(32),
            span_id=_new_id(16),
            parent_span_id=parent.span_id if parent else None,
            user_id=user_id,
            session_id=session_id,
            tags=list(self.default_tags),
        )
        s.attributes["watcher"] = self.name
        s._t0 = time.perf_counter()  # type: ignore[attr-defined]
        s._tok = _ACTIVE.set(s)      # type: ignore[attr-defined]
        return s

    def _finish(self, span: GenAISpan) -> None:
        t0 = getattr(span, "_t0", None)
        if t0 is not None:
            span.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        tok = getattr(span, "_tok", None)
        if tok is not None:
            try:
                _ACTIVE.reset(tok)
            except (ValueError, LookupError):
                pass
        self.sink.emit(span)

    def close(self) -> None:
        self.sink.close()


class _SpanCtx:
    def __init__(
        self,
        watcher: Watcher,
        input_text: str,
        attributes: Dict[str, Any],
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._w = watcher
        self._span = watcher._begin(input_text, user_id=user_id, session_id=session_id)
        if metadata:
            self._span.attributes.update(metadata)
        self._span.attributes.update(attributes)

    def __enter__(self) -> "_SpanCtx":
        return self

    @property
    def span(self) -> GenAISpan:
        return self._span

    def set_output(self, output: Any) -> None:
        self._span.output_text = self._w.extract_output(output)

    def record(self, response: Any) -> "_SpanCtx":
        """Auto-capture model / tokens / finish_reason / tool calls / output
        from a provider response object."""
        extract_response(self._span, response)
        return self

    def set_usage(
        self,
        *,
        prompt: Optional[int] = None,
        completion: Optional[int] = None,
        total: Optional[int] = None,
        cost_usd: Optional[float] = None,
    ) -> "_SpanCtx":
        if prompt is not None:
            self._span.input_tokens = prompt
        if completion is not None:
            self._span.output_tokens = completion
        if total is not None:
            self._span.total_tokens = total
        elif prompt is not None and completion is not None:
            self._span.total_tokens = prompt + completion
        if cost_usd is not None:
            self._span.cost_usd = cost_usd
        return self

    def set_params(self, **params: Any) -> "_SpanCtx":
        """Record request sampling params (temperature, top_p, max_tokens, ...)."""
        self._span.request_params.update({k: v for k, v in params.items() if v is not None})
        return self

    def add_tool_call(self, name: str, arguments: Any = None, result: Any = None) -> "_SpanCtx":
        tc: Dict[str, Any] = {"name": name, "arguments": arguments}
        if result is not None:
            tc["result"] = result
        self._span.tool_calls.append(tc)
        return self

    def set_attribute(self, key: str, value: Any) -> None:
        self._span.attributes[key] = value

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self._span.status = "error"
            self._span.error = str(exc)
        self._w._finish(self._span)
        return False


def watch(name: str, **kwargs: Any) -> Watcher:
    """Create a :class:`Watcher`. Use as a decorator or context manager."""
    return Watcher(name, **kwargs)
