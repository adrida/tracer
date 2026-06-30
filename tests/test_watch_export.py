"""Tests for watch JSONL → fit-ready traces export."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import tracer
from tracer.traces.loader import load_traces
from tracer.traces.watch_export import (
    collect_watch_paths,
    export_watch_traces,
    load_watch_spans,
    span_from_dict,
    watch_to_dataset,
)
from tracer.watch import GenAISpan, LocalFileSink, watch


def _write_span(path: Path, span: GenAISpan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(span.to_dict(), ensure_ascii=False) + "\n")


def test_span_from_dict_roundtrip():
    original = GenAISpan(
        input_text="hello",
        output_text="billing",
        system="acme",
        trace_id="t1",
        status="ok",
    )
    restored = span_from_dict(original.to_dict())
    assert restored.input_text == "hello"
    assert restored.output_text == "billing"
    assert restored.system == "acme"
    assert restored.trace_id == "t1"


def test_load_watch_spans_reads_jsonl(tmp_path):
    path = tmp_path / "app.jsonl"
    _write_span(path, GenAISpan(input_text="a", output_text="x"))
    _write_span(path, GenAISpan(input_text="b", output_text="y"))
    spans = load_watch_spans(path)
    assert len(spans) == 2
    assert spans[0].input_text == "a"


def test_watch_to_dataset_skips_errors_by_default(tmp_path):
    path = tmp_path / "app.jsonl"
    _write_span(path, GenAISpan(input_text="ok", output_text="label", status="ok"))
    _write_span(path, GenAISpan(input_text="bad", output_text="", status="ok"))
    _write_span(path, GenAISpan(input_text="err", output_text="x", status="error"))

    dataset, n_read, n_skipped = watch_to_dataset(path)
    assert n_read == 3
    assert n_skipped == 2
    assert len(dataset.records) == 1
    assert dataset.records[0].input_text == "ok"
    assert dataset.records[0].teacher_label == "label"


def test_export_watch_traces_directory(tmp_path):
    watch_dir = tmp_path / "watch"
    _write_span(watch_dir / "a.jsonl", GenAISpan(input_text="one", output_text="A"))
    _write_span(watch_dir / "b.jsonl", GenAISpan(input_text="two", output_text="B"))

    out = tmp_path / "traces.jsonl"
    result = export_watch_traces(watch_dir, out)

    assert result.n_files == 2
    assert result.n_exported == 2
    assert out.exists()
    dataset = load_traces(out)
    assert len(dataset) == 2
    labels = {r.teacher_label for r in dataset.records}
    assert labels == {"A", "B"}


def test_export_watch_traces_empty_raises(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="No exportable watch spans"):
        export_watch_traces(path, tmp_path / "out.jsonl")


def test_watch_decorator_export_then_load_traces(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    w = watch("classifier")
    @w
    def classify(ticket: str) -> str:
        return "check_balance"

    classify("What is my balance?")

    out = tmp_path / "traces.jsonl"
    result = export_watch_traces(".tracer/watch", out)
    assert result.n_exported == 1

    dataset = load_traces(out)
    assert dataset.records[0].input_text == "What is my balance?"
    assert dataset.records[0].teacher_label == "check_balance"


def test_exported_traces_fit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    w = watch("classifier")
    labels = ["check_balance", "transfer_money", "check_balance"]
    inputs = ["balance?", "send money", "my balance please"]

    @w
    def classify(ticket: str) -> str:
        return labels.pop(0)

    for text in inputs:
        classify(text)

    traces_path = tmp_path / "traces.jsonl"
    export_watch_traces(".tracer/watch", traces_path)

    # Minimal synthetic embeddings so fit() can run without sentence-transformers.
    dataset = load_traces(traces_path)
    rng = np.random.RandomState(0)
    X = rng.randn(len(dataset), 8).astype(np.float32)

    artifact_dir = tmp_path / ".tracer"
    result = tracer.fit(
        traces_path,
        artifact_dir,
        embeddings=X,
        config=tracer.FitConfig(verbose=False, min_deploy_coverage=0.0),
    )
    assert result.manifest.n_traces == 3


def test_collect_watch_paths_single_file(tmp_path):
    path = tmp_path / "one.jsonl"
    path.write_text("{}")
    assert collect_watch_paths(path) == [path]


def test_collect_watch_paths_empty_dir(tmp_path):
    empty = tmp_path / "watch"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No .jsonl watch files"):
        collect_watch_paths(empty)
