"""Convert ``tracer.watch`` JSONL spans into fit-ready trace files.

Watch records GenAI spans (``input_text`` / ``output_text``). ``tracer.fit``
expects canonical traces (``input`` / ``teacher``). This module bridges the two
using :meth:`GenAISpan.to_trace_record`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, Union

from tracer.types import TraceDataset, TraceRecord
from tracer.traces.loader import save_traces
from tracer.watch import GenAISpan


@dataclass
class WatchExportResult:
    """Summary of a watch → traces export."""

    output_path: Path
    n_files: int
    n_spans_read: int
    n_exported: int
    n_skipped: int


def span_from_dict(row: dict) -> GenAISpan:
    """Reconstruct a :class:`GenAISpan` from a JSON object (e.g. watch JSONL)."""
    valid = {f.name for f in fields(GenAISpan)}
    kwargs = {k: v for k, v in row.items() if k in valid}
    return GenAISpan(**kwargs)


def _span_is_exportable(span: GenAISpan) -> bool:
    if span.status == "error":
        return False
    if not span.input_text.strip():
        return False
    if not span.output_text.strip():
        return False
    return True


def collect_watch_paths(source: Union[str, Path]) -> List[Path]:
    """Resolve a watch JSONL file or a directory of ``*.jsonl`` watch files."""
    source = Path(source)
    if source.is_file():
        return [source]
    if source.is_dir():
        paths = sorted(source.glob("*.jsonl"))
        if not paths:
            raise FileNotFoundError(f"No .jsonl watch files found in {source}")
        return paths
    raise FileNotFoundError(f"Watch source not found: {source}")


def load_watch_spans(path: Union[str, Path]) -> List[GenAISpan]:
    """Load GenAI spans from one watch JSONL file."""
    path = Path(path)
    spans: List[GenAISpan] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON in {path} at line {lineno}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected a JSON object in {path} at line {lineno}, "
                    f"got {type(row).__name__}"
                )
            spans.append(span_from_dict(row))
    return spans


def watch_to_dataset(
    source: Union[str, Path],
    *,
    skip_errors: bool = True,
) -> tuple[TraceDataset, int, int]:
    """Convert watch span file(s) into a :class:`TraceDataset`.

    Returns ``(dataset, n_spans_read, n_skipped)``.
    """
    paths = collect_watch_paths(source)
    records: List[TraceRecord] = []
    n_read = 0
    n_skipped = 0

    for path in paths:
        for span in load_watch_spans(path):
            n_read += 1
            if skip_errors and not _span_is_exportable(span):
                n_skipped += 1
                continue
            records.append(span.to_trace_record())

    return TraceDataset(records=records), n_read, n_skipped


def export_watch_traces(
    source: Union[str, Path],
    output: Union[str, Path],
    *,
    skip_errors: bool = True,
) -> WatchExportResult:
    """Write fit-ready traces JSONL from watch span file(s).

    Parameters
    ----------
    source : watch JSONL file or directory (e.g. ``.tracer/watch``)
    output : destination traces JSONL path
    skip_errors : if True (default), drop errored spans and rows with empty
        input or output text
    """
    paths = collect_watch_paths(source)
    dataset, n_read, n_skipped = watch_to_dataset(source, skip_errors=skip_errors)
    if not dataset.records:
        raise ValueError(
            "No exportable watch spans found. "
            "Ensure watched calls completed with input and output text, "
            "or pass skip_errors=False to include partial rows."
        )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_traces(dataset, output)

    return WatchExportResult(
        output_path=output,
        n_files=len(paths),
        n_spans_read=n_read,
        n_exported=len(dataset.records),
        n_skipped=n_skipped,
    )
