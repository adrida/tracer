"""Load teacher traces from JSONL files.

Expected schema per line:
    {"input": "...", "teacher": "label"}

Optional fields: id, ground_truth, embedding, metadata
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Union

from tracer.types import TraceDataset, TraceRecord


def load_traces(path: Union[str, Path]) -> TraceDataset:
    """Load a JSONL trace file into a TraceDataset."""
    path = Path(path)
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON in {path} at line {len(records) + 1}: {exc}"
                ) from exc
            if "input" not in row or "teacher" not in row:
                raise ValueError(
                    f"Trace at line {len(records) + 1} is missing required fields "
                    f"'input' and/or 'teacher'. Got keys: {list(row.keys())}"
                )
            teacher_val = row["teacher"]
            if teacher_val is None or (isinstance(teacher_val, float) and math.isnan(teacher_val)):
                raise ValueError(
                    f"Trace at line {len(records) + 1} has a null/NaN 'teacher' field. "
                    f"This usually happens when json.dumps() serializes float('nan') as "
                    f"unquoted NaN. Filter out rows with missing labels before fitting."
                )
            if not isinstance(teacher_val, str):
                teacher_val = str(teacher_val)
            records.append(TraceRecord(
                input_text=str(row["input"]),
                teacher_label=teacher_val,
                trace_id=str(row["id"]) if row.get("id") is not None else None,
                ground_truth=str(row["ground_truth"]) if row.get("ground_truth") is not None else None,
                metadata=dict(row.get("metadata", {})),
            ))
    return TraceDataset(records=records)


def save_traces(dataset: TraceDataset, path: Union[str, Path]) -> None:
    """Write a TraceDataset to JSONL."""
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for r in dataset.records:
            row = {"input": r.input_text, "teacher": r.teacher_label}
            if r.trace_id is not None:
                row["id"] = r.trace_id
            if r.ground_truth is not None:
                row["ground_truth"] = r.ground_truth
            if r.metadata:
                row["metadata"] = r.metadata
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
