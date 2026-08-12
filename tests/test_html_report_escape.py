"""XSS hardening for the fit HTML report."""
import json
from pathlib import Path

import numpy as np
import pytest

from tracer.api import fit
from tracer.config import FitConfig
from tracer.analysis.html_report import generate_html_report


def _fit_with_poisoned_label(tmp_path: Path):
    """Fit a tiny deployable policy, then poison qualitative_report examples."""
    rng = np.random.RandomState(0)
    n, dim, n_classes = 240, 16, 3
    centers = rng.randn(n_classes, dim) * 4
    labels_int = rng.randint(0, n_classes, size=n)
    X = (centers[labels_int] + rng.randn(n, dim) * 0.6).astype(np.float32)
    names = [f"cls_{i}" for i in range(n_classes)]
    teacher = [names[i] for i in labels_int]
    traces = tmp_path / "traces.jsonl"
    with traces.open("w") as f:
        for i in range(n):
            f.write(json.dumps({"input": f"text {i}", "teacher": teacher[i], "id": str(i)}) + "\n")
    artifact = tmp_path / ".tracer"
    result = fit(traces, artifact, embeddings=X, config=FitConfig(verbose=False))
    assert result.manifest.selected_method is not None

    qr_path = artifact / "qualitative_report.json"
    qr = json.loads(qr_path.read_text())
    payload = "<img src=x onerror=alert(1)>"
    qr["handled_examples"] = [{
        "input_preview": payload,
        "teacher_label": payload,
        "accept_score": 0.99,
    }]
    qr["deferred_examples"] = [{
        "input_preview": payload,
        "teacher_label": '"><script>alert(1)</script>',
    }]
    qr["boundary_pairs"] = [{
        "teacher_label": payload,
        "handled_preview": payload,
        "deferred_preview": payload,
        "handled_score": 0.9,
        "deferred_score": 0.1,
    }]
    qr_path.write_text(json.dumps(qr))
    return artifact


def test_html_report_escapes_trace_text(tmp_path):
    artifact = _fit_with_poisoned_label(tmp_path)
    out = Path(generate_html_report(artifact))
    html = out.read_text(encoding="utf-8")
    assert "<img src=x onerror=alert(1)>" not in html
    assert "<script>alert(1)</script>" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    assert "plotly-latest" not in html


def test_sankey_labels_strip_markup():
    from tracer.analysis.sankey import _safe_node_label
    assert "<" not in _safe_node_label('</script><img src=x onerror=alert(1)>')
    assert _safe_node_label("normal_label") == "normal_label"


def test_html_report_escapes_poisoned_embedding_dim(tmp_path):
    artifact = _fit_with_poisoned_label(tmp_path)
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["embedding_dim"] = '<img src=x onerror=alert(1)>'
    manifest_path.write_text(json.dumps(manifest))
    html = Path(generate_html_report(artifact)).read_text(encoding="utf-8")
    assert "<img src=x onerror=alert(1)>" not in html
    # Non-int embedding_dim is coerced away → "?" display, not raw payload.
    assert "?-dim embeddings" in html


def test_scanner_viz_json_escapes_script_breakout():
    from tracer.scanner import ScanResult, scan_html
    payload = "</script><script>alert(1)</script>"
    r = ScanResult(
        n_traces=10,
        n_classes=1,
        n_clusters=1,
        target=0.9,
        certifiable_share=0.5,
        certified_floor=0.5,
        clusters=[],
        forced=False,
        projection={
            "points": [{"x": 0, "y": 0, "z": 0, "c": 0, "t": payload}],
            "clusters": {"0": {"label": payload, "examples": [payload]}},
        },
    )
    html = scan_html(r, source_name="unit")
    assert "</script><script>alert(1)</script>" not in html
    assert "\\u003c" in html
