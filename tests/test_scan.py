"""Tests for `tracer scan` (scanner.py).

Covers the behaviours of the day-one scan: the data-volume gate (min 1,000
traces unless forced, ~5,000 suggested), --force coarsening, the stable
label-colour map, and the terminal + HTML rendering. Everything here uses only
the base deps (numpy, scikit-learn), so it runs in the core CI matrix without
the embeddings/viz extras and without any network access.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tracer.scanner import (
    scan, format_scan, scan_html, load_scan_traces,
    MIN_SCAN_TRACES, SUGGESTED_SCAN_TRACES, ThinDataError, _label_colors,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_traces(tmp_path, n, n_classes=4, dim=24, sep=4.0, seed=0):
    """Well-separated synthetic traces + matching embeddings, so a clean
    cluster can actually certify."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_classes, dim) * sep
    y = rng.randint(0, n_classes, n)
    X = (centers[y] + rng.randn(n, dim) * 0.6).astype(np.float32)
    p = tmp_path / "traces.jsonl"
    with p.open("w") as f:
        for i in range(n):
            lab = f"intent_{y[i]}"
            f.write(json.dumps({"input": f"question {i} about {lab}", "teacher": lab}) + "\n")
    return p, X


# ── Data-volume gate ────────────────────────────────────────────────────────

def test_thin_data_gate_blocks_below_minimum(tmp_path):
    p, X = _write_traces(tmp_path, 200)
    with pytest.raises(ThinDataError):
        scan(p, embeddings=X, force=False)


def test_thin_data_gate_message_points_at_force_and_suggestion(tmp_path):
    p, X = _write_traces(tmp_path, 200)
    with pytest.raises(ThinDataError) as exc:
        scan(p, embeddings=X, force=False)
    msg = str(exc.value)
    assert f"{MIN_SCAN_TRACES:,}" in msg
    assert f"{SUGGESTED_SCAN_TRACES:,}" in msg
    assert "--force" in msg


def test_scan_runs_above_minimum_without_force(tmp_path):
    p, X = _write_traces(tmp_path, 1200)
    r = scan(p, embeddings=X, force=False)
    assert r.forced is False
    assert r.n_traces == 1200
    # well-separated data should certify a meaningful share
    assert r.certifiable_share > 0.5


# ── --force ───────────────────────────────────────────────────────────────────

def test_force_runs_below_minimum_and_marks_forced(tmp_path):
    p, X = _write_traces(tmp_path, 300)
    r = scan(p, embeddings=X, force=True)
    assert r.forced is True
    # force suppresses the "collect more" nudge
    assert r.traces_needed_estimate is None


def test_force_coarsens_clustering(tmp_path):
    p, X = _write_traces(tmp_path, 1500)
    base = scan(p, embeddings=X, force=False)
    forced = scan(p, embeddings=X, force=True)
    assert forced.n_clusters <= base.n_clusters
    assert forced.forced is True


# ── Label colours ─────────────────────────────────────────────────────────────

def test_label_colors_stable_and_distinct():
    m = _label_colors(["b", "a", "c", "a"])
    assert set(m) == {"a", "b", "c"}
    assert len({m["a"], m["b"], m["c"]}) == 3
    # deterministic regardless of input order or duplicates
    assert _label_colors(["c", "a", "b", "b"]) == m


# ── Rendering ─────────────────────────────────────────────────────────────────

def test_format_scan_renders(tmp_path):
    p, X = _write_traces(tmp_path, 1200)
    out = format_scan(scan(p, embeddings=X))
    assert "certifiable" in out.lower()
    assert "tracer fit" in out


def test_scan_html_has_branding_and_label_toggle(tmp_path):
    p, X = _write_traces(tmp_path, 1200)
    html = scan_html(scan(p, embeddings=X), "synthetic")
    for needle in ("tracerml.ai", "vt-label", "vt-verdict",
                   "function setMode", 'id="run-fit"', 'id="texttip"'):
        assert needle in html, f"missing {needle!r} in scan_html output"


def test_forced_report_shows_banner(tmp_path):
    p, X = _write_traces(tmp_path, 300)
    html = scan_html(scan(p, embeddings=X, force=True), "synthetic")
    assert "Forced scan" in html


# ── Loader aliases ─────────────────────────────────────────────────────────────

def test_loader_accepts_key_aliases(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"query": "hi there", "label": "greeting"}) + "\n")
    inputs, labels = load_scan_traces(p)
    assert inputs == ["hi there"]
    assert labels == ["greeting"]
