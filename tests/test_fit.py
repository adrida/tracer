"""Tests for the tracer package."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_traces(tmpdir, n=300, dim=32, n_classes=4, noise=0.08):
    """Create a synthetic JSONL trace file + embeddings."""
    rng = np.random.RandomState(42)
    centers = rng.randn(n_classes, dim) * 3
    labels_int = rng.randint(0, n_classes, size=n)
    X = centers[labels_int] + rng.randn(n, dim) * 0.8
    label_names = [f"cls_{i}" for i in range(n_classes)]
    teacher = list(label_names[i] for i in labels_int)
    # inject some teacher noise
    for i in range(n):
        if rng.random() < noise:
            teacher[i] = label_names[rng.randint(0, n_classes)]

    traces_path = Path(tmpdir) / "traces.jsonl"
    with traces_path.open("w") as f:
        for i in range(n):
            row = {
                "input": f"sample text {i}",
                "teacher": teacher[i],
                "id": str(i),
                "ground_truth": label_names[labels_int[i]],
            }
            f.write(json.dumps(row) + "\n")
    np.save(traces_path.with_suffix(".npy"), X)
    return traces_path, X, label_names


# ── Loader ────────────────────────────────────────────────────────────────────

def test_load_traces_basic():
    from tracer.traces.loader import load_traces
    with tempfile.TemporaryDirectory() as tmp:
        path, _, _ = _make_traces(tmp)
        ds = load_traces(path)
        assert len(ds) == 300
        assert len(ds.label_space) == 4
        assert all(r.teacher_label is not None for r in ds.records)
        assert all(r.ground_truth is not None for r in ds.records)


def test_load_traces_minimal_fields():
    """JSONL with only input + teacher should load fine."""
    from tracer.traces.loader import load_traces
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "minimal.jsonl"
        with path.open("w") as f:
            f.write(json.dumps({"input": "hello", "teacher": "greet"}) + "\n")
            f.write(json.dumps({"input": "bye", "teacher": "farewell"}) + "\n")
        ds = load_traces(path)
        assert len(ds) == 2
        assert ds.records[0].trace_id is None
        assert ds.records[0].ground_truth is None


def test_load_traces_skips_blank_lines():
    from tracer.traces.loader import load_traces
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "blank.jsonl"
        with path.open("w") as f:
            f.write(json.dumps({"input": "a", "teacher": "x"}) + "\n")
            f.write("\n")
            f.write(json.dumps({"input": "b", "teacher": "y"}) + "\n")
        ds = load_traces(path)
        assert len(ds) == 2


def test_load_traces_malformed_json_raises():
    from tracer.traces.loader import load_traces
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.jsonl"
        with path.open("w") as f:
            f.write("{not valid json}\n")
        with pytest.raises(ValueError, match="Malformed JSON"):
            load_traces(path)


def test_load_traces_missing_fields_raises():
    from tracer.traces.loader import load_traces
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "missing.jsonl"
        with path.open("w") as f:
            f.write(json.dumps({"input": "hello"}) + "\n")  # missing 'teacher'
        with pytest.raises(ValueError, match="missing required fields"):
            load_traces(path)


# ── Fit & route ───────────────────────────────────────────────────────────────

def test_fit_and_route():
    from tracer.api import fit, load_router
    with tempfile.TemporaryDirectory() as tmp:
        path, X, _ = _make_traces(tmp)
        artifact_dir = Path(tmp) / ".tracer"

        result = fit(path, artifact_dir=artifact_dir)

        assert result.manifest.n_traces == 300
        assert result.manifest.embedding_dim == 32
        assert len(result.manifest.label_space) == 4
        assert (artifact_dir / "manifest.json").exists()
        assert (artifact_dir / "pipeline.joblib").exists()
        assert (artifact_dir / "config.json").exists()
        assert (artifact_dir / "frontier.json").exists()

        router = load_router(artifact_dir)
        out = router.predict(X[0])
        assert out["decision"] in ("handled", "deferred")
        assert "label" in out
        assert "accept_score" in out
        assert "stage" in out

        batch = router.predict_batch(X[:20])
        assert len(batch["decisions"]) == 20
        assert len(batch["labels"]) == 20
        assert len(batch["handled"]) == 20


def test_fit_precomputed_embeddings():
    """tracer.fit() accepts embeddings passed directly, not just from .npy file."""
    from tracer.api import fit
    with tempfile.TemporaryDirectory() as tmp:
        path, X, _ = _make_traces(tmp)
        # Remove the auto-discovered .npy so fit must use the passed array
        path.with_suffix(".npy").unlink()
        result = fit(path, artifact_dir=Path(tmp) / ".tracer", embeddings=X)
        assert result.manifest.n_traces == 300


def test_fit_embedding_mismatch_raises():
    """Mismatched trace count and embedding count should raise clearly."""
    from tracer.api import fit
    with tempfile.TemporaryDirectory() as tmp:
        path, X, _ = _make_traces(tmp)
        with pytest.raises(ValueError, match="mismatch"):
            fit(path, artifact_dir=Path(tmp) / ".tracer", embeddings=X[:100])


def test_fit_missing_embeddings_raises():
    from tracer.api import fit
    with tempfile.TemporaryDirectory() as tmp:
        path, _, _ = _make_traces(tmp)
        path.with_suffix(".npy").unlink()
        with pytest.raises(FileNotFoundError):
            fit(path, artifact_dir=Path(tmp) / ".tracer")


def test_router_wrong_dim_raises():
    from tracer.api import fit, load_router
    with tempfile.TemporaryDirectory() as tmp:
        path, X, _ = _make_traces(tmp)
        result = fit(path, artifact_dir=Path(tmp) / ".tracer")
        if result.manifest.selected_method is None:
            pytest.skip("No deployable pipeline at this target")
        router = load_router(Path(tmp) / ".tracer")
        wrong_emb = np.random.randn(64).astype(np.float32)  # 64 != 32
        with pytest.raises(ValueError, match="dimension mismatch"):
            router.predict(wrong_emb)


# ── Continual learning ────────────────────────────────────────────────────────

def test_update():
    from tracer.api import fit, update
    with tempfile.TemporaryDirectory() as tmp:
        path, X, _ = _make_traces(tmp, n=200)
        artifact_dir = Path(tmp) / ".tracer"
        fit(path, artifact_dir=artifact_dir)

        new_path = Path(tmp) / "new_traces.jsonl"
        n_new = 100
        rng = np.random.RandomState(99)
        X_new = rng.randn(n_new, 32).astype(np.float32)
        with new_path.open("w") as f:
            for i in range(n_new):
                f.write(json.dumps({"input": f"new {i}", "teacher": f"cls_{i % 4}"}) + "\n")
        np.save(new_path.with_suffix(".npy"), X_new)

        updated = update(new_path, artifact_dir=artifact_dir)
        assert updated.manifest.n_traces == 300  # 200 + 100


# ── Qualitative report ────────────────────────────────────────────────────────

def test_qualitative_report():
    from tracer.api import fit
    with tempfile.TemporaryDirectory() as tmp:
        path, _, _ = _make_traces(tmp)
        result = fit(path, artifact_dir=Path(tmp) / ".tracer")
        qr = result.qualitative_report
        if qr is not None:
            assert 0.0 <= qr.coverage <= 1.0
            assert len(qr.slices) > 0
            assert all(s.handled_rate + s.deferred_rate == pytest.approx(1.0) for s in qr.slices)
            assert len(qr.handled_examples) > 0


def test_qualitative_report_boundary_pairs():
    from tracer.api import fit
    with tempfile.TemporaryDirectory() as tmp:
        path, _, _ = _make_traces(tmp)
        result = fit(path, artifact_dir=Path(tmp) / ".tracer")
        qr = result.qualitative_report
        if qr and qr.boundary_pairs:
            for bp in qr.boundary_pairs:
                assert bp.teacher_label is not None
                assert len(bp.handled_preview) > 0
                assert len(bp.deferred_preview) > 0


# ── Report ────────────────────────────────────────────────────────────────────

def test_report():
    from tracer.api import fit, report
    with tempfile.TemporaryDirectory() as tmp:
        path, X, _ = _make_traces(tmp)
        artifact_dir = Path(tmp) / ".tracer"
        fit(path, artifact_dir=artifact_dir)
        manifest = report(artifact_dir)
        assert manifest.n_traces == 300
        assert manifest.version == "0.1.0"


# ── CLI demo ──────────────────────────────────────────────────────────────────

def test_demo_runs():
    """The demo CLI command should run without errors."""
    import argparse
    from tracer.cli.main import _cmd_demo
    _cmd_demo(argparse.Namespace())


# ── Progress logging & candidate skipping ────────────────────────────────────

def test_search_surrogate_invokes_on_candidate_callback():
    """The sweep should invoke on_candidate once per candidate with a usable val_f1.

    Supports both the legacy 2-arg signature (name, val_f1) and the preferred
    3-arg signature (name, val_f1, elapsed_seconds).
    """
    from tracer.fit.surrogate import _candidates, search_best_surrogate
    rng = np.random.RandomState(0)
    X_tr = rng.randn(200, 16).astype(np.float32); y_tr = rng.randint(0, 3, size=200)
    X_va = rng.randn(60,  16).astype(np.float32); y_va = rng.randint(0, 3, size=60)

    calls_short, calls_full = [], []
    def cb_short(name, f1):
        calls_short.append((name, f1))
    def cb_full(name, f1, elapsed):
        calls_full.append((name, f1, elapsed))

    expected = set(_candidates(len(X_tr), skip=("gbt", "xgb")).keys())

    search_best_surrogate(X_tr, y_tr, X_va, y_va, on_candidate=cb_short, skip=("gbt", "xgb"))
    search_best_surrogate(X_tr, y_tr, X_va, y_va, on_candidate=cb_full,  skip=("gbt", "xgb"))

    assert {n for n, _ in calls_short}.issuperset(expected - {"sgd_log"}) or \
           len(calls_short) >= len(expected) - 1  # sgd can fail on tiny data
    assert all(0.0 <= f <= 1.0 for _, f in calls_short)
    assert all(isinstance(el, float) and el >= 0 for _, _, el in calls_full)
    # Elapsed timings in the extended callback should match the returned
    # metrics[fit_seconds] value that now accompanies the best model.
    _, _, metrics = search_best_surrogate(X_tr, y_tr, X_va, y_va, skip=("gbt", "xgb"))
    assert "fit_seconds" in metrics and metrics["fit_seconds"] >= 0


def test_search_surrogate_callback_exception_does_not_abort():
    """A raising on_candidate must not derail the sweep — warn and keep going."""
    from tracer.fit.surrogate import search_best_surrogate
    rng = np.random.RandomState(0)
    X_tr = rng.randn(200, 16).astype(np.float32); y_tr = rng.randint(0, 3, size=200)
    X_va = rng.randn(60,  16).astype(np.float32); y_va = rng.randint(0, 3, size=60)

    def cb_raises(name, f1):
        raise RuntimeError("boom")

    import warnings as _w
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        clf, name, _ = search_best_surrogate(X_tr, y_tr, X_va, y_va,
                                             on_candidate=cb_raises, skip=("gbt", "xgb"))
    assert clf is not None and name is not None
    assert any("on_candidate callback raised" in str(w.message) for w in rec)


def test_skip_candidates_removes_models():
    """FitConfig.skip_candidates should remove the named models from _candidates()."""
    from tracer.fit.surrogate import _candidates
    full = set(_candidates(3000).keys())
    skipped = set(_candidates(3000, skip=("gbt", "mlp_1h")).keys())
    assert "mlp_1h" in full and "mlp_1h" not in skipped
    if "gbt" in full:
        assert "gbt" not in skipped


def test_fit_verbose_emits_stderr_progress(capsys):
    """FitConfig.verbose=True should print `[tracer.fit +...s] ...` to stderr."""
    from tracer.api import fit
    from tracer.config import FitConfig
    with tempfile.TemporaryDirectory() as tmp:
        path, _, _ = _make_traces(tmp, n=200, dim=16, n_classes=3)
        fit(path, artifact_dir=Path(tmp) / ".tracer",
            config=FitConfig(verbose=True,
                             target_teacher_agreement=0.90,
                             frontier_targets=(0.90,),
                             skip_candidates=("gbt", "xgb")))
    err = capsys.readouterr().err
    assert "[tracer.fit" in err
    assert "fit_frontier" in err
    # At least one candidate heartbeat should have appeared.
    assert any(tok in err for tok in ("logreg_c1", "mlp_1h", "rf", "et", "dt"))


def test_fit_verbose_false_is_silent(capsys):
    from tracer.api import fit
    from tracer.config import FitConfig
    with tempfile.TemporaryDirectory() as tmp:
        path, _, _ = _make_traces(tmp, n=200, dim=16, n_classes=3)
        fit(path, artifact_dir=Path(tmp) / ".tracer",
            config=FitConfig(verbose=False,
                             target_teacher_agreement=0.90,
                             frontier_targets=(0.90,),
                             skip_candidates=("gbt", "xgb")))
    err = capsys.readouterr().err
    assert "[tracer.fit" not in err


# ── Exports ──────────────────────────────────────────────────────────────────

def test_public_exports():
    """All public types should be importable from the tracer package."""
    import tracer
    assert hasattr(tracer, "fit")
    assert hasattr(tracer, "update")
    assert hasattr(tracer, "load_router")
    assert hasattr(tracer, "report")
    assert hasattr(tracer, "embed")
    assert hasattr(tracer, "FitConfig")
    assert hasattr(tracer, "EmbeddingConfig")
    assert hasattr(tracer, "TraceRecord")
    assert hasattr(tracer, "TraceDataset")
    assert hasattr(tracer, "FitResult")
    assert hasattr(tracer, "QualitativeReport")
    assert hasattr(tracer, "ArtifactManifest")
    assert hasattr(tracer, "SliceInsight")
    assert hasattr(tracer, "BoundaryPair")
    assert hasattr(tracer, "TemporalDelta")
    assert hasattr(tracer, "RepresentativeExample")


if __name__ == "__main__":
    import traceback
    tests = [
        test_load_traces_basic,
        test_load_traces_minimal_fields,
        test_load_traces_skips_blank_lines,
        test_load_traces_malformed_json_raises,
        test_load_traces_missing_fields_raises,
        test_fit_and_route,
        test_fit_precomputed_embeddings,
        test_fit_embedding_mismatch_raises,
        test_fit_missing_embeddings_raises,
        test_router_wrong_dim_raises,
        test_update,
        test_qualitative_report,
        test_qualitative_report_boundary_pairs,
        test_report,
        test_search_surrogate_invokes_on_candidate_callback,
        test_search_surrogate_callback_exception_does_not_abort,
        test_skip_candidates_removes_models,
        test_demo_runs,
        test_public_exports,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception:
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
