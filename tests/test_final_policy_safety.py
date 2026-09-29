"""Final-policy certification and artifact integrity, without model sweeps."""
import json
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from tracer.config import FitConfig
from tracer.fit.certification import certification_split, certify_policy
from tracer.fit.pipeline import _calibrate_threshold, _cp_lower, _holdout_indices
from tracer.policy.artifacts import save_pipeline, write_manifest
from tracer.runtime.router import Router
from tracer.types import ArtifactManifest


class SignClassifier:
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        y = np.asarray(X)[:, 0] > 0
        return np.column_stack([~y, y]).astype(float)


def candidate(clf=None):
    return {"stages": [{"clf": clf or SignClassifier(), "accept_all": True}],
            "summary": {"method": "global", "coverage_cal_total": 1.,
                        "teacher_agreement_cal_total": 1.}}


def traces(path, n=160, dim=1):
    y = np.arange(n) % 2
    path.write_text(''.join(json.dumps({"input": str(i), "teacher": str(v)}) + '\n'
                            for i, v in enumerate(y)))
    return np.tile((y * 2. - 1.)[:, None], (1, dim)), y


def stub_frontier(monkeypatch, best=None, capture=None):
    def build(X, y, targets, **kwargs):
        if capture is not None:
            capture.append((X.copy(), y.copy()))
        model = best or candidate()
        return ([{"target": t, "best": model, "candidates": [model]} for t in targets], {})
    monkeypatch.setattr("tracer.api.fit_frontier", build)


def test_selection_never_returns_a_lower_diagnostic_below_target():
    teacher = np.zeros(74, dtype=int)
    selection, verification = _holdout_indices(teacher, .7)
    predicted = np.ones(74, dtype=int)
    predicted[selection[:50]] = 0
    predicted[verification[:20]] = 0
    assert _calibrate_threshold(np.ones(74), predicted, teacher, .9) is None


def test_certification_checks_ood_and_handles_zero_accepts():
    clf = DummyClassifier(strategy="constant", constant=0).fit([[0.], [1.]], [0, 1])
    gate = {"k": 10, "global_thr": 0., "per_label_thr": {}}
    router = Router(candidate(clf)["stages"], ["0", "1"], SimpleNamespace(embedding_dim=1),
                    ood_gate=gate, train_embeddings=np.zeros((20, 1)))
    X = np.r_[np.full(90, 100.), np.zeros(10)][:, None]
    y = np.r_[np.zeros(95), np.ones(5)].astype(int)
    assert _cp_lower(95, 100, .1) >= .9  # Pre-OOD policy would clear this bound.
    result = certify_policy(router, X, y, .9, .1)
    assert result["status"] == "failed"
    assert (result["n_accepted"], result["n_correct"]) == (10, 5)
    empty = certify_policy(router, np.full((30, 1), 100.), np.zeros(30), .9, .1)
    assert empty["status"] == "failed"
    assert empty["n_accepted"] == 0
    assert empty["teacher_agreement"] is None
    assert empty["teacher_agreement_lower"] == 0


def test_reserved_rows_never_reach_training_or_ood_and_metrics_match_runtime(tmp_path, monkeypatch):
    from tracer.api import fit, load_router
    path = tmp_path / "traces.jsonl"
    X, y = traces(path)
    X = np.column_stack([X, np.arange(len(X))])
    capture, ood_rows = [], []
    stub_frontier(monkeypatch, capture=capture)
    def gate_fit(ref, labels):
        ood_rows.append(ref.copy())
        return {"k": 1, "global_thr": 1e6, "per_label_thr": {}}
    monkeypatch.setattr(importlib.import_module("tracer.fit.ood"), "fit_ood_gate", gate_fit)
    cfg = FitConfig(frontier_targets=(.9,), verbose=False)
    result = fit(path, tmp_path / "model", X, cfg)
    development, held = certification_split(len(X), cfg.certification_fraction, cfg.seed)
    np.testing.assert_array_equal(capture[0][0], X[development])
    np.testing.assert_array_equal(ood_rows[0], X[development])
    np.testing.assert_array_equal(np.load(tmp_path / "model" / "ood_reference.npy"), X[development])
    np.testing.assert_array_equal(np.load(tmp_path / "model" / "index.embeddings.npy"), X)
    router = load_router(tmp_path / "model")
    actual = certify_policy(router, X[held], y[held], .9, .1)
    assert result.manifest.certification == actual
    assert result.manifest.coverage_cal == actual["coverage"]
    assert result.manifest.ood_required


def test_failed_final_policy_is_not_replaced_using_reserved_outcomes(tmp_path, monkeypatch):
    from tracer.api import fit
    path = tmp_path / "traces.jsonl"
    X, _ = traces(path)
    wrong = DummyClassifier(strategy="constant", constant=0).fit([[0.], [1.]], [0, 1])
    bad, good = candidate(wrong), candidate()
    calls = []
    def build(*args, **kwargs):
        return ([{"target": .9, "best": bad, "candidates": [bad, good]}], {})
    monkeypatch.setattr("tracer.api.fit_frontier", build)
    def certify(*args, **kwargs):
        calls.append(1)
        return certify_policy(*args, **kwargs)
    monkeypatch.setattr("tracer.api.certify_policy", certify)
    result = fit(path, tmp_path / "model", X, FitConfig(frontier_targets=(.9,), verbose=False))
    assert result.manifest.selected_method is None
    assert result.manifest.certification["status"] == "failed"
    assert calls == [1]
    assert not (tmp_path / "model" / "pipeline.joblib").exists()


def test_reserved_only_labels_do_not_influence_label_discovery(tmp_path, monkeypatch):
    from tracer.api import fit
    path = tmp_path / "traces.jsonl"
    X, _ = traces(path)
    _, held = certification_split(len(X), .2, 42)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[int(held[0])]["teacher"] = "!reserved_only"
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    capture = []
    stub_frontier(monkeypatch, capture=capture)
    result = fit(path, tmp_path / "model", X, FitConfig(frontier_targets=(.9,), verbose=False))
    assert result.manifest.label_space == ["0", "1"]
    assert set(capture[0][1]) == {0, 1}
    assert result.manifest.certification["n_correct"] == len(held) - 1


def test_direct_refit_failure_preserves_working_generation(tmp_path, monkeypatch):
    from tracer.api import fit, load_router
    path = tmp_path / "traces.jsonl"
    X, _ = traces(path)
    stub_frontier(monkeypatch)
    out = tmp_path / "model"
    config = FitConfig(frontier_targets=(.9,), verbose=False)
    fit(path, out, X, config)
    before = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
    def fail_report(**kwargs):
        raise OSError("report failed after pipeline serialization")
    monkeypatch.setattr("tracer.api.build_qualitative_report", fail_report)
    with pytest.raises(OSError, match="report failed"):
        fit(path, out, np.column_stack([X, X]), config)
    assert {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()} == before
    assert load_router(out).predict(np.array([1.]))["decision"] == "handled"


@pytest.mark.parametrize("damage", ["json", "missing_gate", "missing_reference", "bad_reference", "bad_threshold"])
def test_expected_ood_damage_prevents_loading(tmp_path, damage):
    manifest = ArtifactManifest(version="0.1.0", n_traces=20, label_space=["0", "1"],
                                selected_method="global", embedding_dim=1, ood_required=True)
    write_manifest(tmp_path / "manifest.json", manifest)
    save_pipeline(tmp_path, candidate(), ["0", "1"])
    gate = {"k": 10, "global_thr": 1., "per_label_thr": {}}
    (tmp_path / "ood.json").write_text(json.dumps(gate))
    np.save(tmp_path / "ood_reference.npy", np.zeros((20, 1)))
    if damage == "json":
        (tmp_path / "ood.json").write_text("{broken")
    elif damage == "missing_gate":
        (tmp_path / "ood.json").unlink()
    elif damage == "missing_reference":
        (tmp_path / "ood_reference.npy").unlink()
    elif damage == "bad_reference":
        np.save(tmp_path / "ood_reference.npy", np.full((20, 1), np.nan))
    else:
        (tmp_path / "ood.json").write_text(json.dumps({**gate, "global_thr": float("inf")}))
    with pytest.raises(ValueError, match="OOD"):
        Router.load(tmp_path)


def test_legacy_unguarded_artifacts_load_without_inventing_certification(tmp_path):
    manifest = ArtifactManifest(version="0.1.0", n_traces=20, label_space=["0", "1"],
                                selected_method="global", embedding_dim=1)
    write_manifest(tmp_path / "manifest.json", manifest)
    payload = json.loads((tmp_path / "manifest.json").read_text())
    payload.pop("certification")
    payload.pop("ood_required")
    (tmp_path / "manifest.json").write_text(json.dumps(payload))
    save_pipeline(tmp_path, candidate(), ["0", "1"])
    loaded = Router.load(tmp_path)
    assert loaded.manifest.certification is None
    assert loaded.predict(np.array([1.]))["label"] == "1"
