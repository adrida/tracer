"""Focused checks for label identity, calibration, seeds and persisted embeddings."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from tracer.api import fit, load_router
from tracer.config import FitConfig
from tracer.embeddings.index import EmbeddingIndex
from tracer.fit.ood import fit_ood_gate, ood_mask
from tracer.fit.pipeline import _predict, apply_stage, _calibrate_threshold, _cp_lower, fit_frontier
from tracer.runtime.router import Router

SKIP = ('logreg_c10', 'sgd_log', 'mlp_1h', 'mlp_2h', 'dt', 'rf', 'et', 'gbt', 'xgb')


def test_residual_stage_preserves_noncontiguous_labels():
    X = np.array([[-3.], [-2.], [-1.], [1.], [2.], [3.]])
    y = np.array([1, 1, 1, 3, 3, 3])
    clf = LogisticRegression().fit(X, y)
    expected = clf.predict(X)
    np.testing.assert_array_equal(_predict(clf, X)[0], expected)
    stage = {'clf': clf, 'accept_all': True}
    router = Router([stage], ['zero', 'one', 'two', 'three'], SimpleNamespace(embedding_dim=1))
    assert router.predict(X[-1])['label'] == 'three'
    assert router.predict_batch(X)['labels'] == ['one'] * 3 + ['three'] * 3


def test_small_calibration_chooses_largest_eligible_set():
    scores = np.linspace(.50, .99, 36)
    teacher = np.zeros(36, dtype=int)
    preds = teacher.copy(); preds[:6] = 1
    chosen = _calibrate_threshold(scores, preds, teacher, .8)
    eligible = []
    for threshold in np.unique(scores):
        mask = scores >= threshold
        n = int(mask.sum()); correct = int((preds[mask] == teacher[mask]).sum())
        if n >= 10 and _cp_lower(correct, n, .1) >= .8:
            eligible.append(mask.mean())
    assert chosen['coverage'] == max(eligible)
    assert chosen['teacher_agreement_lower'] >= .8


def test_seed_changes_split_but_same_seed_repeats():
    X = np.random.RandomState(4).normal(size=(240, 3))
    y = np.arange(len(X)) % 3
    opts = dict(targets=(.9,), max_fit_labels=120, skip=SKIP)
    _, a = fit_frontier(X, y, seed=17, **opts)
    _, b = fit_frontier(X, y, seed=17, **opts)
    _, c = fit_frontier(X, y, seed=93, **opts)
    np.testing.assert_array_equal(a['X_train'], b['X_train'])
    assert not np.array_equal(a['X_train'], c['X_train'])


@pytest.mark.parametrize('value', [.5, [[0, 1]], [0, np.nan], [np.inf, 0]])
def test_single_prediction_rejects_invalid_shapes_and_values(value):
    router = Router([], ['a'], SimpleNamespace(embedding_dim=2))
    with pytest.raises(ValueError, match='finite 1-D'):
        router.predict(value)


@pytest.mark.parametrize('value', [.5, [0, 1], [[[0, 1]]], [[0, np.nan]]])
def test_batch_prediction_rejects_invalid_shapes_and_values(value):
    router = Router([], ['a'], SimpleNamespace(embedding_dim=2))
    with pytest.raises(ValueError, match='finite 2-D'):
        router.predict_batch(value)


def test_router_reuses_ood_index(monkeypatch):
    X = np.random.RandomState(3).normal(size=(80, 3)).astype(np.float32)
    gate = fit_ood_gate(X, ['a'] * len(X))
    queries = np.vstack([X[:2], np.full((1, 3), 100)])
    expected = ood_mask(queries, X, ['a'] * len(queries), gate)
    router = Router([], ['a'], SimpleNamespace(embedding_dim=3), ood_gate=gate, train_embeddings=X)
    def unexpected_fit(*args, **kwargs):
        raise AssertionError('OOD index was rebuilt during prediction')
    monkeypatch.setattr(NearestNeighbors, 'fit', unexpected_fit)
    for _ in range(2):
        np.testing.assert_array_equal(router._ood_flags(queries, np.zeros(3)), expected)


def test_faiss_fit_reload_preserves_embedding_scale(tmp_path):
    pytest.importorskip('faiss')
    rng = np.random.RandomState(24)
    y = np.repeat(np.arange(3), 80)
    X = (rng.normal(size=(3, 8))[y] * 10 + rng.normal(size=(240, 8)) * .2).astype(np.float32)
    original = X.copy()
    traces = tmp_path / 'traces.jsonl'
    traces.write_text(''.join(json.dumps({'input': str(i), 'teacher': str(label)}) + '\n' for i, label in enumerate(y)))
    result = fit(traces, tmp_path / 'model', X, FitConfig(frontier_targets=(.9,), skip_candidates=SKIP, verbose=False))
    np.testing.assert_array_equal(X, original)
    np.testing.assert_array_equal(EmbeddingIndex.load(tmp_path / 'model' / 'index').embeddings, original)
    router = load_router(tmp_path / 'model')
    assert result.manifest.certification['teacher_agreement_lower'] >= .9
    # The reserved rows are no longer in the OOD reference; a small number of
    # distant but in-distribution examples may correctly be deferred.
    assert router.predict_batch(original)['handled'].mean() > .9


def test_fit_report_keeps_deferred_confidence(tmp_path, monkeypatch):
    # The lazy-import test reloads the package; patch the function used below.
    from tracer.api import fit

    base_X = np.array([[-3.], [-1.], [-.1], [.1], [1.], [3.]], dtype=np.float32)
    base_y = np.array([0, 0, 0, 1, 1, 1])
    clf = LogisticRegression().fit(base_X, base_y)
    # Enough accepted certification rows; the old six-row fixture cannot
    # support a 90% agreement lower bound regardless of observed accuracy.
    X, y = np.tile(base_X, (100, 1)), np.tile(base_y, 100)
    stage = {'clf': clf, 'accept_all': False, 'acceptor': None, 'threshold': .9}
    best = {'stages': [stage], 'summary': {'method': 'l2d', 'coverage_cal_total': .3,
                                          'teacher_agreement_cal_total': 1}}
    monkeypatch.setattr('tracer.api.fit_frontier', lambda *args, **kwargs:
                        ([{'target': .9, 'best': best, 'candidates': [best]}], {}))
    path = tmp_path / 'traces.jsonl'
    path.write_text(''.join(json.dumps({'input':str(i), 'teacher':str(label)})+'\n' for i,label in enumerate(y)))
    result = fit(path, tmp_path / 'out', X, FitConfig(frontier_targets=(.9,), verbose=False))
    assert result.qualitative_report.deferred_examples
    for example in result.qualitative_report.deferred_examples:
        expected = clf.predict_proba(X[int(example.input_preview):int(example.input_preview)+1]).max()
        assert example.accept_score == pytest.approx(expected)


def test_fit_config_seed_reaches_model(tmp_path):
    import joblib
    y = np.arange(160) % 2
    X = np.column_stack([y * 10., y * 10.]).astype(np.float32)
    path = tmp_path / 'traces.jsonl'
    path.write_text(''.join(json.dumps({'input':str(i), 'teacher':str(label)})+'\n' for i,label in enumerate(y)))
    result = fit(path, tmp_path / 'out', X,
                 FitConfig(seed=291, frontier_targets=(.9,), skip_candidates=SKIP, verbose=False))
    model = joblib.load(result.manifest.pipeline_path)['pipeline']['stages'][0]['clf']
    assert model.named_steps['clf'].random_state == 291


def test_null_teacher_error_uses_file_line(tmp_path):
    from tracer.traces.loader import load_traces
    path = tmp_path / 'traces.jsonl'
    path.write_text('{"input":"a","teacher":"yes"}\n\n\n{"input":"b","teacher":null}\n')
    with pytest.raises(ValueError, match='line 4'):
        load_traces(path)
