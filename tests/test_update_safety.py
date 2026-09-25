"""Refits must preserve the previous generation until a replacement is ready."""
import json
from pathlib import Path

import numpy as np
import pytest

from tracer.api import fit, update, load_router
from tracer.config import FitConfig
from tracer.traces.loader import load_traces

SKIP = ('logreg_c10', 'sgd_log', 'mlp_1h', 'mlp_2h', 'dt', 'rf', 'et', 'gbt', 'xgb')


def write_traces(path, n):
    y = np.arange(n) % 2
    path.write_text(''.join(json.dumps({'input': f'row {i}', 'teacher': str(label)}) + '\n'
                            for i, label in enumerate(y)))
    rng = np.random.RandomState(14)
    return (np.column_stack([y * 8, y * 8]) + rng.normal(0, .1, (n, 2))).astype(np.float32)


@pytest.fixture
def artifacts(tmp_path):
    traces = tmp_path / 'initial.jsonl'
    X = write_traces(traces, 160)
    output = tmp_path / 'artifacts'
    fit(traces, output, X, FitConfig(frontier_targets=(.9,), skip_candidates=SKIP,
                                    seed=19, verbose=False))
    (output / 'user-notes.txt').write_text('keep me')
    return output, X


def snapshot(directory):
    return {str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob('*') if p.is_file()}


@pytest.mark.parametrize('new_X', [np.zeros((1, 2)), np.zeros((2, 3)),
                                  np.zeros(2), np.full((2, 2), np.nan)])
def test_invalid_update_leaves_all_artifacts_unchanged(artifacts, tmp_path, new_X):
    output, _ = artifacts
    traces = tmp_path / 'new.jsonl'
    write_traces(traces, 2)
    before = snapshot(output)
    with pytest.raises(ValueError):
        update(traces, output, new_embeddings=new_X)
    assert snapshot(output) == before
    # A corrected retry must still work and append exactly once.
    result = update(traces, output, new_embeddings=write_traces(traces, 2))
    assert result.manifest.n_traces == 162
    assert len(load_traces(output / 'all_traces.jsonl')) == 162


def test_refit_failure_leaves_original_generation(artifacts, tmp_path, monkeypatch):
    # The lazy-import test reloads the package; patch the function used below.
    from tracer.api import update

    output, _ = artifacts
    traces = tmp_path / 'new.jsonl'
    X = write_traces(traces, 2)
    before = snapshot(output)

    def fail_after_writing(trace_path, artifact_dir, **kwargs):
        (artifact_dir / 'pipeline.joblib').write_bytes(b'partial model')
        raise OSError('disk write failed')

    monkeypatch.setattr('tracer.api.fit', fail_after_writing)
    with pytest.raises(OSError, match='disk write failed'):
        update(traces, output, new_embeddings=X)
    assert snapshot(output) == before


def test_publish_failure_restores_original_generation(artifacts, tmp_path, monkeypatch):
    output, _ = artifacts
    traces = tmp_path / 'new.jsonl'
    X = write_traces(traces, 2)
    before = snapshot(output)
    rename = Path.rename

    def fail_publish(path, target):
        if path.name == 'next':
            raise OSError('publish failed')
        return rename(path, target)

    monkeypatch.setattr(Path, 'rename', fail_publish)
    with pytest.raises(OSError, match='publish failed'):
        update(traces, output, new_embeddings=X)
    assert snapshot(output) == before


def test_explicit_config_is_honored_and_not_mutated(artifacts, tmp_path):
    output, _ = artifacts
    traces = tmp_path / 'new.jsonl'
    X = write_traces(traces, 2)
    config = FitConfig(target_teacher_agreement=.99, frontier_targets=(.99,),
                       skip_candidates=SKIP, seed=27, verbose=False)
    result = update(traces, output, new_embeddings=X, config=config)
    assert config.target_teacher_agreement == .99
    assert result.manifest.target_teacher_agreement == .99
    assert result.manifest.n_retrains == 2
    assert json.loads((output / 'config.json').read_text())['seed'] == 27
    # A strict refit with no deployable policy must not keep the old model.
    if result.manifest.selected_method is None:
        assert not (output / 'pipeline.joblib').exists()
        assert not (output / 'ood.json').exists()


def test_default_update_preserves_config_and_returns_live_paths(artifacts, tmp_path):
    output, X_old = artifacts
    traces = tmp_path / 'new.jsonl'
    X = write_traces(traces, 10)
    result = update(traces, output, new_embeddings=X)
    saved = json.loads((output / 'config.json').read_text())
    assert saved['seed'] == 19
    assert saved['skip_candidates'] == list(SKIP)
    assert result.manifest.n_retrains == 2
    assert (output / 'user-notes.txt').read_text() == 'keep me'
    for name in ('pipeline_path', 'index_path', 'config_path', 'qualitative_report_path'):
        value = getattr(result.manifest, name)
        if value is not None:
            assert Path(value).parent == output
            if name != 'index_path':
                assert Path(value).exists()
    np.testing.assert_allclose(np.load(output / 'index.embeddings.npy'), np.vstack([X_old, X]))
    assert load_router(output).predict(X_old[0])['decision'] == 'handled'


def test_failed_rollback_keeps_recoverable_backup(artifacts, tmp_path, monkeypatch):
    output, _ = artifacts
    traces = tmp_path / 'new.jsonl'
    X = write_traces(traces, 2)
    before = snapshot(output)
    rename = Path.rename
    def fail_publish_and_rollback(path, target):
        if path.name == 'next' or path.name.endswith('-previous'):
            raise OSError('destination unavailable')
        return rename(path, target)
    monkeypatch.setattr(Path, 'rename', fail_publish_and_rollback)
    with pytest.raises(RuntimeError, match='preserved at'):
        update(traces, output, new_embeddings=X)
    backups = list(tmp_path.glob('.artifacts-update-*-previous'))
    assert len(backups) == 1
    assert snapshot(backups[0]) == before
