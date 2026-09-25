"""Small deterministic routing ablation; no model downloads or API calls.

Run with `python benchmarks/routing_reliability.py`. Install the optional faiss
extra to include the fit/reload case. Output is JSON suitable for comparing two
checkouts with the same Python and dependencies.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from tracer.api import fit, load_router
from tracer.config import FitConfig
from tracer.fit.pipeline import _calibrate_threshold, _predict


def run():
    scores = np.linspace(.50, .99, 36)
    teacher = np.zeros(36, dtype=int)
    predictions = teacher.copy()
    predictions[:6] = 1
    chosen = _calibrate_threshold(scores, predictions, teacher, .8)
    X = np.array([[-3.], [-2.], [-1.], [1.], [2.], [3.]])
    y = np.array([1, 1, 1, 3, 3, 3])
    model = LogisticRegression(random_state=42).fit(X, y)
    results = {
        'small_calibration': {key: chosen[key] for key in
                              ('coverage', 'teacher_agreement', 'teacher_agreement_lower')},
        'subset_label_agreement': float((_predict(model, X)[0] == y).mean()),
    }
    try:
        import faiss
    except ImportError:
        results['fit_reload'] = 'skipped: install tracer-llm[faiss]'
        return results

    rng = np.random.RandomState(24)
    centers = rng.normal(size=(3, 8)) * 10
    y_train = np.repeat(np.arange(3), 80)
    y_test = np.repeat(np.arange(3), 200)
    train = (centers[y_train] + rng.normal(size=(240, 8)) * .2).astype(np.float32)
    heldout = (centers[y_test] + rng.normal(size=(600, 8)) * .2).astype(np.float32)
    original = train.copy()
    skip = ('logreg_c10', 'sgd_log', 'mlp_1h', 'mlp_2h', 'dt', 'rf', 'et', 'gbt', 'xgb')
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        traces = root / 'traces.jsonl'
        traces.write_text(''.join(json.dumps({'input': str(i), 'teacher': str(label)}) + '\n'
                                 for i, label in enumerate(y_train)))
        fitted = fit(traces, root / 'model', train,
                     FitConfig(frontier_targets=(.9,), skip_candidates=skip, verbose=False))
        router = load_router(root / 'model')
        output = router.predict_batch(heldout)
        accepted = output['handled']
        labels = np.asarray(output['labels'])
        results['fit_reload'] = {
            'faiss_version': faiss.__version__,
            'training_input_unchanged': bool(np.array_equal(train, original)),
            'calibration_coverage': fitted.manifest.coverage_cal,
            'heldout_coverage': float(accepted.mean()),
            'heldout_teacher_agreement': float((labels[accepted] == y_test[accepted].astype(str)).mean()) if accepted.any() else None,
            'heldout_rows': len(y_test),
        }
    return results


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
