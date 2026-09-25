"""A stage's predictions must be teacher label ids, not predict_proba columns.

The RSB second stage is trained only on the rows the first stage rejected, so
its surrogate usually sees a subset of the label space (e.g. the easy class the
first stage fully absorbs is missing). For such a model, column ``i`` of
``predict_proba`` is ``clf.classes_[i]``, not label id ``i``.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from tracer.fit.pipeline import _predict, apply_stage

LINEAR_ONLY = ("gbt", "rf", "et", "xgb", "dt", "mlp_1h", "mlp_2h")


def test_predict_maps_proba_columns_to_class_labels():
    rng = np.random.RandomState(0)
    # Trained on label ids {1, 3} only (0 and 2 absent from this subset).
    X = np.vstack([rng.randn(50, 2) - 4, rng.randn(50, 2) + 4])
    y = np.r_[np.full(50, 1), np.full(50, 3)]
    clf = LogisticRegression().fit(X, y)

    preds, probs = _predict(clf, X)
    assert probs.shape == (100, 2)
    assert set(np.unique(preds)) == {1, 3}
    assert (preds == y).mean() > 0.99

    stage = {"clf": clf, "accept_all": True}
    stage_preds, _, _ = apply_stage(stage, X)
    assert set(np.unique(stage_preds)) == {1, 3}


def _residual_traces(tmp, n=4000):
    """Class 0 is trivially separable, so stage 1 accepts every class-0 row.

    Classes 1-3 come in two groups. The 'easy' group carries a per-class
    marker; the 'hard' group encodes the class along a direction that the easy
    group uses the opposite way, which a global linear surrogate cannot
    resolve. Those rows are rejected by stage 1 and are exactly what the RSB
    residual stage should pick up, with only labels {1, 2, 3} in its pool.
    """
    rng = np.random.RandomState(0)
    X, y = [], []
    for _ in range(n):
        c = rng.randint(0, 4)
        x = rng.randn(6) * 0.5
        if c == 0:
            x[0] += 10
        elif rng.rand() > 0.3:
            x[c] += 4
            x[5] -= (c - 2) * 8
            x[4] -= 3
        else:
            x[4] += 3
            x[5] += (c - 2) * 2
        X.append(x)
        y.append(c)
    X = np.asarray(X, dtype=np.float32)
    names = ["cls_0", "cls_1", "cls_2", "cls_3"]
    path = Path(tmp) / "traces.jsonl"
    with path.open("w") as f:
        for i, c in enumerate(y):
            f.write(json.dumps({"input": f"t{i}", "teacher": names[c]}) + "\n")
    return path, X, np.asarray([names[c] for c in y])


def test_rsb_residual_stage_routes_with_teacher_labels():
    from tracer.api import fit, load_router
    from tracer.config import FitConfig

    with tempfile.TemporaryDirectory() as tmp:
        path, X, teacher = _residual_traces(tmp)
        out = Path(tmp) / ".tracer"
        cfg = FitConfig(target_teacher_agreement=0.95, frontier_targets=(0.95,),
                        skip_candidates=LINEAR_ONLY, verbose=False)
        result = fit(path, artifact_dir=out, embeddings=X, config=cfg)

        # The residual stage is certified, so RSB beats single-stage L2D.
        assert result.manifest.selected_method == "rsb"
        assert result.manifest.coverage_cal > 0.95

        router = load_router(out)
        batch = router.predict_batch(X)
        stage2 = np.asarray(batch["stage_id"]) == 1
        assert stage2.sum() > 100
        labels = np.asarray(batch["labels"], dtype=object)[stage2]
        # Stage-2 rows carry the class they belong to, not a shifted label.
        assert (labels == teacher[stage2]).mean() >= 0.9
