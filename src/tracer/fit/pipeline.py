"""Parity-gated pipeline construction.

Three candidate families: global, l2d, rsb.
Acceptor features: top1-prob, top2-prob, margin, normalized-entropy.
Threshold calibration: sweep all unique scores, pick max coverage >= target TA.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tracer.fit.surrogate import search_best_surrogate


def _split_buffer(X: np.ndarray, y: np.ndarray, seed: int = 42):
    """Stratified train / val / cal split with fallback for tiny classes."""
    rng = np.random.RandomState(seed)
    train_idx, val_idx, cal_idx = [], [], []

    for cls in np.unique(y):
        ci = np.where(y == cls)[0]
        ci = rng.permutation(ci)
        n = len(ci)
        if n == 1:
            train_idx.extend(ci.tolist()); continue
        if n == 2:
            train_idx.append(int(ci[0])); cal_idx.append(int(ci[1])); continue
        if n == 3:
            train_idx.append(int(ci[0])); val_idx.append(int(ci[1]))
            cal_idx.append(int(ci[2])); continue
        nv = max(1, int(round(0.2 * n)))
        nc = max(1, int(round(0.2 * n)))
        if nv + nc >= n:
            nv, nc = 1, 1
        nt = n - nv - nc
        train_idx.extend(ci[:nt].tolist())
        val_idx.extend(ci[nt:nt + nv].tolist())
        cal_idx.extend(ci[nt + nv:].tolist())

    train_idx = np.array(sorted(train_idx), dtype=int)
    val_idx = np.array(sorted(val_idx), dtype=int)
    cal_idx = np.array(sorted(cal_idx), dtype=int)
    return {
        "X_train": X[train_idx], "y_train": y[train_idx],
        "X_val": X[val_idx], "y_val": y[val_idx],
        "X_cal": X[cal_idx], "y_cal": y[cal_idx],
        "n_fit": len(X),
    }


def _subsample(X, y, max_n, seed=42):
    if len(X) <= max_n:
        return X, y
    rng = np.random.RandomState(seed)
    keep = []
    classes = np.unique(y)
    per_cls = max(1, max_n // len(classes))
    for cls in classes:
        ci = np.where(y == cls)[0]
        keep.extend(rng.choice(ci, size=min(len(ci), per_cls), replace=False).tolist())
    keep = np.array(sorted(keep), dtype=int)
    return X[keep], y[keep]


def _accept_features(probs: np.ndarray) -> np.ndarray:
    sorted_p = -np.sort(-probs, axis=1)
    top1 = sorted_p[:, 0]
    top2 = sorted_p[:, 1] if probs.shape[1] > 1 else np.zeros(len(probs))
    margin = top1 - top2
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
    if probs.shape[1] > 1:
        entropy = entropy / np.log(probs.shape[1])
    else:
        entropy = np.zeros(len(probs))
    return np.column_stack([top1, top2, margin, entropy])


def _fit_acceptor(probs, y_pred, y_teacher):
    correct = (y_pred == y_teacher).astype(int)
    if len(np.unique(correct)) < 2:
        return None
    clf = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)),
    ])
    clf.fit(_accept_features(probs), correct)
    return clf


def _accept_scores(acceptor, probs):
    if acceptor is None:
        return probs.max(axis=1)
    return acceptor.predict_proba(_accept_features(probs))[:, 1]


def _calibrate_threshold(scores, preds, y_teacher, target_ta):
    thresholds = np.unique(np.sort(scores))
    chosen = None
    for t in thresholds:
        accept = scores >= t
        if not accept.any():
            continue
        agreement = float((preds[accept] == y_teacher[accept]).mean())
        coverage = float(accept.mean())
        if agreement >= target_ta:
            if chosen is None or coverage > chosen["coverage"]:
                chosen = {"threshold": float(t), "teacher_agreement": agreement,
                          "coverage": coverage}
    return chosen


def _predict(clf, X):
    probs = clf.predict_proba(X)
    preds = probs.argmax(axis=1).astype(int)
    return preds, probs


def build_global(split, target_ta):
    """Global pipeline: one surrogate, accept all if TA >= target."""
    if len(np.unique(split["y_train"])) < 2 or len(split["X_val"]) == 0:
        return {"method": "global", "stages": [], "summary": {"status": "insufficient_data", "coverage_cal_total": 0.0}}

    clf, name, val_m = search_best_surrogate(
        split["X_train"], split["y_train"], split["X_val"], split["y_val"])
    preds_cal = clf.predict(split["X_cal"])
    ta_cal = float((preds_cal == split["y_cal"]).mean()) if len(preds_cal) else 0.0
    if ta_cal < target_ta:
        return {"method": "global", "stages": [], "summary": {
            "status": "below_target", "model_name": name,
            "val_teacher_f1": val_m["teacher_f1"],
            "teacher_agreement_cal_total": ta_cal, "coverage_cal_total": 0.0}}

    stage = {"stage_name": "global", "model_name": name, "clf": clf,
             "accept_all": True, "teacher_agreement_cal": ta_cal, "coverage_cal": 1.0,
             "val_teacher_f1": val_m["teacher_f1"]}
    return {"method": "global", "stages": [stage], "summary": {
        "status": "ok", "method": "global", "model_name": name, "n_stages": 1,
        "target_teacher_agreement": float(target_ta),
        "teacher_agreement_cal_total": ta_cal, "coverage_cal_total": 1.0,
        "val_teacher_f1": val_m["teacher_f1"], "n_train_labels": split["n_fit"]}}


def _build_accepting_stage(X_tr, y_tr, X_val, y_val, X_cal, y_cal, target_ta, stage_name):
    if len(np.unique(y_tr)) < 2 or len(X_val) == 0 or len(X_cal) == 0:
        return None
    clf, name, val_m = search_best_surrogate(X_tr, y_tr, X_val, y_val)
    preds_val, probs_val = _predict(clf, X_val)
    preds_cal, probs_cal = _predict(clf, X_cal)
    acceptor = _fit_acceptor(probs_val, preds_val, y_val)
    scores_cal = _accept_scores(acceptor, probs_cal)
    ti = _calibrate_threshold(scores_cal, preds_cal, y_cal, target_ta)
    if ti is None:
        return None
    return {"stage_name": stage_name, "model_name": name, "clf": clf,
            "acceptor": acceptor, "accept_all": False, "threshold": ti["threshold"],
            "score_name": "predicted_teacher_match" if acceptor else "max_prob",
            "teacher_agreement_cal": ti["teacher_agreement"],
            "coverage_cal": ti["coverage"],
            "val_teacher_f1": val_m["teacher_f1"],
            "val_teacher_acc": val_m["teacher_acc"]}


def build_l2d(split, target_ta):
    """L2D: surrogate + acceptor-gated deferral."""
    s1 = _build_accepting_stage(
        split["X_train"], split["y_train"], split["X_val"], split["y_val"],
        split["X_cal"], split["y_cal"], target_ta, "stage_1")
    if s1 is None:
        return {"method": "l2d", "stages": [], "summary": {"status": "no_stage", "coverage_cal_total": 0.0}}
    stages = [s1]
    summary = _pipeline_cal_summary("l2d", stages, split["X_cal"], split["y_cal"])
    summary["target_teacher_agreement"] = float(target_ta)
    summary["stage_1"] = {"model_name": s1["model_name"], "coverage_cal": s1["coverage_cal"],
                          "teacher_agreement_cal": s1["teacher_agreement_cal"],
                          "val_teacher_f1": s1["val_teacher_f1"]}
    summary["n_train_labels"] = split["n_fit"]
    return {"method": "l2d", "stages": stages, "summary": summary}


def build_rsb(split, target_ta):
    """RSB: residual two-stage cascade."""
    s1 = _build_accepting_stage(
        split["X_train"], split["y_train"], split["X_val"], split["y_val"],
        split["X_cal"], split["y_cal"], target_ta, "stage_1")
    if s1 is None:
        return {"method": "rsb", "stages": [], "summary": {"status": "no_stage", "coverage_cal_total": 0.0}}
    stages = [s1]

    _, accept_tr, _ = apply_stage(s1, split["X_train"])
    _, accept_val, _ = apply_stage(s1, split["X_val"])
    _, accept_cal, _ = apply_stage(s1, split["X_cal"])
    rej_tr, rej_val, rej_cal = ~accept_tr, ~accept_val, ~accept_cal

    if (rej_tr.sum() >= 50 and rej_val.sum() >= 20 and rej_cal.sum() >= 20
            and len(np.unique(split["y_train"][rej_tr])) >= 2):
        s2 = _build_accepting_stage(
            split["X_train"][rej_tr], split["y_train"][rej_tr],
            split["X_val"][rej_val], split["y_val"][rej_val],
            split["X_cal"][rej_cal], split["y_cal"][rej_cal],
            target_ta, "stage_2")
        if s2 is not None:
            stages.append(s2)

    summary = _pipeline_cal_summary("rsb", stages, split["X_cal"], split["y_cal"])
    summary["target_teacher_agreement"] = float(target_ta)
    summary["stage_1"] = {"model_name": s1["model_name"], "coverage_cal": s1["coverage_cal"],
                          "teacher_agreement_cal": s1["teacher_agreement_cal"],
                          "val_teacher_f1": s1["val_teacher_f1"]}
    if len(stages) == 2:
        summary["stage_2"] = {"model_name": stages[1]["model_name"],
                              "coverage_cal": stages[1]["coverage_cal"],
                              "teacher_agreement_cal": stages[1]["teacher_agreement_cal"],
                              "val_teacher_f1": stages[1]["val_teacher_f1"]}
    summary["n_train_labels"] = split["n_fit"]
    return {"method": "rsb", "stages": stages, "summary": summary}


def apply_stage(stage: dict, X: np.ndarray):
    """Apply a single stage: returns (preds, accept_mask, scores)."""
    preds, probs = _predict(stage["clf"], X)
    if stage.get("accept_all"):
        return preds, np.ones(len(X), dtype=bool), np.ones(len(X))
    scores = _accept_scores(stage.get("acceptor"), probs)
    accept = scores >= stage["threshold"]
    return preds, accept, scores


def route_pipeline(stages: list, X: np.ndarray):
    """Route samples through a multi-stage pipeline."""
    n = len(X)
    preds = np.full(n, -1, dtype=int)
    handled = np.zeros(n, dtype=bool)
    stage_id = np.full(n, -1, dtype=int)
    remaining = np.arange(n)

    for idx, stage in enumerate(stages):
        if len(remaining) == 0:
            break
        sp, sa, _ = apply_stage(stage, X[remaining])
        if sa.any():
            sel = remaining[sa]
            preds[sel] = sp[sa]
            handled[sel] = True
            stage_id[sel] = idx
        remaining = remaining[~sa]
    return preds, handled, stage_id


def evaluate_pipeline(pipeline: dict, X, y_true, y_teacher):
    """Evaluate a pipeline on held-out data."""
    preds, handled, stage_id = route_pipeline(pipeline["stages"], X)
    deferred = ~handled
    final = preds.copy()
    final[deferred] = y_teacher[deferred]
    return {
        "coverage": float(handled.mean()),
        "teacher_agreement_handled": float((preds[handled] == y_teacher[handled]).mean()) if handled.any() else 0.0,
        "gt_f1_handled": float(f1_score(y_true[handled], preds[handled], average="macro", zero_division=0)) if handled.any() else 0.0,
        "gt_acc_handled": float(accuracy_score(y_true[handled], preds[handled])) if handled.any() else 0.0,
        "e2e_gt_acc": float(accuracy_score(y_true, final)),
        "e2e_gt_f1": float(f1_score(y_true, final, average="macro", zero_division=0)),
        "e2e_teacher_agreement": float((final == y_teacher).mean()),
        "n_deferred": int(deferred.sum()),
        "preds": preds,
        "handled": handled,
        "stage_id": stage_id,
    }


def _pipeline_cal_summary(method, stages, X_cal, y_cal):
    preds, handled, _ = route_pipeline(stages, X_cal)
    ta = float((preds[handled] == y_cal[handled]).mean()) if handled.any() else 0.0
    return {"method": method, "status": "ok" if handled.any() else "no_accepted",
            "n_stages": len(stages), "coverage_cal_total": float(handled.mean()),
            "teacher_agreement_cal_total": ta}


def fit_frontier(X, y_teacher, targets, max_fit_labels=8000, min_coverage=0.05):
    """Build global/l2d/rsb for each target TA, return best per target."""
    X_fit, y_fit = _subsample(X, y_teacher, max_fit_labels)
    split = _split_buffer(X_fit, y_fit)
    builders = {"global": build_global, "l2d": build_l2d, "rsb": build_rsb}
    frontier = []

    for target in sorted(set(float(t) for t in targets)):
        candidates = []
        for method_name, builder in builders.items():
            pipeline = builder(split, target)
            pipeline["summary"]["method"] = method_name
            if pipeline["summary"].get("coverage_cal_total", 0.0) < min_coverage:
                if pipeline["summary"].get("status") == "ok":
                    pipeline["summary"]["status"] = "below_min_coverage"
                pipeline["stages"] = []
            candidates.append(pipeline)

        deployable = [c for c in candidates
                      if c["summary"].get("status") == "ok"
                      and c["summary"].get("teacher_agreement_cal_total", 0.0) >= target
                      and c["summary"].get("coverage_cal_total", 0.0) >= min_coverage]
        best = None
        if deployable:
            best = max(deployable, key=lambda c: (
                c["summary"]["coverage_cal_total"],
                c["summary"]["teacher_agreement_cal_total"],
                -c["summary"]["n_stages"]))
        frontier.append({"target": target, "candidates": candidates, "best": best})
    return frontier, split
