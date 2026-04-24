"""Model zoo for surrogate classifiers.

Trains all candidates and selects the best by teacher-label macro-F1
on the validation split.  Tree-based models skip StandardScaler;
linear/neural models use it.

Optional dependency: xgboost (skipped silently if not installed).
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Callable, Iterable, Optional, Tuple

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

SEED = 42


def _candidates(n_samples: int, skip: Iterable[str] = ()) -> dict:
    """Return model factory dict.  Factories are callables → fitted-ready estimator.

    Pass ``skip=("name", ...)`` to exclude candidates from the sweep — useful
    when one model is known to dominate wall-time on the target data (e.g.
    ``skip=("gbt",)`` on large multi-class problems).
    """
    skip_set = set(skip)

    # ── linear / logistic ────────────────────────────────────────────────────
    linear = {
        "logreg_c1": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                                       random_state=SEED)),
        ]),
        "logreg_c10": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=10.0, max_iter=1000, solver="lbfgs",
                                       random_state=SEED)),
        ]),
        "sgd_log": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("clf", SGDClassifier(loss="log_loss", max_iter=200, tol=1e-3,
                                  random_state=SEED)),
        ]),
    }

    # ── neural ───────────────────────────────────────────────────────────────
    neural = {
        "mlp_1h": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-4,
                                  max_iter=140, early_stopping=True,
                                  random_state=SEED)),
        ]),
        "mlp_2h": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(256, 96), alpha=1e-4,
                                  max_iter=140, early_stopping=True,
                                  random_state=SEED)),
        ]),
    }

    # ── tree-based (no scaling needed) ────────────────────────────────────────
    trees = {
        "dt": lambda: DecisionTreeClassifier(
            max_depth=16, min_samples_leaf=2, random_state=SEED),
        "rf": lambda: RandomForestClassifier(
            n_estimators=200, max_features="sqrt",
            min_samples_leaf=2, n_jobs=-1, random_state=SEED),
        "et": lambda: ExtraTreesClassifier(
            n_estimators=200, max_features="sqrt",
            min_samples_leaf=2, n_jobs=-1, random_state=SEED),
    }

    # GradientBoosting is slow on large datasets -- skip above 4k
    if n_samples <= 4_000:
        trees["gbt"] = lambda: GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=SEED)

    # ── xgboost (optional) ───────────────────────────────────────────────────
    try:
        from xgboost import XGBClassifier  # type: ignore
        # Probe that the native library actually loads before registering
        XGBClassifier()
        trees["xgb"] = lambda: XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            n_jobs=-1, random_state=SEED, verbosity=0,
        )
    except Exception:
        pass

    merged = {**linear, **neural, **trees}
    if skip_set:
        merged = {k: v for k, v in merged.items() if k not in skip_set}
    return merged


def _invoke_on_candidate(cb, name: str, val_f1: float, elapsed: float) -> None:
    """Call ``cb(name, val_f1[, elapsed])`` tolerating either signature.

    Older callers expected ``(name, val_f1)`` and we do not want to break
    them. Newer code can accept the optional elapsed seconds. A raising
    callback is swallowed (with a warning) so it can't derail a fit.
    """
    if cb is None:
        return
    try:
        try:
            cb(name, val_f1, elapsed)
        except TypeError:
            cb(name, val_f1)
    except Exception as e:  # pragma: no cover — defensive
        warnings.warn(f"on_candidate callback raised: {e!r}", RuntimeWarning)


def search_best_surrogate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    on_candidate: Optional[Callable[..., None]] = None,
    skip: Iterable[str] = (),
) -> Tuple[Any, str, dict]:
    """Train all candidates and return (best_clf, model_name, metrics).

    Parameters
    ----------
    X_train, y_train : training split
    X_val, y_val     : validation split (used for model selection only)
    on_candidate     : optional callback invoked once per candidate after the
                       fit+predict completes. Accepts either
                       ``cb(name, val_f1)`` (legacy) or
                       ``cb(name, val_f1, elapsed_seconds)`` (preferred).
                       Exceptions raised by the callback are warned and
                       swallowed — they do not abort the sweep.
    skip             : candidate names to omit from the sweep (forwarded to
                       ``_candidates``). Useful for pruning models that
                       dominate wall-time, e.g. ``skip=("gbt",)``.
    """
    n = len(X_train)
    candidates = _candidates(n, skip=skip)

    best_clf     = None
    best_name    = None
    best_metrics = None

    for name, factory in candidates.items():
        t0 = time.perf_counter()
        clf = factory()
        try:
            clf.fit(X_train, y_train)
        except (ValueError, np.linalg.LinAlgError, Exception):
            continue

        # Tree-based models don't have predict_proba by default for DT --
        # all sklearn trees support it, so this is always fine.
        try:
            preds = clf.predict(X_val)
        except Exception:
            continue

        val_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))
        val_acc = float(accuracy_score(y_val, preds))
        elapsed = time.perf_counter() - t0

        metrics = {"teacher_f1": val_f1, "teacher_acc": val_acc, "fit_seconds": elapsed}

        _invoke_on_candidate(on_candidate, name, val_f1, elapsed)

        if best_clf is None or val_f1 > best_metrics["teacher_f1"]:
            best_clf     = clf
            best_name    = name
            best_metrics = metrics

    return best_clf, best_name, best_metrics
