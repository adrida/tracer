"""Distance-based out-of-distribution (OOD) gate.

The parity gate certifies how often the surrogate agrees with the teacher *on
traffic that looks like the calibration data*. It says nothing about inputs that
fall outside that distribution: an off-topic query, gibberish, a prompt-injection
string, or a different-domain item can still get a confident surrogate prediction
and slip through. This gate is the safety net: at fit time it measures how far
in-distribution training points sit from their neighbours, and at inference it
defers any query that lands further out than that, regardless of surrogate
confidence.

Mechanism (deliberately commodity): mean distance to the k nearest training
neighbours, with a 95th-percentile threshold taken globally and per predicted
label (global fallback for sparse labels). This is standard kNN-distance OOD
detection. It is intentionally NOT keyed on the partition cells, the cell
construction is out of scope here; this gate only needs the input embeddings and
the surrogate's predicted label.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def fit_ood_gate(X_train: np.ndarray, pred_labels, k: int = 10,
                 quantile: float = 0.995) -> Optional[dict]:
    """Calibrate the OOD thresholds on the training embeddings.

    Returns a serialisable dict, or None if there is too little data to calibrate.
    pred_labels: the surrogate's predicted label (string) for each training row,
    used to key per-label thresholds (with a global fallback).
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    n = len(X_train)
    if n < 20:
        return None
    from sklearn.neighbors import NearestNeighbors
    k_eff = int(min(k, n - 1))
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(X_train)
    d, _ = nn.kneighbors(X_train)
    mean_d = d[:, 1:].mean(axis=1)  # drop self (col 0)
    global_thr = float(np.quantile(mean_d, quantile))
    labels = np.asarray([str(l) for l in pred_labels])
    per_label: dict = {}
    for lab in np.unique(labels):
        m = labels == lab
        if int(m.sum()) >= 10:
            per_label[str(lab)] = float(np.quantile(mean_d[m], quantile))
    return {"k": k_eff, "quantile": float(quantile),
            "global_thr": global_thr, "per_label_thr": per_label}


def ood_mask(X_query: np.ndarray, X_train: np.ndarray, query_labels, gate, nn=None) -> np.ndarray:
    """Boolean array, True where a query is out-of-distribution (should defer).

    Uses the same mean-k-NN-distance rule the gate was calibrated with, comparing
    against the per-predicted-label threshold (global fallback)."""
    X_query = np.asarray(X_query, dtype=np.float32)
    if gate is None or X_train is None or len(X_train) == 0 or len(X_query) == 0:
        return np.zeros(len(X_query), dtype=bool)
    
    k = int(min(gate.get("k", 10), len(X_train)))
    if k < 1:
        return np.zeros(len(X_query), dtype=bool)

    if nn is not None:
        if getattr(nn, "n_neighbors", None) != k:
            nn = None

    if nn is None:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k).fit(np.asarray(X_train, dtype=np.float32))

    d, _ = nn.kneighbors(X_query)
    mean_d = d.mean(axis=1)
    per = gate.get("per_label_thr", {})
    g = float(gate.get("global_thr", np.inf))
    # Use the looser of the per-label and global thresholds: a safety net should
    # only defer clearly-far inputs, so per-label may loosen the bar for labels
    # with a naturally wide neighbourhood but never tighten it below global
    # (which would over-defer in-distribution traffic that merely phrases things
    # differently from the training sample).
    thr = np.array([max(float(per.get(str(l), g)), g) for l in query_labels], dtype=float)
    return mean_d > thr
