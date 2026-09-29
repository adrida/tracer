"""One-time evaluation of a fixed serving policy on untouched examples.

The binomial confidence statement assumes independent examples drawn from the
same distribution as future requests. It is not a guarantee under distribution
shift, dependent sessions, or repeated adaptive reuse of the holdout.
"""

from __future__ import annotations

import numpy as np

from tracer.fit.pipeline import _cp_lower


def certification_split(n: int, fraction: float, seed: int):
    """Random row split, without stratification or label-dependent balancing.

Reserve before any subsampling so certification retains traffic prevalence.
Never put a reserved example back into development to rescue a rare class.
"""
    order = np.random.RandomState(seed).permutation(n)
    n_cert = min(n, max(1, int(np.ceil(n * fraction)))) if n else 0
    return np.sort(order[n_cert:]), np.sort(order[:n_cert])


def certify_policy(router, X, y, target: float, alpha: float) -> dict:
    """Check exactly one policy, including its OOD filter, with no retries."""
    n = len(X)
    if n:
        routed = router.predict_batch(X)
        handled = np.asarray(routed["handled"], dtype=bool)
        n_accepted = int(handled.sum())
        n_correct = int((routed["preds"][handled] == np.asarray(y)[handled]).sum())
    else:
        n_accepted = n_correct = 0
    lower = _cp_lower(n_correct, n_accepted, alpha)
    return {
        "status": "certified" if n_accepted > 0 and lower >= target else "failed",
        "method": "untouched_final_policy_clopper_pearson",
        "alpha": float(alpha),
        "target": float(target),
        "n_examples": n,
        "n_accepted": n_accepted,
        "n_correct": n_correct,
        "coverage": n_accepted / n if n else 0.0,
        "teacher_agreement": n_correct / n_accepted if n_accepted else None,
        "teacher_agreement_lower": lower,
    }
