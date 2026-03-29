"""Qualitative reporting for TRACER routing policies.

Grounded in XAI literature:
- Slice-level summaries (Slice Finder, Google 2019)
- Representative example cards (influence functions, Koh & Liang 2017)
- Contrastive boundary pairs (counterfactual explanations, Wachter et al 2017)
- Temporal deltas across retrains (Dataset Cartography, Swayamdipta et al 2020)
"""

from __future__ import annotations

from collections import Counter
from statistics import median
from typing import List, Optional

import numpy as np

from tracer.types import (
    BoundaryPair, QualitativeReport, RepresentativeExample,
    SliceInsight, TemporalDelta,
)


def _preview(text: str, limit: int = 160) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit - 3] + "..."


def _length_bin(length: int, cuts) -> str:
    if length <= cuts[0]:
        return "short"
    if length <= cuts[1]:
        return "medium"
    return "long"


def _select_representatives(
    texts: List[str], labels: List[str], decisions: List[str],
    local_labels: Optional[List[Optional[str]]], scores: Optional[np.ndarray],
    trace_ids: Optional[List[Optional[str]]], decision_filter: str, top_k: int,
) -> List[RepresentativeExample]:
    """Select representative examples near the median length."""
    indices = [i for i, d in enumerate(decisions) if d == decision_filter]
    if not indices:
        return []
    lengths = [len(texts[i]) for i in indices]
    target = median(lengths)
    ranked = sorted(indices, key=lambda i: abs(len(texts[i]) - target))
    cards = []
    for i in ranked[:top_k]:
        cards.append(RepresentativeExample(
            input_preview=_preview(texts[i]),
            teacher_label=labels[i],
            decision=decision_filter,
            local_label=local_labels[i] if local_labels else None,
            accept_score=float(scores[i]) if scores is not None else None,
            trace_id=trace_ids[i] if trace_ids else None,
        ))
    return cards


def build_qualitative_report(
    texts: List[str],
    teacher_labels: List[str],
    decisions: List[str],
    local_labels: Optional[List[Optional[str]]] = None,
    accept_scores: Optional[np.ndarray] = None,
    trace_ids: Optional[List[Optional[str]]] = None,
    previous_decisions: Optional[List[str]] = None,
    previous_teacher_labels: Optional[List[str]] = None,
    top_k: int = 5,
) -> QualitativeReport:
    """Build a grounded qualitative report from routing outputs.

    Parameters
    ----------
    texts : input texts for each sample
    teacher_labels : teacher label for each sample
    decisions : "handled" or "deferred" for each sample
    local_labels : surrogate prediction for each sample (optional)
    accept_scores : acceptor confidence for each sample (optional)
    trace_ids : trace identifiers (optional)
    previous_decisions : decisions from a prior fit (for temporal deltas)
    previous_teacher_labels : teacher labels from a prior fit
    top_k : number of representative examples per group
    """
    n = len(texts)
    n_handled = sum(1 for d in decisions if d == "handled")
    n_deferred = n - n_handled
    coverage = n_handled / max(n, 1)

    ta_handled = 0.0
    if local_labels and n_handled > 0:
        agree = sum(1 for i in range(n)
                    if decisions[i] == "handled"
                    and local_labels[i] == teacher_labels[i])
        ta_handled = agree / n_handled

    # -- Length-bin slices --
    lengths = sorted(len(t) for t in texts)
    cuts = (lengths[len(lengths) // 3], lengths[2 * len(lengths) // 3])
    slices: List[SliceInsight] = []

    for bucket in ("short", "medium", "long"):
        idx = [i for i in range(n) if _length_bin(len(texts[i]), cuts) == bucket]
        if not idx:
            continue
        h = sum(1 for i in idx if decisions[i] == "handled")
        lbl_counts = Counter(teacher_labels[i] for i in idx)
        dominant = lbl_counts.most_common(1)[0][0]
        slices.append(SliceInsight(
            slice_name=f"length:{bucket}", predicate=f"length_bin == {bucket}",
            count=len(idx), handled_rate=h / len(idx),
            deferred_rate=1.0 - h / len(idx), dominant_teacher_label=dominant))

    # -- Label-level slices --
    label_groups = {}
    for i in range(n):
        label_groups.setdefault(teacher_labels[i], []).append(i)

    for label, idx in sorted(label_groups.items(), key=lambda x: -len(x[1]))[:15]:
        h = sum(1 for i in idx if decisions[i] == "handled")
        ta = None
        if local_labels:
            agree = sum(1 for i in idx if decisions[i] == "handled"
                        and local_labels[i] == teacher_labels[i])
            ta = agree / max(h, 1) if h > 0 else None
        slices.append(SliceInsight(
            slice_name=f"label:{label}", predicate=f"teacher_label == {label}",
            count=len(idx), handled_rate=h / len(idx),
            deferred_rate=1.0 - h / len(idx),
            teacher_agreement_handled=ta, dominant_teacher_label=label))

    # -- Representative examples --
    handled_examples = _select_representatives(
        texts, teacher_labels, decisions, local_labels,
        accept_scores, trace_ids, "handled", top_k)
    deferred_examples = _select_representatives(
        texts, teacher_labels, decisions, local_labels,
        accept_scores, trace_ids, "deferred", top_k)

    # -- Boundary pairs --
    boundary_pairs: List[BoundaryPair] = []
    for label in sorted(label_groups):
        h_idx = [i for i in label_groups[label] if decisions[i] == "handled"]
        d_idx = [i for i in label_groups[label] if decisions[i] == "deferred"]
        if not h_idx or not d_idx:
            continue
        hi = h_idx[len(h_idx) // 2]
        di = d_idx[len(d_idx) // 2]
        boundary_pairs.append(BoundaryPair(
            handled_preview=_preview(texts[hi]),
            deferred_preview=_preview(texts[di]),
            teacher_label=label,
            handled_score=float(accept_scores[hi]) if accept_scores is not None else None,
            deferred_score=float(accept_scores[di]) if accept_scores is not None else None,
        ))
        if len(boundary_pairs) >= top_k:
            break

    # -- Temporal deltas --
    temporal_deltas: List[TemporalDelta] = []
    if previous_decisions and previous_teacher_labels:
        prev_groups = {}
        for i, lbl in enumerate(previous_teacher_labels):
            prev_groups.setdefault(lbl, []).append(i)
        for label in sorted(set(prev_groups) & set(label_groups)):
            prev_idx = prev_groups[label]
            cur_idx = label_groups[label]
            prev_rate = sum(1 for i in prev_idx if previous_decisions[i] == "handled") / max(len(prev_idx), 1)
            cur_rate = sum(1 for i in cur_idx if decisions[i] == "handled") / max(len(cur_idx), 1)
            temporal_deltas.append(TemporalDelta(
                label=label, previous_handled_rate=prev_rate,
                current_handled_rate=cur_rate, delta=cur_rate - prev_rate))
        temporal_deltas.sort(key=lambda x: abs(x.delta), reverse=True)
        temporal_deltas = temporal_deltas[:top_k]

    summary = (f"Handled {n_handled}/{n} ({coverage:.1%}) by surrogate; "
               f"deferred {n_deferred}/{n} ({1-coverage:.1%}).")

    return QualitativeReport(
        summary=summary, coverage=coverage,
        teacher_agreement_handled=ta_handled,
        slices=slices, handled_examples=handled_examples,
        deferred_examples=deferred_examples,
        boundary_pairs=boundary_pairs, temporal_deltas=temporal_deltas)
