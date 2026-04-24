"""Public API for TRACER.

    import tracer
    result = tracer.fit("traces.jsonl", ".tracer")
    router = tracer.load_router(".tracer")
    result = tracer.update("new_traces.jsonl", ".tracer")
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Union

import joblib
import numpy as np

from tracer.analysis.qualitative import build_qualitative_report
from tracer.config import FitConfig
from tracer.embeddings.index import EmbeddingIndex
from tracer.fit.pipeline import (
    evaluate_pipeline, fit_frontier, route_pipeline, apply_stage, _accept_scores, _predict,
)
from tracer.policy.artifacts import (
    load_manifest, load_pipeline, save_pipeline, save_qualitative_report, write_manifest,
)
from tracer.runtime.router import Router
from tracer.traces.loader import load_traces
from tracer.types import ArtifactManifest, FitResult, QualitativeReport


def fit(
    trace_path: Union[str, Path],
    artifact_dir: Union[str, Path] = ".tracer",
    embeddings: Optional[np.ndarray] = None,
    config: Optional[FitConfig] = None,
) -> FitResult:
    """Fit a TRACER routing policy from teacher traces.

    Parameters
    ----------
    trace_path : path to a JSONL trace file
    artifact_dir : output directory for .tracer artifacts
    embeddings : precomputed embedding matrix (n_traces x dim).
                 If None, looks for a .npy file next to trace_path.
    config : fitting configuration (defaults are sensible)

    Returns
    -------
    FitResult with manifest, qualitative report, and notes.
    """
    trace_path = Path(trace_path)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    config = config or FitConfig()
    notes = []

    dataset = load_traces(trace_path)
    labels = sorted(dataset.label_space)
    label_to_idx = {l: i for i, l in enumerate(labels)}

    # Resolve embeddings
    if embeddings is None:
        emb_path = trace_path.with_suffix(".npy")
        if not emb_path.exists():
            emb_path = trace_path.parent / (trace_path.stem + "_embeddings.npy")
        if emb_path.exists():
            embeddings = np.load(emb_path)
            notes.append(f"Loaded embeddings from {emb_path.name}")
        else:
            raise FileNotFoundError(
                f"No embeddings found. Pass embeddings= or place a .npy file at {emb_path}")

    if len(embeddings) != len(dataset):
        raise ValueError(f"Trace/embedding mismatch: {len(dataset)} traces vs {len(embeddings)} embeddings")

    X = embeddings.astype(np.float32, copy=False)
    y_teacher = np.array([label_to_idx[r.teacher_label] for r in dataset.records], dtype=int)
    y_true = None
    has_gt = all(r.ground_truth is not None for r in dataset.records)
    if has_gt:
        y_true = np.array([label_to_idx.get(r.ground_truth, -1) for r in dataset.records], dtype=int)
        valid = y_true >= 0
        if not valid.all():
            y_true = None
            has_gt = False

    # Fit frontier
    targets = list(config.frontier_targets)
    if config.target_teacher_agreement not in targets:
        targets.append(config.target_teacher_agreement)

    log_fn: Optional[Callable[[str], None]] = None
    if config.verbose:
        _t0 = time.perf_counter()
        def log_fn(msg: str) -> None:  # noqa: E301 — local helper
            elapsed = time.perf_counter() - _t0
            print(f"[tracer.fit +{elapsed:6.1f}s] {msg}", file=sys.stderr, flush=True)

    frontier, split = fit_frontier(X, y_teacher, targets,
                                   max_fit_labels=config.max_fit_labels,
                                   min_coverage=config.min_deploy_coverage,
                                   log=log_fn, skip=config.skip_candidates)

    # Select best pipeline at target TA
    selected = None
    for item in frontier:
        if abs(item["target"] - config.target_teacher_agreement) < 1e-9:
            selected = item
            break

    pipeline_path = None
    method = None
    cov_cal = None
    ta_cal = None
    qual_report = None
    qual_path = None

    if selected and selected["best"] and selected["best"]["stages"]:
        best = selected["best"]
        method = best["summary"]["method"]
        cov_cal = best["summary"].get("coverage_cal_total")
        ta_cal = best["summary"].get("teacher_agreement_cal_total")
        pipeline_path = save_pipeline(artifact_dir, best, labels)
        notes.append(f"Deployed {method} at target TA={config.target_teacher_agreement:.2f}, "
                     f"coverage={cov_cal:.1%}, TA={ta_cal:.3f}")

        # Build qualitative report
        preds, handled, stage_id = route_pipeline(best["stages"], X)
        texts = [r.input_text for r in dataset.records]
        teacher_labels_str = [r.teacher_label for r in dataset.records]
        idx_to_label = {i: l for i, l in enumerate(labels)}
        local_labels_str = [idx_to_label.get(int(p), None) if handled[i] else None
                           for i, p in enumerate(preds)]
        decisions = ["handled" if h else "deferred" for h in handled]

        scores = np.zeros(len(X))
        for i, stage in enumerate(best["stages"]):
            mask = stage_id == i
            if mask.any():
                _, _, s = apply_stage(stage, X[mask])
                scores[mask] = s

        qual_report = build_qualitative_report(
            texts=texts, teacher_labels=teacher_labels_str,
            decisions=decisions, local_labels=local_labels_str,
            accept_scores=scores,
            trace_ids=[r.trace_id for r in dataset.records])
        qual_path = save_qualitative_report(artifact_dir, qual_report)
    else:
        notes.append("No deployable pipeline met the target teacher-parity threshold.")

    # Save traces for continual learning (update needs them)
    from tracer.traces.loader import save_traces
    all_traces_path = artifact_dir / "all_traces.jsonl"
    save_traces(dataset, all_traces_path)

    # Build and save FAISS index
    index = EmbeddingIndex.build(X)
    index_path = artifact_dir / "index"
    index.save(index_path)

    # Save config
    config_path = artifact_dir / "config.json"
    config_path.write_text(json.dumps({
        "target_teacher_agreement": config.target_teacher_agreement,
        "frontier_targets": list(config.frontier_targets),
        "min_deploy_coverage": config.min_deploy_coverage,
        "max_fit_labels": config.max_fit_labels,
        "explore_rate": config.explore_rate,
        "seed": config.seed,
    }, indent=2), encoding="utf-8")

    # Save frontier summary
    frontier_path = artifact_dir / "frontier.json"
    frontier_summary = []
    for item in frontier:
        frontier_summary.append({
            "target": item["target"],
            "best_method": item["best"]["summary"]["method"] if item["best"] else None,
            "best_coverage": item["best"]["summary"].get("coverage_cal_total") if item["best"] else None,
            "best_ta": item["best"]["summary"].get("teacher_agreement_cal_total") if item["best"] else None,
            "candidates": [c["summary"] for c in item["candidates"]],
        })
    frontier_path.write_text(json.dumps(frontier_summary, indent=2, default=str), encoding="utf-8")

    manifest = ArtifactManifest(
        version="0.1.0", n_traces=len(dataset),
        label_space=labels, selected_method=method,
        target_teacher_agreement=config.target_teacher_agreement,
        coverage_cal=cov_cal, teacher_agreement_cal=ta_cal,
        embedding_dim=X.shape[1], n_retrains=1,
        pipeline_path=pipeline_path, index_path=str(index_path),
        config_path=str(config_path),
        qualitative_report_path=qual_path)
    write_manifest(artifact_dir / "manifest.json", manifest)

    return FitResult(
        artifact_dir=str(artifact_dir), manifest=manifest,
        qualitative_report=qual_report, notes=notes)


def update(
    new_trace_path: Union[str, Path],
    artifact_dir: Union[str, Path] = ".tracer",
    new_embeddings: Optional[np.ndarray] = None,
    config: Optional[FitConfig] = None,
) -> FitResult:
    """Refit a TRACER policy with additional traces (continual learning).

    Loads the existing traces from the artifact dir, appends the new ones,
    and re-fits. The .tracer directory is updated in place.
    """
    artifact_dir = Path(artifact_dir)
    manifest = load_manifest(artifact_dir / "manifest.json")

    new_ds = load_traces(new_trace_path)
    new_trace_path = Path(new_trace_path)

    if new_embeddings is None:
        emb_path = new_trace_path.with_suffix(".npy")
        if not emb_path.exists():
            emb_path = new_trace_path.parent / (new_trace_path.stem + "_embeddings.npy")
        if emb_path.exists():
            new_embeddings = np.load(emb_path)
        else:
            raise FileNotFoundError(f"No embeddings for new traces at {emb_path}")

    # Load existing embeddings
    existing_index = EmbeddingIndex.load(artifact_dir / "index")
    X_combined = np.vstack([existing_index.embeddings, new_embeddings.astype(np.float32)])

    # Load existing traces (we need to re-save combined)
    existing_traces_path = artifact_dir / "all_traces.jsonl"
    if existing_traces_path.exists():
        from tracer.traces.loader import load_traces as _lt
        existing_ds = _lt(existing_traces_path)
        combined_records = existing_ds.records + new_ds.records
    else:
        # First update: reconstruct original traces from the initial fit
        # by loading them from the original trace source if we can find it,
        # or infer count from existing embeddings
        combined_records = new_ds.records

    from tracer.types import TraceDataset
    from tracer.traces.loader import save_traces
    combined_ds = TraceDataset(records=combined_records)
    save_traces(combined_ds, existing_traces_path)

    config = config or FitConfig()
    if manifest.target_teacher_agreement:
        config.target_teacher_agreement = manifest.target_teacher_agreement

    return fit(existing_traces_path, artifact_dir, embeddings=X_combined, config=config)


def report(artifact_dir: Union[str, Path] = ".tracer") -> ArtifactManifest:
    """Load and display the artifact manifest."""
    return load_manifest(Path(artifact_dir) / "manifest.json")


def load_router(artifact_dir: Union[str, Path] = ".tracer", embedder=None) -> Router:
    """Load a production router from a .tracer artifact directory.

    Parameters
    ----------
    artifact_dir : path to .tracer/ directory
    embedder : optional Embedder instance - enables text-in prediction.
               If set, router.predict("some text") works directly.
    """
    return Router.load(artifact_dir, embedder=embedder)
