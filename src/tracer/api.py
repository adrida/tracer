"""Public API for TRACER.

    import tracer
    result = tracer.fit("traces.jsonl", ".tracer")
    router = tracer.load_router(".tracer")
    result = tracer.update("new_traces.jsonl", ".tracer")
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, replace
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
from tracer.fit.certification import certification_split, certify_policy
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
    """Fit and publish a complete artifact generation.

    A fitting or publication exception preserves the existing generation.
    A completed noncertifying fit publishes a nondeployable manifest instead.
    Use one writer per artifact directory and reload readers after return;
    fit to a separate directory to evaluate a replacement before promotion.
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.absolute().parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{artifact_dir.name}-fit-",
                                     dir=artifact_dir.absolute().parent) as temp:
        root = Path(temp)
        staged = root / "next"
        if artifact_dir.exists():
            shutil.copytree(artifact_dir, staged)
        else:
            staged.mkdir()
        _clear_model_artifacts(staged)
        result = _fit_artifacts(trace_path, staged, embeddings, config)
        result.artifact_dir = str(artifact_dir)
        for field_name in ("pipeline_path", "index_path", "config_path", "qualitative_report_path"):
            path = getattr(result.manifest, field_name)
            if path is not None:
                setattr(result.manifest, field_name, str(artifact_dir / Path(path).relative_to(staged)))
        write_manifest(staged / "manifest.json", result.manifest)
        _publish_generation(staged, artifact_dir, root, result.notes)
    return result


def _clear_model_artifacts(directory):
    for filename in ("pipeline.joblib", "qualitative_report.json", "ood.json",
                     "ood_reference.npy", "index.faiss", "report.html", "sankey.html"):
        (directory / filename).unlink(missing_ok=True)


def _publish_generation(staged, artifact_dir, root, notes):
    # The backup is outside the temporary root: a failed rollback must not
    # cause TemporaryDirectory cleanup to discard the previous generation.
    backup = root.with_name(root.name + "-previous")
    had_previous = artifact_dir.exists()
    if had_previous:
        artifact_dir.rename(backup)
    try:
        staged.rename(artifact_dir)
    except BaseException:
        if had_previous:
            try:
                backup.rename(artifact_dir)
            except OSError as exc:
                raise RuntimeError(f"Previous artifacts are preserved at {backup}; restore them before retrying") from exc
        raise
    if had_previous:
        try:
            shutil.rmtree(backup)
        except OSError:
            notes.append(f"Fit completed; previous artifact backup retained at {backup}")


def _fit_artifacts(
    trace_path: Union[str, Path],
    artifact_dir: Union[str, Path],
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

    X = _embedding_matrix(embeddings, len(dataset))
    # Even label discovery must not inspect reserved outcomes: a class seen
    # only there is an unknown label and counts as a mismatch if accepted.
    development, certification_rows = certification_split(
        len(X), config.certification_fraction, config.seed)
    labels = sorted({dataset.records[i].teacher_label for i in development})
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y_teacher = np.array([label_to_idx.get(r.teacher_label, -1) for r in dataset.records], dtype=int)
    y_true = None
    has_gt = all(r.ground_truth is not None for r in dataset.records)
    if has_gt:
        y_true = np.array([label_to_idx.get(r.ground_truth, -1) for r in dataset.records], dtype=int)
        valid = y_true >= 0
        if not valid.all():
            y_true = None
            has_gt = False

    # Reserve final certification before any label balancing or fitting. The
    # candidate-selection code never sees these rows or their outcomes.
    X_dev, y_dev = X[development], y_teacher[development]

    # Fit candidate frontier using development data only.
    targets = list(config.frontier_targets)
    if config.target_teacher_agreement not in targets:
        targets.append(config.target_teacher_agreement)

    log_fn: Optional[Callable[[str], None]] = None
    if config.verbose:
        _t0 = time.perf_counter()
        def log_fn(msg: str) -> None:  # noqa: E301, local helper
            elapsed = time.perf_counter() - _t0
            print(f"[tracer.fit +{elapsed:6.1f}s] {msg}", file=sys.stderr, flush=True)

    frontier, split = fit_frontier(X_dev, y_dev, targets,
                                   max_fit_labels=config.max_fit_labels,
                                   min_coverage=config.min_deploy_coverage,
                                   log=log_fn, skip=config.skip_candidates, seed=config.seed)

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
    certificate = {"status": "no_candidate", "n_examples": len(certification_rows),
                   "alpha": config.certification_alpha,
                   "target": config.target_teacher_agreement}
    serving_router = None
    ood_gate = None

    if selected and selected["best"] and selected["best"]["stages"]:
        # Freeze all policy choices, including OOD, before looking at the
        # reserved outcomes. There is no fallback to another candidate if this
        # one fails final certification.
        best = selected["best"]
        from tracer.fit.ood import fit_ood_gate
        from types import SimpleNamespace
        dev_preds, _, _ = route_pipeline(best["stages"], X_dev)
        pred_label_strs = [labels[int(p)] if p >= 0 else "?" for p in dev_preds]
        ood_gate = fit_ood_gate(X_dev, pred_label_strs)
        serving_router = Router(best["stages"], labels,
                                SimpleNamespace(embedding_dim=X.shape[1]),
                                ood_gate=ood_gate,
                                train_embeddings=X_dev if ood_gate is not None else None)
        certificate = certify_policy(serving_router, X[certification_rows],
                                     y_teacher[certification_rows],
                                     config.target_teacher_agreement,
                                     config.certification_alpha)
        if (certificate["status"] == "certified"
                and certificate["coverage"] < config.min_deploy_coverage):
            certificate["status"] = "below_min_coverage"

    if certificate["status"] == "certified":
        best = selected["best"]
        method = best["summary"]["method"]
        # Legacy field names retained; values now measure the exact serving
        # policy on untouched certification data, not selection calibration.
        cov_cal = certificate["coverage"]
        ta_cal = certificate["teacher_agreement"]
        pipeline_path = save_pipeline(artifact_dir, best, labels)
        notes.append(f"Deployed {method} at target TA={config.target_teacher_agreement:.2f}, "
                     f"coverage={cov_cal:.1%}, TA={ta_cal:.3f}")

        # Descriptive report over supplied traces, using the exact runtime
        # policy. This report is not an independent performance evaluation.
        routed = serving_router.predict_batch(X)
        preds, handled, stage_id = routed["preds"], routed["handled"], routed["stage_id"]
        texts = [r.input_text for r in dataset.records]
        teacher_labels_str = [r.teacher_label for r in dataset.records]
        idx_to_label = {i: l for i, l in enumerate(labels)}
        local_labels_str = [idx_to_label.get(int(p), None) if handled[i] else None
                           for i, p in enumerate(preds)]
        decisions = ["handled" if h else "deferred" for h in handled]

        scores = np.zeros(len(X))
        remaining = np.ones(len(X), dtype=bool)
        for stage in best["stages"]:
            if remaining.any():
                _, accepted, stage_scores = apply_stage(stage, X[remaining])
                rows = np.flatnonzero(remaining)
                scores[rows] = stage_scores
                remaining[rows[accepted]] = False

        qual_report = build_qualitative_report(
            texts=texts, teacher_labels=teacher_labels_str,
            decisions=decisions, local_labels=local_labels_str,
            accept_scores=scores,
            trace_ids=[r.trace_id for r in dataset.records])
        qual_path = save_qualitative_report(artifact_dir, qual_report)

        if ood_gate is not None:
            (artifact_dir / "ood.json").write_text(json.dumps(ood_gate))
            np.save(artifact_dir / "ood_reference.npy", X_dev)
    else:
        notes.append(f"No deployable pipeline met the final teacher-agreement check: {certificate['status']}.")

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
    config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    # Save frontier summary
    frontier_path = artifact_dir / "frontier.json"
    frontier_summary = []
    for item in frontier:
        frontier_summary.append({
            "target": item["target"],
            "evaluation_role": "development_selection_only",
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
        qualitative_report_path=qual_path,
        certification=certificate,
        ood_required=method is not None and ood_gate is not None)
    write_manifest(artifact_dir / "manifest.json", manifest)

    return FitResult(
        artifact_dir=str(artifact_dir), manifest=manifest,
        qualitative_report=qual_report, notes=notes)


def _embedding_matrix(values, n_rows, dim=None):
    X = np.asarray(values, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("Embeddings must be a nonempty-width 2-D matrix (n_traces, dim)")
    if len(X) != n_rows:
        raise ValueError(f"Trace/embedding mismatch: {n_rows} traces vs {len(X)} embeddings")
    if dim is not None and X.shape[1] != dim:
        raise ValueError(f"Embedding dimension mismatch: expected {dim}, got {X.shape[1]}")
    if not np.isfinite(X).all():
        raise ValueError("Embeddings must contain only finite values")
    return X


def update(
    new_trace_path: Union[str, Path],
    artifact_dir: Union[str, Path] = ".tracer",
    new_embeddings: Optional[np.ndarray] = None,
    config: Optional[FitConfig] = None,
) -> FitResult:
    """Refit a TRACER policy with additional traces (continual learning).

    Loads the existing traces from the artifact dir, appends the new ones,
    and re-fits in a staging directory. Failed validation or fitting leaves the
    existing artifacts intact. Use one writer per artifact directory; reload
    readers after update() returns. An explicit config takes precedence over
    the saved configuration and is never mutated.
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

    # Load existing traces so we can re-save the combined set. fit() always
    # writes all_traces.jsonl, so it should be present. If it is missing (e.g.
    # it was deleted), the original records cannot be recovered from the index
    # embeddings alone, and appending only the new records would leave the trace
    # set out of sync with X_combined. Fail fast with an actionable message
    # instead of surfacing a cryptic trace/embedding mismatch from fit().
    existing_traces_path = artifact_dir / "all_traces.jsonl"
    if not existing_traces_path.exists():
        raise FileNotFoundError(
            f"{existing_traces_path} not found, cannot continue continual "
            "learning. fit() writes this file; if it was removed, re-run "
            "tracer.fit() on your full trace set to rebuild it before calling "
            "update().")

    from tracer.traces.loader import load_traces as _lt
    existing_ds = _lt(existing_traces_path)
    combined_records = existing_ds.records + new_ds.records
    existing_index = EmbeddingIndex.load(artifact_dir / "index")
    X_existing = _embedding_matrix(existing_index.embeddings, len(existing_ds), manifest.embedding_dim)
    X_new = _embedding_matrix(new_embeddings, len(new_ds), X_existing.shape[1])
    X_combined = np.vstack([X_existing, X_new])

    from tracer.types import TraceDataset
    from tracer.traces.loader import save_traces
    combined_ds = TraceDataset(records=combined_records)
    if config is None:
        from tracer.config import EmbeddingConfig
        config_path = artifact_dir / "config.json"
        saved = json.loads(config_path.read_text()) if config_path.exists() else {}
        if "embedding" in saved:
            saved["embedding"] = EmbeddingConfig(**saved["embedding"])
        saved.setdefault("target_teacher_agreement", manifest.target_teacher_agreement)
        config = FitConfig(**saved)
    else:
        config = replace(config)

    # Keep the temporary generation on the same filesystem as the destination.
    with tempfile.TemporaryDirectory(prefix=f".{artifact_dir.name}-update-",
                                     dir=artifact_dir.absolute().parent) as temp:
        root = Path(temp)
        staged = root / "next"
        shutil.copytree(artifact_dir, staged)
        # A refit must not inherit reports or gates from the previous model.
        _clear_model_artifacts(staged)
        combined_path = root / "combined.jsonl"
        save_traces(combined_ds, combined_path)
        result = _fit_artifacts(combined_path, staged, embeddings=X_combined, config=config)
        result.artifact_dir = str(artifact_dir)
        result.manifest.n_retrains = manifest.n_retrains + 1
        for field_name in ("pipeline_path", "index_path", "config_path", "qualitative_report_path"):
            path = getattr(result.manifest, field_name)
            if path is not None:
                setattr(result.manifest, field_name, str(artifact_dir / Path(path).relative_to(staged)))
        write_manifest(staged / "manifest.json", result.manifest)
        _publish_generation(staged, artifact_dir, root, result.notes)
    return result


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


# Heal the function/subpackage name collision: importing this module pulls in
# the `tracer.fit` pipeline subpackage, which Python binds as the `tracer.fit`
# attribute and would shadow the public `tracer.fit()` function. The lazy
# package __getattr__ can't override an existing attribute, so re-assert the
# function on the package here -- importing tracer.api by ANY path fixes it.
import tracer as _tracer  # noqa: E402

_tracer.fit = fit
