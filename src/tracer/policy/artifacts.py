"""Artifact I/O for the .tracer directory."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from tracer.types import ArtifactManifest


def write_manifest(path: Path, manifest: ArtifactManifest) -> None:
    payload = {
        "version": manifest.version,
        "n_traces": manifest.n_traces,
        "label_space": manifest.label_space,
        "selected_method": manifest.selected_method,
        "target_teacher_agreement": manifest.target_teacher_agreement,
        "coverage_cal": manifest.coverage_cal,
        "teacher_agreement_cal": manifest.teacher_agreement_cal,
        "embedding_dim": manifest.embedding_dim,
        "n_retrains": manifest.n_retrains,
        "pipeline_path": manifest.pipeline_path,
        "index_path": manifest.index_path,
        "config_path": manifest.config_path,
        "qualitative_report_path": manifest.qualitative_report_path,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> ArtifactManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ArtifactManifest(**payload)


def save_pipeline(artifact_dir: Path, pipeline: dict, label_space: list) -> str:
    path = artifact_dir / "pipeline.joblib"
    joblib.dump({"pipeline": pipeline, "label_space": label_space}, path)
    return str(path)


def load_pipeline(artifact_dir: Path) -> dict:
    return joblib.load(artifact_dir / "pipeline.joblib")


def save_qualitative_report(artifact_dir: Path, report) -> str:
    import dataclasses
    path = artifact_dir / "qualitative_report.json"
    path.write_text(json.dumps(dataclasses.asdict(report), indent=2, default=str), encoding="utf-8")
    return str(path)
