"""Configuration for TRACER fit and routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class EmbeddingConfig:
    backend: str = "precomputed"
    model: Optional[str] = None
    endpoint_url: Optional[str] = None
    batch_size: int = 128
    normalize: bool = True


@dataclass
class FitConfig:
    target_teacher_agreement: float = 0.90
    frontier_targets: Tuple[float, ...] = (0.85, 0.90, 0.95)
    min_deploy_coverage: float = 0.05
    max_fit_labels: int = 8_000
    explore_rate: float = 0.05
    retrain_every: int = 100
    min_labels_to_start: int = 100
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    seed: int = 42
