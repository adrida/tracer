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

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")


@dataclass
class FitConfig:
    target_teacher_agreement: float = 0.90
    frontier_targets: Tuple[float, ...] = (0.85, 0.90, 0.95)
    min_deploy_coverage: float = 0.05
    max_fit_labels: int = 8_000
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    seed: int = 42
    # Emit per-candidate + per-stage progress to stderr during fit. Disable
    # for quiet notebook / CI runs.
    verbose: bool = True
    # Candidate surrogate model names to skip during the sweep (e.g. ("gbt",)
    # on large multi-class datasets where sklearn's GradientBoosting dominates
    # wall-time — we have seen 1000× slowdowns vs the next slowest candidate
    # on ~3k × 22-class embedding inputs). See `tracer.fit.surrogate._candidates`
    # for the full set of names.
    skip_candidates: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 < self.target_teacher_agreement <= 1.0:
            raise ValueError(
                "target_teacher_agreement must be in (0, 1], got "
                f"{self.target_teacher_agreement}")
        if not self.frontier_targets:
            raise ValueError("frontier_targets must not be empty")
        for t in self.frontier_targets:
            if not 0.0 < t <= 1.0:
                raise ValueError(
                    f"frontier_targets values must be in (0, 1], got {t}")
        if not 0.0 <= self.min_deploy_coverage <= 1.0:
            raise ValueError(
                "min_deploy_coverage must be in [0, 1], got "
                f"{self.min_deploy_coverage}")
        if self.max_fit_labels <= 0:
            raise ValueError(f"max_fit_labels must be > 0, got {self.max_fit_labels}")
