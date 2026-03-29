"""Core datatypes for TRACER."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class TraceRecord:
    """A single teacher trace: input, teacher prediction, optional ground truth."""
    input_text: str
    teacher_label: str
    trace_id: Optional[str] = None
    ground_truth: Optional[str] = None
    embedding: Any = None  # np.ndarray or None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceDataset:
    """Collection of teacher traces with label space discovery."""
    records: List[TraceRecord]

    @property
    def label_space(self) -> Set[str]:
        return {r.teacher_label for r in self.records}

    def __len__(self) -> int:
        return len(self.records)


@dataclass
class SliceInsight:
    """Handled/deferred statistics for one interpretable slice."""
    slice_name: str
    predicate: str
    count: int
    handled_rate: float
    deferred_rate: float
    teacher_agreement_handled: Optional[float] = None
    dominant_teacher_label: Optional[str] = None


@dataclass
class RepresentativeExample:
    """A single representative example from handled or deferred traffic."""
    input_preview: str
    teacher_label: str
    decision: str  # "handled" or "deferred"
    local_label: Optional[str] = None
    accept_score: Optional[float] = None
    trace_id: Optional[str] = None


@dataclass
class BoundaryPair:
    """Contrastive pair: same label, different routing outcome."""
    handled_preview: str
    deferred_preview: str
    teacher_label: str
    handled_score: Optional[float] = None
    deferred_score: Optional[float] = None


@dataclass
class TemporalDelta:
    """Change in handled rate for a label between two fits."""
    label: str
    previous_handled_rate: float
    current_handled_rate: float
    delta: float


@dataclass
class QualitativeReport:
    """Full qualitative analysis of a routing policy."""
    summary: str
    coverage: float
    teacher_agreement_handled: float
    slices: List[SliceInsight] = field(default_factory=list)
    handled_examples: List[RepresentativeExample] = field(default_factory=list)
    deferred_examples: List[RepresentativeExample] = field(default_factory=list)
    boundary_pairs: List[BoundaryPair] = field(default_factory=list)
    temporal_deltas: List[TemporalDelta] = field(default_factory=list)


@dataclass
class ArtifactManifest:
    """Metadata for a .tracer artifact directory."""
    version: str
    n_traces: int
    label_space: List[str]
    selected_method: Optional[str] = None
    target_teacher_agreement: float = 0.90
    coverage_cal: Optional[float] = None
    teacher_agreement_cal: Optional[float] = None
    embedding_dim: Optional[int] = None
    n_retrains: int = 0
    pipeline_path: Optional[str] = None
    index_path: Optional[str] = None
    config_path: Optional[str] = None
    qualitative_report_path: Optional[str] = None


@dataclass
class FitResult:
    """Result of a tracer.fit() call."""
    artifact_dir: str
    manifest: ArtifactManifest
    qualitative_report: Optional[QualitativeReport] = None
    notes: List[str] = field(default_factory=list)
