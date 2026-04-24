"""TRACER -- Trace-Based Adaptive Cost-Efficient Routing.

Turn LLM traces into parity-gated routing policies that progressively
replace teacher calls -- with formal guarantees and qualitative audit.
"""

from tracer.api import fit, report, load_router, update
from tracer.analysis.html_report import generate_html_report
from tracer.analysis.sankey import generate_sankey
from tracer.runtime.serve import serve
from tracer.embeddings.embedder import Embedder
from tracer.config import FitConfig, EmbeddingConfig
from tracer.types import (
    ArtifactManifest,
    BoundaryPair,
    FitResult,
    QualitativeReport,
    RepresentativeExample,
    SliceInsight,
    TemporalDelta,
    TraceDataset,
    TraceRecord,
)

__version__ = "0.1.2"

__all__ = [
    # Core API
    "fit",
    "update",
    "load_router",
    "report",
    "embed",
    "generate_html_report",
    "generate_sankey",
    "serve",
    "Embedder",
    # Config
    "FitConfig",
    "EmbeddingConfig",
    # Types
    "ArtifactManifest",
    "BoundaryPair",
    "FitResult",
    "QualitativeReport",
    "RepresentativeExample",
    "SliceInsight",
    "TemporalDelta",
    "TraceDataset",
    "TraceRecord",
]


def embed(texts, model="all-MiniLM-L6-v2", **kwargs):
    """Compute embeddings for a list of texts using sentence-transformers.

    Convenience wrapper around tracer.embeddings.index.embed_texts.
    Requires: pip install tracer-llm[embeddings]

    Parameters
    ----------
    texts : list of str
    model : sentence-transformers model name (default: "all-MiniLM-L6-v2")
    **kwargs : passed to embed_texts (batch_size, normalize, device, etc.)

    Returns
    -------
    np.ndarray of shape (len(texts), embedding_dim)
    """
    from tracer.embeddings.index import embed_texts
    return embed_texts(texts, model=model, **kwargs)
