"""TRACER -- Trace-Based Adaptive Cost-Efficient Routing.

Turn LLM traces into parity-gated routing policies that progressively
replace teacher calls -- with formal guarantees and qualitative audit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# `watch` collides with the tracer.watch submodule name AND is cheap (stdlib +
# tracer.types only), so bind it eagerly: that makes `tracer.watch` resolve to
# the function, not the module. (The heavy ML symbols stay lazy below.)
from tracer.watch import GenAISpan, Watcher, watch  # noqa: E402

# Lazy public API (PEP 562). The heavy ML stack (numpy / scikit-learn /
# hdbscan / sentence-transformers) is only imported on first attribute access,
# so `import tracer` -- and lightweight entry points like `tracer cloud` --
# start instantly instead of paying multi-second import cost up front.
#
# `fit` also collides with the tracer.fit subpackage, so it cannot be healed by
# __getattr__ alone (the submodule attribute shadows it). tracer.api re-asserts
# `tracer.fit = <function>` at the end of its own module, so importing api by
# any path fixes the binding.
_LAZY: dict[str, tuple[str, str]] = {
    "fit": ("tracer.api", "fit"),
    "report": ("tracer.api", "report"),
    "load_router": ("tracer.api", "load_router"),
    "update": ("tracer.api", "update"),
    "scan": ("tracer.scanner", "scan"),
    "generate_html_report": ("tracer.analysis.html_report", "generate_html_report"),
    "generate_sankey": ("tracer.analysis.sankey", "generate_sankey"),
    "serve": ("tracer.runtime.serve", "serve"),
    "AsyncBatcher": ("tracer.runtime.batching", "AsyncBatcher"),
    "Embedder": ("tracer.embeddings.embedder", "Embedder"),
    "FitConfig": ("tracer.config", "FitConfig"),
    "EmbeddingConfig": ("tracer.config", "EmbeddingConfig"),
    "ArtifactManifest": ("tracer.types", "ArtifactManifest"),
    "BoundaryPair": ("tracer.types", "BoundaryPair"),
    "FitResult": ("tracer.types", "FitResult"),
    "QualitativeReport": ("tracer.types", "QualitativeReport"),
    "RepresentativeExample": ("tracer.types", "RepresentativeExample"),
    "SliceInsight": ("tracer.types", "SliceInsight"),
    "TemporalDelta": ("tracer.types", "TemporalDelta"),
    "TraceDataset": ("tracer.types", "TraceDataset"),
    "TraceRecord": ("tracer.types", "TraceRecord"),
}

if TYPE_CHECKING:  # static analysers still see the real symbols
    from tracer.api import fit, load_router, report, update  # noqa: F401
    from tracer.config import EmbeddingConfig, FitConfig  # noqa: F401
    from tracer.analysis.html_report import generate_html_report  # noqa: F401
    from tracer.analysis.sankey import generate_sankey  # noqa: F401
    from tracer.embeddings.embedder import Embedder  # noqa: F401
    from tracer.runtime.serve import serve  # noqa: F401
    from tracer.runtime.batching import AsyncBatcher  # noqa: F401
    from tracer.scanner import scan  # noqa: F401
    from tracer.types import (  # noqa: F401
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
    from tracer.watch import GenAISpan, Watcher, watch  # noqa: F401


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'tracer' has no attribute {name!r}")
    import importlib

    obj = getattr(importlib.import_module(target[0]), target[1])
    globals()[name] = obj  # cache so subsequent access skips __getattr__
    return obj


__version__ = "0.3.3"

__all__ = [
    # Core API
    "scan",
    "watch",
    "Watcher",
    "GenAISpan",
    "fit",
    "update",
    "load_router",
    "report",
    "embed",
    "generate_html_report",
    "generate_sankey",
    "serve",
    "AsyncBatcher",
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
