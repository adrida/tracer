"""FAISS index wrapper for nearest-neighbor retrieval in qualitative analysis."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np


class EmbeddingIndex:
    """Thin wrapper around a FAISS index for trace retrieval."""

    def __init__(self, embeddings: np.ndarray, index=None):
        self.embeddings = embeddings.astype(np.float32, copy=False)
        self._index = index

    @classmethod
    def build(cls, embeddings: np.ndarray, metric: str = "cosine") -> "EmbeddingIndex":
        try:
            import faiss
        except ImportError:
            return cls(embeddings, index=None)

        X = embeddings.astype(np.float32, copy=False)
        if metric == "cosine":
            faiss.normalize_L2(X)
            idx = faiss.IndexFlatIP(X.shape[1])
        else:
            idx = faiss.IndexFlatL2(X.shape[1])
        idx.add(X)
        return cls(embeddings, index=idx)

    def search(self, query: np.ndarray, k: int = 5):
        if self._index is None:
            dists = np.linalg.norm(self.embeddings - query.reshape(1, -1), axis=1)
            topk = np.argsort(dists)[:k]
            return topk, dists[topk]
        q = query.astype(np.float32).reshape(1, -1)
        try:
            import faiss
            faiss.normalize_L2(q)
        except ImportError:
            pass
        D, I = self._index.search(q, k)
        return I[0], D[0]

    def save(self, path: Path) -> None:
        np.save(path.with_suffix(".embeddings.npy"), self.embeddings)
        if self._index is not None:
            try:
                import faiss
                faiss.write_index(self._index, str(path.with_suffix(".faiss")))
            except ImportError:
                pass

    @classmethod
    def load(cls, path: Path) -> "EmbeddingIndex":
        emb = np.load(path.with_suffix(".embeddings.npy"))
        idx = None
        faiss_path = path.with_suffix(".faiss")
        if faiss_path.exists():
            try:
                import faiss
                idx = faiss.read_index(str(faiss_path))
            except ImportError:
                pass
        return cls(emb, index=idx)


def embed_texts(
    texts: List[str],
    model: str = "all-MiniLM-L6-v2",
    batch_size: int = 128,
    normalize: bool = True,
    show_progress: bool = True,
    device: Optional[str] = None,
) -> np.ndarray:
    """Compute embeddings using sentence-transformers.

    Parameters
    ----------
    texts : list of input strings
    model : sentence-transformers model name or path.
            Popular choices:
            - "all-MiniLM-L6-v2" (fast, 384-dim, good default)
            - "BAAI/bge-small-en-v1.5" (384-dim, strong retrieval)
            - "BAAI/bge-base-en-v1.5" (768-dim)
            - "BAAI/bge-m3" (1024-dim, multilingual, used in TRACER paper)
    batch_size : samples per forward pass
    normalize : L2-normalize embeddings (recommended for cosine similarity)
    show_progress : show a progress bar
    device : "cpu", "cuda", "mps", or None (auto-detect)

    Returns
    -------
    np.ndarray of shape (len(texts), embedding_dim), dtype float32
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for embedding.\n"
            "Install it with:  pip install tracer-llm[embeddings]"
        )

    encoder = SentenceTransformer(model, device=device)
    try:
        embeddings = encoder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
        )
        return np.asarray(embeddings, dtype=np.float32)
    except RuntimeError as e:
        if "Numpy is not available" in str(e) or "numpy" in str(e).lower():
            raise RuntimeError(
                "PyTorch cannot convert tensors to NumPy. This usually means "
                "your NumPy version is too new for your PyTorch version.\n"
                "Fix with:  pip install 'numpy<2.1'\n"
                "Then restart your Python process / notebook kernel."
            ) from e
        raise
