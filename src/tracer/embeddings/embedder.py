"""Unified embedding interface for TRACER.

Supports three backends:
    Embedder.from_sentence_transformers("BAAI/bge-small-en-v1.5")
    Embedder.from_endpoint("https://api.example.com/embed")
    Embedder.from_callable(my_fn)

Once created, an Embedder plugs into the Router so you can pass raw text:
    router = tracer.load_router(".tracer", embedder=embedder)
    router.predict("What is my balance?")
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union

import numpy as np


class Embedder:
    """Compute embeddings from text. Plug into a Router for text-in routing."""

    def __init__(self, fn: Callable[[List[str]], np.ndarray]):
        self._fn = fn

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts. Returns (n, dim) float32 array."""
        out = self._fn(texts)
        return np.asarray(out, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text. Returns (dim,) float32 array."""
        return self.embed([text])[0]

    # ── Factory: sentence-transformers (local) ───────────────────────────────

    @classmethod
    def from_sentence_transformers(
        cls,
        model: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 128,
        normalize: bool = True,
    ) -> "Embedder":
        """Create an Embedder using a sentence-transformers model.

        Requires: pip install tracer-llm[embeddings]

        Parameters
        ----------
        model : model name or path (e.g. "BAAI/bge-small-en-v1.5")
        device : "cpu", "cuda", "mps", or None (auto-detect)
        batch_size : forward-pass batch size
        normalize : L2-normalize embeddings
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required.\n"
                "Install with: pip install tracer-llm[embeddings]"
            )
        encoder = SentenceTransformer(model, device=device)

        def _embed(texts: List[str]) -> np.ndarray:
            return encoder.encode(
                texts, batch_size=batch_size,
                normalize_embeddings=normalize, show_progress_bar=False,
            )

        embedder = cls(_embed)
        embedder._backend = "sentence-transformers"
        embedder._model = model
        return embedder

    # ── Factory: HTTP endpoint ───────────────────────────────────────────────

    @classmethod
    def from_endpoint(
        cls,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        input_key: str = "input",
        output_key: str = "embedding",
        batch_key: Optional[str] = None,
        batch_output_key: Optional[str] = None,
    ) -> "Embedder":
        """Create an Embedder that calls an external HTTP embedding API.

        The endpoint should accept POST with JSON and return embeddings.

        Single-text mode (default):
            POST url  {"input": "text"}  →  {"embedding": [0.1, 0.2, ...]}

        Batch mode (if batch_key set):
            POST url  {"inputs": ["a","b"]}  →  {"embeddings": [[...], [...]]}

        Parameters
        ----------
        url : endpoint URL
        headers : extra HTTP headers (e.g. {"Authorization": "Bearer ..."})
        input_key : JSON key for the input text(s)
        output_key : JSON key in the response containing the embedding vector
        batch_key : if set, sends all texts in one request under this key
        batch_output_key : response key for the batch of embeddings
        """
        import json
        from urllib.request import Request, urlopen

        _headers = {"Content-Type": "application/json"}
        if headers:
            _headers.update(headers)

        def _embed(texts: List[str]) -> np.ndarray:
            if batch_key:
                # Single request for the whole batch
                payload = json.dumps({batch_key: texts}).encode()
                req = Request(url, data=payload, headers=_headers, method="POST")
                resp = json.loads(urlopen(req).read())
                return np.asarray(resp[batch_output_key or output_key], dtype=np.float32)
            else:
                # One request per text
                results = []
                for text in texts:
                    payload = json.dumps({input_key: text}).encode()
                    req = Request(url, data=payload, headers=_headers, method="POST")
                    resp = json.loads(urlopen(req).read())
                    results.append(resp[output_key])
                return np.asarray(results, dtype=np.float32)

        embedder = cls(_embed)
        embedder._backend = "endpoint"
        embedder._url = url
        return embedder

    # ── Factory: custom callable ─────────────────────────────────────────────

    @classmethod
    def from_callable(cls, fn: Callable) -> "Embedder":
        """Create an Embedder from any function.

        fn should accept a list of strings and return an array-like of shape (n, dim).

        Parameters
        ----------
        fn : callable(texts: list[str]) -> array-like
        """
        embedder = cls(fn)
        embedder._backend = "callable"
        return embedder

    def __repr__(self):
        backend = getattr(self, "_backend", "unknown")
        model = getattr(self, "_model", "")
        url = getattr(self, "_url", "")
        extra = f" model={model}" if model else f" url={url}" if url else ""
        return f"Embedder(backend={backend}{extra})"
