"""Production router: load a fitted .tracer artifact and route live traffic.

Accepts raw text if an Embedder is attached:
    router = tracer.load_router(".tracer", embedder=embedder)
    router.predict("What is my balance?")

Or raw embeddings (backward compatible):
    router = tracer.load_router(".tracer")
    router.predict(embedding_vector)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import numpy as np

from tracer.fit.pipeline import apply_stage, _accept_scores, _predict
from tracer.policy.artifacts import load_manifest, load_pipeline


class Router:
    """Route incoming requests through a fitted TRACER policy.

    Usage (with embedder - recommended):
        from tracer import Embedder
        embedder = Embedder.from_sentence_transformers("BAAI/bge-small-en-v1.5")
        router = tracer.load_router(".tracer", embedder=embedder)
        router.predict("What is my balance?")

    Usage (raw embeddings):
        router = tracer.load_router(".tracer")
        router.predict(np.array([0.1, 0.2, ...]))
    """

    def __init__(self, stages: list, label_space: list, manifest, embedder=None):
        self._stages = stages
        self._label_space = label_space
        self._idx_to_label = {i: l for i, l in enumerate(label_space)}
        self.manifest = manifest
        self.embedder = embedder

    @classmethod
    def load(cls, artifact_dir: Union[str, Path], embedder=None) -> "Router":
        """Load a router from a .tracer artifact directory.

        Parameters
        ----------
        artifact_dir : path to .tracer/
        embedder : optional Embedder instance - enables text-in prediction
        """
        artifact_dir = Path(artifact_dir)
        manifest = load_manifest(artifact_dir / "manifest.json")
        bundle = load_pipeline(artifact_dir)
        stages = bundle["pipeline"]["stages"]
        label_space = bundle["label_space"]
        return cls(stages=stages, label_space=label_space,
                   manifest=manifest, embedder=embedder)

    def _to_embedding(self, input) -> np.ndarray:
        """Convert input (text or array) to a (dim,) float32 embedding."""
        if isinstance(input, str):
            if self.embedder is None:
                raise ValueError(
                    "Got a string but no embedder is configured. Either:\n"
                    "  - Pass a numpy embedding vector instead of text\n"
                    "  - Create a router with an embedder:\n"
                    "      from tracer import Embedder\n"
                    "      embedder = Embedder.from_sentence_transformers('...')\n"
                    "      router = tracer.load_router('.tracer', embedder=embedder)"
                )
            return self.embedder.embed_one(input)
        return np.asarray(input, dtype=np.float32)

    def _to_embeddings(self, inputs) -> np.ndarray:
        """Convert inputs (list of texts or 2D array) to (n, dim) float32."""
        if isinstance(inputs, list) and inputs and isinstance(inputs[0], str):
            if self.embedder is None:
                raise ValueError(
                    "Got a list of strings but no embedder is configured.\n"
                    "Pass a numpy array of embeddings or set an embedder."
                )
            return self.embedder.embed(inputs)
        return np.asarray(inputs, dtype=np.float32)

    def predict(
        self,
        input: Union[str, np.ndarray],
        fallback: Optional[Callable] = None,
    ) -> dict:
        """Route a single sample.

        Parameters
        ----------
        input : text string (requires embedder) or embedding array (dim,)
        fallback : called with no args if deferred; return value used as label

        Returns
        -------
        dict with keys: label, decision ("handled"/"deferred"), accept_score, stage
        """
        embedding = self._to_embedding(input)
        expected_dim = self.manifest.embedding_dim
        if expected_dim is not None and embedding.shape[-1] != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {expected_dim}, "
                f"got {embedding.shape[-1]}."
            )
        X = embedding.reshape(1, -1)
        for idx, stage in enumerate(self._stages):
            preds, accept, scores = apply_stage(stage, X)
            if accept[0]:
                return {
                    "label": self._idx_to_label.get(int(preds[0]), str(preds[0])),
                    "decision": "handled",
                    "accept_score": float(scores[0]),
                    "stage": idx,
                }

        if fallback is not None:
            teacher_label = fallback()
            return {"label": teacher_label, "decision": "deferred",
                    "accept_score": 0.0, "stage": -1}

        return {"label": None, "decision": "deferred",
                "accept_score": 0.0, "stage": -1}

    def predict_batch(
        self, inputs: Union[List[str], np.ndarray],
    ) -> dict:
        """Route a batch.

        Parameters
        ----------
        inputs : list of text strings (requires embedder) or (n, dim) embedding array

        Returns
        -------
        dict with: labels, decisions, handled (bool array), preds, stage_id
        """
        from tracer.fit.pipeline import route_pipeline
        X = self._to_embeddings(inputs)
        preds, handled, stage_id = route_pipeline(self._stages, X)
        labels = []
        decisions = []
        for i in range(len(X)):
            if handled[i]:
                labels.append(self._idx_to_label.get(int(preds[i]), str(preds[i])))
                decisions.append("handled")
            else:
                labels.append(None)
                decisions.append("deferred")
        return {"labels": labels, "decisions": decisions,
                "handled": handled, "preds": preds, "stage_id": stage_id}
