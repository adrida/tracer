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

    def __init__(self, stages: list, label_space: list, manifest, embedder=None,
                 ood_gate=None, train_embeddings=None):
        self._stages = stages
        self._label_space = label_space
        self._idx_to_label = {i: l for i, l in enumerate(label_space)}
        self.manifest = manifest
        self.embedder = embedder
        self._ood_gate = ood_gate
        self._train_embeddings = train_embeddings
        self._ood_nn = None
        if ood_gate is not None and train_embeddings is not None:
            try:
                from sklearn.neighbors import NearestNeighbors
                k = int(min(ood_gate.get("k", 10), len(train_embeddings)))
                if k >= 1:
                    self._ood_nn = NearestNeighbors(n_neighbors=k).fit(
                        np.asarray(train_embeddings, dtype=np.float32)
                    )
            except Exception:
                pass

    @classmethod
    def load(cls, artifact_dir: Union[str, Path], embedder=None) -> "Router":
        """Load a router from a .tracer artifact directory.

        Parameters
        ----------
        artifact_dir : path to .tracer/
        embedder : optional Embedder instance - enables text-in prediction
        """
        import json as _json
        artifact_dir = Path(artifact_dir)
        manifest = load_manifest(artifact_dir / "manifest.json")
        if manifest.selected_method is None:
            raise ValueError(
                f"No router is deployed in {artifact_dir}: the parity gate did not "
                "certify any method (selected_method is null in manifest.json). "
                "Run tracer.fit() again with more data or a lower target_teacher_agreement."
            )
        bundle = load_pipeline(artifact_dir)
        stages = bundle["pipeline"]["stages"]
        label_space = bundle["label_space"]

        # Optional OOD safety net: defer inputs far from the training distribution.
        ood_gate = None
        train_emb = None
        ood_path = artifact_dir / "ood.json"
        if ood_path.exists():
            try:
                ood_gate = _json.loads(ood_path.read_text())
                from tracer.embeddings.index import EmbeddingIndex
                train_emb = EmbeddingIndex.load(artifact_dir / "index").embeddings
            except Exception:
                ood_gate, train_emb = None, None
        return cls(stages=stages, label_space=label_space, manifest=manifest,
                   embedder=embedder, ood_gate=ood_gate, train_embeddings=train_emb)

    def _ood_flags(self, X: np.ndarray, preds) -> np.ndarray:
        """True where each row is out-of-distribution (force-deferred)."""
        if self._ood_gate is None or self._train_embeddings is None:
            return np.zeros(len(X), dtype=bool)
        from tracer.fit.ood import ood_mask
        labels = [self._idx_to_label.get(int(p), "?") for p in preds]
        return ood_mask(X, self._train_embeddings, labels, self._ood_gate, nn=self._ood_nn)

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
                # OOD safety net: defer if the input is far from the training
                # distribution, even when the surrogate is confident.
                if bool(self._ood_flags(X, preds)[0]):
                    break
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
        expected_dim = self.manifest.embedding_dim
        if expected_dim is not None and X.shape[-1] != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {expected_dim}, "
                f"got {X.shape[-1]}."
            )
        preds, handled, stage_id = route_pipeline(self._stages, X)
        # OOD safety net: force-defer rows far from the training distribution.
        if self._ood_gate is not None and np.asarray(handled).any():
            ood = self._ood_flags(X, preds)
            handled = np.asarray(handled) & ~ood
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
