"""Production router: load a fitted .tracer artifact and route live traffic.

Accepts raw text if an Embedder is attached:
    router = tracer.load_router(".tracer", embedder=embedder)
    router.predict("What is my balance?")

Or raw embeddings (backward compatible):
    router = tracer.load_router(".tracer")
    router.predict(embedding_vector)
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any, Awaitable, Callable, List, Optional, Union

import numpy as np

from tracer.fit.pipeline import apply_stage, _accept_scores, _predict
from tracer.policy.artifacts import load_manifest, load_pipeline


def route_scored(stages: list, X: np.ndarray):
    """Route a batch through the staged pipeline, also returning accept scores.

    This mirrors :func:`tracer.fit.pipeline.route_pipeline` (same routing
    decisions, vectorized) but additionally returns, for every item, the
    acceptor score recorded at the stage that handled it -- or, for deferred
    items, the score at the final stage they reached. Having the score available
    lets the batch and async prediction paths return the same per-item payload
    as :meth:`Router.predict` (``label``/``decision``/``accept_score``/``stage``).

    Returns
    -------
    (preds, handled, stage_id, scores) : tuple of np.ndarray, each length ``len(X)``
    """
    n = len(X)
    preds = np.full(n, -1, dtype=int)
    handled = np.zeros(n, dtype=bool)
    stage_id = np.full(n, -1, dtype=int)
    scores = np.zeros(n, dtype=float)
    remaining = np.arange(n)
    for idx, stage in enumerate(stages):
        if len(remaining) == 0:
            break
        sp, sa, ss = apply_stage(stage, X[remaining])
        # Record the score for everything still in flight, then overwrite the
        # accepted subset with the (identical) accepted scores so deferred items
        # keep the score from the last stage they were evaluated against.
        scores[remaining] = ss
        if sa.any():
            sel = remaining[sa]
            preds[sel] = sp[sa]
            handled[sel] = True
            stage_id[sel] = idx
        remaining = remaining[~sa]
    return preds, handled, stage_id, scores


async def _maybe_await(value: Any) -> Any:
    """Return ``value``, awaiting it first if it is awaitable.

    Lets the async prediction paths accept either a plain callable
    (``fallback(x) -> label``) or a coroutine function
    (``async def fallback(x) -> label``) without the caller declaring which.
    """
    if inspect.isawaitable(value):
        return await value
    return value


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

    def _check_dim(self, X: np.ndarray) -> None:
        """Validate the trailing embedding dimension against the manifest."""
        expected_dim = self.manifest.embedding_dim
        if expected_dim is not None and X.shape[-1] != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {expected_dim}, "
                f"got {X.shape[-1]}."
            )

    def _assemble(self, i, preds, handled, stage_id, scores) -> dict:
        """Build the per-item result dict for a handled sample at index ``i``."""
        return {
            "label": self._idx_to_label.get(int(preds[i]), str(preds[i])),
            "decision": "handled",
            "accept_score": float(scores[i]),
            "stage": int(stage_id[i]),
        }

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
        self,
        inputs: Union[List[str], np.ndarray],
        fallback: Optional[Callable[[Any], Any]] = None,
    ) -> dict:
        """Route a batch in a single vectorized pass.

        Parameters
        ----------
        inputs : list of text strings (requires an embedder) or an
            ``(n, dim)`` embedding array.
        fallback : optional callable invoked for each *deferred* item as
            ``fallback(original_input) -> label``. ``original_input`` is the text
            (when ``inputs`` is a list) or the embedding row (when ``inputs`` is
            an array). When omitted, deferred items keep ``label=None`` -- the
            previous behavior, so this stays backward compatible.

        Returns
        -------
        dict with keys: labels, decisions, handled (bool array), preds,
        stage_id, accept_scores.
        """
        X = self._to_embeddings(inputs)
        self._check_dim(X)
        preds, handled, stage_id, scores = route_scored(self._stages, X)
        is_list = isinstance(inputs, list)
        labels: list = []
        decisions: list = []
        for i in range(len(X)):
            if handled[i]:
                labels.append(self._idx_to_label.get(int(preds[i]), str(preds[i])))
                decisions.append("handled")
            else:
                decisions.append("deferred")
                if fallback is not None:
                    original = inputs[i] if is_list else X[i]
                    labels.append(fallback(original))
                else:
                    labels.append(None)
        return {"labels": labels, "decisions": decisions, "handled": handled,
                "preds": preds, "stage_id": stage_id, "accept_scores": scores}

    async def apredict(
        self,
        input: Union[str, np.ndarray],
        fallback: Optional[Callable[[], Any]] = None,
    ) -> dict:
        """Asynchronously route a single sample.

        The surrogate forward pass is synchronous and sub-millisecond, so it runs
        inline; only the optional ``fallback`` (used when the input is deferred)
        is awaited. As with :meth:`predict`, ``fallback`` takes no arguments and
        may be a plain callable or a coroutine function.

        Returns the same dict shape as :meth:`predict`.
        """
        out = self.predict(input)
        if out["decision"] == "deferred" and fallback is not None:
            out = dict(out)
            out["label"] = await _maybe_await(fallback())
        return out

    async def apredict_batch(
        self,
        inputs: Union[List[str], np.ndarray],
        fallback: Optional[Callable[[Any], Any]] = None,
        max_concurrency: int = 8,
    ) -> dict:
        """Asynchronously route a batch with concurrent fallback resolution.

        The surrogate runs once over the whole batch (vectorized), then any
        deferred items are sent to ``fallback(original_input)`` concurrently --
        bounded by ``max_concurrency`` -- instead of serially. ``fallback`` may
        be sync or async. This is the high-throughput counterpart to
        :meth:`predict_batch` for async apps that defer to an async teacher
        (e.g. an LLM API).

        Returns the same dict shape as :meth:`predict_batch`.
        """
        X = self._to_embeddings(inputs)
        self._check_dim(X)
        preds, handled, stage_id, scores = route_scored(self._stages, X)
        is_list = isinstance(inputs, list)
        n = len(X)
        labels: list = [None] * n
        decisions: list = ["deferred"] * n
        deferred_idx: list = []
        for i in range(n):
            if handled[i]:
                decisions[i] = "handled"
                labels[i] = self._idx_to_label.get(int(preds[i]), str(preds[i]))
            else:
                deferred_idx.append(i)

        if fallback is not None and deferred_idx:
            sem = asyncio.Semaphore(max(1, int(max_concurrency)))

            async def _resolve(i: int) -> None:
                original = inputs[i] if is_list else X[i]
                async with sem:
                    labels[i] = await _maybe_await(fallback(original))

            await asyncio.gather(*[_resolve(i) for i in deferred_idx])

        return {"labels": labels, "decisions": decisions, "handled": handled,
                "preds": preds, "stage_id": stage_id, "accept_scores": scores}
