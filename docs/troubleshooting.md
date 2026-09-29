# Troubleshooting

TRACER's student returns fixed labels directly when its policy accepts them.
Otherwise your application retains its teacher path. These diagnostics concern
that policy; an acceptance score is not a certified correctness probability.
See [concepts](concepts.md) and [the API contract](api.md).

## `selected_method` is `null`

**Symptom:** `fit()` or `update()` returns a manifest with no selected method.
`load_router()` raises `ValueError`; `serve()` fails during loading, before its
HTTP server starts. There is no functioning `/health` endpoint for that policy.

**Meaning:** no candidate was selected, or the fixed policy failed its final
teacher-agreement/coverage check. A policy can fail with no observed mistakes
when too few reserved examples were accepted. The target bounds agreement with
the teacher, not independent ground-truth accuracy.

1. Inspect `result.manifest.certification` and `result.notes` for the exact
   status, accepted counts, observed agreement and lower bound.
2. Use `frontier.json` to understand development selection. Its scores are not
   final certification or proof of the best achievable production result.
3. Audit labels and collect representative independent examples. Check whether
   the representation separates the labels your application needs.
4. Keep using the teacher. Do not repeatedly change thresholds or random seeds
   against the reserved outcomes to obtain a passing certificate. A different
   quality requirement is an application decision, not a repair for failure.

A completed null-policy fit **replaces** the selected directory's generation;
it is different from an exception, which preserves the preceding generation.
To review a candidate before promoting it, fit to a separate directory:

```python
import tracer

candidate = tracer.fit("traces.jsonl", artifact_dir=".tracer-candidate", embeddings=X)
print(candidate.manifest.certification)
# Switch your application's configured directory only after your review/evals.
```

## Coverage drops between fits

A lower accepted fraction can reflect harder traffic, changed label proportions,
sampling variation, different selected models, or a regression. It is not by
itself proof that the system adapted correctly or that quality was preserved.

Compare old and new policies on the same external session/time holdout. Report
coverage and accepted agreement together; if independent reference labels are
available, also measure direct accuracy and per-class/worst-group failures.
Use the qualitative report and its per-label `temporal_deltas` for diagnosis,
not as an independent quality estimate. Record a representative sample of all
traffic: training only on the old policy's deferrals changes the distribution.

Updates refit and recheck a policy; there is no promised coverage improvement or
automatic production retraining schedule in the OSS package.

## Embedding dimension mismatch or unexpected behavior

```text
ValueError: Embedding dimension mismatch: expected 384, got 768.
```

`predict()` and `predict_batch()` check width against `manifest.embedding_dim`.
The HTTP server returns **400**, not 500, for this input error. Use the exact
same encoder, revision, preprocessing and normalization as during fitting.
Two encoders can share a dimension and still be incompatible; dimension checks
do not validate model identity.

```python
embedder = tracer.Embedder.from_sentence_transformers("BAAI/bge-small-en-v1.5")
router = tracer.load_router(".tracer", embedder=embedder)  # only if fitted with this encoder
```

The HTTP interface takes `{"embedding": [...]}`, not raw text. It does not
compute embeddings or call your fallback. Read the dimension from the manifest;
`GET /health` does not expose it. `Embedder.from_endpoint()` uses literal
top-level response keys, so `"data.0.embedding"` is not a nested-path adapter.
Use `from_callable()` to transform a provider's nested response.

## A request is deferred, including an unfamiliar input

Ordinary acceptance rejection and the OOD distance guard both return
`decision="deferred"`. There is no separate public OOD-reason field.
Without a supplied fallback, the result has `label=None`. With one, its return
value becomes the label but the decision stays `"deferred"`. An OOD guard is a
heuristic distance check, not a guarantee that all wrong inputs are detected.

Batch results have `None` for deferred labels and never run a teacher. Ignore
internal `preds` where `handled` is false. Calling a fallback does not establish
that its answer is correct; evaluate its outputs separately when reporting
end-to-end quality.

## A saved policy no longer loads

A missing or unreadable declared OOD guard fails loading instead of silently
disabling it. Restore the matching complete artifact generation; do not delete
the guard to make loading succeed. Legacy artifacts without final evidence do
not become certified when reloaded.

`update()` needs `all_traces.jsonl` and its aligned saved embeddings. If either
is lost, reconstruct from your source records and refit; embeddings alone cannot
recover teacher labels. Use one artifact writer at a time, keep enough disk
space for staged generations, and reload readers after publication finishes.
See [artifacts](artifacts.md).
