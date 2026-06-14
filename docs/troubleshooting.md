# Troubleshooting

Three snags people hit most often, what causes them, and how to fix each.

## `selected_method` is `null` (no policy deployed)

**Symptom.** After `tracer.fit(...)`, the manifest shows `selected_method: null`,
the `FitResult` notes say "No deployable pipeline met the target teacher-parity
threshold", and `tracer.load_router(...)` then raises `FileNotFoundError` on
`pipeline.joblib` (the file is only written when a policy actually deploys).
`tracer serve` reports `method=none` on `/health`.

**Why.** TRACER only deploys a surrogate that can hold your parity bar. If no
candidate reaches `target_teacher_agreement` while still covering at least
`min_deploy_coverage` of traffic, nothing is deployed on purpose, rather than
shipping a policy that silently disagrees with your teacher.

Common reasons:

- `target_teacher_agreement` set too high for the data (for example 0.99 on
  noisy teacher labels).
- `min_deploy_coverage` set so high that the only parity-respecting policies are
  filtered out.
- Too few traces, or embeddings that do not separate the classes.
- Heavy teacher label noise, which caps the agreement any surrogate can reach.

**Fix.**

1. Open `.tracer/frontier.json`. Each entry lists `best_coverage` and `best_ta`
   per target. This tells you the best parity actually achievable and at what
   coverage, so you can pick a realistic bar instead of guessing.
2. Lower the bar to something the frontier shows is reachable:
   ```python
   tracer.fit("traces.jsonl", embeddings=X,
              config=tracer.FitConfig(target_teacher_agreement=0.90,
                                      min_deploy_coverage=0.02))
   ```
3. Add more traces, or improve the embedding model so classes separate better.
4. If `best_ta` is capped well below your target across all candidates, suspect
   teacher label noise and audit the traces before re-fitting.

## Coverage drops between fits

**Symptom.** A re-fit (`tracer.update(...)`, or `tracer.fit` on a larger trace
set) reports a lower `coverage_cal` than the previous run at the same
`target_teacher_agreement`.

**Why.** This is expected, not a regression. TRACER holds parity fixed and lets
coverage float. When new traffic is harder, more diverse, noisier, or introduces
labels the surrogate has not seen, the only way to keep matching the teacher at
your target is to defer more inputs. Coverage falls so that the parity bound still holds.

**Fix.** Usually nothing: a lower coverage at the same parity is the system
adapting correctly. To understand the change:

- Check `.tracer/frontier.json` to see the new coverage versus parity curve.
- Read the qualitative report (`.tracer/qualitative_report.json`). Its
  `temporal_deltas` show, per label, how the handled rate moved between fits, so
  you can see which slices got harder.
- If new labels expanded the space, that alone can lower coverage; confirm the
  label count grew.
- If cost matters more than parity for the new traffic, lower
  `target_teacher_agreement` to trade some agreement back for coverage.

## Embedding dimension mismatch

**Symptom.**
```
ValueError: Embedding dimension mismatch: expected 384, got 768.
```
raised from `router.predict(...)` / `router.predict_batch(...)`, or surfaced as
a `500` from the server's `/predict` endpoint.

**Why.** The router records the embedding width it was fitted on in
`manifest.embedding_dim` and checks every incoming vector against it. A mismatch
means the embeddings at route time were produced differently from the ones used
at fit time, almost always a different embedder model.

**Fix.** Use the same embedder at fit time and route time.

- If you fit on precomputed embeddings, route with the same model that produced
  them. `all-MiniLM-L6-v2` and `BAAI/bge-small-en-v1.5` are 384-dim, while
  `BAAI/bge-base-en-v1.5` is 768-dim; fitting on one width and routing on the
  other triggers this error.
  ```python
  embedder = tracer.Embedder.from_sentence_transformers("BAAI/bge-small-en-v1.5")
  router = tracer.load_router(".tracer", embedder=embedder)  # same model as fit
  ```
- For the HTTP server, the vector in `{"embedding": [...]}` must have exactly
  `manifest.embedding_dim` values. Check `GET /health` and the manifest if you
  are unsure of the expected width.
- A dimension match with poor accuracy is a different problem: make sure the
  `normalize` setting also matches between fit and route.
