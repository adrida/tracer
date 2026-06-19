# Changelog

All notable changes to TRACER are recorded here. This project follows semantic
versioning.

## 0.3.0 (2026-06)

### Added
- `tracer.watch`: a decorator (and HTTP fallback) that observes the LLM calls
  your pipeline already makes and ships them, free, to Tracer Cloud, where they
  accumulate and auto-optimize once there are enough. Captures the full request
  and response on the OpenTelemetry GenAI (`gen_ai.*`) schema, with pluggable
  sinks (local file, Tracer Cloud, OTLP, or several at once).
- `@tracer-llm/watch`: a zero-dependency JavaScript/TypeScript mirror of the watch
  decorator (async-context aware), so JS pipelines get the same one-line
  observability as Python.
- `tracer cloud`: a full command-line interface for Tracer Cloud, at parity with
  the dashboard. Browser or password login, then manage tracers (create, quick
  and bulk uploads, agentic `onboard`, rename, delete), training (retrain,
  auto-retrain, promote/rollback, live status), routing (`route`), test
  batteries, the model library, API keys, observability ingest keys, billing,
  analytics, trace selection (`traces`), and a public `scan`.
- Lazy package initialization so `import tracer` stays fast and only pulls in the
  heavy pieces when you actually use them.

### Changed
- `tracer scan` is more robust on messy uploads: tolerant input/label aliasing
  and a clarification path so ambiguous files still produce a result.

### Docs
- New guides for the watch decorator and the `tracer cloud` CLI.

## 0.2.0 (2026-06)

### Added
- `tracer scan`: a fast, conservative day-one read of a traces file, before any
  training. It groups traffic by similarity and measures, on a held-out slice it
  never saw, how much a near-free model can answer at your target agreement,
  using exact Clopper-Pearson bounds, with an optional per-1k and monthly savings
  estimate. Ships a self-contained HTML report with an interactive 3D map of the
  embedding space (hover-to-inspect cells, a Verdict/Label colour toggle, and
  PCA/UMAP/t-SNE layouts). Exposed as `tracer.scan()`.
- Distance-based OOD safety gate: at inference the router defers inputs that fall
  far from the training distribution (kNN distance, global and per-predicted-label
  thresholds) regardless of surrogate confidence, so off-distribution traffic goes
  to the teacher instead of getting a confident guess.
- `tracer fit --trees` to opt in the tree surrogates, and `--skip` to drop named
  candidates from the zoo.
- Trace loaders accept common key aliases for both input and label
  (`input/query/text/prompt/question` and
  `teacher/teacher_output/label/intent/output/answer`).
- Bring-your-own embeddings for `tracer scan`: local sentence-transformers by
  default (`--embed-model`), a precomputed `.npy` (`--embeddings`), or your own
  HTTP embedding endpoint (`--embed-url`, with header and response-key options).

### Changed
- The parity gate now certifies on an exact held-out lower bound instead of an
  in-sample point estimate, so a policy cannot clear the target by in-sample luck
  and then break the contract on real traffic. Coverage is now monotonic in the
  target, and a hybrid select-then-verify procedure recovers coverage at strict
  targets that a plain held-out split discarded.
- Tree surrogates (decision tree, random forest, extra-trees, gradient boosting)
  are now off by default in `tracer fit`; the default zoo is the fast linear and
  MLP heads. Use `--trees` for hard, high-class-count tasks.
- The HTML report is restyled to the light Tracer theme, and the word "audit" is
  dropped across the report and docs.

### Fixed
- Non-monotonic coverage in the gate (a stricter target could deploy more coverage
  than a looser one).
- NaN-robustness in acceptor fitting on degenerate surrogates.

## 0.1.3

Initial public releases: the parity-gated router (`fit`, `update`, `load_router`,
`serve`), the HTML report and Sankey diagram, and the embedder factories.
