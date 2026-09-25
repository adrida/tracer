# Changelog

All notable changes to TRACER are recorded here. This project follows semantic
versioning.

## Unreleased

### Removed (breaking)
- The `tracer` command-line executable and command implementations. Use the
  Python APIs for fitting, scanning, updating, reports, and serving predictions.
- The Tracer app client, authentication, and automatic app telemetry exporters
  in both Python and JavaScript. Existing app credentials no longer trigger
  uploads. Local recording, custom sinks, and explicitly configured generic
  HTTP exports remain available.
- App connection instructions and command-line documentation. All current
  examples use the library APIs.

## 0.3.3 (2026-06)

### Fixed
- `build_global()` and `_build_accepting_stage()` no longer crash with a
  `TypeError` when all surrogate candidates fail to train. They now return a
  structured failure state, consistent with every other non-deployable path.
- `Router.load()` now raises a clear `ValueError` when `manifest.json`
  reports `selected_method: null`, instead of silently loading a stale
  `pipeline.joblib` left on disk from a previous fit.

## 0.3.2 (2026-06)

### Fixed
- `load_traces()` now reports the correct 1-indexed file line number in
  `ValueError` messages. Previously the count excluded blank lines, so
  the reported line was lower than the actual position of the bad record.

## 0.3.0 (2026-06)

### Added
- `tracer.watch`: a decorator that records LLM calls on the OpenTelemetry
  GenAI (`gen_ai.*`) schema, with pluggable local and generic export sinks.
- `@tracer-llm/watch`: a zero-dependency JavaScript/TypeScript mirror of the watch
  decorator (async-context aware), so JS pipelines get the same one-line
  observability as Python.
- Lazy package initialization so `import tracer` stays fast and only pulls in the
  heavy pieces when you actually use them.

### Changed
- Trace ingestion accepts additional input and label aliases.

### Docs
- New guide for the watch decorator.

## 0.2.0 (2026-06)

### Added
- `tracer.scan()`: a fast, conservative day-one read of a traces file, before any
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
- `FitConfig.skip_candidates` to drop named candidates from the zoo, including
  tree surrogates when a lighter sweep is desired.
- Trace loaders accept common key aliases for both input and label
  (`input/query/text/prompt/question` and
  `teacher/teacher_output/label/intent/output/answer`).
- Bring-your-own embeddings for `tracer.scan()`: pass a precomputed array with
  `embeddings=`, or select a local sentence-transformers model with `model=`.

### Changed
- The parity gate now certifies on an exact held-out lower bound instead of an
  in-sample point estimate, so a policy cannot clear the target by in-sample luck
  and then break the contract on real traffic. Coverage is now monotonic in the
  target, and a hybrid select-then-verify procedure recovers coverage at strict
  targets that a plain held-out split discarded.
- The HTML report is restyled to the light Tracer theme, and the word "audit" is
  dropped across the report and docs.

### Fixed
- Non-monotonic coverage in the gate (a stricter target could deploy more coverage
  than a looser one).
- NaN-robustness in acceptor fitting on degenerate surrogates.

## 0.1.3

Initial public releases: the parity-gated router (`fit`, `update`, `load_router`,
`serve`), the HTML report and Sankey diagram, and the embedder factories.
