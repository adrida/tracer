# Changelog

All notable changes to TRACER are recorded here. This project follows semantic
versioning.

## Unreleased

These entries describe repository source. They do not announce a new PyPI or
npm release; check the installed package version before relying on them.

### Documentation
- Document direct student predictions, teacher deferral and the OOD guard.
- Add an [overview](docs/system1.md) covering the teacher/student contract,
  meaningful quality and cost measurements, free self-hosting, and the boundary
  between this library and the hosted application's managed features.
- Clarify that OSS OOD rejection is currently reported as `deferred`, that
  acceptance scores are not certified per-request probabilities, and that
  fixed-label fitting does not train a general zero-shot model or its encoder.
- Correct stale integration examples and unsupported latency, savings and
  monotonic-coverage claims. Keep historical paper results identified as such.

### Fixed
- Make direct `fit()` transactional as well as `update()`: partial reports,
  indexes, or manifests cannot replace a previously working generation.
- Fail loading when a present or required OOD guard is unreadable or missing;
  preserve legacy unguarded artifacts without inventing a certificate.
- Reserve final certification rows before fitting, label discovery or balancing.
  Check the selected serving policy, including its development-only OOD guard,
  once using a one-sided agreement lower bound; never retry other candidates
  using these outcomes. Zero accepted examples cannot certify a policy.
- Preserve the previous artifact generation when an update fails validation,
  fitting, or publication. Honor explicit update configuration without mutating
  the caller, retain saved configuration by default, and increment the fit count.
- Preserve classifier label IDs in residual stages and select the largest
  eligible acceptance set during small-data calibration.
- Keep original embedding scales when building FAISS indexes and reuse the OOD
  nearest-neighbor index across predictions. Existing FAISS artifacts trained
  with unnormalized inputs need refitting to recover the original scale.
- Apply configured seeds to subsampling, splits, calibration, and model training.
- Record async outputs, errors, cancellation, timing, and parent spans correctly;
  isolate tracing storage/export failures from application calls.
- Escape trace and artifact text in reports and reject unsafe watcher filenames
  in Python and JavaScript.
- Preserve deferred confidence scores in reports; validate prediction shapes and
  finite values; report actual file lines for null teacher labels.
- Handle concurrent HTTP requests and client disconnects, reject malformed or
  oversized request bodies, and close the server cleanly on interruption.
- Correct update examples to use `new_embeddings=`.

### Changed (breaking)
- New fits reserve 20% of rows by default (`certification_fraction`) and use
  `certification_alpha=0.10`. Small or uncertain fits may now decline deployment.
  `manifest.certification` records the result; legacy-named `coverage_cal` and
  `teacher_agreement_cal` now measure this final partition for new artifacts.
  The confidence statement requires independent representative examples and
  does not cover distribution shift, correlated sessions, or adaptive reuse.
- Persist `ood_reference.npy` separately from all embeddings used for updates.
  Existing artifacts keep their previous behavior unless their guard is broken.
- Remove unsupported universal coverage, savings and monotonic-growth claims.
- The prediction server binds to `127.0.0.1` by default and no longer sends a
  wildcard CORS header. Explicitly configure a remote bind address when needed;
  browser deployments should configure CORS at their authenticated proxy.
- Watcher filenames must start with a letter or digit and use only ASCII
  letters, digits, dots, underscores, or hyphens, up to 128 characters.
- Invalid prediction inputs return HTTP 400 rather than 500.
- Require Python 3.12 or newer; drop support for Python 3.9–3.11. CI tests
  Python 3.12, 3.13, and 3.14, plus the minimum core dependencies on 3.12.
- Allow NumPy 1.26.4 through 2.x so Python 3.14 can install compatible wheels.
  Require scikit-learn 1.4.2+ and joblib 1.3.2+ in core, and PyTorch 2.4.1+
  for the optional embedding extras. Remove the obsolete NumPy downgrade advice.
- Build releases and run the documented Python sidecar on Python 3.14.

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

Historical release notes. The scan remains a diagnostic; the calibration
procedure described here is superseded by the independent final-policy check
in Unreleased. These entries do not establish current deployment guarantees.

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
- The parity gate introduced held-out lower-bound checks and a hybrid
  select-then-verify procedure. Later review found that selection and final
  certification must use separate data; see Unreleased. This historical
  procedure does not guarantee agreement on future traffic or monotonic
  coverage across targets.
- The HTML report is restyled to the light Tracer theme, and the word "audit" is
  dropped across the report and docs.

### Fixed
- Non-monotonic coverage in the gate (a stricter target could deploy more coverage
  than a looser one).
- NaN-robustness in acceptor fitting on degenerate surrogates.

## 0.1.3

Initial public releases: the parity-gated router (`fit`, `update`, `load_router`,
`serve`), the HTML report and Sankey diagram, and the embedder factories.
