# Contributing to TRACER

Thank you for your interest in contributing.

TRACER builds task-specific classifiers: classical or
shallow neural classifiers make direct fixed-label decisions, while a learned
policy leaves other requests to the caller's teacher. The final check bounds
teacher agreement under stated sampling assumptions. It does not guarantee
ground-truth accuracy, per-request correctness or robustness to distribution
shift. Contributions should preserve those distinctions and provide evidence
for changes in behavior.

## Contribution standards (read first)

TRACER is benchmark-driven, so changes that affect routing behavior are easiest
for us to act on when they come with numbers. Three guidelines follow from that.

1. **Adding a model: bring a benchmark, not just a name.** A request that only
   names a model to add is hard for us to evaluate, because we can't tell
   whether it would actually help. A model proposal is most useful when it
   shows, on a reproducible eval, how the new surrogate compares to the models
   already in the zoo on the coverage-vs-agreement frontier (see "Adding a new
   surrogate model" below). Without that, we'll usually ask for a benchmark
   before we can move forward, and we're glad to point you at how to run one.

2. **Changing a core parameter, threshold, or gate: bring an ablation.** The
   acceptor threshold, the parity gate, the calibration logic, the default
   `FitConfig` values, and the split strategy determine the statistical contract.
   A change to any of them is much easier to accept with before/after
   numbers on the eval showing the frontier does not regress (and, ideally,
   improves). If you think a parameter should be user-tunable, the most
   convincing case is one where a different value measurably wins.

3. **Bug reports: include a reproduction.** The reports we can act on fastest
   name the affected files and lines and include a minimal script that
   reproduces the behavior. Issues #29, #30, and #31 are good examples.

If your idea is exploratory ("how would TRACER handle X?", "is this a good fit
for Y?"), open a **GitHub Discussion** rather than an issue. Issues are for
actionable, evidenced work.

## What "benchmarked" means here

Run against a fixed, public eval so results are reproducible by a maintainer:

- Use a known dataset (the repo's test fixtures, or a public set such as
  Banking77) and a fixed `seed`.
- Report the metric that matters: **coverage at the target teacher agreement**
  together with accepted counts, realized agreement, the final lower bound and
  confidence level. Use an untouched evaluation split, not the examples used
  for classifier/threshold selection. Where ground truth exists, report its
  accuracy separately, including per-class or worst-group behavior.
- Measure full-path latency and cost when claiming efficiency: embeddings,
  student inference, teacher deferrals and relevant serving overhead. A gate's
  score is not a calibrated per-request correctness probability.
- Separate sessions/workflows/time where the dataset permits; random row splits
  do not establish independence. Preserve historical benchmark results and label
  new results with their exact implementation and evaluation protocol.
- Compare against the current behavior on `main`, so the delta is clear.
- Include the exact command and config you ran so the numbers can be checked.

Performance proposals should show their tradeoff. Correctness fixes, truthful
contracts and reproducible failure handling are valuable even when they reduce
previously overstated coverage; they do not need an invented benchmark win.
Never retune a policy against its final certification outcomes to make it pass.

## Setup

Use Python 3.12 or newer. CI tests 3.12, 3.13, and 3.14 with current
dependencies, plus 3.12 with the minimum supported core dependencies.

```bash
git clone https://github.com/adrida/tracer
cd tracer
pip install -e ".[dev]"
```

## Running tests

```bash
TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
pytest tests/ -v
```

Tests use synthetic data and run in temporary directories. No model downloads
or API keys are required. To check the JavaScript recorder:

```bash
cd js
npm ci
npm test -- --maxWorkers=1 --minWorkers=1
npm run build
```

## Quick sanity check

```bash
pytest tests/test_router.py tests/test_watch.py -q
```

## Project structure

```
src/tracer/
  __init__.py            <- public exports (fit, update, load_router, report, embed, types)
  api.py                 <- public API (fit, update, load_router, report)
  config.py              <- FitConfig, EmbeddingConfig
  types.py               <- TraceRecord, QualitativeReport, ArtifactManifest, ...
  fit/
    pipeline.py          <- global / L2D / RSB pipeline construction + calibration
    certification.py     <- untouched final-policy teacher-agreement check
    ood.py               <- distance guard, included in final serving check
    surrogate.py         <- model zoo (LogReg, SGD, MLP, RF, ET, DT, GBT, XGB) + selection
  analysis/
    qualitative.py       <- XAI report: slices, boundary pairs, examples, deltas
    html_report.py       <- self-contained HTML audit report generator
  embeddings/
    index.py             <- FAISS wrapper + embed_texts (sentence-transformers)
    embedder.py          <- Embedder class (sentence-transformers, HTTP, callable)
  traces/
    loader.py            <- JSONL loader / writer + validation
  policy/
    artifacts.py         <- manifest, pipeline, qualitative report I/O
  runtime/
    router.py            <- production Router class
    serve.py             <- lightweight HTTP prediction server (stdlib only)
  scanner.py             <- pre-training traffic diagnostics and reports
  watch.py               <- local trace recording and generic export sinks
```

## Adding a new surrogate model

We're happy to grow the zoo when a model earns its place on the frontier. The
benchmark is what tells us it does.

1. Add a factory to the `_candidates()` dict in `src/tracer/fit/surrogate.py`.
   The model must implement the scikit-learn `fit` / `predict` / `predict_proba`
   interface.
2. Include a benchmark in the PR description showing the new model wins (higher
   coverage at the target agreement, or equal coverage at lower latency) on a
   reproducible eval, against the existing zoo. Show the command and seed.
3. Account for the dependency cost. A new heavy dependency needs to pay for
   itself in the numbers, and should be optional where possible.

If a PR adds a model without (2), we'll ask for the benchmark before reviewing
further. It's usually quick to run, and it's the thing that lets us say yes.

## Changing the gate, calibration, or default config

These parts define selection and final certification. Performance changes need
an ablation on the same external evaluation and seed. Contract/correctness fixes
need focused regressions that demonstrate the broken behavior and its repair.
Verify that no final outcomes influence training, threshold selection, OOD
fitting or retries. Keep no-policy results, exceptions and legacy-artifact
behavior explicit; a completed failed check and a failed write are different.

## Adding a new pipeline family

Implement a `build_<name>(split, target_ta) -> dict` function in
`src/tracer/fit/pipeline.py` following the same structure as `build_global`,
`build_l2d`, and `build_rsb`. Register it in the `builders` dict inside
`fit_frontier`, preserving the builders' keyword options. Include a benchmark
showing where the new family wins. The full selected policy, including residual
stages and OOD checks, must receive the final check and match runtime behavior.

## Documentation and integration boundaries

Check examples against current function signatures and saved artifact fields.
The bundled HTTP server takes vectors and does not call teachers. Generic
embedding/export adapters do not imply a turnkey provider integration or
standards-compliant OTLP transport. The current package exposes Python APIs,
not a packaged CLI; hosted catalogs, credentials and Echo wallets are separate.
Document proposed features as proposals rather than existing OSS capabilities.

## Submitting a PR

1. Fork the repo and create a branch from `main`.
2. Make your changes with tests.
3. Run `pytest tests/ -v`. All tests must pass.
4. Open a pull request with a clear description of what changed, why, and (for
   anything touching routing behavior) the benchmark that backs it.
