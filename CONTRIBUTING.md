# Contributing to TRACER

Thank you for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/adrida/tracer
cd tracer
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -v
```

Tests use synthetic data and run in temporary directories -- no external dependencies or API keys required.

## Quick sanity check

```bash
tracer demo
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
  cli/
    main.py              <- tracer CLI entry point (fit, report, update, demo, serve)
    _ui.py               <- terminal formatting and progress display
```

## Adding a new surrogate model

Add a factory to the `_candidates()` dict in `src/tracer/fit/surrogate.py`. The model must implement the scikit-learn `fit` / `predict` / `predict_proba` interface.

## Adding a new pipeline family

Implement a `build_<name>(split, target_ta) -> dict` function in `src/tracer/fit/pipeline.py` following the same structure as `build_global`, `build_l2d`, and `build_rsb`. Register it in the `builders` dict inside `fit_frontier`.

## Submitting a PR

1. Fork the repo and create a branch from `main`
2. Make your changes with tests
3. Run `pytest tests/ -v` -- all tests must pass
4. Open a pull request with a clear description of what changed and why
