# TRACER

**Trace-Based Adaptive Cost-Efficient Routing**

[![arXiv](https://img.shields.io/badge/arXiv-2604.14531-b31b1b.svg)](https://arxiv.org/abs/2604.14531)
[![Hugging Face](https://img.shields.io/badge/🤗%20HF-Papers-yellow)](https://huggingface.co/papers/2604.14531)
[![PyPI](https://img.shields.io/pypi/v/tracer-llm)](https://pypi.org/project/tracer-llm/)
[![Downloads](https://static.pepy.tech/badge/tracer-llm)](https://pepy.tech/project/tracer-llm)
[![Downloads](https://static.pepy.tech/badge/tracer-llm/month)](https://pepy.tech/project/tracer-llm)
[![Python](https://img.shields.io/pypi/pyversions/tracer-llm)](https://pypi.org/project/tracer-llm/)
[![npm](https://img.shields.io/npm/v/@tracer-llm/watch?label=%40tracer-llm%2Fwatch&color=cb3837&logo=npm)](https://www.npmjs.com/package/@tracer-llm/watch)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](https://github.com/adrida/tracer/actions)
[![Docs](https://img.shields.io/badge/docs-reference-blue)](docs/)

Most LLM-based classification pipelines use a large language model for every single input. In practice, the vast majority of that traffic is predictable - a lightweight traditional ML model (logistic regression, gradient-boosted trees, or a small neural net) can match the LLM's output with near-perfect agreement.

TRACER learns the decision boundary between "easy" and "hard" inputs directly from your LLM's own classification traces. It fits a fast, non-LLM surrogate on the easy partition, gates it with a calibrated acceptor, and defers only the uncertain inputs back to the LLM. Every deferred call produces a new trace, which feeds the next refit - coverage grows automatically over time. The result: **90%+ of classification calls routed to traditional ML, with formal parity guarantees against the teacher LLM and a self-improving routing policy**.

```bash
pip install tracer-llm
```

## Quickstart

Input: a JSONL file where each line contains the original text (`input`) and the label your LLM assigned (`teacher`).

```python
import tracer

# 1. Fit - learn a routing policy from your LLM's classification traces
result = tracer.fit(
    "traces.jsonl",                  # {"input": "...", "teacher": "label"} per line
    embeddings=X,                    # np.ndarray (n, dim) - precomputed text embeddings
    config=tracer.FitConfig(target_teacher_agreement=0.95),
)

# 2. Route - surrogate handles easy inputs, LLM handles the rest
router = tracer.load_router(".tracer", embedder=embedder)
out = router.predict("What is my balance?")
# {"label": "check_balance", "decision": "handled", "accept_score": 0.96}

# 3. Fallback - only invokes the LLM when the surrogate declines
out = router.predict("Some edge case", fallback=lambda: call_my_llm(text))
```
The [API reference](docs/api.md) covers fitting, routing, updates, and reports.
See [concepts](docs/concepts.md) for the pipeline and [watch](docs/watch.md) for
local trace recording and opt-in exports to your own backend.

## Watch your LLM traffic

Before you fit anything, just *watch*. Wrap any LLM call and every request is
recorded locally as an OpenTelemetry GenAI span, no account, no key, nothing
leaves your machine:

```python
import tracer

watch = tracer.watch("support_classifier", system="my-provider", model="my-model")

@watch
def classify(ticket: str) -> str:
    return call_my_llm(ticket)   # traces append to .tracer/watch/*.jsonl
```

The same watched spans map 1:1 to `TraceRecord`, so once you have traffic you can
call `tracer.fit()` to train a router from it. Full guide: [docs/watch.md](docs/watch.md).

## Using from JavaScript / Node.js

**Watch your JS LLM calls (free observability):** [`@tracer-llm/watch`](https://www.npmjs.com/package/@tracer-llm/watch) mirrors the Python decorator with zero dependencies, recording every call as an OpenTelemetry GenAI span (local by default, with opt-in export to your own backend).

```bash
npm install @tracer-llm/watch
```

```js
import { watch } from "@tracer-llm/watch";

const w = watch("support_classifier", { system: "provider-x", model: "model-x" });

// Wrap the function that calls your model; the return value is auto-captured.
const classify = w(async (ticket) => callYourLLM(ticket));
```

Full guide: [docs/javascript.md](docs/javascript.md). To route (not just observe) from JS, log traces, fit offline with `tracer.fit()`, run `tracer.serve()` in a Python sidecar, and call it via `fetch`:

```js
// 1. Log every LLM classification
fs.appendFileSync('traces.jsonl', JSON.stringify({ input: text, teacher: label }) + '\n')

// 2. At inference: embed → POST to TRACER → fallback to LLM only if deferred
let { label, decision } = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: JSON.stringify({ embedding }),  // same model you used at fit time
}).then(r => r.json())

if (decision === 'deferred') label = await callYourLLM(text)
```

See the [JavaScript integration guide](docs/javascript.md) for the full setup including embeddings, docker-compose, batch prediction, and continual learning.

## How it works

```
User query → [Embedder] → [ML Surrogate] → [Acceptor Gate]
                                                |          |
                                            score >= t   score < t
                                                |          |
                                          Local answer   Defer to LLM
                                          (traditional ML)
```

The surrogate is **not another LLM** - it is a classical ML or shallow DL model. The Python API searches linear, neural, and tree-based candidates. For a lighter sweep, pass `FitConfig(skip_candidates=("dt", "rf", "et", "gbt", "xgb"))` to exclude tree models. Inference runs locally on your CPU.

1. **Fit** - train a suite of candidate surrogates on your LLM's classification traces; select the best via cross-validated teacher agreement
2. **Gate** - attach a learned acceptor that estimates, per-input, whether the surrogate will agree with the teacher
3. **Calibrate** - sweep the acceptor threshold to maximise coverage at your target parity (e.g. ≥ 95% teacher agreement)
4. **Guard** - block deployment if the best candidate cannot clear the parity bar on held-out data

## Benchmark results (Banking77 - 77-class intent classification)

| Metric | Value |
|--------|-------|
| Coverage | **92.2%** of traffic handled locally |
| Teacher agreement (handled) | 96.1% |
| End-to-end accuracy | 96.4% |
| **Annual savings** (10k queries/day) | **$302,850** |

_Banking77 is a 77-class task; these results include tree-based candidates. Candidate selection and coverage depend on your data._

## Continual learning flywheel

TRACER is not a one-shot fit. Every deferred input that reaches the LLM produces a new labeled trace, which feeds back into the next refit. As the surrogate sees more of the input distribution, its coverage grows - meaning fewer LLM calls, which in turn cost less, while the quality guarantee holds at every iteration.

```
Day 1:  2,000 traces → 84% coverage → 1,600 calls/day saved
Day 3:  6,000 traces → 90% coverage → 9,000 calls/day saved
Day 5: 10,000 traces → 92% coverage → 9,200 calls/day saved
```

```python
tracer.update("new_traces.jsonl", embeddings=X_new)  # refit with new production traces
```

The parity gate re-calibrates on each update, so coverage only increases when the surrogate actually earns it.

## Embedder options

```python
from tracer import Embedder

embedder = Embedder.from_sentence_transformers("BAAI/bge-small-en-v1.5")  # local
embedder = Embedder.from_endpoint("https://api.example.com/embed", headers={...})  # API
embedder = Embedder.from_callable(my_fn)  # any function
# or skip the embedder and pass raw np.ndarray embeddings directly
```

Need to compute embeddings at fit time?

```bash
pip install tracer-llm[embeddings]   # adds sentence-transformers
```

```python
X = tracer.embed(texts)  # default: all-MiniLM-L6-v2 (384-dim)
```

## Inspect traffic before fitting

`tracer.scan()` groups traces by similarity and estimates the certifiable share
using held-out bounds. Pass the embeddings that correspond to your trace rows:

```python
from pathlib import Path
import tracer
from tracer.scanner import scan_html

scan = tracer.scan("traces.jsonl", embeddings=X, target=0.95)
Path("scan.html").write_text(scan_html(scan), encoding="utf-8")
```

A scan needs at least 1,000 traces; around 5,000 is recommended. Use `force=True`
only for an explicitly marked thin-data estimate. It does not train a router;
use `tracer.fit()` for training. See the [scan guide](docs/scan.md).

## What's in `.tracer/`

| File | Contents |
|------|----------|
| `manifest.json` | Method, coverage, teacher agreement, label space |
| `pipeline.joblib` | Surrogate + acceptor + calibrated thresholds |
| `frontier.json` | All candidates at each quality target |
| `qualitative_report.json` | Per-label slices, boundary pairs, examples |
| `report.html` | Visual HTML report |

## Install

```bash
pip install tracer-llm                # core (numpy + sklearn + joblib)
pip install tracer-llm[embeddings]    # + sentence-transformers
pip install tracer-llm[all]           # everything
```

## Docs

| | |
|---|---|
| [Concepts](docs/concepts.md) | Pipeline internals, model zoo, parity gate |
| [API reference](docs/api.md) | Every function, parameter, and return type |
| [Scan](docs/scan.md) | Inspect traffic before training |
| [Watch](docs/watch.md) | Record calls locally and configure generic exports |
| [JavaScript / Node.js](docs/javascript.md) | Full integration guide for JS pipelines |
| [Artifacts](docs/artifacts.md) | `.tracer/` directory schema |
| [Troubleshooting](docs/troubleshooting.md) | `selected_method=null`, coverage drift, embedding-dim mismatch |
| [AGENTS.md](AGENTS.md) | Integration guide for AI coding assistants |

## Research

### TRACER paper

**TRACER: Trace-Based Adaptive Cost-Efficient Routing for LLM Classification**  
Adam Rida, arXiv 2026

[![arXiv](https://img.shields.io/badge/arXiv-2604.14531-b31b1b.svg)](https://arxiv.org/abs/2604.14531) [![Hugging Face](https://img.shields.io/badge/🤗%20HF-Papers-yellow)](https://huggingface.co/papers/2604.14531)

```bibtex
@article{rida2026tracer,
  title   = {TRACER: Trace-Based Adaptive Cost-Efficient Routing for LLM Classification},
  author  = {Rida, Adam},
  journal = {arXiv preprint arXiv:2604.14531},
  year    = {2026}
}
```

### Pricing Capability

**Pricing Capability: A Framework for Valuing Proprietary Data Assets in the Age of AI**

Adam Rida, Tracer AI, Inc., August 25, 2026

[PDF in this repository](research/pricing-capability.pdf)

This repository preserves the 21-page final submission dated August 25, 2026. The paper defines Capability Alpha under controlled evaluation, develops Price-Implied Uplift, studies Spirit Aviation's selected data-and-software bid, and compares four public AI data and content agreements. The public PDF preserves the final submission's text, bookmarks, citations, and internal navigation, corrects one internal section reference, and adds descriptive document metadata. It is independent research without a peer review or journal acceptance claim.

SHA-256: `72f2dfa8385c4ea4c74297dc2d1a9f1a193fd447f5452138dfef2dda8778c65c`

## License

MIT
