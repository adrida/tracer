# TRACER

**Trace-Based Adaptive Cost-Efficient Routing**

[![PyPI](https://img.shields.io/pypi/v/tracer-llm)](https://pypi.org/project/tracer-llm/)
[![Python](https://img.shields.io/pypi/pyversions/tracer-llm)](https://pypi.org/project/tracer-llm/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](https://github.com/adrida/tracer/actions)
[![Docs](https://img.shields.io/badge/docs-github-blue)](https://github.com/adrida/tracer/tree/main/docs)

Most LLM-based classification pipelines use a large language model for every single input. In practice, the vast majority of that traffic is predictable - a lightweight traditional ML model (logistic regression, gradient-boosted trees, or a small neural net) can match the LLM's output with near-perfect agreement.

TRACER learns the decision boundary between "easy" and "hard" inputs directly from your LLM's own classification traces. It fits a fast, non-LLM surrogate on the easy partition, gates it with a calibrated acceptor, and defers only the uncertain inputs back to the LLM. Every deferred call produces a new trace, which feeds the next refit - coverage grows automatically over time. The result: **90%+ of classification calls routed to traditional ML, with formal parity guarantees against the teacher LLM and a self-improving routing policy**.

```bash
pip install tracer-llm
```

## See it work

```bash
tracer demo
```

```
  TRACER  Demo - Banking77 (77 intents · 1,500 traces)

  Routing Policy
  method      l2d
  coverage    80.7%   of traffic handled by surrogate
  teacher TA  0.951   surrogate matches teacher on handled traffic

  Cost Projection (10k queries/day)
      Without TRACER   10,000 LLM calls/day   $20.00/day
      With TRACER       1,926 LLM calls/day   $ 3.85/day   $5,894 saved/yr
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

> **Want to go deeper?** The [concepts guide](docs/concepts.md) explains the full pipeline, model zoo, and parity gate. The [API reference](docs/api.md) covers every parameter. The [CLI reference](docs/cli.md) covers `tracer fit`, `tracer serve`, and more.

## How it works

```
User query → [Embedder] → [ML Surrogate] → [Acceptor Gate]
                                                |          |
                                            score >= t   score < t
                                                |          |
                                          Local answer   Defer to LLM
                                          (traditional ML)
```

The surrogate is **not another LLM** - it is a classical ML or shallow DL model (the model zoo includes logistic regression, SGD, gradient-boosted trees, random forests, and small feed-forward nets). This is what makes the cost reduction real: inference is CPU-bound, sub-millisecond, and free.

1. **Fit** - train a suite of candidate surrogates on your LLM's classification traces; select the best via cross-validated teacher agreement
2. **Gate** - attach a learned acceptor that estimates, per-input, whether the surrogate will agree with the teacher
3. **Calibrate** - sweep the acceptor threshold to maximise coverage at your target parity (e.g. ≥ 95% teacher agreement)
4. **Guard** - block deployment if the best candidate cannot clear the parity bar on held-out data

## Explainability

Every fit produces a structured audit report alongside the routing policy -- not just aggregate numbers.

| Report component | What it answers |
|-----------------|-----------------|
| **Per-label slice stats** | Which intents are fully self-serve? Which always defer? |
| **Contrastive boundary pairs** | For each label: one handled example, one deferred -- what makes an input "easy" vs "hard"? |
| **Representative examples** | What does typical handled/deferred traffic look like? |
| **Temporal deltas** | Which intents improved between the last refit and this one? |

```python
result = tracer.fit("traces.jsonl", embeddings=X)
qr = result.qualitative_report

print(qr.summary)
# "Handled 9278/10003 (92.8%) by surrogate"

for bp in qr.boundary_pairs[:3]:
    print(f"{bp.teacher_label}: handled={bp.handled_preview!r} / deferred={bp.deferred_preview!r}")

# Generate self-contained HTML report
tracer.generate_html_report(".tracer")
```

## Benchmark results (Banking77 - 77-class intent classification)

| Metric | Value |
|--------|-------|
| Coverage | **92.2%** of traffic handled locally |
| Teacher agreement (handled) | 96.1% |
| End-to-end accuracy | 96.4% |
| **Annual savings** (10k queries/day) | **$302,850** |

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

## CLI

| Command | What it does |
|---------|-------------|
| `tracer demo` | Zero-setup demo on real data |
| `tracer fit traces.jsonl --target 0.95` | Fit a routing policy |
| `tracer update new_traces.jsonl` | Refit with new traces |
| `tracer report-html` | Open the HTML audit report |
| `tracer serve .tracer --port 8000` | HTTP prediction server |

## What's in `.tracer/`

| File | Contents |
|------|----------|
| `manifest.json` | Method, coverage, teacher agreement, label space |
| `pipeline.joblib` | Surrogate + acceptor + calibrated thresholds |
| `frontier.json` | All candidates at each quality target |
| `qualitative_report.json` | Per-label slices, boundary pairs, examples |
| `report.html` | Visual audit report |

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
| [CLI reference](docs/cli.md) | `tracer fit`, `tracer serve`, `tracer demo`, and more |
| [Artifacts](docs/artifacts.md) | `.tracer/` directory schema |
| [AGENTS.md](AGENTS.md) | Integration guide for AI coding assistants |

## Prior work

TRACER builds on and connects three research threads:

**Learn to Defer (L2D)** -- the algorithmic backbone. The surrogate + acceptor gate architecture is grounded in Mozannar & Sontag ([ICML 2020](https://proceedings.mlr.press/v119/mozannar20b.html)), Madras et al. ([NeurIPS 2018](https://papers.nips.cc/paper_files/paper/2018/hash/09d37c08f7b129e96277388757530c72-Abstract.html)), and Mao et al. ([NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7e1bf09d2e4e6e5c5a0b63b66b47ef42-Abstract.html)).

**Post-hoc calibration** -- the threshold sweep that enforces the parity guarantee follows the post-hoc deferral estimator framework of Narasimhan et al. ([ICML 2022](https://proceedings.mlr.press/v162/narasimhan22a.html)).

**LLM routing** -- the problem setting is shared with Hybrid LLM (Ding et al., [ICLR 2024](https://openreview.net/forum?id=02f3mUtqnM)) and RouteLLM (Ong et al., [ICLR 2025](https://openreview.net/forum?id=sSt9fROCn2)). TRACER's distinction: it targets classification specifically, uses the teacher's own trace distribution rather than preference data, and enforces a formal parity guarantee rather than a quality heuristic.

**Interpretability** -- the qualitative audit design is informed by Slice Finder (Chung et al., [ICDE 2019](https://ieeexplore.ieee.org/document/8731353)) and the contrastive explanation literature (Miller, [AIJ 2019](https://www.sciencedirect.com/science/article/pii/S0004370217301126)).

## Paper

A research paper detailing the approach, formal guarantees, ablation studies, limitations, and reproducible experiment tooling is in preparation. It will be linked here upon publication.

## License

MIT
