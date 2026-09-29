# TRACER

*Trace-Based Adaptive Cost-Efficient Routing*

**Distill your LLM traces into fast ML System 1 models.**

[![arXiv](https://img.shields.io/badge/arXiv-2604.14531-b31b1b.svg)](https://arxiv.org/abs/2604.14531)
[![Hugging Face](https://img.shields.io/badge/🤗%20HF-Papers-yellow)](https://huggingface.co/papers/2604.14531)
[![PyPI](https://img.shields.io/pypi/v/tracer-llm)](https://pypi.org/project/tracer-llm/)
[![Downloads](https://static.pepy.tech/badge/tracer-llm)](https://pepy.tech/project/tracer-llm)
[![Downloads](https://static.pepy.tech/badge/tracer-llm/month)](https://pepy.tech/project/tracer-llm)
[![Python](https://img.shields.io/pypi/pyversions/tracer-llm)](https://pypi.org/project/tracer-llm/)
[![npm](https://img.shields.io/npm/v/@tracer-llm/watch?label=%40tracer-llm%2Fwatch&color=cb3837&logo=npm)](https://www.npmjs.com/package/@tracer-llm/watch)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![CI](https://github.com/adrida/tracer/actions/workflows/ci.yml/badge.svg)](https://github.com/adrida/tracer/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-reference-blue)](docs/)

TRACER learns **task-specific ML students** from your teacher's decisions.
The ML students return labels locally for repetitive requests. **Calibrated
deferral** and an **out-of-distribution (OOD) guard** keep the teacher in the
loop for requests the ML path declines. Your teacher can be an LLM, a
foundation model for System 1 decisions, or another classifier.

**No LLM fine-tuning. No GPU required. Your teacher stays in the loop.**

ML fitting and prediction run on CPU with supplied embeddings. You choose the
encoder and teacher; their compute is separate.

![Teacher-labelled traces become an ML student and deferral policy. Blue requests receive a local ML label; orange deferred and red OOD requests go to the teacher LLM.](docs/assets/training-inference.gif)

*Training and inference on illustrative support-ticket traces.*

## Quickstart

From a checkout of this branch ([release status](#release-status)), with
Python 3.12+:

```bash
python -m pip install -e .
```

Supply representative teacher-labelled traces in `traces.jsonl`, with one
embedding per row in `traces.npy`. The trace format is:

```jsonl
{"input": "What is my balance?", "teacher": "check_balance"}
{"input": "Send $50 to Alice", "teacher": "transfer_money"}
```

These two rows illustrate the format; fit on your actual trace dataset.
`new_request.npy` contains one new request encoded with the same model.

```python
import numpy as np
import tracer

result = tracer.fit(
    "traces.jsonl", embeddings=np.load("traces.npy"),
    config=tracer.FitConfig(target_teacher_agreement=0.95),
)
if result.manifest.selected_method is None:
    raise RuntimeError("No policy passed the final check.")
router = tracer.load_router(".tracer")
out = router.predict(np.load("new_request.npy"))
print(out["label"], out["decision"])  # label + "handled", or None + "deferred"
```

Your application sends deferred requests to the teacher, or supplies a
[fallback callback](#ood-and-teacher-fallback).

## Why Tracer?

| | TRACER |
| --- | --- |
| ML student | Task-specific classical ML or a shallow neural classifier |
| Training data | Your existing teacher-labelled traces and embeddings |
| Decisions | Labels from a fixed set: intents, tags, tool choices, workflow actions |
| Accepted traffic | The ML student returns the label locally |
| Declined traffic | Your teacher answers through your application's fallback |
| Unfamiliar inputs | An OOD distance guard can override student acceptance |
| Hardware | CPU-capable ML fitting and inference; no LLM fine-tuning |
| Quality check | The complete serving policy is checked once on reserved data |
| Ownership | MIT licensed; fit and self-host without a Tracer account |

Your embedding, teacher and serving costs still apply. Measure coverage,
latency and savings on your workload.

## What is learned

The ML student learns to predict your teacher's labels directly from embeddings.
The search includes logistic regression, SGD classifiers, trees, forests,
boosted trees and compact MLPs, with optional XGBoost. A learned acceptance
policy controls which student predictions to return; residual stages can try
another student before deferring to the teacher.

The encoder and teacher are not fine-tuned. TRACER learns your task's label set;
it does not train a general pretrained decision model or generate free-form
responses.

For a smaller fitting sweep, `FitConfig(skip_candidates=("dt", "rf", "et",
"gbt", "xgb"))` excludes tree candidates. This changes the search budget, not
the selected quality requirement. Some candidates use multiple CPU cores;
choose resource limits appropriate to your machine. See [concepts](docs/concepts.md)
for methods and [API reference](docs/api.md) for the complete configuration.

## How the quality check works

TRACER selects the ML policy on development data, freezes the complete serving
policy including its OOD guard, then checks that exact policy once on reserved
data before publishing it. If it misses the requirement, the fit does not
publish a usable policy.

1. Reserve final certification rows before label discovery, balancing or fitting.
2. Select student classifiers using validation macro-F1 against teacher labels.
3. Fit acceptance policies and select a global or staged policy for the chosen
   teacher-agreement target and coverage requirement on development data.
4. Freeze that serving policy, including its distance guard, and check it once
   on the reserved rows. A failed final check does not trigger another candidate
   search on those rows.

For the fixed final serving policy, a one-sided Clopper–Pearson lower bound on
**teacher agreement among accepted requests** must meet the target. Defaults
reserve 20% of input rows and use `certification_alpha=0.10`, corresponding to a
90% one-sided confidence level under independent, representative sampling.
Accepted coverage must also meet `min_deploy_coverage`. Zero accepted examples
cannot pass this check.

Teacher agreement is distinct from ground-truth accuracy. This aggregate bound
does not certify each prediction or each class. `accept_score` is a ranking
signal, not a calibrated per-request correctness probability. The distance
guard can reject unfamiliar embeddings; it provides no general guarantee
against distribution shift.

The built-in reserve is a random row split. Correlated sessions, duplicates,
adaptive reuse of the reserve and later traffic changes can invalidate the
sampling interpretation. Keep separate session/time evaluations, inspect rare
classes and measure both accepted quality and coverage before relying on a
policy. Paper results are historical experiments, not measurements of this
revised certification implementation.

## OOD and teacher fallback

An unfamiliar request can receive a confident ML prediction. The
**out-of-distribution (OOD) guard** checks its embedding distance to neighbouring
training examples. When a fitted guard flags the input as too far away, it
forces deferral even if the ML student would otherwise accept it.

These are two different reasons to use the teacher:

- **The acceptance policy declines the prediction:** the fitted rule does not
  accept the student's answer at the chosen operating point.
- **The OOD guard flags the input:** its embedding is too far from the training
  reference under the fitted distance rule.

Both return `decision="deferred"`. The public API currently has two decisions,
`"handled"` and `"deferred"`; it does not expose a separate OOD decision or a
rejection-reason field. A distance guard can miss unfamiliar inputs and is
fitted only when sufficient reference data is available. Evaluate it on your
own out-of-domain examples as well as ordinary traffic.

To accept raw text and connect your teacher, attach the same embedding
implementation used during fit. Matching vector dimensions alone is
insufficient: weights, normalization and preprocessing must match too.

```python
embedder = tracer.Embedder.from_callable(my_embedding_function)
router = tracer.load_router(".tracer", embedder=embedder)

text = "What is my balance?"
out = router.predict(text, fallback=lambda: call_my_teacher(text))
```

`my_embedding_function` accepts a list of strings and returns an embedding
matrix. The optional fallback takes no arguments and runs only for a deferred
request. Its return value becomes `label`; `decision` remains `"deferred"`.
Without a fallback, your application handles deferral explicitly. Batch
prediction returns deferred rows for your application to process; it does not
call the teacher.

## Local serving and updates

A fitted router can run inside your Python application or through the small
built-in HTTP server:

```python
import tracer
tracer.serve(".tracer", host="127.0.0.1", port=8000)
```

It exposes `POST /predict` with `{"embedding": [...]}`, `POST /predict_batch`
with `{"embeddings": [[...], ...]}`, and `GET /health`. The server accepts
embeddings; the calling application owns text encoding and teacher fallback.
It does not add authentication or TLS. You can integrate the same Python
router into your own authenticated service.

`update()` combines stored and new traces and refits the policy:

```python
tracer.update("new_traces.jsonl", new_embeddings=X_new)
```

Coverage may rise or fall. Collect a representative sample of all traffic, not
only deferrals, when estimating overall coverage. `fit()` and `update()` stage
artifacts and roll back on fitting or publication **exceptions**. A completed
refit that fails certification can intentionally publish
`selected_method=null`, replacing the prior on-disk generation. Check the new
manifest before reloading. Use one writer per artifact directory; these file
operations do not provide concurrent-reader or zero-downtime deployment control.

## Record traces and inspect traffic

`tracer.watch()` records your model calls locally by default, with optional
export to your own backend. It does not train a student until you call `fit()`.

```python
import tracer

watch = tracer.watch("support_classifier", system="my-provider", model="my-model")

@watch
def classify(ticket: str) -> str:
    return call_my_teacher(ticket)
```

[Watch](docs/watch.md) documents trace recording and export. For JavaScript,
[`@tracer-llm/watch`](js/README.md) records calls; training and prediction remain
in the Python library. The [JavaScript guide](docs/javascript.md) shows the
Python sidecar integration.

`tracer.scan()` groups embedded traces and reports per-cluster diagnostics. It
needs at least 1,000 traces by default; `force=True` permits an explicitly thin
data estimate. Scanning does not train or certify the final serving policy.
See the [scan guide](docs/scan.md).

## Embeddings and artifacts

Use supplied NumPy arrays, `Embedder.from_callable`, a compatible HTTP endpoint,
or a local sentence-transformers encoder. To install the optional local encoder
support from this checkout:

```bash
python -m pip install -e '.[embeddings]'
```

`tracer.embed(texts)` defaults to `all-MiniLM-L6-v2`. Optional encoders can require
model downloads; `fit()` itself does not train or package the encoder.

| Artifact | Purpose |
| --- | --- |
| `manifest.json` | Selected method, label space and final certification result |
| `pipeline.joblib` | Fitted students, acceptors and thresholds |
| `frontier.json` | Candidate-selection diagnostics, separate from final certification |
| `ood.json`, `ood_reference.npy` | Distance guard and its development-only reference, when fitted |
| `all_traces.jsonl`, `index.embeddings.npy`, optional `index.faiss` | Stored traces, embeddings and optional search index used by updates |
| `qualitative_report.json` | Per-label slices, examples and boundary pairs |
| `report.html` | Optional HTML report generated with `tracer.generate_html_report()` |

Keep the complete artifact directory. If an expected OOD guard or reference is
missing or corrupt, loading fails instead of silently disabling it. Older
artifacts without final-certification metadata do not gain a certificate by
being loaded with newer code. See [artifacts](docs/artifacts.md) and
[troubleshooting](docs/troubleshooting.md).

## Release status

The final-policy certification, staged publication and strict OOD-loading
corrections described here are **unreleased source changes**. Installing
`tracer-llm` from PyPI installs the published package; it does not establish
that these corrections are present. The quickstart above installs from this
checkout. See the [change log](CHANGELOG.md) and [overview](docs/system1.md).

## Documentation

| Guide | Contents |
| --- | --- |
| [Overview](docs/system1.md) | Teacher/student architecture and scope |
| [Change log](CHANGELOG.md) | Source changes and release status |
| [Concepts](docs/concepts.md) | Students, acceptance policies and statistical limits |
| [API reference](docs/api.md) | Functions, configuration and return types |
| [Watch](docs/watch.md) | Local recording and optional exports |
| [Scan](docs/scan.md) | Inspect traffic before fitting |
| [JavaScript](docs/javascript.md) | Python integration from Node.js |
| [Artifacts](docs/artifacts.md) | Saved policy layout |
| [Troubleshooting](docs/troubleshooting.md) | Failed checks, data and embedding problems |
| [AGENTS.md](AGENTS.md) | Grounded integration instructions for coding assistants |

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
