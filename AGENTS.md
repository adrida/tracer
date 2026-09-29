# TRACER integration notes

TRACER learns fixed-label decisions from classification traces. A
teacher supplies past classification labels. A fitted student predicts labels
directly from embeddings; an acceptance policy and optional distance guard
control which predictions to return. The caller handles deferred requests,
optionally through its teacher. Read the [overview](docs/system1.md),
[API reference](docs/api.md) and [change log](CHANGELOG.md) for the current scope.

## Source and release status

These instructions describe this repository checkout. Final-policy certification,
staged artifact publication and strict OOD-loading corrections are unreleased
source changes. Do not imply they are already present in the PyPI package merely
because `pip install tracer-llm` succeeds. To use the corrected source, install
from a checkout containing the changes:

```bash
python -m pip install -e .
```

Python 3.12+ is required. Core dependencies are NumPy, scikit-learn and joblib.
The library is MIT licensed: local fitting, saved artifacts and self-hosted
prediction do not require a Tracer account or license fee. Embeddings, teacher
calls and hosting still have their own resource costs. This checkout supplies a
Python API and a small HTTP server; the JavaScript package records traces.
Do not invent a command-line interface, a general chat model, an OpenRouter
listing or parity with a separately deployed hosted service.

## Identify the decision contract first

Establish the label vocabulary, input representation, teacher and quality target
from the user's existing task. Preserve their selected quality requirement.
Training examples have this shape:

```jsonl
{"input": "What is my balance?", "teacher": "check_balance"}
{"input": "Send $50 to Alice", "teacher": "transfer_money"}
```

Optional fields are `id`, `ground_truth` and `metadata`. Example rows illustrate
the format; a usable policy needs representative training and held-out evidence.
The student learns the label set from the development partition. Unseen labels
are not zero-shot supported classes. Multi-turn context can be encoded into
`input` by the application, but this library does not reconstruct agent state.

Use an existing authorized dataset and encoder when available. Resolve missing
contract information before issuing paid teacher calls or sending private data
to an external embedding endpoint. Existing user authorization takes precedence
over this guide; do not invent another approval step for already authorized work.

## Fit, inspect, load

```python
import numpy as np
import tracer

X = np.load("traces.npy")
result = tracer.fit(
    "traces.jsonl",
    embeddings=X,
    config=tracer.FitConfig(target_teacher_agreement=0.95),
)
print(result.manifest.certification)

if result.manifest.selected_method is None:
    raise RuntimeError("No deployable policy; inspect the certificate and data.")

router = tracer.load_router(".tracer")
out = router.predict(np.load("new_request.npy"))
```

The matrix must contain one finite embedding per trace, in the same row order.
Use the same encoder weights, dimension, normalization and preprocessing during
fitting and prediction. Matching dimensions alone does not establish this.

The fitting search includes linear classifiers, compact MLPs and tree models;
classical ML can make the final accepted decision. Fitting does not fine-tune
an embedding model. Do not describe the library as training only a deferral gate.

Some candidate implementations use multiple CPU cores. Respect workstation and
cloud resource limits; do not run a full fit merely to validate a documentation
change. `skip_candidates` can bound the candidate search, but it must not be used
to silently lower the user's quality target or bypass the final check.

## Runtime results and teacher fallback

```python
embedder = tracer.Embedder.from_callable(my_embedding_function)
router = tracer.load_router(".tracer", embedder=embedder)
text = "What is my balance?"
out = router.predict(text, fallback=lambda: call_my_teacher(text))
```

The callable embedder accepts a list of strings and returns a matrix. A fallback
accepts no arguments and is invoked only after deferral. The return contract is:

| Field | Meaning |
| --- | --- |
| `label` | Student label when handled; fallback return value or `None` when deferred |
| `decision` | Exactly `handled` or `deferred` |
| `accept_score` | Acceptance ranking score; zero on deferred outputs |
| `stage` | Accepted pipeline stage, or `-1` for deferral |

An OOD distance rejection is a plain `deferred` result. There is no public
third decision state or reason field. `accept_score` is not a calibrated
correctness probability. A fallback response still has `decision="deferred"`.

`predict_batch()` returns labels, decisions, the handled mask, internal prediction
indices and stage IDs. It does not call a teacher for deferred rows. The caller
must process those rows explicitly.

## Certification and failure handling

The default reserves 20% of rows before fitting, label balancing or candidate
selection. Development data selects the student, acceptance thresholds and
policy. The final serving policy, including its OOD guard, is frozen before one
check on the reserve. Do not reuse this reserve to rescue a failed candidate.

The one-sided Clopper–Pearson lower bound on accepted teacher agreement must
meet `target_teacher_agreement`. The default `certification_alpha=0.10` gives a
90% one-sided confidence level under independent, representative sampling.
Accepted coverage must also meet `min_deploy_coverage`. No accepted examples
means no certificate, not perfect agreement.

This measures agreement with the teacher, not ground-truth accuracy, individual
correctness or a per-class guarantee. The reserve is a random row split.
Duplicates, correlated sessions, distribution shift and repeated tuning against
held-out results weaken or invalidate that interpretation. Maintain independent
session/time evaluations and report accepted coverage alongside quality.

If `selected_method` is null, `load_router()` refuses to load a serving policy.
A completed `fit()` or `update()` that fails certification may intentionally
publish this null-policy generation. A fitting/publication exception instead
preserves the prior generation through staged rollback. Do not describe every
failed check as preserving the old router, or promise zero-downtime concurrent
readers. Use one writer and reload readers only after inspecting the new manifest.

Expected OOD metadata and reference files are part of the policy. Missing or
corrupt required files fail loading; do not remove them to make loading succeed.
Older artifacts without final-certification metadata are not retroactively
certified by a newer loader.

## Embeddings, updates and deployment

Local sentence-transformers support is optional:

```bash
python -m pip install -e '.[embeddings]'
```

`Embedder.from_sentence_transformers(model)` may download a model;
`Embedder.from_endpoint(url, ...)` calls the configured external HTTP endpoint.
`Embedder.from_callable(fn)` supports an existing encoder. The generic endpoint
adapter expects configured JSON keys; do not claim it implements every provider's
API schema automatically. Precomputed embeddings need no embedding API call.

```python
tracer.update("new_traces.jsonl", new_embeddings=X_new)
tracer.generate_html_report(".tracer")
```

Keep the complete `.tracer/` directory, including stored traces and embedding
index needed by updates. New data must use the same representation. Collect a
representative sample of all traffic when estimating total coverage, rather than
only collecting the old router's deferrals. Coverage can rise or fall.

For a local HTTP sidecar:

```python
tracer.serve(".tracer", host="127.0.0.1", port=8000)
```

The server accepts `POST /predict` with an embedding, `POST /predict_batch` with
an embedding matrix and `GET /health`. Encoding and teacher calls are the caller's
responsibility. Authentication and TLS belong in the integrating application.

`watch()` records model calls locally by default, with optional export to an
explicitly configured backend. `scan()` reports cluster diagnostics; it does not
train or certify a serving policy. See the separate watch and scan guides.

## Paper and claims

Preserve the research title and citation:
**TRACER: Trace-Based Adaptive Cost-Efficient Routing for LLM Classification**,
Adam Rida, [arXiv:2604.14531](https://arxiv.org/abs/2604.14531).

The paper's historical benchmark results are not measurements of this revised
certification implementation. Do not invent achieved coverage, latency, cost
savings or superiority over another model. Report measured outcomes and their
data/split assumptions.
