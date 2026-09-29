# Python API Reference

This API builds and runs a task-specific classifier from
fixed-label classification traces. The student predicts labels directly;
acceptance and OOD rules select which answers to use. The caller owns teacher
execution on deferral. The OSS package has no CLI entry point, integrated hosted
model catalog, managed credentials, or Echo wallet.

## `tracer.scan()`

Group traffic by similarity and measure per-cell held-out teacher agreement.
Returns a `ScanResult`; render it with `tracer.scanner.format_scan` (text) or
`tracer.scanner.scan_html` (HTML). The scan is a diagnostic, not a trained
student, a final-policy certificate, or a net savings measurement. See the
[scan guide](scan.md) for the meaning of its historical `certifiable_share` field.

```python
tracer.scan(
    traces_path,
    target=0.90,
    embeddings=None,
    model="all-MiniLM-L6-v2",
    teacher_price_per_1k=None,
    monthly_calls=None,
    viz_layout="pca",
    seed=7,
    max_clusters=60,
    force=False,
) -> ScanResult
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `traces_path` | `str \| Path` | required | Path to traces JSONL file |
| `target` | `float` | `0.90` | Threshold for per-cell teacher-agreement lower bounds |
| `embeddings` | `np.ndarray \| None` | `None` | Precomputed embeddings `(n, dim)`; computed locally from text if omitted |
| `model` | `str` | `"all-MiniLM-L6-v2"` | Local sentence-transformers model used when `embeddings` is omitted |
| `teacher_price_per_1k` | `float \| None` | `None` | Teacher cost per 1k calls, to estimate savings |
| `monthly_calls` | `int \| None` | `None` | Monthly volume, to project monthly savings |
| `viz_layout` | `str` | `"pca"` | 3D layout for the HTML report (`pca`, `umap`, `tsne`, `auto`) |
| `seed` | `int` | `7` | Split and clustering seed |
| `max_clusters` | `int` | `60` | Maximum number of similarity cells |
| `force` | `bool` | `False` | Scan thin data with coarser grouping and an explicit warning; no production coverage floor is established |

**Raises:** `tracer.scanner.ThinDataError` when fewer than 1,000 traces are passed
and `force=False`. About 5,000 traces is the sweet spot.

**Returns:** `ScanResult` (fields include `certifiable_share`, `clusters`,
`frontier`, `forced`, `savings_per_1k_calls`, `projection`).

---

## `tracer.fit()`

Fit a student classifier and acceptance policy from traces and embeddings. The
encoder is not trained. Embeddings must match the trace rows in order, using the
same encoder and preprocessing that will be used at inference.

```python
tracer.fit(
    trace_path,
    artifact_dir=".tracer",
    embeddings=None,
    config=None,
) -> FitResult
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `trace_path` | `str \| Path` | required | Path to traces JSONL file |
| `artifact_dir` | `str \| Path` | `".tracer"` | Directory to save artifacts |
| `embeddings` | `np.ndarray \| None` | `None` | Precomputed finite embeddings `(n, dim)`. If omitted, tries `<stem>.npy`, then `<stem>_embeddings.npy`; does not call an embedding model automatically |
| `config` | `FitConfig \| None` | `None` | Fit configuration. Defaults to `FitConfig()` |

**Returns:** `FitResult`

```python
result.manifest              # ArtifactManifest
result.manifest.selected_method          # "global", "l2d", "rsb", or None
result.manifest.coverage_cal             # float, e.g. 0.928
result.manifest.teacher_agreement_cal    # float, e.g. 0.950
result.manifest.n_traces                 # int
result.manifest.label_space              # list[str]
result.manifest.embedding_dim            # int
result.manifest.certification            # final check status, counts and bound
result.qualitative_report    # QualitativeReport | None
result.notes                 # list[str], human-readable notes
result.artifact_dir          # str
result.get_sankey()          # generate Sankey diagram (requires tracer-llm[viz])
```

**Example:**

```python
import tracer, numpy as np

result = tracer.fit(
    "traces.jsonl",
    embeddings=np.load("embeddings.npy"),
    config=tracer.FitConfig(target_teacher_agreement=0.95),
)

print(f"Method:   {result.manifest.selected_method}")
print(f"Certification: {result.manifest.certification}")
if result.manifest.selected_method is not None:
    print(f"Coverage: {result.manifest.coverage_cal:.1%}")
    print(f"Teacher agreement: {result.manifest.teacher_agreement_cal:.3f}")
```

The final check uses a reserved sample after the serving policy is fixed. Its
bound concerns teacher agreement on accepted traffic, under independent,
representative sampling; it is not ground-truth accuracy or per-request
certainty. A completed fit can return `selected_method=None`, leaving no student
to load. This result publishes a null-policy generation to `artifact_dir`.
Exceptions during fitting/publication preserve the preceding generation;
ordinary certification rejection is not such an exception. Use a separate
candidate directory when promotion must be explicit. See [concepts](concepts.md).

---

## `tracer.update()`

Refit with new traces. Combines new traces with all historical traces.

```python
tracer.update(
    new_trace_path,
    artifact_dir=".tracer",
    new_embeddings=None,
    config=None,
) -> FitResult
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `new_trace_path` | `str \| Path` | required | Path to NEW traces JSONL file |
| `artifact_dir` | `str \| Path` | `".tracer"` | Existing artifact directory to update |
| `new_embeddings` | `np.ndarray \| None` | `None` | Embeddings for the NEW traces only `(n_new, dim)` |
| `config` | `FitConfig \| None` | `None` | If None, reuses the saved fit configuration. An explicit configuration takes precedence and is not mutated. |

**Returns:** `FitResult` (same as `fit()`)

The replacement model is fitted in a staging directory. Validation or fitting
failures leave existing artifacts intact; a failed publication restores the
previous directory. If restoration itself fails, the exception identifies the
retained backup. Keep one writer per artifact directory and reload readers only
after `update()` returns: directory publication is not a concurrent-reader or
power-loss transaction. Allow disk space for the staged generation and backup.
A completed update that fails its final check replaces the directory with a
null-policy generation. Coverage can rise or fall, and historical fit outcomes
do not certify a new update. Include representative traffic rather than only
old-policy deferrals; do not retune against repeatedly inspected final results.

**Example:**

```python
result = tracer.update(
    "traces_day2.jsonl",
    new_embeddings=X_day2,
)
print(result.manifest.certification)
```

---

## `tracer.load_router()`

Load a production router from a `.tracer/` artifact directory.

```python
tracer.load_router(artifact_dir=".tracer", embedder=None) -> Router
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `artifact_dir` | `str \| Path` | `".tracer"` | Artifact directory |
| `embedder` | `Embedder \| None` | `None` | If set, the router accepts text strings directly |

**Returns:** `Router` instance

Raises `ValueError` for a null selected policy or an invalid declared OOD guard;
missing/corrupt artifacts can also raise their file/serialization errors. Loading
a legacy artifact does not grant it the new final-policy certificate.

```python
# Without embedder (pass embeddings manually)
router = tracer.load_router(".tracer")

# With embedder (pass text directly)
from tracer import Embedder
embedder = Embedder.from_sentence_transformers("BAAI/bge-small-en-v1.5")
router = tracer.load_router(".tracer", embedder=embedder)
```

---

## `Embedder`

Handles converting text to embedding vectors. Three factory methods:

### `Embedder.from_sentence_transformers()`

```python
Embedder.from_sentence_transformers(
    model="all-MiniLM-L6-v2",
    device=None,
    batch_size=128,
    normalize=True,
) -> Embedder
```

Requires: `pip install tracer-llm[embeddings]`

### `Embedder.from_endpoint()`

```python
Embedder.from_endpoint(
    url,
    headers=None,
    input_key="input",
    output_key="embedding",
    batch_key=None,
    batch_output_key=None,
) -> Embedder
```

Calls an external HTTP embedding API. Default: sends one request per text with
`{"input": "text"}`, expects `{"embedding": [...]}` back. Set `batch_key` to send
all texts in one request. Output keys are **literal top-level keys**, not dotted
JSON paths. The adapter does not add a provider's model parameter or parse its
nested response automatically. Use `from_callable()` for provider SDKs or a
custom response transform.

### `Embedder.from_callable()`

```python
Embedder.from_callable(fn) -> Embedder
```

Wraps any function `fn(texts: list[str]) -> array-like (n, dim)`.

### Instance methods

| Method | Description |
|--------|-------------|
| `embedder.embed(texts)` | Batch embed. Returns `np.ndarray (n, dim)` |
| `embedder.embed_one(text)` | Single text. Returns `np.ndarray (dim,)` |

**Example:**

```python
from tracer import Embedder

# sentence-transformers
embedder = Embedder.from_sentence_transformers("BAAI/bge-small-en-v1.5")

# Your endpoint must implement {"input": text} -> {"embedding": [...]}
embedder = Embedder.from_endpoint(
    "https://your-host/embed",
    headers={"Authorization": "Bearer YOUR_ENDPOINT_KEY"},
    input_key="input",
    output_key="embedding",
)

# Custom function
embedder = Embedder.from_callable(lambda texts: my_model.encode(texts))
```

---

## `Router.predict()`

Route a single input. Accepts text (if embedder set) or embedding vector.

```python
router.predict(
    input,
    fallback=None,
) -> dict
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `input` | `str \| np.ndarray` | Text (requires embedder) or embedding `(dim,)` |
| `fallback` | `callable \| None` | Called with no args if deferred. Return value used as label. |

**Returns:**

```python
{
    "label":        str | None,  # student label, fallback result, or None
    "decision":     str,    # "handled" or "deferred"
    "accept_score": float,  # acceptance ranking signal, not correctness probability
    "stage":        int,    # handled stage index, or -1 when deferred
}
```

Both acceptance-rule rejection and OOD rejection return `"deferred"`; this API
does not expose a separate reason field. Without a fallback, a deferred result
has `label=None`, `accept_score=0.0`, and `stage=-1`. With a fallback, its return
value becomes `label` but the decision remains `"deferred"`. The callback is
called without arguments; validate its label and handle its exceptions in your
integration. It is not an automatic provider selection or generation API.

**Example:**

```python
# With embedder (text in)
out = router.predict("What is my balance?")

# With fallback
out = router.predict("What is my balance?",
                     fallback=lambda: call_my_llm("What is my balance?"))

# Without embedder (embedding in)
out = router.predict(embedding_vector)

if out["decision"] == "handled":
    print(f"Surrogate: {out['label']} (score={out['accept_score']:.2f})")
else:
    print("Student deferred; invoke your teacher if no fallback was supplied")
```

---

## `Router.predict_batch()`

Route a batch of inputs. Accepts list of texts or embedding matrix.

```python
router.predict_batch(inputs) -> dict
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `inputs` | `list[str] \| np.ndarray` | Texts (requires embedder) or embeddings `(n, dim)` |

**Returns:**

```python
{
    "labels":    list[str | None],  # None for deferred inputs
    "decisions": list[str],     # "handled" or "deferred" for each
    "handled":   np.ndarray,    # bool array, shape (n,)
    "preds":     np.ndarray,    # internal predictions; use only where handled
    "stage_id":  np.ndarray,    # stage index, or -1 for deferred inputs
}
```

Batch prediction does not call a teacher. Use `handled` or `decisions` to select
which rows need a fallback; an internal prediction is not an accepted answer.

**Example:**

```python
# Batch text
batch = router.predict_batch(["query 1", "query 2", "query 3"])
print(batch["decisions"])  # ["handled", "handled", "deferred"]

# Batch embeddings
batch = router.predict_batch(X_test)
n_handled = batch["handled"].sum()
print(f"Handled: {n_handled}/{len(X_test)}")
```

---

## `tracer.serve()`

Serve a saved policy over a lightweight local HTTP interface:

```python
tracer.serve(artifact_dir=".tracer", host="127.0.0.1", port=8000)
```

This blocking call loads the policy before starting its server. It accepts
**embedding vectors**, not raw text, and never calls a teacher.

| Method | Path | Input | Result |
| --- | --- | --- | --- |
| GET | `/health` | None | `status`, `method`, `coverage`, `teacher_agreement`, `n_labels`, `n_traces` |
| POST | `/predict` | `{"embedding": [0.1, 0.2]}` with the fitted width | Single prediction, including `stage` |
| POST | `/predict_batch` | `{"embeddings": [[0.1, 0.2]]}` with the fitted width | `labels`, `decisions`, boolean `handled` list |

The shown vector widths are placeholders. `/health` does not return the
embedding dimension; read `manifest.embedding_dim`. The server validates body
size up to 8 MiB and gives HTTP 400 for malformed inputs, including dimension
errors. It supplies neither authentication nor TLS/CORS configuration. Keep it
on loopback or protect it through your application's network/proxy. It is not
an OpenAI/OpenRouter-compatible chat endpoint. See the [JS guide](javascript.md).

---

## `tracer.report()`

Load and return the policy manifest from an artifact directory.

```python
tracer.report(artifact_dir=".tracer") -> ArtifactManifest
```

**Example:**

```python
m = tracer.report(".tracer")
print(m.coverage_cal)
print(m.selected_method)
print(m.label_space[:5])
```

---

## `tracer.embed()`

Compute embeddings using sentence-transformers. Requires `pip install tracer-llm[embeddings]`.

```python
tracer.embed(
    texts,
    model="all-MiniLM-L6-v2",
    batch_size=128,
    normalize=True,
    show_progress=True,
    device=None,
) -> np.ndarray
```

**Parameters:**

| Name | Default | Description |
|------|---------|-------------|
| `texts` | required | `list[str]` |
| `model` | `"all-MiniLM-L6-v2"` | sentence-transformers model name |
| `batch_size` | `128` | Forward pass batch size |
| `normalize` | `True` | L2-normalize (recommended for cosine similarity) |
| `show_progress` | `True` | Show tqdm progress bar |
| `device` | `None` | `"cpu"`, `"cuda"`, `"mps"`, or None (auto-detect) |

**Returns:** `np.ndarray` of shape `(len(texts), dim)`, dtype `float32`

**Example model dimensions:**

| Model | Dim |
|-------|-----|
| `all-MiniLM-L6-v2` | 384 |
| `BAAI/bge-small-en-v1.5` | 384 |
| `BAAI/bge-base-en-v1.5` | 768 |
| `BAAI/bge-m3` | 1024 |

These examples are not a ranking on your task. Measure quality, embedding cost
and full request latency with your traffic. Matching dimensions alone do not
make two embedding models interchangeable.

**Example:**

```python
import tracer, numpy as np

texts = ["What's my balance?", "Send $50 to Alice"]
X = tracer.embed(texts, model="BAAI/bge-small-en-v1.5")
np.save("embeddings.npy", X)
```

---

## `tracer.generate_html_report()`

Generate a self-contained HTML report from a `.tracer/` directory.

```python
tracer.generate_html_report(
    artifact_dir,
    output_path=None,
) -> str
```

**Parameters:**

| Name | Default | Description |
|------|---------|-------------|
| `artifact_dir` | required | Path to `.tracer/` directory |
| `output_path` | `<artifact_dir>/report.html` | Where to write the HTML |

**Returns:** `str` path to the generated HTML file.

**Example:**

```python
path = tracer.generate_html_report(".tracer")
print(f"Report at: {path}")

import webbrowser
from pathlib import Path
webbrowser.open(Path(path).resolve().as_uri())
```

---

## `tracer.generate_sankey()`

Generate an interactive Sankey diagram of the routing flow.

```python
tracer.generate_sankey(
    artifact_dir,
    output_path=None,
    fmt="html",
    top_k=15,
    title=None,
) -> str
```

**Parameters:**

| Name | Default | Description |
|------|---------|-------------|
| `artifact_dir` | required | Path to `.tracer/` directory |
| `output_path` | `<artifact_dir>/sankey.<fmt>` | Where to write the output |
| `fmt` | `"html"` | `"html"` for interactive, `"png"`, `"svg"`, `"pdf"`, or `"jpeg"` for static |
| `top_k` | `15` | Number of top labels to show individually (rest grouped as "other") |
| `title` | auto-generated | Custom diagram title |

**Returns:** `str` path to the generated file.

**Requires:** `pip install tracer-llm[viz]`

**Example:**

```python
path = tracer.generate_sankey(".tracer")
tracer.generate_sankey(".tracer", fmt="png", output_path="routing.png")
```

Also available as a method on `FitResult`:

```python
result = tracer.fit("traces.jsonl", embeddings=X)
result.get_sankey()                # interactive HTML
result.get_sankey(fmt="png")       # static image
```

---

## `FitConfig`

Configuration for `fit()` and `update()`.
The following is a field summary; instantiate it through `tracer.FitConfig`.

```python
@dataclass
class FitConfig:
    target_teacher_agreement: float = 0.90
    frontier_targets: tuple = (0.85, 0.90, 0.95)
    min_deploy_coverage: float = 0.05
    max_fit_labels: int = 8_000
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    seed: int = 42
    certification_fraction: float = 0.20
    certification_alpha: float = 0.10
    verbose: bool = True
    skip_candidates: tuple = ()
```

| Field | Description |
|-------|-------------|
| `target_teacher_agreement` | Target for the one-sided lower bound on accepted teacher agreement, under independent representative sampling. |
| `frontier_targets` | Development-only targets to explore. The requested target is added if absent; only one selected policy receives the final check. |
| `min_deploy_coverage` | Minimum coverage fraction to consider a method deployable. |
| `max_fit_labels` | Subsample to this size for efficiency on large datasets (stratified). |
| `embedding` | Saved embedding configuration metadata. Does not make `fit()` compute embeddings or validate encoder identity at runtime. |
| `seed` | Random seed for reproducibility. |
| `certification_fraction` | Fraction reserved before fitting or selection. The final policy is checked once on these examples. |
| `certification_alpha` | One-sided error level for final certification; default 0.10. No distribution-shift or repeated adaptive testing guarantee. |
| `verbose` | Emit fitting progress to stderr. Set `False` for quiet runs. |
| `skip_candidates` | Candidate names to exclude. Use `("dt", "rf", "et", "gbt", "xgb")` for a linear/neural sweep. An empty tuple keeps all available candidates. |

**Example:**

```python
config = tracer.FitConfig(
    target_teacher_agreement=0.95,        # target lower bound on accepted teacher agreement
    frontier_targets=(0.90, 0.95, 0.99),  # explore these targets
)
result = tracer.fit("traces.jsonl", embeddings=X, config=config)
```

---

## Types

### `QualitativeReport`

```python
@dataclass
class QualitativeReport:
    summary: str                          # e.g. "Handled 9278/10003 (92.8%) by surrogate"
    coverage: float
    teacher_agreement_handled: float
    slices: list[SliceInsight]
    handled_examples: list[RepresentativeExample]
    deferred_examples: list[RepresentativeExample]
    boundary_pairs: list[BoundaryPair]
    temporal_deltas: list[TemporalDelta]
```

### `SliceInsight`

```python
@dataclass
class SliceInsight:
    slice_name: str          # e.g. "label:check_balance" or "length:short"
    predicate: str           # human-readable description
    count: int
    handled_rate: float
    deferred_rate: float
    teacher_agreement_handled: float | None
    dominant_teacher_label: str | None
```

### `BoundaryPair`

```python
@dataclass
class BoundaryPair:
    handled_preview: str     # first 160 chars of handled input
    deferred_preview: str    # first 160 chars of deferred input
    teacher_label: str       # same for both
    handled_score: float | None
    deferred_score: float | None
```

### `RepresentativeExample`

```python
@dataclass
class RepresentativeExample:
    input_preview: str       # first 160 chars
    teacher_label: str
    decision: str            # "handled" or "deferred"
    local_label: str | None
    accept_score: float | None
    trace_id: str | None
```

### `TemporalDelta`

```python
@dataclass
class TemporalDelta:
    label: str
    previous_handled_rate: float
    current_handled_rate: float
    delta: float             # current - previous
```

### `ArtifactManifest`

```python
@dataclass
class ArtifactManifest:
    version: str
    n_traces: int
    label_space: list[str]
    selected_method: str | None      # "global", "l2d", "rsb", or None
    target_teacher_agreement: float
    coverage_cal: float | None
    teacher_agreement_cal: float | None
    embedding_dim: int | None
    n_retrains: int
    pipeline_path: str | None
    index_path: str | None
    config_path: str | None
    qualitative_report_path: str | None
    certification: dict | None      # final-policy evidence; absent on legacy artifacts
    ood_required: bool              # loading requires a valid saved guard
```

`coverage_cal` and `teacher_agreement_cal` contain final reserved-sample metrics
for new certified policies; historical artifacts retain their old calibration
meaning. Use `certification` to distinguish them. `QualitativeReport` describes
the supplied trace set, including development examples, rather than a separate
unseen evaluation. Examples throughout this reference illustrate API shape and
are not benchmark claims.
