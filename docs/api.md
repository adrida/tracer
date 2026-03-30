# Python API Reference

## `tracer.fit()`

Fit a routing policy from traces and embeddings.

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
| `embeddings` | `np.ndarray \| None` | `None` | Precomputed embeddings `(n, dim)`. If None, auto-discovered from `<trace_path_stem>.npy` |
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
print(f"Coverage: {result.manifest.coverage_cal:.1%}")
print(f"TA:       {result.manifest.teacher_agreement_cal:.3f}")
```

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
| `config` | `FitConfig \| None` | `None` | If None, re-uses `target_teacher_agreement` from the existing manifest |

**Returns:** `FitResult` (same as `fit()`)

**Example:**

```python
result = tracer.update(
    "traces_day2.jsonl",
    new_embeddings=X_day2,
)
print(f"Coverage now: {result.manifest.coverage_cal:.1%}")
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

Calls an external HTTP embedding API. Default: sends one request per text with `{"input": "text"}`, expects `{"embedding": [...]}` back. Set `batch_key` to send all texts in one request.

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

# OpenAI-compatible endpoint
embedder = Embedder.from_endpoint(
    "https://api.openai.com/v1/embeddings",
    headers={"Authorization": "Bearer sk-..."},
    input_key="input",
    output_key="data.0.embedding",
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
    "label":        str,    # predicted class label
    "decision":     str,    # "handled" or "deferred"
    "accept_score": float,  # acceptor confidence (0–1)
    "stage":        int,    # which pipeline stage handled it
}
```

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
    print(f"Deferred to teacher: {out['label']}")
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
    "labels":    list[str],     # predicted labels for all inputs
    "decisions": list[str],     # "handled" or "deferred" for each
    "handled":   np.ndarray,    # bool array, shape (n,)
    "preds":     np.ndarray,    # label indices, shape (n,)
    "stage_id":  np.ndarray,    # int array, shape (n,)
}
```

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

**Model recommendations:**

| Model | Dim | Speed | Quality |
|-------|-----|-------|---------|
| `all-MiniLM-L6-v2` | 384 | ⚡⚡⚡ | good |
| `BAAI/bge-small-en-v1.5` | 384 | ⚡⚡⚡ | very good |
| `BAAI/bge-base-en-v1.5` | 768 | ⚡⚡ | excellent |
| `BAAI/bge-m3` | 1024 | ⚡ | excellent, multilingual |

**Example:**

```python
import tracer, numpy as np

texts = ["What's my balance?", "Send $50 to Alice", ...]
X = tracer.embed(texts, model="BAAI/bge-small-en-v1.5")
np.save("embeddings.npy", X)
```

---

## `tracer.generate_html_report()`

Generate a self-contained HTML audit report from a `.tracer/` directory.

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
webbrowser.open(f"file://{path}")
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

```python
@dataclass
class FitConfig:
    target_teacher_agreement: float = 0.90
    frontier_targets: tuple = (0.85, 0.90, 0.95)
    min_deploy_coverage: float = 0.05
    max_fit_labels: int = 8_000
    explore_rate: float = 0.05
    retrain_every: int = 100
    min_labels_to_start: int = 100
    seed: int = 42
```

| Field | Description |
|-------|-------------|
| `target_teacher_agreement` | The parity bar. Surrogate must match teacher at least this often on handled traffic. |
| `frontier_targets` | Multiple targets to explore. The best (highest coverage) at `target_teacher_agreement` is selected. |
| `min_deploy_coverage` | Minimum coverage fraction to consider a method deployable. |
| `max_fit_labels` | Subsample to this size for efficiency on large datasets (stratified). |
| `seed` | Random seed for reproducibility. |

**Example:**

```python
config = tracer.FitConfig(
    target_teacher_agreement=0.95,        # 95% parity required
    frontier_targets=(0.90, 0.95, 0.99),  # explore these targets
)
result = tracer.fit("traces.jsonl", config=config)
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
```
