# CLI Reference

## Overview

```
tracer <command> [options]
```

| Command | What it does |
|---------|-------------|
| `tracer demo` | Zero-setup demo with Banking77 data |
| `tracer scan` | Day-one read: how much traffic is certifiably routable, with a 3D map |
| `tracer fit` | Fit a routing policy from your traces |
| `tracer update` | Refit with new traces (continual learning) |
| `tracer report` | Print the policy manifest as JSON |
| `tracer report-html` | Generate an HTML report and open it |
| `tracer sankey` | Generate a Sankey routing flow diagram |
| `tracer serve` | Start a prediction HTTP server |
| `tracer watch export` | Convert watch JSONL spans into fit-ready traces |

---

## `tracer demo`

Runs a complete end-to-end demo. No traces or embeddings needed.

```bash
tracer demo
```

**What it does:**
1. Downloads Banking77 data (77 intent classes, ~1,500 traces) and computes embeddings. Falls back to synthetic data (500 traces, 5 classes) if `sentence-transformers` is not installed.
2. Trains the surrogate model zoo (logreg, RF, ET, and more)
3. Fits and calibrates the routing policy
4. Shows per-label coverage bars
5. Shows contrastive boundary pairs
6. Routes 5 sample queries live
7. Prints a cost projection for 10k queries/day
8. Saves all outputs to `./tracer-demo-output/` and opens the HTML report

**Output directory:** `./tracer-demo-output/`
```
tracer-demo-output/
  traces.jsonl            ← synthetic traces (copy the format)
  embeddings.npy          ← synthetic embeddings
  .tracer/                ← artifacts
    manifest.json
    pipeline.joblib
    frontier.json
    qualitative_report.json
    report.html
    ...
```

---

## `tracer scan`

The fast, conservative first look at a traces file. It groups your traffic by
similarity and, on a held-out slice the grouping never saw, measures how much of
it a near-free model can answer at your target agreement, using exact binomial
(held-out, never in-sample) bounds. It does not train a router; `tracer fit`
does that and certifies more of the same traffic.

```bash
tracer scan <traces> [--target <float>] [--html <path>] [options]
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `traces` | required | Path to traces JSONL file |
| `--target` | `0.90` | Target label agreement to certify against |
| `--teacher-price-per-1k` | none | Your teacher cost per 1k calls, to estimate savings |
| `--monthly-calls` | none | Monthly call volume, to project monthly savings |
| `--html` | none | Also write a self-contained HTML report (3D embedding map) |
| `--no-open` | off | Don't open the HTML report in a browser |
| `--embed-model` | `all-MiniLM-L6-v2` | Local sentence-transformers model for embeddings |
| `--embeddings` | none | Path to a precomputed embeddings `.npy` (skip embedding) |
| `--embed-url` | none | Use your own HTTP embedding endpoint instead of a local model |
| `--embed-header` | none | Header for `--embed-url`, repeatable (e.g. `'Authorization: Bearer ...'`) |
| `--embed-input-key` / `--embed-output-key` | `input` / `embedding` | Request/response JSON keys for the custom endpoint |
| `--embed-batch-key` | none | Send all texts in one request under this key (batch endpoints) |
| `--viz-layout` | `pca` | 3D layout for the report: `pca` (dense connected cloud), `umap`, `tsne`, or `auto`. `umap` needs `umap-learn` installed |
| `--force` | off | Scan thin data anyway (see below) |

**Data volume.** The scan needs about **1,000 traces** for a stable read, and
around **5,000** is the sweet spot where each cell carries enough held-out
examples to certify at 0.90. Below 1,000 traces it stops and asks you to collect
more, unless you pass `--force`, which coarsens the grouping to concentrate the
held-out evidence and reports a best-effort floor (not a guarantee), and says so
loudly.

**Embeddings.** By default scan computes embeddings **locally with
sentence-transformers** (`pip install tracer-llm[embeddings]`; pick the model with
`--embed-model`). You can instead:
- reuse a precomputed matrix with `--embeddings path.npy`, or
- call **your own embedding service** with `--embed-url` (any HTTP endpoint that
  takes JSON and returns a vector; pass auth with `--embed-header`, and the
  `--embed-input-key`/`--embed-output-key`/`--embed-batch-key` flags adapt it to
  standard or custom response shapes). No vendor is assumed or required.

**Examples:**

```bash
# Local embeddings, write and open the HTML report
tracer scan traces.jsonl --html scan.html

# Estimate savings at your teacher price and volume
tracer scan traces.jsonl --teacher-price-per-1k 5.0 --monthly-calls 3000000

# Read a thin sample anyway (best-effort floor, with a warning)
tracer scan small.jsonl --force

# Use your own embedding service instead of a local model
tracer scan traces.jsonl --embed-url https://my-embeddings.internal/embed \
    --embed-header 'Authorization: Bearer $TOKEN'

# Reuse precomputed embeddings
tracer scan traces.jsonl --embeddings traces.npy
```

The **HTML report** shows the headline certifiable share, per-cell verdicts and
proven match rates, an estimated saving, and an interactive 3D map of the
embedding space. Each dot is one request; hover a cell to inspect it, and use the
**Colour by: Verdict / Label** toggle to switch between the free-vs-kept colouring
and a per-label palette.

---

## `tracer fit`

Fit a routing policy from a JSONL traces file.

```bash
tracer fit <traces> [--artifact-dir <dir>] [--target <float>] [--trees] [--skip <models>]
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `traces` | required | Path to traces JSONL file |
| `--artifact-dir` | `.tracer` | Where to save artifacts |
| `--target` | `0.90` | Target teacher agreement (0.0-1.0) |
| `--trees` | off | Add the tree-based surrogates (decision tree, random forest, extra-trees, gradient boosting). **Off by default**, they are slower and the linear + MLP heads are usually enough; turn them on for hard, high-class-count tasks. |
| `--skip` | none | Comma-separated surrogate models to drop from the zoo (e.g. `mlp_1h,mlp_2h`). |

The default zoo is lightweight (logistic regression, SGD, small MLPs) so a fit is fast. `--trees` adds the heavier tree models, which can lift coverage on difficult datasets (e.g. Banking77's 77 classes) at the cost of fit time.

Trace files accept common key aliases: the input can be any of `input, query, text, prompt, question` and the label any of `teacher, teacher_output, label, intent, output, answer`.

**Examples:**

```bash
# Basic fit with 90% parity target (lightweight zoo)
tracer fit traces.jsonl

# Add tree models for a hard, high-class-count task
tracer fit traces.jsonl --trees

# Stricter target, custom artifact dir
tracer fit traces.jsonl --target 0.95 --artifact-dir my-policy

# Very strict -- only handle traffic the surrogate is very confident about
tracer fit traces.jsonl --target 0.99
```

**What gets printed:**
- Selected method (global / l2d / rsb / none)
- Coverage % on calibration set
- Teacher agreement on handled traffic
- Per-label coverage bars (top 15)
- Contrastive boundary pairs
- Path to HTML report (auto-generated)

**Notes:**
- TRACER looks for an embeddings file next to the traces path. If `traces.jsonl` is your traces file, TRACER looks for `traces.npy` (same stem, `.npy` extension).
- Pass embeddings explicitly from Python: `tracer.fit("traces.jsonl", embeddings=X)`
- If the surrogate can't reach the target teacher agreement, `method=none` is printed and all traffic continues to the teacher. This is expected for noisy tasks or insufficient data.

---

## `tracer update`

Refit with new traces. Accumulates all historical traces and refits from scratch.

```bash
tracer update <traces> [--artifact-dir <dir>]
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `traces` | required | Path to new traces JSONL file |
| `--artifact-dir` | `.tracer` | Artifact directory to update |

**Examples:**

```bash
# Day 2: update with 500 new traces
tracer update traces_day2.jsonl

# Using a non-default artifact dir
tracer update new.jsonl --artifact-dir my-policy
```

**What happens:**
1. New traces are appended to `.tracer/all_traces.jsonl`
2. All traces (old + new) are used to refit the policy
3. The method, threshold, and artifacts are all updated
4. Coverage typically grows with each update

---

## `tracer watch export`

Convert `tracer.watch` span JSONL into fit-ready traces (`input` / `teacher`).

```bash
tracer watch export [<source>] [-o <path>] [--include-errors]
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `source` | `.tracer/watch` | Watch JSONL file or directory of `*.jsonl` files |
| `-o`, `--output` | `traces.jsonl` | Output traces JSONL path |
| `--include-errors` | off | Include spans with `status=error` or empty input/output |

**Examples:**

```bash
# Export all watch files under .tracer/watch
tracer watch export

# Export one watcher, then fit
tracer watch export .tracer/watch/support.jsonl -o support_traces.jsonl
tracer fit support_traces.jsonl
```

By default, errored spans and rows with empty input or output text are skipped.

---

## `tracer report`

Print the policy manifest as JSON.

```bash
tracer report [<artifact-dir>]
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `artifact-dir` | `.tracer` | Artifact directory |

**Example output:**

```json
{
  "version": "0.1.0",
  "n_traces": 10003,
  "label_space": ["card_arrival", "transfer_money", ...],
  "method": "l2d",
  "target_ta": 0.95,
  "coverage": 0.928,
  "teacher_agreement": 0.950,
  "embedding_dim": 1024,
  "n_retrains": 1
}
```

**Use cases:**
- Check what policy is currently deployed
- Pipe into `jq` for scripting: `tracer report | jq .coverage`
- Monitor coverage drift over time

---

## `tracer report-html`

Generate a self-contained HTML report and open it in the browser.

```bash
tracer report-html [<artifact-dir>] [--output <path>] [--no-open]
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `artifact-dir` | `.tracer` | Artifact directory |
| `--output` | `<artifact-dir>/report.html` | Output HTML path |
| `--no-open` | off | Don't open browser automatically |

**Examples:**

```bash
# Generate and open in browser
tracer report-html

# Generate without opening (e.g. in CI)
tracer report-html --no-open

# Save to specific path
tracer report-html --output /var/www/tracer-report.html --no-open

# Different artifact dir
tracer report-html my-policy
```

**What the report shows:**
- Coverage, teacher agreement, method at a glance (4 stat cards)
- Per-label coverage table with progress bars (sorted best→worst)
- Coverage by query length (short/medium/long buckets)
- Contrastive boundary pairs with accept scores
- Representative handled and deferred examples

The HTML is fully self-contained -- no external dependencies, works offline, can be shared as a single file.

---

## `tracer sankey`

Generate an interactive Sankey diagram showing how traffic flows from labels to the surrogate or teacher.

```bash
tracer sankey [<artifact-dir>] [--output <path>] [--format <fmt>] [--top-k <n>] [--no-open]
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `artifact-dir` | `.tracer` | Artifact directory |
| `--output` | `<artifact-dir>/sankey.<format>` | Output path |
| `--format` | `html` | `html` (interactive), `png`, or `svg` |
| `--top-k` | `15` | Number of top labels to show individually |
| `--no-open` | off | Don't open browser automatically (html only) |

**Examples:**

```bash
tracer sankey

tracer sankey .tracer --format png --output routing-flow.png

tracer sankey --top-k 20 --no-open
```

**Requires:** `pip install tracer-llm[viz]`

---

## `tracer serve`

Start a lightweight HTTP prediction server (stdlib, zero extra deps).

```bash
tracer serve [<artifact-dir>] [--host <addr>] [--port <int>]
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `artifact-dir` | `.tracer` | Artifact directory |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | Listen port |

**Endpoints:**

| Method | Path | Body | Response |
|--------|------|------|----------|
| `GET` | `/health` | - | `{"status": "ok", "method", "coverage", ...}` |
| `POST` | `/predict` | `{"embedding": [...]}` | `{"label", "decision", "accept_score"}` |
| `POST` | `/predict_batch` | `{"embeddings": [[...], ...]}` | `{"labels", "decisions", "handled"}` |

**Example:**

```bash
tracer serve .tracer --port 8000

# In another terminal:
curl localhost:8000/health
curl -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"embedding": [0.1, 0.2, ...]}'
```

---

## Trace format

Each line of the JSONL file is one trace:

```jsonl
{"input": "What is my account balance?", "teacher": "check_balance"}
{"input": "Send $50 to Alice", "teacher": "transfer_money", "id": "t_001"}
{"input": "Card declined at store", "teacher": "card_blocked", "ground_truth": "card_blocked"}
```

| Field | Required | Description |
|-------|----------|-------------|
| `input` | ✓ | The input text the teacher saw |
| `teacher` | ✓ | The teacher's output label |
| `id` | - | Optional trace identifier |
| `ground_truth` | - | True label (if known) |
| `metadata` | - | Arbitrary dict |

`teacher` is the label the LLM produced -- not the ground-truth label. TRACER learns to imitate the teacher, so it uses teacher labels for training.

---

## Embeddings

TRACER expects one embedding vector per trace, as a numpy array of shape `(n_traces, embedding_dim)`.

**Auto-discovery:** If your traces are at `traces.jsonl`, TRACER looks for `traces.npy` in the same directory. Name your files consistently and embeddings are found automatically.

**Explicit path:** Pass via `--embeddings` (Python API) or save at the auto-discovery path.

**Compute embeddings:**

```bash
pip install tracer-llm[embeddings]
```

```python
import tracer, numpy as np

texts = [...]
X = tracer.embed(texts)                                   # all-MiniLM-L6-v2 (384-dim)
X = tracer.embed(texts, model="BAAI/bge-small-en-v1.5")  # 384-dim, stronger
X = tracer.embed(texts, model="BAAI/bge-m3")              # 1024-dim, multilingual
np.save("traces.npy", X)
```

**Tip:** Always use the same embedding model when calling `update()`. Mixing embedding models across refits will degrade surrogate quality.
