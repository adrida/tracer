# Cold Start Guide

Getting good results from TRACER requires enough traces to train a reliable surrogate and calibrate the parity gate. This guide covers what to expect when starting with a small trace dataset and how to bootstrap toward full coverage.

---

## How many traces do you need?

There is no hard minimum — TRACER will attempt to fit on whatever you provide. But the quality of the routing policy depends directly on how much data the surrogate and acceptor have to learn from.

**Rules of thumb:**

| Trace volume | What to expect |
|---|---|
| **< 50 total** | Fit will likely produce `selected_method: null` (no deployable policy). The train/val/cal split leaves too few samples per class for reliable calibration. |
| **50–200** | A global surrogate _may_ deploy if the classification task is simple (few classes, clean teacher labels). L2D and RSB are unlikely to calibrate well. |
| **200–500** | L2D starts becoming viable. Expect moderate coverage (40–70%) depending on class count and task difficulty. |
| **500–1,500** | The sweet spot for most tasks. All three pipeline families compete meaningfully, and coverage typically reaches 70–90%+. |
| **1,500+** | Diminishing returns on coverage, but useful for high-class-count tasks (50+ intents) or when you need very tight parity bars (≥ 0.98). |

These numbers assume reasonably separable embeddings. Poor embeddings can make even large datasets underperform.

### Per-class minimums

TRACER's `_split_buffer` performs a stratified train/val/cal split. The splitting logic in [`src/tracer/fit/pipeline.py`](../src/tracer/fit/pipeline.py) handles small classes as follows:

- **1 sample in a class** → goes entirely to training (no validation or calibration for that class)
- **2 samples** → 1 train, 1 calibration
- **3 samples** → 1 train, 1 validation, 1 calibration
- **4+ samples** → 60/20/20 train/val/cal split

This means classes with fewer than 4 traces are underrepresented in calibration, which can degrade parity guarantees for those classes.

**Recommendation:** aim for at least **10 traces per class** before fitting, and at least **5 classes** with 20+ traces for reliable acceptor training.

---

## Bootstrapping traces from scratch

If you're evaluating TRACER for a new classification pipeline and don't yet have production LLM traces, here's how to build up a trace dataset:

### Step 1: Seed with synthetic or sampled inputs

Pick 100–300 representative inputs for your classification task. These can come from:

- A test set you already have
- Manually written examples covering each class
- A sample of recent production inputs (if available)

### Step 2: Generate teacher labels

Run each input through your LLM classification pipeline (the "teacher") and record the output:

```python
import json

traces = []
for text in seed_inputs:
    label = call_your_llm(text)  # your existing LLM pipeline
    traces.append({"input": text, "teacher": label})

with open("traces.jsonl", "w") as f:
    for t in traces:
        f.write(json.dumps(t) + "\n")
```

### Step 3: Compute embeddings

```python
import tracer
import numpy as np

# Option A: sentence-transformers (local)
X = tracer.embed([t["input"] for t in traces], model="all-MiniLM-L6-v2")
np.save("traces.npy", X)

# Option B: precomputed from your own embedding model
# Save as traces.npy next to traces.jsonl
```

### Step 4: Fit with relaxed settings

For cold-start, lower the agreement bar and minimum coverage so a policy can deploy:

```python
result = tracer.fit(
    "traces.jsonl",
    embeddings=X,
    config=tracer.FitConfig(
        target_teacher_agreement=0.85,  # relaxed from default 0.90
        min_deploy_coverage=0.02,       # accept even low coverage initially
    ),
)
```

### Step 5: Grow with `tracer.update()`

As your LLM handles more production traffic, append new traces and re-fit:

```python
# After collecting more traces
result = tracer.update("new_traces.jsonl", ".tracer", embeddings=X_new)
```

Each update round adds deferred inputs (the ones the surrogate wasn't confident about) back into the training pool, automatically expanding coverage over time. This is TRACER's core feedback loop.

---

## Diagnosing cold-start issues

### `selected_method: null` after fitting

This means no pipeline met your parity bar. Before collecting more data, check `.tracer/frontier.json`:

```bash
cat .tracer/frontier.json
```

Each entry shows the best achievable coverage and teacher agreement per target. If `best_ta` is close to your target but `best_coverage` is below `min_deploy_coverage`, try:

```python
config = tracer.FitConfig(
    target_teacher_agreement=0.85,  # lower the bar
    min_deploy_coverage=0.01,       # accept minimal coverage
)
```

If `best_ta` is well below your target even at zero coverage, the problem is likely:

1. **Embeddings don't separate classes** — try a stronger embedding model
2. **Teacher label noise** — audit traces where the surrogate disagrees
3. **Too few traces per class** — collect more data for underrepresented classes

### Low coverage despite many traces

If coverage plateaus well below 80% with 500+ traces:

- Check class imbalance: a few dominant classes might be well-covered while rare classes drag the average down
- Try the RSB pipeline family explicitly — it's designed for multi-stage residual routing
- Improve embedding quality: switching from `all-MiniLM-L6-v2` to `all-mpnet-base-v2` can significantly improve class separation

### Coverage varies between fits

See the [troubleshooting guide](troubleshooting.md#coverage-drops-between-fits) for details on why re-fits may show different coverage numbers.

---

## Summary: cold-start checklist

- [ ] Collect at least **50 traces** (ideally 200+) with **≥ 10 per class**
- [ ] Use a good embedding model (sentence-transformers `all-MiniLM-L6-v2` is a solid default)
- [ ] Start with a **relaxed** `target_teacher_agreement` (0.85) and `min_deploy_coverage` (0.02)
- [ ] Check `frontier.json` after each fit to understand what's achievable
- [ ] Use `tracer.update()` to incrementally grow coverage as you collect more production traces
- [ ] Tighten the parity bar once coverage is stable and you have 500+ traces
