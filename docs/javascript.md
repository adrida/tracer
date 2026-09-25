# Using TRACER from JavaScript / Node.js

TRACER is a Python package, but it integrates cleanly into any JS/Node.js pipeline through its Python API and a self-hosted HTTP server. No Python goes into your application code.

---

## The 4-step integration

### 1. Collect traces from your JS pipeline

Every time your LLM classifies an input, append the result to a JSONL file:

```js
import fs from 'fs'

function logTrace(input, label) {
  const line = JSON.stringify({ input, teacher: label })
  fs.appendFileSync('traces.jsonl', line + '\n')
}

// After every LLM classification:
const label = await callYourLLM(userInput)
logTrace(userInput, label)
```

Each line must have `input` (the text) and `teacher` (the label your LLM returned). That's all TRACER needs.

---

### 2. Compute embeddings (offline, once before fit)

The HTTP server at inference time expects embedding vectors, so you need to embed your traces before fitting. Use the same embedding model you plan to use at inference time, this is the only constraint.

**Option A: OpenAI embeddings (natural fit for JS pipelines)**

```bash
pip install tracer-llm openai numpy
```

```python
# embed.py: run once, or in your data pipeline
import json, numpy as np
from openai import OpenAI

client = OpenAI()
texts = [json.loads(l)["input"] for l in open("traces.jsonl")]
response = client.embeddings.create(model="text-embedding-3-small", input=texts)
X = np.array([d.embedding for d in response.data])
np.save("traces.npy", X)  # TRACER auto-discovers this at fit time
```

**Option B: Local embeddings (sentence-transformers, free)**

```bash
pip install tracer-llm[embeddings]
```

```python
import json
import tracer, numpy as np

texts = [json.loads(l)["input"] for l in open("traces.jsonl")]
X = tracer.embed(texts)  # all-MiniLM-L6-v2 (384-dim)
np.save("traces.npy", X)
```

---

### 3. Fit the routing policy

```python
# fit_policy.py
import tracer

tracer.fit(
    "traces.jsonl",
    config=tracer.FitConfig(target_teacher_agreement=0.95),
)
```

TRACER reads `traces.jsonl` and auto-discovers `traces.npy` (same stem, `.npy` extension). Run `python fit_policy.py` offline, in a cron job, a GitHub Action, or manually. It does not touch your application.

---

### 4. Start the HTTP server

Save this as `serve_policy.py` and run it with `python serve_policy.py`:

```python
import tracer

tracer.serve(".tracer", host="127.0.0.1", port=8000)
```

Run this as a sidecar next to your Node app, same server, same docker-compose, same machine. This example binds to `127.0.0.1:8000`. For a container sidecar, bind to `0.0.0.0` inside the container and limit network access to your application.

---

## Predicting from your JS app

At inference time, embed the input with the same model you used at fit time, then POST the embedding to TRACER. If the surrogate handles it, you get the label back immediately with no LLM call. If it defers, you call your LLM as usual and log the new trace.

**With OpenAI embeddings:**

```js
const openai = new OpenAI()

async function route(text) {
  // 1. Embed the input (same model as at fit time)
  const embResponse = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: text,
  })
  const embedding = embResponse.data[0].embedding

  // 2. Ask TRACER whether to handle locally or defer
  const res = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ embedding }),
  })
  const { label, decision } = await res.json()

  if (decision === 'handled') {
    return label  // surrogate answered, no LLM call
  }

  // Deferred: call your LLM and log the new trace for the next refit
  const llmLabel = await callYourLLM(text)
  logTrace(text, llmLabel)
  return llmLabel
}
```

The embedding API call is cheap (fractions of a cent). The TRACER HTTP call is local and sub-millisecond. You only pay the full LLM cost on deferred inputs.

**Batch prediction:**

```js
const res = await fetch('http://localhost:8000/predict_batch', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ embeddings: [embedding1, embedding2, ...] }),
})
const { labels, decisions, handled } = await res.json()
// handled: boolean[]: true means surrogate answered, no LLM needed
```

---

## HTTP API reference

| Method | Path | Body | Response |
|--------|------|------|----------|
| `GET` | `/health` |, | `{"status": "ok", "method", "coverage", ...}` |
| `POST` | `/predict` | `{"embedding": [float, ...]}` | `{"label", "decision", "accept_score"}` |
| `POST` | `/predict_batch` | `{"embeddings": [[float, ...], ...]}` | `{"labels", "decisions", "handled"}` |

`decision` is `"handled"` (surrogate answered, no LLM call needed) or `"deferred"` (call your LLM).

---

## Continual learning

Every deferred input that reaches your LLM is a new trace. Accumulate them and retrain periodically, coverage grows with each refit.

```python
# Run this script on a schedule; supply matching embeddings or a sibling .npy.
import tracer

tracer.update("new_traces.jsonl", embeddings=X_new)
```

Then restart the `serve_policy.py` process to pick up the updated policy. Coverage typically grows from ~84% at day 1 to 90%+ within a week of production traffic.

---

## docker-compose setup

In `serve_policy.py`, use `host="0.0.0.0"` so the Node container can reach the
server. The example exposes port 8000 only within the Compose network.

```yaml
services:
  app:
    build: .
    ports: ['3000:3000']
    depends_on: [tracer]
    environment:
      TRACER_URL: http://tracer:8000

  tracer:
    image: python:3.11-slim
    working_dir: /app
    volumes:
      - ./.tracer:/app/.tracer:ro
      - ./serve_policy.py:/app/serve_policy.py:ro
    command: >
      sh -c "pip install tracer-llm -q && python serve_policy.py"
    expose: ['8000']
    healthcheck:
      test: ['CMD', 'python', '-c', "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 10s
      retries: 3
```

Your Node app reads `process.env.TRACER_URL` and routes through it. Replace the Python sidecar with your preferred deployment method (ECS task, Railway service, Fly.io machine, etc.).

---

## What runs where

| Step | Where | Frequency |
|------|-------|-----------|
| Collect traces | Your JS app | Every LLM call |
| Embed traces | Python script (offline) | Before each fit |
| Fit policy | `tracer.fit()` in a Python script | On a schedule |
| Serve predictions | `tracer.serve()` in a Python sidecar | Always-on |
| Embed input at inference | JS (same model/API) | Every prediction |
| POST to TRACER | Your JS app | Every prediction |

Your application code stays in JS. Python runs in the background, invisibly.
