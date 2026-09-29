# Using TRACER from JavaScript / Node.js

TRACER trains a fixed-label student in Python. Your Node application can use its
direct decisions through the bundled HTTP sidecar and call your teacher when
the policy defers. The JS package records traces; it does not train or execute
the classifier. See [concepts](concepts.md) for the scope and quality contract.

This guide uses Python scripts, not a CLI. It does not connect to a hosted
catalog, Tracer account or Echo wallet. Provider SDK calls below belong to your
application and use your provider credentials and billing.

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

The HTTP server expects embedding vectors. Embed traces in exactly their input
row order, and use the same encoder, revision and normalization at inference.
Matching dimensions alone does not establish compatibility.

**Option A: your existing embedding API (OpenAI SDK example)**

```bash
pip install tracer-llm openai numpy
```

```python
# embed.py: run once, or in your data pipeline
import json, numpy as np
from openai import OpenAI

client = OpenAI()
texts = [json.loads(l)["input"] for l in open("traces.jsonl")]
vectors = []
for start in range(0, len(texts), 64):
    response = client.embeddings.create(
        model="text-embedding-3-small", input=texts[start:start + 64]
    )
    vectors.extend(d.embedding for d in sorted(response.data, key=lambda d: d.index))
X = np.asarray(vectors, dtype=np.float32)
np.save("traces.npy", X)  # TRACER auto-discovers this at fit time
```

These calls use your provider account. Check its input/token limits; the batch
size is illustrative and does not make arbitrary-length traces admissible.

**Option B: Local embeddings (sentence-transformers, no API charge)**

```bash
pip install 'tracer-llm[embeddings]'
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

result = tracer.fit(
    "traces.jsonl",
    config=tracer.FitConfig(target_teacher_agreement=0.95),
)
print(result.manifest.certification)
if result.manifest.selected_method is None:
    raise SystemExit("No student published; keep the teacher path active.")
```

TRACER reads `traces.jsonl` and auto-discovers `traces.npy` (same stem, `.npy` extension). Run `python fit_policy.py` offline, in a cron job, a GitHub Action, or manually. It does not touch your application.

The final reserved-sample check concerns teacher agreement, not ground-truth
accuracy. A completed failed check publishes a null policy; use a separate
candidate directory before promotion if a student is already serving. Embedding,
training and local serving still consume resources.

---

### 4. Start the HTTP server

Save this as `serve_policy.py` and run it with `python serve_policy.py`:

```python
import tracer

tracer.serve(".tracer", host="127.0.0.1", port=8000)
```

Run this as a sidecar next to your Node app, same server, same docker-compose, same machine. This example binds to `127.0.0.1:8000`. For a container sidecar, bind to `0.0.0.0` inside the container and limit network access to your application.

The server defaults to loopback, accepts JSON bodies up to 8 MiB, and uses a
10-second socket timeout. It handles requests concurrently. Invalid input returns
HTTP 400. It provides no authentication or CORS headers; for browser or remote
access, configure authentication, TLS, and CORS at your reverse proxy.

---

## Predicting from your JS app

At inference time, embed the input with the same model you used at fit time, then POST the embedding to TRACER. If the surrogate handles it, you get the label back immediately with no LLM call. If it defers, you call your LLM as usual and log the new trace.

**With OpenAI embeddings:**

Install your application's SDK with `npm install openai`. The example assumes
`callYourLLM(text)` returns a validated string from your fixed label set and
`logTrace()` is the function above.

```js
import OpenAI from 'openai'

const openai = new OpenAI()
const tracerUrl = process.env.TRACER_URL ?? 'http://localhost:8000'

async function route(text) {
  // 1. Embed the input (same model as at fit time)
  const embResponse = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: text,
  })
  const embedding = embResponse.data[0].embedding

  // 2. Ask TRACER whether to handle locally or defer
  const res = await fetch(`${tracerUrl}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ embedding }),
  })
  if (!res.ok) throw new Error(`TRACER request failed: ${res.status}`)
  const { label, decision } = await res.json()

  if (decision === 'handled') {
    return label  // surrogate answered, no LLM call
  }

  if (decision !== 'deferred') throw new Error('Unexpected TRACER decision')

  // Deferred: call your teacher. A separate sampling policy should also
  // collect teacher labels on representative accepted traffic for evaluation.
  const llmLabel = await callYourLLM(text)
  logTrace(text, llmLabel)
  return llmLabel
}
```

This skips teacher generation for accepted inputs. Measure the complete path:
embedding price and latency, local serving, network overhead and teacher calls.
No sub-millisecond or fixed savings claim follows from this example. Serving
errors above propagate explicitly; any retry/fallback-on-error policy is yours
to define and account for.

**Batch prediction:**

```js
const res = await fetch('http://localhost:8000/predict_batch', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ embeddings: [embedding1, embedding2] }),
})
if (!res.ok) throw new Error(`TRACER request failed: ${res.status}`)
const { labels, decisions, handled } = await res.json()
// handled: boolean[]: true means surrogate answered, no LLM needed
```

---

## HTTP API reference

| Method | Path | Body | Response |
|--------|------|------|----------|
| `GET` | `/health` | None | `status`, `method`, `coverage`, `teacher_agreement`, `n_labels`, `n_traces` |
| `POST` | `/predict` | `{"embedding": [float, ...]}` | `label`, `decision`, `accept_score`, `stage` |
| `POST` | `/predict_batch` | `{"embeddings": [[float, ...], ...]}` | `{"labels", "decisions", "handled"}` |

`decision` is `"handled"` (surrogate answered, no LLM call needed) or `"deferred"` (call your LLM).
Deferred labels are `null`. Rejection by the acceptance rule and by the OOD
distance guard share this public state; there is no separate reason field.
`accept_score` is a ranking signal, not certified per-request correctness.
The server does not run a teacher for either single or batch requests, and is
not an OpenAI/OpenRouter-compatible chat-completion endpoint.

---

## Continual learning

Teacher-labeled requests can supply new training traces. Keep a representative
sample of all traffic, including teacher checks on accepted requests; using
only old-policy deferrals changes the training/evaluation distribution. Do not
turn unchecked student predictions into teacher labels.

```python
# Run this script on a schedule; supply matching embeddings or a sibling .npy.
import tracer

result = tracer.update("new_traces.jsonl", new_embeddings=X_new)
print(result.manifest.certification)
```

Inspect the returned certification result before switching serving artifacts.
Restart the sidecar or reload your application router after publication. Updates
can increase or decrease coverage, or publish no policy. There is no automatic
retraining schedule or promised week-one gain in the package. Avoid repeated
tuning against reserved results; retain independent session/time evaluation.

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
    image: python:3.14-slim
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
| Fit policy | `tracer.fit()` in a Python script | When you explicitly run it |
| Serve predictions | `tracer.serve()` in a Python sidecar | While your service runs |
| Embed input at inference | JS (same model/API) | Every prediction |
| POST to TRACER | Your JS app | Every prediction |

Your application code stays in JS. Python runs in the background, invisibly.
