# Starting with a small trace dataset

Choose the teacher-agreement target your application needs, then collect enough
representative traces to test it. Do not lower the target merely to make a model
publish. Dataset size alone does not determine attainable coverage.
The student learns to return your fixed labels directly. It needs examples of
your task; the OSS package does not ship a universal pretrained decision model
or automatically buy labels from a teacher. See [concepts](concepts.md).

## How much evidence is needed?

By default, 20% of rows are reserved for final certification before any training
or model selection. On a fixed policy, at alpha 0.10, at least **22 accepted,
all-correct** reserved examples are needed for a 0.90 agreement lower bound.
A 0.95 target needs at least 45, and a 0.99 target at least 230. Errors require
more evidence. These are certification counts, not total training counts.

The remaining rows must also support fitting, classifier selection and threshold
selection. Rare labels can be absent from one partition; a global agreement
bound is not a per-label guarantee. Collect additional examples and measure
per-label behavior separately when those distinctions matter.

## Prepare the data

Record the actual request and teacher label in JSONL:

```jsonl
{"input": "What is my balance?", "teacher": "check_balance"}
{"input": "Send money to Alice", "teacher": "transfer_money"}
```

Compute embeddings with the same model you will use during serving:

```python
import tracer
from tracer.traces.loader import load_traces

texts = [record.input_text for record in load_traces("traces.jsonl").records]
X = tracer.embed(texts, model="all-MiniLM-L6-v2")
result = tracer.fit(
    "traces.jsonl",
    embeddings=X,
    config=tracer.FitConfig(target_teacher_agreement=0.90),
)
print(result.manifest.certification)
```

This example requires `pip install 'tracer-llm[embeddings]'`. It runs the
embedding model locally; model download and local computation are separate
from the classical student fit. Fitting does not fine-tune that encoder.

Synthetic examples can help develop a classifier, but success on synthetic
reserved data does not establish production quality. Repeated turns from one
session and duplicate inputs also violate the independent-sample assumptions.
Keep external session/time holdouts for production evaluation.

## If no policy is published

Check `manifest.json` first. `selected_method: null` means `load_router()` will
refuse to serve a policy. The final certification status distinguishes no
candidate, failed agreement evidence, and insufficient accepted coverage.
A completed fit with this result replaces the selected artifact directory with
a null-policy generation. Keep an existing serving generation in a separate
directory if you want to review a candidate before switching traffic.

`frontier.json` shows development-only candidate diagnostics. It does not show
the best achievable production result, and its apparent success must not be used
to override a failed final check. Collect more representative data, inspect
teacher-label errors, or investigate representation quality. Do not repeatedly
choose thresholds or random seeds by looking at final certification outcomes.

## Updating

```python
result = tracer.update("new_traces.jsonl", new_embeddings=X_new)
```

An update refits and rechecks a policy; coverage can rise or fall. Include a
representative sample of all traffic, not only old-policy deferrals, if the
measured coverage is intended to describe overall traffic. The same concerns
about adaptive holdout reuse apply across repeated updates.
Teacher agreement is not ground-truth accuracy. Check important labels and
worst-case workflows on independent evaluation data before deployment.
