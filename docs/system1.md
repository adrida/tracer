# Classification from teacher traces

TRACER turns a teacher's classification traces into a task-specific student.
The student directly predicts labels with a local ML model. An acceptance rule
and a distance-based out-of-distribution (OOD) guard determine which predictions
it handles. Your application can send the remaining requests to the teacher.

The student provides a System 1 path for a defined task and label set. Intent
classification, tagging and selection from a stable set of tools are suitable
task shapes. Fit and evaluate the student on representative examples of your
own workload.

## The teacher and student

The **teacher** supplies the reference labels in your traces. It can be an LLM
you already use through a provider such as OpenRouter, or an existing classifier.
The teacher's outputs define the agreement target, so check their quality before
using them as training labels.

The **student** learns to predict those labels from embeddings. The candidate
families include linear classifiers, trees, and shallow neural classifiers.
Classical ML can make the accepted decisions; it is not used only to choose an LLM.
The semantic encoder is supplied separately and must match between fitting and
inference. Fitting the student does not fine-tune that encoder or the teacher.

```text
Teacher-labeled traces + matching embeddings
                    |
            fit and select a student
                    |
         check the fixed serving policy
                    |
                  deploy

New input -> same embedder -> student + acceptance rule + OOD guard
                                      |                    |
                               handled locally          deferred
                                      |                    |
                                student label       caller's teacher
```

Residual stages can try additional students before deferral. Your teacher is
called only if you provide a fallback function or implement the deferred path in
your application. TRACER does not choose or bill a teacher provider for you.

## What a result means

The current OSS single-request API returns:

```python
{
    "label": "check_balance",
    "decision": "handled",
    "accept_score": 0.96,
    "stage": 0,
}
```

- `handled` means a student answered through the fitted policy.
- `deferred` means the student did not handle the request. This includes OOD
  rejections. Without a fallback, `label` is `None`; with a fallback, it contains
  the fallback's return value and the decision remains `deferred`.
- `accept_score` is a ranking signal used by the acceptance rule. It is not a
  calibrated probability that this answer is correct.

The OSS response does **not** currently expose `OOD` as a third decision value or
provide a rejection-reason field. Do not infer OOD from every deferred result.
Batch prediction returns labels and decisions; your application handles fallback
for the deferred rows. See the [API reference](api.md).

## Choose quality before measuring savings

Set `FitConfig(target_teacher_agreement=...)` to define the required agreement
on accepted traffic. In the current source, fitting reserves 20% of rows before
training or selecting candidates. The frozen final policy, including its OOD
guard, is evaluated once on that partition. Its one-sided Clopper-Pearson lower
bound must reach the target. The default `certification_alpha=0.10` specifies
the 90% one-sided confidence level of that check; it is separate from the
agreement target.

This check assumes independent, representative examples. The OSS uses a random
row split; it does not enforce session/time separation or deduplicate inputs.
Repeatedly tuning against the reserved outcomes invalidates the interpretation.
Teacher agreement is also different from correctness against independently
verified labels. The check makes no universal promise about future traffic,
each class, or each request.

Before deploying, inspect `result.manifest.selected_method` and
`result.manifest.certification`. A completed fit that fails the final check can
publish a manifest with `selected_method=None`; that directory cannot be loaded
as a deployed router. To keep a live policy while assessing a replacement, fit
to a separate artifact directory, evaluate it, then promote it yourself.

At minimum, measure on representative held-out traffic:

| Measure | What it answers |
| --- | --- |
| Accepted coverage | What fraction does the student answer directly? |
| Accepted teacher agreement | How often do those labels match the teacher? |
| Ground-truth accuracy, macro-F1 and per-class recall | Does the system solve the actual task, including minority classes? |
| Deferred and OOD behavior | Which inputs need the teacher or further investigation? |
| Complete latency and cost | Are encoding, student serving and teacher calls together worthwhile? |

Do not count every deferral as correct: measure the teacher's actual answer when
evaluating the complete system. The scan and qualitative reports help inspect
data, but they are not substitutes for a final production evaluation.

## Free OSS and the hosted Tracer app

The MIT-licensed core remains usable without a Tracer account. You can record
traces, fit and inspect students, run predictions, refit with new data, and serve
the model yourself. You provide the embedding model and teacher integration;
local compute and any provider usage remain your responsibility.

The [hosted Tracer app](https://app.tracerml.ai) is a separate managed service.
Its accounts, provider integrations and billing are not dependencies of this
package. The application changes below are under development and have their
own release cycle.

| Area | This OSS repository | Hosted application release work |
| --- | --- | --- |
| Training | Python `fit()` and `update()` on your traces | Managed jobs and versioned datasets |
| Teacher | Caller-supplied fallback | Configured teacher and strict JSON label contract |
| Serving | Python API; HTTP sidecar accepts embeddings | Authenticated text-input endpoints and tenant management |
| Trace capture | Local Python/JS wrappers; opt-in generic export | Stored shadow and routed traffic |
| OOD | Distance guard; public outcome remains `deferred` | Separate ML/deferred/OOD analytics |
| Accounts and funds | No Tracer account or wallet required | Shared Echo workspaces, credits and payment integration |
| Artifacts | Local `.tracer/` directory containing `pipeline.joblib` | Managed artifacts use a different runtime format |

Copying a local `.tracer/` directory supports OSS deployment with a compatible
Python environment and the same embedder. A hosted artifact download is not yet
a tested drop-in OSS bundle; a conversion and round-trip path is still needed.
Only load artifacts from sources you trust: `joblib` is executable serialization.

Use the Python API and HTTP sidecar for local integration. There is currently
no command-line entry point. The package contains the fitting and serving
library; you supply the traces and encoder for your task.

## Source and release status

The [changelog](../CHANGELOG.md) distinguishes repository changes from tagged
package releases. Check the installed version before relying on a documented
certification or artifact behavior.

Continue with the [quickstart](../README.md#quickstart-from-this-source),
[policy concepts](concepts.md), [artifact contract](artifacts.md), or
[troubleshooting](troubleshooting.md).
