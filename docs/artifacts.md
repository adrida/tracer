# Artifact reference

Every `tracer.fit()` or `tracer.update()` call writes a `.tracer/` directory.
It contains a task-specific student, its acceptance policy and evidence. The
student directly predicts a label; a deferred result leaves teacher execution
to the caller. See [concepts](concepts.md) for the training and serving contract.

The numerical examples below illustrate schemas, not a measured benchmark or
promised coverage. Successful calls that publish no selected method still write
a manifest, but no `pipeline.joblib` for serving. Exceptions preserve the prior
generation; a completed certification rejection instead publishes a null policy.

## Directory layout

```
.tracer/
  manifest.json             ← policy summary (load with tracer.report())
  pipeline.joblib           ← fitted model (load with tracer.load_router())
  frontier.json             ← all candidate pipelines at each target TA
  config.json               ← FitConfig used for this fit
  all_traces.jsonl          ← all traces accumulated so far (for update)
  index.embeddings.npy      ← full embedding matrix (n × dim)
  index.faiss               ← FAISS index, if faiss-cpu is installed
  ood.json                  ← distance guard metadata, when fitted
  ood_reference.npy         ← development-only guard reference, when fitted
  qualitative_report.json   ← XAI audit (slices, examples, boundary pairs)
  report.html               ← HTML audit report when generated
```

---

## manifest.json

The top-level summary. Human-readable and machine-readable.
The abbreviated example omits the full `certification` object; inspect the
stored object rather than inferring evidence from example metric values.

```json
{
  "version": "0.1.0",
  "n_traces": 10003,
  "label_space": ["card_arrival", "transfer_money", "check_balance"],
  "selected_method": "l2d",
  "target_teacher_agreement": 0.95,
  "coverage_cal": 0.928,
  "teacher_agreement_cal": 0.950,
  "embedding_dim": 1024,
  "n_retrains": 1,
  "pipeline_path": ".tracer/pipeline.joblib",
  "index_path": ".tracer/index",
  "config_path": ".tracer/config.json",
  "qualitative_report_path": ".tracer/qualitative_report.json"
}
```

**Key fields:**

| Field | Meaning |
|-------|---------|
| `selected_method` | Which pipeline was published: `"global"`, `"l2d"`, `"rsb"`, or `null` (selection/final check rejected) |
| `coverage_cal` | For new certified artifacts, fraction of final reserved examples handled by the complete runtime policy. Legacy artifacts retain their old calibration meaning. |
| `teacher_agreement_cal` | For new certified artifacts, observed teacher agreement on handled final reserved examples. |
| `certification` | Status, method, alpha, target, counts, coverage, agreement and one-sided lower bound from the final check. Missing/null on legacy artifacts. |
| `ood_required` | Whether loading requires a valid `ood.json` and `ood_reference.npy`. Defaults to false for legacy artifacts. |
| `n_retrains` | Number of fits in this artifact's history: 1 after initial fitting, incremented by each successful `tracer.update()` |

**Null method:** If `selected_method` is `null`, there is no published local
policy and `load_router()` raises an error. The caller must keep using its
teacher. Inspect `certification.status` to distinguish selection failure,
insufficient final agreement evidence, and insufficient coverage.

The certificate assumes independent, representative examples and no adaptive
reuse of reserved outcomes. It does not establish ground-truth or per-class
accuracy. A present but unreadable guard fails loading even for legacy artifacts.
Legacy guards without a separate reference retain their original stored-index
reference; they do not acquire final certification by being loaded.

---

## frontier.json

Candidate-selection diagnostics for each target in `frontier_targets`. Entries
are marked `evaluation_role: "development_selection_only"`. These examples were
used to choose the policy; their bounds are not final certification. Only the
selected policy at `target_teacher_agreement` is evaluated on the reserved data.

```json
[
  {
    "target": 0.85,
    "best_method": "global",
    "best_coverage": 1.0,
    "best_ta": 0.921,
    "candidates": [
      {"method": "global", "coverage_cal_total": 1.0, "teacher_agreement_cal_total": 0.921},
      {"method": "l2d",    "coverage_cal_total": 0.986, "teacher_agreement_cal_total": 0.850},
      {"method": "rsb",    "coverage_cal_total": 0.972, "teacher_agreement_cal_total": 0.861}
    ]
  },
  {
    "target": 0.90,
    "best_method": "global",
    "best_coverage": 1.0,
    "best_ta": 0.921,
    "candidates": []
  },
  {
    "target": 0.95,
    "best_method": "l2d",
    "best_coverage": 0.928,
    "best_ta": 0.950,
    "candidates": []
  }
]
```

Use this to inspect the development coverage/agreement tradeoff. It does not
authorize a policy that failed its final check or establish production quality:

```python
import json

frontier = json.loads(open(".tracer/frontier.json").read())
for item in frontier:
    if item["best_method"] is None:
        print(f"target={item['target']:.0%}: no development candidate")
        continue
    print(f"target={item['target']:.0%}  "
          f"method={item['best_method']}  "
          f"coverage={item['best_coverage']:.1%}  "
          f"TA={item['best_ta']:.3f}")
```

```
target=85%  method=global  coverage=100.0%  TA=0.921
target=90%  method=global  coverage=100.0%  TA=0.921
target=95%  method=l2d     coverage=92.8%   TA=0.950
```

---

## qualitative_report.json

The structured audit report, computed on supplied traces including development
rows. It explains behavior; it is not an independent held-out accuracy report.
An `accept_score` ranks acceptance and is not a correctness probability. Schema:

```json
{
  "summary": "Handled 9278/10003 (92.8%) by surrogate; deferred 725/10003 (7.2%).",
  "coverage": 0.928,
  "teacher_agreement_handled": 0.950,

  "slices": [
    {
      "slice_name": "label:card_arrival",
      "predicate": "teacher label is 'card_arrival'",
      "count": 145,
      "handled_rate": 0.965,
      "deferred_rate": 0.035,
      "teacher_agreement_handled": 0.971,
      "dominant_teacher_label": "card_arrival"
    },
    {
      "slice_name": "length:short",
      "predicate": "input length < p33",
      "count": 3334,
      "handled_rate": 0.918,
      "deferred_rate": 0.082,
      "teacher_agreement_handled": 0.951,
      "dominant_teacher_label": null
    }
  ],

  "handled_examples": [
    {
      "input_preview": "I want to activate my new card",
      "teacher_label": "activate_my_card",
      "decision": "handled",
      "local_label": "activate_my_card",
      "accept_score": 0.97,
      "trace_id": "t_001"
    }
  ],

  "deferred_examples": [
    {
      "input_preview": "the card I just got isn't working when I try to use it",
      "teacher_label": "card_not_working",
      "decision": "deferred",
      "local_label": null,
      "accept_score": 0.12,
      "trace_id": null
    }
  ],

  "boundary_pairs": [
    {
      "handled_preview":  "activate my card please",
      "deferred_preview": "I need to switch on my card",
      "teacher_label": "activate_my_card",
      "handled_score":  0.95,
      "deferred_score": 0.31
    }
  ],

  "temporal_deltas": [
    {
      "label": "card_arrival",
      "previous_handled_rate": 0.82,
      "current_handled_rate": 0.96,
      "delta": 0.14
    }
  ]
}
```

---

## pipeline.joblib

The fitted routing pipeline. Loaded automatically by `tracer.load_router()`.

Internal structure (for reference):

```python
{
    "label_space": list[str],
    "pipeline": {
        "method": str,          # "global", "l2d", or "rsb"
        "stages": [
            {
                "clf": fitted_classifier,  # sklearn estimator or pipeline
                "acceptor": fitted_acceptor_or_none,
                "accept_all": bool,
                "threshold": float | None,
            },
            # Second stage for RSB, when selected
        ],
        # Additional selection/summary fields may be present.
    },
}
```

This is a schematic: global stages set `accept_all=True` and omit acceptor and
threshold fields. Other stages carry a threshold and an optional logistic
acceptor. Labels are mapped through the bundle's shared `label_space`.

Load and inspect:

```python
import joblib

pipeline = joblib.load(".tracer/pipeline.joblib")
stages = pipeline["pipeline"]["stages"]
student = stages[0]["clf"]
print(type(student))  # pipeline or standalone estimator, depending on candidate
```

---

## Sharing artifacts

Copy the complete `.tracer/` directory together. It contains the fitted student
and policy data, but not external embedding-model weights or provider secrets.
Text inference still requires the same embedder and preprocessing; vector
inference requires matching vectors. Use compatible Python/sklearn/joblib
versions, and load only trusted joblib artifacts. You can:
- Copy it to a server and `tracer.load_router("/path/to/.tracer")`
- Version or archive it in storage appropriate to your data
- Pass it between teammates with compatible runtime dependencies
- Archive it with the traces for audit/reproducibility

`all_traces.jsonl`, reports and embeddings can contain or reveal customer data;
artifact export is not automatic redaction. Review them before publishing.

Storage depends on model and data sizes. Float32 `index.embeddings.npy` takes
approximately `n_traces × dim × 4` bytes plus its header (10k × 1024 is about
40.96 MB). The separate OOD reference, optional FAISS index, traces, reports and
classifier add to this; there is no fixed bundle-size ceiling.

This is the OSS joblib format. The hosted app's individual JSON artifact
downloads are not a documented drop-in replacement for it. The OSS package
does not automatically import hosted model catalogs, credentials or wallets.
