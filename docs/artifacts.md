# Artifact Reference

Every `tracer.fit()` or `tracer.update()` call writes a `.tracer/` directory.

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
  qualitative_report.json   ← XAI audit (slices, examples, boundary pairs)
  report.html               ← HTML report (auto-generated after fit)
```

---

## manifest.json

The top-level summary. Human-readable and machine-readable.

```json
{
  "version": "0.1.0",
  "n_traces": 10003,
  "label_space": ["card_arrival", "transfer_money", "check_balance", ...],
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
| `selected_method` | Which pipeline was deployed: `"global"`, `"l2d"`, `"rsb"`, or `null` (parity gate blocked) |
| `coverage_cal` | Fraction of calibration-set traffic handled by surrogate. Proxy for production coverage. |
| `teacher_agreement_cal` | On calibration-set handled traffic, fraction where surrogate agreed with teacher. Should be ≥ `target_teacher_agreement`. |
| `n_retrains` | How many times this artifact has been updated via `tracer.update()` |

**Null method:** If `selected_method` is `null`, coverage is 0%. All traffic routes to the teacher. This happens when no pipeline could reach the target teacher agreement on the calibration set. It's safe -- not an error.

---

## frontier.json

All candidate pipelines evaluated during fit, for every target TA in `frontier_targets`.

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
    "candidates": [...]
  },
  {
    "target": 0.95,
    "best_method": "l2d",
    "best_coverage": 0.928,
    "best_ta": 0.950,
    "candidates": [...]
  }
]
```

Use this to understand the coverage–quality tradeoff before committing to a target:

```python
import json

frontier = json.loads(open(".tracer/frontier.json").read())
for item in frontier:
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

The structured XAI report. Schema:

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
    "method":      str,         # "global", "l2d", or "rsb"
    "label_space": list[str],
    "stages": [
        {
            "surrogate":  sklearn.Pipeline,   # StandardScaler + classifier
            "acceptor":   sklearn.Pipeline | None,
            "threshold":  float | None,       # calibrated accept threshold
            "label_enc":  LabelEncoder,
        },
        # Stage 2 (RSB only)
    ]
}
```

Load and inspect:

```python
import joblib

pipeline = joblib.load(".tracer/pipeline.joblib")
stages = pipeline["pipeline"]["stages"]
surrogate = stages[0]["surrogate"]
print(type(surrogate.named_steps["clf"]))  # e.g. LogisticRegression
```

---

## Sharing artifacts

The `.tracer/` directory is fully self-contained. You can:
- Copy it to a server and `tracer.load_router("/path/to/.tracer")`
- Commit it to git (only `pipeline.joblib` and `index.embeddings.npy` are large)
- Pass it between teammates -- they only need `tracer-llm` installed
- Archive it with the traces for audit/reproducibility

Typical size:
- `pipeline.joblib`: 1–50 MB depending on model and label count
- `index.embeddings.npy`: `n_traces × dim × 4` bytes (e.g. 10k × 1024 → 40 MB)
- Everything else: < 1 MB
