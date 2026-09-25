# Routing reliability ablation

`routing_reliability.py` exercises three deterministic failure cases without
external data, model downloads, or API calls. The inputs and seeds are in the
script. These are targeted regression measurements, not general workload claims.

Run the same script against each checkout with identical dependencies:

```bash
pip install -e '.[faiss]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 \
  python benchmarks/routing_reliability.py
```

Measured with Python 3.12.13, NumPy 2.5.3, scikit-learn 1.9.1, and FAISS 1.15.1.
The baseline is commit `0c918712c09a677feae80eede7069442c4e259a9`.

| Case | Baseline | Fixed |
|---|---:|---:|
| Small calibration: accepted fraction | 30.56% | 91.67% |
| Small calibration: agreement on accepted rows | 100% | 90.91% |
| Small calibration: Clopper–Pearson lower bound | 81.11% | 80.86% |
| Model trained on a label subset: label agreement | 0% | 100% |
| FAISS fit/reload: held-out coverage | 0% | 99% |
| FAISS fit/reload: agreement on handled held-out rows | Not applicable (none handled) | 100% |
| FAISS fit: caller input unchanged | No | Yes |

The small-calibration case has 36 rows, a target of 80%, and alpha 0.1.
Both selections satisfy the same lower-bound test; the correction chooses the
larger eligible set. The label-subset case compares six predictions against
known labels `{1, 3}`. The FAISS case uses 240 training rows and **600 separate,
newly generated held-out rows**, target agreement 90%, and the same linear model
candidate in both versions. OOD detection remains Euclidean and retains its
existing threshold rule.

The regression suite also covers staged-update rollback, retained recovery
backups, explicit configuration, reproducible seeds, async tracing and
cancellation, report escaping, path validation, and HTTP error/concurrency paths.
