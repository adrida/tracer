# Inspect traffic before fitting

`tracer.scan()` estimates how much labeled traffic a simple surrogate might
handle before you spend time training one. It groups embeddings by similarity
and measures label agreement on a held-out split. It does not produce a routing
policy; use `tracer.fit()` for that.

```python
from pathlib import Path
import numpy as np
import tracer
from tracer.scanner import format_scan, scan_html

X = np.load("traces.npy")  # one embedding per usable input/label row
result = tracer.scan(
    "traces.jsonl",
    embeddings=X,
    target=0.95,
    teacher_price_per_1k=5.0,
    monthly_calls=3_000_000,
)
print(format_scan(result))
Path("scan.html").write_text(scan_html(result, "My traces"), encoding="utf-8")
```

The loader accepts `input`, `query`, `text`, `prompt`, or `question` for input,
and `teacher`, `teacher_output`, `label`, `intent`, `output`, or `answer` for
labels. Rows without usable string fields are skipped, so supplied embeddings
must match the remaining rows in order.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `target` | `0.90` | Minimum held-out agreement bound for a certifiable cell |
| `embeddings` | `None` | Precomputed NumPy array of shape `(n, dim)` |
| `model` | `all-MiniLM-L6-v2` | Local embedding model used when no array is supplied; requires the `embeddings` extra |
| `teacher_price_per_1k` | `None` | Teacher cost per 1,000 calls for the savings estimate |
| `monthly_calls` | `None` | Volume used for a monthly savings estimate |
| `viz_layout` | `pca` | Report projection: `pca`, `umap`, `tsne`, or `auto`; UMAP requires `umap-learn` |
| `seed` | `7` | Split and clustering seed |
| `max_clusters` | `60` | Maximum number of similarity cells |
| `force` | `False` | Allow a clearly marked thin-data estimate |

At least 1,000 usable traces are required by default; around 5,000 is recommended.
`force=True` coarsens the clusters to increase held-out evidence per cell and
marks the result as forced. It does not relax the statistical bound.

The result includes the certifiable share, per-cell examples and bounds, a
frontier across agreement targets, and optional savings estimates. The HTML
report includes a 3D embedding map. Savings are projections at the supplied cost
and volume, not measured production outcomes.
