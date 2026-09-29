# Inspect traffic before fitting

`tracer.scan()` estimates how much labeled traffic a simple surrogate might
handle before you spend time training one. It groups embeddings by similarity
and measures label agreement on a held-out split. It does not produce a routing
policy; use `tracer.fit()` for that.
The learned student can use different decision boundaries from these clusters,
so a scan is neither a bound on achievable student coverage nor a final-policy
certificate. See [the fitting contract](concepts.md).

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
| `target` | `0.90` | Threshold applied to each cell's held-out teacher-agreement lower bound |
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

The result's historical `certifiable_share` field is the held-out traffic share
in cells whose individual bounds clear the target. Clusters and majority labels
are fitted on 70% of rows; the remaining 30% supplies the diagnostic counts.
The scan uses `alpha=max(0.01, 1-target)`, unlike `fit()`'s separately configured
final check. It does not correct for simultaneous selection across cells or
certify the quality of their selected union. Representative independent data
remain necessary; a row split does not remove correlated-session leakage.

The result includes per-cell examples and bounds, a frontier across targets,
and optional cost projections. The HTML report includes a 3D embedding map.
`savings_per_1k_calls` is simply `certifiable_share * teacher_price_per_1k`;
monthly savings multiply that by `monthly_calls / 1000`. These are gross avoided
teacher-cost illustrations, not measured net savings. They exclude embeddings,
training, serving, storage and changes in request mix. Teacher agreement also
does not measure ground-truth accuracy.
