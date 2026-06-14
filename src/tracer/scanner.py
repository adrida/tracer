"""tracer scan: the day-one verdict on a traces file.

Answers three questions in one pass, honestly:
  1. How much of this traffic is predictable enough for a near-free model?
     (certifiable share, measured with exact binomial bounds on HELD-OUT
     data, never in-sample)
  2. What does that predictable traffic look like? (clusters with dominant
     labels and real examples)
  3. What would that be worth? (savings per 1k calls at your teacher price,
     extrapolated monthly if you tell us your volume)

The scan is a diagnostic, deliberately simple: similarity clustering plus
per-cluster exact confidence bounds. It does not train a router; `fit`
does that, learning a router with accept gates on the same traffic.

Usage:
    tracer scan traces.jsonl
    tracer scan traces.jsonl --target 0.95 --teacher-price-per-1k 5.0 \
        --monthly-calls 3000000 --html scan_report.html
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

# Accepted label-field aliases. Day-one users bring middleware dumps with
# heterogeneous schemas; the scan should meet them where they are.
_INPUT_KEYS = ("input", "query", "text", "prompt", "question")
_LABEL_KEYS = ("teacher", "teacher_output", "label", "intent", "output", "answer")

# Data-volume policy. Below MIN the held-out evidence is too thin for the
# bounds to mean anything, so a plain scan refuses and points at --force.
# SUGGESTED is the volume where cells reliably carry enough held-out members
# to certify at 0.90 without coarsening.
MIN_SCAN_TRACES = 1_000
SUGGESTED_SCAN_TRACES = 5_000


class ThinDataError(ValueError):
    """Raised when a non-forced scan is run on fewer than MIN_SCAN_TRACES."""


def _cp_lower(k: int, n: int, alpha: float) -> float:
    """Exact (Clopper-Pearson) lower confidence bound on a binomial rate."""
    if n == 0 or k == 0:
        return 0.0
    from scipy.stats import beta as _beta  # scipy ships with scikit-learn
    if k == n:
        return float(_beta.ppf(alpha, n, 1))
    return float(_beta.ppf(alpha, k, n - k + 1))


def load_scan_traces(path: Union[str, Path]) -> tuple[list[str], list[str]]:
    """Tolerant JSONL loader: accepts common input/label key aliases."""
    inputs: list[str] = []
    labels: list[str] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at line {ln}: {exc}") from exc
            inp = next((row[k] for k in _INPUT_KEYS if isinstance(row.get(k), str)), None)
            lab = next((row[k] for k in _LABEL_KEYS if isinstance(row.get(k), str)), None)
            if inp is None or lab is None:
                continue
            inputs.append(inp)
            labels.append(lab)
    if not inputs:
        raise ValueError(
            "No usable rows. Each line needs an input field "
            f"({'/'.join(_INPUT_KEYS)}) and a label field ({'/'.join(_LABEL_KEYS)})."
        )
    return inputs, labels


@dataclass
class ScanCluster:
    cluster_id: int
    n_fit: int
    n_held: int
    share: float                 # held-traffic share
    dominant_label: str
    held_correct: int            # held members matching the dominant label
    cp_lower: float              # exact lower bound on that match rate
    certifiable: bool
    examples: list[str] = field(default_factory=list)


@dataclass
class ScanResult:
    n_traces: int
    n_classes: int
    n_clusters: int
    target: float
    certifiable_share: float     # held-traffic share inside certified clusters
    certified_floor: float       # worst certified cluster's bound
    clusters: list[ScanCluster] = field(default_factory=list)
    # economics (None when no price given)
    teacher_price_per_1k: Optional[float] = None
    savings_per_1k_calls: Optional[float] = None
    monthly_calls: Optional[int] = None
    monthly_savings: Optional[float] = None
    # certifiable share at a few relaxed targets, so a single 0 at a strict
    # target still shows the user where their data sits. {target: share}
    frontier: dict[float, float] = field(default_factory=dict)
    # guidance when data is thin
    traces_needed_estimate: Optional[int] = None
    # True when run with force=True: thin-data guard bypassed, clustering
    # coarsened to concentrate held-out mass. Bounds are best-effort, not a
    # guarantee. Surfaced as a warning in the terminal and a report banner.
    forced: bool = False
    # 3D projection for the HTML report viz:
    # {"points": [[x,y,z,cluster_id], ...], "clusters": {cid: {"label","cert"}}}
    projection: Optional[dict] = None


# Categorical palette for colour-by-label in the report. Chosen to stay
# distinct and legible on the dark 3D canvas (no neon, no muddy pairs). The
# same hex list is mirrored in the viz script so Python and JS agree on which
# label gets which hue.
_LABEL_PALETTE = [
    "#38bdf8", "#fb923c", "#a3e635", "#f472b6", "#c084fc",
    "#2dd4bf", "#facc15", "#60a5fa", "#fb7185", "#4ade80",
    "#e879f9", "#fbbf24", "#34d399", "#93c5fd", "#f87171",
    "#a78bfa", "#fdba74", "#22d3ee", "#bef264", "#f9a8d4",
]


def _label_colors(labels) -> dict:
    """Stable label -> hex colour map. Sorted so the assignment is
    deterministic across runs and across related datasets."""
    uniq = sorted({str(x) for x in labels})
    return {lab: _LABEL_PALETTE[i % len(_LABEL_PALETTE)] for i, lab in enumerate(uniq)}


def scan(
    traces_path: Union[str, Path],
    *,
    target: float = 0.90,
    embeddings: Optional[np.ndarray] = None,
    model: str = "all-MiniLM-L6-v2",
    teacher_price_per_1k: Optional[float] = None,
    monthly_calls: Optional[int] = None,
    viz_layout: str = "pca",
    seed: int = 7,
    max_clusters: int = 60,
    force: bool = False,
) -> ScanResult:
    """Run the scan. Returns a ScanResult; use format_scan / scan_html to render.

    Honesty contract: every certifiable claim is an exact binomial lower
    bound computed on a held-out 30% slice the clustering never saw.

    force: on thin data a cluster may not have enough held-out members for any
    bound to clear the target, however clean the traffic actually is. With
    force=True the scan does not change the binomial maths (the bound stays the
    bound), it coarsens the clustering so each cell carries roughly the ~22
    held-out members a 0.90 target needs, trading granularity for statistical
    power, and marks the result as forced so the caller can warn loudly.
    """
    inputs, labels = load_scan_traces(traces_path)
    n = len(inputs)
    alpha = max(0.01, 1.0 - target)

    if n < MIN_SCAN_TRACES and not force:
        raise ThinDataError(
            f"scan needs at least {MIN_SCAN_TRACES:,} traces for a stable read "
            f"(we suggest ~{SUGGESTED_SCAN_TRACES:,}); this file has {n:,}. "
            f"Collect more, or pass --force to scan anyway on thin data "
            f"(results will be a best-effort floor, not a guarantee)."
        )

    if embeddings is None:
        from tracer.embeddings.index import embed_texts
        X = embed_texts(inputs, model=model)
    else:
        X = np.asarray(embeddings, dtype=np.float32)
        if X.shape[0] != n:
            raise ValueError(f"embeddings rows {X.shape[0]} != usable traces {n}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    cut = int(0.7 * n)
    fit_idx, held_idx = order[:cut], order[cut:]

    k = max(4, min(max_clusters, int(round(math.sqrt(max(1, len(fit_idx)) / 2)))))
    if force:
        # Concentrate the held-out mass: target ~22 members per cell so a clean
        # cell actually has the evidence to clear 0.90. Never below 2 cells.
        k = max(2, min(k, len(held_idx) // 22 or 2))
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from sklearn.decomposition import PCA
    Xf = X[fit_idx]
    pca = None
    if Xf.shape[1] > 64 and Xf.shape[0] > 64:
        pca = PCA(n_components=64, random_state=seed).fit(Xf)
        Xf = pca.transform(Xf).astype(np.float32)
    km_cls = MiniBatchKMeans if len(fit_idx) >= 5000 else KMeans
    km = km_cls(n_clusters=min(k, len(fit_idx)), random_state=seed, n_init=4).fit(Xf)

    Xh = X[held_idx]
    if pca is not None:
        Xh = pca.transform(Xh).astype(np.float32)
    held_assign = km.predict(Xh)
    fit_assign = km.labels_

    labels_arr = np.asarray(labels)
    py_rng = random.Random(seed)
    clusters: list[ScanCluster] = []
    n_held_total = max(1, len(held_idx))
    certified_traffic = 0
    certified_floor = 1.0
    for cid in range(int(km.n_clusters)):
        fit_members = fit_idx[fit_assign == cid]
        held_members = held_idx[held_assign == cid]
        if len(fit_members) == 0:
            continue
        uniq, counts = np.unique(labels_arr[fit_members], return_counts=True)
        dominant = str(uniq[int(counts.argmax())])
        n_held = int(len(held_members))
        k_corr = int((labels_arr[held_members] == dominant).sum()) if n_held else 0
        lower = _cp_lower(k_corr, n_held, alpha)
        certifiable = lower >= target
        if certifiable:
            certified_traffic += n_held
            certified_floor = min(certified_floor, lower)
        pool = [inputs[i] for i in fit_members[:200]]
        py_rng.shuffle(pool)
        clusters.append(ScanCluster(
            cluster_id=cid,
            n_fit=int(len(fit_members)),
            n_held=n_held,
            share=n_held / n_held_total,
            dominant_label=dominant,
            held_correct=k_corr,
            cp_lower=round(lower, 4),
            certifiable=certifiable,
            examples=pool[:3],
        ))
    clusters.sort(key=lambda c: -c.share)

    certifiable_share = certified_traffic / n_held_total
    if certifiable_share == 0:
        certified_floor = 0.0

    # Frontier: certifiable share at a small set of targets (the user's,
    # plus two relaxed ones). All exact bounds, no recompute beyond a
    # threshold comparison on each cluster's held bound.
    frontier: dict[float, float] = {}
    for tgt in sorted({round(target, 2), 0.80, 0.85, 0.90}):
        share_t = sum(c.n_held for c in clusters if c.cp_lower >= tgt) / n_held_total
        frontier[tgt] = round(share_t, 4)

    savings_1k = None
    monthly = None
    if teacher_price_per_1k is not None:
        savings_1k = certifiable_share * teacher_price_per_1k
        if monthly_calls:
            monthly = savings_1k * monthly_calls / 1000.0

    # Thin-data guidance: exact bounds need held-out mass per cluster.
    # Rough rule from the math: a cluster needs ~22 held members at 100%
    # agreement to certify 0.90 (CP lower of 22/22 at alpha=0.1 is 0.901).
    traces_needed = None
    med_held = sorted(c.n_held for c in clusters)[len(clusters) // 2] if clusters else 0
    if not force and certifiable_share < 0.05 and (med_held < 22 or n < SUGGESTED_SCAN_TRACES):
        # Point at the suggested volume where cells reliably carry enough
        # held-out members, rather than the bare minimum to clear one cell.
        traces_needed = max(0, SUGGESTED_SCAN_TRACES - n)

    # Layout of the embedding space for the HTML report (display only).
    # UMAP gives the most readable, separated clusters; t-SNE is the no-extra-
    # dependency fallback; PCA is the always-available last resort.
    projection = None
    try:
        from sklearn.decomposition import PCA as _PCA3
        full_assign = np.empty(n, dtype=int)
        full_assign[fit_idx] = fit_assign
        full_assign[held_idx] = held_assign
        cap = 4000
        sub = np.random.default_rng(seed)
        sel = np.arange(n) if n <= cap else np.sort(sub.choice(n, size=cap, replace=False))
        Xs = X[sel]
        coords = None
        if viz_layout in ("umap", "auto"):
            try:
                import umap  # optional; separates many-class data into clouds
                coords = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.12,
                                   random_state=seed).fit_transform(Xs)
            except Exception:
                coords = None
        if coords is None and (viz_layout == "tsne" or (viz_layout == "auto" and len(Xs) <= 1500)):
            try:
                from sklearn.manifold import TSNE
                pre = (_PCA3(n_components=min(50, Xs.shape[1]), random_state=seed).fit_transform(Xs)
                       if Xs.shape[1] > 50 else Xs)
                coords = TSNE(n_components=3, init="pca", perplexity=30,
                              learning_rate="auto", random_state=seed).fit_transform(pre)
            except Exception:
                coords = None
        if coords is None:  # default + always-available: dense connected layout
            coords = _PCA3(n_components=3, random_state=seed).fit_transform(Xs)
        coords = np.asarray(coords, dtype=np.float32)
        coords = coords - coords.mean(axis=0)
        # Scale by a high percentile, not the max, so the dense core fills the
        # frame (a few outliers can extend past the edge) instead of being
        # shrunk into a small central blob with lots of empty space.
        scale = float(np.percentile(np.abs(coords), 97)) or 1.0
        coords = coords / scale
        pts = [[round(float(coords[i, 0]), 3), round(float(coords[i, 1]), 3),
                round(float(coords[i, 2]), 3), int(full_assign[sel[i]])] for i in range(len(sel))]
        lab_colors = _label_colors(labels)
        cl = {int(c.cluster_id): {
                "label": c.dominant_label, "cert": bool(c.certifiable),
                "share": round(c.share, 4), "bound": round(c.cp_lower, 4),
                "n": int(c.n_held), "ex": _clean_examples(c.examples),
                "lc": lab_colors.get(c.dominant_label, "#94a3b8"),
              } for c in clusters}
        projection = {"points": pts, "clusters": cl}
    except Exception:
        projection = None

    return ScanResult(
        n_traces=n,
        n_classes=len(set(labels)),
        n_clusters=len(clusters),
        target=target,
        certifiable_share=round(certifiable_share, 4),
        certified_floor=round(certified_floor, 4),
        clusters=clusters,
        teacher_price_per_1k=teacher_price_per_1k,
        savings_per_1k_calls=round(savings_1k, 4) if savings_1k is not None else None,
        monthly_calls=monthly_calls,
        monthly_savings=round(monthly, 2) if monthly is not None else None,
        frontier=frontier,
        traces_needed_estimate=traces_needed,
        forced=force,
        projection=projection,
    )


def format_scan(r: ScanResult) -> str:
    """Coloured terminal rendering, matching the rest of the tracer CLI.

    Colours degrade to plain text automatically off a TTY (see cli._ui._C).
    """
    try:
        from tracer.cli._ui import _C, hr
        c = _C()
        rule = hr
    except Exception:                       # pragma: no cover - cli always present
        class _Plain:
            def __getattr__(self, _): return ""
        c = _Plain()
        def rule(char="-", width=56): return char * width

    L: list[str] = []
    pct = r.certifiable_share * 100
    head_c = c.GREEN if r.certifiable_share > 0 else c.DIM

    L.append("")
    L.append(f"  {c.BOLD}{head_c}{pct:.1f}%{c.RESET}{c.BOLD} of traffic certifiable for a near-free model{c.RESET}")
    L.append(f"  {c.DIM}{r.n_traces:,} traces  ·  {r.n_classes} labels  ·  {r.n_clusters} cells  ·  target {r.target:.0%} agreement{c.RESET}")
    L.append(f"  {rule()}")

    if r.forced:
        L.append(f"  {c.YELLOW}{c.BOLD}⚠  forced scan{c.RESET}  {c.YELLOW}thin data: clustering coarsened to concentrate held-out evidence.{c.RESET}")
        L.append(f"     {c.DIM}Bounds are a best-effort floor, not a guarantee. Collect more traces or run `tracer fit` for the real number.{c.RESET}")

    if r.certifiable_share > 0:
        L.append(f"  {c.DIM}{'worst certified bound':<22}{c.RESET}{c.BOLD}{c.GREEN}{r.certified_floor:.3f}{c.RESET}  {c.DIM}exact, held-out{c.RESET}")
    if r.savings_per_1k_calls is not None:
        L.append(f"  {c.DIM}{'savings / 1k calls':<22}{c.RESET}{c.BOLD}${r.savings_per_1k_calls:.2f}{c.RESET}  {c.DIM}at ${r.teacher_price_per_1k}/1k teacher{c.RESET}")
    if r.monthly_savings is not None:
        L.append(f"  {c.DIM}{'monthly savings':<22}{c.RESET}{c.BOLD}{c.GREEN}${r.monthly_savings:,.0f}{c.RESET}  {c.DIM}at {r.monthly_calls:,} calls/mo{c.RESET}")

    if r.traces_needed_estimate:
        L.append("")
        L.append(f"  {c.YELLOW}⚠{c.RESET}  Not enough held-out evidence yet. Collect roughly "
                 f"{c.BOLD}{r.traces_needed_estimate:,} more traces{c.RESET} and rescan,")
        L.append(f"     {c.DIM}or pass --force to certify on what you have. Exact bounds need ~22 held-out examples per cell.{c.RESET}")

    if r.frontier:
        L.append("")
        L.append(f"  {c.BOLD}{c.YELLOW}Certifiable share by target{c.RESET}  {c.DIM}(lightweight scan estimate){c.RESET}")
        L.append(f"  {rule('·')}")
        for tgt in sorted(r.frontier):
            share = r.frontier[tgt]
            sc = c.GREEN if share >= 0.30 else (c.YELLOW if share > 0 else c.DIM)
            L.append(f"  {c.DIM}target {tgt:.0%}{c.RESET}   {c.BOLD}{sc}{share*100:5.1f}%{c.RESET}")

    L.append("")
    L.append(f"  {c.BOLD}{c.YELLOW}Cells{c.RESET}  {c.DIM}(top {min(20, len(r.clusters))} by traffic share){c.RESET}")
    L.append(f"  {rule('·')}")
    L.append(f"  {c.DIM}{'share':>6} {'held':>5} {'bound':>6}  verdict   dominant label{c.RESET}")
    for cl in r.clusters[:20]:
        if cl.certifiable:
            tag = f"{c.GREEN}✔ free {c.RESET}"
            bcol = c.GREEN
        else:
            tag = f"{c.RED}→ keep {c.RESET}"
            bcol = c.DIM
        L.append(f"  {cl.share*100:>5.1f}% {cl.n_held:>5} {bcol}{cl.cp_lower:>6.3f}{c.RESET}  {tag}  {c.BOLD}{cl.dominant_label[:38]}{c.RESET}")
        if cl.examples:
            ex = " ".join(cl.examples[0].split())[:68]
            L.append(f"  {c.DIM}{'':>19}e.g. “{ex}”{c.RESET}")
    if len(r.clusters) > 20:
        L.append(f"  {c.DIM}... {len(r.clusters) - 20} more cells{c.RESET}")

    L.append("")
    L.append(f"  {c.DIM}Certifiable = an exact binomial lower bound on held-out label agreement clears your target. No in-sample numbers.{c.RESET}")
    L.append(f"  {c.DIM}Next: {c.RESET}{c.CYAN}tracer fit{c.RESET}{c.DIM} trains a real router with accept gates and certifies more on the same traffic.{c.RESET}")
    L.append("")
    return "\n".join(L)


_SCAN_CSS = """
 :root{--ink:#1c1917;--muted:#6b7280;--line:#e7e5e4;--green:#16a34a;--red:#dc2626;--sky:#0ea5e9;--orange:#f97316}
 *{box-sizing:border-box}
 body{font-family:'Manrope',-apple-system,Segoe UI,sans-serif;color:var(--ink);max-width:880px;margin:0 auto;padding:40px 22px 80px;line-height:1.5}
 .logo{display:flex;align-items:center;gap:10px;font-weight:800;font-size:18px;color:var(--ink);text-decoration:none;width:fit-content}
 .logo:hover .src{color:var(--ink)}
 .dots i{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:5px}
 .src{color:var(--muted);font-weight:600}
 .foot-brand{display:block;margin-top:10px}
 .foot-brand a{display:inline-flex;align-items:center;gap:5px;color:var(--ink);font-weight:700;text-decoration:none}
 .foot-brand a:hover{opacity:.7}
 .foot-brand .dots i{width:8px;height:8px;margin-right:3px}
 h1{font-size:44px;line-height:1.06;letter-spacing:-0.02em;margin:26px 0 6px}
 h1 .u{color:var(--green)}
 .sub{font-size:18px;margin:0 0 6px;font-weight:600}
 .meta{color:var(--muted);font-size:13.5px;font-family:'JetBrains Mono',monospace;margin:0 0 16px}
 .save{font-size:16px;margin:0 0 8px}
 .note{background:#f5f5f4;border:1px solid var(--line);border-radius:12px;padding:14px 16px;font-size:15px;color:#44403c;margin:18px 0}
 .note.warn{background:#fff7ed;border-color:#fed7aa;color:#9a3412}
 .note b{color:var(--ink)}
 .vizwrap{margin:22px 0}
 #viz{width:100%;height:600px;border-radius:18px;background:radial-gradient(120% 120% at 50% 0%,#11161f 0%,#070a0f 100%);overflow:hidden;position:relative}
 .vizcap{color:var(--muted);font-size:13.5px;margin-top:10px}
 .legend{position:absolute;left:14px;top:12px;font-family:'JetBrains Mono',monospace;font-size:12px;color:#cbd5e1;z-index:2}
 .legend span{display:inline-flex;align-items:center;gap:6px;margin-right:14px}
 .legend i{width:9px;height:9px;border-radius:50%;display:inline-block}
 table{border-collapse:collapse;width:100%;margin-top:10px;font-size:14px}
 th{text-align:left;color:var(--muted);font-weight:700;font-size:12px;letter-spacing:.04em;text-transform:uppercase;padding:10px;border-bottom:1px solid var(--line)}
 td{padding:12px 10px;border-bottom:1px solid var(--line);vertical-align:top}
 td.num{text-align:right;font-variant-numeric:tabular-nums;font-weight:700}
 .lab{font-weight:700}
 .ex{color:var(--muted);font-size:12.5px;margin-top:3px}
 .pill{padding:4px 10px;border-radius:999px;font-size:12px;font-weight:700;white-space:nowrap}
 .hint{color:var(--muted);font-size:11px;font-weight:500;text-transform:none;letter-spacing:0}
 footer{color:var(--muted);font-size:13px;margin-top:30px;border-top:1px solid var(--line);padding-top:16px}
 code{font-family:'JetBrains Mono',monospace;background:#f5f5f4;padding:1px 6px;border-radius:5px}
 .cellcard{position:absolute;top:12px;right:12px;width:290px;max-width:calc(100% - 24px);background:rgba(8,11,16,.92);border:1px solid rgba(255,255,255,.12);border-radius:14px;padding:14px 16px;color:#e5e7eb;display:none;z-index:3;-webkit-backdrop-filter:blur(8px);backdrop-filter:blur(8px);box-shadow:0 10px 30px rgba(0,0,0,.45)}
 .cellcard.on{display:block}
 .cellcard .cc-pill{display:inline-block;padding:3px 10px;border-radius:999px;font-size:11px;font-weight:800;letter-spacing:.01em;margin-bottom:9px}
 .cellcard .cc-label{font-size:17px;font-weight:800;color:#fff;margin:0 0 9px;line-height:1.2}
 .cellcard .cc-stats{display:flex;gap:18px;font-family:'JetBrains Mono',monospace;font-size:12px;color:#94a3b8;margin-bottom:8px}
 .cellcard .cc-stats b{color:#fff;font-weight:700}
 .cellcard .cc-ex{color:#94a3b8;font-size:12px;line-height:1.5;margin-top:6px;border-top:1px solid rgba(255,255,255,.1);padding-top:8px}
 .cellcard .cc-ex .cc-ex-h{color:#64748b;font-size:10px;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px}
 .cellcard .cc-ex div.q{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:2px}
 .vizhint{display:inline-flex;align-items:center;gap:7px;margin:0;padding:7px 14px;border-radius:999px;background:#f5f5f4;border:1px solid var(--line);color:var(--ink);font-size:12.5px;font-weight:700;transition:opacity .4s;animation:vizfade 2.8s ease-in-out infinite}
 .vizhint.hide{opacity:0;animation:none}
 .vizhint::after{content:"\\2193";color:var(--sky);font-weight:800;font-size:13px}
 @keyframes vizfade{0%,100%{opacity:.82}50%{opacity:1}}
 .scan-cta{margin:26px 0 0;padding:18px 20px;border:1px solid #bbf7d0;border-radius:12px;background:#f0fdf4}
 .scan-cta h3{margin:0 0 6px;font-size:16px;color:var(--ink)}
 .scan-cta p{margin:0 0 10px;font-size:14.5px;color:#3f6212}
 .scan-cta p:last-child{margin-bottom:0}
 .scan-cta code{font-family:'JetBrains Mono',monospace;background:#dcfce7;padding:2px 7px;border-radius:5px;font-size:13px;color:#166534}
 html{scroll-behavior:smooth}
 .top-cta{display:inline-flex;align-items:center;gap:7px;margin:2px 0 16px;padding:9px 17px;border-radius:999px;background:var(--ink);color:#fff;font-size:13.5px;font-weight:700;text-decoration:none}
 .top-cta:hover{opacity:.88}
 .top-cta code{font-family:'JetBrains Mono',monospace;background:rgba(255,255,255,.16);color:#fff;padding:1px 6px;border-radius:5px;font-size:12.5px}
 .vizbar{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin:0 0 10px}
 .viztoggle{display:inline-flex;align-items:center;gap:6px}
 .vt-cap{font-size:12px;color:var(--muted);margin-right:2px}
 .vt{font:inherit;font-size:12.5px;font-weight:700;cursor:pointer;padding:6px 13px;border-radius:999px;border:1px solid var(--line);background:#fff;color:var(--muted)}
 .vt:hover{border-color:var(--ink)}
 .vt.on{background:var(--ink);color:#fff;border-color:var(--ink)}
 .legend .legmore{color:var(--muted)}
 .texttip{position:fixed;max-width:380px;background:#1c1917;color:#f5f5f4;font-size:13px;line-height:1.55;padding:11px 14px;border-radius:10px;box-shadow:0 12px 34px rgba(0,0,0,.3);z-index:50;pointer-events:none;display:none;white-space:pre-wrap;word-break:break-word}
 .texttip.on{display:block}
"""

_VIZ_HTML = """ <div class="vizwrap">
   <div class="vizbar">
     <span id="vizhint" class="vizhint">Hover any cloud to inspect a cell</span>
     <div class="viztoggle"><span class="vt-cap">Colour by</span><button id="vt-verdict" class="vt on" type="button">Verdict</button><button id="vt-label" class="vt" type="button">Label</button></div>
   </div>
   <div id="viz"><div class="legend" id="vizlegend"></div><div id="cellcard" class="cellcard"></div></div>
   <p class="vizcap">Drag to rotate, scroll to zoom. Every dot is one request, placed by meaning so similar questions sit together. Switch the colouring to see verdict (free vs kept) or the dominant label of each cell.</p>
 </div>"""

_VIZ_SCRIPT = """<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
(function(){
 var VIZ=__VIZ_DATA__;
 var panel=document.getElementById('viz'); if(!panel||!window.THREE){return;}
 var card=document.getElementById('cellcard');
 var hint=document.getElementById('vizhint');
 function hideHint(){ if(hint) hint.className='vizhint hide'; }
 var H=600, W=panel.clientWidth;
 var scene=new THREE.Scene();
 var camera=new THREE.PerspectiveCamera(58, W/H, 0.01, 100); camera.position.set(0,0,2.7);
 var renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
 renderer.setSize(W,H); renderer.setPixelRatio(Math.min(2.5,window.devicePixelRatio||1));
 panel.appendChild(renderer.domElement);

 // crisp glowing disc: solid core + a thin halo (not a fuzzy blob)
 var SZ=128, cv=document.createElement('canvas'); cv.width=cv.height=SZ; var g=cv.getContext('2d');
 var grd=g.createRadialGradient(SZ/2,SZ/2,0,SZ/2,SZ/2,SZ/2);
 grd.addColorStop(0.00,'rgba(255,255,255,1)');
 grd.addColorStop(0.55,'rgba(255,255,255,1)');
 grd.addColorStop(1.00,'rgba(255,255,255,0)');
 g.fillStyle=grd; g.beginPath(); g.arc(SZ/2,SZ/2,SZ/2,0,Math.PI*2); g.fill();
 var tex=new THREE.Texture(cv); tex.minFilter=THREE.LinearFilter; tex.magFilter=THREE.LinearFilter;
 tex.anisotropy=4; tex.needsUpdate=true;

 var P=VIZ.points||[], C=VIZ.clusters||{};
 var N=P.length;
 var pos=new Float32Array(N*3), col=new Float32Array(N*3), base=new Float32Array(N*3);
 var verdCol=new Float32Array(N*3), labCol=new Float32Array(N*3);
 var cids=new Int32Array(N);
 var GREEN=[0.20,0.92,0.47], RED=[1.0,0.40,0.42], HILITE=[0.85,1.0,1.0], GREY=[0.58,0.64,0.70];
 function hex2rgb(h){ if(!h){return GREY;} h=h.replace('#',''); if(h.length===3){h=h[0]+h[0]+h[1]+h[1]+h[2]+h[2];}
   return [parseInt(h.substr(0,2),16)/255, parseInt(h.substr(2,2),16)/255, parseInt(h.substr(4,2),16)/255]; }
 for(var i=0;i<N;i++){ pos[3*i]=P[i][0]; pos[3*i+1]=P[i][1]; pos[3*i+2]=P[i][2];
   var id=P[i][3]; cids[i]=id; var cl=C[id];
   var v=(cl&&cl.cert)?GREEN:RED; var lc=hex2rgb(cl&&cl.lc);
   verdCol[3*i]=v[0]; verdCol[3*i+1]=v[1]; verdCol[3*i+2]=v[2];
   labCol[3*i]=lc[0]; labCol[3*i+1]=lc[1]; labCol[3*i+2]=lc[2];
   base[3*i]=v[0]; base[3*i+1]=v[1]; base[3*i+2]=v[2];
   col[3*i]=v[0]; col[3*i+1]=v[1]; col[3*i+2]=v[2]; }
 var mode='verdict';
 var geo=new THREE.BufferGeometry();
 geo.setAttribute('position', new THREE.BufferAttribute(pos,3));
 var colAttr=new THREE.BufferAttribute(col,3); geo.setAttribute('color', colAttr);
 var mat=new THREE.PointsMaterial({size:0.05,map:tex,vertexColors:true,sizeAttenuation:true,transparent:true,opacity:0.78,depthWrite:false,blending:THREE.NormalBlending});
 var cloud=new THREE.Points(geo,mat); scene.add(cloud);

 var controls=null;
 if(THREE.OrbitControls){
   controls=new THREE.OrbitControls(camera, renderer.domElement);
   controls.enableDamping=true; controls.dampingFactor=0.08; controls.enablePan=false;
   controls.autoRotate=true; controls.autoRotateSpeed=2.4; controls.minDistance=1.0; controls.maxDistance=8;
 }

 function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
 var ray=new THREE.Raycaster(); ray.params.Points.threshold=0.05;
 var mouse=new THREE.Vector2(); var hover=-1;
 function paint(id){
   // Keep every dot lit; only the hovered cell changes colour (Goodfire-style).
   for(var i=0;i<N;i++){
     var hit=(id>=0 && cids[i]===id);
     col[3*i]=hit?HILITE[0]:base[3*i]; col[3*i+1]=hit?HILITE[1]:base[3*i+1]; col[3*i+2]=hit?HILITE[2]:base[3*i+2];
   }
   colAttr.needsUpdate=true;
 }
 var legend=document.getElementById('vizlegend');
 function buildLegend(){
   if(!legend){return;}
   if(mode==='verdict'){
     legend.innerHTML='<span><i style="background:#22c55e"></i>auto-answered, free</span><span><i style="background:#ef4444"></i>stays on your model</span>';
     return;
   }
   // Label mode: one swatch per dominant label, ordered by traffic share.
   var seen={}, items=[];
   var ids=Object.keys(C).map(Number).sort(function(a,b){return (C[b].share||0)-(C[a].share||0);});
   for(var j=0;j<ids.length;j++){ var cl=C[ids[j]]; var lab=cl.label||'?';
     if(seen[lab]){continue;} seen[lab]=1;
     items.push('<span><i style="background:'+(cl.lc||'#94a3b8')+'"></i>'+esc(lab)+'</span>');
     if(items.length>=8){ break; } }
   if(ids.length>items.length){ items.push('<span class="legmore">+ more</span>'); }
   legend.innerHTML=items.join('');
 }
 function setMode(m){
   if(m===mode){return;} mode=m;
   var src=(m==='label')?labCol:verdCol;
   for(var i=0;i<N*3;i++){ base[i]=src[i]; }
   var bv=document.getElementById('vt-verdict'), bl=document.getElementById('vt-label');
   if(bv) bv.className=(m==='verdict')?'vt on':'vt';
   if(bl) bl.className=(m==='label')?'vt on':'vt';
   buildLegend(); paint(hover);
 }
 function showCard(id){
   var cl=C[id]; if(!cl){card.className='cellcard';return;}
   var safe=!!cl.cert, pc=safe?'#22c55e':'#ef4444', pt=safe?'Auto-answer, free':'Keep on your model';
   var share=(cl.share!=null)?(cl.share*100).toFixed(1)+'%':'';
   var bound=(cl.bound!=null)?Math.round(cl.bound*100)+'%':'';
   var ex=(cl.ex||[]).map(function(s){return '<div class="q">&ldquo;'+esc(s)+'&rdquo;</div>';}).join('');
   card.innerHTML='<span class="cc-pill" style="background:'+pc+'26;color:'+pc+'">'+pt+'</span>'+
     '<div class="cc-label">'+esc(cl.label||'')+'</div>'+
     '<div class="cc-stats"><span>traffic <b>'+share+'</b></span><span>match rate <b>'+bound+'</b></span></div>'+
     (ex?('<div class="cc-ex"><div class="cc-ex-h">sounds like</div>'+ex+'</div>'):'');
   card.className='cellcard on';
 }
 function setHover(id){
   if(id===hover) return; hover=id; paint(id);
   if(controls) controls.autoRotate=(id<0);
   if(id<0){ card.className='cellcard'; } else { hideHint(); showCard(id); }
 }
 renderer.domElement.addEventListener('pointermove', function(e){
   var r=renderer.domElement.getBoundingClientRect();
   mouse.x=((e.clientX-r.left)/r.width)*2-1; mouse.y=-((e.clientY-r.top)/r.height)*2+1;
   ray.setFromCamera(mouse, camera); var hits=ray.intersectObject(cloud);
   setHover(hits.length? cids[hits[0].index] : -1);
 });
 renderer.domElement.addEventListener('pointerleave', function(){ setHover(-1); });
 renderer.domElement.addEventListener('pointerdown', hideHint);

 var btnV=document.getElementById('vt-verdict'), btnL=document.getElementById('vt-label');
 if(btnV) btnV.addEventListener('click', function(){ setMode('verdict'); });
 if(btnL) btnL.addEventListener('click', function(){ setMode('label'); });
 buildLegend();

 function loop(){ requestAnimationFrame(loop); if(controls){controls.update();} else if(hover<0){cloud.rotation.y+=0.010;} renderer.render(scene,camera); }
 loop();
 window.addEventListener('resize', function(){ var w=panel.clientWidth; renderer.setSize(w,H); camera.aspect=w/H; camera.updateProjectionMatrix(); });
})();
</script>"""


# Full-text tooltip for truncated example rows in the table. Long customer
# inputs get clipped to 120 chars in the cell; hovering shows the whole thing
# in a clean dark bubble that follows the cursor and never overflows the viewport.
_TEXTTIP_SCRIPT = """<script>
(function(){
 var tip=document.getElementById('texttip'); if(!tip) return;
 function place(e){
   var pad=14, w=tip.offsetWidth, h=tip.offsetHeight;
   var x=e.clientX+16, y=e.clientY+16;
   if(x+w+pad>window.innerWidth) x=e.clientX-w-16;
   if(y+h+pad>window.innerHeight) y=e.clientY-h-16;
   tip.style.left=Math.max(pad,x)+'px'; tip.style.top=Math.max(pad,y)+'px';
 }
 document.querySelectorAll('.ex[data-full]').forEach(function(el){
   var full=el.getAttribute('data-full'); if(!full) return;
   el.style.cursor='help';
   el.addEventListener('pointerenter', function(e){ tip.textContent='“'+full+'”'; tip.className='texttip on'; place(e); });
   el.addEventListener('pointermove', place);
   el.addEventListener('pointerleave', function(){ tip.className='texttip'; });
 });
})();
</script>"""


def scan_html(r: ScanResult, source_name: str = "traces") -> str:
    """Self-contained, brand-styled HTML report with a 3D embedding view."""
    import json as _json
    GREEN, RED = "#16a34a", "#dc2626"
    pct = r.certifiable_share * 100

    rows = []
    for c in r.clusters:
        color = GREEN if c.certifiable else RED
        verdict = "Auto-answer, free" if c.certifiable else "Keep on your model"
        exs = "".join(
            f"<div class='ex' data-full=\"{_esc(e)}\">&ldquo;{_esc(e[:120])}{'…' if len(e) > 120 else ''}&rdquo;</div>"
            for e in c.examples[:2]
        )
        rows.append(
            f"<tr><td><div class='lab'>{_esc(c.dominant_label)}</div>{exs}</td>"
            f"<td class='num'>{c.share*100:.1f}%</td>"
            f"<td class='num'>{c.cp_lower*100:.0f}%</td>"
            f"<td><span class='pill' style='background:{color}14;color:{color}'>{verdict}</span></td></tr>"
        )

    money = ""
    if r.savings_per_1k_calls is not None:
        money = f"<b>${r.savings_per_1k_calls:.2f}</b> saved per 1,000 calls"
        if r.monthly_savings is not None:
            money += f" &middot; about <b>${r.monthly_savings:,.0f}/month</b> at {r.monthly_calls:,} calls"

    thin = ""
    if r.traces_needed_estimate:
        thin = (f"<div class='note warn'>Not enough data yet to prove much. Collect about "
                f"<b>{r.traces_needed_estimate:,} more requests</b> and run this again. "
                f"Each group needs roughly 22 held-out examples before we can vouch for it.</div>")
    if r.forced:
        thin = ("<div class='note warn'><b>Forced scan on limited data.</b> The grouping was "
                "coarsened to squeeze the most evidence out of a small sample, so every number "
                "below is a best-effort floor, not a guarantee. Collect more requests, or run "
                "<code>tracer fit</code>, for a number you can quote.</div>")

    has_viz = bool(r.projection and r.projection.get("points"))
    viz_block = _VIZ_HTML if has_viz else ""
    script = _VIZ_SCRIPT.replace("__VIZ_DATA__", _json.dumps(r.projection)) if has_viz else ""
    save_html = f"<p class='save'>{money}</p>" if money else ""

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tracer scan: {_esc(source_name)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Manrope:wght@500;700;800&display=swap" rel="stylesheet">
<style>{_SCAN_CSS}</style></head>
<body>
 <a class="logo" href="https://tracerml.ai" target="_blank" rel="noopener"><span class="dots"><i style="background:#0ea5e9"></i><i style="background:#f97316"></i><i style="background:#dc2626"></i></span>Tracer scan <span class="src">&middot; {_esc(source_name)}</span></a>
 <h1><span class="u">{pct:.0f}%</span> of your traffic can be answered for free</h1>
 <p class="sub">A near-free model already matches your current model on {pct:.0f}% of requests, proven on held-out examples, not guessed.</p>
 <p class="meta">{r.n_traces:,} requests &middot; {r.n_clusters} groups of similar questions &middot; target {r.target:.0%} agreement</p>
 <a href="#run-fit" class="top-cta">Train the real router with <code>tracer fit</code> &rarr;</a>
 {save_html}
 {thin}
 <div class="note"><b>How to read this.</b> We grouped your requests into clusters of similar questions and laid them out in the space below. For each cluster we checked, on examples it never saw, how often a tiny free model agrees with your model. <b style="color:#16a34a">Green</b> means it agrees at least {r.target:.0%} of the time, so it is safe to auto-answer for free. <b style="color:#dc2626">Red</b> means we could not prove that yet, so those stay on your model.</div>
 {viz_block}
 <table>
  <tr><th>What customers ask</th><th style="text-align:right">Share of traffic</th><th style="text-align:right">Match rate <span class="hint">(proven)</span></th><th>Verdict</th></tr>
  {''.join(rows)}
 </table>
 <div class="scan-cta" id="run-fit">
   <h3>This is a fast, conservative estimate. Train the real router for more.</h3>
   <p>This 2-minute scan groups your traffic by similarity, a deliberately conservative read, so the real number is usually higher. <code>tracer fit</code> trains the actual router with accept gates and certifies a larger share of the same traffic.</p>
   <p><code>pip install tracer-llm</code> &nbsp;then&nbsp; <code>tracer fit your_traces.jsonl</code></p>
 </div>
 <footer>Every number here is an exact lower bound measured on held-out data the grouping never saw, no in-sample optimism.
   <span class="foot-brand"><a href="https://tracerml.ai" target="_blank" rel="noopener"><span class="dots"><i style="background:#0ea5e9"></i><i style="background:#f97316"></i><i style="background:#dc2626"></i></span>tracerml.ai</a></span></footer>
 <div id="texttip" class="texttip"></div>
 {script}
 {_TEXTTIP_SCRIPT}
</body></html>"""


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _clean_examples(exs, n: int = 3, maxlen: int = 80) -> list:
    """Adaptive example snippets for the hover card. Strips a shared generic
    lead-in (a greeting or a marker like 'INTENT: ') when every example in the
    cell starts the same way, and truncates, so long or boilerplate-prefixed
    inputs stay readable. Only strips at a ': ' or ', ' boundary so real
    content is never eaten."""
    import os as _os
    cleaned = [" ".join(str(e).split()) for e in exs if e and str(e).strip()]
    if not cleaned:
        return []
    pre = _os.path.commonprefix(cleaned)
    for delim in (": ", ", "):
        idx = pre.rfind(delim)
        if idx != -1:
            cut = idx + len(delim)
            cleaned = [e[cut:] if len(e) > cut else e for e in cleaned]
            break
    out = []
    for e in cleaned[:n]:
        e = e.strip()
        if len(e) > maxlen:
            e = e[:maxlen - 1].rstrip() + "…"
        out.append(e)
    return out
