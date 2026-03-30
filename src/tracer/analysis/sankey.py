"""Generate a Sankey diagram showing the TRACER routing flow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union


_PALETTE = [
    "#4f7cac", "#7fb069", "#e0a458", "#c97b84", "#8f7aa7",
    "#5da9a1", "#d4826a", "#6b9ac4", "#a3b86c", "#c9a96e",
    "#7c9eb2", "#b0855a", "#6aab9e", "#b88fa3", "#8ab07c",
    "#c4956a", "#6994b3", "#a5b563", "#c78e7d", "#7da899",
]

_SURROGATE = "#2d9c6f"
_TEACHER = "#d95f5f"
_BG = "#ffffff"
_TEXT = "#1a1a2e"
_MUTED = "#6b7280"


def generate_sankey(
    artifact_dir: Union[str, Path],
    output_path: Union[str, Path, None] = None,
    fmt: str = "html",
    top_k: int = 15,
    title: Optional[str] = None,
) -> str:
    """Generate a Sankey diagram of the TRACER routing policy.

    Parameters
    ----------
    artifact_dir : path to .tracer/ directory
    output_path : where to write the output; defaults to artifact_dir/sankey.{fmt}
    fmt : "html" for interactive HTML, "png" for static image, "svg" for vector
    top_k : number of top labels to show individually (rest grouped as "other")
    title : optional diagram title

    Returns
    -------
    str - path to the generated file

    Requires
    --------
    pip install tracer-llm[viz]
    """
    try:
        import plotly.graph_objects as _go  # noqa: F401 — ensure plotly available
    except ImportError:
        raise ImportError(
            "plotly is required for Sankey diagrams. "
            "Install with: pip install tracer-llm[viz]"
        ) from None

    artifact_dir = Path(artifact_dir)
    if output_path is None:
        output_path = artifact_dir / f"sankey.{fmt}"
    output_path = Path(output_path)

    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    qr_path = artifact_dir / "qualitative_report.json"
    if not qr_path.exists():
        raise FileNotFoundError(f"No qualitative report found at {qr_path}")
    qr = json.loads(qr_path.read_text())

    coverage = manifest.get("coverage_cal") or qr.get("coverage", 0)
    n_traces = manifest.get("n_traces", 0)
    pct_h = coverage * 100
    pct_d = (1 - coverage) * 100

    if title is None:
        title = (
            f"TRACER routing flow  -  "
            f"{pct_h:.0f}% surrogate / {pct_d:.0f}% LLM  -  "
            f"{n_traces:,} traces, "
            f"{len(manifest.get('label_space', []))} labels"
        )

    fig = _build_sankey_figure(manifest, qr, top_k, dark=False)
    if fig is None:
        raise RuntimeError("plotly is required")

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=15, color=_TEXT, family="Inter, system-ui, sans-serif"),
            x=0.01, y=0.97,
        ),
        width=1200,
        height=800,
        margin=dict(l=30, r=30, t=60, b=40),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "html":
        fig.write_html(str(output_path), include_plotlyjs="cdn")
    elif fmt in ("png", "svg", "pdf", "jpeg"):
        fig.write_image(str(output_path), scale=2)
    else:
        raise ValueError(f"Unsupported format '{fmt}'. Use html, png, svg, pdf, or jpeg.")

    return str(output_path)


def _build_sankey_figure(manifest: dict, qr: dict, top_k: int, dark: bool):
    """Shared figure builder used by both generate_sankey and generate_sankey_div."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    label_slices = sorted(
        [s for s in qr.get("slices", []) if s["slice_name"].startswith("label:")],
        key=lambda s: -s["count"],
    )

    coverage = manifest.get("coverage_cal") or qr.get("coverage", 0)
    n_traces = manifest.get("n_traces", 0)
    top = label_slices[:top_k]
    n_top = len(top)

    shown_handled = sum(s["count"] * s["handled_rate"] for s in top)
    shown_deferred = sum(s["count"] * s["deferred_rate"] for s in top)
    total_handled = n_traces * coverage
    total_deferred = n_traces * (1 - coverage)
    other_handled = max(0, total_handled - shown_handled)
    other_deferred = max(0, total_deferred - shown_deferred)
    n_other_labels = len(manifest.get("label_space", [])) - n_top
    has_other = (other_handled + other_deferred) > 0.5

    node_labels = [s["slice_name"].replace("label:", "").replace("_", " ") for s in top]
    if has_other:
        node_labels.append(f"other ({n_other_labels} labels)")
    surrogate_idx = len(node_labels)
    teacher_idx = surrogate_idx + 1
    pct_h = coverage * 100
    pct_d = (1 - coverage) * 100
    node_labels.append(f"Surrogate  ({pct_h:.0f}%)")
    node_labels.append(f"Teacher / LLM  ({pct_d:.0f}%)")

    if dark:
        _palette_dark = [
            "#58a6ff", "#3fb950", "#d29922", "#f0883e", "#bc8cff",
            "#2ea043", "#79c0ff", "#56d364", "#e3b341", "#ffa657",
            "#a5d6ff", "#7ee787", "#f0c42e", "#ffb86c", "#e8d5ff",
            "#4ac26b", "#388bfd", "#d4a72c", "#ff9a5c", "#c084fc",
        ]
        surrogate_color = "#238636"
        teacher_color   = "#b91c1c"
        node_border     = "rgba(255,255,255,0.08)"
        link_surr       = "rgba(35, 134, 54, 0.35)"
        link_teach      = "rgba(185, 28, 28, 0.40)"
        link_surr_other = "rgba(35, 134, 54, 0.25)"
        link_teach_other= "rgba(185, 28, 28, 0.30)"
        other_color     = "#484f58"
        bg              = "#161b22"
        text_color      = "#c9d1d9"
    else:
        _palette_dark   = _PALETTE
        surrogate_color = _SURROGATE
        teacher_color   = _TEACHER
        node_border     = "rgba(0,0,0,0.15)"
        link_surr       = "rgba(45, 156, 111, 0.25)"
        link_teach      = "rgba(217, 95, 95, 0.30)"
        link_surr_other = "rgba(45, 156, 111, 0.18)"
        link_teach_other= "rgba(217, 95, 95, 0.22)"
        other_color     = "#9ca3af"
        bg              = "#ffffff"
        text_color      = "#1a1a2e"

    node_colors = [_palette_dark[i % len(_palette_dark)] for i in range(n_top)]
    if has_other:
        node_colors.append(other_color)
    node_colors.extend([surrogate_color, teacher_color])

    sources, targets, values, link_colors = [], [], [], []
    for i, s in enumerate(top):
        h = s["count"] * s["handled_rate"]
        d = s["count"] * s["deferred_rate"]
        if h > 0:
            sources.append(i); targets.append(surrogate_idx)
            values.append(h); link_colors.append(link_surr)
        if d > 0:
            sources.append(i); targets.append(teacher_idx)
            values.append(d); link_colors.append(link_teach)
    if has_other:
        other_idx = n_top
        if other_handled > 0:
            sources.append(other_idx); targets.append(surrogate_idx)
            values.append(other_handled); link_colors.append(link_surr_other)
        if other_deferred > 0:
            sources.append(other_idx); targets.append(teacher_idx)
            values.append(other_deferred); link_colors.append(link_teach_other)

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=14, thickness=22,
            line=dict(color=node_border, width=0.5),
            label=node_labels, color=node_colors,
            hovertemplate="%{label}<extra></extra>",
        ),
        link=dict(
            source=sources, target=targets, value=values, color=link_colors,
            hovertemplate="%{source.label} -> %{target.label}<br>%{value:.0f} queries<extra></extra>",
        ),
    )])
    fig.update_layout(
        font=dict(size=11, color=text_color, family="Inter, system-ui, sans-serif"),
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def generate_sankey_div(
    artifact_dir: Union[str, Path],
    top_k: int = 15,
) -> str:
    """Return a self-contained HTML <div> with the Sankey chart embedded inline (dark theme).

    Suitable for injecting directly into the HTML report page.
    Requires plotly to be installed (pip install tracer-llm[viz]).
    Returns an empty string if plotly is not available or the report is missing.
    """
    try:
        from plotly.io import to_html
    except ImportError:
        return ""

    import json as _json
    artifact_dir = Path(artifact_dir)
    manifest_path = artifact_dir / "manifest.json"
    qr_path = artifact_dir / "qualitative_report.json"
    if not manifest_path.exists() or not qr_path.exists():
        return ""

    manifest = _json.loads(manifest_path.read_text())
    qr = _json.loads(qr_path.read_text())

    fig = _build_sankey_figure(manifest, qr, top_k, dark=True)
    if fig is None:
        return ""

    fig.update_layout(height=520)
    return to_html(fig, full_html=False, include_plotlyjs=False)
