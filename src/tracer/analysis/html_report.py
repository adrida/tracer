"""Generate a self-contained HTML audit report from a .tracer/ artifact directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union


_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0d1117; color: #c9d1d9; line-height: 1.6; }
.page { max-width: 1000px; margin: 0 auto; padding: 36px 24px 80px; }

/* ── header ── */
.top-bar { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
.logo { font-size: 1rem; font-weight: 700; letter-spacing: .12em;
        color: #58a6ff; text-transform: uppercase; }
h1 { font-size: 1.5rem; font-weight: 700; color: #f0f6fc; }
.subtitle { color: #8b949e; font-size: .875rem; margin-top: 3px; margin-bottom: 36px; }

/* ── stat cards ── */
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
         gap: 14px; margin-bottom: 36px; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 12px;
        padding: 20px 22px; position: relative; overflow: hidden; }
.card::before { content: ''; position: absolute; top: 0; left: 0; right: 0;
                height: 3px; border-radius: 12px 12px 0 0; }
.card.green::before { background: #238636; }
.card.blue::before  { background: #1f6feb; }
.card.purple::before { background: #8957e5; }
.card.yellow::before { background: #9e6a03; }
.card-label { font-size: .72rem; color: #8b949e; text-transform: uppercase;
              letter-spacing: .1em; font-weight: 600; }
.card-value { font-size: 2rem; font-weight: 800; color: #f0f6fc; margin: 4px 0 2px; }
.card.green .card-value { color: #3fb950; }
.card.blue  .card-value { color: #58a6ff; }
.card.purple .card-value { color: #bc8cff; }
.card.yellow .card-value { color: #d29922; }
.card-sub { font-size: .75rem; color: #8b949e; }

/* ── coverage ring ── */
.coverage-wrap { display: flex; align-items: center; gap: 32px; margin-bottom: 36px;
                 background: #161b22; border: 1px solid #30363d; border-radius: 12px;
                 padding: 24px 28px; }
.ring-svg { flex-shrink: 0; }
.ring-stats { flex: 1; }
.ring-stats h3 { font-size: .8rem; font-weight: 600; color: #8b949e; text-transform: uppercase;
                 letter-spacing: .08em; margin-bottom: 14px; }
.ring-row { display: flex; justify-content: space-between; align-items: center;
            padding: 6px 0; border-bottom: 1px solid #21262d; font-size: .875rem; }
.ring-row:last-child { border-bottom: none; }
.ring-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block;
            margin-right: 8px; flex-shrink: 0; }
.ring-row-label { display: flex; align-items: center; color: #c9d1d9; }
.ring-row-val { font-weight: 600; color: #f0f6fc; }

/* ── sections ── */
.section { background: #161b22; border: 1px solid #30363d; border-radius: 12px;
           padding: 22px 24px; margin-bottom: 20px; }
h2 { font-size: .82rem; font-weight: 600; color: #8b949e; text-transform: uppercase;
     letter-spacing: .1em; margin-bottom: 16px; }

/* ── label table ── */
table { width: 100%; border-collapse: collapse; font-size: .84rem; }
th { text-align: left; color: #8b949e; font-weight: 500; font-size: .75rem;
     text-transform: uppercase; letter-spacing: .07em; padding: 0 8px 10px 0;
     border-bottom: 1px solid #30363d; }
td { padding: 9px 8px 9px 0; border-bottom: 1px solid #1c2128; vertical-align: middle; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: #1c2128; }
.bar-bg { background: #21262d; border-radius: 3px; height: 6px; min-width: 80px; }
.bar-fill { height: 6px; border-radius: 3px; }
.bar-high { background: #238636; }
.bar-mid  { background: #9e6a03; }
.bar-low  { background: #b91c1c; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 20px;
         font-size: .72rem; font-weight: 600; }
.b-green  { background: #0f2e17; color: #3fb950; border: 1px solid #238636; }
.b-blue   { background: #0c1e3a; color: #58a6ff; border: 1px solid #1f6feb; }
.b-purple { background: #1d1135; color: #bc8cff; border: 1px solid #8957e5; }
.b-gray   { background: #21262d; color: #8b949e; border: 1px solid #30363d; }

/* ── filter ── */
.filter-row { display: flex; gap: 10px; margin-bottom: 14px; }
.search { background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
          padding: 6px 12px; color: #c9d1d9; font-size: .84rem; flex: 1;
          outline: none; }
.search:focus { border-color: #58a6ff; }
.filter-count { font-size: .78rem; color: #8b949e; align-self: center; white-space: nowrap; }

/* ── boundary pairs ── */
.pair { background: #0d1117; border-radius: 10px; padding: 14px 16px;
        margin-bottom: 10px; border: 1px solid #21262d; }
.pair-intent { font-size: .72rem; font-weight: 700; color: #8b949e;
               text-transform: uppercase; letter-spacing: .08em; margin-bottom: 10px; }
.pair-row { display: flex; gap: 10px; align-items: flex-start;
            margin-bottom: 6px; }
.pair-row:last-child { margin-bottom: 0; }
.pair-tag { font-size: .7rem; font-weight: 700; padding: 3px 9px; border-radius: 4px;
            white-space: nowrap; min-width: 72px; text-align: center; }
.pt-local    { background: #0f2e17; color: #3fb950; border: 1px solid #238636; }
.pt-deferred { background: #2d1516; color: #f85149; border: 1px solid #b91c1c; }
.pair-text { font-size: .84rem; color: #c9d1d9; line-height: 1.5; flex: 1; }
.pair-score { font-size: .75rem; color: #8b949e; margin-left: 6px; white-space: nowrap; }

/* ── examples ── */
.ex-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.ex-col h3 { font-size: .75rem; color: #8b949e; text-transform: uppercase;
             letter-spacing: .08em; margin-bottom: 10px; }
.ex-item { background: #0d1117; border: 1px solid #21262d; border-radius: 8px;
           padding: 12px 14px; margin-bottom: 8px; }
.ex-item.handled { border-left: 3px solid #238636; }
.ex-item.deferred { border-left: 3px solid #b91c1c; }
.ex-text { font-size: .84rem; color: #c9d1d9; margin-bottom: 6px; line-height: 1.4; }
.ex-meta { display: flex; gap: 8px; flex-wrap: wrap; }
.ex-label { font-size: .72rem; background: #21262d; border-radius: 4px;
            padding: 2px 7px; color: #8b949e; }
.ex-score { font-size: .72rem; color: #8b949e; }

/* ── temporal deltas ── */
.delta-pos { color: #3fb950; font-weight: 600; }
.delta-neg { color: #f85149; font-weight: 600; }
.delta-zero { color: #8b949e; }
.delta-bar-bg { background: #21262d; border-radius: 3px; height: 4px; width: 60px;
               display: inline-block; vertical-align: middle; }
.delta-bar-fill { height: 4px; border-radius: 3px; background: #238636; }

/* ── footer ── */
.footer { margin-top: 48px; text-align: center; font-size: .78rem; color: #484f58; }
.footer a { color: #58a6ff; text-decoration: none; }
.footer a:hover { text-decoration: underline; }
"""

_JS = """
document.addEventListener('DOMContentLoaded', function() {
    var input = document.getElementById('label-search');
    if (!input) return;
    input.addEventListener('input', function() {
        var q = this.value.toLowerCase();
        var rows = document.querySelectorAll('#label-table tbody tr');
        var visible = 0;
        rows.forEach(function(row) {
            var txt = row.textContent.toLowerCase();
            var show = txt.includes(q);
            row.style.display = show ? '' : 'none';
            if (show) visible++;
        });
        document.getElementById('label-count').textContent =
            visible + ' of ' + rows.length + ' labels';
    });
});
"""

_METHOD_BADGE = {"global": "b-green", "l2d": "b-blue", "rsb": "b-purple"}


def _pct(v) -> str:
    return f"{v:.1%}" if v is not None else "-"


def _bar_html(rate: float) -> str:
    pct = int(rate * 100)
    cls = "bar-high" if rate >= 0.85 else "bar-mid" if rate >= 0.60 else "bar-low"
    return (f'<div class="bar-bg">'
            f'<div class="{cls} bar-fill" style="width:{pct}%"></div></div>')


def _score_str(s) -> str:
    return f'<span class="pair-score">score {s:.2f}</span>' if s is not None else ""


def generate_html_report(
    artifact_dir: Union[str, Path],
    output_path: Union[str, Path, None] = None,
) -> str:
    """Generate a self-contained HTML audit report.

    Parameters
    ----------
    artifact_dir : path to .tracer/ directory
    output_path  : where to write the HTML file; defaults to artifact_dir/report.html

    Returns
    -------
    str path to the generated HTML file
    """
    artifact_dir = Path(artifact_dir)
    if output_path is None:
        output_path = artifact_dir / "report.html"
    output_path = Path(output_path)

    manifest_raw = json.loads((artifact_dir / "manifest.json").read_text())
    qr_path = artifact_dir / "qualitative_report.json"
    qr = json.loads(qr_path.read_text()) if qr_path.exists() else None

    method      = manifest_raw.get("selected_method") or "none"
    coverage    = manifest_raw.get("coverage_cal")
    ta          = manifest_raw.get("teacher_agreement_cal")
    n_traces    = manifest_raw.get("n_traces", 0)
    n_labels    = len(manifest_raw.get("label_space", []))
    emb_dim     = manifest_raw.get("embedding_dim")
    target_ta   = manifest_raw.get("target_teacher_agreement", 0.90)
    method_cls  = _METHOD_BADGE.get(method, "b-gray")

    cov_exact = (coverage or 0) * 100
    defer_exact = 100 - cov_exact

    # ── Coverage ring (SVG donut) ────────────────────────────────────────────
    # stroke-dasharray = [green_length, gap]; rotate -90 so arc starts at 12 o'clock
    import math
    _cov = coverage or 0
    r, cx, cy, sw = 42, 56, 56, 24
    circumf = 2 * math.pi * r
    green_len = circumf * _cov      # handled arc
    gap_len   = circumf - green_len  # rest = deferred

    ring_svg = f"""
<svg class="ring-svg" width="112" height="112" viewBox="0 0 112 112">
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
          stroke="#b91c1c" stroke-width="{sw}"/>
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
          stroke="#238636" stroke-width="{sw}"
          stroke-dasharray="{green_len:.2f} {gap_len:.2f}"
          transform="rotate(-90 {cx} {cy})"/>
  <text x="{cx}" y="{cy - 6}" text-anchor="middle" fill="#f0f6fc"
        font-size="15" font-weight="800">{cov_exact:.1f}%</text>
  <text x="{cx}" y="{cy + 12}" text-anchor="middle" fill="#8b949e"
        font-size="9" letter-spacing="1">surrogate</text>
</svg>"""

    # ── Header ───────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TRACER - Audit Report</title>
<style>{_CSS}</style>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
<div class="page">

<div class="top-bar">
  <span class="logo">TRACER</span>
  <span style="color:#30363d">|</span>
  <span style="font-size:.875rem;color:#8b949e">Audit Report</span>
</div>
<h1>Routing Policy</h1>
<div class="subtitle">
  {n_traces:,} traces &nbsp;·&nbsp; {n_labels} labels
  &nbsp;·&nbsp; {emb_dim}-dim embeddings
  &nbsp;·&nbsp; target TA = {target_ta:.0%}
</div>

<div class="cards">
  <div class="card green">
    <div class="card-label">Coverage</div>
    <div class="card-value">{_pct(coverage)}</div>
    <div class="card-sub">handled by surrogate</div>
  </div>
  <div class="card blue">
    <div class="card-label">Teacher Agreement</div>
    <div class="card-value">{_pct(ta)}</div>
    <div class="card-sub">on handled traffic</div>
  </div>
  <div class="card purple">
    <div class="card-label">Method</div>
    <div class="card-value" style="font-size:1.2rem;margin-top:10px">
      <span class="badge {method_cls}">{method.upper()}</span>
    </div>
    <div class="card-sub">selected pipeline</div>
  </div>
  <div class="card yellow">
    <div class="card-label">LLM Calls Saved</div>
    <div class="card-value">{_pct(coverage)}</div>
    <div class="card-sub">vs 100% LLM baseline</div>
  </div>
</div>

<div class="coverage-wrap">
  {ring_svg}
  <div class="ring-stats">
    <h3>Traffic Split</h3>
    <div class="ring-row">
      <div class="ring-row-label">
        <span class="ring-dot" style="background:#238636"></span>Handled by surrogate
      </div>
      <div class="ring-row-val" style="color:#3fb950">{_pct(coverage)}</div>
    </div>
    <div class="ring-row">
      <div class="ring-row-label">
        <span class="ring-dot" style="background:#b91c1c"></span>Deferred to teacher
      </div>
      <div class="ring-row-val" style="color:#f85149">{defer_exact:.1f}%</div>
    </div>
    <div class="ring-row">
      <div class="ring-row-label">
        <span class="ring-dot" style="background:#58a6ff"></span>Teacher agreement (handled)
      </div>
      <div class="ring-row-val">{_pct(ta)}</div>
    </div>
    <div class="ring-row">
      <div class="ring-row-label">
        <span class="ring-dot" style="background:#8b949e"></span>Labels
      </div>
      <div class="ring-row-val">{n_labels}</div>
    </div>
  </div>
</div>
"""

    if qr is None:
        html += "<p style='color:#8b949e'>No qualitative report in this artifact directory.</p>"
    else:
        slices      = qr.get("slices", [])
        label_slices  = [s for s in slices if s["slice_name"].startswith("label:")]
        length_slices = [s for s in slices if s["slice_name"].startswith("length:")]
        pairs       = qr.get("boundary_pairs", [])
        handled_ex  = qr.get("handled_examples", [])
        deferred_ex = qr.get("deferred_examples", [])
        deltas      = qr.get("temporal_deltas", [])

        # ── Sankey routing diagram (top of content) ───────────────────────────
        try:
            from tracer.analysis.sankey import generate_sankey_div
            sankey_div = generate_sankey_div(artifact_dir)
            if sankey_div:
                html += f"""
<div class="section">
  <h2>Routing Flow</h2>
  <p style="font-size:.82rem;color:#8b949e;margin-bottom:12px">
    Green flows are handled by the surrogate. Red flows are deferred to the teacher LLM.
    Drag nodes to rearrange &middot; hover for exact counts.
  </p>
  {sankey_div}
</div>
"""
        except Exception:
            pass

        # ── Per-label coverage table (searchable) ─────────────────────────────
        html += f"""
<div class="section">
  <h2>Per-Label Coverage ({len(label_slices)} labels)</h2>
  <div class="filter-row">
    <input id="label-search" class="search" placeholder="Search labels…" type="text">
    <span class="filter-count" id="label-count">{len(label_slices)} of {len(label_slices)} labels</span>
  </div>
  <table id="label-table">
    <thead>
      <tr>
        <th>Label</th>
        <th>Coverage</th>
        <th style="width:130px"></th>
        <th>Count</th>
        <th>TA (handled)</th>
      </tr>
    </thead>
    <tbody>
"""
        for s in sorted(label_slices, key=lambda x: -x["handled_rate"]):
            label = s["slice_name"].replace("label:", "")
            hr_v  = s["handled_rate"]
            ta_s  = s.get("teacher_agreement_handled")
            ta_str = f"{ta_s:.1%}" if ta_s is not None else "-"
            html += (
                f'<tr><td><code style="color:#8b949e;font-size:.82rem">{label}</code></td>'
                f'<td><b style="color:#f0f6fc">{hr_v:.1%}</b></td>'
                f'<td>{_bar_html(hr_v)}</td>'
                f'<td style="color:#8b949e">{s["count"]}</td>'
                f'<td style="color:#8b949e">{ta_str}</td></tr>\n'
            )
        html += "    </tbody>\n  </table>\n</div>\n"

        # ── Coverage by length ────────────────────────────────────────────────
        if length_slices:
            html += '<div class="section">\n<h2>Coverage by Query Length</h2>\n<table>\n'
            html += '<tr><th>Bucket</th><th>Coverage</th><th style="width:130px"></th><th>Count</th></tr>\n'
            for s in length_slices:
                html += (
                    f'<tr><td>{s["slice_name"]}</td>'
                    f'<td><b style="color:#f0f6fc">{s["handled_rate"]:.1%}</b></td>'
                    f'<td>{_bar_html(s["handled_rate"])}</td>'
                    f'<td style="color:#8b949e">{s["count"]}</td></tr>\n'
                )
            html += '</table>\n</div>\n'

        # ── Boundary pairs ────────────────────────────────────────────────────
        if pairs:
            html += f'<div class="section">\n<h2>Contrastive Boundary Pairs ({len(pairs)})</h2>\n'
            html += '<p style="font-size:.82rem;color:#8b949e;margin-bottom:14px">Same label, opposite routing decision - shows what makes a query \'easy\' vs \'hard\'.</p>\n'
            for p in pairs[:8]:
                hs = _score_str(p.get("handled_score"))
                ds = _score_str(p.get("deferred_score"))
                html += f"""<div class="pair">
  <div class="pair-intent">{p["teacher_label"]}</div>
  <div class="pair-row">
    <span class="pair-tag pt-local">SURROGATE</span>
    <span class="pair-text">{p["handled_preview"]}</span>{hs}
  </div>
  <div class="pair-row">
    <span class="pair-tag pt-deferred">→ LLM</span>
    <span class="pair-text">{p["deferred_preview"]}</span>{ds}
  </div>
</div>\n"""
            html += '</div>\n'

        # ── Representative examples ───────────────────────────────────────────
        if handled_ex or deferred_ex:
            html += '<div class="section">\n<h2>Representative Examples</h2>\n<div class="ex-grid">\n'

            html += '<div class="ex-col">\n<h3>Handled by surrogate</h3>\n'
            for ex in handled_ex[:6]:
                score_str = f'<span class="ex-score">score {ex["accept_score"]:.2f}</span>' if ex.get("accept_score") else ""
                html += (f'<div class="ex-item handled">'
                         f'<div class="ex-text">{ex["input_preview"]}</div>'
                         f'<div class="ex-meta">'
                         f'<span class="ex-label">{ex["teacher_label"]}</span>'
                         f'{score_str}</div></div>\n')
            html += '</div>\n'

            html += '<div class="ex-col">\n<h3>Deferred to teacher</h3>\n'
            for ex in deferred_ex[:6]:
                html += (f'<div class="ex-item deferred">'
                         f'<div class="ex-text">{ex["input_preview"]}</div>'
                         f'<div class="ex-meta">'
                         f'<span class="ex-label">{ex["teacher_label"]}</span>'
                         f'</div></div>\n')
            html += '</div>\n'

            html += '</div>\n</div>\n'

        # ── Temporal deltas ───────────────────────────────────────────────────
        if deltas:
            html += '<div class="section">\n<h2>Temporal Deltas (largest changes this refit)</h2>\n<table>\n'
            html += '<tr><th>Label</th><th>Previous</th><th>Current</th><th>Change</th></tr>\n'
            for d in deltas:
                sign  = "+" if d["delta"] >= 0 else ""
                dcls  = "delta-pos" if d["delta"] > 0.01 else "delta-neg" if d["delta"] < -0.01 else "delta-zero"
                prev_bar = int(d["previous_handled_rate"] * 60)
                cur_bar  = int(d["current_handled_rate"] * 60)
                html += (
                    f'<tr><td><code style="color:#8b949e;font-size:.82rem">{d["label"]}</code></td>'
                    f'<td style="color:#8b949e">{d["previous_handled_rate"]:.1%}'
                    f'<div class="delta-bar-bg"><div class="delta-bar-fill" style="width:{prev_bar}px"></div></div></td>'
                    f'<td style="color:#f0f6fc">{d["current_handled_rate"]:.1%}'
                    f'<div class="delta-bar-bg"><div class="delta-bar-fill" style="width:{cur_bar}px"></div></div></td>'
                    f'<td class="{dcls}">{sign}{d["delta"]:.1%}</td></tr>\n'
                )
            html += '</table>\n</div>\n'

    html += f"""
<div class="footer">
  Generated by <a href="https://github.com/adrida/tracer">tracer-llm</a>
</div>

</div>
<script>{_JS}</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")

    # Always generate the standalone sankey.html alongside the report
    try:
        from tracer.analysis.sankey import generate_sankey
        generate_sankey(artifact_dir, output_path=artifact_dir / "sankey.html", fmt="html")
    except Exception:
        pass

    return str(output_path)
