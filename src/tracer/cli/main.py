"""TRACER CLI.

    tracer demo
    tracer fit traces.jsonl [--artifact-dir .tracer] [--target 0.90]
    tracer report [.tracer]
    tracer report-html [.tracer]
    tracer update new_traces.jsonl [--artifact-dir .tracer]
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap


# ── helpers ───────────────────────────────────────────────────────────────────

def _cmd_fit(args):
    from tracer.api import fit
    from tracer.config import FitConfig
    from tracer.cli._ui import header, section, step, success, warn, stat, bar_line, hr, _C

    c = _C()
    header("fit", f"traces={args.traces}  target-TA={args.target:.0%}")

    config = FitConfig(target_teacher_agreement=args.target)

    with step("Fitting routing policy (model zoo + calibration)"):
        result = fit(args.traces, artifact_dir=args.artifact_dir, config=config)

    _print_fit_result(result)


def _cmd_report(args):
    from tracer.api import report
    from tracer.cli._ui import _C, header, stat, section

    c = _C()
    m = report(args.artifact_dir)
    header("report", args.artifact_dir)
    print(json.dumps({
        "version": m.version,
        "n_traces": m.n_traces,
        "label_space": m.label_space,
        "method": m.selected_method,
        "target_ta": m.target_teacher_agreement,
        "coverage": m.coverage_cal,
        "teacher_agreement": m.teacher_agreement_cal,
        "embedding_dim": m.embedding_dim,
        "n_retrains": m.n_retrains,
    }, indent=2))


def _cmd_report_html(args):
    from tracer.analysis.html_report import generate_html_report
    from tracer.cli._ui import success, info, step, _C
    import webbrowser

    c = _C()
    with step("Generating HTML report"):
        out = generate_html_report(args.artifact_dir, output_path=args.output)

    success(f"HTML report saved  →  {c.CYAN}{out}{c.RESET}")
    if not args.no_open:
        info("Opening in browser...")
        webbrowser.open(f"file://{out}")


def _cmd_serve(args):
    from tracer.runtime.serve import serve
    serve(artifact_dir=args.artifact_dir, host=args.host, port=args.port)


def _cmd_update(args):
    from tracer.api import update
    from tracer.cli._ui import header, step

    header("update", f"new traces={args.traces}")

    with step("Refitting with new traces (continual learning)"):
        result = update(args.traces, artifact_dir=args.artifact_dir)

    _print_fit_result(result)


def _cmd_demo(args):
    """Run a zero-setup demo. Uses Banking77 data if available, synthetic otherwise."""
    import shutil
    from pathlib import Path

    import numpy as np

    from tracer.api import fit, load_router
    from tracer.config import FitConfig
    from tracer.analysis.html_report import generate_html_report
    from tracer.cli._ui import (
        _C, header, section, stat, bar_line, pair_block,
        route_line, cost_table, hr, success, info, warn, step,
    )

    c = _C()

    # ── Detect Banking77 data (already on disk in the repo) ───────────────────
    # Check relative to this file (tracer/src/tracer/cli/) → tracer/notebooks/data/
    _pkg_root = Path(__file__).resolve().parent.parent.parent.parent
    _b77_traces = _pkg_root / "notebooks" / "data" / "banking77_traces.jsonl"
    _b77_emb    = _pkg_root / "notebooks" / "data" / "banking77_traces.npy"
    # Also check from cwd (running from repo root)
    if not _b77_traces.exists():
        _b77_traces = Path.cwd() / "tracer" / "notebooks" / "data" / "banking77_traces.jsonl"
        _b77_emb    = Path.cwd() / "tracer" / "notebooks" / "data" / "banking77_traces.npy"

    use_banking77 = _b77_traces.exists() and _b77_emb.exists()

    # ── Build dataset ─────────────────────────────────────────────────────────
    demo_out_dir = Path.cwd() / "tracer-demo-output"
    shutil.rmtree(demo_out_dir, ignore_errors=True)
    demo_out_dir.mkdir(parents=True)
    traces_path  = demo_out_dir / "traces.jsonl"
    emb_path     = demo_out_dir / "embeddings.npy"
    artifact_dir = demo_out_dir / ".tracer"

    if use_banking77:
        # ── Banking77: real 77-class bank intent data (already downloaded) ────
        # Subsample to 1500 for demo speed; still representative
        DEMO_N = 1_500
        header(
            "Demo  -  Banking77 Intent Classifier  (real data)",
            f"77 intents · {DEMO_N:,} traces · 1024-dim BGE-M3 embeddings",
        )
        info(f"Found Banking77 data at {_b77_traces.parent}")
        print()

        # Load and subsample
        import random as _rnd
        _rnd.seed(42)
        all_lines = _b77_traces.read_text(encoding="utf-8").splitlines()
        sampled_lines = _rnd.sample(all_lines, min(DEMO_N, len(all_lines)))

        # Write sampled traces
        with traces_path.open("w") as f:
            for line in sampled_lines:
                f.write(line + "\n")

        # Load full embeddings, extract matching rows by position in sampled lines
        _all_indices = {line: i for i, line in enumerate(all_lines)}
        row_indices = [_all_indices[line] for line in sampled_lines]
        X_full = np.load(_b77_emb, mmap_mode="r")
        X = X_full[row_indices].astype(np.float32)
        np.save(emb_path, X)

        # Build label-to-embedding index for live routing
        import json as _json
        records = [_json.loads(l) for l in sampled_lines]
        teacher_list = [r["teacher"] for r in records]
        label_to_idx: dict[str, list[int]] = {}
        for i, lbl in enumerate(teacher_list):
            label_to_idx.setdefault(lbl, []).append(i)

        # Pick 5 real Banking77 queries from the sampled data (diverse intents)
        _target_labels = [
            "card_arrival", "exchange_rate", "transfer_not_received_by_bank",
            "card_payment_not_recognised", "change_email",
        ]
        _all_labels_present = set(teacher_list)
        # Fall back to first 5 available labels if any aren't in this subsample
        _fallback_labels = [l for l in sorted(_all_labels_present)
                            if l not in _target_labels]
        _demo_label_list = []
        for lbl in _target_labels:
            if lbl in _all_labels_present:
                _demo_label_list.append(lbl)
            elif _fallback_labels:
                _demo_label_list.append(_fallback_labels.pop(0))
        # Build text lookup from raw records
        _label_text = {}
        for rec in records:
            lbl = rec["teacher"]
            if lbl not in _label_text:
                _label_text[lbl] = rec.get("input", rec.get("text", ""))
        demo_queries = [(
            _label_text.get(lbl, lbl),   # actual text from the dataset
            lbl,
        ) for lbl in _demo_label_list[:5]]

        def _pick_emb(preferred_label):
            idxs = label_to_idx.get(preferred_label)
            if idxs:
                return X[idxs[0]], preferred_label
            # fallback: first available
            first = next(iter(label_to_idx))
            return X[label_to_idx[first][0]], first

        config = FitConfig(target_teacher_agreement=0.95,
                           frontier_targets=(0.90, 0.95))
        model_zoo_desc = "logreg · rf · et"
        use_fast_zoo   = True

    else:
        # ── Synthetic fallback: clean 5-class data ─────────────────────────────
        N, DIM, N_CLS = 2_000, 128, 5
        header(
            "Demo  -  Synthetic Banking Intent Classifier",
            f"5 intents · {N:,} traces · {DIM}-dim embeddings  "
            f"(tip: put Banking77 data in tracer/notebooks/data/ for a real demo)",
        )
        rng = np.random.RandomState(42)
        intent_names = ["check_balance", "transfer_money", "card_blocked",
                        "loan_inquiry", "account_settings"]
        centers    = rng.randn(N_CLS, DIM) * 4.0          # well separated
        labels_int = rng.randint(0, N_CLS, size=N)
        X          = centers[labels_int] + rng.randn(N, DIM) * 0.8
        teacher_list = [intent_names[i] for i in labels_int]
        ground_truth = list(teacher_list)
        for i in range(N):
            if rng.random() < 0.05:                        # 5% noise only
                teacher_list[i] = intent_names[rng.randint(0, N_CLS)]

        sample_texts = {
            "check_balance":    ["What is my current account balance?",
                                 "How much money do I have?",
                                 "Can you check my account balance?"],
            "transfer_money":   ["I want to send money to my friend",
                                 "Transfer $500 to savings",
                                 "How do I wire money abroad?"],
            "card_blocked":     ["My card is not working",
                                 "Why was my card blocked?",
                                 "My card got declined at the store"],
            "loan_inquiry":     ["What are your loan rates?",
                                 "I'd like to apply for a personal loan",
                                 "How much can I borrow?"],
            "account_settings": ["Change my email address",
                                 "How do I reset my password?",
                                 "Update my mailing address"],
        }
        with traces_path.open("w") as f:
            for i in range(N):
                intent  = ground_truth[i]
                texts   = sample_texts[intent]
                text    = texts[i % len(texts)]
                if i >= len(texts):
                    text = f"{text} (query #{i})"
                f.write(json.dumps({"input": text, "teacher": teacher_list[i],
                                    "ground_truth": ground_truth[i],
                                    "id": str(i)}) + "\n")
        np.save(emb_path, X.astype(np.float32))

        # Build label index for live routing
        label_to_idx = {}
        for i, lbl in enumerate(teacher_list):
            label_to_idx.setdefault(lbl, []).append(i)

        demo_queries = [
            ("What is my current account balance?", "check_balance"),
            ("I want to send money to my friend",   "transfer_money"),
            ("My card is not working",               "card_blocked"),
            ("What are your loan rates?",            "loan_inquiry"),
            ("Change my email address",              "account_settings"),
        ]

        def _pick_emb(preferred_label):
            idxs = label_to_idx.get(preferred_label, [0])
            return X[idxs[0]].astype(np.float32), preferred_label

        config = FitConfig(target_teacher_agreement=0.90,
                           frontier_targets=(0.85, 0.90))
        model_zoo_desc = "logreg · mlp · rf · et"
        use_fast_zoo   = False

    # ── Fit ───────────────────────────────────────────────────────────────────
    surrogate_log: list[tuple[str, float]] = []

    from tracer.fit import surrogate as _s_mod
    from tracer.fit import pipeline as _p_mod
    _orig_candidates = _s_mod._candidates

    if use_fast_zoo:
        # Banking77: logreg + trees (fast on high-dim), skip MLP (slow)
        def _demo_candidates(n_samples: int) -> dict:
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            SEED = 42
            return {
                "logreg_c1":  lambda: Pipeline([("scale", StandardScaler()),
                    ("clf", LogisticRegression(C=1.0, max_iter=1000,
                                               random_state=SEED))]),
                "logreg_c10": lambda: Pipeline([("scale", StandardScaler()),
                    ("clf", LogisticRegression(C=10.0, max_iter=1000,
                                               random_state=SEED))]),
                "rf":  lambda: RandomForestClassifier(n_estimators=150, n_jobs=-1,
                                                      random_state=SEED),
                "et":  lambda: ExtraTreesClassifier(n_estimators=150, n_jobs=-1,
                                                     random_state=SEED),
            }
    else:
        # Synthetic: add tree models
        def _demo_candidates(n_samples: int) -> dict:
            from sklearn.linear_model import LogisticRegression
            from sklearn.neural_network import MLPClassifier
            from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            SEED = 42
            return {
                "logreg_c1":  lambda: Pipeline([("scale", StandardScaler()),
                    ("clf", LogisticRegression(C=1.0, max_iter=500,
                                               random_state=SEED))]),
                "logreg_c10": lambda: Pipeline([("scale", StandardScaler()),
                    ("clf", LogisticRegression(C=10.0, max_iter=500,
                                               random_state=SEED))]),
                "mlp_1h":     lambda: Pipeline([("scale", StandardScaler()),
                    ("clf", MLPClassifier(hidden_layer_sizes=(256,), max_iter=100,
                                          early_stopping=True, random_state=SEED))]),
                "rf":         lambda: RandomForestClassifier(n_estimators=100,
                                          n_jobs=-1, random_state=SEED),
                "et":         lambda: ExtraTreesClassifier(n_estimators=100,
                                          n_jobs=-1, random_state=SEED),
            }

    _orig_search = _p_mod.search_best_surrogate

    def _patched_search(X_tr, y_tr, X_val, y_val, on_candidate=None):
        return _orig_search(X_tr, y_tr, X_val, y_val,
                            on_candidate=on_candidate or (
                                lambda n, f: surrogate_log.append((n, f))))

    _s_mod._candidates = _demo_candidates
    _p_mod.search_best_surrogate = _patched_search

    try:
        with step(f"Training model zoo  ({model_zoo_desc})"):
            result = fit(traces_path, artifact_dir=artifact_dir,
                         embeddings=X.astype(np.float32), config=config)
    finally:
        _s_mod._candidates = _orig_candidates
        _p_mod.search_best_surrogate = _orig_search

    # ── Results ───────────────────────────────────────────────────────────────
    m  = result.manifest
    qr = result.qualitative_report

    section("Model Selection")
    if surrogate_log:
        # Deduplicate: each model may appear multiple times (once per pipeline family)
        seen: dict[str, float] = {}
        for name, f1 in surrogate_log:
            if name not in seen or f1 > seen[name]:
                seen[name] = f1
        best_name = max(seen, key=lambda k: seen[k])
        for name, f1 in seen.items():
            marker = f"  {c.GREEN}← selected{c.RESET}" if name == best_name else ""
            print(f"    {c.DIM}{name:<18}{c.RESET}  val F1 = {c.BOLD}{f1:.4f}{c.RESET}{marker}")

    section("Routing Policy")
    method_color = {"global": "GREEN", "l2d": "CYAN", "rsb": "MAGENTA"}.get(
        m.selected_method or "", "WHITE")
    stat("method",     m.selected_method or "none (parity gate blocked)", color=method_color.lower())
    stat("coverage",   f"{m.coverage_cal:.1%}" if m.coverage_cal else "-",
         note="of traffic handled by surrogate", color="green")
    stat("teacher TA", f"{m.teacher_agreement_cal:.3f}" if m.teacher_agreement_cal else "-",
         note="surrogate matches teacher on handled traffic", color="cyan")
    stat("traces",     f"{m.n_traces:,}")
    stat("labels",     str(len(m.label_space)))

    for note in result.notes:
        if note.startswith("Deployed") or note.startswith("Loaded"):
            info(note)
        else:
            warn(note)

    if qr:
        section("Per-Label Coverage  (top 15)")
        label_slices = [s for s in qr.slices if s.slice_name.startswith("label:")]
        for s in sorted(label_slices, key=lambda x: -x.handled_rate)[:15]:
            bar_line(s.slice_name.replace("label:", ""), s.handled_rate, s.count)

        if qr.boundary_pairs:
            section(f"Boundary Pairs  ({len(qr.boundary_pairs)} examples - same label, different routing)")
            print()
            for bp in qr.boundary_pairs[:4]:
                pair_block(bp.teacher_label, bp.handled_preview, bp.deferred_preview,
                           bp.handled_score, bp.deferred_score)

    # ── Live routing ──────────────────────────────────────────────────────────
    # Pull real examples from the qualitative report: 3 handled + 2 deferred.
    # This guarantees a realistic mix instead of cherry-picking embeddings.
    if m.selected_method and qr:
        section("Live Routing  (sample queries from the training set)")
        handled_ex = [e for e in qr.handled_examples if e.local_label][:3]
        deferred_ex = qr.deferred_examples[:2]
        for ex in handled_ex + deferred_ex:
            route_line(ex.input_preview[:70], ex.decision,
                       ex.local_label or "", ex.accept_score)

    # ── Cost ──────────────────────────────────────────────────────────────────
    if m.coverage_cal:
        section("Cost Projection  (10 k queries / day)")
        cost_table(m.coverage_cal)

    # ── HTML ──────────────────────────────────────────────────────────────────
    html_path = generate_html_report(artifact_dir)

    # ── Output paths ──────────────────────────────────────────────────────────
    section("Demo Outputs Saved")
    success(f"Traces          →  {c.CYAN}{traces_path}{c.RESET}")
    success(f"Embeddings      →  {c.CYAN}{emb_path}{c.RESET}")
    success(f"Artifacts       →  {c.CYAN}{artifact_dir}{c.RESET}")
    success(f"HTML report     →  {c.CYAN}{html_path}{c.RESET}")

    print()
    print(f"  {c.DIM}{hr()}{c.RESET}")
    print(f"  {c.BOLD}Next steps{c.RESET}")
    print(f"  {c.DIM}1.{c.RESET}  Open the HTML report:")
    print(f"       {c.CYAN}open {html_path}{c.RESET}")
    print(f"  {c.DIM}2.{c.RESET}  Fit on your own traces:")
    print(f"       {c.CYAN}tracer fit traces.jsonl --target 0.90{c.RESET}")
    print(f"  {c.DIM}3.{c.RESET}  Route with text directly (no manual embedding):")
    print(f"       {c.CYAN}from tracer import Embedder{c.RESET}")
    print(f"       {c.CYAN}embedder = Embedder.from_sentence_transformers('BAAI/bge-small-en-v1.5'){c.RESET}")
    print(f"       {c.CYAN}router = tracer.load_router('.tracer', embedder=embedder){c.RESET}")
    print(f"       {c.CYAN}router.predict('What is my balance?'){c.RESET}")
    print(f"  {c.DIM}4.{c.RESET}  Docs: {c.CYAN}https://github.com/adrida/tracer{c.RESET}")
    print()


def _print_fit_result(result):
    """Colored summary of a FitResult -- used by fit and update commands."""
    from tracer.cli._ui import (
        _C, section, stat, bar_line, pair_block, success, warn, info, hr,
    )
    from tracer.analysis.html_report import generate_html_report

    c  = _C()
    m  = result.manifest
    qr = result.qualitative_report

    section("Routing Policy")
    method_color = {"global": "green", "l2d": "cyan", "rsb": "magenta"}.get(
        m.selected_method or "", "white")
    stat("method",     m.selected_method or "none (parity gate blocked)",
         color=method_color)
    if m.coverage_cal is not None:
        stat("coverage",   f"{m.coverage_cal:.1%}", "handled by surrogate",   color="green")
        stat("teacher TA", f"{m.teacher_agreement_cal:.3f}",
             "agreement on handled traffic", color="cyan")
    stat("traces",     f"{m.n_traces:,}")
    stat("labels",     str(len(m.label_space)))
    stat("artifacts",  str(result.artifact_dir))

    for note in result.notes:
        if note.startswith("Deployed") or note.startswith("Loaded"):
            info(note)
        else:
            warn(note)

    if qr:
        section("Per-Label Coverage")
        label_slices = [s for s in qr.slices if s.slice_name.startswith("label:")]
        for s in sorted(label_slices, key=lambda x: -x.handled_rate)[:15]:
            bar_line(s.slice_name.replace("label:", ""), s.handled_rate, s.count)

        if qr.boundary_pairs:
            print()
            print(f"  {c.BOLD}{c.YELLOW}Boundary Pairs{c.RESET}  {c.DIM}({len(qr.boundary_pairs)} examples){c.RESET}")
            print(f"  {hr('·')}")
            for bp in qr.boundary_pairs[:3]:
                pair_block(bp.teacher_label, bp.handled_preview, bp.deferred_preview,
                           bp.handled_score, bp.deferred_score)

    # Generate HTML automatically
    try:
        html = generate_html_report(result.artifact_dir)
        print()
        success(f"HTML report  →  {c.CYAN}{html}{c.RESET}")
        info(f"open {html}")
    except Exception:
        pass

    print()


def main():
    parser = argparse.ArgumentParser(
        prog="tracer",
        description="TRACER - Turn LLM traces into cost-efficient routing policies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              tracer demo                              Run a zero-setup demo
              tracer fit traces.jsonl                  Fit a routing policy
              tracer fit traces.jsonl --target 0.95    95%% parity target
              tracer report .tracer                    Show policy manifest
              tracer report-html .tracer               Open HTML audit report
              tracer update new.jsonl                  Refit with new traces
        """),
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("demo", help="Zero-setup demo with synthetic data")

    p_fit = sub.add_parser("fit", help="Fit a routing policy from traces")
    p_fit.add_argument("traces", help="Path to traces JSONL file")
    p_fit.add_argument("--artifact-dir", default=".tracer",
                       help="Output artifact directory (default: .tracer)")
    p_fit.add_argument("--target", type=float, default=0.90,
                       help="Target teacher agreement, e.g. 0.90 (default)")

    p_report = sub.add_parser("report", help="Show artifact manifest as JSON")
    p_report.add_argument("artifact_dir", nargs="?", default=".tracer")

    p_html = sub.add_parser("report-html",
                             help="Generate HTML audit report and open in browser")
    p_html.add_argument("artifact_dir", nargs="?", default=".tracer")
    p_html.add_argument("--output", default=None,
                        help="Output path (default: <artifact_dir>/report.html)")
    p_html.add_argument("--no-open", action="store_true",
                        help="Don't open browser automatically")

    p_serve = sub.add_parser("serve",
                              help="Start a prediction HTTP server")
    p_serve.add_argument("artifact_dir", nargs="?", default=".tracer",
                         help="Artifact directory (default: .tracer)")
    p_serve.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    p_serve.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")

    p_update = sub.add_parser("update",
                               help="Refit with new traces (continual learning)")
    p_update.add_argument("traces", help="Path to new traces JSONL file")
    p_update.add_argument("--artifact-dir", default=".tracer")

    args = parser.parse_args()

    dispatch = {
        "fit":         _cmd_fit,
        "report":      _cmd_report,
        "report-html": _cmd_report_html,
        "serve":       _cmd_serve,
        "update":      _cmd_update,
        "demo":        _cmd_demo,
    }

    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
