"""`tracer cloud <subcommand>`, drive Tracer Cloud from the terminal.

Mirrors the web UI: login, then manage tracers, training, routing, tests,
models, observability keys, billing and analytics. Reads (list/get) use the
same read views the dashboard reads; writes (create/train/promote/…)
call the same web API routes the dashboard posts to, authenticated with the
user's Tracer Cloud session as a bearer token.
"""

from __future__ import annotations

import getpass
import json as _json
import os
import sys
import time
from typing import Any

from .client import CloudClient, CloudError

# ANSI helpers (no dependency on the OSS color module to keep this standalone).
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_RESET = "\033[0m"


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(s: str, code: str) -> str:
    return f"{code}{s}{_RESET}" if _supports_color() else s


def _print_json(obj: Any) -> None:
    print(_json.dumps(obj, indent=2, default=str))


def _table(rows: list[dict[str, Any]], cols: list[str]) -> None:
    if not rows:
        print(_c("(none)", _DIM))
        return
    widths = {c: len(c) for c in cols}
    cells = []
    for r in rows:
        row = {}
        for c in cols:
            v = r.get(c)
            s = "" if v is None else str(v)
            if len(s) > 48:
                s = s[:45] + "…"
            row[c] = s
            widths[c] = max(widths[c], len(s))
        cells.append(row)
    header = "  ".join(_c(c.ljust(widths[c]), _BOLD) for c in cols)
    print(header)
    print(_c("  ".join("-" * widths[c] for c in cols), _DIM))
    for row in cells:
        print("  ".join(row[c].ljust(widths[c]) for c in cols))


# ----- tenant / tracer resolution --------------------------------------------
def _resolve_tenant(client: CloudClient, want: str | None) -> dict[str, Any]:
    tenants = client.db("my_tenants", select="*", order="created_at.asc")
    if not tenants:
        raise CloudError("no workspaces found for this account")
    want = want or os.environ.get("TRACER_TENANT")
    if want:
        for t in tenants:
            if want in (t.get("id"), t.get("slug"), t.get("name")):
                return t
        raise CloudError(f"workspace not found: {want}")
    return tenants[0]


def _resolve_tracer(client: CloudClient, ref: str) -> dict[str, Any]:
    # Try by id, then by slug.
    rows = client.db("tracer_summary", select="*", filters={"id": f"eq.{ref}"})
    if not rows:
        rows = client.db("tracer_summary", select="*", filters={"slug": f"eq.{ref}"})
    if not rows:
        raise CloudError(f"tracer not found: {ref}")
    return rows[0]


# ===== command handlers =======================================================
def _cmd_login(client: CloudClient, args) -> None:
    if args.url:
        client.cfg["base_url"] = args.url
    if args.auth_url:
        client.cfg["auth_url"] = args.auth_url

    # Password grant only when credentials are explicitly supplied (headless
    # / CI). Otherwise use the browser device flow, which opens a tab to pick
    # the account and hands the session back to a localhost callback.
    if args.email or args.password:
        email = args.email or input("Email: ").strip()
        password = args.password or getpass.getpass("Password: ")
        client.login_password(email, password)
        print(_c("✓", _GREEN), f"logged in as {email}")
        print(_c(f"  cloud: {client.base_url}", _DIM))
        return

    _login_browser(client)


def _login_browser(client: CloudClient) -> None:
    """Open the browser, capture the Tracer Cloud session on a localhost callback."""
    import http.server
    import secrets
    import threading
    import urllib.parse
    import webbrowser

    state = secrets.token_urlsafe(16)
    captured: dict[str, Any] = {}
    done = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence
            pass

        def do_GET(self):  # noqa: N802
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            ok = q.get("state", [None])[0] == state and "session" in q
            if ok:
                import base64

                try:
                    raw = base64.b64decode(q["session"][0])
                    captured.update(_json.loads(raw))
                except Exception:  # noqa: BLE001
                    ok = False
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            msg = "Connected. You can close this tab and return to the terminal." if ok \
                else "Login failed or cancelled. Return to the terminal."
            self.wfile.write(
                f"<html><body style='font-family:sans-serif;padding:40px'>"
                f"<h2>Tracer CLI</h2><p>{msg}</p></body></html>".encode()
            )
            done.set()

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    url = (
        f"{client.base_url}/app/cli/authorize?"
        + urllib.parse.urlencode({"port": port, "state": state, "mode": "session"})
    )
    print("Opening your browser to sign in…")
    print(_c(f"  {url}", _DIM))
    print(_c("  (if it didn't open, paste that URL into your browser)", _DIM))
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001
        pass

    if not done.wait(timeout=300):
        srv.shutdown()
        raise CloudError("login timed out; no callback received")
    srv.shutdown()

    if not captured.get("access_token"):
        raise CloudError("login failed: no session returned")
    client._store_session(captured, email=captured.get("email"))
    print(_c("✓", _GREEN), f"logged in as {captured.get('email') or 'user'}")
    print(_c(f"  cloud: {client.base_url}", _DIM))


def _cmd_logout(client: CloudClient, args) -> None:
    client.logout()
    print(_c("✓", _GREEN), "logged out")


def _cmd_whoami(client: CloudClient, args) -> None:
    user = client.auth_user()
    tenants = client.db("my_tenants", select="*", order="created_at.asc")
    if args.json:
        _print_json({"user": user, "tenants": tenants})
        return
    print(_c("User", _BOLD), user.get("email"), _c(f"({user.get('id')})", _DIM))
    print(_c("Cloud", _BOLD), client.base_url)
    print(_c("Workspaces", _BOLD))
    _table(tenants, ["id", "name", "slug", "kind"])


def _cmd_tenants(client: CloudClient, args) -> None:
    tenants = client.db("my_tenants", select="*", order="created_at.asc")
    if args.json:
        _print_json(tenants)
        return
    _table(tenants, ["id", "name", "slug", "kind"])


def _cmd_tenant_delete(client: CloudClient, args) -> None:
    tenant = _resolve_tenant(client, args.tenant)
    name = tenant.get("name") or tenant.get("slug") or tenant["id"]
    if tenant.get("kind") == "personal":
        raise CloudError("personal workspaces cannot be deleted (only company workspaces)")
    # SOTA destructive-action guard: require the exact name, typed.
    if args.confirm != name:
        raise CloudError(
            f"refusing to delete: pass --confirm \"{name}\" to confirm deleting this workspace"
        )
    ok = client.rpc("delete_company_tenant", {"_tenant": tenant["id"]})
    deleted = ok[0] if isinstance(ok, list) and ok else ok
    if deleted is True:
        print(_c("✓", _GREEN), "deleted workspace", name)
    else:
        raise CloudError("workspace not found or not deletable")


def _cmd_tracers(client: CloudClient, args) -> None:
    rows = client.db("tracer_summary", select="*", order="created_at.desc", limit=args.limit)
    if args.json:
        _print_json(rows)
        return
    _table(rows, ["id", "name", "slug", "status", "current_version_tag"])
    print(_c(f"\n{len(rows)} tracer(s)", _DIM))


def _cmd_get(client: CloudClient, args) -> None:
    t = _resolve_tracer(client, args.tracer)
    versions = client.db(
        "tracer_versions",
        select="id,version_tag,status,coverage_cal,teacher_agreement_cal,selected_method,created_at",
        filters={"tracer_id": f"eq.{t['id']}"},
        order="created_at.desc",
        limit=20,
    )
    if args.json:
        _print_json({"tracer": t, "versions": versions})
        return
    print(_c(t.get("name", "tracer"), _BOLD), _c(f"({t.get('slug')})", _DIM))
    print(f"  id        {t.get('id')}")
    print(f"  status    {t.get('status')}")
    print(f"  current   {t.get('current_version_tag')} ({t.get('current_version_status')})")
    if t.get("current_pipeline_label"):
        print(f"  pipeline  {t.get('current_pipeline_label')}")
    print(_c("\nVersions", _BOLD))
    _table(versions, ["id", "version_tag", "status", "coverage_cal", "teacher_agreement_cal", "selected_method"])


def _parse_teacher(s: str | None) -> dict[str, str] | None:
    """Accept 'provider:model_id' or a JSON object."""
    if not s:
        return None
    s = s.strip()
    if s.startswith("{"):
        return _json.loads(s)
    if ":" in s:
        provider, model_id = s.split(":", 1)
        # label matches the dashboard's ModelMenuEntry shape the create routes expect.
        return {"provider": provider, "model_id": model_id, "label": model_id}
    raise CloudError("teacher must be 'provider:model_id' or a JSON object")


def _cmd_create(client: CloudClient, args) -> None:
    tenant = _resolve_tenant(client, args.tenant)
    teacher = _parse_teacher(args.teacher)
    label_space = [s.strip() for s in args.label_space.split(",") if s.strip()] if args.label_space else None

    if args.mode == "auto":
        if not teacher:
            raise CloudError("auto mode needs --teacher provider:model_id")
        body = {
            "tenant_id": tenant["id"],
            "name": args.name,
            "teacher": teacher,
            "threshold": args.threshold,
        }
        if label_space:
            body["label_space"] = label_space
        if args.system_prompt:
            body["system_prompt"] = args.system_prompt
        res = client.api("POST", "/api/tracers/auto-create", json_body=body)
    else:  # labelled / unlabelled upload via quick-create (multipart)
        if not args.file:
            raise CloudError("quick/unlabelled mode needs --file <traces.jsonl|csv>")
        if not teacher:
            raise CloudError("provide --teacher provider:model_id")
        fields = {
            "tenant_id": tenant["id"],
            "name": args.name,
            "path": "unlabelled" if args.mode == "unlabelled" else "traces",
            "teacher": _json.dumps(teacher),
            "modality": args.modality,
        }
        if label_space:
            fields["label_space"] = _json.dumps(label_space)
        if args.system_prompt:
            fields["system_prompt"] = args.system_prompt
        res = client.api_multipart("/api/tracers/quick-create", fields, args.file)
    if args.json:
        _print_json(res)
        return
    print(_c("✓", _GREEN), "tracer queued")
    _print_json(res)
    if args.wait and isinstance(res, dict) and res.get("tracer_id"):
        _wait_for_training(client, res["tracer_id"])


def _cmd_rename(client: CloudClient, args) -> None:
    t = _resolve_tracer(client, args.tracer)
    res = client.api("PATCH", f"/api/tracers/{t['id']}", json_body={"name": args.name})
    print(_c("✓", _GREEN), "renamed ->", res.get("name", args.name))


def _cmd_delete(client: CloudClient, args) -> None:
    t = _resolve_tracer(client, args.tracer)
    confirm = args.confirm or t.get("name")
    client.api("DELETE", f"/api/tracers/{t['id']}", json_body={"confirm_name": confirm})
    print(_c("✓", _GREEN), "deleted", t.get("name"))


def _cmd_retrain(client: CloudClient, args) -> None:
    t = _resolve_tracer(client, args.tracer)
    body: dict[str, Any] = {"mode": args.mode}
    if args.trace_ids:
        body["trace_ids"] = [s.strip() for s in args.trace_ids.split(",") if s.strip()]
    if args.alpha is not None:
        body["alpha"] = args.alpha
    res = client.api("POST", f"/api/tracers/{t['id']}/retrain", json_body=body)
    print(_c("✓", _GREEN), "retrain queued:", res)
    if args.wait:
        _wait_for_training(client, t["id"])


def _cmd_auto_retrain(client: CloudClient, args) -> None:
    t = _resolve_tracer(client, args.tracer)
    body = {"enabled": not args.disable, "threshold": args.threshold}
    res = client.api("POST", f"/api/tracers/{t['id']}/auto-retrain", json_body=body)
    print(_c("✓", _GREEN), "auto-retrain:", res)


def _cmd_promote(client: CloudClient, args) -> None:
    t = _resolve_tracer(client, args.tracer)
    body = {"version_id": args.version} if args.version else {}
    res = client.api("POST", f"/api/tracers/{t['id']}/promote", json_body=body)
    print(_c("✓", _GREEN), res.get("action", "promoted"), res)


def _cmd_training(client: CloudClient, args) -> None:
    t = _resolve_tracer(client, args.tracer)
    res = client.api("GET", f"/api/tracers/{t['id']}/training/state")
    _print_json(res)


def _cmd_training_cancel(client: CloudClient, args) -> None:
    t = _resolve_tracer(client, args.tracer)
    res = client.api("POST", f"/api/tracers/{t['id']}/training/cancel")
    print(_c("✓", _GREEN), "cancel requested:", res)


def _wait_for_training(client: CloudClient, tracer_id: str, timeout: float = 900.0) -> None:
    print(_c("waiting for training…", _DIM))
    start = time.time()
    last = None
    while time.time() - start < timeout:
        try:
            st = client.api("GET", f"/api/tracers/{tracer_id}/training/state")
        except CloudError:
            time.sleep(5)
            continue
        status = st.get("status")
        if status != last:
            print(f"  status: {_c(str(status), _CYAN)}")
            last = status
        if status in ("ready", "failed", "error"):
            mark = _GREEN if status == "ready" else _RED
            print(_c("✓" if status == "ready" else "✗", mark), "training", status)
            return
        time.sleep(6)
    print(_c("(timed out waiting; check `tracer cloud training`)", _DIM))


def _cmd_route(client: CloudClient, args) -> None:
    t = _resolve_tracer(client, args.tracer)
    body: dict[str, Any] = {"input": args.input}
    if args.race:
        body["race"] = True
    params = {"mode": "lite"} if args.lite else None
    res = client.api(
        "POST", f"/api/tracers/{t['id']}/route-query", json_body=body, params=params
    )
    if args.json:
        _print_json(res)
        return
    deferred = res.get("deferred")
    decision = "deferred to LLM" if deferred else "handled by ML"
    decision_c = _RED if deferred else _GREEN
    print(_c("routing", _BOLD), _c(decision, decision_c),
          _c("(OOD escalated)" if res.get("ood_escalated") else "", _DIM))
    print(_c("intent ", _BOLD), res.get("dominant_intent"))
    # routed_to is an object now ({provider,model,label,tier,reason}); pull .model.
    _rt = res.get("routed_to")
    _model = res.get("effective_model") or (_rt.get("model") if isinstance(_rt, dict) else _rt)
    print(_c("model  ", _BOLD), _model, _c(f"cell={res.get('cell_id')}", _DIM))
    if res.get("conformal_upper") is not None:
        print(_c("conf.  ", _BOLD), f"upper={res.get('conformal_upper')}")
    lat = (res.get("route_ms") or 0) + (res.get("embed_ms") or 0)
    print(_c("cost   ", _BOLD), f"${res.get('cost_usd')}", _c(f"{lat:.0f}ms", _DIM))
    if args.show_exemplars and res.get("exemplars"):
        print(_c("exemplars", _BOLD))
        for ex in res["exemplars"][:5]:
            print("  -", (ex if isinstance(ex, str) else ex.get("input", ex)))


def _cmd_analytics(client: CloudClient, args) -> None:
    t = _resolve_tracer(client, args.tracer)
    res = client.api("GET", f"/api/tracers/{t['id']}/analytics", params={"range": args.range})
    _print_json(res)


def _cmd_label_space(client: CloudClient, args) -> None:
    t = _resolve_tracer(client, args.tracer)
    # No mutation flags -> read-only view of the current label space.
    if not args.set and args.strict is None:
        rows = client.db(
            "tracer_versions",
            select="id,version_tag,label_space,label_space_strict,status",
            filters={"tracer_id": f"eq.{t['id']}"},
            order="created_at.desc",
            limit=1,
        )
        cur = rows[0] if rows else {}
        if args.json:
            _print_json(cur)
            return
        ls = cur.get("label_space") or []
        print(_c("label space", _BOLD), f"({len(ls)} labels, strict={cur.get('label_space_strict')})")
        for lab in ls:
            print(" ", lab)
        return
    body: dict[str, Any] = {}
    if args.set:
        body["label_space"] = [s.strip() for s in args.set.split(",") if s.strip()]
    if args.strict is not None:
        body["label_space_strict"] = args.strict
    res = client.api("PATCH", f"/api/tracers/{t['id']}/label-space", json_body=body)
    print(_c("✓", _GREEN), "label space updated:", res)


def _cmd_batteries(client: CloudClient, args) -> None:
    t = _resolve_tracer(client, args.tracer)
    if args.action == "list":
        res = client.api("GET", f"/api/tracers/{t['id']}/batteries")
        if args.json:
            _print_json(res)
            return
        _table(res.get("batteries", []), ["id", "name", "n_cases", "created_at"])
    elif args.action == "create":
        cases = _json.loads(open(args.cases).read()) if args.cases else []
        body = {"name": args.name, "cases": cases}
        res = client.api("POST", f"/api/tracers/{t['id']}/batteries", json_body=body)
        bid = res.get("battery_id") if isinstance(res, dict) else None
        print(_c("✓", _GREEN), "battery created:", bid or res, _c(f"({res.get('n_cases')} cases)" if isinstance(res, dict) else "", _DIM))
    elif args.action == "run":
        params = {"version": args.version} if args.version else None
        res = client.api(
            "POST", f"/api/tracers/{t['id']}/batteries/{args.battery}/run", params=params
        )
        if args.json:
            _print_json(res)
            return
        summary = res.get("summary", {})
        print(_c("battery run", _BOLD),
              f"pass={summary.get('n_pass')} fail={summary.get('n_fail')} deferred={summary.get('n_deferred')}")


def _cmd_models(client: CloudClient, args) -> None:
    if args.action == "list":
        rows = client.db("model_catalog", select="*", order="provider.asc")
        if args.json:
            _print_json(rows)
            return
        _table(rows, ["provider", "model_id", "display_label", "tier_hint"])
    elif args.action == "add":
        tenant = _resolve_tenant(client, args.tenant)
        body = {
            "tenant_id": tenant["id"],
            "name": args.name,
            "display_label": args.label or args.name,
            "endpoint_url": args.endpoint,
            "upstream_model_id": args.upstream,
            "api_key": args.api_key,
            "model_kind": args.kind,
            "cost_input_per_1m_usd": args.cost_in,
            "cost_output_per_1m_usd": args.cost_out,
            "visibility": args.visibility,
        }
        res = client.api("POST", "/api/library/custom-models", json_body=body)
        print(_c("✓", _GREEN), "custom model added:", res)
    elif args.action == "delete":
        client.api("DELETE", f"/api/library/custom-models?id={args.id}")
        print(_c("✓", _GREEN), "deleted", args.id)


def _cmd_keys(client: CloudClient, args) -> None:
    """Per-tracer gateway API keys (trc_*)."""
    t = _resolve_tracer(client, args.tracer)
    base = f"/api/tracers/{t['id']}/api-keys"
    if args.action == "list":
        res = client.api("GET", base)
        if args.json:
            _print_json(res)
            return
        _table(res.get("keys", []), ["id", "name", "prefix", "last4", "last_used_at", "revoked_at"])
    elif args.action == "create":
        res = client.api("POST", base, json_body={"name": args.name})
        print(_c("✓ key (shown once):", _GREEN), res.get("full_key"))
    elif args.action == "revoke":
        client.api("DELETE", f"{base}/{args.id}")
        print(_c("✓", _GREEN), "revoked", args.id)


def _cmd_ingest_keys(client: CloudClient, args) -> None:
    """Per-tenant observability ingest keys (trobs_*)."""
    if args.action == "list":
        tenant = _resolve_tenant(client, args.tenant)
        res = client.api("GET", "/api/observe/keys", params={"tenant": tenant["id"]})
        if args.json:
            _print_json(res)
            return
        _table(res.get("keys", []), ["id", "name", "prefix", "last4", "last_used_at", "revoked_at"])
    elif args.action == "create":
        tenant = _resolve_tenant(client, args.tenant)
        res = client.api("POST", "/api/observe/keys", json_body={"tenant_id": tenant["id"], "name": args.name})
        print(_c("✓ ingest key (shown once):", _GREEN), res.get("full_key"))
    elif args.action == "revoke":
        client.api("DELETE", f"/api/observe/keys/{args.id}")
        print(_c("✓", _GREEN), "revoked", args.id)


def _cmd_ingest(client: CloudClient, args) -> None:
    """Send observed trace events to Tracer Cloud.

    Auto-routes by key type, matching the two UI paths:
      * trobs_* (tenant ingest key)   -> POST /v1/observe  (tenant-wide)
      * trc_*   (per-tracer gateway)  -> POST /v1/ingest   (bound to a tracer)
    """
    events = _json.loads(open(args.events).read())
    if isinstance(events, dict):
        events = [events]
    if args.key.startswith("trobs_"):
        res = client.keyed("POST", "/v1/observe", args.key, json_body={"events": events})
    else:
        res = client.keyed(
            "POST", "/v1/ingest", args.key, json_body={"source": args.source, "events": events}
        )
    print(_c("✓", _GREEN), "ingested:", res)


def _cmd_billing(client: CloudClient, args) -> None:
    tenant = _resolve_tenant(client, args.tenant)
    res = client.rpc("get_tenant_billing_summary", {"_tenant": tenant["id"]})
    row = res[0] if isinstance(res, list) and res else res
    if args.json:
        _print_json(row)
        return
    print(_c(f"Billing · {tenant.get('name')}", _BOLD))
    if isinstance(row, dict):
        print(f"  balance     ${row.get('balance_usd')}")
        print(f"  credits     ${row.get('credits_total')}")
        print(f"  usage       ${row.get('usage_total')}")
        print(f"  saved       ${row.get('saved_total')}")
        print(f"  ML queries  {row.get('ml_queries')}  deferred {row.get('deferred_queries')}")


def _cmd_scan(client: CloudClient, args) -> None:
    res = client.api_multipart("/api/scan", {}, args.file)
    # The agent may need to disambiguate the file format. Re-submit with answers
    # (CLI flags override the server's best-guess defaults) so the scan is
    # non-interactive instead of silently no-opping on an ambiguous file.
    if isinstance(res, dict) and res.get("needs_clarification"):
        answers = dict(res.get("defaults") or {})
        if args.input_field:
            answers["input_field"] = args.input_field
        if args.label_field:
            answers["label_field"] = args.label_field
        if args.task:
            answers["task"] = args.task
        print(_c("clarifying format", _DIM), _json.dumps(answers))
        res = client.api_multipart("/api/scan", {"answers": _json.dumps(answers)}, args.file)
    job_id = res.get("job_id") if isinstance(res, dict) else None
    if not job_id:
        _print_json(res)
        return
    print(_c("scan queued", _DIM), job_id)
    print(_c("link", _DIM), f"{client.base_url}/scan/{job_id}")
    start = time.time()
    while time.time() - start < 300:
        st = client.api("GET", f"/api/scan/{job_id}")
        if st.get("status") == "ready":
            print(_c("✓", _GREEN), "scan ready")
            if getattr(args, "json", False):
                _print_json(st)
            else:
                summ = st.get("summary") or {}
                proj = st.get("projection") or {}
                print(f"  type        {st.get('data_type')}")
                print(f"  rows        {st.get('n_rows')}")
                cs = summ.get("certifiable_share")
                if cs is not None:
                    print(f"  certifiable {round(float(cs) * 100)}% of traffic")
                print(f"  points      {len(proj.get('points') or [])}")
                print(_c("view", _DIM), f"{client.base_url}/scan/{job_id}")
            return
        if st.get("error"):
            raise CloudError(f"scan failed: {st['error']}")
        time.sleep(4)
    print(_c("(scan still running; re-poll later)", _DIM))


def _sample_inputs(path: str, n: int = 25) -> list[str]:
    """Up to n representative input strings from a jsonl/csv file, for the
    onboarding planner. Tolerant of common input-field aliases."""
    keys = ("input", "query", "text", "prompt", "question")
    out: list[str] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            head = f.readline()
            f.seek(0)
            if "," in head and not head.lstrip().startswith("{"):
                import csv
                for row in csv.DictReader(f):
                    v = next((row[k] for k in keys if row.get(k)), None) or next(
                        (x for x in row.values() if x and x.strip()), None
                    )
                    if v:
                        out.append(str(v)[:500])
                    if len(out) >= n:
                        break
            else:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = _json.loads(line)
                    except Exception:
                        continue
                    if isinstance(row, dict):
                        v = next((row[k] for k in keys if isinstance(row.get(k), str)), None)
                        if v:
                            out.append(v[:500])
                    elif isinstance(row, str):
                        out.append(row[:500])
                    if len(out) >= n:
                        break
    except OSError as e:
        raise CloudError(f"could not read {path}: {e}")
    return out


def _cmd_onboard(client: CloudClient, args) -> None:
    tenant = _resolve_tenant(client, args.tenant)
    samples = _sample_inputs(args.file)
    body: dict[str, Any] = {
        "tenant_id": tenant["id"],
        "use_case_description": args.describe,
        "sample_inputs": samples,
    }
    if args.teacher:
        body["teacher_model_id"] = args.teacher
    if args.labels:
        body["label_space"] = [s.strip() for s in args.labels.split(",") if s.strip()]
    res = client.api("POST", "/api/tracers/auto-onboard", json_body=body)
    if args.json:
        _print_json(res)
        return
    plan = res.get("plan", {}) if isinstance(res, dict) else {}
    print(_c("onboarding plan", _BOLD), _c(f"({len(samples)} samples)", _DIM))
    for k in ("task", "modality", "embedder", "cascade_mode", "dual_embed", "alpha"):
        if plan.get(k) is not None:
            print(f"  {k:<13}{plan.get(k)}")
    if plan.get("label_space"):
        print(f"  {'label_space':<13}{', '.join(plan['label_space'])}")
    if plan.get("system_prompt"):
        print(f"  {'system':<13}{str(plan['system_prompt'])[:200]}")
    if plan.get("rationale"):
        print(_c("why", _DIM), str(plan["rationale"])[:400])
    pl = res.get("planner", {}) if isinstance(res, dict) else {}
    if pl.get("cost_usd") is not None:
        print(_c(f"planner {pl.get('model')} · ${pl.get('cost_usd')} · {pl.get('latency_ms')}ms", _DIM))


def _cmd_traces(client: CloudClient, args) -> None:
    t = _resolve_tracer(client, args.tracer)
    body: dict[str, Any] = {"limit": args.limit, "order": "random" if args.random else "recent"}
    if args.errors:
        body["errors_only"] = True
    if args.tier:
        body["tier"] = args.tier
    if args.since_minutes:
        body["since_minutes"] = args.since_minutes
    if args.cell is not None:
        body["cell_id"] = args.cell
    res = client.api("POST", f"/api/tracers/{t['id']}/traces/select", json_body=body)
    ids = res.get("trace_ids", []) if isinstance(res, dict) else []
    if args.json:
        _print_json(res)
        return
    if args.ids_only:
        # comma-separated, for: retrain --trace-ids "$(tracer cloud traces X --errors --ids-only)"
        print(",".join(ids))
        return
    print(_c(f"{len(ids)} trace(s)", _BOLD))
    for i in ids:
        print(" ", i)


def _cmd_create_bulk(client: CloudClient, args) -> None:
    import mimetypes
    tenant = _resolve_tenant(client, args.tenant)
    fname = os.path.basename(args.file)
    ctype = mimetypes.guess_type(fname)[0] or (
        "application/x-ndjson" if fname.endswith((".jsonl", ".ndjson")) else "text/plain"
    )
    try:
        data = open(args.file, "rb").read()
    except OSError as e:
        raise CloudError(f"could not read {args.file}: {e}")
    size = len(data)
    print(_c("presigning", _DIM), f"{fname} ({size / 1_048_576:.1f} MB)")
    pre = client.api("POST", "/api/tracers/uploads/presign-bulk", json_body={
        "tenant_id": tenant["id"], "filename": fname, "content_type": ctype, "size_bytes": size,
    })
    url = pre.get("presigned_url") if isinstance(pre, dict) else None
    storage_path = pre.get("storage_path") if isinstance(pre, dict) else None
    if not url or not storage_path:
        raise CloudError(f"presign failed: {pre}")
    print(_c("uploading", _DIM), "to storage...")
    client.put_url(url, data, ctype)
    body: dict[str, Any] = {
        "tenant_id": tenant["id"],
        "name": args.name,
        "teacher": _json.dumps(_parse_teacher(args.teacher)) if args.teacher else None,
        "storage_path": storage_path,
        "filename": fname,
        "content_type": ctype,
    }
    if args.path:
        body["path"] = args.path
    if args.labels:
        body["label_space"] = [s.strip() for s in args.labels.split(",") if s.strip()]
    if args.system_prompt:
        body["system_prompt"] = args.system_prompt
    res = client.api("POST", "/api/tracers/quick-create-bulk", json_body=body)
    tid = res.get("tracer_id") if isinstance(res, dict) else None
    rows = res.get("estimated_total_rows") if isinstance(res, dict) else None
    print(_c("✓", _GREEN), "tracer created:", tid or res, _c(f"(~{rows} rows)" if rows else "", _DIM))
    if args.wait and tid:
        _wait_for_training(client, tid)


# ===== argparse wiring ========================================================
def build_parser(p) -> None:
    import argparse

    p.add_argument("--json", action="store_true", help="Raw JSON output")
    # Shared parent so `--json` is accepted AFTER the subcommand too
    # (e.g. `tracer cloud get <id> --json`), not only before it.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="Raw JSON output")
    sub = p.add_subparsers(dest="cloud_command", parser_class=lambda **kw: argparse.ArgumentParser(parents=[common], **kw))

    pl = sub.add_parser("login", help="Log in to Tracer Cloud")
    pl.add_argument("--email")
    pl.add_argument("--password")
    pl.add_argument("--url", help="Cloud base URL (default https://app.tracerml.ai)")
    pl.add_argument("--auth-url", dest="auth_url")

    sub.add_parser("logout", help="Clear the stored session")
    sub.add_parser("whoami", help="Show the logged-in user + workspaces")
    sub.add_parser("tenants", help="List workspaces")

    ptd = sub.add_parser("tenant-delete", help="Delete an EMPTY company workspace (owner only)")
    ptd.add_argument("tenant", help="workspace id / slug / name")
    ptd.add_argument("--confirm", required=True, help="the workspace name, typed exactly, to confirm")

    pt = sub.add_parser("tracers", help="List tracers")
    pt.add_argument("--limit", type=int, default=50)

    pg = sub.add_parser("get", help="Show a tracer + its versions")
    pg.add_argument("tracer", help="tracer id or slug")

    pc = sub.add_parser("create", help="Create a tracer (and start training)")
    pc.add_argument("name")
    pc.add_argument("--mode", choices=["quick", "unlabelled", "auto"], default="quick",
                    help="quick=labelled upload, unlabelled=teacher labels it, auto=watch-mode")
    pc.add_argument("--file", help="traces file (quick/unlabelled)")
    pc.add_argument("--teacher", help="teacher model 'provider:model_id'")
    pc.add_argument("--label-space", dest="label_space", help="comma-separated labels")
    pc.add_argument("--system-prompt", dest="system_prompt")
    pc.add_argument("--modality", default="text")
    pc.add_argument("--threshold", type=int, default=200, help="auto-mode retrain threshold")
    pc.add_argument("--tenant", help="workspace id/slug/name (default: first)")
    pc.add_argument("--wait", action="store_true", help="block until training finishes")

    pr = sub.add_parser("rename", help="Rename a tracer")
    pr.add_argument("tracer")
    pr.add_argument("name")

    pd = sub.add_parser("delete", help="Delete a tracer")
    pd.add_argument("tracer")
    pd.add_argument("--confirm", help="confirm name (defaults to the tracer's name)")

    prt = sub.add_parser("retrain", help="Retrain a tracer")
    prt.add_argument("tracer")
    prt.add_argument("--mode", choices=["reuse", "merge", "replace", "consume_orphans"], default="reuse")
    prt.add_argument("--trace-ids", dest="trace_ids", help="comma-separated trace ids (replace/merge)")
    prt.add_argument("--alpha", type=float, default=None)
    prt.add_argument("--wait", action="store_true")

    par = sub.add_parser("auto-retrain", help="Enable/disable scheduled retrain")
    par.add_argument("tracer")
    par.add_argument("--threshold", type=int, default=200)
    par.add_argument("--disable", action="store_true")

    pp = sub.add_parser("promote", help="Promote (or roll back) a version to prod")
    pp.add_argument("tracer")
    pp.add_argument("--version", help="version id (rollback target); omit to promote current")

    ptr = sub.add_parser("training", help="Show training state")
    ptr.add_argument("tracer")
    ptc = sub.add_parser("training-cancel", help="Cancel a running training job")
    ptc.add_argument("tracer")

    pq = sub.add_parser("route", help="Run a live routing query (the dashboard's Live Query)")
    pq.add_argument("tracer")
    pq.add_argument("input")
    pq.add_argument("--race", action="store_true")
    pq.add_argument("--lite", action="store_true", help="skip the teacher call (show would_call)")
    pq.add_argument("--show-exemplars", dest="show_exemplars", action="store_true",
                    help="print the matched cell's nearest training exemplars")

    pan = sub.add_parser("analytics", help="Query-event analytics")
    pan.add_argument("tracer")
    pan.add_argument("--range", default="7d", choices=["24h", "7d", "30d"])

    pls = sub.add_parser("label-space", help="View/update the label space")
    pls.add_argument("tracer")
    pls.add_argument("--set", help="comma-separated labels")
    pls.add_argument("--strict", dest="strict", action="store_true", default=None)

    pb = sub.add_parser("batteries", help="Test batteries (list/create/run)")
    pb.add_argument("action", choices=["list", "create", "run"])
    pb.add_argument("tracer")
    pb.add_argument("--name")
    pb.add_argument("--cases", help="JSON file of cases (create)")
    pb.add_argument("--battery", help="battery id (run)")
    pb.add_argument("--version", help="version id to run against")

    pm = sub.add_parser("models", help="Model library (list/add/delete)")
    pm.add_argument("action", choices=["list", "add", "delete"])
    pm.add_argument("--tenant")
    pm.add_argument("--name")
    pm.add_argument("--label")
    pm.add_argument("--endpoint")
    pm.add_argument("--upstream", help="upstream model id")
    pm.add_argument("--api-key", dest="api_key")
    pm.add_argument("--kind", choices=["llm", "embedding"], default="llm")
    pm.add_argument("--cost-in", dest="cost_in", type=float, default=0.0)
    pm.add_argument("--cost-out", dest="cost_out", type=float, default=0.0)
    pm.add_argument("--visibility", choices=["workspace", "private", "org"], default="workspace")
    pm.add_argument("--id", help="custom model id (delete)")

    pk = sub.add_parser("keys", help="Per-tracer gateway API keys")
    pk.add_argument("action", choices=["list", "create", "revoke"])
    pk.add_argument("tracer")
    pk.add_argument("--name")
    pk.add_argument("--id", help="key id (revoke)")

    pik = sub.add_parser("ingest-keys", help="Per-workspace observability ingest keys")
    pik.add_argument("action", choices=["list", "create", "revoke"])
    pik.add_argument("--tenant")
    pik.add_argument("--name")
    pik.add_argument("--id")

    pin = sub.add_parser("ingest", help="Send trace events (auto-routes by key prefix)")
    pin.add_argument("--key", required=True, help="trobs_* -> /v1/observe, or trc_* -> /v1/ingest")
    pin.add_argument("--events", required=True, help="JSON file of events")
    pin.add_argument("--source", default="cli")

    pbi = sub.add_parser("billing", help="Workspace billing summary")
    pbi.add_argument("--tenant")

    psc = sub.add_parser("scan", help="Public scan of a traces file")
    psc.add_argument("file")
    psc.add_argument("--input-field", dest="input_field", help="input column (skips the format clarify prompt)")
    psc.add_argument("--label-field", dest="label_field", help="label/teacher column")
    psc.add_argument("--task", choices=["classification", "other"], help="task type override")

    pon = sub.add_parser("onboard", help="Agentic onboarding plan from a sample file")
    pon.add_argument("file")
    pon.add_argument("--describe", required=True, help="one sentence: what you are labelling")
    pon.add_argument("--tenant", help="workspace id/slug/name")
    pon.add_argument("--teacher", help="teacher model id (e.g. gpt-5-mini)")
    pon.add_argument("--labels", help="comma-separated label space (optional)")

    ptr = sub.add_parser("traces", help="List trace ids (feed into retrain --trace-ids)")
    ptr.add_argument("tracer")
    ptr.add_argument("--errors", action="store_true", help="only errored traces")
    ptr.add_argument("--tier", help="filter by routed tier (e.g. ml, teacher)")
    ptr.add_argument("--since-minutes", dest="since_minutes", type=int, help="only the last N minutes")
    ptr.add_argument("--cell", type=int, help="filter by cell id")
    ptr.add_argument("--limit", type=int, default=500)
    ptr.add_argument("--random", action="store_true", help="random sample instead of most recent")
    ptr.add_argument("--ids-only", dest="ids_only", action="store_true", help="comma-separated ids only (for piping)")

    pcb = sub.add_parser("create-bulk", help="Create a tracer from a large file (presigned upload)")
    pcb.add_argument("file")
    pcb.add_argument("--name", required=True)
    pcb.add_argument("--teacher", help="provider:model_id (or JSON object)")
    pcb.add_argument("--tenant", help="workspace id/slug/name")
    pcb.add_argument("--path", help="upload path kind (server default if omitted)")
    pcb.add_argument("--labels", help="comma-separated label space")
    pcb.add_argument("--system-prompt", dest="system_prompt")
    pcb.add_argument("--wait", action="store_true", help="wait for training to finish")


_DISPATCH = {
    "login": _cmd_login,
    "logout": _cmd_logout,
    "whoami": _cmd_whoami,
    "tenants": _cmd_tenants,
    "tenant-delete": _cmd_tenant_delete,
    "tracers": _cmd_tracers,
    "get": _cmd_get,
    "create": _cmd_create,
    "rename": _cmd_rename,
    "delete": _cmd_delete,
    "retrain": _cmd_retrain,
    "auto-retrain": _cmd_auto_retrain,
    "promote": _cmd_promote,
    "training": _cmd_training,
    "training-cancel": _cmd_training_cancel,
    "route": _cmd_route,
    "analytics": _cmd_analytics,
    "label-space": _cmd_label_space,
    "batteries": _cmd_batteries,
    "models": _cmd_models,
    "keys": _cmd_keys,
    "ingest-keys": _cmd_ingest_keys,
    "ingest": _cmd_ingest,
    "billing": _cmd_billing,
    "scan": _cmd_scan,
    "onboard": _cmd_onboard,
    "traces": _cmd_traces,
    "create-bulk": _cmd_create_bulk,
}


def run(args) -> None:
    cmd = getattr(args, "cloud_command", None)
    if not cmd:
        print("usage: tracer cloud <subcommand>  (try `tracer cloud --help`)")
        sys.exit(1)
    client = CloudClient.load()
    # login/logout don't require an existing session.
    if cmd not in ("login", "logout") and not client.logged_in:
        print(_c("not logged in.", _RED), "run `tracer cloud login` first.")
        sys.exit(1)
    handler = _DISPATCH.get(cmd)
    if not handler:
        print(f"unknown subcommand: {cmd}")
        sys.exit(1)
    try:
        handler(client, args)
    except CloudError as e:
        print(_c("error:", _RED), e)
        sys.exit(1)
