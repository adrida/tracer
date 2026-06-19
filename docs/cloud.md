# `tracer cloud`: drive Tracer Cloud from the terminal

`tracer cloud` mirrors the Tracer Cloud web app from your shell: log in once,
then create and train tracers, run live routing queries, manage tests, models,
observability keys, billing, and more. It talks to the same API the dashboard
uses, as you.

```bash
pip install tracer-llm
tracer cloud login          # opens your browser, pick your account
tracer cloud tracers        # list your tracers
```

## Auth

`tracer cloud login` runs a browser device flow: it opens a page where you pick
your account and approve, then hands the session back to the CLI on a localhost
callback. The session is stored only on this machine in `~/.tracer/cloud.json`
and refreshed automatically. Sign out with `tracer cloud logout`.

Headless / CI: pass credentials to skip the browser.

```bash
tracer cloud login --email you@example.com --password "$TRACER_PW"
```

Point at a different deployment or a local dev server with `--url` (or the
`TRACER_CLOUD_URL` env var). The package hardcodes no infrastructure details;
the client config is fetched from `<url>/api/cli/config` at login.

## Commands

### Account & workspaces
```bash
tracer cloud whoami                 # the logged-in user + workspaces
tracer cloud tenants                # list workspaces
tracer cloud tenant-delete <ref> --confirm "<name>"   # delete an EMPTY company workspace
```

### Tracers
```bash
tracer cloud tracers                # list
tracer cloud get <id|slug>          # a tracer + its versions
tracer cloud rename <id|slug> "New name"
tracer cloud delete <id|slug> [--confirm <name>]

# create + train
tracer cloud create "support router" --mode quick \
    --file traces.jsonl --teacher my-provider:my-model \
    --label-space "billing,fraud,general" --wait
tracer cloud create "watch mode" --mode auto --teacher my-provider:my-model --threshold 200
```
`--mode quick` trains on a labelled upload, `unlabelled` lets the teacher label
it, `auto` starts in watch mode (no data needed). `--wait` blocks until training
finishes. `--teacher` takes `provider:model_id` or a JSON object.

### Training, retraining, promotion
```bash
tracer cloud training <id>                      # job status
tracer cloud training-cancel <id>
tracer cloud retrain <id> --mode reuse|merge|replace [--trace-ids a,b,c] [--wait]
tracer cloud auto-retrain <id> --threshold 500  # or --disable
tracer cloud promote <id> [--version <version_id>]   # promote (or roll back)
```

### Live routing
```bash
tracer cloud route <id> "my card was declined"     # the dashboard's Live Query
tracer cloud route <id> "..." --lite               # skip the teacher call
tracer cloud route <id> "..." --show-exemplars
```

### Tests (batteries)
```bash
tracer cloud batteries list <id>
tracer cloud batteries create <id> --name "smoke" --cases cases.json
tracer cloud batteries run <id> --battery <battery_id> [--version <id>]
```

### Models
```bash
tracer cloud models list
tracer cloud models add --name my-llm --endpoint https://… --upstream model-id \
    --api-key "$KEY" --kind llm --cost-in 0.15 --cost-out 0.6
tracer cloud models delete --id <model_id>
```

### Observability
```bash
tracer cloud ingest-keys create --name "my app"     # mint a workspace ingest key
tracer cloud ingest-keys list
tracer cloud ingest-keys revoke --id <key_id>
tracer cloud ingest --key trobs_… --events events.json   # send observed events
```
Most teams stream live traffic with [`tracer.watch`](watch.md) instead of
posting batches by hand.

### Keys, analytics, billing, scan, label space
```bash
tracer cloud keys list|create|revoke <id>     # per-tracer gateway API keys
tracer cloud analytics <id> --range 24h|7d|30d
tracer cloud label-space <id> [--set a,b,c] [--strict]
tracer cloud billing [--tenant <ref>]
tracer cloud scan path/to/traces.jsonl        # public scan
```

Add `--json` to any read command for raw JSON output.

## Environment variables

| Var | Effect |
|-----|--------|
| `TRACER_CLOUD_URL` | Tracer Cloud base URL (default `https://app.tracerml.ai`) |
| `TRACER_TENANT` | default workspace id/slug/name when a command needs one |
| `TRACER_CLOUD_CONFIG` | path to the stored session (default `~/.tracer/cloud.json`) |
