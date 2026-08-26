"""Tests for the prediction server (runtime/serve.py).

Exercises the real request handler over a live socket: /health, /predict,
/predict_batch and the error paths. The server is started on an ephemeral
port (127.0.0.1:0) in a background thread and shut down cleanly after the
module's tests run, so nothing leaks between test files.
"""

import json
import threading
from http.server import HTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tracer.api import fit
from tracer.config import FitConfig
from tracer.runtime import serve as serve_mod
from tracer.policy.artifacts import load_manifest
from tracer.runtime.router import Router


# ── Fixtures ────────────────────────────────────────────────────────────────

def _make_traces(tmpdir, n=240, dim=16, n_classes=3, noise=0.05):
    """Synthetic, well-separated traces so fit() deploys a surrogate."""
    rng = np.random.RandomState(0)
    centers = rng.randn(n_classes, dim) * 4
    labels_int = rng.randint(0, n_classes, size=n)
    X = centers[labels_int] + rng.randn(n, dim) * 0.6
    names = [f"cls_{i}" for i in range(n_classes)]
    teacher = [names[i] for i in labels_int]
    for i in range(n):
        if rng.random() < noise:
            teacher[i] = names[rng.randint(0, n_classes)]
    path = Path(tmpdir) / "traces.jsonl"
    with path.open("w") as f:
        for i in range(n):
            f.write(json.dumps({"input": f"text {i}", "teacher": teacher[i],
                                "id": str(i)}) + "\n")
    return path, X.astype(np.float32)


@pytest.fixture(scope="module")
def fitted_artifact(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("serve_artifact")
    traces_path, X = _make_traces(tmpdir)
    artifact_dir = Path(tmpdir) / ".tracer"
    result = fit(traces_path, artifact_dir, embeddings=X,
                 config=FitConfig(verbose=False))
    assert result.manifest.selected_method is not None, \
        "test fixture expected a deployable policy"
    return artifact_dir, X


@pytest.fixture(scope="module")
def server(fitted_artifact):
    artifact_dir, _ = fitted_artifact
    # Wire the handler exactly as serve() does, but keep the server handle so we
    # can bind an ephemeral port and shut down cleanly.
    serve_mod._manifest = load_manifest(artifact_dir / "manifest.json")
    serve_mod._router = Router.load(artifact_dir)
    serve_mod._cors_origin = None
    serve_mod._max_body_bytes = 16 * 1024 * 1024
    serve_mod._max_batch = 10_000
    httpd = HTTPServer(("127.0.0.1", 0), serve_mod._Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()
    thread.join(timeout=5)
    httpd.server_close()


# ── Helpers ─────────────────────────────────────────────────────────────────

def _get(base, path):
    with urlopen(f"{base}{path}", timeout=5) as resp:
        return resp.status, json.loads(resp.read())


def _post(base, path, payload):
    req = Request(f"{base}{path}", data=json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except HTTPError as e:
        return e.code, json.loads(e.read())


# ── /health ───────────────────────────────────────────────────────────────────

def test_health(server, fitted_artifact):
    status, body = _get(server, "/health")
    assert status == 200
    assert body["status"] == "ok"
    assert body["method"]  # a deployed method name
    assert body["n_labels"] == 3
    assert 0.0 <= body["coverage"] <= 1.0


# ── /predict ──────────────────────────────────────────────────────────────────

def test_predict_returns_routing_decision(server, fitted_artifact):
    _, X = fitted_artifact
    status, body = _post(server, "/predict", {"embedding": X[0].tolist()})
    assert status == 200
    assert body["decision"] in ("handled", "deferred")
    assert "label" in body and "accept_score" in body and "stage" in body


def test_predict_missing_field_is_400(server):
    status, body = _post(server, "/predict", {"not_embedding": [1, 2, 3]})
    assert status == 400
    assert "embedding" in body["error"]


def test_predict_wrong_dim_is_500(server, fitted_artifact):
    # Router raises a dimension-mismatch ValueError, surfaced as a 500.
    status, body = _post(server, "/predict", {"embedding": [0.1, 0.2, 0.3]})
    assert status == 500
    assert "dimension" in body["error"].lower()


# ── /predict_batch ────────────────────────────────────────────────────────────

def test_predict_batch(server, fitted_artifact):
    _, X = fitted_artifact
    batch = X[:5].tolist()
    status, body = _post(server, "/predict_batch", {"embeddings": batch})
    assert status == 200
    assert len(body["labels"]) == 5
    assert len(body["decisions"]) == 5
    assert len(body["handled"]) == 5
    assert all(d in ("handled", "deferred") for d in body["decisions"])


def test_predict_batch_missing_field_is_400(server):
    status, body = _post(server, "/predict_batch", {"wrong": []})
    assert status == 400
    assert "embeddings" in body["error"]


# ── Routing / unknown paths ───────────────────────────────────────────────────

def test_unknown_get_path_is_404(server):
    try:
        _get(server, "/nope")
        assert False, "expected 404"
    except HTTPError as e:
        assert e.code == 404
        body = json.loads(e.read())
        assert "endpoints" in body


def test_unknown_post_path_is_404(server):
    status, body = _post(server, "/nope", {"x": 1})
    assert status == 404


def test_default_serve_host_is_loopback():
    import inspect
    from tracer.runtime.serve import serve
    assert inspect.signature(serve).parameters["host"].default == "127.0.0.1"


def test_oversized_body_is_413(server):
    serve_mod._max_body_bytes = 64
    try:
        payload = {"embedding": [0.1] * 100}
        raw = json.dumps(payload).encode()
        assert len(raw) > 64
        req = Request(
            f"{server}/predict",
            data=raw,
            headers={"Content-Type": "application/json", "Content-Length": str(len(raw))},
            method="POST",
        )
        try:
            urlopen(req, timeout=5)
            assert False, "expected 413"
        except HTTPError as e:
            assert e.code == 413
            body = json.loads(e.read())
            assert "too large" in body["error"]
    finally:
        serve_mod._max_body_bytes = 16 * 1024 * 1024


def test_batch_over_max_is_413(server, fitted_artifact):
    _, X = fitted_artifact
    serve_mod._max_batch = 2
    try:
        status, body = _post(server, "/predict_batch", {"embeddings": X[:5].tolist()})
        assert status == 413
        assert "batch too large" in body["error"]
    finally:
        serve_mod._max_batch = 10_000


def test_no_cors_header_by_default(server):
    with urlopen(f"{server}/health", timeout=5) as resp:
        assert resp.headers.get("Access-Control-Allow-Origin") is None


def test_cors_origin_when_enabled(server):
    serve_mod._cors_origin = "https://app.example.com"
    try:
        with urlopen(f"{server}/health", timeout=5) as resp:
            assert resp.headers.get("Access-Control-Allow-Origin") == "https://app.example.com"
    finally:
        serve_mod._cors_origin = None


def test_oversized_body_closes_connection(server):
    serve_mod._max_body_bytes = 64
    try:
        raw = json.dumps({"embedding": [0.1] * 100}).encode()
        req = Request(
            f"{server}/predict",
            data=raw,
            headers={"Content-Type": "application/json", "Content-Length": str(len(raw))},
            method="POST",
        )
        try:
            urlopen(req, timeout=5)
            assert False, "expected 413"
        except HTTPError as e:
            assert e.code == 413
            assert e.headers.get("Connection", "").lower() == "close"
    finally:
        serve_mod._max_body_bytes = 16 * 1024 * 1024


def test_cli_serve_host_default_is_loopback(monkeypatch):
    captured = {}

    def fake_serve(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("tracer.runtime.serve.serve", fake_serve)
    monkeypatch.setattr(sys, "argv", ["tracer", "serve", "/tmp/nonexistent-artifact"])
    from tracer.cli.main import main
    main()
    assert captured["host"] == "127.0.0.1"
    assert captured.get("cors_origin") is None
    assert captured["max_body_bytes"] == 16 * 1024 * 1024
    assert captured["max_batch"] == 10_000
