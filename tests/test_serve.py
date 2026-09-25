"""Tests for the prediction server (runtime/serve.py).

Exercises the real request handler over a live socket: /health, /predict,
/predict_batch and the error paths. The server is started on an ephemeral
port (127.0.0.1:0) in a background thread and shut down cleanly after the
module's tests run, so nothing leaks between test files.
"""

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import tracer
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
    result = tracer.fit(traces_path, artifact_dir, embeddings=X,
                        config=tracer.FitConfig(verbose=False))
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
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), serve_mod._Handler)
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


def test_predict_wrong_dim_is_400(server, fitted_artifact):
    status, body = _post(server, "/predict", {"embedding": [0.1, 0.2, 0.3]})
    assert status == 400
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


@pytest.mark.parametrize('body', [[], None, 2, 'string'])
def test_non_object_request_is_400(server, body):
    status, result = _post(server, '/predict', body)
    assert status == 400
    assert 'JSON object' in result['error']
    assert _get(server, '/health')[0] == 200


def test_slow_prediction_does_not_block_health(server, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    results = []
    def slow_predict(_):
        entered.set()
        assert release.wait(3)
        return {'label': 'a', 'decision': 'handled', 'accept_score': 1, 'stage': 0}
    monkeypatch.setattr(serve_mod._router, 'predict', slow_predict)
    thread = threading.Thread(target=lambda: results.append(_post(server, '/predict', {'embedding': [0]})))
    thread.start()
    try:
        assert entered.wait(2)
        with urlopen(server + '/health', timeout=1) as response:
            assert response.status == 200
    finally:
        release.set()
        thread.join(timeout=3)
    assert results[0][0] == 200


@pytest.mark.parametrize('stage', ['headers', 'body'])
@pytest.mark.parametrize('error', [BrokenPipeError, ConnectionResetError])
def test_disconnect_during_response_is_ignored(stage, error):
    from types import SimpleNamespace
    handler = object.__new__(serve_mod._Handler)
    handler.send_response = lambda *_: None
    handler.send_header = lambda *_: None
    def disconnected(*_): raise error('peer closed')
    handler.end_headers = disconnected if stage == 'headers' else lambda: None
    handler.wfile = SimpleNamespace(write=disconnected)
    handler._json_response(200, {'ok': True})
    assert handler.close_connection


def test_invalid_content_length_is_rejected_before_reading():
    import io
    handler = object.__new__(serve_mod._Handler)
    handler.rfile = io.BytesIO(b'{}')
    for value in ('-1', '0', str(8 * 1024 * 1024 + 1)):
        handler.headers = {'Content-Length': value}
        with pytest.raises(ValueError, match='request body'):
            handler._read_body()
        assert handler.rfile.tell() == 0


def test_serve_uses_threaded_loopback_and_closes_on_interrupt(fitted_artifact, monkeypatch):
    from unittest.mock import Mock
    artifact_dir, _ = fitted_artifact
    instance = Mock()
    instance.serve_forever.side_effect = KeyboardInterrupt
    factory = Mock(return_value=instance)
    monkeypatch.setattr(serve_mod, 'ThreadingHTTPServer', factory)
    serve_mod.serve(artifact_dir)
    assert factory.call_args.args[0] == ('127.0.0.1', 8000)
    instance.server_close.assert_called_once()
    instance.shutdown.assert_not_called()
