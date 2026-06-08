"""Tests for the Embedder factories (embeddings/embedder.py).

`from_callable` and `from_endpoint` are exercised with no optional
dependencies: the endpoint factory is tested against a stdlib mock HTTP
server. The `from_sentence_transformers` factory pulls in the optional
`sentence-transformers` extra, so its test is skipped when that extra is not
installed, keeping the core CI matrix fast.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tracer.embeddings.embedder import Embedder


# ── from_callable ─────────────────────────────────────────────────────────────

def test_from_callable_shapes_and_dtype():
    def fn(texts):
        return [[float(len(t)), 1.0] for t in texts]

    emb = Embedder.from_callable(fn)
    out = emb.embed(["a", "bbb"])
    assert out.shape == (2, 2)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[:, 0], [1.0, 3.0])


def test_embed_one_returns_1d():
    emb = Embedder.from_callable(lambda texts: [[1.0, 2.0, 3.0] for _ in texts])
    one = emb.embed_one("hello")
    assert one.shape == (3,)
    assert one.dtype == np.float32


def test_from_callable_repr():
    emb = Embedder.from_callable(lambda texts: [[0.0] for _ in texts])
    assert "callable" in repr(emb)


# ── from_endpoint (mock HTTP server) ───────────────────────────────────────────

class _EmbedHandler(BaseHTTPRequestHandler):
    """Echoes a deterministic embedding derived from input length.

    Single mode:  {"input": "ab"}            -> {"embedding": [2.0]}
    Batch mode:   {"inputs": ["a", "bb"]}    -> {"embeddings": [[1.0], [2.0]]}
    """

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        self.server.request_count += 1
        self.server.last_auth = self.headers.get("Authorization")

        if "inputs" in body:
            payload = {"embeddings": [[float(len(t))] for t in body["inputs"]]}
        else:
            payload = {"embedding": [float(len(body["input"]))]}

        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


@pytest.fixture
def mock_endpoint():
    httpd = HTTPServer(("127.0.0.1", 0), _EmbedHandler)
    httpd.request_count = 0
    httpd.last_auth = None
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    yield httpd, f"http://127.0.0.1:{port}"
    httpd.shutdown()
    thread.join(timeout=5)
    httpd.server_close()


def test_from_endpoint_single_mode(mock_endpoint):
    httpd, url = mock_endpoint
    emb = Embedder.from_endpoint(url)
    out = emb.embed(["a", "bb", "ccc"])
    assert out.shape == (3, 1)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[:, 0], [1.0, 2.0, 3.0])
    # One HTTP request per text in single mode.
    assert httpd.request_count == 3


def test_from_endpoint_batch_mode(mock_endpoint):
    httpd, url = mock_endpoint
    emb = Embedder.from_endpoint(url, batch_key="inputs",
                                 batch_output_key="embeddings")
    out = emb.embed(["a", "bb"])
    assert out.shape == (2, 1)
    np.testing.assert_allclose(out[:, 0], [1.0, 2.0])
    # The whole batch goes in a single request.
    assert httpd.request_count == 1


def test_from_endpoint_passes_headers(mock_endpoint):
    httpd, url = mock_endpoint
    emb = Embedder.from_endpoint(url, headers={"Authorization": "Bearer xyz"})
    emb.embed(["a"])
    assert httpd.last_auth == "Bearer xyz"


def test_from_endpoint_repr(mock_endpoint):
    _, url = mock_endpoint
    emb = Embedder.from_endpoint(url)
    assert "endpoint" in repr(emb)


# ── from_sentence_transformers (optional extra) ────────────────────────────────

def test_from_sentence_transformers_smoke():
    # Skipped unless the optional `sentence-transformers` extra is installed,
    # so the core CI matrix stays fast.
    pytest.importorskip("sentence_transformers")
    emb = Embedder.from_sentence_transformers("all-MiniLM-L6-v2")
    out = emb.embed(["hello world", "another sentence"])
    assert out.ndim == 2
    assert out.shape[0] == 2
    assert out.dtype == np.float32
    assert "sentence-transformers" in repr(emb)
