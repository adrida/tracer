"""Lightweight prediction server for a fitted TRACER policy.

    tracer serve .tracer --port 8000

Exposes:
    POST /predict   {"embedding": [0.1, 0.2, ...]}  →  {"label", "decision", "accept_score", "stage"}
    POST /predict_batch  {"embeddings": [[...], ...]}  →  {"labels", "decisions", ...}
    GET  /health    →  {"status": "ok", "method", "coverage", "n_labels"}

Zero external dependencies - uses http.server from stdlib.

Security defaults (opt out explicitly for LAN/production exposure):
    - Binds to 127.0.0.1 (not 0.0.0.0)
    - No CORS headers unless --cors-origin is set
    - Request bodies capped (default 16 MiB)
    - Batch size capped (default 10_000 rows)
"""

from __future__ import annotations

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional, Union

import numpy as np


_router = None
_manifest = None
# Module-level knobs set by serve() before the handler accepts traffic.
_cors_origin: Optional[str] = None
_max_body_bytes: int = 16 * 1024 * 1024
_max_batch: int = 10_000
# Cap total float elements after JSON parse (batch_rows * dim amplification).
_max_elements: int = 10_000 * 4096


class _BodyTooLarge(ValueError):
    """Request body exceeded max_body_bytes."""


class _Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/health":
            self._json_response(200, {
                "status": "ok",
                "method": _manifest.selected_method,
                "coverage": _manifest.coverage_cal,
                "teacher_agreement": _manifest.teacher_agreement_cal,
                "n_labels": len(_manifest.label_space),
                "n_traces": _manifest.n_traces,
            })
        else:
            self._json_response(404, {"error": "not found",
                                       "endpoints": ["GET /health", "POST /predict",
                                                      "POST /predict_batch"]})

    def do_POST(self):
        try:
            body = self._read_body()
        except _BodyTooLarge as e:
            self._json_response(413, {"error": str(e)})
            return
        except Exception as e:
            self._json_response(400, {"error": str(e)})
            return

        if self.path == "/predict":
            self._handle_predict(body)
        elif self.path == "/predict_batch":
            self._handle_predict_batch(body)
        else:
            self._json_response(404, {"error": "not found"})

    def do_OPTIONS(self):
        # Preflight only when CORS is explicitly enabled.
        if not _cors_origin:
            self.send_error(404)
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", _cors_origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _handle_predict(self, body: dict):
        emb = body.get("embedding")
        if emb is None:
            self._json_response(400, {"error": "missing 'embedding' field"})
            return
        try:
            x = np.asarray(emb, dtype=np.float32)
            if x.size > _max_elements:
                self._json_response(
                    413,
                    {"error": f"embedding too large: {x.size} > max_elements={_max_elements}"},
                )
                return
            out = _router.predict(x)
            self._json_response(200, out)
        except Exception as e:
            self._json_response(500, {"error": str(e)})

    def _handle_predict_batch(self, body: dict):
        embs = body.get("embeddings")
        if embs is None:
            self._json_response(400, {"error": "missing 'embeddings' field"})
            return
        if not isinstance(embs, list):
            self._json_response(400, {"error": "'embeddings' must be a list"})
            return
        if len(embs) > _max_batch:
            self._json_response(
                413,
                {"error": f"batch too large: {len(embs)} > max_batch={_max_batch}"},
            )
            return
        try:
            X = np.asarray(embs, dtype=np.float32)
            if X.size > _max_elements:
                self._json_response(
                    413,
                    {"error": f"embeddings too large: {X.size} > max_elements={_max_elements}"},
                )
                return
            out = _router.predict_batch(X)
            # Convert numpy arrays to lists for JSON serialization
            result = {
                "labels": out["labels"],
                "decisions": out["decisions"],
                "handled": out["handled"].tolist(),
            }
            self._json_response(200, result)
        except Exception as e:
            self._json_response(500, {"error": str(e)})

    def _read_body(self) -> dict:
        raw_len = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_len)
        except (TypeError, ValueError) as e:
            raise ValueError("invalid Content-Length") from e
        if length <= 0:
            raise ValueError("empty request body")
        if length > _max_body_bytes:
            # Do not read the body; close so unread bytes cannot pin the worker.
            raise _BodyTooLarge(
                f"request body too large: {length} > max_body_bytes={_max_body_bytes}"
            )
        raw = self.rfile.read(length)
        return json.loads(raw)

    def _json_response(self, code: int, data: dict):
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if _cors_origin:
            self.send_header("Access-Control-Allow-Origin", _cors_origin)
        if code in (413, 400, 500):
            # Avoid keep-alive reuse after partial/oversized requests.
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # Quiet logging - only errors
        if args and str(args[1]).startswith("5"):
            super().log_message(fmt, *args)


def serve(
    artifact_dir: Union[str, Path] = ".tracer",
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    cors_origin: Optional[str] = None,
    max_body_bytes: int = 16 * 1024 * 1024,
    max_batch: int = 10_000,
):
    """Start a prediction server for a fitted TRACER policy.

    Parameters
    ----------
    artifact_dir : path to .tracer/ directory
    host : bind address (default: 127.0.0.1 — loopback only)
    port : listen port (default: 8000)
    cors_origin : if set, emit Access-Control-Allow-Origin with this value
                  (including ``*``). Default ``None`` = no CORS headers.
    max_body_bytes : reject request bodies larger than this (default 16 MiB)
    max_batch : reject /predict_batch with more rows than this (default 10_000)

    Notes
    -----
    There is no authentication. Only expose beyond localhost behind a reverse
    proxy that enforces auth/TLS, or bind deliberately with ``host='0.0.0.0'``.
    ``pipeline.joblib`` is loaded via joblib/pickle — only serve artifacts you trust.
    """
    global _router, _manifest, _cors_origin, _max_body_bytes, _max_batch

    from tracer.runtime.router import Router
    from tracer.policy.artifacts import load_manifest

    artifact_dir = Path(artifact_dir)
    _manifest = load_manifest(artifact_dir / "manifest.json")
    _router = Router.load(artifact_dir)
    _cors_origin = cors_origin
    _max_body_bytes = int(max_body_bytes)
    _max_batch = int(max_batch)

    server = HTTPServer((host, port), _Handler)
    # Bound stuck clients (slow-body / idle) so one hung socket cannot pin the process.
    server.timeout = 30
    server.socket.settimeout(30)
    method = _manifest.selected_method or "none"
    cov = f"{_manifest.coverage_cal:.1%}" if _manifest.coverage_cal else "n/a"
    print(f"\n  TRACER serve")
    print(f"  method={method}  coverage={cov}  labels={len(_manifest.label_space)}")
    print(f"  listening on http://{host}:{port}")
    if host in ("0.0.0.0", "::", "[::]"):
        print("  WARNING: bound on all interfaces with no auth — use a reverse proxy")
    if cors_origin:
        print(f"  CORS Allow-Origin: {cors_origin}")
    print(f"  endpoints:")
    predict_ex = '{"embedding": [...]}'
    batch_ex = '{"embeddings": [[...], ...]}'
    print(f"    POST /predict        {predict_ex}")
    print(f"    POST /predict_batch  {batch_ex}")
    print("    GET  /health")
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down.")
        server.shutdown()
