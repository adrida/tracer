"""Lightweight prediction server for a fitted TRACER policy.

    import tracer
    tracer.serve(".tracer", port=8000)

Exposes:
    POST /predict   {"embedding": [0.1, 0.2, ...]}  →  {"label", "decision", "accept_score", "stage"}
    POST /predict_batch  {"embeddings": [[...], ...]}  →  {"labels", "decisions", ...}
    GET  /health    →  {"status": "ok", "method", "coverage", "n_labels"}

Zero external dependencies - uses http.server from stdlib.
"""

from __future__ import annotations

import json
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Union

import numpy as np


_router = None
_manifest = None


class _Handler(BaseHTTPRequestHandler):

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

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
        except Exception as e:
            self._json_response(400, {"error": str(e)})
            return

        if self.path == "/predict":
            self._handle_predict(body)
        elif self.path == "/predict_batch":
            self._handle_predict_batch(body)
        else:
            self._json_response(404, {"error": "not found"})

    def _handle_predict(self, body: dict):
        emb = body.get("embedding")
        if emb is None:
            self._json_response(400, {"error": "missing 'embedding' field"})
            return
        try:
            x = np.asarray(emb, dtype=np.float32)
            out = _router.predict(x)
            self._json_response(200, out)
        except (ValueError, TypeError) as e:
            self._json_response(400, {"error": str(e)})
        except Exception as e:
            self._json_response(500, {"error": str(e)})

    def _handle_predict_batch(self, body: dict):
        embs = body.get("embeddings")
        if embs is None:
            self._json_response(400, {"error": "missing 'embeddings' field"})
            return
        try:
            X = np.asarray(embs, dtype=np.float32)
            out = _router.predict_batch(X)
            # Convert numpy arrays to lists for JSON serialization
            result = {
                "labels": out["labels"],
                "decisions": out["decisions"],
                "handled": out["handled"].tolist(),
            }
            self._json_response(200, result)
        except (ValueError, TypeError) as e:
            self._json_response(400, {"error": str(e)})
        except Exception as e:
            self._json_response(500, {"error": str(e)})

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > 8 * 1024 * 1024:
            raise ValueError("request body must be between 1 byte and 8 MiB")
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise ValueError("incomplete request body")
        body = json.loads(raw)
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")
        return body

    def _json_response(self, code: int, data: dict):
        body = json.dumps(data, default=str).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True

    def log_message(self, fmt, *args):
        # Quiet logging - only errors
        if len(args) > 1 and str(args[1]).startswith("5"):
            super().log_message(fmt, *args)


def serve(
    artifact_dir: Union[str, Path] = ".tracer",
    host: str = "127.0.0.1",
    port: int = 8000,
):
    """Start a prediction server for a fitted TRACER policy.

    Parameters
    ----------
    artifact_dir : path to .tracer/ directory
    host : bind address (default: 127.0.0.1; set explicitly for remote access)
    port : listen port (default: 8000)
    """
    global _router, _manifest

    from tracer.runtime.router import Router
    from tracer.policy.artifacts import load_manifest

    artifact_dir = Path(artifact_dir)
    _manifest = load_manifest(artifact_dir / "manifest.json")
    _router = Router.load(artifact_dir)

    server = ThreadingHTTPServer((host, port), _Handler)
    method = _manifest.selected_method or "none"
    cov = f"{_manifest.coverage_cal:.1%}" if _manifest.coverage_cal else "n/a"
    print(f"\n  TRACER serve")
    print(f"  method={method}  coverage={cov}  labels={len(_manifest.label_space)}")
    print(f"  listening on http://{host}:{port}")
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
    finally:
        server.server_close()
