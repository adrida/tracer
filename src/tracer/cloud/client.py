"""Auth + thin REST client for Tracer Cloud.

Stdlib-only (urllib) so the OSS package gains no runtime dependency. Holds a
Tracer Cloud session in ``~/.tracer/cloud.json`` and presents it to the
Tracer Cloud web API as a bearer token, exactly as the browser session does.
"""

from __future__ import annotations

import io
import json
import mimetypes
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

# The package ships ZERO infrastructure identifiers. Only the public Tracer
# Cloud base URL is baked in; the auth base URL + public client key are fetched
# from {base_url}/api/cli/config at login (so they can be rotated server-side
# without breaking installed CLIs). All overridable by env to point at a local
# dev server or a different deployment.
DEFAULT_BASE_URL = os.environ.get("TRACER_CLOUD_URL", "https://app.tracerml.ai")
DEFAULT_AUTH_URL = os.environ.get("TRACER_CLOUD_AUTH_URL", "")
DEFAULT_AUTH_KEY = os.environ.get("TRACER_CLOUD_AUTH_KEY", "")

CONFIG_PATH = Path(
    os.environ.get("TRACER_CLOUD_CONFIG", str(Path.home() / ".tracer" / "cloud.json"))
)


class CloudError(RuntimeError):
    """A Cloud request failed; message is safe to print to the user."""


# A real User-Agent: the default "Python-urllib/x" trips Cloudflare's bot
# protection on app.tracerml.ai (HTTP 403, CF error 1010).
USER_AGENT = "tracer-cli/0.3.0 (+https://tracerml.ai)"


def _http(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    timeout: float = 60.0,
) -> tuple[int, bytes, dict[str, str]]:
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("User-Agent", USER_AGENT)
    req.add_header("Accept", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers or {})
    except urllib.error.URLError as e:
        raise CloudError(f"network error reaching {url}: {e.reason}") from e


def _json_or_text(body: bytes) -> Any:
    if not body:
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return body.decode("utf-8", "replace")


class CloudClient:
    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}

    # ----- config persistence -------------------------------------------------
    @classmethod
    def load(cls) -> "CloudClient":
        if CONFIG_PATH.exists():
            try:
                return cls(json.loads(CONFIG_PATH.read_text()))
            except (ValueError, OSError):
                pass
        return cls({})

    def save(self) -> None:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(self.cfg, indent=2))
        try:
            os.chmod(CONFIG_PATH, 0o600)
        except OSError:
            pass

    @property
    def base_url(self) -> str:
        return (self.cfg.get("base_url") or DEFAULT_BASE_URL).rstrip("/")

    @property
    def auth_url(self) -> str:
        return (self.cfg.get("auth_url") or DEFAULT_AUTH_URL).rstrip("/")

    @property
    def auth_key(self) -> str:
        return self.cfg.get("auth_key") or DEFAULT_AUTH_KEY

    @property
    def logged_in(self) -> bool:
        return bool(self.cfg.get("access_token"))

    # ----- auth ----------------------------------------------------------------
    def _ensure_auth_config(self) -> None:
        """Fetch the public auth base URL + client key from the Tracer Cloud
        bootstrap endpoint if we don't already have them. Keeps the package free
        of any hardcoded infra identifiers; the values are public (the same the
        browser ships) and cached into the local config."""
        if self.cfg.get("auth_url") and self.cfg.get("auth_key"):
            return
        if DEFAULT_AUTH_URL and DEFAULT_AUTH_KEY:  # env override
            self.cfg["auth_url"], self.cfg["auth_key"] = DEFAULT_AUTH_URL, DEFAULT_AUTH_KEY
            return
        status, body, _ = _http("GET", f"{self.base_url}/api/cli/config")
        cfg = _json_or_text(body)
        if status != 200 or not isinstance(cfg, dict) or not cfg.get("auth_url") or not cfg.get("auth_key"):
            raise CloudError(
                f"could not fetch CLI config from {self.base_url}/api/cli/config "
                f"(status {status}); set --url or TRACER_CLOUD_URL"
            )
        self.cfg["auth_url"] = cfg["auth_url"]
        self.cfg["auth_key"] = cfg["auth_key"]

    def login_password(self, email: str, password: str) -> dict[str, Any]:
        """Password grant -> store the Tracer Cloud session."""
        self._ensure_auth_config()
        url = f"{self.auth_url}/auth/v1/token?grant_type=password"
        status, body, _ = _http(
            "POST",
            url,
            headers={"apikey": self.auth_key, "Content-Type": "application/json"},
            data=json.dumps({"email": email, "password": password}).encode(),
        )
        payload = _json_or_text(body)
        if status != 200 or not isinstance(payload, dict) or "access_token" not in payload:
            msg = payload.get("error_description") if isinstance(payload, dict) else payload
            raise CloudError(f"login failed: {msg or status}")
        self._store_session(payload, email=email)
        return payload

    def _store_session(self, payload: dict[str, Any], email: str | None = None) -> None:
        self._ensure_auth_config()  # persist auth config (covers browser login)
        self.cfg["base_url"] = self.base_url
        self.cfg["auth_url"] = self.auth_url
        self.cfg["auth_key"] = self.auth_key
        self.cfg["access_token"] = payload["access_token"]
        self.cfg["refresh_token"] = payload.get("refresh_token")
        # expires_at may be absolute (epoch) or we derive from expires_in.
        exp = payload.get("expires_at")
        if not exp and payload.get("expires_in"):
            exp = int(time.time()) + int(payload["expires_in"])
        self.cfg["expires_at"] = exp
        user = payload.get("user") or {}
        self.cfg["email"] = email or user.get("email")
        self.cfg["user_id"] = user.get("id")
        self.save()

    def _refresh(self) -> None:
        rt = self.cfg.get("refresh_token")
        if not rt:
            raise CloudError("session expired and no refresh token; run `tracer cloud login`")
        self._ensure_auth_config()
        url = f"{self.auth_url}/auth/v1/token?grant_type=refresh_token"
        status, body, _ = _http(
            "POST",
            url,
            headers={"apikey": self.auth_key, "Content-Type": "application/json"},
            data=json.dumps({"refresh_token": rt}).encode(),
        )
        payload = _json_or_text(body)
        if status != 200 or not isinstance(payload, dict) or "access_token" not in payload:
            raise CloudError("session refresh failed; run `tracer cloud login` again")
        self._store_session(payload, email=self.cfg.get("email"))

    def _access_token(self) -> str:
        if not self.logged_in:
            raise CloudError("not logged in; run `tracer cloud login`")
        exp = self.cfg.get("expires_at")
        if isinstance(exp, (int, float)) and time.time() > float(exp) - 60:
            self._refresh()
        return self.cfg["access_token"]

    def logout(self) -> None:
        for k in ("access_token", "refresh_token", "expires_at", "email", "user_id"):
            self.cfg.pop(k, None)
        self.save()

    # ----- web API (session-authed routes, via bearer JWT) --------------------
    def api(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        if params:
            url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        headers = {"Authorization": f"Bearer {self._access_token()}"}
        data = None
        if json_body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(json_body).encode()
        status, body, _ = _http(method, url, headers=headers, data=data)
        payload = _json_or_text(body)
        if status >= 400:
            err = payload.get("error") if isinstance(payload, dict) else payload
            raise CloudError(f"{method} {path} -> {status}: {err}")
        return payload

    def api_multipart(self, path: str, fields: dict[str, str], file_path: str | None) -> Any:
        """POST multipart/form-data (used by the upload + create routes)."""
        boundary = f"----tracer{uuid.uuid4().hex}"
        buf = io.BytesIO()

        def w(s: str) -> None:
            buf.write(s.encode())

        for name, value in fields.items():
            if value is None:
                continue
            w(f"--{boundary}\r\n")
            w(f'Content-Disposition: form-data; name="{name}"\r\n\r\n')
            w(f"{value}\r\n")
        if file_path:
            fp = Path(file_path)
            ctype = mimetypes.guess_type(fp.name)[0] or "application/octet-stream"
            w(f"--{boundary}\r\n")
            w(f'Content-Disposition: form-data; name="file"; filename="{fp.name}"\r\n')
            w(f"Content-Type: {ctype}\r\n\r\n")
            buf.write(fp.read_bytes())
            w("\r\n")
        w(f"--{boundary}--\r\n")

        headers = {
            "Authorization": f"Bearer {self._access_token()}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }
        status, body, _ = _http(
            "POST", f"{self.base_url}{path}", headers=headers, data=buf.getvalue()
        )
        payload = _json_or_text(body)
        if status >= 400:
            err = payload.get("error") if isinstance(payload, dict) else payload
            raise CloudError(f"POST {path} -> {status}: {err}")
        return payload

    def put_url(self, url: str, data: bytes, content_type: str) -> None:
        """Raw PUT to an external presigned URL (R2 bulk upload). No auth header;
        the URL itself carries the signature."""
        status, body, _ = _http("PUT", url, headers={"Content-Type": content_type}, data=data)
        if status >= 400:
            raise CloudError(f"presigned PUT -> {status}: {body[:200]!r}")

    # ----- Tracer Cloud data API (the read views the dashboard uses) ----------
    def db(
        self,
        view: str,
        *,
        select: str = "*",
        filters: dict[str, str] | None = None,
        order: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        q: dict[str, str] = {"select": select}
        if filters:
            q.update(filters)
        if order:
            q["order"] = order
        if limit:
            q["limit"] = str(limit)
        self._ensure_auth_config()
        url = f"{self.auth_url}/rest/v1/{view}?" + urllib.parse.urlencode(q)
        headers = {
            "apikey": self.auth_key,
            "Authorization": f"Bearer {self._access_token()}",
            "Accept": "application/json",
        }
        status, body, _ = _http("GET", url, headers=headers)
        payload = _json_or_text(body)
        if status >= 400:
            msg = payload.get("message") if isinstance(payload, dict) else payload
            raise CloudError(f"db read {view} -> {status}: {msg}")
        return payload if isinstance(payload, list) else []

    def rpc(self, fn: str, args: dict[str, Any]) -> Any:
        self._ensure_auth_config()
        url = f"{self.auth_url}/rest/v1/rpc/{fn}"
        headers = {
            "apikey": self.auth_key,
            "Authorization": f"Bearer {self._access_token()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        status, body, _ = _http("POST", url, headers=headers, data=json.dumps(args).encode())
        payload = _json_or_text(body)
        if status >= 400:
            msg = payload.get("message") if isinstance(payload, dict) else payload
            raise CloudError(f"rpc {fn} -> {status}: {msg}")
        return payload

    def auth_user(self) -> dict[str, Any]:
        self._ensure_auth_config()
        url = f"{self.auth_url}/auth/v1/user"
        headers = {"apikey": self.auth_key, "Authorization": f"Bearer {self._access_token()}"}
        status, body, _ = _http("GET", url, headers=headers)
        payload = _json_or_text(body)
        if status >= 400:
            raise CloudError(f"whoami failed: {status}")
        return payload if isinstance(payload, dict) else {}

    # ----- bearer-key endpoints (gateway/ingest, no session) ------------------
    def keyed(self, method: str, path: str, key: str, json_body: Any | None = None) -> Any:
        headers = {"Authorization": f"Bearer {key}"}
        data = None
        if json_body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(json_body).encode()
        status, body, _ = _http(method, f"{self.base_url}{path}", headers=headers, data=data)
        payload = _json_or_text(body)
        if status >= 400:
            err = payload.get("error") if isinstance(payload, dict) else payload
            raise CloudError(f"{method} {path} -> {status}: {err}")
        return payload
