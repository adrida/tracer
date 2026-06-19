"""Unit tests for tracer.cloud.client.CloudClient. The HTTP layer (`_http`) is
monkeypatched, so these are hermetic, no network, no real Tracer Cloud."""
import json
import time

import pytest

import tracer.cloud.client as cc
from tracer.cloud.client import CloudClient, CloudError


def _far_future():
    return int(time.time()) + 9999


def _base_cfg(**over):
    cfg = {
        "base_url": "http://app.test",
        "auth_url": "http://sb.test",
        "auth_key": "anon",
        "access_token": "aaa.bbb.ccc",
        "refresh_token": "refresh-1",
        "expires_at": _far_future(),
        "email": "u@test",
    }
    cfg.update(over)
    return cfg


def _install_fake_http(monkeypatch, handler):
    """handler(method, url, headers, data) -> (status, body_bytes)."""
    calls = []

    def fake_http(method, url, *, headers=None, data=None, timeout=60.0):
        calls.append({"method": method, "url": url, "headers": headers or {}, "data": data})
        status, body = handler(method, url, headers or {}, data)
        return status, body, {}

    monkeypatch.setattr(cc, "_http", fake_http)
    return calls


# ----- config + resolution --------------------------------------------------- #
def test_defaults_when_cfg_empty():
    c = CloudClient({})
    assert c.base_url == cc.DEFAULT_BASE_URL
    assert c.auth_url == cc.DEFAULT_AUTH_URL
    assert c.auth_key == cc.DEFAULT_AUTH_KEY
    assert c.logged_in is False


def test_save_load_roundtrip(tmp_path, monkeypatch):
    p = tmp_path / "cloud.json"
    monkeypatch.setattr(cc, "CONFIG_PATH", p)
    c = CloudClient(_base_cfg())
    c.save()
    loaded = CloudClient.load()
    assert loaded.cfg["access_token"] == "aaa.bbb.ccc"
    assert loaded.base_url == "http://app.test"


def test_logout_clears_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "CONFIG_PATH", tmp_path / "c.json")
    c = CloudClient(_base_cfg())
    assert c.logged_in
    c.logout()
    assert c.logged_in is False
    assert "access_token" not in c.cfg


# ----- web API requests ------------------------------------------------------ #
def test_api_get_builds_bearer_request(monkeypatch):
    calls = _install_fake_http(monkeypatch, lambda *a: (200, b'{"ok":true}'))
    c = CloudClient(_base_cfg())
    out = c.api("GET", "/api/tracers/x")
    assert out == {"ok": True}
    assert calls[0]["url"] == "http://app.test/api/tracers/x"
    assert calls[0]["headers"]["Authorization"] == "Bearer aaa.bbb.ccc"


def test_api_params_and_json_body(monkeypatch):
    calls = _install_fake_http(monkeypatch, lambda *a: (200, b'{}'))
    c = CloudClient(_base_cfg())
    c.api("POST", "/api/x", json_body={"a": 1}, params={"range": "7d", "skip": None})
    assert "range=7d" in calls[0]["url"]
    assert "skip=" not in calls[0]["url"]  # None params dropped
    assert json.loads(calls[0]["data"]) == {"a": 1}


def test_api_raises_on_error_status(monkeypatch):
    _install_fake_http(monkeypatch, lambda *a: (404, b'{"error":"nope"}'))
    c = CloudClient(_base_cfg())
    with pytest.raises(CloudError) as ei:
        c.api("GET", "/api/x")
    assert "nope" in str(ei.value)


def test_db_builds_rest_query(monkeypatch):
    calls = _install_fake_http(monkeypatch, lambda *a: (200, b"[]"))
    c = CloudClient(_base_cfg())
    c.db("my_tenants", select="id,name", filters={"id": "eq.1"}, order="created_at.asc", limit=5)
    url = calls[0]["url"]
    assert url.startswith("http://sb.test/rest/v1/my_tenants?")
    assert "select=id%2Cname" in url
    assert "id=eq.1" in url and "order=created_at.asc" in url and "limit=5" in url
    assert calls[0]["headers"]["apikey"] == "anon"
    assert calls[0]["headers"]["Authorization"] == "Bearer aaa.bbb.ccc"


def test_rpc_posts_args(monkeypatch):
    calls = _install_fake_http(monkeypatch, lambda *a: (200, b"true"))
    c = CloudClient(_base_cfg())
    out = c.rpc("delete_company_tenant", {"_tenant": "t1"})
    assert out is True
    assert calls[0]["url"] == "http://sb.test/rest/v1/rpc/delete_company_tenant"
    assert json.loads(calls[0]["data"]) == {"_tenant": "t1"}


def test_keyed_uses_provided_key(monkeypatch):
    calls = _install_fake_http(monkeypatch, lambda *a: (200, b'{"ingested":1}'))
    c = CloudClient(_base_cfg())
    out = c.keyed("POST", "/v1/observe", "trobs_K", json_body={"events": []})
    assert out == {"ingested": 1}
    assert calls[0]["headers"]["Authorization"] == "Bearer trobs_K"


def test_multipart_frames_file_and_fields(monkeypatch):
    calls = _install_fake_http(monkeypatch, lambda *a: (200, b'{"ok":true}'))
    c = CloudClient(_base_cfg())
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.write(fd, b'{"input":"x"}\n'); os.close(fd)
    c.api_multipart("/api/scan", {"tenant_id": "t1"}, path)
    body = calls[0]["data"]
    assert b'name="tenant_id"' in body
    assert b'name="file"; filename=' in body
    assert b'{"input":"x"}' in body
    ctype = calls[0]["headers"]["Content-Type"]
    assert ctype.startswith("multipart/form-data; boundary=")


# ----- auth: login + refresh ------------------------------------------------- #
def test_login_password_stores_session(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "CONFIG_PATH", tmp_path / "c.json")

    def handler(method, url, headers, data):
        assert "grant_type=password" in url
        return 200, json.dumps({
            "access_token": "tok-new", "refresh_token": "ref-new", "expires_in": 3600,
            "user": {"id": "u1", "email": "u@test"},
        }).encode()

    _install_fake_http(monkeypatch, handler)
    c = CloudClient({"auth_url": "http://sb.test", "auth_key": "anon"})
    c.login_password("u@test", "pw")
    assert c.cfg["access_token"] == "tok-new"
    assert c.cfg["user_id"] == "u1"
    assert c.cfg["expires_at"] > time.time()


def test_access_token_refreshes_when_expired(monkeypatch):
    state = {"refreshed": False}

    def handler(method, url, headers, data):
        if "grant_type=refresh_token" in url:
            state["refreshed"] = True
            return 200, json.dumps({"access_token": "tok-2", "refresh_token": "ref-2",
                                    "expires_in": 3600}).encode()
        return 200, b"{}"

    monkeypatch.setattr(cc, "CONFIG_PATH", __import__("pathlib").Path("/tmp/_unused_cloud.json"))
    _install_fake_http(monkeypatch, handler)
    c = CloudClient(_base_cfg(expires_at=int(time.time()) - 10))  # already expired
    tok = c._access_token()
    assert state["refreshed"] is True
    assert tok == "tok-2"


def test_access_token_no_refresh_when_fresh(monkeypatch):
    state = {"refreshed": False}

    def handler(method, url, headers, data):
        if "grant_type=refresh_token" in url:
            state["refreshed"] = True
        return 200, b"{}"

    _install_fake_http(monkeypatch, handler)
    c = CloudClient(_base_cfg(expires_at=_far_future()))
    assert c._access_token() == "aaa.bbb.ccc"
    assert state["refreshed"] is False


def test_ensure_auth_config_fetches_from_bootstrap(monkeypatch):
    # No auth_url/auth_key in cfg and no env override -> fetch /api/cli/config.
    monkeypatch.setattr(cc, "DEFAULT_AUTH_URL", "")
    monkeypatch.setattr(cc, "DEFAULT_AUTH_KEY", "")

    def handler(method, url, headers, data):
        if url.endswith("/api/cli/config"):
            return 200, json.dumps({"auth_url": "http://fetched", "auth_key": "fetched-key"}).encode()
        return 200, b"[]"

    calls = _install_fake_http(monkeypatch, handler)
    c = CloudClient({"base_url": "http://app.test", "access_token": "a.b.c",
                     "expires_at": _far_future()})
    c.db("my_tenants")
    assert c.cfg["auth_url"] == "http://fetched"
    assert c.cfg["auth_key"] == "fetched-key"
    # the bootstrap endpoint was hit before the rest query
    assert any(call["url"] == "http://app.test/api/cli/config" for call in calls)


def test_api_without_login_raises(monkeypatch):
    _install_fake_http(monkeypatch, lambda *a: (200, b"{}"))
    c = CloudClient({})  # no token
    with pytest.raises(CloudError):
        c.api("GET", "/api/x")
