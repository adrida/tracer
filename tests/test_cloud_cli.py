"""Unit tests for tracer.cloud.cli helpers + command guards. No network: the
client is a fake recording double."""
import json
from argparse import Namespace

import pytest

import tracer.cloud.cli as cli
from tracer.cloud.client import CloudError


# ----- _parse_teacher -------------------------------------------------------- #
def test_parse_teacher_colon_form():
    # label is included to match the dashboard's ModelMenuEntry shape the create routes expect.
    assert cli._parse_teacher("openai:gpt-5-mini") == {
        "provider": "openai",
        "model_id": "gpt-5-mini",
        "label": "gpt-5-mini",
    }


def test_parse_teacher_json_form():
    out = cli._parse_teacher('{"provider":"anthropic","model_id":"claude"}')
    assert out["provider"] == "anthropic"


def test_parse_teacher_none():
    assert cli._parse_teacher(None) is None


def test_parse_teacher_invalid_raises():
    with pytest.raises(CloudError):
        cli._parse_teacher("just-a-model")


# ----- ingest auto-routes by key prefix -------------------------------------- #
class _FakeClient:
    def __init__(self, tenants=None):
        self.keyed_calls = []
        self.rpc_calls = []
        self._tenants = tenants or [
            {"id": "t-1", "name": "Acme", "slug": "acme", "kind": "company"},
        ]

    def keyed(self, method, path, key, json_body=None):
        self.keyed_calls.append((method, path, key, json_body))
        return {"ingested": len(json_body.get("events", []))}

    def db(self, view, **kw):
        return list(self._tenants)

    def rpc(self, fn, args):
        self.rpc_calls.append((fn, args))
        return True


def test_ingest_trobs_goes_to_observe(tmp_path):
    f = tmp_path / "ev.json"
    f.write_text(json.dumps([{"input": "a"}, {"input": "b"}]))
    c = _FakeClient()
    cli._cmd_ingest(c, Namespace(key="trobs_x", events=str(f), source="cli", json=False))
    method, path, key, body = c.keyed_calls[0]
    assert path == "/v1/observe"
    assert "source" not in body and body["events"][0]["input"] == "a"


def test_ingest_trc_goes_to_ingest(tmp_path):
    f = tmp_path / "ev.json"
    f.write_text(json.dumps([{"input": "a"}]))
    c = _FakeClient()
    cli._cmd_ingest(c, Namespace(key="trc_x", events=str(f), source="myapp", json=False))
    method, path, key, body = c.keyed_calls[0]
    assert path == "/v1/ingest"
    assert body["source"] == "myapp"


# ----- tenant-delete guards -------------------------------------------------- #
def test_tenant_delete_refuses_personal():
    c = _FakeClient(tenants=[{"id": "p1", "name": "Me", "slug": "me", "kind": "personal"}])
    with pytest.raises(CloudError) as ei:
        cli._cmd_tenant_delete(c, Namespace(tenant="me", confirm="Me", json=False))
    assert "personal" in str(ei.value).lower()
    assert c.rpc_calls == []  # never reached the RPC


def test_tenant_delete_refuses_wrong_confirm():
    c = _FakeClient(tenants=[{"id": "t1", "name": "Acme", "slug": "acme", "kind": "company"}])
    with pytest.raises(CloudError) as ei:
        cli._cmd_tenant_delete(c, Namespace(tenant="acme", confirm="wrong", json=False))
    assert "confirm" in str(ei.value).lower()
    assert c.rpc_calls == []


def test_tenant_delete_calls_rpc_when_confirmed():
    c = _FakeClient(tenants=[{"id": "t1", "name": "Acme", "slug": "acme", "kind": "company"}])
    cli._cmd_tenant_delete(c, Namespace(tenant="acme", confirm="Acme", json=False))
    assert c.rpc_calls == [("delete_company_tenant", {"_tenant": "t1"})]


# ----- _table doesn't crash on empty ----------------------------------------- #
def test_table_empty(capsys):
    cli._table([], ["a", "b"])
    out = capsys.readouterr().out
    assert "none" in out.lower()
