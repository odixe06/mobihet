"""Quota report: bounded retry, partial failure, reuse of verified rows, hard timeout."""
import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("kaggle_quota", ROOT / ".agents/skills/kaggle-training-notebook/scripts/kaggle_quota.py")
kq = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kq)
OK = {"ok": True, "gpu_left": 0.0, "gpu_total": 30.0, "gpu_reserved": 0.0, "tpu_left": 0.0, "tpu_total": 0.0, "refresh_at": ""}


def args(**kw):
    a = dict(attempts=3, backoff=5.0, timeout=90.0, connect_timeout=10.0, read_timeout=30.0)
    return types.SimpleNamespace(**{**a, **kw})


def test_retry_then_ok_keeps_zero_quota_distinct():
    calls, sleeps = [], []

    def once(user, a):
        calls.append(user)
        return {"ok": False, "error_type": "ConnectTimeout"} if len(calls) < 3 else dict(OK, user=user)

    r = kq.query("a", args(), sleep=sleeps.append, once=once)
    assert r["status"] == "OK" and r["gpu_left"] == 0.0 and r["attempts"] == 3
    assert len(sleeps) == 2 and 2.5 <= sleeps[0] <= 7.5 and 5 <= sleeps[1] <= 15


def test_exhausted_retries_are_unverified_not_zero():
    r = kq.query("a", args(), sleep=lambda s: None, once=lambda u, a: {"ok": False, "error_type": "ReadTimeout"})
    assert r["status"] == "UNVERIFIED" and r["gpu_left"] is None and r["errors"] == ["ReadTimeout"] * 3


def test_identity_mismatch_is_unverified():
    r = kq.query("a", args(), sleep=lambda s: None, once=lambda u, a: dict(OK, user="b"))
    assert r["status"] == "UNVERIFIED" and "IdentityMismatch" in r["errors"]


def test_partial_failure_then_retry_only_failed(tmp_path, monkeypatch):
    seen, healed = [], []

    def fake_query(user, a, **kw):
        seen.append(user)
        if user == "bad" and not healed:
            return {"status": "UNVERIFIED", "gpu_left": None, "attempts": 3, "errors": ["WallTimeout"], "checked_at": kq.now().isoformat()}
        return {"status": "OK", **{k: OK[k] for k in kq.FIELDS}, "attempts": 1, "checked_at": kq.now().isoformat()}

    monkeypatch.setattr(kq, "query", fake_query)
    monkeypatch.setitem(sys.modules, "kaggle_account", types.SimpleNamespace(known_users=lambda: ["a", "b", "bad"]))
    out = tmp_path / "q.json"
    assert kq.main(["--out", str(out)]) == 3
    rep = json.loads(out.read_text())
    assert rep["failed"] == ["bad"] and rep["verified"] == ["a", "b"] and rep["at"]
    seen.clear(); healed.append(1)
    assert kq.main(["--out", str(out), "--retry-failed"]) == 0
    assert seen == ["bad"] and json.loads(out.read_text())["failed"] == []


def test_wall_timeout_kills_hung_child(monkeypatch, tmp_path):
    hang = tmp_path / "hang.py"
    hang.write_text("import time; time.sleep(30)")
    monkeypatch.setattr(kq.Path, "resolve", lambda self: hang if self.name == "kaggle_quota.py" else Path.absolute(self))
    r = kq.query_once("a", args(timeout=1.0))
    assert r == {"ok": False, "error_type": "WallTimeout"}


def test_status_retry_only_on_query_error(monkeypatch):
    from scripts import watch_kaggle_batch as w
    seq = iter(["QUERY_ERROR", "QUERY_ERROR", "RUNNING"])
    monkeypatch.setattr(w, "_query_once", lambda spec: {"status": next(seq)})
    assert w.query({}, sleep=lambda s: None) == {"status": "RUNNING", "attempts": 3}
    monkeypatch.setattr(w, "_query_once", lambda spec: {"status": "QUERY_ERROR"})
    assert w.query({}, sleep=lambda s: None)["status"] == "QUERY_ERROR"
