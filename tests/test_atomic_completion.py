"""
Atomic clarification completion safety tests (Phase 3 final).

Covers: successful atomic completion, rollback on failures at each step,
concurrency/duplicate prevention, expiry, and session isolation.
"""

from __future__ import annotations

import contextlib
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from app import db
from app.intent import resolve_clarification


def _seed_pending_request(
    session_id="sess-1",
    tool="create_order",
    args=None,
    missing=None,
):
    if args is None:
        args = {"amount": "10.00", "description": "consulting services"}
    if missing is None:
        missing = ["currency"]
    return db.create_pending_request(session_id, tool, args, missing)


class _FailingConnection:
    """Proxy around a real sqlite3 connection that raises on matching SQL."""

    def __init__(self, real, match_substring: str, exc=None):
        self._real = real
        self._match = match_substring
        self._exc = exc or db.sqlite3.OperationalError("forced failure")

    def execute(self, sql, *params):
        if isinstance(sql, str) and self._match in sql:
            raise self._exc
        return self._real.execute(sql, *params)

    def executemany(self, sql, *params):
        return self._real.executemany(sql, *params)

    def executescript(self, sql):
        return self._real.executescript(sql)

    def __enter__(self):
        self._real.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._real.__exit__(exc_type, exc, tb)

    def __getattr__(self, name):
        return getattr(self._real, name)


@contextlib.contextmanager
def _failing_db(match_substring: str):
    real = db._connect()
    wrapped = _FailingConnection(real, match_substring)
    try:
        with wrapped._real:  # commit/rollback on real connection
            yield wrapped
    finally:
        wrapped._real.close()


# --------------------------------------------------------------------------- #
# Atomic completion basics
# --------------------------------------------------------------------------- #

def test_atomic_completion_success(env):
    req = _seed_pending_request()
    result = db.complete_unresolved_request(
        req["id"], "sess-1", "create_order",
        {"amount": "10.00", "currency": "USD", "description": "consulting services"},
    )
    assert result["status"] == "completed"
    action = result["action"]
    assert action["id"]
    assert action["tool"] == "create_order"
    assert action["status"] == "pending"

    actions = db.list_pending_actions("pending")
    assert len(actions) == 1
    assert actions[0]["id"] == action["id"]

    audits = db.list_audit()
    assert len(audits) == 1
    assert audits[0]["request_id"] == action["request_id"]
    assert audits[0]["result"]["action_id"] == action["id"]

    ur = db.get_unresolved_request(req["id"])
    assert ur["status"] == "resolved"
    assert ur["pending_action_id"] == action["id"]


def test_resolve_clarification_uses_atomic_completion(env):
    req = _seed_pending_request()
    intent = resolve_clarification("sess-1", "USD")
    assert intent is not None
    assert intent.classification == "WRITE"
    assert intent.completion_outcome == "completed"
    assert intent.action is not None
    assert intent.action["status"] == "pending"
    assert len(db.list_pending_actions("pending")) == 1
    assert len(db.list_audit()) == 1
    ur = db.get_unresolved_request(req["id"])
    assert ur["status"] == "resolved"
    assert ur["pending_action_id"] == intent.action["id"]


# --------------------------------------------------------------------------- #
# No pending action before completion
# --------------------------------------------------------------------------- #

def test_no_pending_action_before_completion(env):
    _seed_pending_request()
    assert db.list_pending_actions("pending") == []
    assert db.list_audit() == []


# --------------------------------------------------------------------------- #
# Rollback: failure during pending action insert
# --------------------------------------------------------------------------- #

def test_rollback_on_pending_action_insert_failure(env, monkeypatch):
    req = _seed_pending_request()

    @contextlib.contextmanager
    def failing_db():
        with _failing_db("INSERT INTO pending_actions") as conn:
            yield conn

    monkeypatch.setattr(db, "_db", failing_db)
    with pytest.raises(db.sqlite3.OperationalError):
        db.complete_unresolved_request(
            req["id"], "sess-1", "create_order",
            {"amount": "10.00", "currency": "USD", "description": "consulting services"},
        )

    assert db.list_pending_actions() == []
    assert db.list_audit() == []
    ur = db.get_unresolved_request(req["id"])
    assert ur["status"] == "awaiting_clarification"
    assert ur.get("pending_action_id") is None


# --------------------------------------------------------------------------- #
# Rollback: failure during audit insert
# --------------------------------------------------------------------------- #

def test_rollback_on_audit_insert_failure(env, monkeypatch):
    req = _seed_pending_request()

    @contextlib.contextmanager
    def failing_db():
        with _failing_db("INSERT INTO audit_log") as conn:
            yield conn

    monkeypatch.setattr(db, "_db", failing_db)
    with pytest.raises(db.sqlite3.OperationalError):
        db.complete_unresolved_request(
            req["id"], "sess-1", "create_order",
            {"amount": "10.00", "currency": "USD", "description": "consulting services"},
        )

    assert db.list_pending_actions() == []
    assert db.list_audit() == []
    ur = db.get_unresolved_request(req["id"])
    assert ur["status"] == "awaiting_clarification"
    assert ur.get("pending_action_id") is None


# --------------------------------------------------------------------------- #
# Rollback: failure during unresolved-request update (claim)
# --------------------------------------------------------------------------- #

def test_rollback_on_unresolved_update_failure(env, monkeypatch):
    req = _seed_pending_request()

    @contextlib.contextmanager
    def failing_db():
        # Only fail the claim/update that sets pending_action_id (not expiry updates)
        with _failing_db("pending_action_id = ?") as conn:
            yield conn

    monkeypatch.setattr(db, "_db", failing_db)
    with pytest.raises(db.sqlite3.OperationalError):
        db.complete_unresolved_request(
            req["id"], "sess-1", "create_order",
            {"amount": "10.00", "currency": "USD", "description": "consulting services"},
        )

    assert db.list_pending_actions() == []
    assert db.list_audit() == []
    ur = db.get_unresolved_request(req["id"])
    assert ur["status"] == "awaiting_clarification"
    assert ur.get("pending_action_id") is None


# --------------------------------------------------------------------------- #
# Duplicate / repeated completion
# --------------------------------------------------------------------------- #

def test_repeated_completion_same_request(env):
    req = _seed_pending_request()
    r1 = db.complete_unresolved_request(
        req["id"], "sess-1", "create_order",
        {"amount": "10.00", "currency": "USD", "description": "consulting services"},
    )
    assert r1["status"] == "completed"
    action_id = r1["action"]["id"]

    r2 = db.complete_unresolved_request(
        req["id"], "sess-1", "create_order",
        {"amount": "10.00", "currency": "USD", "description": "consulting services"},
    )
    assert r2["status"] in {"already_completed", "conflict"}
    assert r2.get("pending_action_id") == action_id or r2.get("pending_action_id") is not None
    assert len(db.list_pending_actions("pending")) == 1
    assert len(db.list_audit()) == 1


def test_concurrent_completion_attempts(env):
    req = _seed_pending_request()
    barrier = threading.Barrier(2)
    validated = {"amount": "10.00", "currency": "USD", "description": "consulting services"}

    def attempt():
        barrier.wait(timeout=5)
        try:
            return db.complete_unresolved_request(
                req["id"], "sess-1", "create_order", validated
            )
        except Exception as e:
            return {"status": "error", "error": str(e)}

    with ThreadPoolExecutor(max_workers=2) as ex:
        results = list(ex.map(lambda _: attempt(), range(2)))

    statuses = [r["status"] for r in results]
    assert statuses.count("completed") == 1
    assert sum(1 for s in statuses if s in {"already_completed", "conflict"}) == 1
    assert len(db.list_pending_actions("pending")) == 1
    assert len(db.list_audit()) == 1


# --------------------------------------------------------------------------- #
# Session isolation and independent requests
# --------------------------------------------------------------------------- #

def test_different_sessions_independent_requests(env):
    r1 = _seed_pending_request(session_id="s-a", args={"amount": "10.00"}, missing=["currency"])
    r2 = _seed_pending_request(session_id="s-b", args={"amount": "20.00"}, missing=["currency"])

    c1 = db.complete_unresolved_request(
        r1["id"], "s-a", "create_order",
        {"amount": "10.00", "currency": "USD", "description": "PayPilot AI order"},
    )
    c2 = db.complete_unresolved_request(
        r2["id"], "s-b", "create_order",
        {"amount": "20.00", "currency": "EUR", "description": "PayPilot AI order"},
    )
    assert c1["status"] == "completed"
    assert c2["status"] == "completed"
    assert c1["action"]["id"] != c2["action"]["id"]
    assert len(db.list_pending_actions("pending")) == 2
    assert len(db.list_audit()) == 2

    u1 = db.get_unresolved_request(r1["id"])
    u2 = db.get_unresolved_request(r2["id"])
    assert u1["status"] == "resolved" and u1["pending_action_id"] == c1["action"]["id"]
    assert u2["status"] == "resolved" and u2["pending_action_id"] == c2["action"]["id"]


def test_session_mismatch_rejected(env):
    req = _seed_pending_request(session_id="s-a")
    result = db.complete_unresolved_request(
        req["id"], "s-b", "create_order",
        {"amount": "10.00", "currency": "USD", "description": "x"},
    )
    assert result["status"] == "session_mismatch"
    assert db.list_pending_actions() == []
    assert db.get_unresolved_request(req["id"])["status"] == "awaiting_clarification"


# --------------------------------------------------------------------------- #
# Expiry
# --------------------------------------------------------------------------- #

def test_expired_request_during_completion(env):
    from app import db as db_module
    with db_module._db() as conn:
        conn.execute(
            """
            INSERT INTO unresolved_requests
                (id, session_id, tool, args, missing, created_at, expires_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'awaiting_clarification')
            """,
            (
                "expired-id",
                "sess-exp",
                "create_order",
                '{"amount": "10.00"}',
                '["currency"]',
                "2000-01-01T00:00:00+00:00",
                "2000-01-01T00:10:00+00:00",
            ),
        )
    result = db.complete_unresolved_request(
        "expired-id", "sess-exp", "create_order",
        {"amount": "10.00", "currency": "USD", "description": "x"},
    )
    assert result["status"] == "expired"
    assert db.list_pending_actions() == []
    assert db.list_audit() == []
    ur = db.get_unresolved_request("expired-id")
    assert ur["status"] == "expired"
