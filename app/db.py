"""SQLite persistence: transactions, pending actions and the audit log."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .config import get_settings

_LOCK = threading.RLock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _db_path() -> Path:
    path = get_settings().db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path()), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def _db():
    """Commit/rollback and always close the SQLite connection."""
    conn = _connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _ensure_column(
    conn: sqlite3.Connection, table: str, column: str, ddl: str
) -> None:
    columns = {
        row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def init_db() -> None:
    with _LOCK:
        with _db() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    paypal_id TEXT,
                    parent_id TEXT,
                    status TEXT,
                    amount TEXT,
                    currency TEXT,
                    description TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS pending_actions (
                    id TEXT PRIMARY KEY,
                    tool TEXT NOT NULL,
                    args TEXT NOT NULL,
                    request_id TEXT,
                    baseline TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    resolved_at TEXT,
                    result TEXT,
                    error TEXT
                );

                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    tool TEXT NOT NULL,
                    request_id TEXT,
                    args TEXT,
                    result TEXT,
                    error TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_transactions_created
                    ON transactions(created_at);
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp
                    ON audit_log(timestamp);
                CREATE INDEX IF NOT EXISTS idx_pending_status
                    ON pending_actions(status);
                """
            )
            # Migration for databases created before idempotency was added.
            _ensure_column(conn, "pending_actions", "request_id", "TEXT")
            _ensure_column(conn, "pending_actions", "baseline", "TEXT")
            _ensure_column(conn, "audit_log", "request_id", "TEXT")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pending_request "
                "ON pending_actions(request_id)"
            )


def _json(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, default=str)


def _loads(value: Optional[str]) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


# --------------------------------------------------------------------------- #
# Transactions
# --------------------------------------------------------------------------- #
def add_transaction(
    kind: str,
    paypal_id: Optional[str],
    status: Optional[str],
    amount: Optional[str] = None,
    currency: Optional[str] = None,
    description: Optional[str] = None,
    parent_id: Optional[str] = None,
) -> int:
    with _LOCK, _db() as conn:
        cur = conn.execute(
            """
            INSERT INTO transactions
                (kind, paypal_id, parent_id, status, amount, currency, description, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (kind, paypal_id, parent_id, status, amount, currency, description, _now()),
        )
        return int(cur.lastrowid)


def update_transaction_status(paypal_id: str, status: str) -> None:
    with _LOCK, _db() as conn:
        conn.execute(
            "UPDATE transactions SET status = ? WHERE paypal_id = ?",
            (status, paypal_id),
        )


def list_transactions(limit: int = 200) -> list[dict]:
    with _LOCK, _db() as conn:
        rows = conn.execute(
            "SELECT * FROM transactions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


# --------------------------------------------------------------------------- #
# Pending actions
# --------------------------------------------------------------------------- #
def create_pending_action(
    tool: str, args: dict, request_id: Optional[str] = None
) -> dict:
    action_id = uuid.uuid4().hex
    request_id = request_id or uuid.uuid4().hex
    now = _now()
    with _LOCK, _db() as conn:
        conn.execute(
            """
            INSERT INTO pending_actions
                (id, tool, args, request_id, status, created_at)
            VALUES (?, ?, ?, ?, 'pending', ?)
            """,
            (action_id, tool, _json(args), request_id, now),
        )
    return {
        "id": action_id,
        "tool": tool,
        "args": args,
        "request_id": request_id,
        "baseline": None,
        "status": "pending",
        "created_at": now,
        "resolved_at": None,
        "result": None,
        "error": None,
    }


def _row_to_action(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "tool": row["tool"],
        "args": _loads(row["args"]) or {},
        "request_id": row["request_id"],
        "baseline": _loads(row["baseline"]),
        "status": row["status"],
        "created_at": row["created_at"],
        "resolved_at": row["resolved_at"],
        "result": _loads(row["result"]),
        "error": _loads(row["error"]),
    }


def get_pending_action(action_id: str) -> Optional[dict]:
    with _LOCK, _db() as conn:
        row = conn.execute(
            "SELECT * FROM pending_actions WHERE id = ?", (action_id,)
        ).fetchone()
    return _row_to_action(row) if row else None


def list_pending_actions(status: Optional[str] = "pending", limit: int = 100) -> list[dict]:
    with _LOCK, _db() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM pending_actions WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM pending_actions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [_row_to_action(row) for row in rows]


def claim_pending_action(action_id: str) -> bool:
    """Atomically move a pending action to executing.

    The status guard is the single source of truth: exactly one caller can win
    the UPDATE, which prevents double approval and duplicate PayPal writes.
    """
    with _LOCK, _db() as conn:
        cur = conn.execute(
            """
            UPDATE pending_actions
               SET status = 'executing'
             WHERE id = ? AND status = 'pending'
            """,
            (action_id,),
        )
        return cur.rowcount == 1


def release_pending_action(action_id: str) -> None:
    """Return an unexecuted action to the queue (e.g. validation failed)."""
    with _LOCK, _db() as conn:
        conn.execute(
            """
            UPDATE pending_actions
               SET status = 'pending', resolved_at = NULL, result = NULL, error = NULL
             WHERE id = ? AND status = 'executing'
            """,
            (action_id,),
        )


def set_action_baseline(action_id: str, baseline: Any) -> None:
    """Store the live pre-execution snapshot used to reconcile ambiguity later."""
    with _LOCK, _db() as conn:
        conn.execute(
            "UPDATE pending_actions SET baseline = ? WHERE id = ?",
            (_json(baseline), action_id),
        )


def resolve_pending_action(
    action_id: str,
    status: str,
    result: Any = None,
    error: Any = None,
) -> None:
    with _LOCK, _db() as conn:
        conn.execute(
            """
            UPDATE pending_actions
               SET status = ?, resolved_at = ?, result = ?, error = ?
             WHERE id = ?
            """,
            (status, _now(), _json(result), _json(error), action_id),
        )


def resolve_and_audit(
    action_id: str,
    status: str,
    actor: str,
    tool: str,
    args: Any = None,
    result: Any = None,
    error: Any = None,
    request_id: Optional[str] = None,
) -> None:
    """Resolve a pending action and write its audit record in one transaction.

    The status update and the audit row commit together so an observer can never
    see an executed action without its matching audit entry (or vice versa).
    """
    with _LOCK, _db() as conn:
        conn.execute(
            """
            UPDATE pending_actions
               SET status = ?, resolved_at = ?, result = ?, error = ?
             WHERE id = ?
            """,
            (status, _now(), _json(result), _json(error), action_id),
        )
        conn.execute(
            """
            INSERT INTO audit_log
                (timestamp, actor, tool, request_id, args, result, error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (_now(), actor, tool, request_id, _json(args), _json(result), _json(error)),
        )


# --------------------------------------------------------------------------- #
# Audit log
# --------------------------------------------------------------------------- #
def log_audit(
    actor: str,
    tool: str,
    args: Any = None,
    result: Any = None,
    error: Any = None,
    request_id: Optional[str] = None,
) -> int:
    with _LOCK, _db() as conn:
        cur = conn.execute(
            """
            INSERT INTO audit_log
                (timestamp, actor, tool, request_id, args, result, error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (_now(), actor, tool, request_id, _json(args), _json(result), _json(error)),
        )
        return int(cur.lastrowid)


def list_audit(limit: int = 200) -> list[dict]:
    with _LOCK, _db() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    entries = []
    for row in rows:
        item = dict(row)
        item["args"] = _loads(item.get("args"))
        item["result"] = _loads(item.get("result"))
        item["error"] = _loads(item.get("error"))
        entries.append(item)
    return entries

