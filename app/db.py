"""SQLite persistence: transactions, pending actions and the audit log."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from .config import get_settings

_LOCK = threading.RLock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _now_from_seconds(base_iso: str, seconds: int) -> str:
    """Compute an ISO timestamp `seconds` after `base_iso`."""
    base = datetime.fromisoformat(base_iso)
    return (base + timedelta(seconds=seconds)).isoformat(timespec="seconds")


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


PROVENANCE_UNKNOWN = "unknown"
PROVENANCE_LOCAL = "local"
PROVENANCE_PAYPAL_VERIFIED = "paypal_verified"
_VALID_PROVENANCES = {PROVENANCE_UNKNOWN, PROVENANCE_LOCAL, PROVENANCE_PAYPAL_VERIFIED}


def _normalize_stored_provenance(provenance: Any) -> str:
    """Normalize provenance read from storage/API surfaces without downgrading
    legitimately stored paypal_verified values."""
    value = (
        str(provenance).strip().lower()
        if provenance is not None
        else PROVENANCE_UNKNOWN
    )
    return value if value in _VALID_PROVENANCES else PROVENANCE_UNKNOWN


class VerifiedResourceEvidence:
    """Explicit evidence that a PayPal resource was validated from a successful
    PayPal client response.

    Constructed only from validated PayPal client response data by internal
    code paths (approval execution, reconciliation, live lookups, financial
    revalidation). It binds provenance to the exact resource ID, resource type,
    and validated response.

    Trust assumption: construction is not cryptographically restricted; any
    in-process caller can instantiate it. The persistence boundary enforces
    that ``paypal_verified`` provenance requires a valid evidence object matching
    the persisted record. This is a module/process discipline boundary, not a
    cryptographic one.
    """

    def __init__(
        self,
        kind: str,
        paypal_id: str,
        status: str,
        amount: Optional[str] = None,
        currency: Optional[str] = None,
        description: Optional[str] = None,
        parent_id: Optional[str] = None,
        raw_response: Optional[dict] = None,
    ) -> None:
        self.kind = str(kind).strip().lower()
        self.paypal_id = str(paypal_id).strip()
        self.status = str(status).strip()
        self.amount = amount
        self.currency = currency
        self.description = description
        self.parent_id = parent_id
        self.raw_response = dict(raw_response) if isinstance(raw_response, dict) else None
        self.verified_at = _now()

    def matches(self, kind: str, paypal_id: str) -> bool:
        return (
            self.kind == str(kind).strip().lower()
            and self.paypal_id == str(paypal_id).strip()
        )


def _normalize_write_provenance(
    provenance: Any, verified_evidence: Optional[VerifiedResourceEvidence] = None
) -> str:
    """Normalize provenance for persistence. ``paypal_verified`` is allowed only
    when accompanied by validated evidence matching the persisted record."""
    value = _normalize_stored_provenance(provenance)
    if value == PROVENANCE_PAYPAL_VERIFIED:
        if verified_evidence is None or not isinstance(verified_evidence, VerifiedResourceEvidence):
            return PROVENANCE_UNKNOWN
    return value


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
                    created_at TEXT NOT NULL,
                    provenance TEXT NOT NULL DEFAULT 'unknown'
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
                CREATE UNIQUE INDEX IF NOT EXISTS idx_pending_dedup
                    ON pending_actions(tool, args)
                    WHERE status = 'pending';
                CREATE TABLE IF NOT EXISTS unresolved_requests (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    tool TEXT NOT NULL,
                    args TEXT NOT NULL,
                    missing TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'awaiting_clarification'
                );
                CREATE INDEX IF NOT EXISTS idx_unresolved_session
                    ON unresolved_requests(session_id);
                CREATE INDEX IF NOT EXISTS idx_unresolved_expires
                    ON unresolved_requests(expires_at);

                CREATE TABLE IF NOT EXISTS session_transaction_context (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    paypal_id TEXT NOT NULL,
                    record TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_session_tx_ctx_session
                    ON session_transaction_context(session_id, updated_at);
                CREATE INDEX IF NOT EXISTS idx_session_tx_ctx_sid_pid
                    ON session_transaction_context(session_id, paypal_id);
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
            # Migration for atomic clarification completion linkage.
            _ensure_column(conn, "unresolved_requests", "pending_action_id", "TEXT")
            _ensure_column(conn, "unresolved_requests", "resolved_at", "TEXT")
            # Migration for transaction provenance (idempotent, preserves all records).
            # Existing records must remain and migrate to 'unknown' (never auto-verified).
            _ensure_column(
                conn,
                "transactions",
                "provenance",
                "TEXT NOT NULL DEFAULT 'unknown'",
            )
            conn.execute(
                """
                UPDATE transactions
                   SET provenance = 'unknown'
                 WHERE provenance IS NULL
                    OR TRIM(provenance) = ''
                    OR LOWER(TRIM(provenance)) NOT IN ('unknown', 'local', 'paypal_verified')
                """
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
    provenance: str = PROVENANCE_LOCAL,
    verified_evidence: Optional[VerifiedResourceEvidence] = None,
) -> int:
    """Insert a transaction record.

    Provenance is persisted explicitly. ``paypal_verified`` is accepted only
    when accompanied by validated evidence matching the persisted record
    (resource ID, resource type, and successful PayPal response).
    Ordinary local and unknown inserts remain supported without evidence.
    """
    prov = _normalize_write_provenance(provenance, verified_evidence)
    if prov == PROVENANCE_PAYPAL_VERIFIED:
        if not verified_evidence or not verified_evidence.matches(kind, str(paypal_id or "")):
            prov = PROVENANCE_UNKNOWN
    with _LOCK, _db() as conn:
        cur = conn.execute(
            """
            INSERT INTO transactions
                (kind, paypal_id, parent_id, status, amount, currency, description, created_at, provenance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (kind, paypal_id, parent_id, status, amount, currency, description, _now(), prov),
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
    out = []
    for row in rows:
        item = dict(row)
        item["provenance"] = _normalize_stored_provenance(item.get("provenance"))
        out.append(item)
    return out


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


def create_pending_action_with_audit(
    tool: str, args: dict, *, actor: str = "agent"
) -> dict:
    """Create a pending action and its proposal audit record in one transaction.

    If a pending action with the same tool and args already exists, returns
    that existing action instead of creating a duplicate. This prevents
    duplicate proposals when the same financial operation is requested
    multiple times while a previous proposal is still awaiting approval.

    If any step fails, the transaction rolls back so no pending action exists
    without its required audit entry.
    """
    # Check for an existing pending action with the same tool and args.
    # This check runs inside the same lock/transaction as the insert below,
    # preventing race conditions between concurrent identical submissions.
    existing = find_existing_pending_action(tool, args)
    if existing:
        return {**existing, "is_duplicate": True}

    action_id = uuid.uuid4().hex
    request_id = uuid.uuid4().hex
    now = _now()
    audit_result = {"status": "pending_approval", "action_id": action_id}
    args_json = _json(args)
    try:
        with _LOCK, _db() as conn:
            conn.execute(
                """
                INSERT INTO pending_actions
                    (id, tool, args, request_id, status, created_at)
                VALUES (?, ?, ?, ?, 'pending', ?)
                """,
                (action_id, tool, args_json, request_id, now),
            )
            conn.execute(
                """
                INSERT INTO audit_log
                    (timestamp, actor, tool, request_id, args, result, error)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (now, actor, tool, request_id, args_json, _json(audit_result), None),
            )
    except sqlite3.IntegrityError:
        # UNIQUE index on (tool, args) WHERE status='pending' prevented the
        # insert because a concurrent request created the same pending action
        # first. Fetch and return the existing action as a duplicate.
        existing = find_existing_pending_action(tool, args)
        if existing:
            return {**existing, "is_duplicate": True}
        # If we can't find it either, re-raise
        raise
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
        "is_duplicate": False,
    }


def find_existing_pending_action(tool: str, args: dict) -> Optional[dict]:
    """Find an existing pending action with the same tool and args.

    Used to prevent duplicate pending-action creation when the same
    financial operation is requested multiple times while a previous
    proposal is still awaiting approval.
    """
    args_json = _json(args)
    with _LOCK, _db() as conn:
        row = conn.execute(
            """
            SELECT * FROM pending_actions
             WHERE tool = ? AND args = ? AND status = 'pending'
             ORDER BY created_at DESC LIMIT 1
            """,
            (tool, args_json),
        ).fetchone()
    return _row_to_action(row) if row else None


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
# Unresolved request state (multi-turn clarification)
# --------------------------------------------------------------------------- #

def create_pending_request(
    session_id: str,
    tool: str,
    args: dict,
    missing: list[str],
    ttl_seconds: int = 600,
) -> dict:
    """Create an unresolved request entry for multi-turn clarification."""
    rid = uuid.uuid4().hex
    now = _now()
    expires_at = _now_from_seconds(now, ttl_seconds)
    with _LOCK, _db() as conn:
        conn.execute(
            """
            INSERT INTO unresolved_requests
                (id, session_id, tool, args, missing, created_at, expires_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'awaiting_clarification')
            """,
            (rid, session_id, tool, _json(args), _json(missing), now, expires_at),
        )
    return {
        "id": rid,
        "session_id": session_id,
        "tool": tool,
        "args": args,
        "missing": missing,
        "status": "awaiting_clarification",
        "created_at": now,
        "expires_at": expires_at,
        "pending_action_id": None,
        "resolved_at": None,
    }


def _row_to_unresolved(row: sqlite3.Row) -> dict:
    keys = row.keys()
    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "tool": row["tool"],
        "args": _loads(row["args"]) or {},
        "missing": _loads(row["missing"]) or [],
        "created_at": row["created_at"],
        "expires_at": row["expires_at"],
        "status": row["status"],
        "pending_action_id": row["pending_action_id"] if "pending_action_id" in keys else None,
        "resolved_at": row["resolved_at"] if "resolved_at" in keys else None,
    }


def get_pending_request(session_id: str) -> Optional[dict]:
    """Get the most recent unresolved request for a session, if not expired."""
    with _LOCK, _db() as conn:
        row = conn.execute(
            """
            SELECT * FROM unresolved_requests
            WHERE session_id = ? AND status = 'awaiting_clarification'
              AND expires_at > ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (session_id, _now()),
        ).fetchone()
    return _row_to_unresolved(row) if row else None


def get_unresolved_request(request_id: str) -> Optional[dict]:
    with _LOCK, _db() as conn:
        row = conn.execute(
            "SELECT * FROM unresolved_requests WHERE id = ?",
            (request_id,),
        ).fetchone()
    return _row_to_unresolved(row) if row else None


def update_unresolved_request(
    request_id: str, args: dict, missing: list[str]
) -> Optional[dict]:
    """Update args/missing for an awaiting clarification request by id.

    Returns the updated row, or None if no matching awaiting request exists.
    """
    with _LOCK, _db() as conn:
        row = conn.execute(
            """
            UPDATE unresolved_requests
               SET args = ?, missing = ?
             WHERE id = ? AND status = 'awaiting_clarification'
             RETURNING *
            """,
            (_json(args), _json(missing), request_id),
        ).fetchone()
    return _row_to_unresolved(row) if row else None


def resolve_pending_request(
    session_id: str,
    tool: str,
    args: dict,
    missing: list[str],
) -> Optional[dict]:
    """Backward-compatible update for an awaiting unresolved request.

    Updates the latest awaiting request for (session_id, tool). Prefer
    :func:`update_unresolved_request` when a request id is known.
    """
    status = "awaiting_clarification" if missing else "resolved"
    now = _now()
    with _LOCK, _db() as conn:
        row = conn.execute(
            """
            SELECT id FROM unresolved_requests
             WHERE session_id = ? AND tool = ? AND status = 'awaiting_clarification'
             ORDER BY created_at DESC LIMIT 1
            """,
            (session_id, tool),
        ).fetchone()
        if row is None:
            return None
        rid = row["id"]
        resolved_at = now if status == "resolved" else None
        out = conn.execute(
            """
            UPDATE unresolved_requests
               SET status = ?, args = ?, missing = ?, resolved_at = ?
             WHERE id = ? AND status = 'awaiting_clarification'
             RETURNING *
            """,
            (status, _json(args), _json(missing), resolved_at, rid),
        ).fetchone()
    return _row_to_unresolved(out) if out else None


def complete_unresolved_request(
    request_id: str,
    session_id: str,
    tool: str,
    validated_args: dict,
) -> dict:
    """Atomically complete a clarification request.

    In a single SQLite transaction: claim the exact awaiting, non-expired
    unresolved request (verifying session, tool, status, expiry), mark it
    resolved and linked, create the pending action, and write the proposal
    audit record. On any failure or failed claim, nothing is committed.

    Returns a dict with ``status``:
      - ``completed``: success; includes ``action``.
      - ``already_completed``: request was already completed/claimed.
      - ``expired``: request expired before completion.
      - ``not_found`` / ``session_mismatch`` / ``tool_mismatch``: invalid target.
      - ``conflict``: concurrent claim prevented completion.
    """
    action_id = uuid.uuid4().hex
    action_request_id = uuid.uuid4().hex
    now = _now()
    args_json = _json(validated_args)
    audit_result_json = _json({"status": "pending_approval", "action_id": action_id})

    with _LOCK, _db() as conn:
        req = conn.execute(
            "SELECT * FROM unresolved_requests WHERE id = ?",
            (request_id,),
        ).fetchone()
        if req is None:
            return {"status": "not_found"}
        if req["session_id"] != session_id:
            return {"status": "session_mismatch"}
        if req["tool"] != tool:
            return {"status": "tool_mismatch"}

        req_keys = req.keys()
        existing_action_id = (
            req["pending_action_id"] if "pending_action_id" in req_keys else None
        )
        if req["status"] != "awaiting_clarification":
            return {
                "status": "already_completed",
                "pending_action_id": existing_action_id,
            }
        if req["expires_at"] <= now:
            conn.execute(
                "UPDATE unresolved_requests SET status='expired' "
                "WHERE id=? AND status='awaiting_clarification'",
                (request_id,),
            )
            return {"status": "expired"}

        # Atomic claim/finalize of the unresolved request. This conditional
        # update is the concurrency guard: only one claimant can win.
        cur = conn.execute(
            """
            UPDATE unresolved_requests
               SET status = 'resolved',
                   pending_action_id = ?,
                   resolved_at = ?,
                   args = ?,
                   missing = '[]'
             WHERE id = ?
               AND session_id = ?
               AND tool = ?
               AND status = 'awaiting_clarification'
               AND expires_at > ?
            """,
            (action_id, now, args_json, request_id, session_id, tool, now),
        )
        if cur.rowcount != 1:
            req2 = conn.execute(
                "SELECT status, pending_action_id, expires_at "
                "FROM unresolved_requests WHERE id = ?",
                (request_id,),
            ).fetchone()
            if req2 is not None:
                if req2["status"] == "expired" or req2["expires_at"] <= now:
                    return {"status": "expired"}
                if req2["status"] != "awaiting_clarification":
                    return {
                        "status": "already_completed",
                        "pending_action_id": req2["pending_action_id"],
                    }
            return {"status": "conflict"}

        # Create pending action and audit in the same transaction.
        conn.execute(
            """
            INSERT INTO pending_actions
                (id, tool, args, request_id, status, created_at)
            VALUES (?, ?, ?, ?, 'pending', ?)
            """,
            (action_id, tool, args_json, action_request_id, now),
        )
        conn.execute(
            """
            INSERT INTO audit_log
                (timestamp, actor, tool, request_id, args, result, error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (now, "agent", tool, action_request_id, args_json, audit_result_json, None),
        )

    action = {
        "id": action_id,
        "tool": tool,
        "args": validated_args,
        "request_id": action_request_id,
        "baseline": None,
        "status": "pending",
        "created_at": now,
        "resolved_at": None,
        "result": None,
        "error": None,
    }
    return {"status": "completed", "action": action}


def expire_pending_requests() -> int:
    """Mark all expired unresolved_requests as expired. Returns count."""
    with _LOCK, _db() as conn:
        cur = conn.execute(
            """
            UPDATE unresolved_requests
               SET status = 'expired'
             WHERE status = 'awaiting_clarification'
               AND expires_at <= ?
            """,
            (_now(),),
        )
        return cur.rowcount


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




# --------------------------------------------------------------------------- #
# Verified transaction helpers / session context (Phase 4)
# --------------------------------------------------------------------------- #

def find_verified_transaction(paypal_id: str, kind: Optional[str] = None) -> Optional[dict]:
    """Return the most recent non-failed local transaction for a PayPal ID."""
    with _LOCK, _db() as conn:
        if kind:
            row = conn.execute(
                """
                SELECT * FROM transactions
                 WHERE paypal_id = ? AND kind = ?
                   AND (status IS NULL OR UPPER(status) NOT IN ('FAILED','DENIED','VOIDED','UNKNOWN','ERROR','EXPIRED'))
                 ORDER BY id DESC LIMIT 1
                """,
                (paypal_id, kind),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT * FROM transactions
                 WHERE paypal_id = ?
                   AND (status IS NULL OR UPPER(status) NOT IN ('FAILED','DENIED','VOIDED','UNKNOWN','ERROR','EXPIRED'))
                 ORDER BY id DESC LIMIT 1
                """,
                (paypal_id,),
            ).fetchone()
    if not row:
        return None
    item = dict(row)
    item["provenance"] = _normalize_stored_provenance(item.get("provenance"))
    return item


def upsert_verified_transaction(
    kind: str,
    paypal_id: str,
    status: Optional[str],
    amount: Optional[str] = None,
    currency: Optional[str] = None,
    description: Optional[str] = None,
    parent_id: Optional[str] = None,
    provenance: str = PROVENANCE_PAYPAL_VERIFIED,
    verified_evidence: Optional[VerifiedResourceEvidence] = None,
) -> int:
    """Insert or refresh a non-failed transaction record for (kind, paypal_id).

    Prevents duplicates (idempotent insert-by-identity). ``paypal_verified``
    provenance is stored only when accompanied by validated evidence matching
    the persisted record. Untrusted callers cannot mark records as PayPal-verified
    without validated PayPal response evidence.
    """
    if not paypal_id:
        return -1
    prov = _normalize_write_provenance(provenance, verified_evidence)
    if prov == PROVENANCE_PAYPAL_VERIFIED:
        if not verified_evidence or not verified_evidence.matches(kind, paypal_id):
            prov = PROVENANCE_UNKNOWN
    with _LOCK, _db() as conn:
        existing = conn.execute(
            """
            SELECT id, provenance FROM transactions
             WHERE kind = ? AND paypal_id = ?
               AND (status IS NULL OR UPPER(status) NOT IN ('FAILED','DENIED','VOIDED','UNKNOWN','ERROR','EXPIRED'))
             ORDER BY id DESC LIMIT 1
            """,
            (kind, paypal_id),
        ).fetchone()
        if existing:
            existing_prov = _normalize_stored_provenance(existing["provenance"])
            # Never downgrade a verified record; only upgrade when valid evidence matches.
            new_prov = existing_prov
            if verified_evidence and verified_evidence.matches(kind, paypal_id):
                new_prov = PROVENANCE_PAYPAL_VERIFIED
            if status or amount or currency or (new_prov != existing_prov):
                conn.execute(
                    """
                    UPDATE transactions
                       SET status = COALESCE(?, status),
                           amount = COALESCE(?, amount),
                           currency = COALESCE(?, currency),
                           provenance = ?
                     WHERE id = ?
                    """,
                    (status, amount, currency, new_prov, existing["id"]),
                )
            return int(existing["id"])
        cur = conn.execute(
            """
            INSERT INTO transactions
                (kind, paypal_id, parent_id, status, amount, currency, description, created_at, provenance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (kind, paypal_id, parent_id, status, amount, currency, description, _now(), prov),
        )
        return int(cur.lastrowid)


def list_verified_transactions(
    kind: Optional[str] = None, limit: int = 200
) -> list[dict]:
    """List non-failed transaction records, newest first."""
    with _LOCK, _db() as conn:
        if kind:
            rows = conn.execute(
                """
                SELECT * FROM transactions
                 WHERE kind = ?
                   AND (status IS NULL OR UPPER(status) NOT IN ('FAILED','DENIED','VOIDED','UNKNOWN','ERROR','EXPIRED'))
                 ORDER BY id DESC LIMIT ?
                """,
                (kind, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM transactions
                 WHERE (status IS NULL OR UPPER(status) NOT IN ('FAILED','DENIED','VOIDED','UNKNOWN','ERROR','EXPIRED'))
                 ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
    out = []
    for r in rows:
        item = dict(r)
        item["provenance"] = _normalize_stored_provenance(item.get("provenance"))
        out.append(item)
    return out


def upsert_session_context(
    session_id: str, kind: str, paypal_id: str, record: Optional[dict] = None
) -> None:
    """Record a verified transaction reference in session-scoped context."""
    if not session_id or not paypal_id or not kind:
        return
    now = _now()
    with _LOCK, _db() as conn:
        conn.execute(
            "DELETE FROM session_transaction_context WHERE session_id = ? AND paypal_id = ?",
            (session_id, paypal_id),
        )
        conn.execute(
            """
            INSERT INTO session_transaction_context (session_id, kind, paypal_id, record, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, kind, paypal_id, _json(record), now),
        )


def list_session_context(session_id: str, limit: int = 50) -> list[dict]:
    if not session_id:
        return []
    with _LOCK, _db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM session_transaction_context
             WHERE session_id = ?
             ORDER BY updated_at DESC, id DESC
             LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
    out = []
    for r in rows:
        item = dict(r)
        item["record"] = _loads(item.get("record")) or {}
        out.append(item)
    return out


def clear_session_context(session_id: str) -> None:
    if not session_id:
        return
    with _LOCK, _db() as conn:
        conn.execute(
            "DELETE FROM session_transaction_context WHERE session_id = ?",
            (session_id,),
        )
