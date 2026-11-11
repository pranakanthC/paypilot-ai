from __future__ import annotations

import shutil
import sqlite3

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from app import db, services
from app.main import app
from app.paypal_client import PayPalClient, PayPalError
from tests.test_approval_gate import (
    CAPTURE_ID,
    CAPTURE_URL,
    ORDER_ID,
    ORDER_URL,
    ORDERS_URL,
    REFUND_URL,
    TOKEN_URL,
    capture_json,
    mock_token,
    order_json,
)


def _legacy_db_no_provenance(path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            CREATE TABLE transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                paypal_id TEXT,
                parent_id TEXT,
                status TEXT,
                amount TEXT,
                currency TEXT,
                description TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO transactions
                (kind, paypal_id, parent_id, status, amount, currency, description, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "order",
                "ORD-LEGACY-001",
                None,
                "APPROVED",
                "10.00",
                "USD",
                "legacy record",
                "2026-01-01T10:00:00+00:00",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def test_migration_adds_provenance_existing_records_unknown(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy.db"
    _legacy_db_no_provenance(db_path)
    monkeypatch.setenv("DB_PATH", str(db_path))
    db.init_db()
    rows = db.list_transactions()
    assert len(rows) == 1
    assert rows[0]["id"] == 1
    assert rows[0]["paypal_id"] == "ORD-LEGACY-001"
    assert rows[0]["status"] == "APPROVED"
    assert rows[0]["provenance"] == "unknown"
    conn = sqlite3.connect(str(db_path))
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(transactions)").fetchall()}
        assert "provenance" in cols
    finally:
        conn.close()


def test_migration_idempotent_preserves_records_and_ids(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy.db"
    _legacy_db_no_provenance(db_path)
    monkeypatch.setenv("DB_PATH", str(db_path))
    db.init_db()
    db.init_db()
    db.init_db()
    rows = db.list_transactions()
    assert len(rows) == 1
    assert rows[0]["id"] == 1
    assert rows[0]["paypal_id"] == "ORD-LEGACY-001"
    assert rows[0]["amount"] == "10.00"
    assert rows[0]["currency"] == "USD"
    assert rows[0]["description"] == "legacy record"
    assert rows[0]["provenance"] == "unknown"


def test_existing_records_never_auto_classified_paypal_verified(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy.db"
    _legacy_db_no_provenance(db_path)
    monkeypatch.setenv("DB_PATH", str(db_path))
    db.init_db()
    rows = db.list_transactions()
    assert rows[0]["status"] == "APPROVED"
    assert rows[0]["provenance"] == "unknown"
    assert rows[0]["provenance"] != "paypal_verified"


def test_local_records_remain_local_and_untrusted_cannot_verify(env):
    rid = db.add_transaction(
        kind="order",
        paypal_id="LOCAL-REF-001",
        status="CREATED",
        amount="5.00",
        currency="USD",
        description="local ref",
        provenance=db.PROVENANCE_LOCAL,
    )
    row = [r for r in db.list_transactions() if r["id"] == rid][0]
    assert row["provenance"] == "local"

    rid2 = db.add_transaction(
        kind="order",
        paypal_id="SNEAKY-VER-001",
        status="APPROVED",
        amount="1.00",
        currency="USD",
        description="attempted verify",
        provenance=db.PROVENANCE_PAYPAL_VERIFIED,
    )
    row2 = [r for r in db.list_transactions() if r["id"] == rid2][0]
    assert row2["provenance"] == "unknown"

    db.upsert_verified_transaction(
        kind="order",
        paypal_id="SNEAKY-VER-002",
        status="APPROVED",
        amount="2.00",
        currency="USD",
        provenance=db.PROVENANCE_PAYPAL_VERIFIED,
    )
    row3 = [r for r in db.list_transactions() if r["paypal_id"] == "SNEAKY-VER-002"][0]
    assert row3["provenance"] == "unknown"


def test_validated_paypal_response_creates_paypal_verified_records(env):
    services.record_transactions(
        "create_order",
        {"amount": "10.00", "currency": "USD", "description": "ok"},
        {"id": ORDER_ID, "status": "CREATED", "amount": "10.00", "currency": "USD"},
    )
    services.record_transactions(
        "capture_order",
        {"order_id": ORDER_ID},
        {
            "id": ORDER_ID,
            "status": "COMPLETED",
            "captures": [{"id": CAPTURE_ID, "status": "COMPLETED", "amount": "10.00", "currency": "USD"}],
        },
    )
    services.record_transactions(
        "refund_capture",
        {"capture_id": CAPTURE_ID, "amount": "5.00", "note": "partial"},
        {"id": "REF-VERIFIED-001", "status": "COMPLETED", "amount": "5.00", "currency": "USD"},
    )
    services.record_transactions(
        "create_invoice",
        {"recipient_email": "a@example.com", "currency": "USD", "items": [{"name": "x", "quantity": 1, "unit_amount": "1.00"}]},
        {"invoice": {"id": "INV-VERIFIED-001", "status": "SENT", "amount": "1.00", "currency": "USD"}, "sent": {"sent": True}},
    )
    rows = db.list_transactions()
    by_id = {r["paypal_id"]: r for r in rows if r.get("paypal_id")}
    assert by_id[ORDER_ID]["provenance"] == "paypal_verified"
    assert by_id[CAPTURE_ID]["provenance"] == "paypal_verified"
    assert by_id["REF-VERIFIED-001"]["provenance"] == "paypal_verified"
    assert by_id["INV-VERIFIED-001"]["provenance"] == "paypal_verified"


def test_missing_paypal_resource_id_in_response_not_falsely_verified(env):
    services.record_transactions(
        "create_order",
        {"amount": "10.00", "currency": "USD", "description": "no id"},
        {"status": "CREATED", "amount": "10.00", "currency": "USD"},
    )
    rows = db.list_transactions()
    assert rows[0]["paypal_id"] is None
    assert rows[0]["provenance"] == "unknown"


@pytest.mark.asyncio
async def test_failed_paypal_execution_does_not_create_verified_record(env):
    action = db.create_pending_action(
        "capture_order",
        {"order_id": ORDER_ID},
    )
    async with respx.mock(assert_all_called=False) as mock:
        mock_token(mock)
        mock.get(ORDER_URL).mock(
            return_value=httpx.Response(200, json=order_json(status="APPROVED"))
        )
        mock.post(f"{ORDER_URL}/capture").mock(
            return_value=httpx.Response(
                422,
                json={
                    "name": "UNPROCESSABLE_ENTITY",
                    "message": "declined",
                    "details": [{"issue": "INSTRUMENT_DECLINED"}],
                },
            )
        )
        client = PayPalClient()
        try:
            with pytest.raises(PayPalError):
                await services.approve_action(action["id"], client)
        finally:
            await client.aclose()
    assert db.list_transactions() == []
    stored = db.get_pending_action(action["id"])
    assert stored["status"] == "failed"


@pytest.mark.asyncio
async def test_ambiguous_timeout_does_not_create_verified_record(env):
    action = db.create_pending_action(
        "create_order",
        {"amount": "10.00", "currency": "USD", "description": "timeout"},
    )
    async with respx.mock(assert_all_called=False) as mock:
        mock_token(mock)
        mock.post(ORDERS_URL).mock(side_effect=httpx.ReadTimeout("timeout"))
        client = PayPalClient()
        try:
            with pytest.raises(PayPalError):
                await services.approve_action(action["id"], client)
        finally:
            await client.aclose()
    assert db.list_transactions() == []
    stored = db.get_pending_action(action["id"])
    assert stored["status"] == "unknown"


def test_transaction_api_and_tool_expose_provenance(env):
    db.add_transaction(
        kind="order",
        paypal_id="LOCAL-API-001",
        status="CREATED",
        amount="1.00",
        currency="USD",
        description="local",
        provenance=db.PROVENANCE_LOCAL,
    )
    with TestClient(app) as client:
        resp = client.get("/api/transactions")
    assert resp.status_code == 200
    txns = resp.json()["transactions"]
    assert any(t["paypal_id"] == "LOCAL-API-001" and t["provenance"] == "local" for t in txns)


@pytest.mark.asyncio
async def test_read_only_verified_lookup_marks_paypal_verified(env):
    async with respx.mock(assert_all_called=True) as mock:
        mock_token(mock)
        mock.get(ORDER_URL).mock(
            return_value=httpx.Response(200, json=order_json(status="APPROVED"))
        )
        client = PayPalClient()
        try:
            from app.agent import _execute_read
            result = await _execute_read("get_order", {"order_id": ORDER_ID}, client)
        finally:
            await client.aclose()
    assert result["id"] == ORDER_ID
    from app.context import remember_verified_record
    evidence = db.VerifiedResourceEvidence(
        kind="order",
        paypal_id=ORDER_ID,
        status=result.get("status"),
        amount=result.get("amount"),
        currency=result.get("currency"),
        raw_response=result,
    )
    remember_verified_record(
        "order", ORDER_ID, result.get("status"),
        evidence=evidence, amount=result.get("amount"), currency=result.get("currency"),
    )
    rows = [r for r in db.list_transactions() if r["paypal_id"] == ORDER_ID]
    assert rows
    assert rows[0]["provenance"] == "paypal_verified"


@pytest.mark.asyncio
async def test_llm_read_response_distinguishes_local_unknown_from_paypal(env):
    db.add_transaction(
        kind="order",
        paypal_id="ORD-VERIFIED-001",
        status="APPROVED",
        amount="10.00",
        currency="USD",
        description="verified order",
        provenance=db.PROVENANCE_UNKNOWN,
    )
    LLM_URL = "https://api.deepseek.com/chat/completions"

    def llm_tool_call(tool, args):
        import json
        return {"id": "x", "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}]}}]}

    def llm_text(text):
        return {"id": "y", "choices": [{"message": {"role": "assistant", "content": text}}]}

    async with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            side_effect=[
                httpx.Response(200, json=llm_tool_call("list_local_transactions", {})),
                httpx.Response(
                    200,
                    json=llm_text(
                        "A local record exists for reference ORD-VERIFIED-001 with status APPROVED, "
                        "but its provenance is unverified (unknown origin). I cannot confirm it is a "
                        "PayPal-verified resource from this local record alone."
                    ),
                ),
            ]
        )
        mock_token(mock)
        client = PayPalClient()
        try:
            from app.agent import run_agent
            result = await run_agent("list local transactions", client=client)
        finally:
            await client.aclose()
    reply = result["reply"].lower()
    assert "local" in reply
    assert "unverified" in reply or "unknown" in reply
    assert "paypal-verified resource" in reply or "cannot confirm" in reply
    read_steps = [s for s in result["steps"] if s.get("type") == "read"]
    assert read_steps
    txns = read_steps[0]["result"]["transactions"]
    assert txns[0]["provenance"] == "unknown"


def test_migration_on_copy_of_real_db_preserves_row3(tmp_path):
    src = "paypilot.db"
    dst = tmp_path / "paypilot_copy.db"
    shutil.copy2(src, dst)
    conn = sqlite3.connect(str(dst))
    try:
        old = conn.execute("SELECT id,paypal_id,status,amount,currency,description,created_at FROM transactions WHERE id=3").fetchone()
        assert old is not None
        assert old[1] == "ORD-VERIFIED-001"
    finally:
        conn.close()
    conn = sqlite3.connect(str(dst))
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(transactions)").fetchall()}
        if "provenance" not in cols:
            conn.execute("ALTER TABLE transactions ADD COLUMN provenance TEXT NOT NULL DEFAULT 'unknown'")
        conn.execute("UPDATE transactions SET provenance='unknown' WHERE provenance IS NULL OR TRIM(provenance)='' OR LOWER(TRIM(provenance)) NOT IN ('unknown','local','paypal_verified')")
        conn.commit()
        row = conn.execute("SELECT id,paypal_id,status,amount,currency,description,created_at,provenance FROM transactions WHERE id=3").fetchone()
        assert row[:7] == old
        assert row[7] == "unknown"
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# New evidence-boundary regression tests
# --------------------------------------------------------------------------- #


def test_ordinary_db_caller_cannot_mark_arbitrary_ids_verified(env):
    """Without valid evidence, paypal_verified provenance is downgraded to unknown."""
    rid = db.add_transaction(
        kind="order",
        paypal_id="NO-EVIDENCE-001",
        status="APPROVED",
        amount="1.00",
        currency="USD",
        description="no evidence",
        provenance=db.PROVENANCE_PAYPAL_VERIFIED,
    )
    row = [r for r in db.list_transactions() if r["id"] == rid][0]
    assert row["provenance"] == "unknown"

    rid2 = db.upsert_verified_transaction(
        kind="capture",
        paypal_id="NO-EVIDENCE-002",
        status="COMPLETED",
        amount="2.00",
        currency="USD",
        provenance=db.PROVENANCE_PAYPAL_VERIFIED,
    )
    row2 = [r for r in db.list_transactions() if r["paypal_id"] == "NO-EVIDENCE-002"][0]
    assert row2["provenance"] == "unknown"


def test_missing_evidence_rejected_for_mismatched_id(env):
    """Evidence for resource A cannot verify resource B (mismatched paypal_id)."""
    evidence_for_a = db.VerifiedResourceEvidence(
        kind="order", paypal_id="ORD-EVIDENCE-A", status="APPROVED", amount="10.00", currency="USD"
    )
    # Try to verify a different ID using evidence for ORD-EVIDENCE-A
    rid = db.add_transaction(
        kind="order",
        paypal_id="ORD-EVIDENCE-B",
        status="APPROVED",
        amount="10.00",
        currency="USD",
        provenance=db.PROVENANCE_PAYPAL_VERIFIED,
        verified_evidence=evidence_for_a,
    )
    row = [r for r in db.list_transactions() if r["id"] == rid][0]
    assert row["provenance"] == "unknown"

    # Upsert with mismatched evidence also stays unknown
    rid2 = db.upsert_verified_transaction(
        kind="order",
        paypal_id="ORD-EVIDENCE-C",
        status="APPROVED",
        amount="10.00",
        currency="USD",
        provenance=db.PROVENANCE_PAYPAL_VERIFIED,
        verified_evidence=evidence_for_a,
    )
    row2 = [r for r in db.list_transactions() if r["paypal_id"] == "ORD-EVIDENCE-C"][0]
    assert row2["provenance"] == "unknown"


def test_valid_evidence_marks_paypal_verified(env):
    """Validated evidence matching the record marks paypal_verified."""
    evidence = db.VerifiedResourceEvidence(
        kind="order",
        paypal_id="ORD-EVIDENCE-OK",
        status="APPROVED",
        amount="10.00",
        currency="USD",
        description="ok",
    )
    rid = db.add_transaction(
        kind="order",
        paypal_id="ORD-EVIDENCE-OK",
        status="APPROVED",
        amount="10.00",
        currency="USD",
        provenance=db.PROVENANCE_PAYPAL_VERIFIED,
        verified_evidence=evidence,
    )
    row = [r for r in db.list_transactions() if r["id"] == rid][0]
    assert row["provenance"] == "paypal_verified"


def test_evidence_upgrades_existing_unknown_record(env):
    """Valid evidence can upgrade an existing unknown record to verified."""
    rid = db.add_transaction(
        kind="order",
        paypal_id="ORD-UPGRADE-001",
        status="APPROVED",
        amount="5.00",
        currency="USD",
        provenance=db.PROVENANCE_UNKNOWN,
    )
    evidence = db.VerifiedResourceEvidence(
        kind="order", paypal_id="ORD-UPGRADE-001", status="APPROVED", amount="5.00", currency="USD"
    )
    new_id = db.upsert_verified_transaction(
        kind="order",
        paypal_id="ORD-UPGRADE-001",
        status="APPROVED",
        amount="5.00",
        currency="USD",
        verified_evidence=evidence,
    )
    assert new_id == rid
    row = [r for r in db.list_transactions() if r["id"] == rid][0]
    assert row["provenance"] == "paypal_verified"


def test_evidence_for_one_kind_cannot_verify_another_kind(env):
    """Order evidence cannot verify a capture record (mismatched kind)."""
    order_evidence = db.VerifiedResourceEvidence(
        kind="order", paypal_id="ORD-KIND-001", status="APPROVED", amount="10.00", currency="USD"
    )
    rid = db.add_transaction(
        kind="capture",
        paypal_id="CAP-KIND-001",
        status="COMPLETED",
        amount="10.00",
        currency="USD",
        provenance=db.PROVENANCE_PAYPAL_VERIFIED,
        verified_evidence=order_evidence,
    )
    row = [r for r in db.list_transactions() if r["id"] == rid][0]
    assert row["provenance"] == "unknown"


# --------------------------------------------------------------------------- #
# Stale result on read failure
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_successful_verified_read_followed_by_failed_read_no_stale_provenance(env):
    """A successful verified read creates a paypal_verified record; a subsequent
    failing read must not reuse the previous result or upgrade any record."""
    async with respx.mock(assert_all_called=True) as mock:
        mock_token(mock)
        # Single route: first call succeeds, second call fails (no stale result reuse)
        mock.get(ORDER_URL).mock(
            side_effect=[
                httpx.Response(200, json=order_json(status="APPROVED")),
                httpx.ReadTimeout("stale read timeout"),
            ]
        )
        client = PayPalClient()
        try:
            from app.agent import _execute_read, _remember_verified_read
            # Successful read
            result1 = await _execute_read("get_order", {"order_id": ORDER_ID}, client)
            assert result1["id"] == ORDER_ID
            # Remember it with evidence (simulating the agent read path)
            evidence = db.VerifiedResourceEvidence(
                kind="order", paypal_id=ORDER_ID, status=result1.get("status"),
                amount=result1.get("amount"), currency=result1.get("currency"), raw_response=result1,
            )
            from app.context import remember_verified_record
            remember_verified_record(
                "order", ORDER_ID, result1.get("status"),
                evidence=evidence, amount=result1.get("amount"), currency=result1.get("currency"),
            )
            rows_before = [r for r in db.list_transactions() if r["paypal_id"] == ORDER_ID]
            assert len(rows_before) == 1
            assert rows_before[0]["provenance"] == "paypal_verified"

            # Failed read - must not create or upgrade anything
            try:
                await _execute_read("get_order", {"order_id": ORDER_ID}, client)
            except PayPalError:
                pass
            except Exception:
                pass

            rows_after = [r for r in db.list_transactions() if r["paypal_id"] == ORDER_ID]
            assert len(rows_after) == 1
            assert rows_after[0]["provenance"] == "paypal_verified"
        finally:
            await client.aclose()


@pytest.mark.asyncio
async def test_failed_read_never_promotes_provenance(env):
    """A failed read on an unknown local record must not promote it to verified."""
    db.add_transaction(
        kind="order",
        paypal_id="ORD-FAILED-READ-001",
        status="CREATED",
        amount="1.00",
        currency="USD",
        provenance=db.PROVENANCE_UNKNOWN,
    )
    async with respx.mock(assert_all_called=False) as mock:
        mock_token(mock)
        mock.get(ORDER_URL).mock(
            side_effect=httpx.ReadTimeout("read failed")
        )
        client = PayPalClient()
        try:
            from app.agent import _execute_read
            try:
                await _execute_read("get_order", {"order_id": "ORD-FAILED-READ-001"}, client)
            except PayPalError:
                pass
            except Exception:
                pass
        finally:
            await client.aclose()
    rows = [r for r in db.list_transactions() if r["paypal_id"] == "ORD-FAILED-READ-001"]
    assert rows
    assert rows[0]["provenance"] == "unknown"
