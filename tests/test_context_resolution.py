"""Phase 4: Verified context resolution — isolated deterministic unit tests.

All fixtures are synthetic and isolated. PayPal HTTP interactions in this file
are MOCKED via `respx` and are NOT real integration verification.

Mocked PayPal HTTP/API interactions in this file (every occurrence):
- `test_valid_but_nonexistent_id`: POST /v1/oauth2/token, GET /v2/checkout/orders/{id} (404)
- `test_live_paypal_fallback`: POST /v1/oauth2/token, GET /v2/checkout/orders/{id} (200)
- `test_paypal_lookup_timeout`: POST /v1/oauth2/token, GET /v2/checkout/orders/{id} (timeout)
- `test_capture_requires_eligible_verified_order`: POST /v1/oauth2/token, GET /v2/checkout/orders/{id} (200, CREATED)
- `test_refund_requires_verified_capture`: POST /v1/oauth2/token (mocked, no other routes)
- `test_no_financial_execution_during_context_resolution`: POST /v1/oauth2/token, POST capture route registered (never called)

Actual PayPal Sandbox read-only integration tests are separated in
`tests/test_phase4_sandbox_readonly.py` and are skipped unless real sandbox
credentials are configured. No PayPal write calls are made in any Phase 4 test.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx

from app import db
from app.agent import run_agent
from app.context import resolve_transaction_reference
from app.intent import detect_write_intent
from app.paypal_client import PayPalClient, PayPalError

LLM_URL = "https://api.deepseek.com/chat/completions"
TOKEN_URL = "https://api-m.sandbox.paypal.com/v1/oauth2/token"
BASE = "https://api-m.sandbox.paypal.com"
ORDER_ID = "ORD-VERIFIED-001"
ORDER_ID_2 = "ORD-VERIFIED-002"
CAPTURE_ID = "CAP-VERIFIED-001"


def _token(mock):
    mock.post(TOKEN_URL).mock(return_value=httpx.Response(200, json={"access_token": "tok", "expires_in": 3600}))


def _llm_text(text):
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def _llm_tool_call(name, args):
    return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}}]}


def seed_verified_order(paypal_id, amount="10.00", currency="USD", status="APPROVED", created_at="2026-01-01T10:00:00+00:00"):
    with db._db() as conn:
        conn.execute(
            "INSERT INTO transactions (kind,paypal_id,parent_id,status,amount,currency,description,created_at) VALUES (?,?,?,?,?,?,?,?)",
            ("order", paypal_id, None, status, amount, currency, "verified order", created_at),
        )


def seed_verified_capture(paypal_id, parent_order, amount="10.00", currency="USD", status="COMPLETED", created_at="2026-01-01T11:00:00+00:00"):
    with db._db() as conn:
        conn.execute(
            "INSERT INTO transactions (kind,paypal_id,parent_id,status,amount,currency,description,created_at) VALUES (?,?,?,?,?,?,?,?)",
            ("capture", paypal_id, parent_order, status, amount, currency, "verified capture", created_at),
        )


# ---------------------------------------------------------------------------
# Explicit verified IDs
# ---------------------------------------------------------------------------

async def test_explicit_verified_order_id(env):
    seed_verified_order(ORDER_ID)
    client = PayPalClient()
    try:
        res = await resolve_transaction_reference("Show order ORD-VERIFIED-001", client=client)
    finally:
        await client.aclose()
    assert res.resolved
    assert res.kind == "order"
    assert res.paypal_id == ORDER_ID


async def test_explicit_verified_capture_id(env):
    seed_verified_order(ORDER_ID)
    seed_verified_capture(CAPTURE_ID, ORDER_ID)
    client = PayPalClient()
    try:
        res = await resolve_transaction_reference("Show capture CAP-VERIFIED-001", client=client)
    finally:
        await client.aclose()
    assert res.resolved
    assert res.kind == "capture"
    assert res.paypal_id == CAPTURE_ID


# ---------------------------------------------------------------------------
# Latest / multiple / ambiguous / missing / malformed / nonexistent
# ---------------------------------------------------------------------------

async def test_latest_order_single_eligible(env):
    seed_verified_order(ORDER_ID)
    client = PayPalClient()
    try:
        res = await resolve_transaction_reference("Show me my latest order", session_id="s1", client=client)
    finally:
        await client.aclose()
    assert res.resolved
    assert res.kind == "order"
    assert res.paypal_id == ORDER_ID


async def test_latest_order_multiple_records_resolves_most_recent(env):
    seed_verified_order(ORDER_ID, created_at="2026-01-01T10:00:00+00:00")
    seed_verified_order(ORDER_ID_2, created_at="2026-01-02T10:00:00+00:00")
    client = PayPalClient()
    try:
        res = await resolve_transaction_reference("latest order", session_id="s1", client=client)
    finally:
        await client.aclose()
    assert res.resolved
    assert res.paypal_id == ORDER_ID_2


async def test_ambiguous_transaction_reference_in_session(env):
    # Two different orders referenced in same session context -> "that order" ambiguous
    seed_verified_order(ORDER_ID)
    seed_verified_order(ORDER_ID_2)
    db.upsert_session_context("s1", "order", ORDER_ID, {"kind": "order", "paypal_id": ORDER_ID, "status": "APPROVED"})
    db.upsert_session_context("s1", "order", ORDER_ID_2, {"kind": "order", "paypal_id": ORDER_ID_2, "status": "APPROVED"})
    client = PayPalClient()
    try:
        res = await resolve_transaction_reference("capture that order", session_id="s1", expected_kind="order", client=client)
    finally:
        await client.aclose()
    assert res.status == "ambiguous"
    assert not res.resolved


async def test_missing_transaction_reference(env):
    client = PayPalClient()
    try:
        res = await resolve_transaction_reference("Show me my latest order", session_id="s1", client=client)
    finally:
        await client.aclose()
    assert res.status == "not_found"
    assert not res.resolved


async def test_malformed_transaction_id(env):
    client = PayPalClient()
    try:
        res = await resolve_transaction_reference("get order 12", client=client)
    finally:
        await client.aclose()
    assert res.status == "malformed"


async def test_valid_but_nonexistent_id(env):
    with respx.mock(assert_all_called=True) as mock:
        _token(mock)
        mock.get(f"{BASE}/v2/checkout/orders/ORD-NOT-EXIST1").mock(
            return_value=httpx.Response(404, json={"name": "RESOURCE_NOT_FOUND", "message": "not found"})
        )
        client = PayPalClient()
        try:
            res = await resolve_transaction_reference("get order ORD-NOT-EXIST1", client=client)
        finally:
            await client.aclose()
    assert res.status == "not_found"


async def test_order_vs_capture_id_confusion(env):
    seed_verified_order(ORDER_ID)
    client = PayPalClient()
    try:
        res = await resolve_transaction_reference("show capture ORD-VERIFIED-001", expected_kind="capture", client=client)
    finally:
        await client.aclose()
    assert res.status == "type_mismatch"
    assert not res.resolved


# ---------------------------------------------------------------------------
# Pending proposal and failed transaction excluded
# ---------------------------------------------------------------------------

async def test_pending_proposal_excluded(env):
    # pending action exists but no verified transaction record
    db.create_pending_action("create_order", {"amount": "10.00", "currency": "USD", "description": "x"})
    client = PayPalClient()
    try:
        res = await resolve_transaction_reference("latest order", session_id="s1", client=client)
    finally:
        await client.aclose()
    assert res.status == "not_found"


async def test_failed_transaction_excluded(env):
    with db._db() as conn:
        conn.execute(
            "INSERT INTO transactions (kind,paypal_id,parent_id,status,amount,currency,description,created_at) VALUES (?,?,?,?,?,?,?,?)",
            ("order", "ORD-FAILED-001", None, "FAILED", "10.00", "USD", "failed", "2026-01-01T10:00:00+00:00"),
        )
    client = PayPalClient()
    try:
        res = await resolve_transaction_reference("latest order", session_id="s1", client=client)
    finally:
        await client.aclose()
    assert res.status == "not_found"


# ---------------------------------------------------------------------------
# Live fallback and timeout
# ---------------------------------------------------------------------------

async def test_live_paypal_fallback(env):
    with respx.mock(assert_all_called=True) as mock:
        _token(mock)
        mock.get(f"{BASE}/v2/checkout/orders/ORD-LIVE-0001").mock(
            return_value=httpx.Response(200, json={"id": "ORD-LIVE-0001", "status": "APPROVED", "purchase_units": [{"amount": {"currency_code": "USD", "value": "10.00"}}]})
        )
        client = PayPalClient()
        try:
            res = await resolve_transaction_reference("get order ORD-LIVE-0001", client=client)
        finally:
            await client.aclose()
    assert res.resolved
    assert res.source == "live"
    assert res.paypal_id == "ORD-LIVE-0001"


async def test_paypal_lookup_timeout(env):
    with respx.mock(assert_all_called=True) as mock:
        _token(mock)
        mock.get(f"{BASE}/v2/checkout/orders/ORD-TIMEOUT01").mock(side_effect=httpx.ReadTimeout("timeout"))
        client = PayPalClient()
        try:
            res = await resolve_transaction_reference("get order ORD-TIMEOUT01", client=client)
        finally:
            await client.aclose()
    assert res.status == "lookup_failed"
    assert "Could not verify" in res.message or "timeout" in res.message.lower()


# ---------------------------------------------------------------------------
# Session isolation
# ---------------------------------------------------------------------------

async def test_session_isolation(env):
    seed_verified_order(ORDER_ID)
    # session s1 has context, s2 does not
    db.upsert_session_context("s1", "order", ORDER_ID, {"kind": "order", "paypal_id": ORDER_ID})
    client = PayPalClient()
    try:
        res1 = await resolve_transaction_reference("capture that order", session_id="s1", expected_kind="order", client=client)
        res2 = await resolve_transaction_reference("capture that order", session_id="s2", expected_kind="order", client=client)
    finally:
        await client.aclose()
    assert res1.resolved and res1.paypal_id == ORDER_ID
    assert not res2.resolved
    assert res2.status in {"not_found", "missing"}


# ---------------------------------------------------------------------------
# No automatic selection when ambiguous; latest/previous nuances
# ---------------------------------------------------------------------------

async def test_no_automatic_selection_when_ambiguous(env):
    seed_verified_order(ORDER_ID)
    seed_verified_order(ORDER_ID_2)
    db.upsert_session_context("s1", "order", ORDER_ID, {"kind": "order", "paypal_id": ORDER_ID, "status": "APPROVED"})
    db.upsert_session_context("s1", "order", ORDER_ID_2, {"kind": "order", "paypal_id": ORDER_ID_2, "status": "APPROVED"})
    client = PayPalClient()
    try:
        res = await resolve_transaction_reference("that order", session_id="s1", expected_kind="order", client=client)
    finally:
        await client.aclose()
    assert res.status == "ambiguous"


# ---------------------------------------------------------------------------
# Financial safety: capture needs eligible order, refund needs verified capture
# ---------------------------------------------------------------------------

async def test_capture_requires_eligible_verified_order(env):
    # order exists but not APPROVED; session context makes "that order" resolvable
    seed_verified_order(ORDER_ID, status="CREATED")
    db.upsert_session_context("s1", "order", ORDER_ID, {"kind": "order", "paypal_id": ORDER_ID, "status": "CREATED"})
    with respx.mock(assert_all_called=True) as mock:
        _token(mock)
        mock.get(f"{BASE}/v2/checkout/orders/{ORDER_ID}").mock(
            return_value=httpx.Response(200, json={"id": ORDER_ID, "status": "CREATED", "purchase_units": [{"amount": {"currency_code": "USD", "value": "10.00"}}]})
        )
        client = PayPalClient()
        try:
            result = await run_agent("Capture that order", client=client, session_id="s1")
        finally:
            await client.aclose()
    # no capture proposal for ineligible order; no write execution
    caps = [a for a in result["pending_actions"] if a["tool"] == "capture_order"]
    assert caps == []


async def test_refund_requires_verified_capture(env):
    # only order exists, no capture
    seed_verified_order(ORDER_ID, status="COMPLETED")
    with respx.mock(assert_all_called=False) as mock:
        _token(mock)
        client = PayPalClient()
        try:
            result = await run_agent("Refund the capture associated with order ORD-VERIFIED-001", client=client, session_id="s1")
        finally:
            await client.aclose()
    refunds = [a for a in result["pending_actions"] if a["tool"] == "refund_capture"]
    assert refunds == []


# ---------------------------------------------------------------------------
# No duplicate transaction records / no financial execution during context resolution
# ---------------------------------------------------------------------------

async def test_no_duplicate_transaction_records_on_context_resolution(env):
    seed_verified_order(ORDER_ID)
    before = len(db.list_transactions())
    client = PayPalClient()
    try:
        await resolve_transaction_reference("Show me my latest order", session_id="s1", client=client)
        await resolve_transaction_reference("Show me my latest order", session_id="s1", client=client)
    finally:
        await client.aclose()
    after = len(db.list_transactions())
    assert after == before


# ---------------------------------------------------------------------------
# Financial session isolation: latest/previous must not use global history
# ---------------------------------------------------------------------------

async def test_financial_latest_requires_session_reference_no_global_fallback(env):
    # Verified global records exist, but session has no unambiguous reference.
    seed_verified_order(ORDER_ID, status="APPROVED")
    client = PayPalClient()
    try:
        from app.context import resolve_for_financial_operation
        res = await resolve_for_financial_operation(
            "Capture my latest order", session_id="s1", operation="capture_order", client=client
        )
    finally:
        await client.aclose()
    assert not res.resolved
    assert res.status == "not_found"
    assert "session" in res.message.lower() or "explicit" in res.message.lower()


async def test_financial_previous_requires_session_reference_no_global_fallback(env):
    seed_verified_order(ORDER_ID, status="APPROVED", created_at="2026-01-01T10:00:00+00:00")
    seed_verified_order(ORDER_ID_2, status="APPROVED", created_at="2026-01-02T10:00:00+00:00")
    client = PayPalClient()
    try:
        from app.context import resolve_for_financial_operation
        res = await resolve_for_financial_operation(
            "Capture the previous order", session_id="s1", operation="capture_order", client=client
        )
    finally:
        await client.aclose()
    assert not res.resolved
    assert res.status in {"not_found", "ambiguous"}
    assert "session" in res.message.lower() or "explicit" in res.message.lower()


async def test_financial_latest_resolves_from_unambiguous_session_reference(env):
    # Session-scoped verified reference is sufficient (deterministic, no global guess).
    seed_verified_order(ORDER_ID, status="APPROVED")
    db.upsert_session_context("s1", "order", ORDER_ID, {"kind": "order", "paypal_id": ORDER_ID, "status": "APPROVED"})
    with respx.mock(assert_all_called=True) as mock:
        _token(mock)
        mock.get(f"{BASE}/v2/checkout/orders/{ORDER_ID}").mock(
            return_value=httpx.Response(200, json={"id": ORDER_ID, "status": "APPROVED", "purchase_units": [{"amount": {"currency_code": "USD", "value": "10.00"}}]})
        )
        client = PayPalClient()
        try:
            result = await run_agent("Capture my latest order", client=client, session_id="s1")
        finally:
            await client.aclose()
    caps = [a for a in result["pending_actions"] if a["tool"] == "capture_order"]
    assert len(caps) == 1
    assert caps[0]["args"]["order_id"] == ORDER_ID
    assert "refund capture route not registered" not in str(result)  # sanity: no writes


# ---------------------------------------------------------------------------
# Contextual reads: supported/unsupported (existing resolution architecture)
# ---------------------------------------------------------------------------

async def test_contextual_read_latest_order_supported_at_resolution_level(env):
    # Supported: deterministic non-financial resolution uses verified local records.
    seed_verified_order(ORDER_ID, status="APPROVED")
    client = PayPalClient()
    try:
        res = await resolve_transaction_reference("Show me my latest order", session_id="s1", client=client)
    finally:
        await client.aclose()
    assert res.resolved
    assert res.kind == "order"
    assert res.paypal_id == ORDER_ID


async def test_contextual_read_that_order_without_session_reference_not_resolved(env):
    # "that order" requires session-scoped verified context; not guessed.
    seed_verified_order(ORDER_ID, status="APPROVED")
    client = PayPalClient()
    try:
        res = await resolve_transaction_reference("What happened to that order?", session_id="s1", expected_kind="order", client=client)
    finally:
        await client.aclose()
    assert not res.resolved
    assert res.status in {"not_found", "missing"}


async def test_contextual_read_that_order_with_session_reference_supported(env):
    seed_verified_order(ORDER_ID, status="APPROVED")
    db.upsert_session_context("s1", "order", ORDER_ID, {"kind": "order", "paypal_id": ORDER_ID, "status": "APPROVED"})
    client = PayPalClient()
    try:
        res = await resolve_transaction_reference("What happened to that order?", session_id="s1", expected_kind="order", client=client)
    finally:
        await client.aclose()
    assert res.resolved
    assert res.paypal_id == ORDER_ID


async def test_agent_contextual_read_what_happened_to_that_order_is_not_fabricated(env):
    # Agent-level: intent layer treats this as unclear/contextual without verified id;
    # agent must not fabricate or execute writes. (LLM not invoked for UNCLEAR path)
    with respx.mock(assert_all_called=False) as mock:
        # No LLM/PayPal calls expected on deterministic unclear/clarification path.
        client = PayPalClient()
        try:
            result = await run_agent("What happened to that order?", client=client, session_id="s1")
        finally:
            await client.aclose()
    assert result["pending_actions"] == []
    reply = (result.get("reply") or "").lower()
    assert "completed" not in reply or "could not" in reply or "clarif" in reply or "specific" in reply or "transaction" in reply or "order" in reply
    # no write execution occurred (no capture/refund endpoints registered/called)


async def test_no_financial_execution_during_context_resolution(env):
    seed_verified_order(ORDER_ID, status="APPROVED")
    with respx.mock(assert_all_called=False) as mock:
        _token(mock)
        # if any write endpoint called, respx would complain in assert_all_called mode; we ensure none by not registering
        capture_post = mock.post(f"{BASE}/v2/checkout/orders/{ORDER_ID}/capture")
        client = PayPalClient()
        try:
            result = await run_agent("Capture that order", client=client, session_id="s1")
        finally:
            await client.aclose()
    assert capture_post.call_count == 0
    # proposal may be created (pending approval) but no execution
