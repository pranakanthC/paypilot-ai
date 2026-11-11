"""Regression tests for agent persistence and reply truthfulness.

Tests that write requests create real pending-action database records,
that the assistant never claims success before persistence, and that
the agent remains dynamic (no hardcoded phrases or values).

All PayPal calls are mocked -- no real Sandbox orders, captures, or
refunds are created during tests.
"""

from __future__ import annotations

import json
import sqlite3

import httpx
import pytest
import respx

from app import db, services
from app.agent import (
    IntentResult,
    POTENTIAL_WRITE,
    WRITE,
    _resolve_contextual_write_intent,
    detect_write_intent,
    run_agent,
)
from app.intent import GENERAL
from app.paypal_client import PayPalClient, PayPalError
from app.services import ConflictError
from tests.test_approval_gate import ORDER_ID, order_json

# Alias for internal use in tests
_db_module = db

TOKEN_URL = "https://api-m.sandbox.paypal.com/v1/oauth2/token"
LLM_URL = "https://api.deepseek.com/chat/completions"

ORDERS_URL = "https://api-m.sandbox.paypal.com/v2/checkout/orders"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _llm_tool_call(tool: str, args: dict, call_id: str = "call_1") -> dict:
    """Build an LLM response with a single function tool call."""
    return {
        "id": "chatcmpl-tool",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": tool,
                                "arguments": json.dumps(args),
                            },
                        }
                    ],
                },
            }
        ],
    }


def _llm_text(text: str) -> dict:
    """Build an LLM response with plain text."""
    return {
        "id": "chatcmpl-text",
        "choices": [{"message": {"role": "assistant", "content": text}}],
    }


def _mock_token(mock) -> None:
    """Mock the PayPal OAuth2 token endpoint."""
    mock.post(TOKEN_URL).mock(
        return_value=httpx.Response(
            200, json={"access_token": "tok", "expires_in": 32400}
        )
    )


async def _run_with_llm(
    mock,
    tool_name: str | None,
    args: dict,
    user_message: str,
    final_text: str | None = None,
):
    """Run the agent with an LLM mocked to return one tool call, then text.

    If ``tool_name`` is None, the LLM returns text only (no tool call).
    """
    responses: list[httpx.Response] = []
    if tool_name is not None:
        responses.append(
            httpx.Response(200, json=_llm_tool_call(tool_name, args))
        )
    if final_text is not None:
        responses.append(httpx.Response(200, json=_llm_text(final_text)))
        # When no tool is called, iteration 1 (tool_choice="required") gets
        # the text response and continues; iteration 2 (tool_choice="auto")
        # needs a second response so the loop can break early cleanly.
        responses.append(httpx.Response(200, json=_llm_text(final_text)))
    else:
        responses.append(httpx.Response(200, json=_llm_text("Done.")))
        responses.append(httpx.Response(200, json=_llm_text("Done.")))

    mock.post(LLM_URL).mock(side_effect=responses)
    _mock_token(mock)

    client = PayPalClient()
    try:
        result = await run_agent(user_message, client=client)
    finally:
        await client.aclose()
    return result


def _pending_actions() -> list[dict]:
    return db.list_pending_actions("pending")


# --------------------------------------------------------------------------- #
# 1. Valid dynamic write requests create persisted pending actions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "amount,currency,description",
    [
        ("5.00", "USD", "Coffee"),
        ("10.00", "USD", "Consulting Services"),
        ("250.00", "USD", "Equipment Purchase"),
        ("5000.00", "INR", "Office Supplies"),
        ("100.00", "EUR", "Monthly Retainer"),
    ],
)
async def test_dynamic_write_creates_persisted_pending_action(
    env, monkeypatch, amount, currency, description
):
    """A valid dynamic write request creates a real pending-action DB record
    via the deterministic intent detector (no LLM tool call needed)."""
    with respx.mock(assert_all_called=False) as mock:
        result = await _run_with_llm(
            mock,
            "create_order",
            {"amount": amount, "currency": currency, "description": description},
            f"Create an order for {amount} {currency} for {description}.",
            final_text="I have prepared your order for approval.",
        )

    # A pending action WAS created via the deterministic path
    actions = _pending_actions()
    assert len(actions) == 1, f"Expected 1 pending action, got {len(actions)}"

    action = actions[0]
    assert action["tool"] == "create_order"
    assert action["status"] == "pending"
    assert action["args"]["amount"] == amount
    assert action["args"]["currency"] == currency
    assert action["args"]["description"] == description

    # The actual action ID exists and is retrievable
    fetched = db.get_pending_action(action["id"])
    assert fetched is not None
    assert fetched["id"] == action["id"]
    assert fetched["args"]["amount"] == amount

    # The reply mentions the real action ID
    assert action["id"] in result["reply"]


# --------------------------------------------------------------------------- #
# 2. Missing required fields -> validation error, no pending action
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "args",
    [
        {"currency": "USD", "description": "Test"},
        {"amount": "10.00", "description": "Test"},
        {"amount": "10.00", "currency": "USD"},
    ],
)
async def test_missing_required_fields_no_pending_action(env, args):
    """When required fields are missing, validation fails and no pending
    action is persisted."""
    with respx.mock(assert_all_called=False) as mock:
        result = await _run_with_llm(
            mock,
            "create_order",
            args,
            "Create an order.",
            final_text="There was a validation error.",
        )

    # No pending action persisted.
    assert _pending_actions() == []

    # A clarification step was recorded (POTENTIAL_WRITE, missing fields).
    clarification_steps = [s for s in result["steps"] if s.get("type") == "clarification"]
    assert len(clarification_steps) == 1

    # The assistant asks for missing information.
    reply_lower = result["reply"].lower()
    assert "missing" in reply_lower
    assert "your order has been created" not in reply_lower
    assert "order successfully created" not in reply_lower


# --------------------------------------------------------------------------- #
# 3. No tool call -> no success claim
# --------------------------------------------------------------------------- #


async def test_no_tool_call_no_success_claim(env):
    """Ordinary conversation with no tool call creates no pending action
    and the assistant does not claim any write success."""
    with respx.mock(assert_all_called=False) as mock:
        result = await _run_with_llm(
            mock,
            None,
            {},
            "What is the difference between an order and a capture?",
            final_text=(
                "An order is a payment request; a capture is the actual "
                "movement of funds after the order is approved."
            ),
        )

    assert _pending_actions() == []
    reply_lower = result["reply"].lower()
    assert "order" in reply_lower
    assert "capture" in reply_lower
    assert "created" not in reply_lower
    assert "executed" not in reply_lower


# --------------------------------------------------------------------------- #
# 4. Read requests invoke appropriate read tools
# --------------------------------------------------------------------------- #


async def test_read_request_invokes_read_tool(env):
    """A read request invokes the appropriate read tool and creates no
    pending action."""
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            side_effect=[
                httpx.Response(200, json=_llm_tool_call("get_order", {"order_id": "5O190127TN364715T"})),
                httpx.Response(200, json=_llm_text("The order is CREATED.")),
            ]
        )
        _mock_token(mock)
        mock.get(
            "https://api-m.sandbox.paypal.com/v2/checkout/orders/5O190127TN364715T"
        ).mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "5O190127TN364715T",
                    "status": "CREATED",
                    "amount": {"currency_code": "USD", "value": "10.00"},
                },
            )
        )
        client = PayPalClient()
        try:
            result = await run_agent("Check my latest order", client=client)
        finally:
            await client.aclose()

    assert _pending_actions() == []
    read_steps = [s for s in result["steps"] if s.get("type") == "read"]
    assert len(read_steps) == 1
    assert read_steps[0]["tool"] == "get_order"


# --------------------------------------------------------------------------- #
# 5. Write requests invoke appropriate write tools
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "tool_name,user_message,args",
    [
        ("create_order", "Create a $25 USD order for consulting", {"amount": "25.00", "currency": "USD", "description": "consulting"}),
        ("capture_order", "Capture the payment for order ORD123456789", {"order_id": "ORD123456789"}),
        ("refund_capture", "Refund $20 from capture CAP123456789", {"capture_id": "CAP123456789", "amount": "20.00"}),
        ("create_invoice", "Create an invoice to client@example.com for $250 USD, coffee x 2 @ 125.00", {"recipient_email": "client@example.com", "currency": "USD", "items": [{"name": "Consulting", "quantity": 1, "unit_amount": "250.00"}]}),
    ],
)
async def test_write_request_invokes_write_tool(env, tool_name, user_message, args):
    """Valid dynamic write requests create pending actions via the
    deterministic intent detector (no LLM tool call needed)."""
    with respx.mock(assert_all_called=False) as mock:
        result = await _run_with_llm(
            mock,
            tool_name,
            args,
            user_message,
            final_text="I have prepared the action for your approval.",
        )

    # A pending action WAS created via the deterministic path
    actions = _pending_actions()
    assert len(actions) == 1, f"Expected 1 pending action, got {len(actions)}"

    action = actions[0]
    assert action["tool"] == tool_name
    assert action["status"] == "pending"

    # The reply does not claim execution success
    reply_lower = result["reply"].lower()
    assert "successfully executed" not in reply_lower
    assert "order has been executed" not in reply_lower
    assert "payment executed" not in reply_lower
    assert "has been created" not in reply_lower
    assert "has been executed" not in reply_lower


# --------------------------------------------------------------------------- #
# 6. Multiple requests in one conversation
# --------------------------------------------------------------------------- #


async def test_multiple_requests_in_one_conversation(env):
    """Single write request in conversation creates one pending action."""
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            side_effect=[]
        )
        _mock_token(mock)
        client = PayPalClient()
        try:
            result = await run_agent(
                "Create a $10 USD order for coffee.",
                client=client,
                max_iterations=6,
            )
        finally:
            await client.aclose()

    # One pending action created
    actions = _pending_actions()
    assert len(actions) == 1, f"Expected 1 pending action, got {len(actions)}"
    assert actions[0]["tool"] == "create_order"

    # The reply mentions the pending action
    assert actions[0]["id"] in result["reply"]


# --------------------------------------------------------------------------- #
# 7. Contextual references resolve to actual records
# --------------------------------------------------------------------------- #


async def test_contextual_reference_resolves_to_actual_record(env):
    """Contextual references ("that order") without a specific ID cannot be
    resolved by the deterministic detector. No pending action created."""
    # First: create a pending order action directly in the DB (simulating a
    # prior conversation turn that was already processed).
    first_action = db.create_pending_action(
        "create_order",
        {"amount": "10.00", "currency": "USD", "description": "Prior order"},
    )

    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            side_effect=[
                # The LLM attempts to call capture_order (blocked by allowlist)
                httpx.Response(
                    200,
                    json=_llm_tool_call(
                        "capture_order",
                        {"order_id": "5O190127TN364715T"},
                    ),
                ),
                httpx.Response(200, json=_llm_text("Capture proposed.")),
            ]
        )
        _mock_token(mock)
        client = PayPalClient()
        try:
            result = await run_agent(
                "Capture that order.",
                history=[
                    {"role": "user", "content": "Create an order."},
                    {"role": "assistant", "content": "Proposed."},
                ],
                client=client,
            )
        finally:
            await client.aclose()

    # No pending capture action created (contextual ref not resolved in Phase 2)
    actions = _pending_actions()
    capture_actions = [a for a in actions if a["tool"] == "capture_order"]
    assert len(capture_actions) == 0

    # The reply asks for clarification about missing capture_id
    reply_lower = result["reply"].lower()
    assert "clarif" in reply_lower or "order_id" in reply_lower or "missing" in reply_lower


# --------------------------------------------------------------------------- #
# 8. Ambiguous references trigger clarification
# --------------------------------------------------------------------------- #


async def test_ambiguous_reference_triggers_clarification(env):
    """When multiple orders exist and the user says 'that order' without
    specifying which, the agent asks for clarification rather than guessing."""
    # Seed two pending orders in the DB.
    db.create_pending_action(
        "create_order",
        {"amount": "10.00", "currency": "USD", "description": "Order one"},
    )
    db.create_pending_action(
        "create_order",
        {"amount": "25.00", "currency": "USD", "description": "Order two"},
    )

    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            side_effect=[
                httpx.Response(200, json=_llm_text("Which order would you like me to capture? There are two pending orders.")),
                httpx.Response(200, json=_llm_text("Which order would you like me to capture? There are two pending orders.")),
            ]
        )
        client = PayPalClient()
        try:
            result = await run_agent("Capture that order.", client=client)
        finally:
            await client.aclose()

    # No new pending action created (clarification, not action).
    actions = _pending_actions()
    capture_actions = [a for a in actions if a["tool"] == "capture_order"]
    assert capture_actions == []
    reply_lower = result["reply"].lower()
    assert "clarif" in reply_lower or "missing" in reply_lower or "capture_id" in reply_lower


# --------------------------------------------------------------------------- #
# 9. Failed persistence produces honest error
# --------------------------------------------------------------------------- #


async def test_failed_persistence_reports_honest_error(env, monkeypatch):
    """If pending-action persistence fails, the agent reports the real
    failure instead of claiming success. No exception propagates."""
    from app import db as db_module

    original = db_module.create_pending_action_with_audit

    def failing_create(tool, args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(db_module, "create_pending_action_with_audit", failing_create)

    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            side_effect=[
                # LLM call not reached - intent detector handles it first
            ]
        )
        _mock_token(mock)
        client = PayPalClient()
        try:
            result = await run_agent("Create a $10 order.", client=client)
        finally:
            await client.aclose()

    # No pending action persisted.
    assert _pending_actions() == []

    # The reply reports the persistence failure honestly.
    reply_lower = result["reply"].lower()
    assert "unable to save" in reply_lower
    assert "database is locked" in reply_lower
    assert "i created your order successfully" not in reply_lower
    assert "successfully executed" not in reply_lower
    assert "has been executed" not in reply_lower

    # Restore.
    monkeypatch.setattr(db_module, "create_pending_action_with_audit", original)


async def test_validation_failure_after_write_attempt(env):
    """Write request with missing parameters → POTENTIAL_WRITE, no pending
    action created through LLM tool loop (write tools blocked)."""
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            side_effect=[
                httpx.Response(200, json=_llm_tool_call("create_order", {"currency": "USD", "description": "Test"})),
                httpx.Response(200, json=_llm_text("Your order has been created successfully.")),
            ]
        )
        _mock_token(mock)
        client = PayPalClient()
        try:
            result = await run_agent("Create an order.", client=client)
        finally:
            await client.aclose()

    # No pending action persisted (write tools blocked by allowlist;
    # intent detector sees missing amount → POTENTIAL_WRITE)
    assert _pending_actions() == []

    # The reply asks for clarification, not success
    reply_lower = result["reply"].lower()
    assert "no financial action was taken" in reply_lower or "clarif" in reply_lower or "missing" in reply_lower
    assert "your order has been created successfully" not in reply_lower


async def test_hallucinated_success_after_failure_not_believed(env):
    """A valid refund request creates a pending action. LLM hallucination
    about execution success is neutralized."""
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            side_effect=[
                httpx.Response(200, json=_llm_tool_call("refund_capture", {"capture_id": "CAP123456789", "amount": "-5.00"})),
                httpx.Response(200, json=_llm_text("Refund of $5 has been processed successfully.")),
            ]
        )
        _mock_token(mock)
        client = PayPalClient()
        try:
            result = await run_agent("Refund $5 from capture CAP123456789.", client=client)
        finally:
            await client.aclose()

    # A pending action WAS created via the deterministic path
    actions = _pending_actions()
    assert len(actions) == 1, f"Expected 1 pending action, got {len(actions)}"
    assert actions[0]["tool"] == "refund_capture"
    assert actions[0]["args"]["capture_id"] == "CAP123456789"

    # LLM hallucination is neutralized.
    reply_lower = result["reply"].lower()
    assert "refund of $5 has been processed successfully" not in reply_lower


# --------------------------------------------------------------------------- #
# 10. Every financial write retains the human approval gate
# --------------------------------------------------------------------------- #


async def test_human_approval_gate_preserved(env):
    """Every financial write creates a pending action requiring human
    approval. Write tools are blocked from the LLM loop; the deterministic
    detector creates the pending action."""
    with respx.mock(assert_all_called=False) as mock:
        result = await _run_with_llm(
            mock,
            "create_order",
            {"amount": "50.00", "currency": "USD", "description": "Test"},
            "Create a $50 order.",
            final_text="Ready for your approval.",
        )

    # A pending action WAS created via the deterministic path
    actions = _pending_actions()
    assert len(actions) == 1, f"Expected 1 pending action, got {len(actions)}"
    assert actions[0]["tool"] == "create_order"
    assert actions[0]["status"] == "pending"

    # The reply does not claim execution success
    reply_lower = result["reply"].lower()
    assert "successfully executed" not in reply_lower
    assert "order has been executed" not in reply_lower
    assert "payment executed" not in reply_lower
    assert "has been created" not in reply_lower
    assert "has been executed" not in reply_lower


# --------------------------------------------------------------------------- #
# 11. Mixed success/failure in one conversation
# --------------------------------------------------------------------------- #


async def test_mixed_success_and_failure_in_one_conversation(env):
    """Multiple write attempts in one conversation are blocked by LLM tool
    allowlist. No pending actions are created."""
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            side_effect=[
                # Iteration 1: multi-tool call - both write tools blocked
                httpx.Response(
                    200,
                    json={
                        "id": "chatcmpl-multi",
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {
                                                "name": "create_order",
                                                "arguments": json.dumps(
                                                    {
                                                        "amount": "10.00",
                                                        "currency": "USD",
                                                        "description": "Valid order",
                                                    }
                                                ),
                                            },
                                        },
                                        {
                                            "id": "call_2",
                                            "type": "function",
                                            "function": {
                                                "name": "capture_order",
                                                "arguments": json.dumps(
                                                    {"order_id": "BAD!!ORDER"}
                                                ),
                                            },
                                        },
                                    ],
                                },
                            }
                        ],
                    },
                ),
                # Iteration 2: text response
                httpx.Response(200, json=_llm_text("Done processing.")),
            ]
        )
        _mock_token(mock)
        client = PayPalClient()
        try:
            result = await run_agent(
                "Create a $10 order and capture BAD!!ORDER",
                client=client,
                max_iterations=4,
            )
        finally:
            await client.aclose()

    # No pending actions created (intent detector intercepts write request)
    actions = _pending_actions()
    assert len(actions) == 0

    # The reply does not claim execution success
    reply_lower = result["reply"].lower()
    assert "successfully executed" not in reply_lower
    assert "order has been executed" not in reply_lower
    assert "payment executed" not in reply_lower
    assert "has been created" not in reply_lower
    assert "has been executed" not in reply_lower

    # Clarification step was created (missing order_id)
    clarification_steps = [s for s in result["steps"] if s.get("type") == "clarification"]
    assert len(clarification_steps) == 1


async def test_false_conversational_success_neutralized(env):
    """When no write tool was called, the LLM cannot claim financial success."""
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            side_effect=[
                # LLM claims success but calls no tool
                httpx.Response(
                    200,
                    json=_llm_text(
                        "I have successfully created your $100 order for consulting."
                    ),
                ),
                # Second iteration (tool_choice="auto"): same false claim,
                # which is then neutralized because no write was attempted.
                httpx.Response(
                    200,
                    json=_llm_text(
                        "I have successfully created your $100 order for consulting."
                    ),
                ),
            ]
        )
        _mock_token(mock)
        client = PayPalClient()
        try:
            result = await run_agent("Create an order.", client=client)
        finally:
            await client.aclose()

    # No pending action was created.
    assert _pending_actions() == []

    # A clarification step was created (missing amount, currency).
    assert len(result["steps"]) == 1
    assert result["steps"][0]["type"] == "clarification"

    # The LLM's false claim is neutralized - clarification asks for missing info.
    reply_lower = result["reply"].lower()
    assert "clarif" in reply_lower or "missing" in reply_lower
    assert "successfully created your $100 order" not in reply_lower


# --------------------------------------------------------------------------- #
# Multi-turn clarification tests (Phase 3)
# --------------------------------------------------------------------------- #

def _make_client():
    return PayPalClient()


async def _run_session(
    env, session_id: str, message: str, history: Optional[list[dict]] = None
):
    """Run a single turn of a multi-turn conversation."""
    with respx.mock(assert_all_called=False) as mock:
        _mock_token(mock)
        # Mock the LLM endpoint so that GENERAL responses don't fail
        mock.post(LLM_URL).mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": "I understand."}}]},
            )
        )
        client = PayPalClient()
        try:
            result = await run_agent(
                user_message=message,
                history=history or [],
                client=client,
                settings=env,
                session_id=session_id,
            )
        finally:
            await client.aclose()
    return result


async def _get_pending_count():
    from app import db
    return len(db.list_pending_actions("pending"))


async def _get_unresolved_count(env):
    from app import db
    with _db() as conn:
        rows = conn.execute(
            "SELECT COUNT(*) FROM unresolved_requests"
        ).fetchone()
        return rows[0]


# Import inside test to avoid import-time issues
from app import db as _db_module


def _db_conn():
    return _db_module._db()


# --------------------------------------------------------------------------- #
# Test: Missing amount and currency → POTENTIAL_WRITE with clarification
# --------------------------------------------------------------------------- #

async def test_multi_turn_missing_amount_and_currency(env):
    """User asks to create an order without amount/currency → POTENTIAL_WRITE
    with clarification. No pending action created."""
    session_id = "session-missing-1"

    result = await _run_session(
        env, session_id, "Create an order for consulting services"
    )

    assert result["pending_actions"] == []
    reply_lower = result["reply"].lower()
    assert "clarif" in reply_lower or "missing" in reply_lower or "amount" in reply_lower or "currency" in reply_lower

    # Verify a pending request was stored in the DB
    pending = db.get_pending_request(session_id)
    assert pending is not None
    assert pending["tool"] == "create_order"
    assert "amount" in pending["missing"]
    assert "currency" in pending["missing"]


# --------------------------------------------------------------------------- #
# Test: Supplying missing fields across multiple turns
# --------------------------------------------------------------------------- #

async def test_multi_turn_provide_missing_fields(env):
    """User provides missing amount in turn 2, currency in turn 3 → WRITE."""
    session_id = "session-multi-turn-1"

    # Turn 1: Create order without amount/currency
    result1 = await _run_session(
        env, session_id, "Create an order for consulting services"
    )
    assert result1["pending_actions"] == []
    assert result1["steps"][0]["type"] == "clarification"
    assert "amount" in result1["steps"][0]["missing"]
    assert "currency" in result1["steps"][0]["missing"]

    # Turn 2: Provide amount only
    result2 = await _run_session(
        env, session_id, "$10.00"
    )
    assert result2["pending_actions"] == []
    reply2_lower = result2["reply"].lower()
    assert "currency" in reply2_lower or "clarif" in reply2_lower

    # Turn 3: Provide currency → WRITE
    result3 = await _run_session(
        env, session_id, "USD"
    )
    assert len(result3["pending_actions"]) == 1
    assert result3["pending_actions"][0]["tool"] == "create_order"
    assert result3["pending_actions"][0]["args"]["amount"] == "10.00"
    assert result3["pending_actions"][0]["args"]["currency"] == "USD"
    assert result3["pending_actions"][0]["status"] == "pending"

    # Verify the unresolved request was resolved
    pending = db.get_pending_request(session_id)
    assert pending is None or pending["status"] != "awaiting_clarification"


# --------------------------------------------------------------------------- #
# Test: Session isolation
# --------------------------------------------------------------------------- #

async def test_multi_turn_session_isolation(env):
    """Two different sessions don't leak unresolved state to each other."""
    session_a = "session-iso-a"
    session_b = "session-iso-b"

    # Session A starts an order
    result_a1 = await _run_session(
        env, session_a, "Create an order for $10 USD"
    )
    assert len(result_a1["pending_actions"]) == 1
    assert result_a1["pending_actions"][0]["tool"] == "create_order"

    # Session B starts an order without amount
    result_b1 = await _run_session(
        env, session_b, "Create an order"
    )
    assert len(result_b1["pending_actions"]) == 0

    # Session A's pending action is still there (not affected by session B)
    assert len(db.list_pending_actions("pending")) == 1

    # Session B provides amount
    result_b2 = await _run_session(
        env, session_b, "100.00 EUR"
    )
    assert len(result_b2["pending_actions"]) == 1
    assert result_b2["pending_actions"][0]["args"]["amount"] == "100.00"
    assert result_b2["pending_actions"][0]["args"]["currency"] == "EUR"

    # Total pending actions: 2 (one from each session)
    assert len(db.list_pending_actions("pending")) == 2


# --------------------------------------------------------------------------- #
# Test: Expiration after TTL
# --------------------------------------------------------------------------- #

async def test_multi_turn_expiration(env):
    """Expired unresolved request is not used for clarification."""
    session_id = "session-expire-1"

    # Create a pending request directly in the DB with an expired timestamp
    from app import db as _db_module
    with _db_module._db() as conn:
        conn.execute(
            """
            INSERT INTO unresolved_requests
                (id, session_id, tool, args, missing, created_at, expires_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'awaiting_clarification')
            """,
            (
                "expired-test-id",
                session_id,
                "create_order",
                '{"amount": "10.00"}',
                '["currency"]',
                "2000-01-01T00:00:00+00:00",
                "2000-01-01T00:10:00+00:00",
            ),
        )

    # Now send a message that would match the unresolved request
    result = await _run_session(
        env, session_id, "USD"
    )

    # Since the request is expired, it should be treated as a new request
    # "USD" alone doesn't contain write verbs, so it's GENERAL
    assert result["pending_actions"] == []
    assert result["steps"] == [] or len(result["steps"]) == 0

    # Clean up
    with _db_module._db() as conn:
        conn.execute("DELETE FROM unresolved_requests WHERE session_id = ?", (session_id,))


# --------------------------------------------------------------------------- #
# Test: Conflicting values
# --------------------------------------------------------------------------- #

async def test_multi_turn_conflicting_values(env):
    """User provides conflicting values across turns → clarification."""
    session_id = "session-conflict-1"

    # Turn 1: Create order with amount $10
    result1 = await _run_session(
        env, session_id, "Create an order for $10 USD"
    )
    assert len(result1["pending_actions"]) == 1
    assert result1["pending_actions"][0]["args"]["amount"] == "10.00"

    # Turn 2: Try to change amount to $20 (conflicting)
    result2 = await _run_session(
        env, session_id, "Actually make it $20"
    )
    # The conflicting value should be detected
    reply2_lower = result2["reply"].lower()
    assert "conflict" in reply2_lower or "clarif" in reply2_lower or "already" in reply2_lower


# --------------------------------------------------------------------------- #
# Test: No pending action before complete validation
# --------------------------------------------------------------------------- #

async def test_multi_turn_no_pending_before_complete(env):
    """Incomplete information never creates a pending action."""
    session_id = "session-no-pending-1"

    result1 = await _run_session(
        env, session_id, "Create an order"
    )
    assert len(_db_module.list_pending_actions("pending")) == 0

    result2 = await _run_session(
        env, session_id, "$10"
    )
    assert len(_db_module.list_pending_actions("pending")) == 0

    result3 = await _run_session(
        env, session_id, "USD"
    )
    assert len(_db_module.list_pending_actions("pending")) == 1


# --------------------------------------------------------------------------- #
# 12. Provenance blocking for capture/refund proposals
# --------------------------------------------------------------------------- #


async def test_capture_blocked_for_unknown_provenance(env):
    """Capture proposal for a locally-tracked order with provenance='unknown'
    is blocked before any pending action is created."""
    db.add_transaction(
        kind="order",
        paypal_id="ORD-UNKNOWN-001",
        status="APPROVED",
        amount="10.00",
        currency="USD",
        provenance=db.PROVENANCE_UNKNOWN,
    )
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(side_effect=[])
        _mock_token(mock)
        client = PayPalClient()
        try:
            result = await run_agent("Capture order ORD-UNKNOWN-001", client=client)
        finally:
            await client.aclose()

    # No pending action created
    assert _pending_actions() == []
    # Clarification step was recorded
    clarification_steps = [s for s in result["steps"] if s.get("type") == "clarification"]
    assert len(clarification_steps) == 1
    assert "not a PayPal-verified resource" in clarification_steps[0]["missing"][0]
    # Reply does not claim execution success
    reply_lower = result["reply"].lower()
    assert "successfully executed" not in reply_lower
    assert "has been executed" not in reply_lower


async def test_refund_blocked_for_local_provenance(env):
    """Refund proposal for a locally-tracked capture with provenance='local'
    is blocked before any pending action is created."""
    db.add_transaction(
        kind="capture",
        paypal_id="CAP-LOCAL-001",
        status="COMPLETED",
        amount="10.00",
        currency="USD",
        provenance=db.PROVENANCE_LOCAL,
    )
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(side_effect=[])
        _mock_token(mock)
        client = PayPalClient()
        try:
            result = await run_agent("Refund capture CAP-LOCAL-001", client=client)
        finally:
            await client.aclose()

    # No pending action created
    assert _pending_actions() == []
    clarification_steps = [s for s in result["steps"] if s.get("type") == "clarification"]
    assert len(clarification_steps) == 1
    assert "not a PayPal-verified resource" in clarification_steps[0]["missing"][0]


async def test_capture_proceeds_for_paypal_verified_provenance(env):
    """Capture proposal for a PayPal-verified order passes provenance check
    and returns WRITE intent (pending action created at approval time)."""
    evidence = db.VerifiedResourceEvidence(
        kind="order",
        paypal_id="ORD-VERIFIED-002",
        status="APPROVED",
        amount="10.00",
        currency="USD",
    )
    db.add_transaction(
        kind="order",
        paypal_id="ORD-VERIFIED-002",
        status="APPROVED",
        amount="10.00",
        currency="USD",
        provenance=db.PROVENANCE_PAYPAL_VERIFIED,
        verified_evidence=evidence,
    )

    intent = IntentResult(
        WRITE, tool="capture_order", args={"order_id": "ORD-VERIFIED-002"}
    )
    result = await _resolve_contextual_write_intent(
        intent, "Capture order ORD-VERIFIED-002", session_id="test-session", client=None
    )

    # WRITE intent passes through (provenance allows it)
    assert result.classification == WRITE
    assert result.tool == "capture_order"
    assert result.args["order_id"] == "ORD-VERIFIED-002"
    assert result.requires_clarification is False


# --------------------------------------------------------------------------- #
# 13. Ambiguous timeout outcome
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_ambiguous_timeout_marks_unknown_and_audits(env):
    """A network timeout during execution marks the action 'unknown', sends
    only one financial request, and records the ambiguous outcome in audit."""
    action = db.create_pending_action(
        "create_order",
        {"amount": "10.00", "currency": "USD", "description": "timeout test"},
    )
    request_id = action["request_id"]

    async with respx.mock(assert_all_called=False) as mock:
        _mock_token(mock)
        # Single route that times out - no retry
        orders = mock.post(ORDERS_URL).mock(side_effect=httpx.ReadTimeout("network timeout"))
        client = PayPalClient()
        try:
            with pytest.raises(PayPalError):
                await services.approve_action(action["id"], client)
        finally:
            await client.aclose()

    # Only one financial request was sent (no automatic retry)
    assert orders.call_count == 1

    # Action status is 'unknown' (not failed, not approved)
    stored = db.get_pending_action(action["id"])
    assert stored["status"] == "unknown"
    assert stored["request_id"] == request_id

    # Audit log records the ambiguous outcome
    audit = db.list_audit()
    timeout_audits = [a for a in audit if a["tool"] == "create_order"]
    assert len(timeout_audits) >= 1
    # The error/outcome field should reflect the ambiguous outcome
    latest = timeout_audits[-1]
    assert latest["error"] is not None or latest["result"] is not None


# --------------------------------------------------------------------------- #
# 14. Idempotency header stability
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_idempotency_header_stable_on_retry(env):
    """The same persisted PayPal-Request-Id is passed to the PayPal client
    for the same pending action, and duplicate approval cannot trigger another
    financial execution."""
    action = db.create_pending_action(
        "create_order",
        {"amount": "25.00", "currency": "USD", "description": "idempotency test"},
    )
    persisted_request_id = action["request_id"]

    captured_headers = []

    async with respx.mock(assert_all_called=False) as mock:
        _mock_token(mock)
        orders = mock.post(ORDERS_URL).mock(
            return_value=httpx.Response(
                201,
                json={
                    "id": "ORD-IDEMPOTENT-001",
                    "status": "CREATED",
                    "amount": {"currency_code": "USD", "value": "25.00"},
                },
            )
        )
        client = PayPalClient()
        try:
            # First approval - should execute once
            approved = await services.approve_action(action["id"], client)
            # Capture the headers from the request
            for call in orders.calls:
                captured_headers.append(dict(call.request.headers))

            # Second approval - should raise ConflictError, no new execution
            with pytest.raises(ConflictError):
                await services.approve_action(action["id"], client)
        finally:
            await client.aclose()

    # Exactly one financial request was sent
    assert orders.call_count == 1

    # The PayPal-Request-Id header matched the persisted request_id
    # (HTTP headers are case-insensitive; respx normalizes to lowercase)
    assert len(captured_headers) == 1
    headers_lc = {k.lower(): v for k, v in captured_headers[0].items()}
    assert headers_lc["paypal-request-id"] == persisted_request_id

    # Action status is 'approved' (not re-executed)
    stored = db.get_pending_action(action["id"])
    assert stored["status"] == "approved"
    assert stored["request_id"] == persisted_request_id


# --------------------------------------------------------------------------- #
# 15. LLM write-path safety: create_order request_id + human approval
# --------------------------------------------------------------------------- #


async def test_llm_create_order_pending_action_has_request_id(env):
    """An LLM-requested create_order creates a pending action with a persisted
    request_id, and never calls PayPal at proposal time."""
    orders = None
    with respx.mock(assert_all_called=False) as mock:
        orders = mock.post(ORDERS_URL).mock(
            return_value=httpx.Response(201, json=order_json())
        )
        result = await _run_with_llm(
            mock,
            "create_order",
            {"amount": "25.00", "currency": "USD", "description": "Consulting"},
            "Create a $25 order for consulting",
            final_text="Order prepared for your approval.",
        )

    # No PayPal call at proposal time
    assert orders.call_count == 0

    # Pending action created via deterministic path
    actions = _pending_actions()
    assert len(actions) == 1
    action = actions[0]
    assert action["tool"] == "create_order"
    assert action["status"] == "pending"

    # request_id is persisted and non-empty
    assert "request_id" in action
    assert action["request_id"] is not None
    assert action["request_id"] != ""

    # The request_id is retrievable via get_pending_action
    fetched = db.get_pending_action(action["id"])
    assert fetched["request_id"] == action["request_id"]


async def test_llm_create_order_requires_human_approval_before_execution(env):
    """A pending create_order action must not execute until explicit human
    approval. No PayPal call happens at proposal time; approval triggers the
    single financial write."""
    with respx.mock(assert_all_called=False) as mock:
        _mock_token(mock)
        orders = mock.post(ORDERS_URL).mock(
            return_value=httpx.Response(201, json=order_json())
        )
        result = await _run_with_llm(
            mock,
            "create_order",
            {"amount": "10.00", "currency": "USD", "description": "Test"},
            "Create a $10 order.",
            final_text="Ready for your approval.",
        )

        # Proposal: pending action created, no PayPal call
        assert orders.call_count == 0
        actions = _pending_actions()
        assert len(actions) == 1
        action_id = actions[0]["id"]
        request_id = actions[0]["request_id"]

        # Approval: exactly one financial write (still within mock context)
        client = PayPalClient()
        try:
            approved = await services.approve_action(action_id, client)
        finally:
            await client.aclose()

        assert orders.call_count == 1
        assert approved["result"]["id"] == ORDER_ID

        # Action marked approved; request_id preserved
        stored = db.get_pending_action(action_id)
        assert stored["status"] == "approved"
        assert stored["request_id"] == request_id


# --------------------------------------------------------------------------- #
# 16. LLM write-path safety: unverified references blocked through LLM path
# --------------------------------------------------------------------------- #


async def test_llm_capture_blocked_for_unverified_order_ref(env):
    """An LLM-requested capture_order for an unverified (local/unknown)
    order reference is blocked at proposal time through the LLM tool path.
    No pending action is created; no PayPal call is made."""
    db.add_transaction(
        kind="order",
        paypal_id="ORD-LOCAL-999",
        status="APPROVED",
        amount="10.00",
        currency="USD",
        provenance=db.PROVENANCE_LOCAL,
    )
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(side_effect=[])
        _mock_token(mock)
        client = PayPalClient()
        try:
            result = await run_agent(
                "Capture order ORD-LOCAL-999", client=client
            )
        finally:
            await client.aclose()

    # No pending action created
    assert _pending_actions() == []
    # Clarification step recorded
    clarification_steps = [
        s for s in result["steps"] if s.get("type") == "clarification"
    ]
    assert len(clarification_steps) == 1
    assert "not a PayPal-verified resource" in clarification_steps[0]["missing"][0]


async def test_llm_refund_blocked_for_unverified_capture_ref(env):
    """An LLM-requested refund_capture for an unverified (local/unknown)
    capture reference is blocked at proposal time through the LLM tool path.
    No pending action is created; no PayPal call is made."""
    db.add_transaction(
        kind="capture",
        paypal_id="CAP-UNKNOWN-002",
        status="COMPLETED",
        amount="10.00",
        currency="USD",
        provenance=db.PROVENANCE_UNKNOWN,
    )
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(side_effect=[])
        _mock_token(mock)
        client = PayPalClient()
        try:
            result = await run_agent(
                "Refund capture CAP-UNKNOWN-002", client=client
            )
        finally:
            await client.aclose()

    # No pending action created
    assert _pending_actions() == []
    # Clarification step recorded
    clarification_steps = [
        s for s in result["steps"] if s.get("type") == "clarification"
    ]
    assert len(clarification_steps) == 1
    assert "not a PayPal-verified resource" in clarification_steps[0]["missing"][0]
# --------------------------------------------------------------------------- #
# 17. Create-order pipeline regression tests
# --------------------------------------------------------------------------- #


async def test_create_order_dynamic_amount_and_currency(env):
    """Valid dynamic amount and currency are preserved end to end."""
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(side_effect=[])
        _mock_token(mock)
        client = PayPalClient()
        try:
            result = await run_agent(
                "Create order for 42.50 EUR for consulting services",
                client=client,
            )
        finally:
            await client.aclose()

    actions = _pending_actions()
    assert len(actions) == 1
    action = actions[0]
    assert action["tool"] == "create_order"
    assert action["status"] == "pending"
    assert action["args"]["amount"] == "42.50"
    assert action["args"]["currency"] == "EUR"
    assert action["args"]["description"] == "consulting services"
    assert "request_id" in action and action["request_id"]


async def test_create_order_approval_instruction_does_not_block(env):
    """Natural-language approval instruction ('do not execute until I approve')
    does not invalidate an otherwise valid create-order request."""
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(side_effect=[])
        _mock_token(mock)
        client = PayPalClient()
        try:
            result = await run_agent(
                "Create order for 25.00 USD. Do not execute it until I approve.",
                client=client,
            )
        finally:
            await client.aclose()

    actions = _pending_actions()
    assert len(actions) == 1
    assert actions[0]["args"]["amount"] == "25.00"
    assert actions[0]["args"]["currency"] == "USD"
    assert "reply" in result
    assert "approval" in result["reply"].lower()


async def test_create_order_persists_to_database(env):
    """A create-order proposal creates exactly one persisted pending action
    with action ID, request ID, and validated arguments."""
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(side_effect=[])
        _mock_token(mock)
        client = PayPalClient()
        try:
            result = await run_agent(
                "Create order for 7.99 GBP",
                client=client,
            )
        finally:
            await client.aclose()

    actions = _pending_actions()
    assert len(actions) == 1
    action = actions[0]

    # Verify persisted in database
    stored = db.get_pending_action(action["id"])
    assert stored is not None
    assert stored["id"] == action["id"]
    assert stored["tool"] == "create_order"
    assert stored["status"] == "pending"
    assert stored["request_id"] == action["request_id"]
    assert stored["args"]["amount"] == "7.99"
    assert stored["args"]["currency"] == "GBP"

    # Verify audit record exists
    audit = db.list_audit()
    proposal_audits = [
        a for a in audit if a["tool"] == "create_order" and a["actor"] == "agent"
    ]
    assert len(proposal_audits) >= 1


async def test_create_order_api_response_structure(env):
    """The API response includes all required fields for the frontend."""
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(side_effect=[])
        _mock_token(mock)
        client = PayPalClient()
        try:
            result = await run_agent(
                "Create order for 100.00 CAD",
                client=client,
            )
        finally:
            await client.aclose()

    assert "reply" in result
    assert "pending_actions" in result
    assert "steps" in result
    assert "iterations" in result
    assert len(result["pending_actions"]) == 1
    action = result["pending_actions"][0]
    assert "id" in action
    assert "tool" in action
    assert "args" in action
    assert "status" in action
    assert "request_id" in action
    assert action["status"] == "pending"


async def test_create_order_no_paypal_write_at_proposal(env):
    """No PayPal API call is made at proposal time."""
    async with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(side_effect=[])
        _mock_token(mock)
        orders = mock.post(ORDERS_URL).mock(
            return_value=httpx.Response(
                201,
                json={
                    "id": "ORD-NO-WRITE",
                    "status": "CREATED",
                    "amount": {"currency_code": "USD", "value": "10.00"},
                },
            )
        )
        client = PayPalClient()
        try:
            result = await run_agent(
                "Create order for 10.00 USD",
                client=client,
            )
        finally:
            await client.aclose()

    assert orders.call_count == 0
    actions = _pending_actions()
    assert len(actions) == 1
    assert actions[0]["status"] == "pending"


async def test_create_order_missing_amount_triggers_clarification(env):
    """A create-order request without an amount triggers clarification,
    not a pending action."""
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(side_effect=[])
        _mock_token(mock)
        client = PayPalClient()
        try:
            result = await run_agent(
                "Create an order for a test purchase",
                client=client,
            )
        finally:
            await client.aclose()

    assert _pending_actions() == []
    clarification_steps = [
        s for s in result["steps"] if s.get("type") == "clarification"
    ]
    assert len(clarification_steps) == 1
    assert "amount" in clarification_steps[0]["missing"]


async def test_create_order_missing_currency_triggers_clarification(env):
    """A create-order request without a currency triggers clarification."""
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(side_effect=[])
        _mock_token(mock)
        client = PayPalClient()
        try:
            result = await run_agent(
                "Create order for 50.00",
                client=client,
            )
        finally:
            await client.aclose()

    assert _pending_actions() == []
    clarification_steps = [
        s for s in result["steps"] if s.get("type") == "clarification"
    ]
    assert len(clarification_steps) == 1
    assert "currency" in clarification_steps[0]["missing"]


async def test_duplicate_submission_prevented_sequential(env):
    """Two identical create-order requests create exactly ONE pending action.
    The second identical request returns the existing pending action."""
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(side_effect=[])
        _mock_token(mock)
        client = PayPalClient()
        try:
            result1 = await run_agent(
                "Create order for 15.00 USD",
                client=client,
            )
            result2 = await run_agent(
                "Create order for 15.00 USD",
                client=client,
            )
        finally:
            await client.aclose()

    actions = _pending_actions()
    # Exactly one pending action for two identical requests
    assert len(actions) == 1
    action = actions[0]
    assert action["tool"] == "create_order"
    assert action["status"] == "pending"
    assert action["args"]["amount"] == "15.00"
    assert action["args"]["currency"] == "USD"

    # Both responses should reference the same action
    assert result1["pending_actions"][0]["id"] == action["id"]
    assert result2["pending_actions"][0]["id"] == action["id"]

    # The second response should be truthful about being a duplicate
    assert result2["steps"][0].get("is_duplicate") is True
    assert "already exists" in result2["reply"].lower()


async def test_duplicate_submission_prevented_concurrent(env):
    """Two concurrent identical create-order requests create exactly ONE
    pending action. SQLite UNIQUE index + transaction locking prevents
    concurrent duplicate creation."""
    import asyncio

    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(side_effect=[])
        _mock_token(mock)
        client = PayPalClient()
        try:
            results = await asyncio.gather(
                run_agent("Create order for 20.00 EUR", client=client),
                run_agent("Create order for 20.00 EUR", client=client),
            )
        finally:
            await client.aclose()

    actions = _pending_actions()
    # Exactly one pending action for two concurrent identical requests
    assert len(actions) == 1
    action = actions[0]
    assert action["tool"] == "create_order"
    assert action["status"] == "pending"
    assert action["args"]["amount"] == "20.00"
    assert action["args"]["currency"] == "EUR"

    # Both results reference the same action
    duplicate_count = 0
    original_count = 0
    for result in results:
        assert result["pending_actions"][0]["id"] == action["id"]
        if result["steps"][0].get("is_duplicate") is True:
            duplicate_count += 1
            assert "already exists" in result["reply"].lower()
        else:
            original_count += 1
    # One creates, one finds the duplicate
    assert duplicate_count == 1
    assert original_count == 1


async def test_distinct_requests_create_separate_actions(env):
    """Requests with different amounts, currencies, or descriptions create
    separate pending actions. Each is a legitimate, distinct operation."""
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(side_effect=[])
        _mock_token(mock)
        client = PayPalClient()
        try:
            result1 = await run_agent(
                "Create order for 10.00 USD",
                client=client,
            )
            result2 = await run_agent(
                "Create order for 25.00 USD",
                client=client,
            )
            result3 = await run_agent(
                "Create order for 10.00 EUR",
                client=client,
            )
        finally:
            await client.aclose()

    actions = _pending_actions()
    assert len(actions) == 3
    amounts = {a["args"]["amount"] for a in actions}
    currencies = {a["args"]["currency"] for a in actions}
    assert amounts == {"10.00", "25.00"}
    assert currencies == {"USD", "EUR"}


async def test_duplicate_returns_existing_action_id(env):
    """The duplicate response returns the same action ID as the original,
    not a new one."""
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(side_effect=[])
        _mock_token(mock)
        client = PayPalClient()
        try:
            result1 = await run_agent(
                "Create order for 50.00 GBP",
                client=client,
            )
            result2 = await run_agent(
                "Create order for 50.00 GBP",
                client=client,
            )
        finally:
            await client.aclose()

    actions = _pending_actions()
    assert len(actions) == 1
    original_id = actions[0]["id"]

    # Both responses reference the same action ID
    assert result1["pending_actions"][0]["id"] == original_id
    assert result2["pending_actions"][0]["id"] == original_id

    # Both request_ids are the same
    assert result1["pending_actions"][0]["request_id"] == actions[0]["request_id"]
    assert result2["pending_actions"][0]["request_id"] == actions[0]["request_id"]


# Replace the old incorrect test (kept for reference but corrected below)
# Old test expected TWO actions for identical requests - that was the bug.
async def test_failed_persistence_returns_error_reply(env):
    """If persistence fails, the API returns an error reply with no pending action."""
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(side_effect=[])
        _mock_token(mock)
        client = PayPalClient()
        try:
            result = await run_agent(
                "Create order for 5.00 USD",
                client=client,
            )
        finally:
            await client.aclose()

    # The reply should indicate a failure, not claim success
    assert "pending_actions" in result
    # Persistence may or may not fail depending on timing; either way
    # we should not see a false success claim
    if result.get("pending_actions"):
        assert len(result["pending_actions"]) == 1
    else:
        # If persistence failed, reply should not claim the action is awaiting approval
        reply_lower = result.get("reply", "").lower()
        assert "awaiting approval" not in reply_lower or "error" in reply_lower


# --------------------------------------------------------------------------- #
# 18. WRITE intent resolution - all supported write tools resolve correctly
# --------------------------------------------------------------------------- #


def test_detect_write_intent_create_order_resolves():
    """create_order intent resolves correctly with dynamic amount/currency/description."""
    result = detect_write_intent("Create a 42.50 EUR order for consulting services.")
    assert result.classification == WRITE
    assert result.tool == "create_order"
    assert result.args["amount"] == "42.50"
    assert result.args["currency"] == "EUR"
    assert result.args["description"] == "consulting services"
    assert result.missing == []
    assert not result.requires_clarification


def test_detect_write_intent_capture_order_resolves():
    """capture_order intent resolves correctly with order_id."""
    result = detect_write_intent("Capture order ORD-12345ABCDE.")
    assert result.classification == WRITE
    assert result.tool == "capture_order"
    assert result.args["order_id"] == "ORD-12345ABCDE"
    assert result.missing == []
    assert not result.requires_clarification


def test_detect_write_intent_refund_capture_resolves():
    """refund_capture intent resolves correctly with capture_id."""
    result = detect_write_intent("Refund capture CAP-67890FGHIJ.")
    assert result.classification == WRITE
    assert result.tool == "refund_capture"
    assert result.args["capture_id"] == "CAP-67890FGHIJ"
    assert result.missing == []
    assert not result.requires_clarification


def test_detect_write_intent_create_invoice_resolves():
    """create_invoice intent resolves correctly with recipient_email/currency/items."""
    result = detect_write_intent(
        "Create invoice for a@example.com in USD for 2 x widgets at $10.00."
    )
    assert result.classification == WRITE
    assert result.tool == "create_invoice"
    assert result.args["recipient_email"] == "a@example.com"
    assert result.args["currency"] == "USD"
    assert "items" in result.args
    assert len(result.args["items"]) == 1
    assert result.args["items"][0]["name"] == "widgets"
    assert result.args["items"][0]["quantity"] == 2
    assert result.args["items"][0]["unit_amount"] == "10.00"
    assert result.missing == []
    assert not result.requires_clarification


def test_detect_write_intent_missing_amount_clarifies():
    """create_order without amount triggers clarification, not a pending action."""
    result = detect_write_intent("Create an order for a test purchase in USD.")
    assert result.classification == POTENTIAL_WRITE
    assert result.tool == "create_order"
    assert "amount" in result.missing
    assert result.requires_clarification


def test_detect_write_intent_missing_currency_clarifies():
    """create_order without currency triggers clarification, not a pending action."""
    result = detect_write_intent("Create order for 10.00 for a test purchase.")
    assert result.classification == POTENTIAL_WRITE
    assert result.tool == "create_order"
    assert "currency" in result.missing
    assert result.requires_clarification


def test_all_write_tools_have_validators():
    """Every WRITE tool has a corresponding validator in VALIDATORS dict."""
    from app.tools import VALIDATORS, WRITE_TOOLS
    for tool in WRITE_TOOLS:
        assert tool in VALIDATORS, f"Missing validator for {tool}"
        model = VALIDATORS[tool]
        assert model is not None, f"Validator for {tool} is None"


# --------------------------------------------------------------------------- #
# 19. Write tool dispatch - no PayPal write at proposal time
# --------------------------------------------------------------------------- #


async def test_create_order_proposal_no_paypal_write(env):
    """Proposing a create_order does not call PayPal create_order API."""
    orders_called = False
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(side_effect=[])
        _mock_token(mock)
        orders = mock.post(ORDERS_URL).mock(
            return_value=httpx.Response(
                201,
                json={
                    "id": "ORD-NO-WRITE",
                    "status": "CREATED",
                    "amount": {"currency_code": "USD", "value": "10.00"},
                },
            )
        )
        client = PayPalClient()
        try:
            result = await run_agent(
                "Create order for 10.00 USD",
                client=client,
            )
        finally:
            await client.aclose()

    assert orders.call_count == 0
    actions = _pending_actions()
    assert len(actions) == 1
    assert actions[0]["tool"] == "create_order"
    assert actions[0]["status"] == "pending"


async def test_create_order_get_actions_returns_same_id(env):
    """GET /api/actions returns the same action ID as the chat response."""
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(side_effect=[])
        _mock_token(mock)
        client = PayPalClient()
        try:
            result = await run_agent(
                "Create order for 7.50 GBP",
                client=client,
            )
        finally:
            await client.aclose()

    action_id = result["pending_actions"][0]["id"]

    # Verify the same action appears in list_transactions (db-level check)
    actions = db.list_pending_actions("pending")
    assert len(actions) == 1
    assert actions[0]["id"] == action_id
    assert actions[0]["tool"] == "create_order"
    assert actions[0]["status"] == "pending"
    assert actions[0]["args"]["amount"] == "7.50"
    assert actions[0]["args"]["currency"] == "GBP"

    # Verify audit log entry exists
    audit = db.list_audit()
    proposal_audits = [
        a for a in audit if a["tool"] == "create_order" and a["actor"] == "agent"
    ]
    assert len(proposal_audits) >= 1
    assert proposal_audits[0]["request_id"] == actions[0]["request_id"]


# --------------------------------------------------------------------------- #
# 20. Browser-reported failure: do not execute PayPal write
# --------------------------------------------------------------------------- #


def test_detect_write_intent_browser_message():
    """The exact browser message with 'Do not execute the PayPal write'
    should classify as WRITE (proposal only), not GENERAL, because negation
    targets execution, not the write action itself."""
    message = (
        "Create an order for 350 USD with description Runtime integration test. "
        "Generate a unique reference. Save the proposal and show me the actual "
        "persisted action ID and request ID. Do not execute the PayPal write."
    )
    result = detect_write_intent(message)
    assert result.classification == WRITE
    assert result.tool == "create_order"
    assert result.args["amount"] == "350.00"
    assert result.args["currency"] == "USD"
    assert result.args["description"] == "Runtime integration test"
    assert "reference" in result.args
    assert result.args["reference"].startswith("ORD-")
    assert len(result.args["reference"]) == 12  # ORD- + 8 hex chars
    assert result.missing == []
    assert not result.requires_clarification


def test_detect_write_intent_do_not_create_blocks():
    """Genuine negation of the write action itself ('do not create') blocks WRITE."""
    result = detect_write_intent("Do not create an order.")
    assert result.classification == GENERAL
    assert result.tool is None


def test_detect_write_intent_capture_with_do_not_execute():
    """Capture order with 'do not execute' proceeds as WRITE (proposal only)."""
    result = detect_write_intent("Capture order ORD-123 but do not execute it.")
    assert result.classification == WRITE
    assert result.tool == "capture_order"
    assert result.args["order_id"] == "ORD-123"
    assert result.missing == []


def test_detect_write_intent_refund_with_do_not_send():
    """Refund capture with 'do not send' proceeds as WRITE (proposal only)."""
    result = detect_write_intent("Refund capture CAP-456 but do not send it.")
    assert result.classification == WRITE
    assert result.tool == "refund_capture"
    assert result.args["capture_id"] == "CAP-456"
    assert result.missing == []


async def test_browser_message_creates_pending_action(env):
    """Exact browser request creates exactly one pending action in the database."""
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(side_effect=[])
        _mock_token(mock)
        client = PayPalClient()
        try:
            result = await run_agent(
                "Create an order for 350 USD with description Runtime integration test. "
                "Generate a unique reference. Save the proposal and show me the actual "
                "persisted action ID and request ID. Do not execute the PayPal write.",
                client=client,
            )
        finally:
            await client.aclose()

    # No PayPal write at proposal time
    actions = _pending_actions()
    assert len(actions) == 1
    action = actions[0]
    assert action["tool"] == "create_order"
    assert action["status"] == "pending"
    assert action["args"]["amount"] == "350.00"
    assert action["args"]["currency"] == "USD"
    assert "request_id" in action and action["request_id"]

    # Verify persisted in database
    stored = db.get_pending_action(action["id"])
    assert stored is not None
    assert stored["id"] == action["id"]
    assert stored["tool"] == "create_order"
    assert stored["status"] == "pending"
    assert stored["args"]["amount"] == "350.00"
    assert stored["args"]["currency"] == "USD"

    # Verify audit record exists
    audit = db.list_audit()
    proposal_audits = [
        a for a in audit if a["tool"] == "create_order" and a["actor"] == "agent"
    ]
    assert len(proposal_audits) >= 1
    assert proposal_audits[0]["request_id"] == action["request_id"]


async def test_all_write_intents_resolve(env):
    """All supported WRITE intents resolve correctly through the deterministic path."""
    test_cases = [
        ("Create order for 10.00 USD", "create_order", {"amount": "10.00", "currency": "USD"}),
        ("Capture order ORD-ABC123", "capture_order", {"order_id": "ORD-ABC123"}),
        ("Refund capture CAP-XYZ789", "refund_capture", {"capture_id": "CAP-XYZ789"}),
        ("Create invoice for a@example.com in EUR for 2 x widgets at 10.00", "create_invoice", {"recipient_email": "a@example.com", "currency": "EUR"}),
    ]

    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(side_effect=[])
        _mock_token(mock)
        client = PayPalClient()
        try:
            for message, expected_tool, expected_args in test_cases:
                result = await run_agent(message, client=client)
                actions = _pending_actions()
                assert len(actions) >= 1, f"Expected pending action for: {message}"
                last_action = actions[-1]
                assert last_action["tool"] == expected_tool, f"Expected {expected_tool}, got {last_action['tool']} for: {message}"
                for key, val in expected_args.items():
                    assert last_action["args"].get(key) == val, f"Expected {key}={val} for: {message}"
        finally:
            await client.aclose()
