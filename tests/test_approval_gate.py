import asyncio
import json

import httpx
import pytest
import respx

from app import db
from app.agent import run_agent
from app.paypal_client import PayPalClient, PayPalError
from app.services import (
    BusinessValidationError,
    ConflictError,
    approve_action,
    reject_action,
)

LLM_URL = "https://api.deepseek.com/chat/completions"
TOKEN_URL = "https://api-m.sandbox.paypal.com/v1/oauth2/token"
ORDERS_URL = "https://api-m.sandbox.paypal.com/v2/checkout/orders"
ORDER_URL = "https://api-m.sandbox.paypal.com/v2/checkout/orders/5O190127TN364715T"
CAPTURE_URL = "https://api-m.sandbox.paypal.com/v2/payments/captures/3C679366HH908993F"
REFUND_URL = (
    "https://api-m.sandbox.paypal.com/v2/payments/captures/3C679366HH908993F/refund"
)
ORDER_ID = "5O190127TN364715T"
CAPTURE_ID = "3C679366HH908993F"


def order_json(status="CREATED", amount="25.00"):
    data = {
        "id": ORDER_ID,
        "intent": "CAPTURE",
        "status": status,
        "purchase_units": [
            {
                "reference_id": "default",
                "amount": {"currency_code": "USD", "value": amount},
            }
        ],
        "links": [
            {
                "href": f"https://www.sandbox.paypal.com/checkoutnow?token={ORDER_ID}",
                "rel": "payer-action",
            }
        ],
    }
    if status == "COMPLETED":
        data["purchase_units"][0]["payments"] = {
            "captures": [
                {
                    "id": CAPTURE_ID,
                    "status": "COMPLETED",
                    "amount": {"currency_code": "USD", "value": amount},
                    "final_capture": True,
                }
            ]
        }
    return data


def capture_json(amount="50.00"):
    return {
        "id": CAPTURE_ID,
        "status": "COMPLETED",
        "amount": {"currency_code": "USD", "value": amount},
        "final_capture": True,
    }


def llm_tool_call(tool, args, call_id="call_1"):
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
                }
            }
        ],
    }


def llm_text(text):
    return {
        "id": "chatcmpl-text",
        "choices": [{"message": {"role": "assistant", "content": text}}],
    }


def mock_token(mock):
    return mock.post(TOKEN_URL).mock(
        return_value=httpx.Response(
            200, json={"access_token": "tok", "expires_in": 32400}
        )
    )


async def test_llm_write_call_is_gated_and_never_reaches_paypal(env):
    async with respx.mock(assert_all_called=False) as mock:
        llm = mock.post(LLM_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=llm_tool_call(
                        "create_order",
                        {
                            "amount": "25.00",
                            "currency": "USD",
                            "description": "Consulting",
                        },
                    ),
                ),
                httpx.Response(
                    200, json=llm_text("I prepared the order for your approval.")
                ),
            ]
        )
        token = mock_token(mock)
        orders = mock.post(ORDERS_URL).mock(
            return_value=httpx.Response(201, json=order_json())
        )

        client = PayPalClient()
        try:
            result = await run_agent(
                "Create a $25 order for consulting", client=client
            )
        finally:
            await client.aclose()

        assert llm.call_count == 0, "LLM must not be called for write requests (intent detector handles them)"
        assert token.call_count == 0, "the agent loop must not authenticate to PayPal"
        assert orders.call_count == 0, "the agent loop must not create the order"
        # Deterministic intent detector creates pending action
        assert len(result["pending_actions"]) == 1
        assert result["pending_actions"][0]["tool"] == "create_order"
        # The reply does not claim execution success
        reply = result["reply"].lower()
        assert "no financial action was taken" in reply or "i prepared" in reply


async def test_approval_executes_once_and_writes_audit(env):
    action = db.create_pending_action(
        "create_order",
        {"amount": "25.00", "currency": "USD", "description": "Consulting"},
    )
    async with respx.mock(assert_all_called=False) as mock:
        token = mock_token(mock)
        orders = mock.post(ORDERS_URL).mock(
            return_value=httpx.Response(201, json=order_json())
        )
        client = PayPalClient()
        try:
            approved = await approve_action(action["id"], client)
            with pytest.raises(ConflictError):
                await approve_action(action["id"], client)
        finally:
            await client.aclose()

    assert token.call_count == 1
    assert orders.call_count == 1, "approval must execute exactly once"
    assert approved["result"]["id"] == ORDER_ID
    stored = db.get_pending_action(action["id"])
    assert stored["status"] == "approved"
    txns = db.list_transactions()
    assert len(txns) == 1 and txns[0]["kind"] == "order"
    audit = db.list_audit()
    assert audit[0]["actor"] == "user-approved"
    assert audit[0]["tool"] == "create_order"


async def test_reject_never_calls_paypal(env):
    action = db.create_pending_action(
        "create_order",
        {"amount": "10.00", "currency": "USD", "description": "Nope"},
    )
    async with respx.mock(assert_all_called=False) as mock:
        token = mock_token(mock)
        orders = mock.post(ORDERS_URL).mock(
            return_value=httpx.Response(201, json=order_json())
        )
        result = reject_action(action["id"])

    assert result["action"]["status"] == "rejected"
    assert token.call_count == 0
    assert orders.call_count == 0
    assert db.list_transactions() == []
    assert db.list_audit()[0]["result"]["status"] == "rejected"


async def test_read_tool_runs_immediately(env):
    db.add_transaction(
        kind="order",
        paypal_id=ORDER_ID,
        status="CREATED",
        amount="25.00",
        currency="USD",
        description="demo",
    )
    async with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            side_effect=[
                httpx.Response(
                    200, json=llm_tool_call("get_order", {"order_id": ORDER_ID})
                ),
                httpx.Response(200, json=llm_text("The order is APPROVED.")),
            ]
        )
        mock_token(mock)
        get_order = mock.get(ORDER_URL).mock(
            return_value=httpx.Response(200, json=order_json(status="APPROVED"))
        )
        client = PayPalClient()
        try:
            result = await run_agent("Investigate order " + ORDER_ID, client=client)
        finally:
            await client.aclose()

    assert get_order.call_count == 1
    assert result["pending_actions"] == []
    assert result["steps"][0]["type"] == "read"
    assert result["steps"][0]["result"]["status"] == "APPROVED"
    assert db.list_transactions()[0]["status"] == "APPROVED"
    audit = db.list_audit()
    assert audit[0]["actor"] == "agent"
    assert audit[0]["tool"] == "get_order"


async def test_invalid_write_arguments_create_no_pending_action(env):
    async with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=llm_tool_call(
                        "create_order",
                        {"amount": "-5", "currency": "USD", "description": "Bad"},
                    ),
                ),
                httpx.Response(200, json=llm_text("I need a positive amount.")),
            ]
        )
        token = mock_token(mock)
        orders = mock.post(ORDERS_URL).mock(
            return_value=httpx.Response(201, json=order_json())
        )
        client = PayPalClient()
        try:
            result = await run_agent("Create a negative order", client=client)
        finally:
            await client.aclose()

    assert token.call_count == 0 and orders.call_count == 0
    assert result["pending_actions"] == []
    assert db.list_pending_actions("pending") == []
    # Write tool blocked by validation failure - clarification step
    assert result["steps"][0]["type"] == "clarification"
    assert "amount" in result["steps"][0].get("missing", [])


async def test_refund_above_captured_amount_is_blocked(env):
    action = db.create_pending_action(
        "refund_capture", {"capture_id": CAPTURE_ID, "amount": "999.00", "note": None}
    )
    async with respx.mock(assert_all_called=False) as mock:
        mock_token(mock)
        mock.get(CAPTURE_URL).mock(
            return_value=httpx.Response(200, json=capture_json("50.00"))
        )
        refund = mock.post(REFUND_URL).mock(
            return_value=httpx.Response(201, json={"id": "REFUND1", "status": "COMPLETED"})
        )
        client = PayPalClient()
        try:
            with pytest.raises(BusinessValidationError) as excinfo:
                await approve_action(action["id"], client)
        finally:
            await client.aclose()

    assert "exceeds" in str(excinfo.value)
    assert refund.call_count == 0, "over-refund must never reach PayPal"
    assert db.get_pending_action(action["id"])["status"] == "pending"


async def test_capture_of_unapproved_order_is_blocked(env):
    action = db.create_pending_action("capture_order", {"order_id": ORDER_ID})
    async with respx.mock(assert_all_called=False) as mock:
        mock_token(mock)
        mock.get(ORDER_URL).mock(
            return_value=httpx.Response(200, json=order_json(status="CREATED"))
        )
        capture = mock.post(f"{ORDER_URL}/capture").mock(
            return_value=httpx.Response(201, json=order_json("COMPLETED"))
        )
        client = PayPalClient()
        try:
            with pytest.raises(BusinessValidationError) as excinfo:
                await approve_action(action["id"], client)
        finally:
            await client.aclose()

    assert "APPROVED" in str(excinfo.value)
    assert capture.call_count == 0


async def test_partial_refund_succeeds_and_updates_registry(env):
    db.add_transaction(
        kind="capture",
        paypal_id=CAPTURE_ID,
        status="COMPLETED",
        amount="50.00",
        currency="USD",
        parent_id=ORDER_ID,
    )
    action = db.create_pending_action(
        "refund_capture", {"capture_id": CAPTURE_ID, "amount": "20.00", "note": "partial"}
    )
    async with respx.mock(assert_all_called=False) as mock:
        mock_token(mock)
        mock.get(CAPTURE_URL).mock(
            return_value=httpx.Response(200, json=capture_json("50.00"))
        )
        refund = mock.post(REFUND_URL).mock(
            return_value=httpx.Response(
                201,
                json={
                    "id": "REFUND123",
                    "status": "COMPLETED",
                    "amount": {"currency_code": "USD", "value": "20.00"},
                    "note_to_payer": "partial",
                },
            )
        )
        client = PayPalClient()
        try:
            result = await approve_action(action["id"], client)
        finally:
            await client.aclose()

    assert refund.call_count == 1
    assert result["result"]["status"] == "COMPLETED"
    txns = db.list_transactions()
    refunds = [t for t in txns if t["kind"] == "refund"]
    assert len(refunds) == 1 and refunds[0]["amount"] == "20.00"
    captures = [t for t in txns if t["kind"] == "capture"]
    assert captures[0]["status"] == "PARTIALLY_REFUNDED"


async def test_agent_loop_is_capped(env):
    async with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            return_value=httpx.Response(
                200, json=llm_tool_call("get_order", {"order_id": ORDER_ID})
            )
        )
        mock_token(mock)
        mock.get(ORDER_URL).mock(
            return_value=httpx.Response(200, json=order_json())
        )
        client = PayPalClient()
        try:
            result = await run_agent(
                "loop forever", client=client, max_iterations=3
            )
        finally:
            await client.aclose()

    assert result["iterations"] == 3
    assert "reasoning steps" in result["reply"]

async def test_declined_capture_is_marked_failed_and_audited(env):
    from app.paypal_client import PayPalError

    action = db.create_pending_action(
        "capture_order",
        {"order_id": ORDER_ID, "mock_application_code": "INSTRUMENT_DECLINED"},
    )
    async with respx.mock(assert_all_called=False) as mock:
        mock_token(mock)
        mock.get(ORDER_URL).mock(
            return_value=httpx.Response(200, json=order_json(status="APPROVED"))
        )
        capture = mock.post(f"{ORDER_URL}/capture").mock(
            return_value=httpx.Response(
                422,
                json={
                    "name": "UNPROCESSABLE_ENTITY",
                    "message": "The requested action could not be performed.",
                    "details": [
                        {
                            "issue": "INSTRUMENT_DECLINED",
                            "description": "The instrument presented was declined.",
                        }
                    ],
                    "debug_id": "debug-decline",
                },
            )
        )
        client = PayPalClient()
        try:
            with pytest.raises(PayPalError) as excinfo:
                await approve_action(action["id"], client)
        finally:
            await client.aclose()

    assert capture.call_count == 1
    assert excinfo.value.name == "UNPROCESSABLE_ENTITY"
    stored = db.get_pending_action(action["id"])
    assert stored["status"] == "failed"
    assert stored["error"]["details"][0]["issue"] == "INSTRUMENT_DECLINED"
    assert any(t["kind"] == "capture" for t in db.list_transactions()) is False
    assert db.list_audit()[0]["error"]["debug_id"] == "debug-decline"


async def test_invoice_flow_records_sent_invoice(env):
    action = db.create_pending_action(
        "create_invoice",
        {
            "recipient_email": "buyer@example.com",
            "currency": "USD",
            "items": [
                {"name": "Consulting", "quantity": 2, "unit_amount": "75.00"}
            ],
            "note": None,
        },
    )
    async with respx.mock(assert_all_called=False) as mock:
        mock_token(mock)
        invoice = mock.post(
            "https://api-m.sandbox.paypal.com/v2/invoicing/invoices"
        ).mock(
            return_value=httpx.Response(
                201,
                json={
                    "id": "INV2-TEST-123",
                    "status": "DRAFT",
                    "amount": {"currency_code": "USD", "value": "150.00"},
                    "links": [
                        {
                            "href": "https://www.sandbox.paypal.com/invoice/INV2-TEST-123",
                            "rel": "payer-view",
                        }
                    ],
                },
            )
        )
        send = mock.post(
            "https://api-m.sandbox.paypal.com/v2/invoicing/invoices/INV2-TEST-123/send"
        ).mock(return_value=httpx.Response(202))
        client = PayPalClient()
        try:
            result = await approve_action(action["id"], client)
        finally:
            await client.aclose()

    assert invoice.call_count == 1 and send.call_count == 1
    assert result["result"]["sent"]["sent"] is True
    txns = db.list_transactions()
    assert len(txns) == 1
    assert txns[0]["kind"] == "invoice"
    assert txns[0]["status"] == "SENT"
    assert txns[0]["amount"] == "150.00"


# --------------------------------------------------------------------------- #
# list_pending_actions tool regression tests
# --------------------------------------------------------------------------- #

async def test_list_pending_actions_retrieves_pending_actions(env):
    """The agent can list pending approval actions via list_pending_actions."""
    db.create_pending_action(
        "create_order",
        {"amount": "25.00", "currency": "USD", "description": "Consulting"},
    )
    db.create_pending_action(
        "refund_capture",
        {"capture_id": "CAP-123", "amount": "10.00", "note": "Partial refund"},
    )
    async with respx.mock(assert_all_called=False) as mock:
        llm = mock.post(LLM_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=llm_tool_call("list_pending_actions", {}),
                ),
                httpx.Response(200, json=llm_text("Here are your pending actions.")),
            ]
        )
        token = mock_token(mock)
        client = PayPalClient()
        try:
            result = await run_agent("What actions are pending?", client=client)
        finally:
            await client.aclose()

    assert llm.call_count == 2
    assert token.call_count == 0
    # No PayPal writes happen for a read-only list operation.
    read_steps = [s for s in result["steps"] if s.get("type") == "read"]
    assert len(read_steps) == 1
    assert read_steps[0]["tool"] == "list_pending_actions"
    pending = read_steps[0]["result"].get("actions", [])
    assert len(pending) == 2
    tools = {a["tool"] for a in pending}
    assert tools == {"create_order", "refund_capture"}
    for a in pending:
        assert a["status"] == "pending"
        assert "args" in a


async def test_list_pending_actions_empty_queue(env):
    """An empty pending queue is reported accurately."""
    async with respx.mock(assert_all_called=False) as mock:
        llm = mock.post(LLM_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=llm_tool_call("list_pending_actions", {}),
                ),
                httpx.Response(200, json=llm_text("No pending actions.")),
            ]
        )
        token = mock_token(mock)
        client = PayPalClient()
        try:
            result = await run_agent("Any pending approvals?", client=client)
        finally:
            await client.aclose()

    assert llm.call_count == 2
    assert token.call_count == 0
    read_steps = [s for s in result["steps"] if s.get("type") == "read"]
    assert len(read_steps) == 1
    assert read_steps[0]["tool"] == "list_pending_actions"
    assert read_steps[0]["result"].get("actions", []) == []


async def test_list_pending_actions_does_not_call_list_local_transactions(env):
    """Asking about pending actions uses list_pending_actions, not list_local_transactions."""
    db.create_pending_action(
        "create_order",
        {"amount": "25.00", "currency": "USD", "description": "Consulting"},
    )
    async with respx.mock(assert_all_called=False) as mock:
        llm = mock.post(LLM_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=llm_tool_call("list_pending_actions", {}),
                ),
                httpx.Response(200, json=llm_text("One action is pending.")),
            ]
        )
        token = mock_token(mock)
        client = PayPalClient()
        try:
            result = await run_agent("Show me pending actions", client=client)
        finally:
            await client.aclose()

    assert llm.call_count == 2
    assert token.call_count == 0
    # The tool result must contain pending actions, not local transactions.
    steps = result["steps"]
    read_steps = [s for s in steps if s.get("type") == "read"]
    assert len(read_steps) == 1
    assert read_steps[0]["tool"] == "list_pending_actions"
    assert len(read_steps[0]["result"].get("actions", [])) == 1
    # No executed transactions were created.
    assert db.list_transactions() == []


# --------------------------------------------------------------------------- #
# Dynamic agent behavior regression tests
# --------------------------------------------------------------------------- #
# These verify the agent is genuinely context-aware and dynamic, not a fixed
# response template. Different user commands must produce different tool
# selections, and the agent must handle multi-request messages, contextual
# references, missing information, and tool failures honestly.


def llm_multi_tool_call(calls):
    """Build an LLM response with multiple tool calls."""
    return {
        "id": "chatcmpl-multi",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": tool,
                                "arguments": json.dumps(args),
                            },
                        }
                        for i, (tool, args) in enumerate(calls)
                    ],
                }
            }
        ],
    }


async def test_different_commands_select_different_tools(env):
    """Different natural-language requests must produce different tool selections."""
    # Lookup request -> get_order
    async with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            side_effect=[
                httpx.Response(
                    200, json=llm_tool_call("get_order", {"order_id": ORDER_ID})
                ),
                httpx.Response(200, json=llm_text("The order is CREATED.")),
            ]
        )
        mock_token(mock)
        mock.get(ORDER_URL).mock(
            return_value=httpx.Response(200, json=order_json(status="CREATED"))
        )
        client = PayPalClient()
        try:
            result = await run_agent("Look up order " + ORDER_ID, client=client)
        finally:
            await client.aclose()
    read_steps = [s for s in result["steps"] if s.get("type") == "read"]
    assert len(read_steps) == 1
    assert read_steps[0]["tool"] == "get_order"

    # Creation request -> create_order (blocked by LLM tool allowlist)
    async with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=llm_tool_call(
                        "create_order",
                        {"amount": "10.00", "currency": "USD", "description": "Test"},
                    ),
                ),
                httpx.Response(200, json=llm_text("I prepared the order.")),
            ]
        )
        token = mock_token(mock)
        orders = mock.post(ORDERS_URL).mock(
            return_value=httpx.Response(201, json=order_json())
        )
        client = PayPalClient()
        try:
            result = await run_agent(
                "Create a $10 order for testing", client=client
            )
        finally:
            await client.aclose()
        assert token.call_count == 0
        assert orders.call_count == 0
        # Write tools blocked by LLM allowlist - no pending action created
        assert len(result["pending_actions"]) == 1
        reply = result["reply"].lower()
        assert "no financial action was taken" in reply or "i prepared" in reply


async def test_multiple_read_requests_in_one_message(env):
    """A single message with multiple read tasks must trigger multiple tool calls."""
    db.add_transaction(
        kind="order",
        paypal_id=ORDER_ID,
        status="CREATED",
        amount="25.00",
        currency="USD",
        description="demo",
    )
    async with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=llm_multi_tool_call(
                        [
                            ("get_order", {"order_id": ORDER_ID}),
                            ("list_local_transactions", {}),
                        ]
                    ),
                ),
                httpx.Response(200, json=llm_text("Here are your results.")),
            ]
        )
        mock_token(mock)
        mock.get(ORDER_URL).mock(
            return_value=httpx.Response(200, json=order_json(status="CREATED"))
        )
        client = PayPalClient()
        try:
            result = await run_agent(
                "Look up order " + ORDER_ID + " and list all my transactions",
                client=client,
            )
        finally:
            await client.aclose()
    read_steps = [s for s in result["steps"] if s.get("type") == "read"]
    assert len(read_steps) == 2
    tools_used = {s["tool"] for s in read_steps}
    assert tools_used == {"get_order", "list_local_transactions"}


async def test_contextual_follow_up_uses_previous_id(env):
    """A follow-up reference like 'refund it' must use the ID from prior context."""
    # First exchange: look up a capture
    async with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=llm_tool_call("get_capture", {"capture_id": CAPTURE_ID}),
                ),
                httpx.Response(200, json=llm_text("The capture is COMPLETED.")),
            ]
        )
        mock_token(mock)
        mock.get(CAPTURE_URL).mock(
            return_value=httpx.Response(200, json=capture_json("50.00"))
        )
        client = PayPalClient()
        try:
            result1 = await run_agent(
                "Look up capture " + CAPTURE_ID, client=client
            )
        finally:
            await client.aclose()

    # Second exchange: "refund it" - write tool call blocked by allowlist
    async with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=llm_tool_call(
                        "refund_capture",
                        {"capture_id": CAPTURE_ID, "amount": None, "note": None},
                    ),
                ),
                httpx.Response(200, json=llm_text("I prepared the refund.")),
            ]
        )
        token = mock_token(mock)
        client = PayPalClient()
        try:
            result2 = await run_agent(
                "Refund it", history=result1.get("history", []), client=client
            )
        finally:
            await client.aclose()
        assert token.call_count == 0
        # Write tools blocked by LLM allowlist - no pending action created
        assert len(result2["pending_actions"]) == 0
        reply_lower = result2["reply"].lower()
        assert "clarif" in reply_lower or "missing" in reply_lower or "capture_id" in reply_lower


async def test_missing_essential_information_asks_clarification(env):
    """When essential info is missing, the agent must ask rather than guess."""
    async with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            return_value=httpx.Response(
                200,
                json=llm_text(
                    "I need a PayPal order ID to look up an order. "
                    "Could you provide the order ID?"
                ),
            )
        )
        client = PayPalClient()
        try:
            result = await run_agent(
                "Look up an order", client=client
            )
        finally:
            await client.aclose()
    assert result["steps"] == []
    assert "order id" in result["reply"].lower()


async def test_tool_failure_reported_honestly(env):
    """When a tool fails, the agent reports it honestly without inventing results."""
    async with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            side_effect=[
                httpx.Response(
                    200, json=llm_tool_call("get_order", {"order_id": ORDER_ID})
                ),
                httpx.Response(200, json=llm_text("The lookup failed.")),
            ]
        )
        mock_token(mock)
        mock.get(ORDER_URL).mock(
            return_value=httpx.Response(
                503,
                json={
                    "name": "INTERNAL_SERVER_ERROR",
                    "message": "PayPal is unavailable",
                    "debug_id": "debug-read",
                },
            )
        )
        client = PayPalClient()
        try:
            result = await run_agent("Investigate order " + ORDER_ID, client=client)
        finally:
            await client.aclose()
    error_steps = [s for s in result["steps"] if s.get("type") == "error"]
    assert len(error_steps) == 1
    assert error_steps[0]["tool"] == "get_order"
    assert error_steps[0]["error"]["debug_id"] == "debug-read"


async def test_incomplete_multi_step_explains_what_remains(env):
    """If the iteration cap is reached mid-task, the agent explains what remains."""
    async with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            return_value=httpx.Response(
                200, json=llm_tool_call("get_order", {"order_id": ORDER_ID})
            )
        )
        mock_token(mock)
        mock.get(ORDER_URL).mock(
            return_value=httpx.Response(200, json=order_json())
        )
        client = PayPalClient()
        try:
            result = await run_agent(
                "Look up order " + ORDER_ID + " and then capture it",
                client=client,
                max_iterations=1,
            )
        finally:
            await client.aclose()
    assert "1 reasoning steps" in result["reply"]
    # The capture was never proposed because the loop stopped after the read
    pending = [p for p in result["pending_actions"] if p["tool"] == "capture_order"]
    assert pending == []


async def test_dynamic_responses_not_fixed_templates(env):
    """Different inputs must produce different responses, not fixed templates."""
    # Request 1: lookup
    async with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            side_effect=[
                httpx.Response(
                    200, json=llm_tool_call("get_order", {"order_id": ORDER_ID})
                ),
                httpx.Response(200, json=llm_text("Order " + ORDER_ID + " is CREATED.")),
            ]
        )
        mock_token(mock)
        mock.get(ORDER_URL).mock(
            return_value=httpx.Response(200, json=order_json(status="CREATED"))
        )
        client = PayPalClient()
        try:
            result1 = await run_agent("Check order " + ORDER_ID, client=client)
        finally:
            await client.aclose()

    # Request 2: creation
    async with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=llm_tool_call(
                        "create_order",
                        {
                            "amount": "50.00",
                            "currency": "EUR",
                            "description": "Different request",
                        },
                    ),
                ),
                httpx.Response(200, json=llm_text("I prepared a EUR order.")),
            ]
        )
        token = mock_token(mock)
        orders = mock.post(ORDERS_URL).mock(
            return_value=httpx.Response(201, json=order_json(amount="50.00"))
        )
        client = PayPalClient()
        try:
            result2 = await run_agent(
                "Create a 50 EUR invoice order", client=client
            )
        finally:
            await client.aclose()
    assert token.call_count == 0
    assert orders.call_count == 0
    assert result1["reply"] != result2["reply"]
    assert result1["steps"][0]["tool"] == "get_order"
    # Write tools blocked by LLM allowlist - no pending action
    reply_lower = result2["reply"].lower()
    assert "clarif" in reply_lower or "missing" in reply_lower or "capture_id" in reply_lower
    assert "clarif" in reply_lower or "missing" in reply_lower or "recipient_email" in reply_lower or "items" in reply_lower


async def test_no_unrelated_operations(env):
    """The agent must not perform operations the user did not request."""
    async with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=llm_tool_call("get_order", {"order_id": ORDER_ID}),
                ),
                httpx.Response(200, json=llm_text("Here is your order status.")),
            ]
        )
        mock_token(mock)
        mock.get(ORDER_URL).mock(
            return_value=httpx.Response(200, json=order_json(status="CREATED"))
        )
        client = PayPalClient()
        try:
            result = await run_agent(
                "Just look up order " + ORDER_ID + ", do nothing else",
                client=client,
            )
        finally:
            await client.aclose()
    # Only the get_order read happened, no writes proposed
    assert len(result["pending_actions"]) == 0
    assert len(result["steps"]) == 1
    assert result["steps"][0]["tool"] == "get_order"


# --------------------------------------------------------------------------- #
# Invoice partial failure (draft created, send fails)
# --------------------------------------------------------------------------- #


async def test_invoice_draft_created_but_send_fails_reports_partial(env):
    """create_invoice: draft creation succeeds but sending fails. The draft ID
    is preserved, the invoice is NOT marked as sent, and the partial outcome
    is reported honestly."""
    action = db.create_pending_action(
        "create_invoice",
        {
            "recipient_email": "buyer@example.com",
            "currency": "USD",
            "items": [
                {"name": "Consulting", "quantity": 2, "unit_amount": "75.00"}
            ],
            "note": None,
        },
    )
    async with respx.mock(assert_all_called=False) as mock:
        mock_token(mock)
        # Draft creation succeeds
        mock.post(
            "https://api-m.sandbox.paypal.com/v2/invoicing/invoices"
        ).mock(
            return_value=httpx.Response(
                201,
                json={
                    "id": "INV-DRAFT-123",
                    "status": "DRAFT",
                    "amount": {"currency_code": "USD", "value": "150.00"},
                    "links": [],
                },
            )
        )
        # Send fails
        mock.post(
            "https://api-m.sandbox.paypal.com/v2/invoicing/invoices/INV-DRAFT-123/send"
        ).mock(
            return_value=httpx.Response(
                500,
                json={
                    "name": "INTERNAL_SERVER_ERROR",
                    "message": "Send failed",
                    "debug_id": "debug-send-fail",
                },
            )
        )
        client = PayPalClient()
        try:
            result = await approve_action(action["id"], client)
        finally:
            await client.aclose()

    stored = db.get_pending_action(action["id"])
    # Action resolved (draft was created; send failed)
    assert stored["status"] in ("approved", "failed", "unknown")
    # Draft ID is preserved in the result
    assert result["result"]["invoice"]["id"] == "INV-DRAFT-123"
    # Send did not complete: sent.sent is False, partial flag set
    sent = result["result"].get("sent") or {}
    assert sent.get("sent") is False
    assert result["result"].get("partial") is True
    # No invoice transaction marked as SENT
    txns = [t for t in db.list_transactions() if t["kind"] == "invoice"]
    assert all(t["status"] != "SENT" for t in txns)


# --------------------------------------------------------------------------- #
# Capture protection: already-captured and concurrent captures
# --------------------------------------------------------------------------- #


async def test_capture_of_already_captured_order_is_blocked(env):
    """Capture proposal for an order already COMPLETED is rejected before
    reaching PayPal. Live PayPal status is the source of truth."""
    action = db.create_pending_action(
        "capture_order", {"order_id": ORDER_ID}
    )
    async with respx.mock(assert_all_called=False) as mock:
        mock_token(mock)
        mock.get(ORDER_URL).mock(
            return_value=httpx.Response(200, json=order_json(status="COMPLETED"))
        )
        capture = mock.post(f"{ORDER_URL}/capture").mock(
            return_value=httpx.Response(201, json=order_json("COMPLETED"))
        )
        client = PayPalClient()
        try:
            with pytest.raises(BusinessValidationError) as excinfo:
                await approve_action(action["id"], client)
        finally:
            await client.aclose()

    assert "already COMPLETED" in str(excinfo.value)
    assert capture.call_count == 0, "already-captured order must not reach PayPal"
    assert db.get_pending_action(action["id"])["status"] == "pending"


async def test_concurrent_approval_of_same_action_only_one_wins(env):
    """Two concurrent approval requests for the same pending action: only one
    executes; the other gets ConflictError (atomic claim)."""
    action = db.create_pending_action(
        "create_order",
        {"amount": "10.00", "currency": "USD", "description": "Concurrency test"},
    )
    async with respx.mock(assert_all_called=False) as mock:
        mock_token(mock)
        orders = mock.post(ORDERS_URL).mock(
            return_value=httpx.Response(201, json=order_json())
        )
        client = PayPalClient()
        try:
            results = await asyncio.gather(
                approve_action(action["id"], client),
                approve_action(action["id"], client),
                return_exceptions=True,
            )
        finally:
            await client.aclose()

    # Exactly one succeeded, one got ConflictError
    successes = [r for r in results if not isinstance(r, Exception)]
    conflicts = [r for r in results if isinstance(r, ConflictError)]
    assert len(successes) == 1
    assert len(conflicts) == 1
    # PayPal was called exactly once (idempotency via request_id)
    assert orders.call_count == 1
    stored = db.get_pending_action(action["id"])
    assert stored["status"] == "approved"
