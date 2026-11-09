import json

import httpx
import pytest
import respx

from app import db
from app.agent import run_agent
from app.paypal_client import PayPalClient
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

        assert llm.call_count == 2
        assert token.call_count == 0, "the agent loop must not authenticate to PayPal"
        assert orders.call_count == 0, "the agent loop must not create the order"
        assert len(result["pending_actions"]) == 1
        action = result["pending_actions"][0]
        assert action["tool"] == "create_order"
        assert action["status"] == "pending"
        assert db.list_transactions() == []


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
    assert result["steps"][0]["type"] == "error"
    assert db.list_audit()[0]["error"]["type"] == "validation"


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
