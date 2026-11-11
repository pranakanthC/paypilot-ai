"""Regression tests for scripts/verify_sandbox.py async behavior.

These tests run fully offline with respx and the shared `env` fixture.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx

from app.paypal_client import PayPalClient, PayPalError

TOKEN_URL = "https://api-m.sandbox.paypal.com/v1/oauth2/token"
ORDERS_URL = "https://api-m.sandbox.paypal.com/v2/checkout/orders"

# Existing PayPal Sandbox order to verify/approve (matches the script constant).
EXISTING_ORDER_ID = "4KX64890GG2954126"
EXISTING_ORDER_URL = f"https://api-m.sandbox.paypal.com/v2/checkout/orders/{EXISTING_ORDER_ID}"
EXISTING_ORDER_CAPTURE_URL = f"{EXISTING_ORDER_URL}/capture"
CAPTURE_ID = "3C679366HH908993F"
CAPTURE_URL = f"https://api-m.sandbox.paypal.com/v2/payments/captures/{CAPTURE_ID}"
REFUND_URL = f"{CAPTURE_URL}/refund"


def order_json(status="CREATED", amount="10.00", include_approval_link=True):
    links = [
        {
            "href": f"https://www.sandbox.paypal.com/checkoutnow?token={EXISTING_ORDER_ID}",
            "rel": "payer-action",
        }
    ]
    if include_approval_link:
        links.append(
            {
                "href": f"https://www.sandbox.paypal.com/cgi-bin/webscr?cmd=_express-checkout&token={EXISTING_ORDER_ID}",
                "rel": "approve",
            }
        )
    return {
        "id": EXISTING_ORDER_ID,
        "intent": "CAPTURE",
        "status": status,
        "purchase_units": [
            {
                "reference_id": "default",
                "amount": {"currency_code": "USD", "value": amount},
            }
        ],
        "links": links,
    }


def capture_body():
    return {
        "id": EXISTING_ORDER_ID,
        "status": "COMPLETED",
        "purchase_units": [
            {
                "payments": {
                    "captures": [
                        {
                            "id": CAPTURE_ID,
                            "status": "COMPLETED",
                            "amount": {"currency_code": "USD", "value": "10.00"},
                        }
                    ]
                }
            }
        ],
    }


def _mock_successful_sandbox(mock):
    mock.post(TOKEN_URL).mock(
        return_value=httpx.Response(
            200, json={"access_token": "tok", "expires_in": 32400}
        )
    )
    mock.get(EXISTING_ORDER_URL).mock(
        return_value=httpx.Response(200, json=order_json())
    )
    mock.post(EXISTING_ORDER_CAPTURE_URL).mock(
        return_value=httpx.Response(201, json=capture_body())
    )
    mock.get(CAPTURE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": CAPTURE_ID,
                "status": "COMPLETED",
                "amount": {"currency_code": "USD", "value": "10.00"},
            },
        )
    )
    mock.post(REFUND_URL).mock(
        return_value=httpx.Response(
            201,
            json={
                "id": "REFUND-1",
                "status": "COMPLETED",
                "amount": {"currency_code": "USD", "value": "5.00"},
            },
        )
    )


async def test_main_is_async_and_returns_int(env):
    import scripts.verify_sandbox as vs

    assert asyncio.iscoroutinefunction(vs.main)
    with respx.mock(assert_all_called=False) as mock:
        _mock_successful_sandbox(mock)
        code = await vs.main()
    assert isinstance(code, int)
    assert code == 0


async def test_all_async_operations_are_awaited(env, monkeypatch):
    """Every async PayPalClient operation is awaited (never left as a coroutine).

    The script must not call create_order (no new order) and must await
    get_order, capture_order, get_capture, refund_capture, and aclose.
    """
    import scripts.verify_sandbox as vs

    calls = []

    class FakeClient:
        def __init__(self, settings):
            self.settings = settings

        async def get_access_token(self, force=False):
            calls.append("get_access_token")
            return "tok"

        async def get_order(self, order_id):
            calls.append("get_order")
            return order_json(include_approval_link=True)

        async def capture_order(self, order_id, **kwargs):
            calls.append("capture_order")
            # FakeClient bypasses _normalize_order, so return data in the
            # structure the script accesses directly (captures under purchase_units[].captures).
            return {
                "id": order_id,
                "status": "COMPLETED",
                "purchase_units": [
                    {
                        "captures": [
                            {
                                "id": CAPTURE_ID,
                                "status": "COMPLETED",
                                "amount": {"currency_code": "USD", "value": "10.00"},
                            }
                        ]
                    }
                ],
            }

        async def get_capture(self, capture_id):
            calls.append("get_capture")
            return {"id": capture_id, "status": "COMPLETED"}

        async def refund_capture(self, capture_id, **kwargs):
            calls.append("refund_capture")
            return {"id": "REF-1", "status": "COMPLETED"}

        async def aclose(self):
            calls.append("aclose")

    import app.paypal_client as pp

    monkeypatch.setattr(pp, "PayPalClient", FakeClient)
    monkeypatch.setattr(pp, "PayPalError", PayPalError)

    code = await vs.main()
    assert code == 0
    assert calls == [
        "get_access_token",
        "get_order",
        "capture_order",
        "get_capture",
        "refund_capture",
        "aclose",
    ]
    assert "create_order" not in calls, "no new order must be created"


async def test_approval_url_extracted_and_printed(env, capsys):
    """The buyer approval URL is extracted from order links and printed clearly."""
    import scripts.verify_sandbox as vs

    with respx.mock(assert_all_called=False) as mock:
        _mock_successful_sandbox(mock)
        await vs.main()

    captured = capsys.readouterr()
    output = captured.out
    assert "=== Buyer Approval URL ===" in output
    assert (
        f"https://www.sandbox.paypal.com/cgi-bin/webscr?cmd=_express-checkout&token={EXISTING_ORDER_ID}"
        in output
    )
    assert "Open the URL above in a browser" in output


async def test_approval_url_missing_handled_gracefully(env, capsys):
    """When the order has no approve/payer-action link, the script reports it
    instead of crashing."""
    import scripts.verify_sandbox as vs

    with respx.mock(assert_all_called=False) as mock:
        mock.post(TOKEN_URL).mock(
            return_value=httpx.Response(
                200, json={"access_token": "tok", "expires_in": 32400}
            )
        )
        mock.get(EXISTING_ORDER_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": EXISTING_ORDER_ID,
                    "status": "COMPLETED",
                    "links": [],
                },
            )
        )
        code = await vs.main()

    captured = capsys.readouterr()
    output = captured.out
    assert "=== Buyer Approval URL ===" not in output
    assert "no approve/payer-action link in order" in output
    # The script still completes (non-zero since a step failed).
    assert code == 1


async def test_access_token_never_printed(env, capsys):
    import scripts.verify_sandbox as vs

    with respx.mock(assert_all_called=False) as mock:
        mock.post(TOKEN_URL).mock(
            return_value=httpx.Response(
                200, json={"access_token": "super-secret-access-token", "expires_in": 32400}
            )
        )
        mock.get(EXISTING_ORDER_URL).mock(
            return_value=httpx.Response(200, json=order_json())
        )
        mock.post(EXISTING_ORDER_CAPTURE_URL).mock(
            return_value=httpx.Response(201, json=capture_body())
        )
        mock.get(CAPTURE_URL).mock(
            return_value=httpx.Response(
                200,
                json={"id": CAPTURE_ID, "status": "COMPLETED"},
            )
        )
        mock.post(REFUND_URL).mock(
            return_value=httpx.Response(
                201, json={"id": "REFUND-1", "status": "COMPLETED"}
            )
        )
        await vs.main()

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "super-secret-access-token" not in output
    assert "Bearer " not in output


async def test_missing_credentials_handled_gracefully(env, monkeypatch):
    import scripts.verify_sandbox as vs

    monkeypatch.delenv("PAYPAL_CLIENT_ID", raising=False)
    monkeypatch.delenv("PAYPAL_CLIENT_SECRET", raising=False)

    code = await vs.main()
    assert code == 1


async def test_api_error_handled_gracefully(env):
    """A PayPal API error fails the step but never crashes the script."""
    import scripts.verify_sandbox as vs

    with respx.mock(assert_all_called=False) as mock:
        mock.post(TOKEN_URL).mock(
            return_value=httpx.Response(
                200, json={"access_token": "tok", "expires_in": 32400}
            )
        )
        mock.get(EXISTING_ORDER_URL).mock(
            return_value=httpx.Response(
                400,
                json={
                    "name": "INVALID_REQUEST",
                    "message": "bad request",
                    "debug_id": "dbg-123",
                },
            )
        )
        code = await vs.main()
    assert code == 1


async def test_timeout_handled_gracefully(env):
    """A connect timeout fails gracefully instead of propagating."""
    import scripts.verify_sandbox as vs

    with respx.mock(assert_all_called=False) as mock:
        mock.post(TOKEN_URL).mock(side_effect=httpx.ConnectTimeout("connect timed out"))
        code = await vs.main()
    assert code == 1


async def test_main_coroutine_runs_successfully(env):
    """The main coroutine (run by asyncio.run in __main__) executes the full workflow."""
    import scripts.verify_sandbox as vs

    with respx.mock(assert_all_called=False) as mock:
        _mock_successful_sandbox(mock)
        code = await vs.main()
    assert code == 0


# --------------------------------------------------------------------------- #
# Approval URL extraction (scripts.verify_sandbox.extract_approval_url)
# --------------------------------------------------------------------------- #


def test_extract_approval_url_prefers_approve_rel():
    import scripts.verify_sandbox as vs

    order = {
        "id": EXISTING_ORDER_ID,
        "status": "CREATED",
        "links": [
            {
                "href": f"https://www.sandbox.paypal.com/checkoutnow?token={EXISTING_ORDER_ID}",
                "rel": "payer-action",
            },
            {
                "href": f"https://www.sandbox.paypal.com/cgi-bin/webscr?cmd=_express-checkout&token={EXISTING_ORDER_ID}",
                "rel": "approve",
            },
        ],
    }
    url = vs.extract_approval_url(order)
    assert url == (
        f"https://www.sandbox.paypal.com/cgi-bin/webscr?cmd=_express-checkout&token={EXISTING_ORDER_ID}"
    )


def test_extract_approval_url_falls_back_to_payer_action_rel():
    import scripts.verify_sandbox as vs

    order = {
        "id": EXISTING_ORDER_ID,
        "status": "CREATED",
        "links": [
            {
                "href": f"https://www.sandbox.paypal.com/checkoutnow?token={EXISTING_ORDER_ID}",
                "rel": "payer-action",
            }
        ],
    }
    url = vs.extract_approval_url(order)
    assert url == (
        f"https://www.sandbox.paypal.com/checkoutnow?token={EXISTING_ORDER_ID}"
    )


def test_extract_approval_url_returns_none_when_no_approval_link():
    import scripts.verify_sandbox as vs

    order = {
        "id": EXISTING_ORDER_ID,
        "status": "COMPLETED",
        "links": [
            {"href": "https://api-m.sandbox.paypal.com/v2/checkout/orders/4KX64890GG2954126", "rel": "self"}
        ],
    }
    assert vs.extract_approval_url(order) is None


async def test_create_order_does_not_auto_capture_human_gate_preserved(env):
    """create_order must only create an order (intent CAPTURE, user_action CONTINUE);
    it must not initiate a capture. Capture remains an explicit merchant-side
    action gated by human approval."""
    import scripts.verify_sandbox as vs

    captured_body = {}

    def create_order_handler(request):
        captured_body.update(json.loads(request.content))
        return httpx.Response(201, json=order_json())

    async with respx.mock(assert_all_called=False) as mock:
        mock.post(TOKEN_URL).mock(
            return_value=httpx.Response(
                200, json={"access_token": "tok", "expires_in": 32400}
            )
        )
        mock.post(ORDERS_URL).mock(side_effect=create_order_handler)
        # No capture URL is mocked: create_order must not call it.
        client = PayPalClient()
        try:
            await client.create_order("10.00", "USD", "Test order")
        finally:
            await client.aclose()

    body = captured_body
    assert body["intent"] == "CAPTURE"
    experience = body["payment_source"]["paypal"]["experience_context"]
    assert experience["user_action"] == "CONTINUE"
    # The create_order request body contains no capture instruction at top level.
    assert "capture" not in body
