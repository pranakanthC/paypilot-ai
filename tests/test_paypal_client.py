import json

import httpx
import pytest
import respx

from app.paypal_client import PayPalClient, PayPalError

TOKEN_URL = "https://api-m.sandbox.paypal.com/v1/oauth2/token"
ORDERS_URL = "https://api-m.sandbox.paypal.com/v2/checkout/orders"
ORDER_URL = "https://api-m.sandbox.paypal.com/v2/checkout/orders/5O190127TN364715T"
CAPTURE_URL = "https://api-m.sandbox.paypal.com/v2/payments/captures/3C679366HH908993F"
REFUND_URL = (
    "https://api-m.sandbox.paypal.com/v2/payments/captures/3C679366HH908993F/refund"
)
ORDER_ID = "5O190127TN364715T"
CAPTURE_ID = "3C679366HH908993F"


def order_json():
    return {
        "id": ORDER_ID,
        "intent": "CAPTURE",
        "status": "CREATED",
        "purchase_units": [
            {
                "reference_id": "default",
                "amount": {"currency_code": "USD", "value": "49.99"},
            }
        ],
        "links": [
            {
                "href": f"https://www.sandbox.paypal.com/checkoutnow?token={ORDER_ID}",
                "rel": "payer-action",
            }
        ],
    }


async def test_token_cache_mock_header_and_error_mapping(env):
    seen = {}

    def capture_handler(request):
        seen["mock_header"] = request.headers.get("paypal-mock-response")
        return httpx.Response(
            422,
            json={
                "name": "UNPROCESSABLE_ENTITY",
                "message": "The requested action could not be performed.",
                "details": [
                    {
                        "issue": "INSTRUMENT_DECLINED",
                        "description": "The instrument was declined.",
                    }
                ],
                "debug_id": "debug-123",
            },
        )

    async with respx.mock(assert_all_called=False) as mock:
        token = mock.post(TOKEN_URL).mock(
            return_value=httpx.Response(
                200, json={"access_token": "tok", "expires_in": 32400}
            )
        )
        mock.post(ORDERS_URL).mock(
            return_value=httpx.Response(201, json=order_json())
        )
        mock.get(ORDER_URL).mock(
            return_value=httpx.Response(200, json=order_json())
        )
        mock.post(f"{ORDER_URL}/capture").mock(side_effect=capture_handler)

        client = PayPalClient()
        try:
            order = await client.create_order("49.99", "USD", "Demo")
            await client.get_order(ORDER_ID)
            with pytest.raises(PayPalError) as excinfo:
                await client.capture_order(ORDER_ID, "INSTRUMENT_DECLINED")
        finally:
            await client.aclose()

    assert order["id"] == ORDER_ID
    assert order["amount"] == "49.99"
    assert order["approval_url"].startswith("https://www.sandbox.paypal.com/")
    assert token.call_count == 1, "the OAuth token must be cached across calls"
    assert json.loads(seen["mock_header"]) == {
        "mock_application_codes": "INSTRUMENT_DECLINED"
    }
    assert excinfo.value.name == "UNPROCESSABLE_ENTITY"
    assert excinfo.value.debug_id == "debug-123"
    assert excinfo.value.details[0]["issue"] == "INSTRUMENT_DECLINED"


async def test_refund_request_and_normalization(env):
    captured_body = {}

    def refund_handler(request):
        captured_body.update(json.loads(request.content))
        return httpx.Response(
            201,
            json={
                "id": "REFUND-1",
                "status": "COMPLETED",
                "amount": {"currency_code": "USD", "value": "10.00"},
                "note_to_payer": "partial refund",
            },
        )

    async with respx.mock(assert_all_called=False) as mock:
        mock.post(TOKEN_URL).mock(
            return_value=httpx.Response(
                200, json={"access_token": "tok", "expires_in": 32400}
            )
        )
        mock.post(REFUND_URL).mock(side_effect=refund_handler)

        client = PayPalClient()
        try:
            refund = await client.refund_capture(
                CAPTURE_ID, amount="10.00", currency="USD", note="partial refund"
            )
        finally:
            await client.aclose()

    assert captured_body == {
        "amount": {"value": "10.00", "currency_code": "USD"},
        "note_to_payer": "partial refund",
    }
    assert refund == {
        "id": "REFUND-1",
        "status": "COMPLETED",
        "amount": "10.00",
        "currency": "USD",
        "note_to_payer": "partial refund",
        "create_time": None,
        "update_time": None,
        "links": [],
    }
