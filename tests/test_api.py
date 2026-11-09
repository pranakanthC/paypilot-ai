import httpx
import respx
from fastapi.testclient import TestClient

from app import db
from app.main import app
from tests.test_approval_gate import (
    LLM_URL,
    ORDERS_URL,
    ORDER_ID,
    TOKEN_URL,
    llm_text,
    llm_tool_call,
    mock_token,
    order_json,
)


def test_health_and_static_page(env):
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "PayPilot AI" in page.text
        health = client.get("/api/health").json()
        assert health["status"] == "ok"
        assert health["paypal_configured"] is True
        assert health["llm_configured"] is True


def test_chat_endpoint_with_mocked_llm(env):
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            return_value=httpx.Response(
                200, json=llm_text("I can only explain what the PayPal API returns.")
            )
        )
        with TestClient(app) as client:
            response = client.post(
                "/api/chat", json={"message": "What can you do?", "history": []}
            )
    assert response.status_code == 200
    body = response.json()
    assert "PayPal API" in body["reply"]
    assert body["steps"] == []


def test_chat_llm_failure_is_clean_json(env):
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            return_value=httpx.Response(
                401, json={"error": {"message": "Invalid API key"}}
            )
        )
        with TestClient(app) as client:
            response = client.post("/api/chat", json={"message": "hello"})
    assert response.status_code == 502
    error = response.json()["error"]
    assert error["type"] == "llm"
    assert "Invalid API key" in error["message"]


def test_demo_seed_records_transaction_and_audit(env):
    with respx.mock(assert_all_called=False) as mock:
        mock_token(mock)
        orders = mock.post(ORDERS_URL).mock(
            return_value=httpx.Response(201, json=order_json())
        )
        with TestClient(app) as client:
            response = client.post("/api/demo/seed", json={})
            txns = client.get("/api/transactions").json()["transactions"]
            audit = client.get("/api/audit").json()["audit"]
    assert response.status_code == 200
    body = response.json()
    assert body["order"]["id"] == ORDER_ID
    assert body["approval_url"].startswith("https://www.sandbox.paypal.com/")
    assert "sandbox" in body["instructions"].lower()
    assert orders.call_count == 1
    assert txns[0]["kind"] == "order"
    assert audit[0]["tool"] == "create_order"


def test_approve_and_reject_endpoints(env):
    approve_action = db.create_pending_action(
        "create_order",
        {"amount": "15.00", "currency": "USD", "description": "API test"},
    )
    reject_action = db.create_pending_action(
        "create_order",
        {"amount": "12.00", "currency": "USD", "description": "Reject me"},
    )
    with respx.mock(assert_all_called=False) as mock:
        mock_token(mock)
        orders = mock.post(ORDERS_URL).mock(
            return_value=httpx.Response(201, json=order_json(amount="15.00"))
        )
        with TestClient(app) as client:
            approved = client.post(
                f"/api/actions/{approve_action['id']}/approve"
            )
            rejected = client.post(
                f"/api/actions/{reject_action['id']}/reject"
            )
            pending = client.get("/api/pending").json()["pending"]

    assert approved.status_code == 200
    assert approved.json()["action"]["status"] == "approved"
    assert rejected.status_code == 200
    assert rejected.json()["action"]["status"] == "rejected"
    assert pending == []
    assert orders.call_count == 1


def test_missing_action_and_validation_errors(env):
    with TestClient(app) as client:
        missing = client.post("/api/actions/does-not-exist/approve")
        empty_chat = client.post("/api/chat", json={"message": ""})
    assert missing.status_code == 404
    assert missing.json()["error"]["type"] == "not_found"
    assert empty_chat.status_code == 422
    assert empty_chat.json()["error"]["type"] == "request_validation"


def test_missing_paypal_config_returns_clean_error(env, monkeypatch):
    monkeypatch.delenv("PAYPAL_CLIENT_ID", raising=False)
    monkeypatch.delenv("PAYPAL_CLIENT_SECRET", raising=False)
    with TestClient(app) as client:
        health = client.get("/api/health").json()
        response = client.post("/api/demo/seed", json={})
    assert health["paypal_configured"] is False
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "config"
    assert "PAYPAL_CLIENT_ID" in response.json()["error"]["message"]

def test_declined_capture_end_to_end_via_api(env):
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=llm_tool_call(
                        "capture_order",
                        {
                            "order_id": ORDER_ID,
                            "mock_application_code": "INSTRUMENT_DECLINED",
                        },
                    ),
                ),
                httpx.Response(
                    200,
                    json=llm_text("I prepared the capture for your approval."),
                ),
            ]
        )
        mock_token(mock)
        mock.get(
            f"https://api-m.sandbox.paypal.com/v2/checkout/orders/{ORDER_ID}"
        ).mock(
            return_value=httpx.Response(200, json=order_json(status="APPROVED"))
        )
        mock.post(
            f"https://api-m.sandbox.paypal.com/v2/checkout/orders/{ORDER_ID}/capture"
        ).mock(
            return_value=httpx.Response(
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
                    "debug_id": "debug-e2e",
                },
            )
        )
        with TestClient(app) as client:
            chat = client.post(
                "/api/chat",
                json={
                    "message": "Capture the order with a simulated decline.",
                    "history": [],
                },
            )
            pending = client.get("/api/pending").json()["pending"]
            assert len(pending) == 1
            approve = client.post(f"/api/actions/{pending[0]['id']}/approve")
            pending_after = client.get("/api/pending").json()["pending"]
            audit = client.get("/api/audit").json()["audit"]

    assert chat.status_code == 200
    assert approve.status_code == 502
    error = approve.json()["error"]
    assert error["type"] == "paypal"
    assert error["name"] == "UNPROCESSABLE_ENTITY"
    assert error["details"][0]["issue"] == "INSTRUMENT_DECLINED"
    assert error["debug_id"] == "debug-e2e"
    assert pending_after == []
    assert any(a["actor"] == "user-approved" and a["error"] for a in audit)
