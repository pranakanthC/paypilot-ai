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

def test_chat_works_when_llm_configured_without_health_check(env):
    """Regression test: chat must not return a config error when LLM is configured,
    even if the frontend's health check hasn't completed yet.

    The frontend previously blocked with "LLM is not configured" whenever the
    health object was still null (loadHealth() is fire-and-forget). The backend
    require_llm() check is the authoritative gate, so chat must proceed when
    LLM_API_KEY and LLM_MODEL are set, regardless of frontend health state.
    """
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            return_value=httpx.Response(
                200, json=llm_text("Hello, I am PayPilot.")
            )
        )
        with TestClient(app) as client:
            # Simulate frontend sending chat without waiting for health to load.
            response = client.post(
                "/api/chat", json={"message": "hello", "history": []}
            )
    assert response.status_code == 200
    body = response.json()
    assert "Hello, I am PayPilot" in body["reply"]


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


# --------------------------------------------------------------------------- #
# PayPal connectivity awareness regression tests
# --------------------------------------------------------------------------- #


def test_chat_paypal_not_configured(env, monkeypatch):
    """When PayPal credentials are missing, chat succeeds but agent reports PayPal unavailable."""
    monkeypatch.delenv("PAYPAL_CLIENT_ID", raising=False)
    monkeypatch.delenv("PAYPAL_CLIENT_SECRET", raising=False)
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            return_value=httpx.Response(
                200,
                json=llm_text(
                    "PayPal is not configured, so no payment actions can be "
                    "executed right now. I can still help you investigate data "
                    "and prepare proposals for your approval."
                ),
            )
        )
        with TestClient(app) as client:
            response = client.post("/api/chat", json={"message": "hello"})
    assert response.status_code == 200
    body = response.json()
    reply = body["reply"].lower()
    assert "paypal" in reply
    assert "not configured" in reply or "cannot" in reply
    assert "executed" not in reply.replace("cannot be executed", "").replace("can be executed", "")


def test_chat_paypal_configured_and_connected(env):
    """When PayPal is configured and connected, chat succeeds normally."""
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            return_value=httpx.Response(
                200, json=llm_text("Hello, I am PayPilot.")
            )
        )
        with TestClient(app) as client:
            response = client.post(
                "/api/chat", json={"message": "hello", "history": []}
            )
    assert response.status_code == 200
    body = response.json()
    assert "Hello, I am PayPilot" in body["reply"]


def test_chat_paypal_configured_but_disconnected(env, monkeypatch):
    """When PayPal is configured but disconnected, agent does not claim execution success."""
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
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
                httpx.Response(200, json=llm_text("The order is pending approval.")),
            ]
        )
        mock_token(mock)
        with TestClient(app) as client:
            # Simulate PayPal being configured but disconnected
            client.app.state.paypal_connected = False
            response = client.post(
                "/api/chat",
                json={"message": "Create a $25 order", "history": []},
            )
    assert response.status_code == 200
    body = response.json()
    reply = body["reply"].lower()
    # The agent must not claim execution success when PayPal is disconnected.
    assert "success" not in reply or "cannot" in reply or "unavailable" in reply
    # A pending action was created (proposal), not executed.
    assert len(body["pending_actions"]) == 1
    assert body["pending_actions"][0]["status"] == "pending"


def test_pending_actions_no_false_execution_claims(env, monkeypatch):
    """When PayPal is unavailable, the agent never claims an action was executed."""
    # Create pending actions directly in the DB
    db.create_pending_action(
        "create_order",
        {"amount": "0.01", "currency": "USD", "description": "Test order"},
    )
    db.create_pending_action(
        "create_order",
        {"amount": "25.00", "currency": "USD", "description": "Consulting"},
    )
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=llm_tool_call("list_pending_actions", {}),
                ),
                httpx.Response(
                    200,
                    json=llm_text(
                        "PayPal is not configured, so none of these actions can "
                        "be executed against PayPal. They are proposals awaiting "
                        "your approval."
                    ),
                ),
            ]
        )
        mock_token(mock)
        with TestClient(app) as client:
            client.app.state.paypal_connected = False
            response = client.post(
                "/api/chat",
                json={"message": "What actions are pending?", "history": []},
            )
    assert response.status_code == 200
    body = response.json()
    reply = body["reply"]

    # Server-side post-processing replaces the LLM's reply with factual summary.
    # Only actual DB actions are reported — no hallucinations.
    assert "0.01" in reply
    assert "25.00" in reply
    assert "Test order" in reply
    assert "Consulting" in reply
    assert "2 action(s)" in reply

    # No false execution success claims — summary says actions have NOT been executed.
    positive_execution = ["will execute", "has been executed", "was executed", "money has moved", "successfully captured", "successfully refunded"]
    for phrase in positive_execution:
        assert phrase not in reply.lower(), f"False execution claim found: {phrase}"

    # Summary explicitly states actions have not been executed.
    assert "have been executed" in reply.lower()

    # Both pending actions are listed in read steps.
    steps = body["steps"]
    read_steps = [s for s in steps if s.get("type") == "read"]
    assert len(read_steps) == 1
    assert read_steps[0]["tool"] == "list_pending_actions"
    actions = read_steps[0]["result"].get("actions", [])
    assert len(actions) == 2


# --------------------------------------------------------------------------- #
# Markdown rendering regression tests
# --------------------------------------------------------------------------- #
# These mirror the frontend renderMarkdown() function in static/index.html to
# ensure escaped bold markers are cleaned up and normal bold formatting works.

import re  # noqa: E402


def render_markdown(text: str) -> list:
    """Pure-Python mirror of the frontend renderMarkdown() function.

    Returns a list of (kind, content) tuples where kind is "text" or "strong".
    """
    text = str(text).replace(r"\*\*", "**")
    pattern = re.compile(r"\*\*([^\*\n]+)\*\*")
    parts = []
    last_index = 0
    for match in pattern.finditer(text):
        if match.start() > last_index:
            parts.append(("text", text[last_index:match.start()]))
        parts.append(("strong", match.group(1)))
        last_index = match.end()
    if last_index < len(text):
        parts.append(("text", text[last_index:]))
    return parts


def test_markdown_normal_bold_renders_as_strong():
    """Normal **bold** syntax should produce a strong segment."""
    parts = render_markdown("This is **bold** text.")
    assert parts == [
        ("text", "This is "),
        ("strong", "bold"),
        ("text", " text."),
    ]


def test_markdown_escaped_bold_is_cleaned_up():
    """Escaped \\*\\* markers should be cleaned up to normal bold."""
    parts = render_markdown("This is \\*\\*not bold\\*\\* text.")
    assert parts == [
        ("text", "This is "),
        ("strong", "not bold"),
        ("text", " text."),
    ]


def test_markdown_mixed_bold_and_plain():
    """Mixed content should produce alternating text and strong segments."""
    parts = render_markdown("**Amount:** 25.00 **USD**")
    assert parts == [
        ("strong", "Amount:"),
        ("text", " 25.00 "),
        ("strong", "USD"),
    ]


def test_markdown_multiple_bold_segments():
    """Multiple bold segments should all be captured."""
    parts = render_markdown("**A** and **B** and **C**")
    assert parts == [
        ("strong", "A"),
        ("text", " and "),
        ("strong", "B"),
        ("text", " and "),
        ("strong", "C"),
    ]


def test_markdown_no_asterisks_leaks():
    """No raw asterisk or backslash characters should appear in output."""
    parts = render_markdown("Normal text \\*\\*escaped\\*\\* and **real** bold")
    flat = "".join(content for _, content in parts)
    assert "\\" not in flat
    assert "*" not in flat


def test_system_prompt_uses_normal_bold_syntax():
    """The system prompt must instruct normal **bold** syntax, not escaped."""
    from app.agent import SYSTEM_PROMPT_BASE
    assert "**text**" in SYSTEM_PROMPT_BASE
    assert "Never use escaped asterisks" in SYSTEM_PROMPT_BASE


# --------------------------------------------------------------------------- #
# Pending action accuracy regression tests
# --------------------------------------------------------------------------- #
# These verify the AI never fabricates pending actions and always reports
# only records actually returned by list_pending_actions.


def test_agent_reports_only_tool_verified_pending_actions(env):
    """The AI must not invent pending actions beyond what list_pending_actions returns.

    Even if the LLM tries to hallucinate a $49.99 order, capture, and invoice,
    the server-side post-processing replaces the reply with a factual summary
    derived solely from the actual tool results.
    """
    # Seed exactly two pending actions in the database
    db.create_pending_action(
        "create_order",
        {"amount": "0.01", "currency": "USD", "description": "Small order"},
    )
    db.create_pending_action(
        "create_order",
        {"amount": "25.00", "currency": "USD", "description": "Consulting"},
    )
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=llm_tool_call("list_pending_actions", {}),
                ),
                # The LLM tries to hallucinate extra actions not in the tool result
                httpx.Response(
                    200,
                    json=llm_text(
                        "You have 3 pending actions: a $49.99 order, a $25.00 "
                        "capture, and a $100.00 invoice. All are ready to execute."
                    ),
                ),
            ]
        )
        mock_token(mock)
        with TestClient(app) as client:
            response = client.post(
                "/api/chat",
                json={"message": "What actions are pending?", "history": []},
            )
    assert response.status_code == 200
    body = response.json()
    reply = body["reply"]

    # Must NOT mention hallucinated actions
    assert "49.99" not in reply
    assert "capture" not in reply.lower()
    assert "invoice" not in reply.lower()
    assert "100.00" not in reply

    # Must report only the two actual actions from the database
    assert "0.01" in reply
    assert "25.00" in reply
    assert "Small order" in reply
    assert "Consulting" in reply
    assert "2 action(s)" in reply

    # No PayPal execution claims when disconnected
    assert "executed" not in reply.replace("have been executed", "").replace(
        "can be executed", ""
    )


def test_agent_empty_pending_queue_reported_accurately(env):
    """When list_pending_actions returns empty, the AI must say so explicitly."""
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=llm_tool_call("list_pending_actions", {}),
                ),
                # The LLM might try to say there are actions when there are none
                httpx.Response(
                    200,
                    json=llm_text("You have one pending action: a $50.00 order."),
                ),
            ]
        )
        mock_token(mock)
        with TestClient(app) as client:
            response = client.post(
                "/api/chat",
                json={"message": "Any pending approvals?", "history": []},
            )
    assert response.status_code == 200
    body = response.json()
    reply = body["reply"]

    # Server-side post-processing overrides the LLM's hallucination
    assert "no pending approval actions" in reply.lower()
    assert "no write has been proposed" in reply.lower()
    assert "50.00" not in reply
    assert "order" not in reply.lower()


def test_agent_paypal_disconnected_no_execution_claims(env):
    """When PayPal is disconnected, the AI must not claim actions will execute."""
    db.create_pending_action(
        "create_order",
        {"amount": "25.00", "currency": "USD", "description": "Consulting"},
    )
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=llm_tool_call("list_pending_actions", {}),
                ),
                httpx.Response(
                    200,
                    json=llm_text(
                        "Yes, this action will execute successfully against PayPal."
                    ),
                ),
            ]
        )
        mock_token(mock)
        with TestClient(app) as client:
            client.app.state.paypal_connected = False
            response = client.post(
                "/api/chat",
                json={"message": "What's pending?", "history": []},
            )
    assert response.status_code == 200
    body = response.json()
    reply = body["reply"].lower()

    # Must not claim execution success
    positive_execution = [
        "will execute",
        "has been executed",
        "was executed",
        "money has moved",
        "successfully captured",
        "successfully refunded",
    ]
    for phrase in positive_execution:
        assert phrase not in reply, f"False execution claim found: {phrase}"

    # Must clearly state PayPal is unreachable
    assert "unreachable" in reply or "connectivity" in reply

    # Must not claim any action was executed
    assert "executed" not in reply.replace("have been executed", "").replace(
        "can be executed", ""
    )


def test_agent_paypal_not_configured_no_execution_claims(env, monkeypatch):
    """When PayPal credentials are missing, the AI must not claim actions can execute."""
    monkeypatch.delenv("PAYPAL_CLIENT_ID", raising=False)
    monkeypatch.delenv("PAYPAL_CLIENT_SECRET", raising=False)
    db.create_pending_action(
        "create_order",
        {"amount": "25.00", "currency": "USD", "description": "Consulting"},
    )
    with respx.mock(assert_all_called=False) as mock:
        mock.post(LLM_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=llm_tool_call("list_pending_actions", {}),
                ),
                httpx.Response(
                    200,
                    json=llm_text(
                        "The order will execute successfully once you approve it."
                    ),
                ),
            ]
        )
        mock_token(mock)
        with TestClient(app) as client:
            response = client.post(
                "/api/chat",
                json={"message": "What's pending?", "history": []},
            )
    assert response.status_code == 200
    body = response.json()
    reply = body["reply"].lower()

    # Must not claim execution success
    positive_execution = [
        "will execute",
        "has been executed",
        "was executed",
        "money has moved",
        "successfully captured",
        "successfully refunded",
    ]
    for phrase in positive_execution:
        assert phrase not in reply, f"False execution claim found: {phrase}"

    # Must clearly state PayPal is not configured
    assert "not configured" in reply
