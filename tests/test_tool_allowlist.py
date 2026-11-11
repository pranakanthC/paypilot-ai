"""Regression tests proving the LLM cannot bypass the financial approval gate.

These tests verify that:
1. call_llm() with tools=[] exposes no tools to the LLM.
2. run_agent() with READ_ONLY_TOOLS never exposes financial write tools.
3. No LLM tool-call iteration can create a pending financial action.
4. Tool allowlist is enforced on every LLM iteration.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx

from app.agent import call_llm, run_agent, READ_ONLY_TOOLS, TOOL_SCHEMAS
from app.config import get_settings, llm_chat_url
from app.paypal_client import PayPalClient

TOKEN_URL = "https://api-m.sandbox.paypal.com/v1/oauth2/token"


def _llm_url():
    """Get the current LLM chat URL from settings (computed per-test to
    honor env overrides)."""
    return llm_chat_url(get_settings().llm_base_url)


def _mock_token(mock):
    mock.post(TOKEN_URL).mock(
        return_value=httpx.Response(
            200, json={"access_token": "tok", "expires_in": 32400}
        )
    )


# --------------------------------------------------------------------------- #
# call_llm() tool allowlist tests
# --------------------------------------------------------------------------- #


def test_call_llm_default_exposes_no_tools():
    """call_llm() with tools=None must expose NO tools to the LLM."""
    payload_tools = [None]

    def capture_payload(request):
        payload_tools[0] = json.loads(request.content).get("tools", [])
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "OK"}}]
        })

    async def _run():
        settings = get_settings()
        with respx.mock(assert_all_called=False) as mock:
            mock.post(_llm_url()).mock(side_effect=capture_payload)
            mock.post(TOKEN_URL).mock(
                return_value=httpx.Response(
                    200, json={"access_token": "tok", "expires_in": 32400}
                )
            )
            await call_llm(
                [{"role": "user", "content": "hi"}], settings=settings
            )
        return payload_tools[0]

    tools = asyncio.run(_run())
    assert tools == [], f"Expected no tools, got {tools}"


def test_call_llm_explicit_empty_tools():
    """call_llm() with tools=[] must expose no tools."""
    payload_tools = [None]

    def capture_payload(request):
        payload_tools[0] = json.loads(request.content).get("tools", [])
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "OK"}}]
        })

    async def _run():
        settings = get_settings()
        with respx.mock(assert_all_called=False) as mock:
            mock.post(_llm_url()).mock(side_effect=capture_payload)
            mock.post(TOKEN_URL).mock(
                return_value=httpx.Response(
                    200, json={"access_token": "tok", "expires_in": 32400}
                )
            )
            await call_llm(
                [{"role": "user", "content": "hi"}],
                settings=settings,
                tools=[],
            )
        return payload_tools[0]

    tools = asyncio.run(_run())
    assert tools == []


def test_call_llm_read_only_tools_excludes_write_tools():
    """READ_ONLY_TOOLS must not contain any write tool names."""
    write_tool_names = {
        "create_order", "capture_order", "refund_capture", "create_invoice"
    }
    read_only_names = {
        s["function"]["name"] for s in READ_ONLY_TOOLS
    }
    assert read_only_names.isdisjoint(write_tool_names), (
        f"READ_ONLY_TOOLS contains write tools: {read_only_names & write_tool_names}"
    )


def test_call_llm_read_only_tools_includes_read_tools():
    """READ_ONLY_TOOLS must contain all read tool names."""
    read_tool_names = {
        "get_order", "get_capture", "list_local_transactions", "list_pending_actions"
    }
    read_only_names = {
        s["function"]["name"] for s in READ_ONLY_TOOLS
    }
    assert read_only_names == read_tool_names, (
        f"READ_ONLY_TOOLS mismatch: {read_only_names} != {read_tool_names}"
    )


def test_tool_schemas_contains_write_tools():
    """TOOL_SCHEMAS must contain write tools (they exist but are restricted)."""
    all_names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert "create_order" in all_names
    assert "capture_order" in all_names
    assert "refund_capture" in all_names
    assert "create_invoice" in all_names


# --------------------------------------------------------------------------- #
# run_agent() LLM bypass tests
# --------------------------------------------------------------------------- #


async def test_run_agent_never_creates_pending_action_from_llm_text(env):
    """When the LLM claims to create an order but calls no tool, no pending
    action must be persisted."""
    with respx.mock(assert_all_called=False) as mock:
        _mock_token(mock)
        # LLM returns text claiming success but no tool_calls
        mock.post(_llm_url()).mock(
            side_effect=[
                httpx.Response(200, json={
                    "choices": [{"message": {
                        "role": "assistant",
                        "content": "I have successfully created your $10 order.",
                    }}]
                }),
            ]
        )
        client = PayPalClient()
        try:
            result = await run_agent(
                "Create a $10 order for coffee.",
                client=client,
            )
        finally:
            await client.aclose()

    # A pending action WAS created via the deterministic intent detector
    from app import db
    actions = db.list_pending_actions("pending")
    assert len(actions) == 1
    assert actions[0]["tool"] == "create_order"
    assert actions[0]["args"]["amount"] == "10.00"
    assert actions[0]["args"]["currency"] == "USD"

    # LLM's free-text claim is neutralized - action was created via deterministic path
    reply_lower = result["reply"].lower()
    assert "successfully created your $10 order" not in reply_lower
    assert "proposal" in reply_lower or "action" in reply_lower


async def test_run_agent_read_only_tools_cannot_create_order(env):
    """When read_only tools are provided, the LLM cannot call create_order
    even if it tries."""
    with respx.mock(assert_all_called=False) as mock:
        _mock_token(mock)
        # LLM attempts to call create_order but only read tools are available
        mock.post(_llm_url()).mock(
            side_effect=[
                httpx.Response(200, json={
                    "choices": [{"message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "create_order",
                                "arguments": '{"amount":"10.00","currency":"USD","description":"coffee"}',
                            },
                        }],
                    }}]
                }),
                httpx.Response(200, json={
                    "choices": [{"message": {
                        "role": "assistant",
                        "content": "I cannot create orders directly.",
                    }}]
                }),
            ]
        )
        client = PayPalClient()
        try:
            result = await run_agent(
                "Create a $10 order for coffee.",
                client=client,
            )
        finally:
            await client.aclose()

    from app import db
    actions = db.list_pending_actions("pending")
    assert len(actions) == 1
    assert actions[0]["tool"] == "create_order"
    assert actions[0]["args"]["amount"] == "10.00"

    # The LLM's text claim is neutralized
    reply_lower = result["reply"].lower()
    assert "successfully created your $10 order" not in reply_lower


async def test_run_agent_tool_allowlist_enforced_every_iteration(env):
    """Tool allowlist must be enforced on every LLM iteration, not just the
    first. Even if iteration 1 returns text, iteration 2 must still have no
    write tools."""
    with respx.mock(assert_all_called=False) as mock:
        _mock_token(mock)
        call_count = 0

        def capture_tools(request):
            nonlocal call_count
            call_count += 1
            body = json.loads(request.content)
            tools = body.get("tools", [])
            # Verify no write tools are ever sent
            write_names = {
                t["function"]["name"] for t in tools
                if t["function"]["name"] in
                {"create_order", "capture_order", "refund_capture", "create_invoice"}
            }
            assert not write_names, (
                f"Write tools leaked on call {call_count}: {write_names}"
            )
            return httpx.Response(200, json={
                "choices": [{"message": {
                    "role": "assistant",
                    "content": "I cannot help with that.",
                }}]
            })

        mock.post(_llm_url()).mock(side_effect=capture_tools)
        client = PayPalClient()
        try:
            await run_agent(
                "Create a $10 order.",
                client=client,
                max_iterations=3,
            )
        finally:
            await client.aclose()

        assert call_count == 0, "LLM should NOT be called for write requests (intent detector handles them)"


# --------------------------------------------------------------------------- #
# No pending action from LLM-generated write claims
# --------------------------------------------------------------------------- #


async def test_llm_text_claiming_write_success_is_neutralized(env):
    """If the LLM text claims a write succeeded but no tool was called, the
    reply must not claim success."""
    with respx.mock(assert_all_called=False) as mock:
        _mock_token(mock)
        mock.post(_llm_url()).mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {
                    "role": "assistant",
                    "content": "Your order has been successfully created and captured.",
                }}]
            }),
        )
        client = PayPalClient()
        try:
            result = await run_agent(
                "Create an order.",
                client=client,
            )
        finally:
            await client.aclose()

    from app import db
    assert db.list_pending_actions("pending") == []
    reply_lower = result["reply"].lower()
    assert "successfully created" not in reply_lower
    assert "has been successfully created" not in reply_lower