"""The LLM agent loop with a hard approval gate for all write tools."""

from __future__ import annotations

import json
from typing import Any, Optional

import httpx

from . import db
from .config import Settings, get_settings, llm_chat_url
from .paypal_client import PayPalClient, PayPalError
from .tools import (
    READ_TOOLS,
    TOOL_SCHEMAS,
    WRITE_TOOLS,
    ToolValidationError,
    summarize_step,
    validate_tool_args,
)

MAX_ITERATIONS = 6
MAX_TOOL_CONTENT_CHARS = 20000

SYSTEM_PROMPT = """You are PayPilot, an intelligent payment operations specialist.

Your role is to help merchants understand, investigate, and manage their payment operations through natural conversation.

Communicate like an experienced payments specialist: professional, trustworthy, calm, approachable, and solution-oriented. Handle technical complexity in the background and present users with clear, understandable information.

Core principle: Always communicate the business outcome, not the underlying technical process. Users should understand what happened, what the current situation is, whether any action is required, and what happens next. Do not make users understand APIs, backend systems, authentication, tool calls, database operations, or programming concepts.

Universal rules:
1. Use clear, natural, professional English. Keep responses concise but sufficiently informative.
2. Explain the result before providing any context. Use short paragraphs and meaningful headings when necessary.
3. Never use asterisks, Markdown bold, or Markdown italic formatting.
4. Never expose internal tool names, function names, raw JSON, API payloads, stack traces, or internal implementation details.
5. Never invent transaction details, payment statuses, amounts, IDs, or operation results.
6. Clearly distinguish verified facts from assumptions. Ask only for information that is genuinely required.
7. Maintain conversation context and do not repeatedly ask for information already provided.
8. Always explain important consequences before requesting approval for a financial action.

Status vocabulary (use these exact terms):
- Proposed: PayPilot has suggested an action. Nothing has been executed.
- Awaiting Approval: The action requires user approval before execution.
- Approved: The user has authorized the action. This does not necessarily mean it has been executed.
- Processing: The operation is being executed.
- Completed: The operation has been confirmed successful.
- Failed: The operation was unsuccessful.
- Unknown: The final outcome could not be confirmed.
Never use these statuses interchangeably.

Financial safety:
- Never claim an action has been completed before receiving confirmation.
- Never treat an approval as a completed transaction.
- Never execute a write operation without the required human approval.
- Explain the amount, currency, recipient, and purpose before approval when applicable.
- If an operation has an uncertain outcome, explain that its final status could not be confirmed.
- Never blindly retry an operation when doing so could cause duplicate financial activity.

READ tools (get_order, get_capture, list_local_transactions) run automatically.
WRITE tools (create_order, capture_order, refund_capture, create_invoice) are proposals. They are NEVER executed by you. When you call one, the system creates a pending approval request and notifies the user in the UI. Never claim a write happened; say that you have prepared it and it awaits explicit approval.

Use tools to look up IDs before explaining anything about them. If the user gives no ID and no local transaction matches, ask for the PayPal ID.
Amounts must be positive, currencies must be ISO 4217 codes, and refunds must not exceed the captured amount.
Only use the PayPal sandbox. Never suggest live-money operations.

When a write action is proposed, summarize exactly what will happen, the amount, the currency, and that the user must approve or reject it."""


class LLMError(Exception):
    def __init__(self, message: str, status_code: int = 0, details: Any = None) -> None:
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(message)

    def to_dict(self) -> dict:
        return {
            "type": "llm",
            "status_code": self.status_code,
            "message": self.message,
            "details": self.details,
        }


def _truncate(text: str) -> str:
    if len(text) <= MAX_TOOL_CONTENT_CHARS:
        return text
    return text[:MAX_TOOL_CONTENT_CHARS] + "...[truncated]"


async def call_llm(
    messages: list[dict],
    settings: Optional[Settings] = None,
    tool_choice: str = "auto",
) -> dict:
    """Call an OpenAI-compatible /chat/completions endpoint."""
    settings = settings or get_settings()
    settings.require_llm()
    url = llm_chat_url(settings.llm_base_url)
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "tools": TOOL_SCHEMAS,
        "tool_choice": tool_choice,
        "temperature": 0.2,
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(90.0)) as client:
            response = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {settings.llm_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
    except httpx.HTTPError as exc:
        raise LLMError(f"Could not reach the LLM at {url}: {exc}") from exc

    if response.status_code >= 400:
        detail: Any
        try:
            body = response.json()
            detail = body.get("error") or body
        except ValueError:
            detail = response.text[:1000]
        message = "The LLM request failed."
        if isinstance(detail, dict):
            message = detail.get("message") or message
        raise LLMError(message, status_code=response.status_code, details=detail)

    try:
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise LLMError("The LLM returned no choices.", details=data)
        return choices[0]["message"]
    except (ValueError, KeyError, IndexError) as exc:
        raise LLMError("The LLM returned an unexpected response shape.") from exc


def _sanitize_history(history: Optional[list[dict]]) -> list[dict]:
    clean: list[dict] = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            clean.append({"role": role, "content": content[:8000]})
    return clean[-20:]


def _parse_arguments(raw: Any) -> dict:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise ValueError("Tool arguments must be a JSON object")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Tool arguments must be a JSON object")
    return parsed


def _tool_message(tool_call_id: str, result: Any) -> dict:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": _truncate(json.dumps(result, default=str)),
    }


async def run_agent(
    user_message: str,
    history: Optional[list[dict]] = None,
    client: Optional[PayPalClient] = None,
    settings: Optional[Settings] = None,
    max_iterations: int = MAX_ITERATIONS,
) -> dict:
    """Run the agent loop. Returns reply text, visible steps and proposals."""
    settings = settings or get_settings()
    settings.require_llm()

    owns_client = client is None
    client = client or PayPalClient(settings)

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(_sanitize_history(history))
    messages.append({"role": "user", "content": user_message})

    steps: list[dict] = []
    pending_actions: list[dict] = []
    proposed: dict[tuple[str, str], str] = {}
    last_content: Optional[str] = None

    try:
        for iteration in range(1, max_iterations + 1):
            # Force tool usage on the first iteration (ministral needs it),
            # then allow free-text responses on subsequent iterations.
            choice = "required" if iteration == 1 else "auto"
            assistant = await call_llm(messages, settings, tool_choice=choice)
            content = assistant.get("content")
            if isinstance(content, str) and content.strip():
                last_content = content.strip()
            tool_calls = assistant.get("tool_calls") or []

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": content,
            }
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            messages.append(assistant_message)

            if not tool_calls:
                return {
                    "reply": last_content or "I could not produce a response.",
                    "pending_actions": pending_actions,
                    "steps": steps,
                    "iterations": iteration,
                }

            for tool_call in tool_calls:
                call_id = tool_call.get("id") or "call_unknown"
                function = tool_call.get("function") or {}
                tool_name = function.get("name") or ""
                raw_args = function.get("arguments")

                try:
                    parsed_args = _parse_arguments(raw_args)
                    validated_args = validate_tool_args(tool_name, parsed_args)
                except (ValueError, json.JSONDecodeError) as exc:
                    error = {"type": "validation", "message": str(exc)}
                    messages.append(_tool_message(call_id, {"error": error}))
                    steps.append(
                        {
                            "type": "error",
                            "tool": tool_name,
                            "summary": f"Rejected {tool_name or 'tool'}: invalid arguments",
                            "error": error,
                        }
                    )
                    continue
                except ToolValidationError as exc:
                    error = exc.to_dict()
                    db.log_audit("agent", tool_name, parsed_args, error=error)
                    messages.append(_tool_message(call_id, {"error": error}))
                    steps.append(
                        {
                            "type": "error",
                            "tool": tool_name,
                            "summary": f"Rejected {tool_name}: invalid arguments",
                            "error": error,
                        }
                    )
                    continue

                if tool_name in WRITE_TOOLS:
                    key = (tool_name, json.dumps(validated_args, sort_keys=True))
                    if key in proposed:
                        existing_id = proposed[key]
                        response = {
                            "status": "pending_approval",
                            "action_id": existing_id,
                            "message": "Duplicate proposal ignored; this action is already awaiting approval.",
                            "executed": False,
                        }
                        messages.append(_tool_message(call_id, response))
                        continue

                    action = db.create_pending_action(tool_name, validated_args)
                    proposed[key] = action["id"]
                    pending_actions.append(action)
                    db.log_audit(
                        "agent",
                        tool_name,
                        validated_args,
                        {"status": "pending_approval", "action_id": action["id"]},
                        request_id=action["request_id"],
                    )
                    response = {
                        "status": "pending_approval",
                        "action_id": action["id"],
                        "request_id": action["request_id"],
                        "message": (
                            "The action was NOT executed. It is stored as a pending "
                            "action and requires explicit user approval in the UI."
                        ),
                        "executed": False,
                    }
                    messages.append(_tool_message(call_id, response))
                    steps.append(
                        {
                            "type": "proposal",
                            "tool": tool_name,
                            "action_id": action["id"],
                            "request_id": action["request_id"],
                            "summary": summarize_step(tool_name, validated_args),
                            "args": validated_args,
                        }
                    )
                    continue

                if tool_name in READ_TOOLS:
                    try:
                        result = await _execute_read(
                            tool_name, validated_args, client
                        )
                        db.log_audit("agent", tool_name, validated_args, result)
                        messages.append(_tool_message(call_id, result))
                        steps.append(
                            {
                                "type": "read",
                                "tool": tool_name,
                                "summary": summarize_step(tool_name, validated_args),
                                "result": result,
                            }
                        )
                    except PayPalError as exc:
                        error = exc.to_dict()
                        db.log_audit(
                            "agent", tool_name, validated_args, error=error
                        )
                        messages.append(_tool_message(call_id, {"error": error}))
                        steps.append(
                            {
                                "type": "error",
                                "tool": tool_name,
                                "summary": f"{summarize_step(tool_name, validated_args)} failed",
                                "error": error,
                            }
                        )
                    except Exception as exc:  # defensive: always answer the model
                        error = {"type": "tool_error", "message": str(exc)}
                        db.log_audit(
                            "agent", tool_name, validated_args, error=error
                        )
                        messages.append(_tool_message(call_id, {"error": error}))
                        steps.append(
                            {
                                "type": "error",
                                "tool": tool_name,
                                "summary": f"{summarize_step(tool_name, validated_args)} failed",
                                "error": error,
                            }
                        )
                    continue

                error = {"type": "unknown_tool", "message": f"Unknown tool: {tool_name}"}
                messages.append(_tool_message(call_id, {"error": error}))
                steps.append(
                    {
                        "type": "error",
                        "tool": tool_name,
                        "summary": f"Unknown tool: {tool_name}",
                        "error": error,
                    }
                )

        reply = last_content or "I reached the maximum number of reasoning steps."
        if pending_actions:
            reply = (
                f"{reply}\n\nI prepared {len(pending_actions)} action(s) for your "
                "approval. Nothing has been executed yet."
            )
        else:
            reply = (
                f"{reply}\n\nI stopped after {max_iterations} reasoning steps. "
                "Please refine the request or provide a PayPal ID."
            )
        return {
            "reply": reply,
            "pending_actions": pending_actions,
            "steps": steps,
            "iterations": max_iterations,
        }
    finally:
        if owns_client:
            await client.aclose()


async def _execute_read(tool: str, args: dict, client: PayPalClient) -> dict:
    """Execute a read tool. Kept separate so the write gate stays obvious."""
    if tool == "get_order":
        result = await client.get_order(args["order_id"])
        if result.get("status"):
            db.update_transaction_status(args["order_id"], result["status"])
        return result
    if tool == "get_capture":
        return await client.get_capture(args["capture_id"])
    if tool == "list_local_transactions":
        return {"transactions": db.list_transactions(limit=50)}
    raise ValueError(f"Unknown read tool: {tool}")
