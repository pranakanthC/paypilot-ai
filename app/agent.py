"""The LLM agent loop with a hard approval gate for all write tools."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

import httpx

from . import db
from .config import Settings, get_settings, llm_chat_url
from .intent import (
    GENERAL,
    POTENTIAL_WRITE,
    READ,
    UNCLEAR,
    WRITE,
    IntentResult,
    detect_write_intent,
    resolve_clarification,
)
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

SYSTEM_PROMPT_BASE = r"""You are PayPilot, an intelligent payment operations specialist.

Communicate like an experienced payments specialist: professional, trustworthy, calm, approachable, and solution-oriented. Handle technical complexity in the background and present users with clear, understandable information.

Core principle: Always communicate the business outcome, not the underlying technical process. Users should understand what happened, what the current situation is, whether any action is required, and what happens next. Do not make users understand APIs, backend systems, authentication, tool calls, database operations, or programming concepts.

Universal rules:
1. Use clear, natural, professional English. Keep responses concise but sufficiently informative.
2. Explain the result before providing any context. Use short paragraphs and meaningful headings when necessary.
3. Use Markdown bold syntax (`**text**`) for emphasis. Never use escaped asterisks (`\*\*text\*\*`) or raw unrendered Markdown syntax.
4. Never expose internal tool names, function names, raw JSON, API payloads, stack traces, or internal implementation details.
5. Never invent transaction details, payment statuses, amounts, IDs, or operation results.
6. Clearly distinguish verified facts from assumptions. Ask only for information that is genuinely required.
7. Maintain conversation context and do not repeatedly ask for information already provided.
8. Always explain important consequences before requesting approval for a financial action.

Dynamic request handling:
- If the user's message contains multiple requests, handle each one. Do not stop after the first read tool if other requested tasks remain feasible. Make as many tool calls as needed within the iteration limit.
- Resolve contextual references such as "that order", "the previous transaction", "refund it", or "capture it" by looking at the conversation history. If the reference is ambiguous (multiple orders or transactions were discussed), ask which one the user means.
- Never infer an order ID, capture ID, or financial amount when it is ambiguous. Ask for the specific ID or amount.
- Do not perform unrelated operations. Only act on what the user explicitly asks for.
- If essential information is missing (e.g. an order ID for a lookup, an amount for a refund), ask a concise clarification question rather than guessing.
- When you have gathered all needed information, complete the full request before replying. Do not leave tasks half-done if they can still be completed within the iteration limit.

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

READ tools (get_order, get_capture, list_local_transactions, list_pending_actions) run automatically.
WRITE tools (create_order, capture_order, refund_capture, create_invoice) are proposals. They are NEVER executed by you. When you call one, the system creates a pending approval request and notifies the user in the UI. Never claim a write happened; say that you have prepared it and it awaits explicit approval.

Local transaction records (from list_local_transactions) include `provenance` for each record's identifier/origin:
- `paypal_verified`: the identifier and relevant details were established by a successful, validated PayPal API response.
- `local`: a locally recorded reference (not verified by PayPal).
- `unknown`: the record exists locally but its origin is unverified.
Always distinguish local existence from PayPal existence. Never infer manual insertion or claim PayPal verification without evidence. When provenance is `unknown`, explain that the record exists locally but its origin is unverified. A stored `status` (including `APPROVED`) is not proof of PayPal completion or provenance by itself.

Use tools to look up IDs before explaining anything about them. If the user gives no ID and no local transaction matches, ask for the PayPal ID.
Amounts must be positive, currencies must be ISO 4217 codes, and refunds must not exceed the captured amount.
Only use the PayPal sandbox. Never suggest live-money operations.

When a write action is proposed, summarize exactly what will happen, the amount, the currency, and that the user must approve or reject it."""

# Tool-use reinforcement: a separate system message placed immediately before
# each LLM call so Ministral reliably selects the right tool. The main system
# prompt tells the model to describe outcomes rather than call tools, so this
# reinforcement overrides that behavior for financial operations.
TOOL_REINFORCEMENT = {
    "role": "system",
    "content": (
        "MANDATORY TOOL USE OVERRIDE:\n"
        "When the user requests a financial operation (create order, capture payment, refund capture, create invoice): "
        "you MUST call the corresponding write tool immediately. Do not write text describing what you would do - call the tool. "
        "The system creates the pending action and handles user approval.\n"
        "When the user requests a read operation (get order, get capture, list transactions, list pending actions): "
        "call the appropriate read tool.\n"
        "For general questions or greetings: respond normally without calling tools.\n"
        "If required financial details are missing or ambiguous: ask the user for clarification; never invent amounts, order IDs, or capture IDs."
    ),
}


def _build_system_prompt(
    paypal_configured: bool, paypal_connected: bool
) -> str:
    """Build the system prompt with PayPal execution availability context.

    The agent must never claim a write action can execute when PayPal is
    unconfigured or disconnected. This context is authoritative and comes
    from the server, never inferred from pending action existence.
    """
    if paypal_configured and paypal_connected:
        paypal_status = (
            "PAYPAL EXECUTION AVAILABLE: PayPal sandbox credentials are "
            "configured and the sandbox is reachable. Approved write actions "
            "can be executed against the real PayPal sandbox API."
        )
    elif paypal_configured and not paypal_connected:
        paypal_status = (
            "PAYPAL EXECUTION UNAVAILABLE: PayPal sandbox credentials are "
            "configured but the sandbox is currently unreachable. No write "
            "action can be executed against PayPal until connectivity is "
            "restored. Never claim an action will succeed."
        )
    else:
        paypal_status = (
            "PAYPAL EXECUTION UNAVAILABLE: PayPal sandbox credentials are "
            "not configured. No write action can be executed against PayPal. "
            "Valid Sandbox client ID and secret are required before any order "
            "creation, capture, refund, or invoice can be executed. Never "
            "claim an action will succeed or that money has moved."
        )

    return f"""{SYSTEM_PROMPT_BASE}

{paypal_status}

If PayPal execution is unavailable, explain clearly that the action is only a
proposal awaiting approval and cannot be executed until PayPal credentials are
configured and the sandbox is reachable. Do not claim an order was created,
a payment was captured, or a refund was issued merely because an action is
pending or approved."""


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


READ_ONLY_TOOLS: list[dict] = [
    schema
    for schema in TOOL_SCHEMAS
    if schema["function"]["name"]
    in {"get_order", "get_capture", "list_local_transactions", "list_pending_actions"}
]


async def call_llm(
    messages: list[dict],
    settings: Optional[Settings] = None,
    tool_choice: str = "auto",
    tools: Optional[list[dict]] = None,
) -> dict:
    """Call an OpenAI-compatible /chat/completions endpoint.

    ``tools`` is an explicit server-side allowlist. When omitted, no tools are
    exposed to the LLM, guaranteeing that financial write tools can never leak
    into an LLM invocation by default.
    """
    settings = settings or get_settings()
    settings.require_llm()
    url = llm_chat_url(settings.llm_base_url)
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "tools": tools if tools is not None else [],
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


# Patterns that indicate the LLM is claiming a financial write was completed.
# If no write was attempted server-side and no pending action exists, any such
# claim is fabricated and must be neutralized.
_WRITE_SUCCESS_PATTERNS = [
    r"\b(?:created|executed|sent|processed|completed|captured|refunded)\b.*\b(?:order|capture|refund|invoice|payment)\b",
    r"\b(?:order|capture|refund|invoice|payment)\b.*\b(?:created|executed|sent|processed|completed|captured|refunded)\b",
    r"\bsuccessfully (?:created|executed|sent|processed|completed|captured|refunded)\b",
    r"\bhas been (?:created|executed|sent|processed|completed|captured|refunded)\b",
    r"\bwas (?:created|executed|sent|processed|completed|captured|refunded)\b",
]

# Negation phrases that indicate the LLM is denying (not claiming) a write.
_NEGATION_PHRASES = [
    " cannot ", " can not ", " can't ", " won't ", " will not ",
    " unable to ", " not be ", " cannot be ", " could not ", " couldn't ",
    " did not ", " didn't ", " does not ", " doesn't ", " no ",
]


def _text_claims_write_success(text: str) -> bool:
    """Return True if ``text`` claims a financial write was completed.

    Negated statements (e.g. "cannot be executed", "no X can be verbed")
    are denials, not claims.
    """
    lowered = text.lower()
    has_negation = any(phrase in lowered for phrase in _NEGATION_PHRASES)
    has_negation = has_negation or lowered.startswith("no ")
    if has_negation:
        return False
    for pattern in _WRITE_SUCCESS_PATTERNS:
        if re.search(pattern, lowered):
            return True
    return False


async def _handle_verified_write(
    tool: str, validated_args: dict
) -> dict:
    """Create a pending action from a verified write request and return a
    factual reply. No PayPal call is made here; the action awaits human
    approval via /api/actions/{id}/approve."""
    action = db.create_pending_action_with_audit(tool, validated_args)
    step = {
        "type": "proposal",
        "tool": tool,
        "action_id": action["id"],
        "summary": summarize_step(tool, validated_args),
        "args": validated_args,
    }
    reply = _build_write_proposal_reply([action], [step], [])
    return {
        "reply": reply,
        "pending_actions": [action],
        "steps": [step],
        "iterations": 0,
    }



async def _resolve_contextual_write_intent(
    intent: IntentResult,
    user_message: str,
    *,
    session_id: Optional[str],
    client: Optional[PayPalClient],
) -> IntentResult:
    """Apply Phase 4 verified context resolution to capture/refund intents.

    Resolves contextual references and explicit association patterns to verified
    resources. Revalidates eligibility for financial operations via read-only
    PayPal lookups (no writes). Returns updated intent; may become WRITE with
    resolved ID, or POTENTIAL_WRITE/UNCLEAR with clarification.
    """
    from .context import (
        resolve_for_financial_operation,
        resolve_transaction_reference,
        is_contextual_reference,
    )
    from .intent import extract_order_id, extract_capture_id

    tool = intent.tool
    args = dict(intent.args or {})
    text = user_message or ""

    if tool == "capture_order":
        # Block capture suggestions for locally-tracked orders that are not
        # PayPal-verified. Unknown IDs (not in local DB) fall through to the
        # existing behavior; live revalidation happens at approval/execution.
        if args.get("order_id"):
            local = db.find_verified_transaction(args["order_id"], "order")
            if local and local.get("provenance") != db.PROVENANCE_PAYPAL_VERIFIED:
                return IntentResult(
                    POTENTIAL_WRITE,
                    tool=tool,
                    args=args,
                    missing=[f"order_id ({args['order_id']} is not a PayPal-verified resource)"],
                    requires_clarification=True,
                )
            return intent
        # Detect contextual/associative reference
        if not is_contextual_reference(text):
            return intent
        if client is None:
            return IntentResult(
                POTENTIAL_WRITE,
                tool=tool,
                args=args,
                missing=["order_id (verified context unavailable)"],
                requires_clarification=True,
            )
        res = await resolve_for_financial_operation(
            text, session_id=session_id, operation="capture_order", client=client
        )
        if res.resolved and res.paypal_id:
            args["order_id"] = res.paypal_id
            # Validate resolved args form a complete write request
            try:
                validated = validate_tool_args(tool, args)
                return IntentResult(WRITE, tool=tool, args=validated)
            except Exception as exc:
                msg = str(exc)
                return IntentResult(
                    POTENTIAL_WRITE, tool=tool, args=args,
                    missing=[f"order_id (resolved but invalid: {msg})"],
                    requires_clarification=True,
                )
        # Not resolved -> clarify with precise reason
        if res.status == "ambiguous":
            detail = "order_id (ambiguous transaction reference; please specify the order ID)"
        elif res.status == "malformed":
            detail = "order_id (malformed transaction ID)"
        elif res.status == "not_found":
            detail = f"order_id ({res.message or 'verified eligible order not found'})"
        elif res.status == "type_mismatch":
            detail = f"order_id ({res.message or 'referenced ID is not an eligible order'})"
        elif res.status == "lookup_failed":
            detail = f"order_id ({res.message or 'could not verify order'})"
        elif res.status == "missing":
            detail = "order_id"
        else:
            detail = f"order_id ({res.message or 'could not resolve contextual reference'})"
        return IntentResult(
            POTENTIAL_WRITE, tool=tool, args=args, missing=[detail], requires_clarification=True
        )

    if tool == "refund_capture":
        # Block refund suggestions for locally-tracked captures that are not
        # PayPal-verified. Unknown IDs (not in local DB) fall through to the
        # existing behavior; live revalidation happens at approval/execution.
        if args.get("capture_id"):
            local = db.find_verified_transaction(args["capture_id"], "capture")
            if local and local.get("provenance") != db.PROVENANCE_PAYPAL_VERIFIED:
                return IntentResult(
                    POTENTIAL_WRITE,
                    tool=tool,
                    args=args,
                    missing=[f"capture_id ({args['capture_id']} is not a PayPal-verified resource)"],
                    requires_clarification=True,
                )
            return intent
        # Try association "capture/refund ... associated with order <id>" or contextual ref
        if not is_contextual_reference(text):
            return intent
        if client is None:
            return IntentResult(
                POTENTIAL_WRITE,
                tool=tool,
                args=args,
                missing=["capture_id (verified context unavailable)"],
                requires_clarification=True,
            )
        res = await resolve_for_financial_operation(
            text, session_id=session_id, operation="refund_capture", client=client
        )
        if res.resolved and res.paypal_id:
            args["capture_id"] = res.paypal_id
            # preserve optional amount/note if already extracted
            try:
                validated = validate_tool_args(tool, args)
                return IntentResult(WRITE, tool=tool, args=validated)
            except Exception as exc:
                msg = str(exc)
                return IntentResult(
                    POTENTIAL_WRITE, tool=tool, args=args,
                    missing=[f"capture_id (resolved but invalid: {msg})"],
                    requires_clarification=True,
                )
        if res.status == "ambiguous":
            detail = "capture_id (ambiguous transaction reference; please specify the capture ID)"
        elif res.status == "malformed":
            detail = "capture_id (malformed transaction ID)"
        elif res.status == "not_found":
            detail = f"capture_id ({res.message or 'verified refundable capture not found'})"
        elif res.status == "type_mismatch":
            detail = f"capture_id ({res.message or 'referenced ID is not a capture'})"
        elif res.status == "lookup_failed":
            detail = f"capture_id ({res.message or 'could not verify capture'})"
        elif res.status == "missing":
            detail = "capture_id"
        else:
            detail = f"capture_id ({res.message or 'could not resolve contextual reference'})"
        return IntentResult(
            POTENTIAL_WRITE, tool=tool, args=args, missing=[detail], requires_clarification=True
        )

    return intent


async def run_agent(
    user_message: str,
    history: Optional[list[dict]] = None,
    client: Optional[PayPalClient] = None,
    settings: Optional[Settings] = None,
    max_iterations: int = MAX_ITERATIONS,
    paypal_configured: bool = False,
    paypal_connected: bool = False,
    session_id: Optional[str] = None,
) -> dict:
    """Run the agent loop. Returns reply text, visible steps and proposals."""
    settings = settings or get_settings()
    settings.require_llm()

    # Phase 3: Multi-turn clarification support.
    # Before running the intent detector, check if this session has an
    # unresolved request waiting for clarification. If so, merge the new
    # message's values and either complete atomically (pending action + audit +
    # resolved state in one transaction) or return a clarification.
    if session_id:
        clarification_result = resolve_clarification(
            session_id, user_message
        )
        if clarification_result is not None:
            if clarification_result.classification == WRITE:
                # Completion was handled atomically in resolve_clarification;
                # do not create a second pending action. Never claim success
                # unless a persisted action exists for this completion.
                if clarification_result.completion_outcome == "completed":
                    action = clarification_result.action
                    if not action:
                        return {
                            "reply": _build_persistence_failure_reply(
                                "clarification completed without a persisted action"
                            ),
                            "pending_actions": [],
                            "steps": [
                                {
                                    "type": "error",
                                    "tool": clarification_result.tool,
                                    "summary": "Clarification completion failed to persist an action",
                                    "error": {
                                        "type": "persistence_error",
                                        "message": "no pending action persisted",
                                    },
                                }
                            ],
                            "iterations": 0,
                        }
                    step = {
                        "type": "proposal",
                        "tool": action.get("tool", clarification_result.tool),
                        "action_id": action["id"],
                        "summary": summarize_step(
                            action.get("tool", clarification_result.tool),
                            clarification_result.args,
                        ),
                        "args": clarification_result.args,
                    }
                    reply = _build_write_proposal_reply([action], [step], [])
                    return {
                        "reply": reply,
                        "pending_actions": [action],
                        "steps": [step],
                        "iterations": 0,
                    }
                outcome = clarification_result.completion_outcome or "conflict"
                if outcome == "expired":
                    return {
                        "reply": (
                            "The previous clarification request expired before it "
                            "could be completed. No pending action was created. "
                            "Please submit the request again."
                        ),
                        "pending_actions": [],
                        "steps": [
                            {
                                "type": "clarification",
                                "tool": clarification_result.tool,
                                "missing": ["expired"],
                            }
                        ],
                        "iterations": 0,
                    }
                if outcome == "error":
                    detail = clarification_result.ambiguity or "persistence failure"
                    return {
                        "reply": _build_persistence_failure_reply(detail),
                        "pending_actions": [],
                        "steps": [
                            {
                                "type": "error",
                                "tool": clarification_result.tool,
                                "summary": f"Failed to persist clarification completion: {detail}",
                                "error": {"type": "persistence_error", "message": detail},
                            }
                        ],
                        "iterations": 0,
                    }
                # already_completed / conflict: another attempt completed it or
                # it was otherwise claimed. Do not report a new proposal.
                return {
                    "reply": (
                        "This request was already completed or could not be "
                        "completed again. No new pending action was created."
                    ),
                    "pending_actions": [],
                    "steps": [
                        {
                            "type": "clarification",
                            "tool": clarification_result.tool,
                            "ambiguity": f"completion_{outcome}",
                        }
                    ],
                    "iterations": 0,
                }
            elif clarification_result.classification == POTENTIAL_WRITE:
                reply = _build_clarification_reply(
                    clarification_result.tool, clarification_result.missing
                )
                return {
                    "reply": reply,
                    "pending_actions": [],
                    "steps": [
                        {
                            "type": "clarification",
                            "tool": clarification_result.tool,
                            "missing": clarification_result.missing,
                        }
                    ],
                    "iterations": 0,
                }
            elif clarification_result.classification == UNCLEAR:
                reply = _build_unclear_reply(
                    clarification_result.ambiguity or "unclear request"
                )
                return {
                    "reply": reply,
                    "pending_actions": [],
                    "steps": [
                        {
                            "type": "clarification",
                            "tool": None,
                            "ambiguity": clarification_result.ambiguity,
                        }
                    ],
                    "iterations": 0,
                }
            # GENERAL from completion outcomes (expired/already_completed/error)
            # with no pending proposal: return a safe, factual reply.
            outcome = clarification_result.completion_outcome
            if outcome in {"already_completed", "conflict"}:
                return {
                    "reply": (
                        "This request was already completed or could not be "
                        "completed again. No new pending action was created."
                    ),
                    "pending_actions": [],
                    "steps": [],
                    "iterations": 0,
                }
            if outcome == "expired":
                return {
                    "reply": (
                        "The previous clarification request expired before it "
                        "could be completed. No pending action was created. "
                        "Please submit the request again."
                    ),
                    "pending_actions": [],
                    "steps": [],
                    "iterations": 0,
                }
            if outcome == "error":
                detail = clarification_result.ambiguity or "persistence failure"
                return {
                    "reply": _build_persistence_failure_reply(detail),
                    "pending_actions": [],
                    "steps": [],
                    "iterations": 0,
                }

    # Phase 2: Deterministic intent detection runs BEFORE the LLM agent loop.
    # WRITE requests create pending actions server-side without LLM involvement.
    # POTENTIAL_WRITE/UNCLEAR return clarification questions.
    # READ/GENERAL proceed to the LLM loop with restricted tool allowlists.
    intent = detect_write_intent(user_message)

    # Phase 4: deterministic, session-isolated verified context resolution.
    # Authoritative for contextual transaction references; never guesses or
    # treats pending proposals/failed ops as verified. Financial ops are
    # revalidated and never executed here (proposal only, approval still required).
    if intent.classification in {POTENTIAL_WRITE, WRITE} and intent.tool in {"capture_order", "refund_capture"}:
        try:
            intent = await _resolve_contextual_write_intent(
                intent, user_message, session_id=session_id, client=client
            )
        except Exception:
            # Fail safe: if resolution cannot be completed, require clarification.
            if intent.tool == "capture_order":
                intent = IntentResult(
                    POTENTIAL_WRITE,
                    tool=intent.tool,
                    args=intent.args,
                    missing=["order_id (could not verify contextual reference)"],
                    requires_clarification=True,
                )
            else:
                intent = IntentResult(
                    POTENTIAL_WRITE,
                    tool=intent.tool,
                    args=intent.args,
                    missing=["capture_id (could not verify contextual reference)"],
                    requires_clarification=True,
                )

    if intent.classification == WRITE:
        # Server-side validation before persisting any pending action.
        try:
            validated_args = validate_tool_args(intent.tool, intent.args)
        except ToolValidationError as exc:
            db.log_audit("agent", intent.tool, intent.args, error=exc.to_dict())
            return {
                "reply": _build_validation_failure_reply(exc),
                "pending_actions": [],
                "steps": [{"type": "error", "tool": intent.tool, "summary": f"Rejected {intent.tool}: validation failed", "error": exc.to_dict()}],
                "iterations": 0,
            }

        try:
            action = db.create_pending_action_with_audit(intent.tool, validated_args)
        except Exception as exc:  # persistence failure
            db.log_audit(
                "agent",
                intent.tool,
                validated_args,
                error={"type": "persistence_error", "message": str(exc)},
            )
            return {
                "reply": _build_persistence_failure_reply(str(exc)),
                "pending_actions": [],
                "steps": [{"type": "error", "tool": intent.tool, "summary": f"Failed to persist {intent.tool}: {exc}", "error": {"type": "persistence_error", "message": str(exc)}}],
                "iterations": 0,
            }

        if action.get("is_duplicate"):
            # A pending proposal for this exact operation already exists.
            # Return a truthful response identifying the existing proposal
            # rather than claiming a new one was created.
            reply = _build_duplicate_proposal_reply(action, validated_args)
            return {
                "reply": reply,
                "pending_actions": [action],
                "steps": [
                    {
                        "type": "proposal",
                        "tool": intent.tool,
                        "action_id": action["id"],
                        "request_id": action["request_id"],
                        "summary": summarize_step(intent.tool, validated_args),
                        "args": validated_args,
                        "is_duplicate": True,
                    }
                ],
                "iterations": 0,
            }

        reply = _build_deterministic_write_reply(action, validated_args)
        return {
            "reply": reply,
            "pending_actions": [action],
            "steps": [
                {
                    "type": "proposal",
                    "tool": intent.tool,
                    "action_id": action["id"],
                    "request_id": action["request_id"],
                    "summary": summarize_step(intent.tool, validated_args),
                    "args": validated_args,
                }
            ],
            "iterations": 0,
        }

    if intent.classification == POTENTIAL_WRITE:
        if session_id:
            db.create_pending_request(
                session_id=session_id,
                tool=intent.tool,
                args=intent.args,
                missing=intent.missing,
            )
        reply = _build_clarification_reply(intent.tool, intent.missing)
        return {
            "reply": reply,
            "pending_actions": [],
            "steps": [{"type": "clarification", "tool": intent.tool, "missing": intent.missing}],
            "iterations": 0,
        }

    if intent.classification == UNCLEAR:
        reply = _build_unclear_reply(intent.ambiguity)
        return {
            "reply": reply,
            "pending_actions": [],
            "steps": [{"type": "clarification", "tool": None, "ambiguity": intent.ambiguity}],
            "iterations": 0,
        }

    # READ/GENERAL: proceed to LLM agent loop (existing behavior)
    owns_client = client is None
    client = client or PayPalClient(settings)

    system_prompt = _build_system_prompt(paypal_configured, paypal_connected)
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    messages.extend(_sanitize_history(history))
    messages.append({"role": "user", "content": user_message})

    steps: list[dict] = []
    pending_actions: list[dict] = []
    proposed: dict[tuple[str, str], str] = {}
    last_content: Optional[str] = None
    pending_actions_read_result: Optional[list[dict]] = None
    broke_early: bool = False
    write_attempted: bool = False
    write_failures: list[dict] = []

    try:
        for iteration in range(1, max_iterations + 1):
            # Use "auto" so general conversation is never forced to call tools.
            # The reinforcement message prompts tool use when financial
            # operations are needed. "required" is not reliably enforced by
            # Ministral and would force general conversation to call tools.
            choice = "auto"
            # Re-insert reinforcement before each LLM call (it is consumed by
            # the model's response, so it must be re-added each iteration).
            messages.append(dict(TOOL_REINFORCEMENT))
            assistant = await call_llm(
                messages, settings, tools=READ_ONLY_TOOLS, tool_choice=choice
            )
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
                # "auto" means general conversation is not forced to call tools.
                # The reinforcement message already prompted tool use when
                # needed. Break and use the conversational response.
                reply = last_content or "I could not produce a response."
                broke_early = True
                break

            for tool_call in tool_calls:
                call_id = tool_call.get("id") or "call_unknown"
                function = tool_call.get("function") or {}
                tool_name = function.get("name") or ""
                raw_args = function.get("arguments")

                # Reject any tool call not in the server-side allowlist.
                # This prevents the LLM from bypassing tool restrictions by
                # hallucinating a tool call for a tool that was not provided.
                allowed_tool_names = {
                    t["function"]["name"] for t in READ_ONLY_TOOLS
                }
                if tool_name not in allowed_tool_names:
                    messages.append(
                        _tool_message(
                            call_id,
                            {
                                "error": {
                                    "type": "tool_not_allowed",
                                    "message": f"Tool '{tool_name}' is not available.",
                                }
                            }
                        )
                    )
                    steps.append(
                        {
                            "type": "error",
                            "tool": tool_name,
                            "summary": f"Rejected {tool_name}: tool not available",
                            "error": {
                                "type": "tool_not_allowed",
                                "message": f"Tool '{tool_name}' is not available.",
                            },
                        }
                    )
                    continue

                is_write = tool_name in WRITE_TOOLS
                if is_write:
                    write_attempted = True

                try:
                    parsed_args = _parse_arguments(raw_args)
                    validated_args = validate_tool_args(tool_name, parsed_args)
                except (ValueError, json.JSONDecodeError) as exc:
                    error = {"type": "validation", "message": str(exc)}
                    if is_write:
                        write_failures.append(
                            {
                                "tool": tool_name,
                                "args": raw_args,
                                "reason": "validation_failed",
                                "error": str(exc),
                            }
                        )
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
                    if is_write:
                        write_failures.append(
                            {
                                "tool": tool_name,
                                "args": parsed_args,
                                "reason": "validation_failed",
                                "error": str(exc),
                            }
                        )
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

                    try:
                        action = db.create_pending_action_with_audit(
                            tool_name, validated_args
                        )
                    except Exception as exc:  # persistence failure
                        write_failures.append(
                            {
                                "tool": tool_name,
                                "args": validated_args,
                                "reason": "persistence_failed",
                                "error": str(exc),
                            }
                        )
                        db.log_audit(
                            "agent",
                            tool_name,
                            validated_args,
                            error={
                                "type": "persistence_error",
                                "message": str(exc),
                            },
                        )
                        messages.append(
                            _tool_message(
                                call_id,
                                {
                                    "error": {
                                        "type": "persistence_error",
                                        "message": str(exc),
                                    }
                                },
                            )
                        )
                        steps.append(
                            {
                                "type": "error",
                                "tool": tool_name,
                                "summary": f"Failed to persist {tool_name}: {exc}",
                                "error": {
                                    "type": "persistence_error",
                                    "message": str(exc),
                                },
                            }
                        )
                        continue
                    proposed[key] = action["id"]
                    pending_actions.append(action)
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
                        # Capture list_pending_actions results for factual
                        # post-processing. The LLM's free-text reply must never
                        # invent pending actions that don't exist in the tool result.
                        if tool_name == "list_pending_actions":
                            pending_actions_read_result = result.get("actions", [])
                        # Phase 4: remember verified read results in session context
                        try:
                            _remember_verified_read(session_id, tool_name, result)
                        except Exception:
                            pass
                        continue
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
                        continue
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

        # Server-side factual reply: never trust the LLM's free-text output as
        # proof of what was done. Build from actual recorded steps and pending
        # actions only. This prevents the LLM from claiming success for writes
        # that were never proposed, never persisted, or never executed.
        if pending_actions_read_result is not None:
            reply = _build_pending_actions_summary(
                pending_actions_read_result, paypal_configured, paypal_connected
            )
        elif pending_actions:
            reply = _build_write_proposal_reply(
                pending_actions, steps, write_failures
            )
        elif write_attempted and write_failures:
            # A write was attempted but no pending action was persisted.
            # Return a factual server-generated error, never LLM text that
            # could claim success.
            reply = _build_write_failure_reply(write_failures)
        elif last_content:
            # No write was attempted and no pending action exists, so the LLM's
            # text cannot be backed by a server-side record. Neutralize any
            # claim of financial write success to prevent hallucinated success.
            if (
                not write_attempted
                and not pending_actions
                and not write_failures
                and _text_claims_write_success(last_content)
            ):
                reply = (
                    "No financial action was taken. No order, capture, refund, "
                    "or invoice was created or executed."
                )
            else:
                reply = last_content
        elif broke_early:
            reply = "I could not produce a response."
        else:
            reply = (
                f"I stopped after {max_iterations} reasoning steps. "
                "Please refine the request or provide a PayPal ID."
            )
        return {
            "reply": reply,
            "pending_actions": pending_actions,
            "steps": steps,
            "iterations": iteration,
        }
    finally:
        if owns_client:
            await client.aclose()


def _build_pending_actions_summary(
    actions: list[dict],
    paypal_configured: bool,
    paypal_connected: bool,
) -> str:
    """Build a factually accurate summary of pending actions from tool results.

    Only includes actions actually returned by list_pending_actions. Never
    invents action IDs, amounts, descriptions, statuses, or action types.
    """
    if not actions:
        return (
            "There are no pending approval actions waiting for your review. "
            "No write has been proposed that requires your approval."
        )

    lines = [f"There are {len(actions)} action(s) awaiting your approval:"]
    for action in actions:
        args = action.get("args") or {}
        if isinstance(args, str):
            args = json.loads(args)
        amount = args.get("amount", "—")
        currency = args.get("currency", "—")
        description = args.get("description", "—")
        tool = action.get("tool", "—")
        status = action.get("status", "pending")

        # Map internal tool name to business-friendly label.
        tool_labels = {
            "create_order": "Create order",
            "capture_order": "Capture payment",
            "refund_capture": "Refund capture",
            "create_invoice": "Send invoice",
        }
        label = tool_labels.get(tool, tool.replace("_", " ").title())

        lines.append(
            f"• {label}: {amount} {currency} — {description} (status: {status})"
        )

    # PayPal connectivity warning
    if not paypal_configured:
        lines.append(
            "\nPayPal sandbox credentials are not configured. None of these "
            "actions can be executed until valid credentials are added."
        )
    elif not paypal_connected:
        lines.append(
            "\nPayPal sandbox is currently unreachable. None of these actions "
            "can be executed until connectivity is restored."
        )

    lines.append(
        "\nNone of these actions have been executed. They require your explicit "
        "approval before any PayPal write occurs."
    )

    return "\n".join(lines)


def _build_write_proposal_reply(
    pending_actions: list[dict], steps: list[dict], write_failures: list[dict]
) -> str:
    """Build a factual reply for write action proposals.

    Uses only actual persisted pending actions and their recorded step
    summaries -- never LLM-invented details. Explicitly states that nothing
    has been executed and that each action requires explicit user approval.
    Includes any write attempts that failed without persisting a pending action.
    """
    lines = [
        f"I prepared {len(pending_actions)} action(s) for your review. "
        "Nothing has been executed yet -- each requires your explicit approval."
    ]
    for action in pending_actions:
        raw_args = action.get("args")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                args = {}
        else:
            args = raw_args
        # Find the matching proposal step for a human-friendly summary.
        summary = None
        for step in steps:
            if (
                step.get("type") == "proposal"
                and step.get("action_id") == action["id"]
            ):
                summary = step.get("summary")
                break
        if summary:
            lines.append(f"• {summary} (action {action['id']})")
        else:
            tool = action.get("tool", "—")
            lines.append(f"• {tool} (action {action['id']})")

    if write_failures:
        lines.append("")
        lines.append("The following write attempt(s) could not be saved:")
        for failure in write_failures:
            tool = failure.get("tool", "—")
            reason = failure.get("reason", "unknown")
            error_msg = failure.get("error", "")
            lines.append(
                f"• {tool}: {reason} — {error_msg} (no pending action created)"
            )

    lines.append(
        "Review the pending actions in the approval queue and approve or reject "
        "each one. No PayPal write occurs until you approve."
    )
    return "\n".join(lines)


def _build_write_failure_reply(write_failures: list[dict]) -> str:
    """Build a factual server-generated error reply when a write was attempted
    but no pending action was persisted.

    Never falls back to LLM text that could claim success.
    """
    lines = [
        "I was unable to save your request. No pending action was created "
        "and nothing was executed."
    ]
    lines.append("")
    lines.append("Write attempt(s) that failed:")
    for failure in write_failures:
        tool = failure.get("tool", "—")
        reason = failure.get("reason", "unknown")
        error_msg = failure.get("error", "")
        lines.append(f"• {tool}: {reason} — {error_msg}")
    lines.append("")
    lines.append(
        "Review the pending actions in the approval queue and approve or reject "
        "each one. No PayPal write occurs until you approve."
    )
    return "\n".join(lines)


def _build_deterministic_write_reply(action: dict, validated_args: dict) -> str:
    """Build a factual reply for a server-side verified write proposal.

    Uses only the actual persisted pending action and its validated arguments.
    Never claims execution — only proposal creation.
    """
    tool = action.get("tool", "—")
    action_id = action.get("id", "—")
    args = action.get("args") or {}
    if isinstance(args, str):
            args = json.loads(args)

    tool_labels = {
        "create_order": "Create order",
        "capture_order": "Capture payment",
        "refund_capture": "Refund capture",
        "create_invoice": "Create invoice",
    }
    label = tool_labels.get(tool, tool.replace("_", " ").title())

    lines = [
        f"I prepared a {label.lower()} proposal for your review.",
        f"• {label}: action {action_id}",
    ]

    # Show the actual stored arguments transparently
    detail_parts: list[str] = []
    if args.get("amount"):
        detail_parts.append(f"amount: {args['amount']}")
    if args.get("currency"):
        detail_parts.append(f"currency: {args['currency']}")
    if args.get("description"):
        detail_parts.append(f"description: {args['description']}")
    if args.get("order_id"):
        detail_parts.append(f"order ID: {args['order_id']}")
    if args.get("capture_id"):
        detail_parts.append(f"capture ID: {args['capture_id']}")
    if args.get("recipient_email"):
        detail_parts.append(f"recipient: {args['recipient_email']}")
    if args.get("items"):
        detail_parts.append(f"items: {len(args['items'])} item(s)")
    if description := args.get("description"):
        detail_parts.append(f"description: {description}")

    if detail_parts:
        lines.append(f"  Details: {', '.join(detail_parts)}")

    lines.append("")
    lines.append("Status: pending — awaiting your explicit approval.")
    lines.append(
        "No PayPal write occurs until you approve this proposal in the UI."
    )
    return "\n".join(lines)


def _build_duplicate_proposal_reply(action: dict, validated_args: dict) -> str:
    """Build a truthful reply when a pending proposal already exists.

    Never claims a new proposal was created. Identifies the existing
    pending action and directs the user to approve or reject it.
    """
    tool = action.get("tool", "—")
    action_id = action.get("id", "—")
    args = action.get("args") or {}
    if isinstance(args, str):
        args = json.loads(args)

    tool_labels = {
        "create_order": "Create order",
        "capture_order": "Capture payment",
        "refund_capture": "Refund capture",
        "create_invoice": "Create invoice",
    }
    label = tool_labels.get(tool, tool.replace("_", " ").title())

    lines = [
        f"A {label.lower()} proposal for this exact operation already exists "
        "and is awaiting your approval.",
        f"• {label}: action {action_id}",
    ]

    # Show the actual stored arguments transparently
    detail_parts: list[str] = []
    if args.get("amount"):
        detail_parts.append(f"amount: {args['amount']}")
    if args.get("currency"):
        detail_parts.append(f"currency: {args['currency']}")
    if args.get("description"):
        detail_parts.append(f"description: {args['description']}")
    if args.get("order_id"):
        detail_parts.append(f"order ID: {args['order_id']}")
    if args.get("capture_id"):
        detail_parts.append(f"capture ID: {args['capture_id']}")
    if args.get("recipient_email"):
        detail_parts.append(f"recipient: {args['recipient_email']}")
    if args.get("items"):
        detail_parts.append(f"items: {len(args['items'])} item(s)")
    if description := args.get("description"):
        detail_parts.append(f"description: {description}")

    if detail_parts:
        lines.append(f"  Details: {', '.join(detail_parts)}")

    lines.append("")
    lines.append("Status: pending — no new proposal was created.")
    lines.append(
        "Approve or reject the existing pending action in the Approval Queue."
    )
    return "\n".join(lines)


def _build_clarification_reply(tool: str, missing: list[str]) -> str:
    """Build a factual clarification reply when required parameters are missing.

    Lists exactly what information is needed. Never invents values.
    """
    tool_labels = {
        "create_order": "Create order",
        "capture_order": "Capture payment",
        "refund_capture": "Refund capture",
        "create_invoice": "Create invoice",
    }
    label = tool_labels.get(tool, tool.replace("_", " ").title())

    lines = [
        f"I can prepare a {label.lower()} proposal, but I need more information "
        "before saving it:",
    ]
    for field in missing:
        lines.append(f"• {field}")
    lines.append("")
    lines.append(
        "Please provide the missing details so I can create the pending action "
        "for your approval."
    )
    return "\n".join(lines)


def _build_unclear_reply(ambiguity: str) -> str:
    """Build a clarification reply when the financial intent is ambiguous.

    Never invents values or creates a pending action.
    """
    lines = [
        "I'm not sure which PayPal action you'd like me to propose.",
    ]
    if ambiguity:
        lines.append(f"• {ambiguity}")
    lines.append("")
    lines.append(
        "Please clarify: create an order, capture a payment, refund a capture, "
        "or create an invoice? Include the relevant details (amount, currency, "
        "order/capture ID, recipient, items)."
    )
    return "\n".join(lines)


def _build_validation_failure_reply(exc: ToolValidationError) -> str:
    """Build a factual reply when server-side validation fails.

    Never invents corrected values — reports what was invalid.
    """
    lines = [
        "I couldn't save your request because the provided information "
        "couldn't be validated:",
    ]
    errors = exc.errors if isinstance(exc.errors, list) else [exc.errors]
    for err in errors:
        loc = err.get("loc", [])
        msg = err.get("msg", str(err))
        field = loc[-1] if loc else "field"
        lines.append(f"• {field}: {msg}")
    lines.append("")
    lines.append(
        "Please correct the issue and try again. A pending action is only "
        "created once the request passes validation."
    )
    return "\n".join(lines)


def _build_persistence_failure_reply(error_message: str) -> str:
    """Build a factual reply when pending action persistence fails."""
    lines = [
        "I was unable to save your request due to a server-side error.",
        f"• {error_message}",
        "",
        "No pending action was created. Please try again.",
    ]
    return "\n".join(lines)


def _remember_verified_read(session_id: Optional[str], tool: str, result: dict) -> None:
    """Persist only verified read results into session-scoped context.

    Stores references to verified PayPal resources actually returned by read-only
    tools. Never stores pending proposals or unverified data.
    """
    if not session_id or not isinstance(result, dict):
        return
    from .context import remember_verified_record, remember_session_reference
    if tool == "get_order":
        oid = result.get("id")
        status = (result.get("status") or "").upper()
        if oid and status and status not in {"FAILED", "DENIED", "VOIDED", "UNKNOWN", "ERROR"}:
            evidence = db.VerifiedResourceEvidence(
                kind="order",
                paypal_id=oid,
                status=status,
                amount=result.get("amount"),
                currency=result.get("currency"),
                raw_response=result,
            )
            remember_verified_record(
                "order", oid, status,
                evidence=evidence,
                amount=result.get("amount"), currency=result.get("currency"),
                description=None,
            )
            remember_session_reference(session_id, "order", oid, {
                "kind": "order", "paypal_id": oid, "status": status,
                "amount": result.get("amount"), "currency": result.get("currency"),
                "raw": result,
            })
    elif tool == "get_capture":
        cid = result.get("id")
        status = (result.get("status") or "").upper()
        if cid and status and status not in {"FAILED", "DENIED", "VOIDED", "UNKNOWN", "ERROR"}:
            evidence = db.VerifiedResourceEvidence(
                kind="capture",
                paypal_id=cid,
                status=status,
                amount=result.get("amount"),
                currency=result.get("currency"),
                raw_response=result,
            )
            remember_verified_record(
                "capture", cid, status,
                evidence=evidence,
                amount=result.get("amount"), currency=result.get("currency"),
                description=None,
            )
            remember_session_reference(session_id, "capture", cid, {
                "kind": "capture", "paypal_id": cid, "status": status,
                "amount": result.get("amount"), "currency": result.get("currency"),
                "raw": result,
            })


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
    if tool == "list_pending_actions":
        return {"actions": db.list_pending_actions("pending")}
    raise ValueError(f"Unknown read tool: {tool}")
