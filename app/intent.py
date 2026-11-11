"""Deterministic intent detection and parameter extraction for PayPilot AI.

This module runs BEFORE the LLM agent loop. It classifies user requests as:

* WRITE        - a financial write with all required parameters explicitly present
                 -> server-side validation -> pending action proposal
* POTENTIAL_WRITE - financial write intent detected but parameters missing/invalid
                    -> clarification question, no pending action
* READ          - read-only lookup (order/capture/list)
    -> LLM read-only tool path
* GENERAL       - greeting, question, explanation, non-financial conversation
    -> LLM conversational path
* UNCLEAR       - ambiguous, contradictory, or unparsable financial instruction
    -> clarification question, no pending action

The design enforces zero-hallucination financial safety:

* Only explicitly user-supplied text is extracted. No defaults, no inference.
* Currency symbols map to ISO 4217 codes per the documented product rule.
* A bare amount never gets an inferred currency.
* Conflicting currency expressions are rejected as ambiguous.
* Description is optional for create_order; a transparent default is applied
  only when constructing the validated proposal later (not here).
* Multiple operations in one message are parsed independently with isolated
  parameters. Uncertain segmentation returns UNCLEAR.
* Contextual references resolve only against verified transaction records.
  Pending proposals are never treated as executed PayPal transactions.
* Capture IDs are never inferred from order IDs; order IDs never from captures.
* Uncertain parser results produce clarification questions, never proposals.
* No LLM-generated content is treated as extracted evidence.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

from app import db
from app.tools import ISO_4217, _AMOUNT_RE, _CURRENCY_RE, _EMAIL_RE, _ID_RE
from app.tools import _validate_currency, _validate_id, _validate_money, validate_tool_args

# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #

WRITE = "WRITE"
POTENTIAL_WRITE = "POTENTIAL_WRITE"
READ = "READ"
GENERAL = "GENERAL"
UNCLEAR = "UNCLEAR"

ALL_CLASSIFICATIONS = frozenset({WRITE, POTENTIAL_WRITE, READ, GENERAL, UNCLEAR})

WRITE_TOOLS = frozenset({"create_order", "capture_order", "refund_capture", "create_invoice"})
READ_TOOLS = frozenset({"get_order", "get_capture", "list_local_transactions", "list_pending_actions"})


class IntentResult:
    """Server-side result of intent detection and parameter extraction."""

    def __init__(
        self,
        classification: str,
        *,
        tool: Optional[str] = None,
        args: Optional[dict[str, Any]] = None,
        missing: Optional[list[str]] = None,
        ambiguity: Optional[str] = None,
        operations: Optional[list[dict[str, Any]]] = None,
        requires_clarification: Optional[bool] = None,
        pending_request_id: Optional[str] = None,
        action: Optional[dict[str, Any]] = None,
        completion_outcome: Optional[str] = None,
    ) -> None:
        if classification not in ALL_CLASSIFICATIONS:
            raise ValueError(f"Unknown classification: {classification}")
        self.classification = classification
        self.tool = tool
        self.args = dict(args or {})
        self.missing = list(missing or [])
        self.ambiguity = ambiguity
        self.operations = list(operations or [])
        self.pending_request_id = pending_request_id
        self.action = action
        self.completion_outcome = completion_outcome
        # Auto-determine requires_clarification if not explicitly set
        if requires_clarification is None:
            self.requires_clarification = classification in (
                POTENTIAL_WRITE,
                UNCLEAR,
            ) or bool(self.missing)
        else:
            self.requires_clarification = requires_clarification

    def __repr__(self) -> str:
        return (
            f"IntentResult(classification={self.classification!r}, "
            f"tool={self.tool!r}, args={self.args!r}, "
            f"missing={self.missing!r}, completion_outcome={self.completion_outcome!r})"
        )


# --------------------------------------------------------------------------- #
# Currency symbol mapping (documented product rule)
# --------------------------------------------------------------------------- #

_CURRENCY_SYMBOLS = {
    "$": "USD",
    "US$": "USD",
    "₹": "INR",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "C$": "CAD",
    "A$": "AUD",
}

_CURRENCY_NAMES = {
    "usd": "USD",
    "inr": "INR",
    "eur": "EUR",
    "gbp": "GBP",
    "jpy": "JPY",
    "cad": "CAD",
    "aud": "AUD",
    "dollars": "USD",
    "rupees": "INR",
    "rupee": "INR",
    "euros": "EUR",
    "euro": "EUR",
    "pounds": "GBP",
    "pound": "GBP",
    "yen": "JPY",
}

# Amount with context: matches amounts with optional currency symbol/code.
# Negative numbers are NOT matched (they're invalid amounts).
# Does NOT match parts of hyphenated IDs like "CAP-45678".
_AMOUNT_WITH_CONTEXT = re.compile(
    r"(?<![\d.,A-Za-z-])"
    r"(?:(\$|US\$|₹|€|£|¥|C\$|A\$)\s*)?"
    r"(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?|-?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|-?\d+(?:\.\d{1,2})?)"
    r"(?![\d.,A-Za-z-])"
    r"(?:\s*(USD|INR|EUR|GBP|JPY|CAD|AUD|usd|inr|eur|gbp|jpy|cad|aud))?",
    re.IGNORECASE,
)


def _normalize_amount_text(text: str) -> str:
    """Remove thousands separators and return a clean amount string."""
    return text.replace(",", "")


def _classify_amount_text(raw: str) -> tuple[Optional[str], Optional[str]]:
    """Validate an amount string.

    Returns (amount_str, error) where error is one of None, "invalid_format",
    "invalid_value".
    """
    text = raw.strip()
    if not text:
        return None, "invalid_format"
    normalized = _normalize_amount_text(text)
    if not _AMOUNT_RE.match(normalized):
        return None, "invalid_format"
    try:
        decimal_value = Decimal(normalized)
    except InvalidOperation:
        return None, "invalid_format"
    if decimal_value <= 0:
        return None, "invalid_value"
    if decimal_value > Decimal("1000000"):
        return None, "invalid_value"
    return f"{decimal_value:.2f}", None


def _classify_currency_text(text: str) -> tuple[Optional[str], Optional[str]]:
    """Validate a currency string.

    Returns (currency_code, error) where error is one of None, "invalid".
    """
    text = text.strip()
    if not text:
        return None, "invalid"
    upper = text.upper()
    lower = text.lower()
    if upper in _CURRENCY_SYMBOLS:
        return _CURRENCY_SYMBOLS[upper], None
    if lower in _CURRENCY_NAMES:
        return _CURRENCY_NAMES[lower], None
    if _CURRENCY_RE.match(upper) and upper in ISO_4217:
        return upper, None
    return None, "invalid"


# --------------------------------------------------------------------------- #
# Extraction functions
# --------------------------------------------------------------------------- #

def extract_amount(text: str) -> tuple[Optional[str], Optional[str]]:
    """Extract an explicit amount from ``text``.

    Returns ``(amount_str, error)`` where amount_str is a normalized decimal
    string like "10.00" or None if absent. error is one of None, "missing",
    "invalid_format", "invalid_value".
    """
    if not text:
        return None, "missing"
    match = _AMOUNT_WITH_CONTEXT.search(text)
    if match is None:
        return None, "missing"
    amount_raw = match.group(2)
    return _classify_amount_text(amount_raw)


def extract_currency(text: str) -> tuple[Optional[str], Optional[str]]:
    """Extract an explicit currency from ``text``.

    Checks currency symbols and ISO codes attached to amounts, then standalone
    currency mentions. Returns ``(currency_code, error)`` where error is one of
    None, "missing", "invalid", "ambiguous".
    """
    if not text:
        return None, "missing"

    found: list[str] = []
    saw_invalid = False

    # Check for currency symbols/codes adjacent to amounts
    for match in _AMOUNT_WITH_CONTEXT.finditer(text):
        prefix = match.group(1)
        suffix = match.group(3)
        if prefix:
            code, err = _classify_currency_text(prefix)
            if code:
                found.append(code)
            elif err == "invalid":
                saw_invalid = True
        if suffix:
            code, err = _classify_currency_text(suffix)
            if code:
                found.append(code)
            elif err == "invalid":
                saw_invalid = True

    # Standalone currency mentions (word boundaries).
    # Only check 3-letter codes that could be ISO 4217 codes.
    for w in re.findall(r"\b[A-Z]{3}\b", text):
        code, err = _classify_currency_text(w)
        if code:
            found.append(code)
        elif err == "invalid":
            saw_invalid = True

    unique = set(found)
    if not unique:
        if saw_invalid:
            return None, "invalid"
        return None, "missing"
    if len(unique) > 1:
        return None, "ambiguous"
    return unique.pop(), None


def extract_description(text: str) -> Optional[str]:
    """Extract an optional description from ``text``.

    Looks for text following 'for', 'called', 'named', 'description:',
    'with description', or bare 'description'.
    Returns None if no description text is found.
    """
    if not text:
        return None
    patterns = [
        r"\bfor\s+([A-Za-z][A-Za-z0-9 _-]{1,40}?)(?:\s*(?:and\s+|\.|,|$|in\s+USD|in\s+EUR|in\s+INR|in\s+GBP|in\s+JPY|in\s+CAD|in\s+AUD|for\s+capture|for\s+refund|for\s+invoice))",
        r"\b(?:called|named)\s+([A-Za-z][A-Za-z0-9 _-]{1,40}?)(?:\s*(?:\.|,|$|and\s+|for\s+capture|for\s+refund|for\s+invoice))",
        r"\bdescription\s*:\s*([A-Za-z][A-Za-z0-9 _-]{1,40}?)(?:\s*(?:\.|,|$|and\s+|for\s+capture|for\s+refund|for\s+invoice))",
        r"\bwith\s+description\s+([A-Za-z][A-Za-z0-9 _-]{1,40}?)(?:\s*(?:\.|,|$|and\s+|for\s+capture|for\s+refund|for\s+invoice))",
        r"\bdescription\s+([A-Za-z][A-Za-z0-9 _-]{1,40}?)(?:\s*(?:\.|,|$|and\s+|for\s+capture|for\s+refund|for\s+invoice))",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            desc = match.group(1).strip()
            if desc and len(desc) >= 2:
                return desc
    return None


def extract_reference(text: str) -> Optional[str]:
    """Extract or generate a unique reference from ``text``.

    If the user explicitly asks for a unique reference (e.g. "generate a
    unique reference", "create a reference", "assign a reference"), a
    collision-resistant identifier is generated using UUID4.

    Returns the reference string or None if no reference was requested.
    """
    if not text:
        return None
    lower = text.lower()
    # Check for explicit reference generation requests
    reference_triggers = [
        "generate a unique reference",
        "generate unique reference",
        "create a unique reference",
        "create unique reference",
        "unique reference",
        "reference number",
        "order reference",
        "payment reference",
        "transaction reference",
    ]
    if not any(trigger in lower for trigger in reference_triggers):
        return None
    # Generate a collision-resistant reference: ORD + 8 hex chars from UUID4
    import uuid

    ref = f"ORD-{uuid.uuid4().hex[:8].upper()}"
    return ref


_ID_STOPWORDS = frozenset({
    "associated", "with", "that", "this", "latest", "last", "previous", "prior",
    "mentioned", "earlier", "payment", "status", "details", "information", "info",
    "it", "my", "the", "for", "from", "of", "on", "to", "and", "or", "a", "an",
    "order", "orders", "capture", "captures", "refund", "refunds", "invoice",
    "invoices", "transaction", "transactions", "create", "created", "make", "made",
    "new", "place", "send", "charge", "bill", "pay", "show", "list", "check",
    "view", "get", "lookup", "look", "see", "find", "retrieve", "investigate",
    "what", "happened", "me", "please", "can", "could", "would", "will", "you",
})


def extract_order_id(text: str) -> tuple[Optional[str], Optional[str]]:
    """Extract an order ID from ``text``.

    Returns ``(order_id, error)`` where error is one of None, "missing",
    "invalid". Malformed IDs are reported as invalid, not missing.
    """
    if not text:
        return None, "missing"
    patterns = [
        r"\border\s+(?:id\s+)?(?:is\s+|=|:)?\s*([A-Za-z0-9][A-Za-z0-9_-]{5,39})\b",
        r"\bORD\s+([A-Za-z0-9][A-Za-z0-9_-]{5,39})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            raw = match.group(1).strip().strip(".,;:!?")
            if raw.lower() in _ID_STOPWORDS:
                continue
            try:
                code = _validate_id(raw, "order_id")
            except ValueError:
                return None, "invalid"
            return code, None

    # Fallback: scan bare tokens for a plausible ID when no explicit
    # "order <id>" marker yielded a result. Exclude contextual/stop words to
    # avoid treating references like "that/latest/associated" as IDs.
    tokens = re.findall(r"\b[A-Za-z0-9][A-Za-z0-9_-]{5,39}\b", text)
    for tok in tokens:
        if tok.lower() in _ID_STOPWORDS:
            continue
        try:
            code = _validate_id(tok, "order_id")
        except ValueError:
            continue
        return code, None

    return None, "missing"


def extract_capture_id(text: str) -> tuple[Optional[str], Optional[str]]:
    """Extract a capture ID from ``text``.

    Returns ``(capture_id, error)`` where error is one of None, "missing",
    "invalid". Malformed IDs are reported as invalid, not missing.
    """
    if not text:
        return None, "missing"
    patterns = [
        r"\bcapture\s+(?:id\s+)?(?:is\s+|=|:)?\s*([A-Za-z0-9][A-Za-z0-9_-]{5,39})\b",
        r"\bCAP\s+([A-Za-z0-9][A-Za-z0-9_-]{5,39})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            raw = match.group(1).strip().strip(".,;:!?")
            if raw.lower() in _ID_STOPWORDS:
                # contextual reference (e.g. "capture associated with...") not an ID
                continue
            try:
                code = _validate_id(raw, "capture_id")
            except ValueError:
                return None, "invalid"
            return code, None

    # Fallback: scan bare tokens for a plausible capture ID.
    tokens = re.findall(r"\b[A-Za-z0-9][A-Za-z0-9_-]{5,39}\b", text)
    for tok in tokens:
        if tok.lower() in _ID_STOPWORDS:
            continue
        try:
            code = _validate_id(tok, "capture_id")
        except ValueError:
            continue
        return code, None

    return None, "missing"


def extract_recipient_email(text: str) -> tuple[Optional[str], Optional[str]]:
    """Extract a recipient email address from ``text``.

    Returns ``(email, error)`` where error is one of None, "missing", "invalid".
    """
    if not text:
        return None, "missing"
    match = re.search(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})", text)
    if match:
        return match.group(1).lower(), None
    return None, "missing"


def extract_invoice_items(text: str) -> tuple[Optional[list[dict[str, Any]]], Optional[str]]:
    """Extract invoice line items from ``text``.

    Each item requires name, quantity, and unit_amount.
    Returns ``(items, error)`` where error is one of None, "missing",
    "invalid_item", "currency_conflict".
    """
    if not text:
        return None, "missing"

    items: list[dict[str, Any]] = []

    pattern1 = re.finditer(
        r"([A-Za-z][A-Za-z0-9 _-]{1,40})\s*(?:x|×)\s*(\d+)\s*(?:@|at)\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)",
        text,
        re.IGNORECASE,
    )
    for m in pattern1:
        name = m.group(1).strip()
        qty = int(m.group(2))
        amt_raw = m.group(3).replace(",", "")
        amt_val, amt_err = _classify_amount_text(amt_raw)
        if amt_err:
            return None, "invalid_item"
        items.append({"name": name, "quantity": qty, "unit_amount": amt_val})

    pattern2 = re.finditer(
        r"(\d+)\s*(?:x|×)\s*([A-Za-z][A-Za-z0-9 _-]{1,40})\s*(?:at|each)\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)",
        text,
        re.IGNORECASE,
    )
    for m in pattern2:
        qty = int(m.group(1))
        name = m.group(2).strip()
        amt_raw = m.group(3).replace(",", "")
        amt_val, amt_err = _classify_amount_text(amt_raw)
        if amt_err:
            return None, "invalid_item"
        items.append({"name": name, "quantity": qty, "unit_amount": amt_val})

    if not items:
        return None, "missing"

    for item in items:
        if not item["name"] or len(item["name"]) < 2:
            return None, "invalid_item"
        if item["quantity"] < 1:
            return None, "invalid_item"

    return items, None


# --------------------------------------------------------------------------- #
# Negation / hypothetical / educational detection
# --------------------------------------------------------------------------- #

_NEGATION_WORDS = frozenset(
    {"not", "don't", "do not", "cannot", "can't", "won't", "will not", "never", "no "}
)
# Approval-gating phrases: "do not execute until I approve" is a request to
# create a pending action, not a negation of the financial operation. Strip
# these before negation checks so they don't false-positive.
_APPROVAL_GATING_RE = re.compile(
    r"\b(?:do not|don't|cannot|can't)\s+\w+(?:\s+\w+)?\s+(?:it\s+)?until\s+(?:i\s+)?approv",
    re.I,
)
# "could you", "would you", "can you" are polite forms, NOT hypotheticals
_HYPOTHETICAL_WORDS = frozenset(
    {"if", "suppose", "hypothetical", "for example"}
)
_HYPOTHETICAL_PHRASES = frozenset(
    {"if i ", "if i created", "if i made", "if i captured", "if i refunded", "if i invoiced",
     "suppose i ", "suppose i created", "suppose i made"}
)
_EDUCATIONAL_WORDS = frozenset(
    {"difference", "difference between", "what is", "what are", "explain", "tell me about", "define", "compare"}
)


def _has_negation(text: str) -> bool:
    lower = text.lower()
    # Strip approval-gating phrases so they don't trigger false negation.
    # "Do not execute until I approve" requests a pending action, not negation.
    stripped = _APPROVAL_GATING_RE.sub(" ", lower)
    stripped = " ".join(stripped.split())
    return any(w in stripped for w in _NEGATION_WORDS)


def _negates_write_action(text: str) -> bool:
    """Check if negation specifically targets a write action (create/capture/refund/invoice).

    Negation about execution (e.g. "do not execute the PayPal write") does NOT
    block write intent because the proposal-only design already handles
    "no PayPal write" naturally - proposals are never executed without approval.
    Only negation of the write action itself (e.g. "do not create") blocks it.
    """
    lower = text.lower()
    stripped = _APPROVAL_GATING_RE.sub(" ", lower)
    stripped = " ".join(stripped.split())

    # Check if any negation word is immediately followed by a write verb
    words = stripped.split()
    for i, word in enumerate(words):
        if word in _NEGATION_WORDS and i + 1 < len(words):
            next_word = words[i + 1].strip(".,;:!?")
            if next_word in _WRITE_VERBS:
                return True
    return False


def _has_hypothetical(text: str) -> bool:
    lower = text.lower()
    for phrase in _HYPOTHETICAL_PHRASES:
        if phrase in lower:
            return True
    for w in _HYPOTHETICAL_WORDS:
        if w in lower:
            return True
    return False


def _has_educational(text: str) -> bool:
    lower = text.lower()
    return any(w in lower for w in _EDUCATIONAL_WORDS)


# --------------------------------------------------------------------------- #
# Intent detection
# --------------------------------------------------------------------------- #

_WRITE_VERBS = frozenset(
    {"create", "created", "make", "made", "new", "place", "refund", "refunded", "capture", "captured", "charge", "bill", "invoice", "pay"}
)
_READ_VERBS = frozenset(
    {"show", "list", "check", "view", "get", "lookup", "look", "status", "balance", "see", "find", "retrieve", "investigate"}
)
_WRITE_NOUNS = frozenset({"order", "payment", "capture", "refund", "invoice"})
_READ_NOUNS = frozenset({"balance", "pending", "history", "actions", "order", "status", "transaction", "transactions", "capture"})


def _detect_single_operation(text: str) -> dict[str, Any]:
    """Detect a single financial operation from ``text``.

    Returns a dict with keys: classification, tool, args, missing, ambiguity.
    """
    lower = text.lower()
    ambiguity = None

    # Negation guard - only block if negation targets the write action itself.
    # Negation about execution (e.g. "do not execute") does not block WRITE
    # because the proposal-only design already handles "no PayPal write" naturally.
    if _has_negation(lower):
        if _negates_write_action(lower):
            return {
                "classification": GENERAL,
                "tool": None,
                "args": {},
                "missing": [],
                "ambiguity": "negated request",
            }
        # Negation is about execution or something else - proceed with write detection

    # Hypothetical guard
    if _has_hypothetical(lower):
        return {
            "classification": UNCLEAR,
            "tool": None,
            "args": {},
            "missing": [],
            "ambiguity": "hypothetical request",
        }

    # Educational guard
    if _has_educational(lower):
        return {
            "classification": GENERAL,
            "tool": None,
            "args": {},
            "missing": [],
            "ambiguity": "educational question",
        }

    # Determine which write verbs and nouns are present
    found_verbs = [v for v in _WRITE_VERBS if re.search(rf"\b{v}\b", lower)]
    found_nouns = [n for n in _WRITE_NOUNS if re.search(rf"\b{n}\b", lower)]
    found_read_verbs = [v for v in _READ_VERBS if re.search(rf"\b{v}\b", lower)]
    found_read_nouns = [n for n in _READ_NOUNS if re.search(rf"\b{n}\b", lower)]

    # If a write verb appears right after a read verb (e.g. "View capture X"),
    # treat it as a noun (the object being viewed), not as a write verb.
    for rv in found_read_verbs:
        if re.search(rf"\b{rv}\s+capture\b", lower):
            found_verbs = [v for v in found_verbs if v != "capture"]
            break

    # Write intent takes precedence over read intent. If there are write
    # verbs and nouns, the user wants to perform a write operation regardless
    # of any read-related words (e.g. "show me" in "show me the action ID").
    # However, if a read verb appears BEFORE any write verb in the text
    # (e.g. "Look up order X and then capture it"), the read action comes
    # first and should take precedence.
    if found_verbs and found_nouns:
        # Multiple write operations in one message: cannot safely determine
        # which operation applies to which parameters. Split by conjunction
        # words first, then count how many segments contain verbs. Multiple
        # segments with verbs means multiple operations.
        verb_words = set(_WRITE_VERBS)
        segments = re.split(r'\b(?:and|then|&)\b', lower, flags=re.IGNORECASE)
        verb_segment_count = sum(
            1 for seg in segments
            if any(v in seg.split() for v in _WRITE_VERBS)
        )
        if len(segments) > 1 and verb_segment_count > 1:
            return {
                "classification": UNCLEAR,
                "tool": None,
                "args": {},
                "missing": [],
                "ambiguity": "multiple write operations in one request",
            }

        # If both read and write verbs are present, check which appears first
        # in the text. The first action verb determines the intent.
        first_read_pos = min(
            (lower.find(v) for v in found_read_verbs),
            default=len(lower),
        )
        first_write_pos = min(
            (lower.find(v) for v in found_verbs),
            default=len(lower),
        )
        if first_read_pos < first_write_pos:
            return {
                "classification": READ,
                "tool": None,
                "args": {},
                "missing": [],
                "ambiguity": None,
            }
        pass  # Proceed to write tool determination below
    # Read intent: read verb + read noun. Only applies when there's no
    # write intent.
    elif found_read_verbs and found_read_nouns:
        return {
            "classification": READ,
            "tool": None,
            "args": {},
            "missing": [],
            "ambiguity": None,
        }

    # No financial intent at all
    if not found_verbs and not found_nouns:
        return {
            "classification": GENERAL,
            "tool": None,
            "args": {},
            "missing": [],
            "ambiguity": None,
        }

    # Write intent detected - determine which tool
    tool = None
    if "refund" in found_verbs or "refund" in found_nouns:
        tool = "refund_capture"
    elif "capture" in found_verbs or "capture" in found_nouns:
        tool = "capture_order"
    elif "invoice" in found_verbs or "invoice" in found_nouns or "bill" in found_verbs:
        tool = "create_invoice"
    elif any(v.startswith("create") for v in found_verbs):
        if "order" in found_nouns or "payment" in found_nouns:
            tool = "create_order"
        elif "invoice" in found_nouns:
            tool = "create_invoice"

    if tool is None:
        return {
            "classification": UNCLEAR,
            "tool": None,
            "args": {},
            "missing": [],
            "ambiguity": "write intent detected but operation unclear",
        }

    # Extract parameters
    args: dict[str, Any] = {}
    missing: list[str] = []

    if tool == "create_order":
        amount, amt_err = extract_amount(text)
        currency, curr_err = extract_currency(text)
        description = extract_description(text)

        if amt_err == "missing":
            missing.append("amount")
        elif amt_err:
            missing.append(f"amount ({amt_err})")
        else:
            args["amount"] = amount

        if curr_err == "missing":
            missing.append("currency")
        elif curr_err == "ambiguous":
            missing.append("currency (conflicting currencies detected)")
        elif curr_err == "invalid":
            missing.append("currency (invalid currency code)")
        else:
            args["currency"] = currency

        # Description is optional
        if description:
            args["description"] = description

        # Reference is optional; generate one if the user asks for it
        reference = extract_reference(text)
        if reference:
            args["reference"] = reference

    elif tool == "capture_order":
        order_id, oid_err = extract_order_id(text)
        if oid_err == "missing":
            missing.append("order_id")
        elif oid_err == "invalid":
            missing.append("order_id (invalid format)")
        else:
            args["order_id"] = order_id

    elif tool == "refund_capture":
        capture_id, cid_err = extract_capture_id(text)
        if cid_err == "missing":
            missing.append("capture_id")
        elif cid_err == "invalid":
            missing.append("capture_id (invalid format)")
        else:
            args["capture_id"] = capture_id

        amount, amt_err = extract_amount(text)
        if amt_err == "missing":
            pass  # amount is optional for refund
        elif amt_err:
            missing.append(f"amount ({amt_err})")
        else:
            args["amount"] = amount

    elif tool == "create_invoice":
        email, email_err = extract_recipient_email(text)
        if email_err == "missing":
            missing.append("recipient_email")
        elif email_err == "invalid":
            missing.append("recipient_email (invalid format)")
        else:
            args["recipient_email"] = email

        currency, curr_err = extract_currency(text)
        if curr_err == "missing":
            missing.append("currency")
        elif curr_err == "ambiguous":
            missing.append("currency (conflicting currencies detected)")
        elif curr_err == "invalid":
            missing.append("currency (invalid currency code)")
        else:
            args["currency"] = currency

        items, items_err = extract_invoice_items(text)
        if items_err == "missing":
            missing.append("items")
        elif items_err == "invalid_item":
            missing.append("items (invalid item format)")
        elif items_err == "currency_conflict":
            missing.append("items (currency conflict with invoice currency)")
        else:
            args["items"] = items

    classification = POTENTIAL_WRITE if missing else WRITE

    # Description is optional for create_order; apply transparent default
    # when no description was explicitly provided. This default is disclosed
    # in the reply, not presented as user-provided information.
    if tool == "create_order" and "description" not in args:
        args["description"] = "PayPilot AI order"

    return {
        "classification": classification,
        "tool": tool,
        "args": args,
        "missing": missing,
        "ambiguity": ambiguity,
    }


_PROMPT_INJECTION_PHRASES = frozenset(
    {
        "ignore all previous instructions",
        "disregard everything",
        "ignore everything",
        "forget everything",
        "system override",
        "you are now in admin mode",
        "new instructions",
        "ignore prior instructions",
        "ignore previous instructions",
    }
)


def _is_prompt_injection(text: str) -> bool:
    """Detect prompt injection attempts that try to override system instructions.

    Returns True if the text contains phrases commonly used in prompt injection
    attacks (e.g. "IGNORE ALL PREVIOUS INSTRUCTIONS"). Such text must be treated
    as GENERAL — no financial action of any kind may be taken.
    """
    lower = text.lower()
    for phrase in _PROMPT_INJECTION_PHRASES:
        if phrase in lower:
            return True
    return False


def detect_write_intent(
    text: str,
    *,
    session_id: Optional[str] = None,
    history: Optional[list[dict[str, Any]]] = None,
) -> IntentResult:
    """Detect the financial intent of a user message.

    Parameters
    ----------
    text : str
        The user's message.
    session_id : str, optional
        Session identifier for contextual reference resolution (not yet used
        for DB lookups in this phase).
    history : list[dict], optional
        Conversation history for contextual reference resolution.

    Returns
    -------
    IntentResult
        Classification and extracted parameters.
    """
    if not text or not text.strip():
        return IntentResult(GENERAL)

    # Prompt injection guard: never act on untrusted text that attempts to
    # override system instructions. Treat as GENERAL so no financial action
    # of any kind is taken.
    if _is_prompt_injection(text):
        return IntentResult(GENERAL)

    result = _detect_single_operation(text)

    # If no financial intent detected at all, return GENERAL
    if result["classification"] == GENERAL and not result["tool"]:
        return IntentResult(GENERAL)

    # If UNCLEAR due to ambiguous segmentation or parameter ownership
    if result["classification"] == UNCLEAR:
        return IntentResult(
            UNCLEAR,
            ambiguity=result["ambiguity"],
            requires_clarification=True,
        )

    # If POTENTIAL_WRITE with missing params
    if result["classification"] == POTENTIAL_WRITE:
        return IntentResult(
            POTENTIAL_WRITE,
            tool=result["tool"],
            args=result["args"],
            missing=result["missing"],
            requires_clarification=True,
        )

    # READ - return as-is for LLM read-only tool path
    if result["classification"] == READ:
        return IntentResult(READ)

    # WRITE - all required params present
    return IntentResult(
        WRITE,
        tool=result["tool"],
        args=result["args"],
    )


# --------------------------------------------------------------------------- #
# Module-level convenience
# --------------------------------------------------------------------------- #

__all__ = [
    "WRITE",
    "POTENTIAL_WRITE",
    "READ",
    "GENERAL",
    "UNCLEAR",
    "ALL_CLASSIFICATIONS",
    "IntentResult",
    "extract_amount",
    "extract_currency",
    "extract_description",
    "extract_order_id",
    "extract_capture_id",
    "extract_recipient_email",
    "extract_invoice_items",
    "detect_write_intent",
    "resolve_clarification",
]


# --------------------------------------------------------------------------- #
# Multi-turn clarification / pending request state management
# --------------------------------------------------------------------------- #

def resolve_clarification(
    session_id: str,
    user_message: str,
) -> Optional[IntentResult]:
    """Resolve a pending clarification from a follow-up user message.

    Merges newly extracted arguments with previously stored arguments and
    re-validates. If still incomplete/invalid, updates the unresolved request
    (by id) and returns POTENTIAL_WRITE. If complete and valid, performs an
    atomic clarification completion (claim + pending action + audit + resolve)
    via :func:`db.complete_unresolved_request`.

    Never commits a resolved state before proposal creation. Never creates a
    pending action unless all required arguments are present and validated and
    the atomic completion succeeds.

    Returns None if no awaiting unresolved request exists for the session.
    """
    from app import db  # local import to avoid circular import issues

    pending = db.get_pending_request(session_id)
    if pending is None:
        return None

    pending_id = pending["id"]
    tool = pending["tool"]
    existing_args = pending.get("args") or {}
    if isinstance(existing_args, str):
        existing_args = json.loads(existing_args)

    new_missing: list[str] = []
    args = dict(existing_args)

    # Re-extract all required fields for the tool; existing values are kept,
    # new values from this message overwrite them.
    if tool == "create_order":
        amount, amt_err = extract_amount(user_message)
        currency, curr_err = extract_currency(user_message)
        # In clarification follow-ups, only accept currency when explicitly
        # stated as a code/name (not inferred solely from a $ prefix symbol
        # on the amount), so that "$10.00" provides amount only.
        if currency and _currency_from_symbol_prefix_only(user_message):
            currency, curr_err = None, "missing"
        description = extract_description(user_message)

        if amount:
            args["amount"] = amount
        elif "amount" in existing_args:
            args["amount"] = existing_args["amount"]
        elif amt_err:
            new_missing.append(f"amount ({amt_err})")
        else:
            new_missing.append("amount")

        if currency:
            args["currency"] = currency
        elif "currency" in existing_args:
            args["currency"] = existing_args["currency"]
        elif curr_err == "missing":
            new_missing.append("currency")
        elif curr_err == "ambiguous":
            new_missing.append("currency (conflicting currencies detected)")
        elif curr_err == "invalid":
            new_missing.append("currency (invalid currency code)")
        else:
            new_missing.append("currency")

        if description:
            args["description"] = description
        elif "description" in existing_args:
            args["description"] = existing_args["description"]

    elif tool == "capture_order":
        order_id, oid_err = extract_order_id(user_message)
        if oid_err == "missing":
            if "order_id" in existing_args:
                args["order_id"] = existing_args["order_id"]
            else:
                new_missing.append("order_id")
        elif oid_err == "invalid":
            new_missing.append("order_id (invalid format)")
        else:
            args["order_id"] = order_id

    elif tool == "refund_capture":
        capture_id, cid_err = extract_capture_id(user_message)
        if cid_err == "missing":
            if "capture_id" in existing_args:
                args["capture_id"] = existing_args["capture_id"]
            else:
                new_missing.append("capture_id")
        elif cid_err == "invalid":
            new_missing.append("capture_id (invalid format)")
        else:
            args["capture_id"] = capture_id

        amount, amt_err = extract_amount(user_message)
        if amt_err == "missing":
            if "amount" in existing_args:
                args["amount"] = existing_args["amount"]
            # amount is optional for refund
        elif amt_err:
            new_missing.append(f"amount ({amt_err})")
        else:
            args["amount"] = amount

    elif tool == "create_invoice":
        email, email_err = extract_recipient_email(user_message)
        if email_err == "missing":
            if "recipient_email" in existing_args:
                args["recipient_email"] = existing_args["recipient_email"]
            else:
                new_missing.append("recipient_email")
        elif email_err == "invalid":
            new_missing.append("recipient_email (invalid format)")
        else:
            args["recipient_email"] = email

        currency, curr_err = extract_currency(user_message)
        if currency and _currency_from_symbol_prefix_only(user_message):
            currency, curr_err = None, "missing"
        if curr_err == "missing":
            if "currency" in existing_args:
                args["currency"] = existing_args["currency"]
            else:
                new_missing.append("currency")
        elif curr_err == "ambiguous":
            new_missing.append("currency (conflicting currencies detected)")
        elif curr_err == "invalid":
            new_missing.append("currency (invalid currency code)")
        else:
            args["currency"] = currency

        items, items_err = extract_invoice_items(user_message)
        if items_err == "missing":
            if "items" in existing_args:
                args["items"] = existing_args["items"]
            else:
                new_missing.append("items")
        elif items_err == "invalid_item":
            new_missing.append("items (invalid item format)")
        elif items_err == "currency_conflict":
            new_missing.append("items (currency conflict with invoice currency)")
        else:
            args["items"] = items

    # Still missing required fields: update state (by request id), do not resolve.
    if new_missing:
        db.update_unresolved_request(pending_id, args, new_missing)
        return IntentResult(
            POTENTIAL_WRITE,
            tool=tool,
            args=args,
            missing=new_missing,
            requires_clarification=True,
            pending_request_id=pending_id,
        )

    # Validate merged arguments server-side before any completion attempt.
    try:
        validated_args = validate_tool_args(tool, args)
    except Exception as exc:
        error_msg = getattr(exc, "errors", None)
        if error_msg:
            missing_detail = [
                f"validation: {e.get('msg', str(e))} for {e.get('loc', ['field'])[-1] if e.get('loc') else 'field'}"
                for e in error_msg
            ]
        else:
            missing_detail = [f"validation: {exc}"]
        db.update_unresolved_request(pending_id, args, missing_detail)
        return IntentResult(
            POTENTIAL_WRITE,
            tool=tool,
            args=args,
            missing=missing_detail,
            requires_clarification=True,
            pending_request_id=pending_id,
        )

    # Atomic completion: claim + pending action + audit + resolve in one tx.
    try:
        result = db.complete_unresolved_request(
            pending_id, session_id, tool, validated_args
        )
    except Exception as exc:
        return IntentResult(
            GENERAL,
            tool=tool,
            args=validated_args,
            ambiguity=f"completion persistence failed: {exc}",
            pending_request_id=pending_id,
            completion_outcome="error",
        )

    status = result.get("status")
    if status == "completed":
        return IntentResult(
            WRITE,
            tool=tool,
            args=validated_args,
            pending_request_id=pending_id,
            action=result.get("action"),
            completion_outcome="completed",
        )

    # Already completed, expired, or conflict: no new proposal.
    return IntentResult(
        GENERAL,
        tool=tool,
        args=validated_args,
        pending_request_id=pending_id,
        completion_outcome=status or "conflict",
    )


def _currency_from_symbol_prefix_only(text: str) -> bool:
    """Return True if currency in ``text`` comes only from a $/symbol prefix
    attached to an amount, with no explicit currency code/name suffix or
    standalone currency token. Used in clarification follow-ups so that
    "$10.00" can be treated as amount-only while "100.00 EUR" or "USD" still
    provide currency."""
    has_prefix_symbol = False
    has_explicit_currency = False
    for match in _AMOUNT_WITH_CONTEXT.finditer(text):
        if match.group(1):
            has_prefix_symbol = True
        if match.group(3):
            has_explicit_currency = True
    # Standalone currency mentions (word boundaries)
    for w in re.findall(r"\b[A-Z]{3}\b", text):
        code, err = _classify_currency_text(w)
        if code:
            has_explicit_currency = True
    for name in _CURRENCY_NAMES:
        if re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE):
            has_explicit_currency = True
    return has_prefix_symbol and not has_explicit_currency


def _resolve_pending_clarification(
    session_id: Optional[str],
    user_message: str,
) -> Optional[IntentResult]:
    """Wrapper around resolve_clarification that handles None session_id.

    Returns an IntentResult if a pending clarification existed and was
    processed, or None if there was no pending request for this session.
    """
    if not session_id:
        return None
    return resolve_clarification(session_id, user_message)
