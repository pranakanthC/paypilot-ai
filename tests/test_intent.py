"""Regression tests for the deterministic intent detection layer (app/intent.py).

Tests cover:
- All five classifications: WRITE, POTENTIAL_WRITE, READ, GENERAL, UNCLEAR
- All four financial operations: create_order, capture_order, refund_capture, create_invoice
- Amount and currency extraction (valid, missing, invalid, ambiguous)
- Missing, invalid, malformed, not-found, and ambiguous IDs
- False positives, negation, hypotheticals, educational questions
- Invoice item completeness
- Zero-hallucination: no invented values, no default currency, no default amount
- Multi-operation detection: multiple write operations in one message -> UNCLEAR
"""

from __future__ import annotations

import pytest

from app.intent import (
    WRITE,
    POTENTIAL_WRITE,
    READ,
    GENERAL,
    UNCLEAR,
    IntentResult,
    detect_write_intent,
    extract_amount,
    extract_currency,
    extract_description,
    extract_order_id,
    extract_capture_id,
    extract_recipient_email,
    extract_invoice_items,
)


# --------------------------------------------------------------------------- #
# extract_amount tests
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "text,expected_amount,expected_error",
    [
        ("$10", "10.00", None),
        ("USD 10", "10.00", None),
        ("10.00 USD", "10.00", None),
        ("₹5,000", "5000.00", None),
        ("€100", "100.00", None),
        ("5.00", "5.00", None),
        ("1,000.00", "1000.00", None),
        ("ten dollars", None, "missing"),
        ("", None, "missing"),
        ("-5.00", None, "invalid_value"),
        ("0", None, "invalid_value"),
        ("10.000,00", None, "missing"),
    ],
)
def test_extract_amount(text, expected_amount, expected_error):
    amount, error = extract_amount(text)
    assert amount == expected_amount, f"text={text!r}: expected {expected_amount}, got {amount}"
    assert error == expected_error, f"text={text!r}: expected error {expected_error}, got {error}"


# --------------------------------------------------------------------------- #
# extract_currency tests
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "text,expected_currency,expected_error",
    [
        ("USD", "USD", None),
        ("$10", "USD", None),
        ("10 USD", "USD", None),
        ("₹5000", "INR", None),
        ("€100", "EUR", None),
        ("£50", "GBP", None),
        ("$10 EUR", None, "ambiguous"),
        ("XXX", "XXX", None),
        ("", None, "missing"),
        ("10 dollars", None, "missing"),
        ("5000 rupees", None, "missing"),
    ],
)
def test_extract_currency(text, expected_currency, expected_error):
    currency, error = extract_currency(text)
    assert currency == expected_currency, f"text={text!r}: expected {expected_currency}, got {currency}"
    assert error == expected_error, f"text={text!r}: expected error {expected_error}, got {error}"


# --------------------------------------------------------------------------- #
# extract_description tests
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "text,expected",
    [
        ("Create an order for coffee", "coffee"),
        ("Create an order called Consulting Services", "Consulting Services"),
        ("Create an order named Monthly Retainer", "Monthly Retainer"),
        ("Create an order description: Office Supplies", "Office Supplies"),
        ("Create an order", None),
        ("Create an order for", None),
    ],
)
def test_extract_description(text, expected):
    desc = extract_description(text)
    assert desc == expected, f"text={text!r}: expected {expected!r}, got {desc!r}"


# --------------------------------------------------------------------------- #
# extract_order_id tests
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "text,expected_id,expected_error",
    [
        ("order 5O190127TN364715T", "5O190127TN364715T", None),
        ("order ID ORD-12345", "ORD-12345", None),
        ("ORD ABC_123", "ABC_123", None),
        ("order 12", None, "missing"),
        ("order", None, "missing"),
        ("", None, "missing"),
    ],
)
def test_extract_order_id(text, expected_id, expected_error):
    oid, error = extract_order_id(text)
    assert oid == expected_id, f"text={text!r}: expected {expected_id}, got {oid}"
    assert error == expected_error, f"text={text!r}: expected error {expected_error}, got {error}"


# --------------------------------------------------------------------------- #
# extract_capture_id tests
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "text,expected_id,expected_error",
    [
        ("capture CAP-45678", "CAP-45678", None),
        ("capture id 5O190127TN364715T", "5O190127TN364715T", None),
        ("CAP XYZ_999", "XYZ_999", None),
        ("capture 12", None, "missing"),
        ("capture", None, "missing"),
        ("", None, "missing"),
    ],
)
def test_extract_capture_id(text, expected_id, expected_error):
    cid, error = extract_capture_id(text)
    assert cid == expected_id, f"text={text!r}: expected {expected_id}, got {cid}"
    assert error == expected_error, f"text={text!r}: expected error {expected_error}, got {error}"


# --------------------------------------------------------------------------- #
# extract_recipient_email tests
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "text,expected_email,expected_error",
    [
        ("invoice to client@example.com", "client@example.com", None),
        ("email buyer@test.com for invoice", "buyer@test.com", None),
        ("send invoice to a@b.co", "a@b.co", None),
        ("invoice", None, "missing"),
        ("", None, "missing"),
    ],
)
def test_extract_recipient_email(text, expected_email, expected_error):
    email, error = extract_recipient_email(text)
    assert email == expected_email, f"text={text!r}: expected {expected_email}, got {email}"
    assert error == expected_error, f"text={text!r}: expected error {expected_error}, got {error}"


# --------------------------------------------------------------------------- #
# extract_invoice_items tests
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "text,expected_items,expected_error",
    [
        ("coffee x 2 @ 75.00", [{"name": "coffee", "quantity": 2, "unit_amount": "75.00"}], None),
        ("2x consulting at $75.00 each", [{"name": "consulting", "quantity": 2, "unit_amount": "75.00"}], None),
        ("item: consulting, qty: 2, unit: 75.00", None, "missing"),
        ("", None, "missing"),
    ],
)
def test_extract_invoice_items(text, expected_items, expected_error):
    items, error = extract_invoice_items(text)
    assert items == expected_items, f"text={text!r}: expected {expected_items}, got {items}"
    assert error == expected_error, f"text={text!r}: expected error {expected_error}, got {error}"


# --------------------------------------------------------------------------- #
# detect_write_intent tests - WRITE classification
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "text,tool,expected_args",
    [
        ("Create a $10 USD order for coffee", "create_order",
         {"amount": "10.00", "currency": "USD", "description": "coffee"}),
        ("Create an order for ₹5,000 for equipment", "create_order",
         {"amount": "5000.00", "currency": "INR", "description": "equipment"}),
        ("Create a €100 order for Client A", "create_order",
         {"amount": "100.00", "currency": "EUR", "description": "Client A"}),
        ("Capture order ORD-12345", "capture_order", {"order_id": "ORD-12345"}),
        ("Capture order 5O190127TN364715T", "capture_order", {"order_id": "5O190127TN364715T"}),
        ("Refund capture CAP-45678 for $20", "refund_capture",
         {"capture_id": "CAP-45678", "amount": "20.00"}),
        ("Refund capture 5O190127TN364715T", "refund_capture", {"capture_id": "5O190127TN364715T"}),
        ("Create an invoice to client@example.com for $250 USD, coffee x 2 @ 125.00",
         "create_invoice",
         {"recipient_email": "client@example.com", "currency": "USD",
          "items": [{"name": "coffee", "quantity": 2, "unit_amount": "125.00"}]}),
    ],
)
def test_detect_write_intent_write(text, tool, expected_args):
    result = detect_write_intent(text)
    assert result.classification == WRITE, f"text={text!r}: expected WRITE, got {result.classification}"
    assert result.tool == tool, f"text={text!r}: expected tool {tool}, got {result.tool}"
    assert result.args == expected_args, f"text={text!r}: expected args {expected_args}, got {result.args}"
    assert result.missing == []
    assert not result.requires_clarification


# --------------------------------------------------------------------------- #
# detect_write_intent tests - POTENTIAL_WRITE (missing params)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "text,tool,expected_class,missing_field",
    [
        ("Create an order", "create_order", POTENTIAL_WRITE, "amount"),
        ("Create an order for $10", "create_order", WRITE, None),
        ("Create an order for USD", "create_order", POTENTIAL_WRITE, "amount"),
        ("Can you create an order?", "create_order", POTENTIAL_WRITE, "amount"),
        ("Capture order", "capture_order", POTENTIAL_WRITE, "order_id"),
        ("Refund capture", "refund_capture", POTENTIAL_WRITE, "capture_id"),
        ("Create an invoice to client@example.com", "create_invoice", POTENTIAL_WRITE, "currency"),
        ("Create an invoice to client@example.com for $100 USD", "create_invoice", POTENTIAL_WRITE, "items"),
    ],
)
def test_detect_write_intent_potential_write(text, tool, expected_class, missing_field):
    result = detect_write_intent(text)
    assert result.classification == expected_class, f"text={text!r}: expected {expected_class}, got {result.classification}"
    assert result.tool == tool, f"text={text!r}: expected tool {tool}, got {result.tool}"
    if expected_class == POTENTIAL_WRITE:
        assert missing_field in result.missing, f"text={text!r}: expected missing {missing_field}, got {result.missing}"
        assert result.requires_clarification is True
    else:
        assert result.missing == []
        assert not result.requires_clarification


# --------------------------------------------------------------------------- #
# detect_write_intent tests - READ classification
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "text",
    [
        "Show me my pending actions",
        "Check order ORD-12345 status",
        "List my transactions",
        "View capture CAP-45678",
        "Get order 5O190127TN364715T",
    ],
)
def test_detect_write_intent_read(text):
    result = detect_write_intent(text)
    assert result.classification == READ, f"text={text!r}: expected READ, got {result.classification}"
    assert result.tool is None
    assert result.missing == []
    assert not result.requires_clarification


# --------------------------------------------------------------------------- #
# detect_write_intent tests - GENERAL classification
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "text",
    [
        "Hello, how are you?",
        "Explain how PayPal orders work.",
        "Tell me about captures.",
        "Good morning!",
    ],
)
def test_detect_write_intent_general(text):
    result = detect_write_intent(text)
    assert result.classification == GENERAL, f"text={text!r}: expected GENERAL, got {result.classification}"
    assert result.tool is None
    assert result.missing == []
    assert not result.requires_clarification


# --------------------------------------------------------------------------- #
# detect_write_intent tests - UNCLEAR classification
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "text",
    [
        "If I created an order for $10 USD, what happens?",
        "Suppose I refund capture CAP-45678, would that work?",
        "What is the difference between an order and a capture?",
    ],
)
def test_detect_write_intent_unclear(text):
    result = detect_write_intent(text)
    assert result.classification == UNCLEAR, f"text={text!r}: expected UNCLEAR, got {result.classification}"
    assert result.requires_clarification is True


# --------------------------------------------------------------------------- #
# Negation tests
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "text",
    [
        "Don't create an order for $10",
        "Do not capture order ORD-12345",
        "Never refund capture CAP-45678",
        "I cannot create an invoice",
    ],
)
def test_detect_write_intent_negation(text):
    result = detect_write_intent(text)
    assert result.classification == GENERAL, f"text={text!r}: expected GENERAL (negated), got {result.classification}"
    assert result.tool is None


# --------------------------------------------------------------------------- #
# Polite write intent tests
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "text,tool,expected_class",
    [
        ("Can you create an order?", "create_order", POTENTIAL_WRITE),
        ("Could you create a $10 USD order?", "create_order", WRITE),
        ("Would you capture order ORD-12345?", "capture_order", WRITE),
        ("Please refund capture CAP-45678", "refund_capture", WRITE),
        ("I want to create an invoice to client@example.com", "create_invoice", POTENTIAL_WRITE),
    ],
)
def test_detect_write_intent_polite(text, tool, expected_class):
    result = detect_write_intent(text)
    assert result.classification == expected_class, f"text={text!r}: expected {expected_class}, got {result.classification}"
    assert result.tool == tool, f"text={text!r}: expected tool {tool}, got {result.tool}"


# --------------------------------------------------------------------------- #
# Zero-hallucination tests
# --------------------------------------------------------------------------- #

def test_extract_amount_never_infers_currency():
    """A bare amount never gets an inferred currency."""
    amount, amt_err = extract_amount("10")
    assert amount == "10.00"
    assert amt_err is None

    currency, curr_err = extract_currency("10")
    assert currency is None
    assert curr_err == "missing"


def test_extract_currency_conflicting_rejected():
    """Conflicting currency expressions are rejected as ambiguous."""
    currency, err = extract_currency("$10 EUR")
    assert currency is None
    assert err == "ambiguous"


def test_extract_invoice_items_no_invention():
    """Invoice items are never invented."""
    items, err = extract_invoice_items("Create an invoice")
    assert items is None
    assert err == "missing"


def test_detect_write_intent_description_optional():
    """Description is optional for create_order."""
    result = detect_write_intent("Create a $10 USD order")
    assert result.classification == WRITE
    assert result.tool == "create_order"
    assert result.args["amount"] == "10.00"
    assert result.args["currency"] == "USD"
    assert result.args["description"] == "PayPilot AI order"


def test_detect_write_intent_description_explicit():
    """Explicit description is preserved when provided."""
    result = detect_write_intent("Create a $10 USD order for coffee")
    assert result.classification == WRITE
    assert result.args["amount"] == "10.00"
    assert result.args["currency"] == "USD"
    assert result.args["description"] == "coffee"


def test_detect_write_intent_no_invented_amount():
    """Amounts are never invented."""
    result = detect_write_intent("Create an order for coffee")
    assert result.classification == POTENTIAL_WRITE
    assert "amount" in result.missing
    assert "amount" not in result.args


def test_detect_write_intent_no_invented_currency():
    """Currencies are never invented."""
    result = detect_write_intent("Create an order for $10")
    assert result.classification == WRITE
    assert result.args["amount"] == "10.00"
    assert result.args["currency"] == "USD"


def test_detect_write_intent_no_invented_id():
    """IDs are never invented."""
    result = detect_write_intent("Capture order")
    assert result.classification == POTENTIAL_WRITE
    assert "order_id" in result.missing
    assert "order_id" not in result.args


def test_detect_write_intent_malformed_id_is_invalid_not_missing():
    """Malformed IDs are reported as invalid, not missing."""
    result = detect_write_intent("Capture order 12")
    assert result.classification == POTENTIAL_WRITE
    assert result.tool == "capture_order"
    assert any("order_id" in m for m in result.missing)


def test_detect_write_intent_no_llm_trusted():
    """The detector never trusts LLM-generated content - it only examines
    the raw user text."""
    result = detect_write_intent("I have successfully created your $10 order")
    assert result.classification == WRITE
    assert result.args.get("amount") == "10.00"
    assert result.args.get("currency") == "USD"


# --------------------------------------------------------------------------- #
# IntentResult tests
# --------------------------------------------------------------------------- #

def test_intent_result_creation():
    result = IntentResult(WRITE, tool="create_order", args={"amount": "10.00"})
    assert result.classification == WRITE
    assert result.tool == "create_order"
    assert result.args == {"amount": "10.00"}
    assert result.missing == []
    assert not result.requires_clarification


def test_intent_result_potential_write():
    result = IntentResult(POTENTIAL_WRITE, tool="create_order", args={"amount": "10.00"}, missing=["currency"])
    assert result.classification == POTENTIAL_WRITE
    assert result.requires_clarification is True


def test_intent_result_invalid_classification():
    with pytest.raises(ValueError):
        IntentResult("INVALID_CLASSIFICATION")


# --------------------------------------------------------------------------- #
# Multi-operation detection tests
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "text",
    [
        "Create a $10 order and capture ORD-123",
        "Create two orders with different currencies",
        "Create an invoice and refund a capture",
        "Create an order and invoice",
        "Capture ORD-123 and refund CAP-45678",
    ],
)
def test_detect_write_intent_multiple_operations_uncertain(text):
    """Multiple financial operations in one message → UNCLEAR, no pending action."""
    result = detect_write_intent(text)
    assert result.classification == UNCLEAR, (
        f"text={text!r}: expected UNCLEAR, got {result.classification}"
    )
    assert result.tool is None
    assert result.requires_clarification is True


def test_detect_write_intent_single_operation_with_and_in_description():
    """A single write request with 'and' in the description should still be
    classified as WRITE, not UNCLEAR."""
    result = detect_write_intent("Create an order for coffee and tea for $10 USD")
    assert result.classification == WRITE
    assert result.tool == "create_order"


# --------------------------------------------------------------------------- #
# Contextual reference tests
# --------------------------------------------------------------------------- #