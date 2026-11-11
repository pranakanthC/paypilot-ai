import pytest

from app.tools import ToolValidationError, validate_tool_args


def test_amounts_normalize_to_two_decimals():
    assert validate_tool_args(
        "create_order",
        {"amount": "25", "currency": "USD", "description": "Test"},
    )["amount"] == "25.00"
    assert validate_tool_args(
        "create_order",
        {"amount": 25.5, "currency": "usd", "description": "Test"},
    ) == {"amount": "25.50", "currency": "USD", "description": "Test", "reference": None}


@pytest.mark.parametrize(
    "amount",
    ["0", "0.00", "-1", "-0.01", "1.234", "abc", "", "1e3", "  ", "1000001"],
)
def test_invalid_amounts_are_rejected(amount):
    with pytest.raises(ToolValidationError):
        validate_tool_args(
            "create_order",
            {"amount": amount, "currency": "USD", "description": "Test"},
        )


@pytest.mark.parametrize(
    "currency", ["US", "USDD", "12A", "us", "", "U$D", "ABC", "ZZZ"]
)
def test_invalid_currencies_are_rejected(currency):
    with pytest.raises(ToolValidationError):
        validate_tool_args(
            "create_order",
            {"amount": "10.00", "currency": currency, "description": "Test"},
        )


@pytest.mark.parametrize(
    "order_id", ["abc", "short", "", "../evil", "a b c d e f", "O" * 41]
)
def test_invalid_order_ids_are_rejected(order_id):
    with pytest.raises(ToolValidationError):
        validate_tool_args("get_order", {"order_id": order_id})
    with pytest.raises(ToolValidationError):
        validate_tool_args(
            "capture_order", {"order_id": order_id}
        )


def test_currency_accepts_lowercase_iso_codes():
    args = validate_tool_args(
        "create_order",
        {"amount": "10.00", "currency": "jpy", "description": "Test"},
    )
    assert args["currency"] == "JPY"


def test_id_format_accepts_paypal_shaped_ids():
    assert validate_tool_args("get_order", {"order_id": "5O190127TN364715T"})[
        "order_id"
    ] == "5O190127TN364715T"
    assert validate_tool_args(
        "get_capture", {"capture_id": "3C679366HH908993F"}
    )["capture_id"] == "3C679366HH908993F"
    assert validate_tool_args(
        "get_order", {"order_id": "INV2-ABC_DEF-1234"}
    )["order_id"] == "INV2-ABC_DEF-1234"


def test_refund_args_are_optional_and_validated():
    args = validate_tool_args("refund_capture", {"capture_id": "3C679366HH908993F"})
    assert args == {"capture_id": "3C679366HH908993F", "amount": None, "note": None}

    args = validate_tool_args(
        "refund_capture",
        {"capture_id": "3C679366HH908993F", "amount": "5", "note": "  sorry  "},
    )
    assert args["amount"] == "5.00"
    assert args["note"] == "sorry"

    with pytest.raises(ToolValidationError):
        validate_tool_args(
            "refund_capture",
            {"capture_id": "3C679366HH908993F", "amount": "-5"},
        )


def test_capture_mock_code_is_restricted():
    args = validate_tool_args(
        "capture_order",
        {"order_id": "5O190127TN364715T", "mock_application_code": "INSTRUMENT_DECLINED"},
    )
    assert args["mock_application_code"] == "INSTRUMENT_DECLINED"

    with pytest.raises(ToolValidationError):
        validate_tool_args(
            "capture_order",
            {"order_id": "5O190127TN364715T", "mock_application_code": "SOMETHING_ELSE"},
        )


def test_invoice_validation():
    args = validate_tool_args(
        "create_invoice",
        {
            "recipient_email": "buyer@example.com",
            "currency": "eur",
            "items": [
                {"name": "Consulting", "quantity": 2, "unit_amount": "125.5"}
            ],
        },
    )
    assert args["currency"] == "EUR"
    assert args["items"][0]["unit_amount"] == "125.50"

    with pytest.raises(ToolValidationError):
        validate_tool_args(
            "create_invoice",
            {
                "recipient_email": "not-an-email",
                "currency": "USD",
                "items": [{"name": "X", "quantity": 1, "unit_amount": "1.00"}],
            },
        )

    with pytest.raises(ToolValidationError):
        validate_tool_args(
            "create_invoice",
            {"recipient_email": "buyer@example.com", "currency": "USD", "items": []},
        )

    with pytest.raises(ToolValidationError):
        validate_tool_args(
            "create_invoice",
            {
                "recipient_email": "buyer@example.com",
                "currency": "USD",
                "items": [{"name": "X", "quantity": 0, "unit_amount": "1.00"}],
            },
        )


def test_extra_fields_and_unknown_tools_are_rejected():
    with pytest.raises(ToolValidationError):
        validate_tool_args(
            "create_order",
            {
                "amount": "10.00",
                "currency": "USD",
                "description": "Test",
                "unexpected": True,
            },
        )
    with pytest.raises(ToolValidationError):
        validate_tool_args("delete_everything", {})


def test_read_tools_reject_unknown_fields():
    with pytest.raises(ToolValidationError):
        validate_tool_args(
            "get_order", {"order_id": "5O190127TN364715T", "extra": 1}
        )
