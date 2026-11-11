"""LLM tool schemas, server-side validation and PayPal execution helpers.

READ tools run immediately. WRITE tools are validated here and are executed
only by the approval endpoint, never inside the agent loop.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from . import db
from .paypal_client import PayPalClient, PayPalError

READ_TOOLS = {"get_order", "get_capture", "list_local_transactions", "list_pending_actions"}
WRITE_TOOLS = {"create_order", "capture_order", "refund_capture", "create_invoice"}

_AMOUNT_RE = re.compile(r"^-?(?:0|[1-9]\d{0,8})(?:\.\d{1,2})?$")
_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{5,39}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Active ISO 4217 alphabetic codes (plus a few recently withdrawn ones so older
# sandbox data remains usable). PayPal support depends on the merchant country.
ISO_4217 = frozenset(
    """
    AED AFN ALL AMD ANG AOA ARS AUD AWG AZN BAM BBD BDT BGN BHD BIF BMD BND BOB
    BOV BRL BSD BTN BWP BYN BZD CAD CDF CHE CHF CHW CLF CLP CNY COP COU CRC CUC
    CUP CVE CZK DJF DKK DOP DZD EGP ERN ETB EUR FJD FKP GBP GEL GHS GIP GMD GNF
    GTQ GYD HKD HNL HTG HUF IDR ILS INR IQD IRR ISK JMD JOD JPY KES KGS KHR KMF
    KPW KRW KWD KYD KZT LAK LBP LKR LRD LSL LYD MAD MDL MGA MKD MMK MNT MOP MRU
    MUR MVR MWK MXN MXV MYR MZN NAD NGN NIO NOK NPR NZD OMR PAB PEN PGK PHP PKR
    PLN PYG QAR RON RSD RUB RWF SAR SBD SCR SDG SEK SGD SHP SLE SLL SOS SRD SSP
    STN SVC SYP SZL THB TJS TMT TND TOP TRY TTD TWD TZS UAH UGX USD USN UYI UYU
    UYW UZS VED VEF VES VND VUV WST XAF XAG XAU XBA XBB XBC XBD XCD XCG XDR XOF
    XPD XPF XPT XSU XTS XUA XXX YER ZAR ZMW ZWG ZWL
    """.split()
)

MockCode = Literal["INSTRUMENT_DECLINED", "PAYMENT_DENIED"]


class ToolValidationError(Exception):
    """Raised when the model supplies invalid tool arguments."""

    def __init__(self, tool: str, errors: Any) -> None:
        self.tool = tool
        self.errors = errors
        super().__init__(f"Invalid arguments for {tool}")

    def to_dict(self) -> dict:
        return {"type": "validation", "tool": self.tool, "errors": self.errors}


def _validate_money(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError("amount must be a number")
    if isinstance(value, (int, float, Decimal)):
        value = str(value)
    if not isinstance(value, str):
        raise ValueError("amount must be a string or number")
    text = value.strip()
    if not _AMOUNT_RE.match(text):
        raise ValueError(
            "amount must be a positive number with at most 2 decimal places"
        )
    try:
        decimal_value = Decimal(text)
    except InvalidOperation as exc:  # pragma: no cover - regex guards this
        raise ValueError("amount is not a valid decimal number") from exc
    if decimal_value <= 0:
        raise ValueError("amount must be greater than zero")
    if decimal_value > Decimal("1000000"):
        raise ValueError("amount must not exceed 1,000,000")
    return f"{decimal_value:.2f}"


def _validate_id(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    text = value.strip()
    if not _ID_RE.match(text):
        raise ValueError(
            f"{field_name} must be 6-40 characters of letters, numbers, '-' or '_'"
        )
    return text


def _validate_currency(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("currency must be a string")
    text = value.strip().upper()
    if not _CURRENCY_RE.match(text) or text not in ISO_4217:
        raise ValueError("currency must be a valid ISO 4217 code, e.g. USD")
    return text


# --------------------------------------------------------------------------- #
# Read tool argument models
# --------------------------------------------------------------------------- #
class GetOrderArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str

    @field_validator("order_id")
    @classmethod
    def check_order_id(cls, value: Any) -> str:
        return _validate_id(value, "order_id")


class GetCaptureArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    capture_id: str

    @field_validator("capture_id")
    @classmethod
    def check_capture_id(cls, value: Any) -> str:
        return _validate_id(value, "capture_id")


class ListLocalTransactionsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ListPendingActionsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Write tool argument models
# --------------------------------------------------------------------------- #
class CreateOrderArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    amount: str
    currency: str
    description: str = Field(min_length=1, max_length=127)
    reference: Optional[str] = None

    @field_validator("amount", mode="before")
    @classmethod
    def check_amount(cls, value: Any) -> str:
        return _validate_money(value)

    @field_validator("currency")
    @classmethod
    def check_currency(cls, value: Any) -> str:
        return _validate_currency(value)

    @field_validator("description")
    @classmethod
    def check_description(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("description must not be empty")
        return text


class CaptureOrderArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str
    mock_application_code: Optional[MockCode] = None

    @field_validator("order_id")
    @classmethod
    def check_order_id(cls, value: Any) -> str:
        return _validate_id(value, "order_id")


class RefundCaptureArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    capture_id: str
    amount: Optional[str] = None
    note: Optional[str] = Field(default=None, max_length=255)

    @field_validator("capture_id")
    @classmethod
    def check_capture_id(cls, value: Any) -> str:
        return _validate_id(value, "capture_id")

    @field_validator("amount", mode="before")
    @classmethod
    def check_amount(cls, value: Any) -> Optional[str]:
        if value is None or value == "":
            return None
        return _validate_money(value)

    @field_validator("note")
    @classmethod
    def check_note(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = value.strip()
        return text or None


class InvoiceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100)
    quantity: int = Field(ge=1, le=1000)
    unit_amount: str

    @field_validator("name")
    @classmethod
    def check_name(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("item name must not be empty")
        return text

    @field_validator("unit_amount", mode="before")
    @classmethod
    def check_unit_amount(cls, value: Any) -> str:
        return _validate_money(value)


class CreateInvoiceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recipient_email: str
    items: list[InvoiceItem] = Field(min_length=1, max_length=50)
    currency: str
    note: Optional[str] = Field(default=None, max_length=255)

    @field_validator("recipient_email")
    @classmethod
    def check_email(cls, value: str) -> str:
        text = value.strip()
        if not _EMAIL_RE.match(text):
            raise ValueError("recipient_email must be a valid email address")
        return text

    @field_validator("currency")
    @classmethod
    def check_currency(cls, value: Any) -> str:
        return _validate_currency(value)

    @field_validator("note")
    @classmethod
    def check_note(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = value.strip()
        return text or None


VALIDATORS: dict[str, type[BaseModel]] = {
    "get_order": GetOrderArgs,
    "get_capture": GetCaptureArgs,
    "list_local_transactions": ListLocalTransactionsArgs,
    "list_pending_actions": ListPendingActionsArgs,
    "create_order": CreateOrderArgs,
    "capture_order": CaptureOrderArgs,
    "refund_capture": RefundCaptureArgs,
    "create_invoice": CreateInvoiceArgs,
}


def validate_tool_args(tool: str, args: Any) -> dict:
    """Validate tool arguments server-side, independent of the LLM."""
    model = VALIDATORS.get(tool)
    if model is None:
        raise ToolValidationError(tool, f"Unknown tool: {tool}")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ToolValidationError(tool, "Arguments must be a JSON object")
    try:
        validated = model.model_validate(args)
    except ValidationError as exc:
        raise ToolValidationError(
            tool,
            [
                {
                    "loc": list(error.get("loc", [])),
                    "msg": error.get("msg"),
                    "type": error.get("type"),
                }
                for error in exc.errors()
            ],
        ) from exc
    return validated.model_dump(mode="json")


# --------------------------------------------------------------------------- #
# Tool schemas exposed to the LLM (OpenAI-compatible function calling)
# --------------------------------------------------------------------------- #
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": (
                "Read a PayPal Orders v2 order by ID. Use this to check the real "
                "status, amount and capture IDs of an order."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "PayPal order ID, e.g. 5O190127TN364715T",
                    }
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_capture",
            "description": (
                "Read a PayPal Payments v2 capture by ID. Use this to confirm "
                "whether a payment was captured and its captured amount."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "capture_id": {
                        "type": "string",
                        "description": "PayPal capture ID, e.g. 3C679366HH908993F",
                    }
                },
                "required": ["capture_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_local_transactions",
            "description": (
                "List local transaction records stored by this PayPilot instance "
                "(orders, captures, refunds, invoices). Each record includes "
                "`provenance`: `paypal_verified` (established by a successful, "
                "validated PayPal API response), `local` (locally recorded, not "
                "PayPal-verified), or `unknown` (exists locally but origin "
                "unverified). Use this when the user does not know an ID or asks "
                "about local records; distinguish local existence from PayPal "
                "verification."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_pending_actions",
            "description": (
                "List actions that have been proposed but not yet approved or "
                "rejected. Each entry shows the action type, amount, currency, "
                "description and status. Use this when the user asks about pending "
                "approvals, the approval queue, or actions awaiting review."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_order",
            "description": (
                "PROPOSAL ONLY. Propose creating a PayPal sandbox order. It is "
                "not executed until the user approves it in the UI."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "string",
                        "description": "Positive amount with up to 2 decimals, e.g. 49.99",
                    },
                    "currency": {
                        "type": "string",
                        "description": "3-letter ISO currency code, e.g. USD",
                    },
                    "description": {
                        "type": "string",
                        "description": "Short description of the order",
                    },
                },
                "required": ["amount", "currency", "description"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "capture_order",
            "description": (
                "PROPOSAL ONLY. Propose capturing an approved PayPal order. It is "
                "not executed until the user approves it in the UI."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "PayPal order ID to capture",
                    },
                    "mock_application_code": {
                        "type": "string",
                        "enum": ["INSTRUMENT_DECLINED", "PAYMENT_DENIED"],
                        "description": (
                            "Optional sandbox-only mock failure code used to demo "
                            "declined captures via the PayPal-Mock-Response header."
                        ),
                    },
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "refund_capture",
            "description": (
                "PROPOSAL ONLY. Propose refunding a captured payment. Omit amount "
                "for a full refund. It is not executed until the user approves it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "capture_id": {
                        "type": "string",
                        "description": "PayPal capture ID to refund",
                    },
                    "amount": {
                        "type": "string",
                        "description": "Optional refund amount, <= captured amount",
                    },
                    "note": {
                        "type": "string",
                        "description": "Optional note to the payer",
                    },
                },
                "required": ["capture_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_invoice",
            "description": (
                "PROPOSAL ONLY. Propose creating and sending a PayPal draft "
                "invoice. It is not executed until the user approves it in the UI."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient_email": {
                        "type": "string",
                        "description": "Recipient email address",
                    },
                    "currency": {
                        "type": "string",
                        "description": "3-letter ISO currency code, e.g. USD",
                    },
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "quantity": {"type": "integer", "minimum": 1},
                                "unit_amount": {"type": "string"},
                            },
                            "required": ["name", "quantity", "unit_amount"],
                            "additionalProperties": False,
                        },
                    },
                    "note": {
                        "type": "string",
                        "description": "Optional note shown on the invoice",
                    },
                },
                "required": ["recipient_email", "currency", "items"],
                "additionalProperties": False,
            },
        },
    },
]


async def execute_tool(
    tool: str,
    args: dict,
    client: PayPalClient,
    request_id: Optional[str] = None,
) -> dict:
    """Execute a validated tool against PayPal or the local registry.

    Every mutating call receives the stable ``PayPal-Request-Id`` stored on the
    pending action so a retried approval cannot create a duplicate resource.
    """
    if tool == "get_order":
        return await client.get_order(args["order_id"])
    if tool == "get_capture":
        return await client.get_capture(args["capture_id"])
    if tool == "list_local_transactions":
        return {"transactions": db.list_transactions(limit=50)}
    if tool == "list_pending_actions":
        return {"actions": db.list_pending_actions("pending")}
    if tool == "create_order":
        return await client.create_order(
            amount=args["amount"],
            currency=args["currency"],
            description=args["description"],
            request_id=request_id,
            reference=args.get("reference"),
        )
    if tool == "capture_order":
        return await client.capture_order(
            args["order_id"],
            mock_application_code=args.get("mock_application_code"),
            request_id=request_id,
        )
    if tool == "refund_capture":
        return await client.refund_capture(
            capture_id=args["capture_id"],
            amount=args.get("amount"),
            note=args.get("note"),
            request_id=request_id,
        )
    if tool == "create_invoice":
        draft = await client.create_draft_invoice(
            recipient_email=args["recipient_email"],
            items=args["items"],
            currency=args["currency"],
            note=args.get("note"),
            request_id=f"{request_id}-create" if request_id else None,
        )
        try:
            sent = await client.send_invoice(
                draft["id"],
                note=args.get("note"),
                request_id=f"{request_id}-send" if request_id else None,
            )
        except PayPalError as exc:
            # Draft was created but sending failed. Return the draft ID and the
            # send error honestly; do not claim the invoice was sent.
            return {
                "invoice": draft,
                "sent": {"sent": False, "error": exc.to_dict()},
                "partial": True,
            }
        return {"invoice": draft, "sent": sent}
    raise ToolValidationError(tool, f"Unknown tool: {tool}")


def summarize_step(tool: str, args: dict) -> str:
    if tool == "get_order":
        return f"Looked up order {args.get('order_id')}"
    if tool == "get_capture":
        return f"Looked up capture {args.get('capture_id')}"
    if tool == "list_local_transactions":
        return "Listed local transactions"
    if tool == "list_pending_actions":
        return "Listed pending approval actions"
    labels = {
        "create_order": "Create order",
        "capture_order": "Capture order",
        "refund_capture": "Refund capture",
        "create_invoice": "Create and send invoice",
    }
    return f"Proposed: {labels.get(tool, tool)} (awaiting approval)"

