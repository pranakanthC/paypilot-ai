"""Async PayPal REST client with cached OAuth2 tokens and normalized results.

Only the sandbox base URL is configured by default:
https://api-m.sandbox.paypal.com
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any, Optional

import httpx

from .config import Settings, get_settings

# PayPal error names that are useful to expose to the agent / UI.
_TOKEN_EXPIRY_MARGIN = 60  # seconds


class PayPalError(Exception):
    """A normalized PayPal API or transport error."""

    def __init__(
        self,
        status_code: int,
        name: Optional[str],
        message: Optional[str],
        details: Any = None,
        debug_id: Optional[str] = None,
    ) -> None:
        self.status_code = status_code
        self.name = name or "PAYPAL_ERROR"
        self.message = message or "PayPal returned an error."
        self.details = details
        self.debug_id = debug_id
        super().__init__(f"{self.name}: {self.message}")

    @property
    def ambiguous(self) -> bool:
        """True when PayPal may or may not have applied the write.

        Network failures and 5xx responses are ambiguous: the request can have
        reached PayPal. 4xx responses are definitive rejections.
        """
        return self.status_code == 0 or self.status_code >= 500

    def to_dict(self) -> dict:
        return {
            "type": "paypal",
            "status_code": self.status_code,
            "name": self.name,
            "message": self.message,
            "details": self.details,
            "debug_id": self.debug_id,
            "ambiguous": self.ambiguous,
        }


def _parse_pp_error(status_code: int, payload: Any) -> PayPalError:
    if not isinstance(payload, dict):
        return PayPalError(status_code, "PAYPAL_ERROR", str(payload)[:500])
    name = payload.get("name") or payload.get("error") or "PAYPAL_ERROR"
    message = (
        payload.get("message")
        or payload.get("error_description")
        or "PayPal returned an error."
    )
    details = payload.get("details")
    if details is None and payload.get("error"):
        details = payload.get("error_description")
    return PayPalError(
        status_code=status_code,
        name=name,
        message=message,
        details=details,
        debug_id=payload.get("debug_id"),
    )


def _value(amount: Optional[dict]) -> Optional[str]:
    if not isinstance(amount, dict):
        return None
    value = amount.get("value")
    return str(value) if value is not None else None


def _idempotency_headers(request_id: Optional[str]) -> dict:
    if not request_id:
        return {}
    return {"PayPal-Request-Id": request_id}


class PayPalClient:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._token_lock = asyncio.Lock()
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(30.0))

    async def aclose(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------ #
    # Authentication
    # ------------------------------------------------------------------ #
    async def get_access_token(self, force: bool = False) -> str:
        self.settings.require_paypal()
        async with self._token_lock:
            if (
                not force
                and self._token
                and time.monotonic() < self._token_expires_at - _TOKEN_EXPIRY_MARGIN
            ):
                return self._token

            credentials = (
                f"{self.settings.paypal_client_id}:{self.settings.paypal_client_secret}"
            )
            encoded = base64.b64encode(credentials.encode("utf-8")).decode("ascii")
            try:
                response = await self._http.post(
                    f"{self.settings.paypal_base_url}/v1/oauth2/token",
                    headers={
                        "Authorization": f"Basic {encoded}",
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Accept": "application/json",
                    },
                    data={"grant_type": "client_credentials"},
                )
            except httpx.HTTPError as exc:
                raise PayPalError(
                    0, "NETWORK_ERROR", f"Could not reach PayPal: {exc}"
                ) from exc

            if response.status_code != 200:
                try:
                    payload = response.json()
                except ValueError:
                    payload = response.text
                raise _parse_pp_error(response.status_code, payload)

            payload = response.json()
            token = payload.get("access_token")
            if not token:
                raise PayPalError(
                    500, "TOKEN_ERROR", "PayPal did not return an access token."
                )
            expires_in = int(payload.get("expires_in", 32400))
            self._token = token
            self._token_expires_at = time.monotonic() + expires_in
            return token

    # ------------------------------------------------------------------ #
    # Low-level request helper
    # ------------------------------------------------------------------ #
    async def _request(
        self,
        method: str,
        path: str,
        json_body: Optional[dict] = None,
        extra_headers: Optional[dict] = None,
    ) -> Any:
        token = await self.get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)

        url = f"{self.settings.paypal_base_url}{path}"
        response: Optional[httpx.Response] = None
        for attempt in range(2):
            try:
                response = await self._http.request(
                    method, url, headers=headers, json=json_body
                )
            except httpx.HTTPError as exc:
                raise PayPalError(
                    0, "NETWORK_ERROR", f"Could not reach PayPal: {exc}"
                ) from exc

            if response.status_code == 401 and attempt == 0:
                token = await self.get_access_token(force=True)
                headers["Authorization"] = f"Bearer {token}"
                continue
            break

        assert response is not None
        if response.status_code >= 400:
            try:
                payload = response.json()
            except ValueError:
                payload = response.text
            raise _parse_pp_error(response.status_code, payload)

        if response.status_code in (202, 204) or not response.content:
            return {}
        try:
            return response.json()
        except ValueError:
            return {}

    # ------------------------------------------------------------------ #
    # Orders v2
    # ------------------------------------------------------------------ #
    async def create_order(
        self,
        amount: str,
        currency: str,
        description: str,
        request_id: Optional[str] = None,
        reference: Optional[str] = None,
    ) -> dict:
        full_description = description
        if reference:
            full_description = f"{description} (Ref: {reference})"
        body = {
            "intent": "CAPTURE",
            "purchase_units": [
                {
                    "amount": {"currency_code": currency, "value": amount},
                    "description": full_description,
                }
            ],
            "payment_source": {
                "paypal": {
                    "experience_context": {
                        "brand_name": "PayPilot AI",
                        "shipping_preference": "NO_SHIPPING",
                        "user_action": "CONTINUE",
                        "return_url": "http://127.0.0.1:8000/api/paypal/return",
                        "cancel_url": "http://127.0.0.1:8000/api/paypal/cancel",
                    }
                }
            },
        }
        data = await self._request(
            "POST",
            "/v2/checkout/orders",
            body,
            _idempotency_headers(request_id),
        )
        return self._normalize_order(data)

    async def get_order(self, order_id: str) -> dict:
        data = await self._request("GET", f"/v2/checkout/orders/{order_id}")
        return self._normalize_order(data)

    async def capture_order(
        self,
        order_id: str,
        mock_application_code: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> dict:
        headers = _idempotency_headers(request_id)
        if mock_application_code:
            headers["PayPal-Mock-Response"] = json.dumps(
                {"mock_application_codes": mock_application_code}
            )
        data = await self._request(
            "POST", f"/v2/checkout/orders/{order_id}/capture", {}, headers
        )
        return self._normalize_order(data)

    # ------------------------------------------------------------------ #
    # Payments v2
    # ------------------------------------------------------------------ #
    async def get_capture(self, capture_id: str) -> dict:
        data = await self._request("GET", f"/v2/payments/captures/{capture_id}")
        return self._normalize_capture(data)

    async def refund_capture(
        self,
        capture_id: str,
        amount: Optional[str] = None,
        currency: Optional[str] = None,
        note: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> dict:
        if amount and not currency:
            capture = await self.get_capture(capture_id)
            currency = capture.get("currency")
        body: dict[str, Any] = {}
        if amount:
            body["amount"] = {"value": amount, "currency_code": currency}
        if note:
            body["note_to_payer"] = note
        data = await self._request(
            "POST",
            f"/v2/payments/captures/{capture_id}/refund",
            body,
            _idempotency_headers(request_id),
        )
        return self._normalize_refund(data)

    # ------------------------------------------------------------------ #
    # Invoicing v2
    # ------------------------------------------------------------------ #
    async def create_draft_invoice(
        self,
        recipient_email: str,
        items: list[dict],
        currency: str,
        note: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> dict:
        body = {
            "detail": {
                "currency_code": currency,
                "note": note or "Invoice created by PayPilot AI",
            },
            "primary_recipients": [
                {"billing_info": {"email_address": recipient_email}}
            ],
            "items": [
                {
                    "name": item["name"],
                    "quantity": str(item["quantity"]),
                    "unit_amount": {
                        "currency_code": currency,
                        "value": item["unit_amount"],
                    },
                }
                for item in items
            ],
        }
        data = await self._request(
            "POST",
            "/v2/invoicing/invoices",
            body,
            _idempotency_headers(request_id),
        )
        return self._normalize_invoice(data, recipient_email)

    async def send_invoice(
        self,
        invoice_id: str,
        note: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> dict:
        body: dict[str, Any] = {}
        if note:
            body["note"] = note
        await self._request(
            "POST",
            f"/v2/invoicing/invoices/{invoice_id}/send",
            body,
            _idempotency_headers(request_id),
        )
        return {"id": invoice_id, "sent": True}

    # ------------------------------------------------------------------ #
    # Normalizers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_order(data: dict) -> dict:
        purchase_units = []
        all_captures = []
        first_amount = None
        first_currency = None
        for unit in data.get("purchase_units") or []:
            amount = unit.get("amount") or {}
            captures = []
            for capture in (unit.get("payments") or {}).get("captures") or []:
                capture_refunds = []
                for refund in (capture.get("refunds") or []):
                    capture_refunds.append(
                        {
                            "id": refund.get("id"),
                            "status": refund.get("status"),
                            "amount": _value(refund.get("amount")),
                            "currency": (
                                (refund.get("amount") or {}).get("currency_code")
                            ),
                        }
                    )
                normalized_capture = {
                    "id": capture.get("id"),
                    "status": capture.get("status"),
                    "amount": _value(capture.get("amount")),
                    "currency": (capture.get("amount") or {}).get("currency_code"),
                    "final_capture": capture.get("final_capture"),
                    "seller_protection": capture.get("seller_protection"),
                    "seller_receivable_breakdown": capture.get(
                        "seller_receivable_breakdown"
                    ),
                    "refunds": capture_refunds,
                    "create_time": capture.get("create_time"),
                    "update_time": capture.get("update_time"),
                    "links": capture.get("links") or [],
                }
                captures.append(normalized_capture)
                all_captures.append(normalized_capture)
            unit_amount = _value(amount)
            unit_currency = amount.get("currency_code")
            if first_amount is None:
                first_amount = unit_amount
                first_currency = unit_currency
            purchase_units.append(
                {
                    "reference_id": unit.get("reference_id"),
                    "status": unit.get("status"),
                    "amount": unit_amount,
                    "currency": unit_currency,
                    "captures": captures,
                }
            )

        links = data.get("links") or []
        approval_url = None
        for link in links:
            if link.get("rel") in ("payer-action", "approve"):
                approval_url = link.get("href")
                break

        return {
            "id": data.get("id"),
            "intent": data.get("intent"),
            "status": data.get("status"),
            "create_time": data.get("create_time"),
            "update_time": data.get("update_time"),
            "amount": first_amount,
            "currency": first_currency,
            "approval_url": approval_url,
            "payer": data.get("payer"),
            "purchase_units": purchase_units,
            "captures": all_captures,
            "links": links,
        }

    @staticmethod
    def _normalize_capture(data: dict) -> dict:
        refunds = []
        for refund in (data.get("refunds") or []):
            refunds.append(
                {
                    "id": refund.get("id"),
                    "status": refund.get("status"),
                    "amount": _value(refund.get("amount")),
                    "currency": (refund.get("amount") or {}).get("currency_code"),
                }
            )
        return {
            "id": data.get("id"),
            "status": data.get("status"),
            "amount": _value(data.get("amount")),
            "currency": (data.get("amount") or {}).get("currency_code"),
            "final_capture": data.get("final_capture"),
            "seller_protection": data.get("seller_protection"),
            "seller_receivable_breakdown": data.get(
                "seller_receivable_breakdown"
            ),
            "refunds": refunds,
            "create_time": data.get("create_time"),
            "update_time": data.get("update_time"),
            "links": data.get("links") or [],
        }

    @staticmethod
    def _normalize_refund(data: dict) -> dict:
        return {
            "id": data.get("id"),
            "status": data.get("status"),
            "amount": _value(data.get("amount")),
            "currency": (data.get("amount") or {}).get("currency_code"),
            "note_to_payer": data.get("note_to_payer"),
            "create_time": data.get("create_time"),
            "update_time": data.get("update_time"),
            "links": data.get("links") or [],
        }

    @staticmethod
    def _normalize_invoice(data: dict, recipient_email: Optional[str] = None) -> dict:
        amount = data.get("amount") or {}
        links = data.get("links") or []
        invoice_link = None
        for link in links:
            if link.get("rel") in ("payer-view", "self"):
                invoice_link = link.get("href")
                break
        return {
            "id": data.get("id"),
            "status": data.get("status"),
            "invoice_number": data.get("invoice_number"),
            "recipient_email": recipient_email,
            "amount": _value(amount),
            "currency": amount.get("currency_code"),
            "create_time": data.get("create_time"),
            "detail": data.get("detail"),
            "link": invoice_link,
            "links": links,
        }

