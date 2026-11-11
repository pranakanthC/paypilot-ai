"""
Phase 4: Actual PayPal Sandbox read-only integration tests (separated from mocks).

These tests are SKIPPED unless real PayPal sandbox credentials are configured
(PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET present and not the test fixtures).
They perform ONLY read-only PayPal API calls (token + GET lookups). No PayPal
write/capture/refund/invoice-send calls are made.

This file exists to clearly separate mocked deterministic tests from real
integration verification. Results from mocked tests must NOT be reported as
live integration success.
"""

from __future__ import annotations

import os

import pytest

from app.paypal_client import PayPalClient, PayPalError


def _real_sandbox_configured() -> bool:
    cid = os.getenv("PAYPAL_CLIENT_ID") or ""
    secret = os.getenv("PAYPAL_CLIENT_SECRET") or ""
    base = (os.getenv("PAYPAL_BASE_URL") or "https://api-m.sandbox.paypal.com").lower()
    # Exclude dummy/test fixture credentials
    if not cid or not secret:
        return False
    if cid in {"test-client-id", "dummy", "changeme"} or secret in {"test-client-secret", "dummy", "changeme"}:
        return False
    if "sandbox" not in base:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _real_sandbox_configured(),
    reason="Real PayPal Sandbox credentials not configured; read-only integration tests skipped (no live API verification performed).",
)


@pytest.mark.asyncio
async def test_sandbox_readonly_get_order_not_found_is_honest():
    """Read-only live lookup of a nonexistent order must return an honest not-found error (no write calls)."""
    client = PayPalClient()
    try:
        with pytest.raises(PayPalError):
            await client.get_order("ORD-NOT-EXIST-000000")
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_sandbox_readonly_get_capture_not_found_is_honest():
    """Read-only live lookup of a nonexistent capture must return an honest not-found error (no write calls)."""
    client = PayPalClient()
    try:
        with pytest.raises(PayPalError):
            await client.get_capture("CAP-NOT-EXIST-000000")
    finally:
        await client.aclose()
