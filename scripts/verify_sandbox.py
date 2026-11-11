"""Real PayPal Sandbox verification script.

Run this with your own sandbox credentials in `.env`:

    Copy-Item .env.example .env
    # edit .env: add PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET
    python scripts/verify_sandbox.py

It performs live calls against https://api-m.sandbox.paypal.com and prints the
result of each. No live-money credentials are ever used. Sandbox only.

Mark each result VERIFIED or UNVERIFIED in VERIFICATION.md after running.

The script retrieves an existing sandbox order (4KX64890GG2954126) and prints
its buyer approval URL so you can open it in a browser to approve the order.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import httpx  # noqa: E402  (import after path setup)

from app.config import get_settings  # noqa: E402


# Existing PayPal Sandbox order to verify/approve. No new order is created.
EXISTING_ORDER_ID = "4KX64890GG2954126"


def show(label: str, ok: bool, detail: str = "") -> None:
    status = "VERIFIED" if ok else "FAILED"
    extra = f" - {detail}" if detail else ""
    print(f"[{status}] {label}{extra}")


def extract_approval_url(order: dict) -> Optional[str]:
    """Return the buyer approval URL from a PayPal order's links, if present.

    Prefers rel="approve" and falls back to rel="payer-action". Returns None
    when no approval link is available (order may already be approved/completed).
    """
    links = order.get("links") or []
    for rel in ("approve", "payer-action"):
        for link in links:
            if link.get("rel") == rel:
                href = link.get("href")
                if href:
                    return href
    return None


async def main() -> int:
    settings = get_settings()
    if not settings.paypal_configured:
        print("ERROR: PAYPAL_CLIENT_ID / PAYPAL_CLIENT_SECRET are missing in .env")
        print("       Copy .env.example to .env and add your sandbox keys.")
        return 1

    base = settings.paypal_base_url.rstrip("/")
    results = 0
    total = 0

    # Use the app's PayPalClient for a consistent OAuth2 path.
    from app.paypal_client import PayPalClient, PayPalError  # noqa: E402

    client = PayPalClient(settings)
    oid = EXISTING_ORDER_ID
    approval_url: Optional[str] = None
    cap_id: Optional[str] = None

    try:
        # 1. OAuth2 token
        total += 1
        try:
            token = await client.get_access_token()
            # Never print the access token value itself.
            show("OAuth2 token", bool(token), "token cached")
            results += 1 if token else 0
        except PayPalError as exc:
            show("OAuth2 token", False, f"{exc.name}: {exc.message}")

        # 2. Retrieve the existing sandbox order by ID (no new order created).
        total += 1
        try:
            order = await client.get_order(oid)
            approval_url = extract_approval_url(order)
            show(
                "Get existing order",
                bool(order.get("id")),
                f"id={oid} status={order.get('status')}",
            )
            results += 1 if order.get("id") else 0
        except PayPalError as exc:
            show(
                "Get existing order",
                False,
                f"{exc.name}: {exc.message} (debug_id={exc.debug_id})",
            )

        # 3. Print the buyer approval URL clearly (no secrets exposed).
        total += 1
        if approval_url:
            print("\n=== Buyer Approval URL ===")
            print(approval_url)
            print("==========================\n")
            print(
                "Open the URL above in a browser to approve the order in the "
                "PayPal sandbox buyer UI."
            )
            show("Approval URL", True, "printed above")
            results += 1
        else:
            show("Approval URL", False, "no approve/payer-action link in order")

        # 4. Capture order (requires buyer approval first in sandbox UI)
        total += 1
        if oid and approval_url:
            print(
                "  NOTE: Capture requires you to approve order "
                f"{oid} via the URL above first."
            )
            try:
                captured = await client.capture_order(oid, request_id=f"{oid}-capture")
                cap_id = None
                for cu in (captured.get("purchase_units") or []):
                    for c in (cu.get("captures") or []):
                        cap_id = c.get("id")
                show(
                    "Capture order",
                    bool(cap_id),
                    f"capture_id={cap_id} status={captured.get('status')}",
                )
                results += 1 if cap_id else 0
            except PayPalError as exc:
                show(
                    "Capture order",
                    False,
                    f"{exc.name}: {exc.message} (debug_id={exc.debug_id})",
                )
        else:
            show("Capture order", False, "no order id or approval URL")

        # 5. Get capture
        total += 1
        if cap_id:
            try:
                cap = await client.get_capture(cap_id)
                show(
                    "Get capture",
                    bool(cap.get("id")),
                    f"status={cap.get('status')} amount={cap.get('amount')} {cap.get('currency')}",
                )
                results += 1 if cap.get("id") else 0
            except PayPalError as exc:
                show("Get capture", False, f"{exc.name}: {exc.message} (debug_id={exc.debug_id})")
        else:
            show("Get capture", False, "no capture id (capture failed/skipped)")

        # 6. Refund capture (partial)
        total += 1
        if cap_id:
            try:
                refunded = await client.refund_capture(
                    cap_id, amount="5.00", note="verify_sandbox.py partial refund",
                    request_id=f"{cap_id}-refund",
                )
                show(
                    "Refund capture",
                    bool(refunded.get("id")),
                    f"refund_id={refunded.get('id')} status={refunded.get('status')} amount={refunded.get('amount')}",
                )
                results += 1 if refunded.get("id") else 0
            except PayPalError as exc:
                show(
                    "Refund capture",
                    False,
                    f"{exc.name}: {exc.message} (debug_id={exc.debug_id})",
                )
        else:
            show("Refund capture", False, "no capture id")

    except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
        show("Sandbox verification", False, f"timeout: {type(exc).__name__}")
        return 1
    except PayPalError as exc:
        show("Sandbox verification", False, f"{exc.name}: {exc.message}")
        return 1
    finally:
        await client.aclose()

    print(f"\n{results}/{total} live sandbox calls succeeded.")
    print("Mark each item in VERIFICATION.md section 3 as VERIFIED or UNVERIFIED.")
    return 0 if results == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
