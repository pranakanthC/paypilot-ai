"""Real PayPal Sandbox verification script.

Run this with your own sandbox credentials in `.env`:

    Copy-Item .env.example .env
    # edit .env: add PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET
    python scripts/verify_sandbox.py

It performs live calls against https://api-m.sandbox.paypal.com and prints the
result of each. No live-money credentials are ever used. Sandbox only.

Mark each result VERIFIED or UNVERIFIED in VERIFICATION.md after running.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import httpx  # noqa: E402  (import after path setup)

from app.config import get_settings  # noqa: E402


def show(label: str, ok: bool, detail: str = "") -> None:
    status = "VERIFIED" if ok else "FAILED"
    extra = f" - {detail}" if detail else ""
    print(f"[{status}] {label}{extra}")


def main() -> int:
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
    try:
        # 1. OAuth2 token
        total += 1
        try:
            token = client.get_access_token()
            show("OAuth2 token", bool(token), "token cached")
            results += 1 if token else 0
        except PayPalError as exc:
            show("OAuth2 token", False, f"{exc.name}: {exc.message}")

        # 2. Create order
        total += 1
        try:
            order = client.create_order(
                "10.00",
                "USD",
                "verify_sandbox.py order",
                request_id="verify-1",
            )
            oid = order.get("id")
            show("Create order", bool(oid), f"id={oid} status={order.get('status')}")
            results += 1 if oid else 0
        except PayPalError as exc:
            show("Create order", False, f"{exc.name}: {exc.message} (debug_id={exc.debug_id})")
            oid = None

        # 3. Get order
        if oid:
            total += 1
            try:
                got = client.get_order(oid)
                show(
                    "Get order",
                    bool(got.get("id")),
                    f"status={got.get('status')} amount={got.get('amount')} {got.get('currency')}",
                )
                results += 1 if got.get("id") else 0
            except PayPalError as exc:
                show("Get order", False, f"{exc.name}: {exc.message} (debug_id={exc.debug_id})")

        # 4. Idempotency: retry the same create_order with the same request_id
        total += 1
        try:
            again = client.create_order(
                "10.00",
                "USD",
                "verify_sandbox.py order",
                request_id="verify-1",
            )
            show(
                "Idempotency key reuse",
                again.get("id") == oid,
                f"retry id={again.get('id')} original id={oid}",
            )
            results += 1 if again.get("id") == oid else 0
        except PayPalError as exc:
            # PayPal may return the original order on duplicate request_id (idempotent),
            # or an error depending on timing/state. Report honestly.
            show(
                "Idempotency key reuse",
                False,
                f"{exc.name}: {exc.message} (debug_id={exc.debug_id})",
            )

        # 5. Capture order (requires buyer approval first in sandbox UI)
        total += 1
        if oid:
            print(
                "  NOTE: Capture requires you to approve order "
                f"{oid} in the PayPal sandbox buyer UI first."
            )
            try:
                captured = client.capture_order(oid, request_id=f"{oid}-capture")
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
                cap_id = None
        else:
            show("Capture order", False, "no order id (create failed)")
            cap_id = None

        # 6. Get capture
        total += 1
        if cap_id:
            try:
                cap = client.get_capture(cap_id)
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

        # 7. Refund capture (partial)
        total += 1
        if cap_id:
            try:
                refunded = client.refund_capture(
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

    finally:
        import asyncio

        asyncio.run(client.aclose())

    print(f"\n{results}/{total} live sandbox calls succeeded.")
    print("Mark each item in VERIFICATION.md section 3 as VERIFIED or UNVERIFIED.")
    return 0 if results == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
