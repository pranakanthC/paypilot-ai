"""Approval-gated execution of write actions and transaction recording."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from . import db
from .paypal_client import PayPalClient, PayPalError
from .tools import execute_tool, validate_tool_args


class NotFoundError(Exception):
    pass


class ConflictError(Exception):
    pass


class BusinessValidationError(Exception):
    def __init__(self, message: str, details: Any = None) -> None:
        self.message = message
        self.details = details
        super().__init__(message)

    def to_dict(self) -> dict:
        return {
            "type": "business_validation",
            "message": self.message,
            "details": self.details,
        }


def _money(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise BusinessValidationError(f"Invalid monetary value: {value!r}") from exc


async def validate_business_rules(
    tool: str, args: dict, client: PayPalClient
) -> Any:
    """Rules that need live PayPal data, applied again at approval time.

    Returns a live evidence snapshot (if any) so an ambiguous execution can be
    reconciled later against the state PayPal actually reports.
    """
    if tool == "refund_capture":
        capture = await client.get_capture(args["capture_id"])
        captured = _money(capture.get("amount") or "0")
        status = (capture.get("status") or "").upper()
        requested_arg = args.get("amount")
        requested = _money(requested_arg) if requested_arg is not None else captured

        if status == "REFUNDED":
            raise BusinessValidationError(
                f"Capture {args['capture_id']} is already fully refunded.",
                {"capture_status": status, "captured": str(captured)},
            )
        if status not in ("COMPLETED", "PARTIALLY_REFUNDED"):
            raise BusinessValidationError(
                f"Capture {args['capture_id']} is {status or 'UNKNOWN'} and cannot "
                "be refunded yet.",
                {"capture_status": status, "captured": str(captured)},
            )
        if requested <= 0:
            raise BusinessValidationError(
                f"Capture {args['capture_id']} has no refundable balance left.",
                {"capture_status": status, "captured": str(captured)},
            )

        # Compute the remaining refundable balance from live refund data reported
        # by PayPal. A capture may report refunds inline (refunds array) or
        # aggregate them in seller_receivable_breakdown.refunded_amount. Only
        # COMPLETED and PENDING refunds reduce the available balance.
        total_refunded = Decimal("0")
        has_refund_data = False

        refunds = capture.get("refunds") or []
        if refunds:
            has_refund_data = True
            for refund in refunds:
                refund_status = (refund.get("status") or "").upper()
                if refund_status in ("COMPLETED", "PENDING"):
                    try:
                        total_refunded += _money(refund.get("amount") or "0")
                    except (InvalidOperation, TypeError, ValueError):
                        pass

        if not has_refund_data:
            breakdown = capture.get("seller_receivable_breakdown") or {}
            refunded_amount = breakdown.get("refunded_amount") or {}
            refunded_value = refunded_amount.get("value")
            if refunded_value:
                has_refund_data = True
                try:
                    total_refunded = _money(refunded_value)
                except (InvalidOperation, TypeError, ValueError):
                    pass

        remaining = captured - total_refunded

        if has_refund_data:
            if requested > remaining:
                raise BusinessValidationError(
                    f"Refund amount {requested} exceeds the remaining refundable "
                    f"balance {remaining} for capture {args['capture_id']}.",
                    {
                        "capture_status": status,
                        "captured": str(captured),
                        "refunded": str(total_refunded),
                        "remaining": str(remaining),
                        "requested": str(requested),
                    },
                )
        else:
            # No live refund data available: fall back to the conservative,
            # status-based check. For PARTIALLY_REFUNDED, an explicit amount
            # cannot be verified safely without live refund data.
            if status == "PARTIALLY_REFUNDED" and requested_arg is not None:
                raise BusinessValidationError(
                    f"Capture {args['capture_id']} already has refunds. PayPal does not "
                    "expose a live refund list for a capture, so this partial amount "
                    "cannot be verified safely. Omit the amount to refund the remaining "
                    "balance, or confirm the amount in the PayPal dashboard.",
                    {
                        "capture_status": status,
                        "captured": str(captured),
                        "requested": str(requested),
                    },
                )
            if status == "COMPLETED" and requested > captured:
                raise BusinessValidationError(
                    f"Refund amount {requested} exceeds the captured amount {captured} "
                    f"for capture {args['capture_id']}.",
                    {
                        "capture_status": status,
                        "captured": str(captured),
                        "remaining": str(captured),
                        "requested": str(requested),
                    },
                )

        return {
            "capture_status": status,
            "captured": str(captured),
            "requested": str(requested) if requested_arg is not None else None,
            "refunded": str(total_refunded) if has_refund_data else None,
            "remaining": str(remaining) if has_refund_data else None,
        }

    if tool == "capture_order":
        order = await client.get_order(args["order_id"])
        status = (order.get("status") or "").upper()
        if status == "COMPLETED":
            raise BusinessValidationError(
                f"Order {args['order_id']} is already COMPLETED and has been "
                "captured. A capture cannot be executed twice.",
                {"order_id": args["order_id"], "status": status},
            )
        if status != "APPROVED":
            raise BusinessValidationError(
                f"Order {args['order_id']} is {status or 'UNKNOWN'} and cannot be "
                "captured. The buyer must approve it first (status APPROVED).",
                {"order_id": args["order_id"], "status": status},
            )
        return {"order_status": status}
    return None


def _validated_paypal_id(value: Any) -> Optional[str]:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    return None


def record_transactions(tool: str, args: dict, result: dict) -> None:
    """Update the local registry after a successfully executed action.

    Records are marked ``paypal_verified`` only when a successful, validated
    PayPal API response establishes the resource ID and relevant details. If a
    required PayPal resource ID cannot be established from the response, the
    record is stored as unverified/local provenance (never falsely verified).
    """
    if tool == "create_order":
        paypal_id = _validated_paypal_id((result or {}).get("id"))
        trusted = paypal_id is not None
        evidence = (
            db.VerifiedResourceEvidence(
                kind="order",
                paypal_id=paypal_id,
                status=(result or {}).get("status"),
                amount=args.get("amount") or (result or {}).get("amount"),
                currency=args.get("currency") or (result or {}).get("currency"),
                description=args.get("description"),
                raw_response=result,
            )
            if trusted
            else None
        )
        db.add_transaction(
            kind="order",
            paypal_id=paypal_id,
            status=(result or {}).get("status"),
            amount=args.get("amount") or (result or {}).get("amount"),
            currency=args.get("currency") or (result or {}).get("currency"),
            description=args.get("description"),
            provenance=db.PROVENANCE_PAYPAL_VERIFIED if trusted else db.PROVENANCE_UNKNOWN,
            verified_evidence=evidence,
        )
        return

    if tool == "capture_order":
        order_id = args["order_id"]
        captures = (result or {}).get("captures") or []
        for capture in captures:
            capture_id = _validated_paypal_id((capture or {}).get("id"))
            trusted = capture_id is not None
            evidence = (
                db.VerifiedResourceEvidence(
                    kind="capture",
                    paypal_id=capture_id,
                    status=(capture or {}).get("status"),
                    amount=(capture or {}).get("amount"),
                    currency=(capture or {}).get("currency"),
                    description=f"Capture of order {order_id}",
                    parent_id=order_id,
                    raw_response=capture,
                )
                if trusted
                else None
            )
            db.add_transaction(
                kind="capture",
                paypal_id=capture_id,
                parent_id=order_id,
                status=(capture or {}).get("status"),
                amount=(capture or {}).get("amount"),
                currency=(capture or {}).get("currency"),
                description=f"Capture of order {order_id}",
                provenance=db.PROVENANCE_PAYPAL_VERIFIED if trusted else db.PROVENANCE_UNKNOWN,
                verified_evidence=evidence,
            )
        if captures:
            db.update_transaction_status(order_id, (result or {}).get("status") or "COMPLETED")
        return

    if tool == "refund_capture":
        capture_id = args["capture_id"]
        refund_id = _validated_paypal_id((result or {}).get("id"))
        trusted = refund_id is not None
        evidence = (
            db.VerifiedResourceEvidence(
                kind="refund",
                paypal_id=refund_id,
                status=(result or {}).get("status"),
                amount=(result or {}).get("amount") or args.get("amount"),
                currency=(result or {}).get("currency"),
                description=args.get("note") or f"Refund of capture {capture_id}",
                parent_id=capture_id,
                raw_response=result,
            )
            if trusted
            else None
        )
        db.add_transaction(
            kind="refund",
            paypal_id=refund_id,
            parent_id=capture_id,
            status=(result or {}).get("status"),
            amount=(result or {}).get("amount") or args.get("amount"),
            currency=(result or {}).get("currency"),
            description=args.get("note") or f"Refund of capture {capture_id}",
            provenance=db.PROVENANCE_PAYPAL_VERIFIED if trusted else db.PROVENANCE_UNKNOWN,
            verified_evidence=evidence,
        )
        try:
            capture = db.list_transactions(limit=10000)
            captured_row = next(
                (
                    t
                    for t in capture
                    if t.get("kind") == "capture" and t.get("paypal_id") == capture_id
                ),
                None,
            )
            if captured_row and captured_row.get("amount"):
                refunded = sum(
                    Decimal(str(t["amount"]))
                    for t in capture
                    if t.get("kind") == "refund"
                    and t.get("parent_id") == capture_id
                    and t.get("amount")
                )
                if refunded >= Decimal(str(captured_row["amount"])):
                    db.update_transaction_status(capture_id, "REFUNDED")
                else:
                    db.update_transaction_status(capture_id, "PARTIALLY_REFUNDED")
        except InvalidOperation:
            pass
        return

    if tool == "create_invoice":
        invoice = (result or {}).get("invoice") or {}
        sent = (result or {}).get("sent") or {}
        invoice_id = _validated_paypal_id(invoice.get("id"))
        # Draft invoice ID comes from a successful PayPal create-draft response.
        trusted = invoice_id is not None
        total = invoice.get("amount")
        if not total:
            try:
                total = str(
                    sum(
                        Decimal(str(item["unit_amount"])) * int(item["quantity"])
                        for item in args.get("items", [])
                    )
                )
            except (InvalidOperation, KeyError, TypeError, ValueError):
                total = None
        evidence = (
            db.VerifiedResourceEvidence(
                kind="invoice",
                paypal_id=invoice_id,
                status=invoice.get("status"),
                amount=total,
                currency=invoice.get("currency") or args.get("currency"),
                description=f"Invoice to {args.get('recipient_email')}",
                raw_response=invoice,
            )
            if trusted
            else None
        )
        db.add_transaction(
            kind="invoice",
            paypal_id=invoice_id,
            status="SENT" if sent.get("sent") else invoice.get("status"),
            amount=total,
            currency=invoice.get("currency") or args.get("currency"),
            description=f"Invoice to {args.get('recipient_email')}",
            provenance=db.PROVENANCE_PAYPAL_VERIFIED if trusted else db.PROVENANCE_UNKNOWN,
            verified_evidence=evidence,
        )


async def approve_action(action_id: str, client: PayPalClient) -> dict:
    """Execute a pending write action after explicit user approval.

    The pending action is atomically claimed before any PayPal call. A double
    click, a retry or a concurrent request can only ever win the claim once.
    """
    action = db.get_pending_action(action_id)
    if action is None:
        raise NotFoundError(f"Pending action {action_id} was not found.")
    if action["status"] != "pending":
        raise ConflictError(
            f"Action {action_id} is already {action['status']} and cannot be executed."
        )
    if not db.claim_pending_action(action_id):
        current = db.get_pending_action(action_id)
        current_status = current["status"] if current else "unknown"
        raise ConflictError(
            f"Action {action_id} is already {current_status} and cannot be executed."
        )

    tool = action["tool"]
    request_id = action.get("request_id")
    try:
        validated = validate_tool_args(tool, action["args"])
        baseline = await validate_business_rules(tool, validated, client)
        if baseline is not None:
            db.set_action_baseline(action_id, baseline)
    except Exception as exc:
        error = (
            exc.to_dict()
            if hasattr(exc, "to_dict")
            else {"type": "validation", "message": str(exc)}
        )
        # Nothing was sent to PayPal, so the action may safely return to the queue.
        db.release_pending_action(action_id)
        db.log_audit(
            "user-approved",
            tool,
            action["args"],
            error=error,
            request_id=request_id,
        )
        raise

    try:
        result = await execute_tool(
            tool, validated, client, request_id=request_id
        )
    except PayPalError as exc:
        error = exc.to_dict()
        # A write that may have reached PayPal is never retried automatically.
        outcome = "unknown" if exc.ambiguous else "failed"
        error = {**error, "outcome": outcome}
        db.resolve_and_audit(
            action_id,
            outcome,
            "user-approved",
            tool,
            validated,
            error=error,
            request_id=request_id,
        )
        raise
    except Exception as exc:
        error = (
            exc.to_dict()
            if hasattr(exc, "to_dict")
            else {"type": "execution", "message": str(exc)}
        )
        db.resolve_and_audit(
            action_id,
            "failed",
            "user-approved",
            tool,
            validated,
            error=error,
            request_id=request_id,
        )
        raise

    # PayPal already applied the write. Local bookkeeping must not make the
    # action look unexecuted, so a recording error is audited and tolerated.
    try:
        record_transactions(tool, validated, result)
    except Exception as exc:  # defensive
        db.log_audit(
            "system",
            tool,
            validated,
            error={"type": "record_error", "message": str(exc)},
            request_id=request_id,
        )
    db.resolve_and_audit(
        action_id,
        "approved",
        "user-approved",
        tool,
        validated,
        result=result,
        request_id=request_id,
    )
    return {
        "action": db.get_pending_action(action_id),
        "result": result,
    }


async def reconcile_action(action_id: str, client: PayPalClient) -> dict:
    """Resolve an ``unknown`` write by reading the live state from PayPal.

    Only definitive evidence changes the status. If PayPal cannot prove the
    outcome, the action stays ``unknown`` and is reported as unresolved.
    """
    action = db.get_pending_action(action_id)
    if action is None:
        raise NotFoundError(f"Pending action {action_id} was not found.")
    if action["status"] != "unknown":
        raise ConflictError(
            f"Action {action_id} is {action['status']}, not unknown; "
            "there is nothing to reconcile."
        )

    tool = action["tool"]
    args = action["args"]
    request_id = action.get("request_id")
    baseline = action.get("baseline") or {}

    resolved_status: str | None = None
    evidence: dict[str, Any] = {}
    message = ""

    if tool == "capture_order":
        order = await client.get_order(args["order_id"])
        status = (order.get("status") or "").upper()
        evidence = {"order_status": status, "order": order}
        if status == "COMPLETED":
            resolved_status = "approved"
            message = "PayPal reports the order COMPLETED; the capture was applied."
            try:
                record_transactions("capture_order", args, order)
            except Exception as exc:  # defensive
                db.log_audit(
                    "reconciler",
                    tool,
                    args,
                    error={"type": "record_error", "message": str(exc)},
                    request_id=request_id,
                )
        elif status in ("APPROVED", "CREATED", "PAYER_ACTION_REQUIRED", "SAVED"):
            resolved_status = "failed"
            message = (
                f"PayPal reports the order {status}; the capture was not applied."
            )
        else:
            message = f"PayPal reports order status {status or 'UNKNOWN'}; still unresolved."

    elif tool == "refund_capture":
        capture = await client.get_capture(args["capture_id"])
        status = (capture.get("status") or "").upper()
        baseline_status = str(baseline.get("capture_status") or "").upper()
        evidence = {
            "capture_status": status,
            "baseline_status": baseline_status,
            "capture": capture,
        }
        if baseline_status == "COMPLETED" and status in (
            "REFUNDED",
            "PARTIALLY_REFUNDED",
        ):
            resolved_status = "approved"
            message = (
                f"PayPal reports the capture {status} after it was COMPLETED; "
                "the refund was applied."
            )
        elif baseline_status == "PARTIALLY_REFUNDED" and status == "REFUNDED":
            resolved_status = "approved"
            message = (
                "PayPal reports the capture fully REFUNDED after a partial refund; "
                "the refund was applied."
            )
        elif baseline_status == status:
            resolved_status = "failed"
            message = (
                f"PayPal still reports the capture {status or 'UNKNOWN'}; "
                "the refund was not applied."
            )
        else:
            message = (
                f"PayPal reports the capture {status or 'UNKNOWN'} (baseline "
                f"{baseline_status or 'UNKNOWN'}); the refund outcome cannot be "
                "attributed safely. Verify it in the PayPal dashboard."
            )
    else:
        raise BusinessValidationError(
            f"Automatic reconciliation is not supported for {tool}. PayPal does "
            "not expose a lookup by idempotency key; verify it in the dashboard.",
            {"tool": tool, "request_id": request_id},
        )

    if resolved_status is None:
        db.log_audit(
            "reconciler",
            tool,
            args,
            {"status": "unknown", "message": message, "evidence": evidence},
            request_id=request_id,
        )
        return {
            "action": db.get_pending_action(action_id),
            "resolved": False,
            "message": message,
            "evidence": evidence,
        }

    if resolved_status == "approved":
        db.resolve_and_audit(
            action_id,
            "approved",
            "reconciler",
            tool,
            args,
            result={"reconciled": True, "message": message, "evidence": evidence},
            request_id=request_id,
        )
    else:
        db.resolve_and_audit(
            action_id,
            "failed",
            "reconciler",
            tool,
            args,
            error={
                "type": "reconciled",
                "message": message,
                "evidence": evidence,
            },
            request_id=request_id,
        )
    db.log_audit(
        "reconciler",
        tool,
        args,
        {"status": resolved_status, "message": message, "evidence": evidence},
        request_id=request_id,
    )
    return {
        "action": db.get_pending_action(action_id),
        "resolved": True,
        "message": message,
        "evidence": evidence,
    }


def reject_action(action_id: str) -> dict:
    action = db.get_pending_action(action_id)
    if action is None:
        raise NotFoundError(f"Pending action {action_id} was not found.")
    if action["status"] != "pending":
        raise ConflictError(
            f"Action {action_id} is already {action['status']} and cannot be rejected."
        )
    if not db.claim_pending_action(action_id):
        current = db.get_pending_action(action_id)
        current_status = current["status"] if current else "unknown"
        raise ConflictError(
            f"Action {action_id} is already {current_status} and cannot be rejected."
        )
    db.resolve_and_audit(
        action_id,
        "rejected",
        "user",
        action["tool"],
        action["args"],
        result={"status": "rejected", "action_id": action_id},
        request_id=action.get("request_id"),
    )
    return {"action": db.get_pending_action(action_id)}
