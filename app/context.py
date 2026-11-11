"""Deterministic, session-isolated context resolution over verified PayPal records.

Verified sources only:
1. Verified local transaction records (non-failed records created/recorded from
   actual executed PayPal operations).
2. Read-only live PayPal API lookup fallback for explicit valid identifiers.

Never treated as verified/completed: pending/unapproved proposals, failed
operations, unverified LLM-generated IDs, demo/sample records.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from . import db
from .paypal_client import PayPalClient, PayPalError
from .tools import _validate_id

ResourceKind = Literal["order", "capture"]

_FAILED_STATUSES = {"FAILED", "DENIED", "VOIDED", "UNKNOWN", "ERROR", "EXPIRED"}

_ID_STOPWORDS = frozenset({
    "that", "this", "it", "my", "the", "a", "an", "latest", "last", "previous",
    "prior", "earlier", "mentioned", "one", "same", "associated", "with", "for",
    "from", "of", "on", "to", "and", "or", "me", "payment", "order", "capture",
    "refund", "invoice", "status", "transaction", "transactions", "show", "check",
    "what", "happened", "get", "look", "lookup", "find", "see", "view",
})

_EXPLICIT_ORDER_PATTERNS = [
    re.compile(r"\border\s+(?:id\s+)?(?:is\s+|=|:)?\s*([A-Za-z0-9][A-Za-z0-9_-]{0,80})\b", re.I),
]
_EXPLICIT_CAPTURE_PATTERNS = [
    re.compile(r"\bcapture\s+(?:id\s+)?(?:is\s+|=|:)?\s*([A-Za-z0-9][A-Za-z0-9_-]{0,80})\b", re.I),
]

_LATEST_RE = re.compile(r"\b(latest|last|most recent|newest)\b", re.I)
_PREVIOUS_RE = re.compile(r"\b(previous|prior|before that|one before)\b", re.I)
_THAT_RE = re.compile(
    r"\b(that|this|mentioned earlier|earlier|the one|same order|same capture|same payment)\b|\bit\b",
    re.I,
)
_ASSOC_CAPTURE_FOR_ORDER_RE = re.compile(
    r"(?:capture|refund)\b.*\b(?:for|of|on|associated with)\s+(?:the\s+)?order\s+([A-Za-z0-9][A-Za-z0-9_-]{0,80})",
    re.I,
)
_ASSOC_ORDER_RE = re.compile(
    r"associated with\s+(?:the\s+)?order\s+([A-Za-z0-9][A-Za-z0-9_-]{0,80})", re.I
)


@dataclass
class ResolutionResult:
    status: str
    kind: Optional[ResourceKind] = None
    paypal_id: Optional[str] = None
    record: Optional[dict[str, Any]] = None
    source: Optional[str] = None  # local|live
    message: str = ""
    candidates: list[dict[str, Any]] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return self.status == "resolved" and self.paypal_id is not None


def _is_verified_status(status: Optional[str]) -> bool:
    if status is None:
        return True
    return str(status).upper() not in _FAILED_STATUSES


def _is_verified_local_row(row: dict[str, Any]) -> bool:
    if not row or not row.get("paypal_id"):
        return False
    return _is_verified_status(row.get("status"))


def _tx_sort_key(row: dict[str, Any]) -> tuple:
    return (str(row.get("created_at") or ""), int(row.get("id") or 0))


def list_verified(kind: Optional[ResourceKind] = None, limit: int = 500) -> list[dict[str, Any]]:
    rows = db.list_transactions(limit=limit)
    out = []
    for r in rows:
        if not _is_verified_local_row(r):
            continue
        if kind and r.get("kind") != kind:
            continue
        out.append(dict(r))
    out.sort(key=_tx_sort_key, reverse=True)
    return out


def find_verified_local(paypal_id: str, kind: Optional[ResourceKind] = None) -> Optional[dict[str, Any]]:
    for r in list_verified(kind=kind, limit=5000):
        if r.get("paypal_id") == paypal_id:
            return r
    if kind is not None:
        for r in list_verified(kind=None, limit=5000):
            if r.get("paypal_id") == paypal_id:
                return r
    return None


def _validate_id_fmt(value: str, label: str) -> tuple[Optional[str], Optional[str]]:
    try:
        return _validate_id(value, label), None
    except Exception:
        return None, "malformed"


def _extract_explicit_id(text: str) -> tuple[Optional[ResourceKind], Optional[str], Optional[str]]:
    t = text or ""
    # Try capture-specific markers first, but avoid contextual stopwords
    for pat in _EXPLICIT_CAPTURE_PATTERNS:
        m = pat.search(t)
        if m:
            raw = m.group(1).strip().strip(".,;:!?")
            if raw.lower() in _ID_STOPWORDS:
                # e.g. "capture that order" - contextual, not an explicit capture id
                break
            vid, err = _validate_id_fmt(raw, "capture_id")
            return "capture", (vid if err is None else raw), err
    for pat in _EXPLICIT_ORDER_PATTERNS:
        m = pat.search(t)
        if m:
            raw = m.group(1).strip().strip(".,;:!?")
            if raw.lower() in _ID_STOPWORDS:
                break
            vid, err = _validate_id_fmt(raw, "order_id")
            return "order", (vid if err is None else raw), err
    return None, None, "missing"


def is_contextual_reference(text: str) -> bool:
    t = text or ""
    return bool(
        _ASSOC_CAPTURE_FOR_ORDER_RE.search(t)
        or _ASSOC_ORDER_RE.search(t)
        or _LATEST_RE.search(t)
        or _PREVIOUS_RE.search(t)
        or _THAT_RE.search(t)
    )


def remember_verified_record(
    kind: ResourceKind,
    paypal_id: str,
    status: Optional[str],
    *,
    evidence: Optional[db.VerifiedResourceEvidence] = None,
    amount: Optional[str] = None,
    currency: Optional[str] = None,
    description: Optional[str] = None,
    parent_id: Optional[str] = None,
) -> None:
    """Persist a verified PayPal resource record only when validated evidence
    is supplied and matches the exact resource being persisted.

    Arbitrary caller-provided IDs, amounts, currencies, or statuses cannot
    promote records to paypal_verified without matching evidence.
    """
    if not paypal_id or not _is_verified_status(status):
        return
    if not evidence or not isinstance(evidence, db.VerifiedResourceEvidence):
        return
    if not evidence.matches(kind, paypal_id):
        return
    try:
        db.upsert_verified_transaction(
            kind=kind,
            paypal_id=paypal_id,
            status=status,
            amount=amount or evidence.amount,
            currency=currency or evidence.currency,
            description=description,
            parent_id=parent_id,
            provenance=db.PROVENANCE_PAYPAL_VERIFIED,
            verified_evidence=evidence,
        )
    except Exception:
        pass


def remember_session_reference(
    session_id: Optional[str], kind: ResourceKind, paypal_id: str, record: Optional[dict[str, Any]] = None
) -> None:
    if not session_id or not paypal_id:
        return
    try:
        rec = record or {"kind": kind, "paypal_id": paypal_id}
        db.upsert_session_context(session_id, kind, paypal_id, rec)
    except Exception:
        pass


def get_session_context(session_id: Optional[str]) -> list[dict[str, Any]]:
    if not session_id:
        return []
    try:
        return db.list_session_context(session_id)
    except Exception:
        return []


def _distinct_verified_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for r in records:
        pid = r.get("paypal_id")
        if not pid or pid in seen:
            continue
        if not _is_verified_local_row(r):
            continue
        seen.add(pid)
        out.append(r)
    return out


async def _live_lookup(
    client: PayPalClient, kind: ResourceKind, paypal_id: str
) -> ResolutionResult:
    try:
        if kind == "order":
            data = await client.get_order(paypal_id)
            if not isinstance(data, dict) or data.get("id") != paypal_id:
                return ResolutionResult(status="not_found", kind=kind, paypal_id=paypal_id, message=f"No verified order found for ID {paypal_id}.")
            rec = {"kind": "order", "paypal_id": data.get("id"), "status": data.get("status"), "amount": data.get("amount"), "currency": data.get("currency"), "raw": data}
            evidence = db.VerifiedResourceEvidence(
                kind="order",
                paypal_id=paypal_id,
                status=data.get("status"),
                amount=data.get("amount"),
                currency=data.get("currency"),
                raw_response=data,
            )
            remember_verified_record("order", paypal_id, data.get("status"), evidence=evidence, amount=data.get("amount"), currency=data.get("currency"))
            return ResolutionResult(status="resolved", kind="order", paypal_id=paypal_id, record=rec, source="live", message=f"Verified order {paypal_id} via live PayPal lookup.")
        else:
            data = await client.get_capture(paypal_id)
            if not isinstance(data, dict) or data.get("id") != paypal_id:
                return ResolutionResult(status="not_found", kind=kind, paypal_id=paypal_id, message=f"No verified capture found for ID {paypal_id}.")
            rec = {"kind": "capture", "paypal_id": data.get("id"), "status": data.get("status"), "amount": data.get("amount"), "currency": data.get("currency"), "raw": data}
            evidence = db.VerifiedResourceEvidence(
                kind="capture",
                paypal_id=paypal_id,
                status=data.get("status"),
                amount=data.get("amount"),
                currency=data.get("currency"),
                raw_response=data,
            )
            remember_verified_record("capture", paypal_id, data.get("status"), evidence=evidence, amount=data.get("amount"), currency=data.get("currency"))
            return ResolutionResult(status="resolved", kind="capture", paypal_id=paypal_id, record=rec, source="live", message=f"Verified capture {paypal_id} via live PayPal lookup.")
    except PayPalError as exc:
        name = str(exc.name or "").upper()
        if exc.status_code == 404 or "NOT_FOUND" in name or "RESOURCE_NOT_FOUND" in name:
            return ResolutionResult(status="not_found", kind=kind, paypal_id=paypal_id, message=f"No {kind} found for ID {paypal_id}.")
        return ResolutionResult(status="lookup_failed", kind=kind, paypal_id=paypal_id, message=f"Could not verify {kind} {paypal_id}: {exc.message}")
    except Exception as exc:
        return ResolutionResult(status="lookup_failed", kind=kind, paypal_id=paypal_id, message=f"Could not verify {kind} {paypal_id}: {exc}")


def _session_verified_records(session_id: Optional[str], kind: Optional[ResourceKind]) -> list[dict[str, Any]]:
    ctx = get_session_context(session_id)
    out = []
    for e in ctx:
        rec = e.get("record") or {}
        ekind = rec.get("kind") or e.get("kind")
        pid = rec.get("paypal_id") or e.get("paypal_id")
        if not pid or not ekind:
            continue
        if kind and ekind != kind:
            continue
        local = find_verified_local(pid, ekind)
        if local:
            out.append(local)
            continue
        # Session context is only populated from verified resolutions/reads.
        # Accept as a verified session reference unless explicitly failed.
        st = rec.get("status")
        if st is not None and str(st).upper() in _FAILED_STATUSES:
            continue
        merged = {"kind": ekind, "paypal_id": pid, **{k: v for k, v in rec.items() if k not in {"kind", "paypal_id"}}}
        out.append(merged)
    return _distinct_verified_candidates(out)


async def resolve_transaction_reference(
    text: str,
    *,
    session_id: Optional[str] = None,
    expected_kind: Optional[ResourceKind] = None,
    client: Optional[PayPalClient] = None,
    allow_live_lookup: bool = True,
    for_financial: bool = False,
) -> ResolutionResult:
    t = (text or "").strip()
    if not t:
        return ResolutionResult(status="missing", kind=expected_kind, message="No transaction reference provided.")

    assoc_m = _ASSOC_CAPTURE_FOR_ORDER_RE.search(t) or _ASSOC_ORDER_RE.search(t)
    wants_capture_assoc = bool(re.search(r"\b(capture|refund)\b", t, re.I)) and assoc_m is not None
    if wants_capture_assoc and assoc_m:
        raw_oid = assoc_m.group(1).strip().strip(".,;:!?")
        oid, err = _validate_id_fmt(raw_oid, "order_id")
        if err:
            return ResolutionResult(status="malformed", kind="order", message="The referenced order ID is malformed.")
        assert oid
        order_local = find_verified_local(oid, "order")
        if order_local is None and allow_live_lookup and client is not None:
            o_res = await _live_lookup(client, "order", oid)
            if not o_res.resolved:
                return ResolutionResult(status=o_res.status, kind="order", paypal_id=oid, message=o_res.message)
            order_local = o_res.record or {"kind": "order", "paypal_id": oid}
        elif order_local is None:
            return ResolutionResult(status="not_found", kind="order", paypal_id=oid, message=f"No verified order found for ID {oid}.")
        caps = _distinct_verified_candidates([c for c in list_verified("capture", limit=5000) if c.get("parent_id") == oid])
        if len(caps) == 0:
            raw = (order_local or {}).get("raw") or {}
            live_caps = raw.get("captures") or []
            if len(live_caps) == 1 and live_caps[0].get("id"):
                cid = str(live_caps[0]["id"])
                rec = {"kind": "capture", "paypal_id": cid, "status": live_caps[0].get("status"), "amount": live_caps[0].get("amount"), "currency": live_caps[0].get("currency"), "parent_id": oid}
                if session_id:
                    remember_session_reference(session_id, "capture", cid, rec)
                return ResolutionResult(status="resolved", kind="capture", paypal_id=cid, record=rec, source="live", message=f"Resolved capture {cid} associated with order {oid}.")
            return ResolutionResult(status="not_found", kind="capture", message=f"No verified capture associated with order {oid} was found.")
        if len(caps) > 1:
            return ResolutionResult(status="ambiguous", kind="capture", candidates=caps, message=f"Multiple verified captures are associated with order {oid}; please specify the capture ID.")
        cap = caps[0]
        if session_id:
            remember_session_reference(session_id, "capture", cap["paypal_id"], cap)
        return ResolutionResult(status="resolved", kind="capture", paypal_id=cap["paypal_id"], record=cap, source="local", message=f"Resolved verified capture for order {oid}.")

    ref_kind, ref_id, ref_issue = _extract_explicit_id(t)
    if ref_issue == "malformed":
        return ResolutionResult(status="malformed", kind=ref_kind or expected_kind, paypal_id=None, message=f"The provided {(ref_kind or 'transaction')} ID is malformed.")
    if ref_id and ref_issue is None:
        kind = ref_kind or expected_kind
        local_any = find_verified_local(ref_id, None)
        if local_any:
            actual_kind = local_any.get("kind")
            if kind and actual_kind != kind:
                return ResolutionResult(status="type_mismatch", kind=kind, paypal_id=ref_id, record=local_any, source="local", message=f"ID {ref_id} is a verified {actual_kind}, not a {kind}.")
            if session_id:
                remember_session_reference(session_id, actual_kind, ref_id, local_any)
            return ResolutionResult(status="resolved", kind=actual_kind, paypal_id=ref_id, record=local_any, source="local", message=f"Resolved verified {actual_kind} {ref_id}.")
        if allow_live_lookup and client is not None and kind in ("order", "capture"):
            res = await _live_lookup(client, kind, ref_id)
            if res.resolved and session_id and res.kind and res.paypal_id:
                remember_session_reference(session_id, res.kind, res.paypal_id, res.record)
            return res
        if kind is None:
            return ResolutionResult(status="ambiguous", kind=None, paypal_id=ref_id, message=f"ID {ref_id} was provided but its type cannot be determined from verified records. Please specify order or capture.")
        return ResolutionResult(status="not_found", kind=kind, paypal_id=ref_id, message=f"No verified {kind} found for ID {ref_id}.")

    if not is_contextual_reference(t):
        return ResolutionResult(status="missing", kind=expected_kind, message="No transaction reference found.")

    desired = expected_kind
    tl = t.lower()
    if desired is None:
        if re.search(r"\border\b", tl) and not re.search(r"\bcapture\b", tl):
            desired = "order"
        elif re.search(r"\bcapture\b", tl):
            desired = "capture"
        elif re.search(r"\brefund\b", tl):
            desired = "capture"
        elif re.search(r"\bpayment\b", tl):
            desired = "capture"

    is_latest = bool(_LATEST_RE.search(t))
    is_prev = bool(_PREVIOUS_RE.search(t))
    is_that = bool(_THAT_RE.search(t))

    # Contextual references
    if (is_that or ((is_latest or is_prev) and for_financial)) and not session_id:
        return ResolutionResult(
            status="missing",
            kind=desired,
            message=(
                "A session with a verified reference or an explicit verified ID is required "
                "to resolve this contextual reference."
            ),
        )

    if is_latest and not is_prev:
        pool: list[dict[str, Any]] = []
        sess_pool: list[dict[str, Any]] = []
        if session_id:
            sess_pool = _session_verified_records(session_id, desired)
            pool = sess_pool
        if not pool and not for_financial:
            # Non-financial read resolution may use deterministic verified local records
            # (never arbitrary). Financial operations must not fall back to global history.
            pool = _distinct_verified_candidates(list_verified(desired, limit=5000))
        if not pool:
            if for_financial:
                return ResolutionResult(
                    status="not_found",
                    kind=desired,
                    message=(
                        f"No unambiguous verified {desired or 'transaction'} reference found in this session. "
                        "Provide an explicit verified ID or a verified reference from the current session."
                    ),
                )
            if session_id:
                return ResolutionResult(status="not_found", kind=desired, message=f"No verified {desired or 'transaction'} records found in this session.")
            return ResolutionResult(status="not_found", kind=desired, message=f"No verified {desired or 'transaction'} records found.")
        # For financial operations, require unambiguous session reference (not global history).
        if for_financial and not sess_pool:
            return ResolutionResult(
                status="not_found",
                kind=desired,
                message=(
                    f"No unambiguous verified {desired or 'transaction'} reference found in this session. "
                    "Provide an explicit verified ID or a verified reference from the current session."
                ),
            )
        if len(pool) >= 2 and _tx_sort_key(pool[0]) == _tx_sort_key(pool[1]):
            return ResolutionResult(status="ambiguous", kind=desired, candidates=pool[:5], message="Multiple records are equally recent; please specify the ID.")
        chosen = pool[0]
        if session_id:
            remember_session_reference(session_id, chosen.get("kind", desired or "order"), chosen["paypal_id"], chosen)
        return ResolutionResult(status="resolved", kind=chosen.get("kind", desired), paypal_id=chosen["paypal_id"], record=chosen, source="local", message=f"Resolved latest verified {chosen.get('kind')} {chosen['paypal_id']}.")

    if is_prev:
        pool: list[dict[str, Any]] = []
        sess: list[dict[str, Any]] = []
        if session_id:
            sess = _session_verified_records(session_id, desired)
            pool = sess if len(sess) >= 2 else []
        if len(pool) < 2 and not for_financial:
            pool = _distinct_verified_candidates(list_verified(desired, limit=5000))
        if for_financial:
            # Require unambiguous verified reference(s) from current session.
            if len(sess) < 2:
                if len(sess) == 1:
                    return ResolutionResult(
                        status="ambiguous",
                        kind=desired,
                        candidates=sess,
                        message=(
                            "Only one verified transaction reference exists in this session; 'previous' is ambiguous. "
                            "Provide an explicit verified ID or a verified reference from the current session."
                        ),
                    )
                return ResolutionResult(
                    status="not_found",
                    kind=desired,
                    message=(
                        f"No unambiguous verified previous {desired or 'transaction'} reference found in this session. "
                        "Provide an explicit verified ID or a verified reference from the current session."
                    ),
                )
            pool = sess
        if len(pool) < 2:
            if len(pool) == 1:
                return ResolutionResult(status="ambiguous", kind=desired, candidates=pool, message="Only one verified transaction is available; 'previous' is ambiguous. Please specify the ID.")
            return ResolutionResult(status="not_found", kind=desired, message="No verified previous transaction found.")
        chosen = pool[1]
        if session_id:
            remember_session_reference(session_id, chosen.get("kind", desired or "order"), chosen["paypal_id"], chosen)
        return ResolutionResult(status="resolved", kind=chosen.get("kind", desired), paypal_id=chosen["paypal_id"], record=chosen, source="local", message=f"Resolved previous verified {chosen.get('kind')} {chosen['paypal_id']}.")

    if is_that:
        # "that/this/mentioned earlier" must resolve only from session-scoped
        # verified context. Never infer from unrelated/global records.
        sess = _session_verified_records(session_id, desired)
        if not sess:
            return ResolutionResult(status="not_found", kind=desired, message=f"No verified {desired or 'transaction'} was previously referenced in this session.")
        if len(sess) > 1:
            return ResolutionResult(status="ambiguous", kind=desired, candidates=sess[:10], message="Multiple transactions were referenced in this session; please specify which one you mean.")
        chosen = sess[0]
        remember_session_reference(session_id, chosen.get("kind", desired or "order"), chosen["paypal_id"], chosen)
        return ResolutionResult(status="resolved", kind=chosen.get("kind", desired), paypal_id=chosen["paypal_id"], record=chosen, source="local", message=f"Resolved referenced {chosen.get('kind')} {chosen['paypal_id']} from session context.")

    return ResolutionResult(status="ambiguous", kind=desired, message="The transaction reference is ambiguous. Please specify the transaction ID.")


async def resolve_for_financial_operation(
    text: str,
    *,
    session_id: Optional[str],
    operation: str,
    client: PayPalClient,
) -> ResolutionResult:
    if operation == "capture_order":
        res = await resolve_transaction_reference(text, session_id=session_id, expected_kind="order", client=client, allow_live_lookup=True, for_financial=True)
        if not res.resolved or not res.paypal_id:
            return res
        try:
            order = await client.get_order(res.paypal_id)
        except PayPalError as exc:
            return ResolutionResult(status="lookup_failed", kind="order", paypal_id=res.paypal_id, message=f"Could not revalidate order {res.paypal_id}: {exc.message}")
        except Exception as exc:
            return ResolutionResult(status="lookup_failed", kind="order", paypal_id=res.paypal_id, message=f"Could not revalidate order {res.paypal_id}: {exc}")
        if not isinstance(order, dict) or order.get("id") != res.paypal_id:
            return ResolutionResult(status="not_found", kind="order", paypal_id=res.paypal_id, message=f"Could not verify order {res.paypal_id}.")
        status = (order.get("status") or "").upper()
        evidence = db.VerifiedResourceEvidence(
            kind="order",
            paypal_id=order.get("id"),
            status=order.get("status"),
            amount=order.get("amount"),
            currency=order.get("currency"),
            raw_response=order,
        )
        rec = {"kind": "order", "paypal_id": order.get("id"), "status": order.get("status"), "amount": order.get("amount"), "currency": order.get("currency"), "raw": order}
        remember_verified_record("order", res.paypal_id, order.get("status"), evidence=evidence, amount=order.get("amount"), currency=order.get("currency"))
        if status != "APPROVED":
            return ResolutionResult(status="not_found", kind="order", paypal_id=res.paypal_id, record=rec, source="live", message=f"Order {res.paypal_id} is {status or 'UNKNOWN'} and not eligible for capture.")
        if session_id:
            remember_session_reference(session_id, "order", res.paypal_id, rec)
        return ResolutionResult(status="resolved", kind="order", paypal_id=res.paypal_id, record=rec, source="live", message=f"Verified APPROVED order {res.paypal_id}.")

    if operation == "refund_capture":
        res = await resolve_transaction_reference(text, session_id=session_id, expected_kind="capture", client=client, allow_live_lookup=True, for_financial=True)
        if not res.resolved or not res.paypal_id:
            return res
        try:
            capture = await client.get_capture(res.paypal_id)
        except PayPalError as exc:
            return ResolutionResult(status="lookup_failed", kind="capture", paypal_id=res.paypal_id, message=f"Could not revalidate capture {res.paypal_id}: {exc.message}")
        except Exception as exc:
            return ResolutionResult(status="lookup_failed", kind="capture", paypal_id=res.paypal_id, message=f"Could not revalidate capture {res.paypal_id}: {exc}")
        if not isinstance(capture, dict) or capture.get("id") != res.paypal_id:
            return ResolutionResult(status="not_found", kind="capture", paypal_id=res.paypal_id, message=f"Could not verify capture {res.paypal_id}.")
        status = (capture.get("status") or "").upper()
        evidence = db.VerifiedResourceEvidence(
            kind="capture",
            paypal_id=capture.get("id"),
            status=capture.get("status"),
            amount=capture.get("amount"),
            currency=capture.get("currency"),
            raw_response=capture,
        )
        rec = {"kind": "capture", "paypal_id": capture.get("id"), "status": capture.get("status"), "amount": capture.get("amount"), "currency": capture.get("currency"), "raw": capture}
        remember_verified_record("capture", res.paypal_id, capture.get("status"), evidence=evidence, amount=capture.get("amount"), currency=capture.get("currency"))
        if status not in {"COMPLETED", "PARTIALLY_REFUNDED"}:
            return ResolutionResult(status="not_found", kind="capture", paypal_id=res.paypal_id, record=rec, source="live", message=f"Capture {res.paypal_id} is {status or 'UNKNOWN'} and not refundable.")
        if session_id:
            remember_session_reference(session_id, "capture", res.paypal_id, rec)
        return ResolutionResult(status="resolved", kind="capture", paypal_id=res.paypal_id, record=rec, source="live", message=f"Verified refundable capture {res.paypal_id}.")

    return ResolutionResult(status="missing", message="Unsupported operation.")
