# PayPilot AI - Verification

This document reports what was actually run, what was mocked, and what remains
UNVERIFIED against the real PayPal sandbox. Nothing here is fabricated.

---

## 1. Unit tests (actually run, real output)

Command: `python -m pytest` from the project root.

Result: **340 passed, 0 failed**.

Every test runs offline with a temporary SQLite database and `respx` mocking of
HTTP (PayPal and LLM endpoints). No real credentials are required.

### Test coverage
- `tests/test_agent_persistence.py`: Agent persistence, pending-action creation,
  API response correctness, duplicate submission prevention (sequential +
  concurrent), distinct legitimate requests, missing amount/currency
  clarification, failed persistence error replies, LLM write-path restrictions,
  provenance blocking for unverified order/capture references, human approval
  enforcement, double approval prevention, idempotency header stability.
- `tests/test_approval_gate.py`: LLM write gating (zero PayPal calls), single
  execution under concurrency, reject never calls PayPal, invalid write args
  create no pending action, refund above captured amount blocked, capture of
  unapproved order blocked, declined capture marked failed, invoice flow.
- `tests/test_atomic_completion.py`: Atomic clarification completion,
  rollback, concurrency, session/expiry cases.
- `tests/test_api.py`: Health endpoint, static page, chat with mocked LLM,
  pending actions API, error handling.
- `tests/test_context_resolution.py`: Deterministic session-isolated context
  resolution, financial safety regressions.
- `tests/test_invariants.py`: Safety invariants (prompt injection, atomic
  single execution, stable idempotency key, ambiguous `unknown` without retry,
  UTC audit trail, error handling).
- `tests/test_intent.py`: Intent detection, parameter extraction,
  clarification handling.
- `tests/test_paypal_client.py`: OAuth2 token caching, error mapping,
  refund request/response normalization.
- `tests/test_tool_allowlist.py`: Read vs write tool separation.
- `tests/test_verify_sandbox_async.py`: Sandbox verification async coverage.
- `tests/test_transaction_provenance.py`: Provenance migration (existing
  records → `unknown`), idempotent migration, local vs verified provenance,
  validated PayPal response recording, failed/ambiguous operations cannot
  produce verified records, migration on copy of real database.

No PayPal financial write operations were performed during testing.

---

## 2. Actual local HTTP verification (run against the running server)

The FastAPI server was started (`uvicorn app.main:app`) and the real HTTP
workflow was exercised:

| Check | Endpoint | Result | Details |
|-------|----------|--------|---------|
| Health | `GET /api/health` | VERIFIED | Returns `paypal_configured`, `paypal_connected`, `llm_configured`, `llm_connected`, database path, demo_enabled |
| Chat proposal | `POST /api/chat` | VERIFIED | `create_order` request with dynamic amount/currency returns `pending_actions` with action ID, request ID, status=`pending` |
| Pending actions list | `GET /api/actions` | VERIFIED | Returns persisted pending actions with correct fields |
| Duplicate prevention | `POST /api/chat` (identical) | VERIFIED | Second identical request returns existing action; `is_duplicate: true`; reply truthfully states "already exists" |
| Distinct inputs | `POST /api/chat` (different amount) | VERIFIED | Different amount creates separate pending action |
| Database persistence | `paypilot.db` (read-only) | VERIFIED | Pending action row present with correct `tool`, `status`, `args`, `request_id` |

No PayPal financial write operations were performed. No Sandbox orders were
created.

---

## 3. Read-only PayPal Sandbox checks (UNVERIFIED)

OAuth2 authentication and token caching were verified against the real PayPal
Sandbox. Read-only order and capture retrieval could **not** be completed
because no valid existing Sandbox resource ID was available (the hardcoded test
ID `4KX64890GG2954126` returned `RESOURCE_NOT_FOUND`).

| Item | Method/endpoint | Status |
|------|----------------|--------|
| OAuth2 token | `POST /v1/oauth2/token` | VERIFIED |
| Token caching | 2nd call returns cached token | VERIFIED |
| Get order | `GET /v2/checkout/orders/{id}` | UNVERIFIED (no valid existing ID available) |
| Get capture | `GET /v2/payments/captures/{id}` | UNVERIFIED (no valid capture ID available) |
| Create order | `POST /v2/checkout/orders` | UNVERIFIED (financial write - not performed) |
| Capture order | `POST /v2/checkout/orders/{id}/capture` | UNVERIFIED (financial write - not performed) |
| Refund capture | `POST /v2/payments/captures/{id}/refund` | UNVERIFIED (financial write - not performed) |
| Invoice create/send | `POST /v2/invoicing/invoices` | UNVERIFIED (financial write - not performed) |

**Real sandbox confirmed:** OAuth2 and token caching only. No live PayPal
sandbox write calls were made. No live read calls beyond OAuth were completed.

To verify read-only order/capture retrieval against the real sandbox, obtain a
valid existing Sandbox order ID from the PayPal Developer Dashboard (Sandbox →
Accounts → a business account with processed orders) and run
`scripts/verify_sandbox.py` with your own credentials.

---

## 4. LLM health check (VERIFIED)

`GET /api/health` returns `llm_configured: true`, `llm_connected: true`,
`llm_model: ministral-3:8b`, `llm_base_url: http://localhost:11434/v1`.

PayPal health check returns `paypal_configured: true`, `paypal_connected: true`,
`paypal_base_url: https://api-m.sandbox.paypal.com`.

---

## 5. Browser visual verification

**NOT PERFORMED.** No browser automation was available. The frontend code
(`static/index.html`) was statically inspected and confirmed to consume the
actual API response schema (`send()` → `refreshAll()` → `loadActions()` →
`GET /api/actions` → `renderQueue()`), but actual browser interaction was not
tested.

---

## 6. Duplicate submission safety (VERIFIED)

| Scenario | Result |
|----------|--------|
| Sequential identical requests | Exactly 1 pending action created; 2nd request returns existing action with `is_duplicate: true` |
| Concurrent identical requests | Exactly 1 pending action created; SQLite UNIQUE partial index + transaction locking prevents duplicates |
| Distinct amounts/currencies | Separate pending actions created (legitimate differentiation preserved) |
| Persistence UNIQUE index | `idx_pending_dedup` on `(tool, args) WHERE status='pending'` provides database-level defense |

---

## 7. Files changed (this verification cycle)

Only `VERIFICATION.md` was modified during this documentation-only correction cycle.
No application code, tests, `.env`, database, or other project files were changed.

---

## 8. Remaining limitations

| Limitation | Detail |
|------------|--------|
| Browser visual verification | Not performed. Frontend statically inspected but not tested through actual browser interaction. |
| Live PayPal order/capture retrieval | UNVERIFIED. No valid existing Sandbox resource ID was available; read-only GET lookups could not be completed. |
| Live PayPal financial writes | NOT PERFORMED. No order creation, capture, refund, or invoice send was executed. |
| PayPal Sandbox credentials | Configured — OAuth2 authentication and token caching were verified against the real PayPal Sandbox. No valid existing order/capture resource ID was available for read-only GET lookups. |

---

## 9. What is NOT in this document

- No claims about verified PayPal financial operations (none were performed).
- No instructions to seed sandbox orders or perform unapproved financial operations.
- No credential values or `.env` contents.
- No outdated test counts from previous runs.
- No unverified operation labeled as VERIFIED.
