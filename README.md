# PayPilot AI - Agentic Payment Operations

PayPilot AI is an AI agent for PayPal **sandbox** payment operations. It looks up real
orders and captures through the PayPal REST API, explains failures and pending states
from live data, and proposes write actions (create order, capture, refund, invoice).
Every agent-proposed write is blocked behind an explicit **Approve / Reject** decision
in a dark payment-operations command center, and every executed action is written to an
audit log.

The hackathon build focuses on the safety properties that make an agentic payments tool
defensible: a structural human approval gate, atomic single execution, idempotent PayPal
writes, live-only refund validation, honest handling of ambiguous failures, a complete
UTC audit trail, and clean error paths. These are enforced in code and covered by
invariant tests (I1-I7), not only by the system prompt.

Built for the PayPal AI Hackathon on Devpost.

---

## What it does

- **Investigates payments** with READ tools that run automatically:
  - `get_order(order_id)` -> PayPal Orders v2
  - `get_capture(capture_id)` -> PayPal Payments v2
  - `list_local_transactions()` -> the app's SQLite registry
- **Explains** order status, capture status, pending/unapproved orders, and PayPal
  error payloads (name, message, details, `debug_id`) using only API results.
- **Proposes write actions** with WRITE tools:
  - `create_order(amount, currency, description)`
  - `capture_order(order_id, mock_application_code?)`
  - `refund_capture(capture_id, amount?, note)`
  - `create_invoice(recipient_email, items, currency)`
- **Never executes an agent write without approval.** The agent loop stores a
  `pending_actions` row and returns a proposal. The only way an agent-proposed write can
  reach PayPal is `POST /api/actions/{id}/approve`. The separate **Seed sandbox order**
  button is an explicit user action that creates one sandbox order directly.
- **Executes each approval exactly once.** Approval uses an atomic SQLite claim, so a
  double-click, retry, or two concurrent requests produce one PayPal write and one
  transaction record.
- **Reuses a stable `PayPal-Request-Id`.** The ID is generated when the proposal is
  created, stored with the pending action, and sent on every create/capture/refund and
  invoice write. Retrying the same action reuses the same key.
- **Validates refunds against live PayPal state.** The approve path refetches the capture
  instead of trusting local rows. `REFUNDED` is blocked; `COMPLETED` is refundable;
  `PARTIALLY_REFUNDED` requires an explicit amount or an amount-less request so PayPal
  calculates the remaining balance.
- **Handles ambiguous failures honestly.** Timeouts and 5xx responses mark the action
  `unknown` and are never auto-retried. A human can reconcile through live reads with
  `POST /api/actions/{id}/reconcile`; definitive 4xx responses are marked `failed`.
- **Supports sandbox failure testing** by sending the optional `PayPal-Mock-Response`
  header on capture, e.g. `INSTRUMENT_DECLINED`, so a declined capture can be
  demonstrated and then explained by the agent.
- **Shows its work** as collapsible tool steps in the chat and a live activity feed.
- **Audits everything** with a UTC timestamp, request ID and PayPal `debug_id` where
  available: agent reads, proposals, approvals, rejections, results and failures.

The app uses real PayPal sandbox APIs. There is no mock data path in the application;
HTTP mocking exists only inside the pytest suite.

---

## Architecture

```
+---------------------------+
| Browser                   |
| static/index.html         |
|  - chat + tool steps      |
|  - pending approvals      |
|  - AG Grid tables         |
+-------------+-------------+
              | JSON / HTTP
              v
+---------------------------+         +-----------------------------+
| FastAPI  app/main.py      |         | OpenAI-compatible LLM API   |
|                           |         | POST /chat/completions      |
| POST /api/chat -----------+-------->| model: deepseek-chat        |
|   agent.run_agent         |<--------| tool calling                |
|     |                     |         +-----------------------------+
|     | READ tools (auto)
|     v
|  app/paypal_client.py ----+--------> https://api-m.sandbox.paypal.com
|     ^                     |          - POST /v1/oauth2/token
|     |                     |          - /v2/checkout/orders...
|     | WRITE tools         |          - /v2/payments/captures...
|     | (proposal only)     |          - /v2/invoicing/invoices...
|     |                     |
|  app/tools.py             |
|     |                     |
|     v                     |
|  app/services.py <--------+-- POST /api/actions/{id}/approve
|     |  revalidate + execute + record + audit
|     v
|  SQLite  app/db.py
|    transactions | pending_actions | audit_log
+---------------------------+
```

The approval boundary is deliberately simple: **the only path that executes an
agent-proposed write is `app/services.py:approve_action`**, reached only from an HTTP
request that names a pending action ID. The agent loop imports no write execution
function. The user-initiated `POST /api/demo/seed` button also creates a sandbox order
directly, but it is not callable by the agent.

That path is also the single source of execution truth. It atomically claims the pending
row before touching PayPal, revalidates against live reads, executes with the stored
`PayPal-Request-Id`, records the result, and audits the outcome. Concurrent or repeated
approvals lose the claim and return the already-recorded result without writing again.
`app/services.py:reconcile_action` only performs live reads and updates an `unknown`
action; it never retries a write.

---

## PayPal APIs used

All calls go to `https://api-m.sandbox.paypal.com` by default.

| Purpose | Method and endpoint |
| --- | --- |
| OAuth2 client credentials (cached) | `POST /v1/oauth2/token` |
| Create order | `POST /v2/checkout/orders` |
| Get order | `GET /v2/checkout/orders/{order_id}` |
| Capture order | `POST /v2/checkout/orders/{order_id}/capture` |
| Get capture | `GET /v2/payments/captures/{capture_id}` |
| Refund capture | `POST /v2/payments/captures/{capture_id}/refund` |
| Create draft invoice | `POST /v2/invoicing/invoices` |
| Send invoice | `POST /v2/invoicing/invoices/{invoice_id}/send` |

The OAuth2 access token is cached in memory until shortly before expiry and reused
across requests.

The optional capture failure demo uses the sandbox-only header:

```
PayPal-Mock-Response: {"mock_application_codes":"INSTRUMENT_DECLINED"}
```

---

## AI model

- **Default model:** `ministral-3:8b` via local Ollama
- **Default base URL:** `http://localhost:11434/v1`
- **API key:** `ollama` (Ollama placeholder, accepted by the local server)
- **Chat endpoint used:** `POST http://localhost:11434/v1/chat/completions`
- **Protocol:** OpenAI-compatible Chat Completions with `tools` / tool calling
- **Loop cap:** 6 iterations per user message

You can point `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` at any OpenAI-compatible provider (for example DeepSeek at `https://api.deepseek.com`).

### Ollama installation and Ministral model setup

1. Install Ollama from https://ollama.com/ (Windows, macOS or Linux).
2. Start Ollama and pull the Ministral model:

```powershell
ollama pull ministral-3:8b
```

3. Verify both endpoints respond:

```powershell
curl http://localhost:11434/api/tags
curl http://localhost:11434/v1/models
```

4. Copy `.env.example` to `.env` (Ollama values are already set as defaults):

```
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=ministral-3:8b
```

Ollama exposes an OpenAI-compatible API at `/v1/chat/completions`, so PayPilot's agent loop can call it directly with tool schemas. Ministral supports tool calling; the first agent iteration uses `tool_choice: "required"` to force tool invocation, then `tool_choice: "auto"` for free-text responses.

Older README note: DeepSeek (`https://api.deepseek.com`) remains a valid alternative — set `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` accordingly.

---

## Setup

Requirements: Python 3.11+.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# edit .env and add your keys
```

On macOS/Linux the venv activation is `source .venv/bin/activate`, and the copy is
`cp .env.example .env`.

### Get PayPal sandbox credentials

1. Sign in at <https://developer.paypal.com/dashboard/>.
2. Open **Apps & Credentials** -> **Sandbox** -> **Create App** (merchant app).
3. Copy the **Client ID** and **Secret** into `.env` as `PAYPAL_CLIENT_ID` and
   `PAYPAL_CLIENT_SECRET`.
4. In **Sandbox** -> **Accounts**, use the **Personal** (buyer) account to approve
   orders. Keep those credentials handy for the demo.
5. Invoicing requires a sandbox business account with invoicing enabled. If your
   sandbox app was created under a business account that supports invoicing, the
   invoice tool works; otherwise PayPal returns a clear permission error.

### Get an LLM key

Create a DeepSeek API key at <https://platform.deepseek.com/> and put it in
`LLM_API_KEY`. If you use another provider, set `LLM_BASE_URL` (for example
`https://api.openai.com/v1`) and `LLM_MODEL`.

### Run

```powershell
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. The header shows whether PayPal and the LLM are
configured. API docs are at <http://127.0.0.1:8000/docs>.

### Docker

```powershell
docker build -t paypilot-ai .
docker run --env-file .env -p 8000:8000 paypilot-ai
```

---

## Configuration

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `PAYPAL_CLIENT_ID` | yes | - | Sandbox app client ID |
| `PAYPAL_CLIENT_SECRET` | yes | - | Sandbox app secret |
| `PAYPAL_BASE_URL` | no | `https://api-m.sandbox.paypal.com` | PayPal API base |
| `LLM_BASE_URL` | no | `https://api.deepseek.com` | OpenAI-compatible base |
| `LLM_API_KEY` | yes | - | LLM API key |
| `LLM_MODEL` | no | `deepseek-chat` | Model name |
| `DB_PATH` | no | `paypilot.db` | SQLite file path |

---

## HTTP API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Status plus whether PayPal/LLM are configured |
| `POST` | `/api/chat` | `{message, history?}` -> agent reply, steps, proposals |
| `GET` | `/api/pending` | Pending approval actions |
| `GET` | `/api/actions` | All pending actions with status, result and request ID |
| `POST` | `/api/actions/{id}/approve` | Atomically claim, revalidate, execute, audit |
| `POST` | `/api/actions/{id}/reject` | Reject atomically without touching PayPal |
| `POST` | `/api/actions/{id}/reconcile` | Read-only live reconciliation of an `unknown` action |
| `GET` | `/api/transactions` | Local registry of orders/captures/refunds/invoices |
| `GET` | `/api/audit` | Audit log, newest first |
| `POST` | `/api/demo/seed` | Create a sandbox order, return buyer approval link |
| `POST` | `/api/demo/decline` | Seed a declined-capture demo action for the UI |

Errors are always JSON shaped like:

```json
{"error": {"type": "paypal", "name": "UNPROCESSABLE_ENTITY", "message": "...", "details": [...], "debug_id": "..."}}
```

No stack traces are returned to the client.

---

## Safety invariants (I1-I7)

These are the properties the build is designed around. Each is enforced in code and has
at least one test in `tests/test_invariants.py`.

| ID | Invariant | Enforcement |
| --- | --- | --- |
| **I1** | Prompt injection cannot bypass the approval gate. | Agent loop has no write executor; a hostile prompt can at most create a pending proposal. Zero PayPal writes are issued by the loop. |
| **I2** | Concurrent or double approvals execute exactly once. | `claim_pending_action` performs an atomic `UPDATE ... WHERE status IN ('pending','approved')`; the loser returns the recorded result. |
| **I3** | Writes are idempotent across retries. | A `PayPal-Request-Id` is generated and stored at proposal time and reused on order, capture, refund and invoice writes. |
| **I4** | Refund validation uses live capture state. | Approval refetches the capture and never trusts cached refund rows. `REFUNDED` blocks; `PARTIALLY_REFUNDED` requires an explicit amount or amount-less PayPal calculation. |
| **I5** | Ambiguous failures are never auto-retried. | Timeout / status 0 / 5xx -> action `unknown`; definitive 4xx -> `failed`; only a human-triggered read-only reconcile resolves `unknown`. |
| **I6** | Every state change is auditable. | Audit rows carry UTC time, actor, action, `request_id`, result, and PayPal `debug_id` when available; no secrets are logged. |
| **I7** | Failures are clean JSON, not crashes. | LLM timeouts and PayPal errors return structured JSON; the server stays up. |

### Approval gate details

1. **Tool separation.** `app/tools.py` declares `READ_TOOLS` and `WRITE_TOOLS`.
   Only READ tools have an execution path inside `agent.run_agent`.
2. **Proposal, not execution.** When the LLM calls a WRITE tool, the agent validates
   the arguments with pydantic, writes a row to `pending_actions` with
   `status = 'pending'` and a fresh `request_id`, returns a tool result saying
   `executed: false`, and shows the proposal in the UI.
3. **Server-side validation independent of the model.**
   - positive amount, max 2 decimals, <= 1,000,000
   - 3-letter uppercase ISO 4217 currency, checked against the active code list
   - PayPal-shaped ID format (also prevents path injection in URLs)
   - valid recipient email, 1-50 invoice items, positive quantities
   - no unknown or extra fields (`extra="forbid"`)
4. **Business validation at approval time.** The approve endpoint refetches live data:
   - refunds are checked against the live capture status and amount
   - an order must be `APPROVED` before it can be captured
5. **Single execution endpoint for agent proposals.** `services.approve_action`
   atomically claims the action, revalidates, executes via `paypal_client` with the
   stored request ID, records the transaction, marks the action approved, and writes an
   `audit_log` row with actor `user-approved`. The only other write entry point is the
   explicit demo seed, which logs actor `user`.
6. **Reject is inert and atomic.** Rejection claims the row, updates it and the audit log,
   and never builds a PayPal request.
7. **Ambiguous failures stop.** Catchable client errors with no HTTP status, timeouts and
   5xx responses mark the action `unknown` and require reconciliation. Nothing is
   automatically retried.
8. **Reconciliation is read-only.** `reconcile_action` fetches the live order/capture and
   resolves the action to `executed` or `failed` without issuing a write.
9. **Loop cap.** The agent loop stops after 6 LLM iterations.

Tests assert that an LLM write tool call produces zero PayPal token/order requests, that
approval executes exactly once under concurrency, that request IDs are stable and reused,
and that ambiguous errors become `unknown` without a retry.

---

## Demo walkthrough

1. Click **Seed sandbox order** in the demo bar. A PayPal sandbox order is created and
   the buyer approval link appears in the demo modal.
2. Open the link, sign in with the sandbox **Personal** account, and approve.
3. Ask: *"Investigate order <ORDER_ID> using live PayPal data."* The agent calls
   `get_order`, shows the status and amount, and summarizes the capture state. The center
   lifecycle stepper advances to **Approved**.
4. Ask: *"Capture order <ORDER_ID> with a simulated INSTRUMENT_DECLINED failure."*
   The agent proposes `capture_order` with the mock code in the approval queue. Nothing
   has been sent to PayPal. Press **Approve and execute** twice quickly to show the
   single-execution claim; PayPal declines once and the card shows the normalized error.
   Press **Ask PayPilot to explain** to ground the explanation in the real payload.
5. Ask: *"Refund capture <CAPTURE_ID> for 10.00 USD."* Approve it and show the live
   capture validation passing, then the **Approvals**, **Transactions** and **Audit log**
   views. The safety strip stays at zero bypassed writes and one execution per action.

A timed 2m45s script is in [DEMO.md](DEMO.md).

---

## Tests

```powershell
pytest
```

Current state: **67 tests pass**. The suite covers:

- Every validation rule (amounts, currencies, IDs, emails, invoice items, extras).
- The approval gate: an LLM WRITE tool call never causes a PayPal token or write
  request; approval executes exactly once; rejection never calls PayPal.
- I1-I7 in `tests/test_invariants.py`: injection resistance, atomic single execution,
  stable and reused `PayPal-Request-Id`, live refund validation, ambiguous `unknown`
  handling without retry, audit fields, and clean error paths.
- Business rules: over-refunds and captures of unapproved orders are blocked.
- READ tools execute immediately and are audited.
- The agent loop cap.
- FastAPI endpoints, clean JSON errors for missing config and LLM failures, demo seed,
  approve/reject/reconcile, actions, transactions and audit.

HTTP is mocked with `respx` **only in tests**. See [VERIFICATION.md](VERIFICATION.md) for
the exact commands, real output, and which live sandbox checks remain `UNVERIFIED`.

---

## Project structure

```
.
|-- app/
|   |-- main.py           FastAPI app, endpoints, error handlers
|   |-- agent.py          LLM tool-calling loop + approval gate
|   |-- tools.py          OpenAI tool schemas, pydantic validation, executors
|   |-- services.py       Atomic approval, live revalidation, reconcile, registry
|   |-- paypal_client.py  OAuth2 cache, Orders/Payments/Invoicing, idempotency, errors
|   |-- db.py             SQLite tables, migrations, atomic claims, audit
|   `-- config.py         .env loading and settings
|-- static/index.html     Dark command-center UI (vanilla JS + AG Grid CDN)
|-- scripts/
|   `-- verify_sandbox.py Live PayPal sandbox verification script
|-- tests/
|   |-- test_invariants.py I1-I7 coverage
|   `-- ...               existing validation, API, client and approval tests
|-- .env.example
|-- requirements.txt
|-- Dockerfile
|-- README.md
|-- DEMO.md
|-- DEVPOST.md
|-- VERIFICATION.md
`-- LICENSE
```

---

## Limitations

- Designed for the **PayPal sandbox** only. Do not point it at live money.
- There is no authentication or multi-user authorization. Run it locally; do not expose
  it publicly as-is.
- Invoicing depends on the sandbox account's invoicing permissions.
- Refunds require a `COMPLETED` capture. PayPal has no "list refunds for a capture"
  endpoint, so validation uses the live capture status and amount; for
  `PARTIALLY_REFUNDED`, an amount-less refund asks PayPal to calculate the remaining
  balance. The app does not invent a remaining balance from local rows.
- Reconciliation for an `unknown` create-order action is limited to reading known order
  IDs; the app never retries the write automatically. A human decides what to do next.
- The app stores the local registry and audit log in SQLite; it is not a system of
  record for PayPal state, and READ tools always prefer live API data.
- Live sandbox and live LLM calls require your own credentials. The automated test suite
  mocks HTTP so it runs offline; the live sandbox checks are marked `UNVERIFIED` until
  you run `scripts/verify_sandbox.py` with real credentials.
- Docker build/run was not executed in this environment because Docker is unavailable.

---

## What was built during the hackathon

- The complete PayPal async client with cached OAuth2 tokens, normalized responses,
  error mapping, stable idempotency headers and the sandbox `PayPal-Mock-Response`
  capture header.
- The pydantic tool schemas and independent server-side validation.
- The LLM agent loop with visible tool steps and a hard proposal-only gate for writes,
  including prompt-injection-resistant behavior.
- Atomic single-execution approval, reject and read-only reconciliation.
- Live refund validation and conservative handling of ambiguous partial refunds.
- The SQLite registry, pending-action workflow, migrations and UTC audit log.
- The FastAPI endpoints, structured error handling and demo seed/decline flows.
- The dark command-center UI: copilot chat, payment lifecycle stepper, approval queue,
  activity feed, safety counters, dark AG Grid tables, demo modal and toasts.
- The pytest suite covering validation, the approval boundary and invariants I1-I7.
- `scripts/verify_sandbox.py`, `VERIFICATION.md`, setup, demo and Devpost files.

---

## License

MIT. See [LICENSE](LICENSE).
