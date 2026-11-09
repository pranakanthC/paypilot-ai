# PayPilot AI - Agentic Payment Operations

## Inspiration

Payment operations work is full of small investigations: why is this order still
pending, was this capture refunded, what does `INSTRUMENT_DECLINED` actually mean for
this transaction, and which amount is safe to refund? The answers live in API responses
that are verbose and easy to misread. LLMs are good at reading and explaining those
responses, but they should not be trusted to move money on their own.

PayPilot AI explores a middle ground: an agent that can freely read PayPal sandbox data,
explain what it sees, and prepare operational actions, while a human explicitly approves
every write. The hackathon build turns that idea into enforceable safety invariants and
a payment-operations command center.

## What it does

- Connects to the real PayPal **sandbox** REST API with OAuth2 client credentials and a
  cached access token.
- Answers questions about orders and captures by calling READ tools:
  - `get_order(order_id)` -> `GET /v2/checkout/orders/{order_id}`
  - `get_capture(capture_id)` -> `GET /v2/payments/captures/{capture_id}`
  - `list_local_transactions()` -> the app's SQLite registry
- Proposes WRITE actions without executing them:
  - `create_order(amount, currency, description)`
  - `capture_order(order_id, mock_application_code?)`
  - `refund_capture(capture_id, amount?, note)`
  - `create_invoice(recipient_email, items, currency)`
- Shows each tool call as a collapsible step in the chat, with a live activity feed and a
  payment lifecycle stepper.
- Stores every proposal as a pending action. The approval queue shows the exact tool,
  arguments, before/after state and stable `PayPal-Request-Id`.
- On approval, atomically claims the action, revalidates the arguments on the server,
  applies live business rules, executes the PayPal call, updates the registry and writes
  the audit log.
- Proves single execution: double-clicking Approve, retrying, or racing two requests
  produces one PayPal write and one transaction record.
- Validates refunds against live capture state, not cached local rows. `REFUNDED` is
  blocked; `PARTIALLY_REFUNDED` requires an explicit amount or an amount-less request so
  PayPal calculates the remaining balance.
- Treats ambiguous failures honestly: timeouts and 5xx responses become `unknown`, are
  never auto-retried, and can be resolved with a read-only reconcile action. Definitive
  4xx responses become `failed`.
- Supports sandbox failure testing by sending the `PayPal-Mock-Response` header on
  capture, including `INSTRUMENT_DECLINED`, and then explains the real PayPal error
  payload, including `name`, `message`, `details` and `debug_id`.
- Serves a dark command-center UI with a copilot chat, lifecycle stepper, approval queue,
  activity feed, safety counters, dark AG Grid transaction/audit tables, a demo modal and
  a "Seed sandbox order" flow that returns the buyer approval link.

## The safety invariants

These are the core of the build. Each has at least one test in
`tests/test_invariants.py`.

| ID | Invariant | How it is enforced |
| --- | --- | --- |
| **I1** | Prompt injection cannot bypass approval. | The agent loop has no write executor; hostile prompts can only create pending proposals. |
| **I2** | Double/concurrent approval executes once. | Atomic SQLite claim; the loser returns the recorded result. |
| **I3** | Writes are idempotent across retries. | `PayPal-Request-Id` generated at proposal time, stored, reused on every write. |
| **I4** | Refunds use live capture state. | Approval refetches the capture; no cached refund arithmetic. |
| **I5** | Ambiguous failures are not retried. | Timeout/5xx -> `unknown`; explicit human reconciliation; 4xx -> `failed`. |
| **I6** | Every state change is audited. | UTC time, actor, request ID, result and `debug_id`; no secrets. |
| **I7** | Failures are clean JSON. | LLM timeout and PayPal errors return structured JSON, never a crash. |

## How I built it

**Backend:** Python 3.11+, FastAPI, httpx, pydantic and SQLite from the standard library.

- `app/paypal_client.py` implements OAuth2 client credentials against
  `https://api-m.sandbox.paypal.com/v1/oauth2/token` with an in-memory cached token. It
  wraps Orders v2 (`POST/GET /v2/checkout/orders`, `POST /v2/checkout/orders/{id}/capture`),
  Payments v2 (`GET /v2/payments/captures/{id}`,
  `POST /v2/payments/captures/{id}/refund`) and Invoicing v2
  (`POST /v2/invoicing/invoices`, `POST /v2/invoicing/invoices/{id}/send`). Every write
  accepts a stable `PayPal-Request-Id`; invoice create/send use `{rid}-create` and
  `{rid}-send`. Responses are normalized and PayPal errors are mapped into a structured
  exception that distinguishes definitive failures from ambiguous ones. Captures accept
  an optional mock application code that becomes the sandbox `PayPal-Mock-Response`
  header.
- `app/tools.py` defines OpenAI-compatible function schemas plus pydantic models for
  every tool. Validation is independent of the model: positive amounts with at most two
  decimals, ISO 4217 currencies, PayPal-shaped IDs, valid email addresses, bounded
  invoice items, and no extra fields.
- `app/agent.py` runs the tool-calling loop. READ tools execute immediately and their
  results go back to the model. WRITE tools never execute in the loop: the arguments are
  validated, a `pending_actions` row is created with a fresh `request_id`, and the model
  receives a tool result saying the action is pending and was not executed. The system
  prompt pins the agent to tool results, forbids claiming a write was performed, and
  explicitly resists instructions embedded in API data. The loop is capped at six
  iterations.
- `app/services.py` is the only execution path for agent-proposed writes. Approval
  performs an atomic claim, revalidates arguments, fetches live data for business rules,
  calls PayPal with the stored request ID, records the transaction, marks the action
  approved and writes an audit row with actor `user-approved`. It records a baseline
  snapshot before execution so the UI can show before/after state. Ambiguous errors mark
  the action `unknown`; a separate read-only `reconcile_action` resolves the action
  without issuing a write.
- `app/db.py` uses SQLite tables `transactions`, `pending_actions` and `audit_log`, with
  migrations for `request_id` and `baseline`, atomic claim/release helpers, and audit
  helpers. WAL mode keeps concurrent reads responsive.
- `app/main.py` exposes the API: `POST /api/chat`, `GET /api/pending`,
  `GET /api/actions`, `POST /api/actions/{id}/approve`,
  `POST /api/actions/{id}/reject`, `POST /api/actions/{id}/reconcile`,
  `GET /api/transactions`, `GET /api/audit`, `POST /api/demo/seed`,
  `POST /api/demo/decline` and `GET /api/health`, with structured JSON errors rather
  than stack traces.

**AI:** The agent uses an OpenAI-compatible Chat Completions API. The default model is
`ministral-3:8b` served by local Ollama at `http://localhost:11434/v1` with the tool
schemas and `tool_choice` orchestration. The base URL, key and model are configurable
through `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL`, so any OpenAI-compatible provider
(such as DeepSeek) works as an alternative.

**Frontend:** One static HTML file with vanilla JavaScript and AG Grid Community from a
CDN. No build step. The refreshed UI is a dark navy "payment operations command center":
glass panels over a gradient mesh, Inter and JetBrains Mono, a left copilot chat, a center
payment lifecycle stepper, a right approval queue and activity feed, safety counters,
dark AG Grid tables, a demo modal, skeleton loaders, toasts, empty states, copyable IDs
and a sandbox badge. It is responsive at 1280 and 1920 widths. Failed or ambiguous
actions offer a one-click explanation or reconciliation follow-up that feeds the real
PayPal payload back into the chat.

**Testing:** pytest with pytest-asyncio and respx. HTTP is mocked only in tests. The
suite has 67 passing tests covering validation, the approval boundary, API behavior and
invariants I1-I7. Python and UI checks are reproducible; live PayPal and LLM calls are
marked `UNVERIFIED` in [VERIFICATION.md](VERIFICATION.md) because no credentials were
available in the build environment.

## Challenges

- **Making the approval gate structural, not a prompt instruction.** The system prompt
  says writes need approval, but that is not a security boundary. I separated read and
  write tool sets and kept the write execution function out of the agent loop entirely.
  For agent proposals, the only caller is the approve endpoint.
- **Making single execution true under concurrency.** A status check followed by an
  update is a race. I moved the check into an atomic SQLite claim and made the losing
  request return the recorded result instead of writing again.
- **Idempotency that survives retries.** The request ID is generated when the proposal
  is created, not when the button is clicked, so retries and duplicate approvals reuse
  the same key.
- **Refund validation against a live API with no refund list.** PayPal has no
  "list refunds for a capture" endpoint. The validator uses the live capture status and
  amount; for `PARTIALLY_REFUNDED` it asks PayPal to calculate the remaining balance
  instead of inventing one from local rows.
- **Ambiguous failures.** A timeout does not mean the write failed. The app marks the
  action `unknown`, never retries automatically, and gives a human a read-only
  reconciliation path. This is deliberately conservative.
- **Explaining instead of hallucinating.** The system prompt pins the agent to tool
  results and tells it to state what is missing. Error payloads are normalized so the
  model can cite the exact PayPal issue and debug ID.
- **A usable agent UI.** Chat alone hides what the agent did. Structured steps, an
  approval queue with before/after state, a lifecycle stepper and an audit grid make the
  reasoning and the money movement auditable.

## What I learned

- Tool schemas are part of the security surface. Constraining them with
  `additionalProperties: false` and validating again on the server matters more than
  trusting model output.
- Approval gates are a product feature, not just a checkbox. Showing the exact args,
  request ID and result builds trust and makes the audit log useful.
- Idempotency and single execution are different problems. One protects against network
  retries; the other protects against concurrent human clicks.
- Ambiguous failures are the hardest part of payments. "Unknown, do not retry, reconcile
  with a read" is a better default than optimistic retries.
- Mocking HTTP in tests is what makes safety claims verifiable offline.

## What's next

- PayPal webhook ingestion so the agent can react to payment events without polling.
- A durable job queue for approvals and reconciliation across restarts.
- Authentication, roles and maker-checker approval for multi-user operations.
- More read and write tools: authorizations, voids, dispute lookup, payout status.
- Persisted conversation state and an evaluation harness for explanation quality.
- Optional support for other payment providers behind the same approval interface.

## Tools used

- Python 3.11+, FastAPI, Uvicorn
- httpx, pydantic, python-dotenv
- SQLite (standard library)
- HTML, CSS, vanilla JavaScript, AG Grid Community (CDN)
- DeepSeek `deepseek-chat` via an OpenAI-compatible chat API (optional alternative)
- PayPal sandbox REST APIs: OAuth2, Orders v2, Payments v2, Invoicing v2
- pytest, pytest-asyncio, respx
- Docker (Dockerfile included; not built in this environment)
