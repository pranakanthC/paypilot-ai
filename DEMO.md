# PayPilot AI - 2m45s Demo Script

Total runtime: **2 minutes 45 seconds**. The spoken script is written to fit the
timecodes; pause points are marked. Do not rush the two safety proofs.

## Before recording (10 minutes)

1. `.env` contains valid `PAYPAL_CLIENT_ID`, `PAYPAL_CLIENT_SECRET`, `LLM_API_KEY`,
    and `LLM_MODEL=ministral-3:8b` (Ollama at `http://localhost:11434/v1`).
2. Start clean so the audit grid is readable:
   - stop the server, delete `paypilot.db`, then run `uvicorn app.main:app --reload`.
3. Open <http://127.0.0.1:8000>. Confirm the header pills say **PayPal configured** and
   **LLM configured**, and the **Sandbox** badge is visible.
4. Have the PayPal **sandbox Personal (buyer)** account credentials ready.
5. Two tabs only: PayPilot and the PayPal sandbox login. Close notifications and other
   tabs. Set browser zoom to 110% and window size to 1440x900.
6. Warm up the LLM once with a harmless question, then reload the page. This avoids a
   cold-start pause on camera.
7. Do a full rehearsal once. The capture must be **successful** after the decline before
   the refund step will work.
8. Never show `.env`, keys, or the browser devtools.
9. Record at 1080p/30fps. Keep the cursor deliberate and the mouse still while speaking.

---

## 0:00 - 0:15 | The problem

**On screen:** PayPilot AI home page in the dark command-center layout.

**Say:**

> "Payment operations teams spend their day reading PayPal responses: why an order is
> stuck, what a decline means, and how much is still refundable. LLMs are great at that
> reading work, but you cannot let a model move money on its own. PayPilot AI is an agent
> that can read everything and write nothing without a human approval."

Point at the three safety counters: **0 bypassed writes**, **0 duplicate executions**,
**0 silent retries**.

---

## 0:15 - 0:35 | Seed and approve a real sandbox order

**Do:** Click **Seed sandbox order** in the demo bar.

**Say:**

> "This is a real Orders v2 call to the PayPal sandbox. PayPilot returns the buyer
> approval link and records the order in its local registry."

**Do:** Open the approval link in the PayPal tab, sign in with the sandbox **Personal**
account, confirm **49.99 USD**, and approve. Return to PayPilot.

**Say:**

> "The buyer approved it. The center lifecycle stepper has moved to Approved, and the
> order ID is on screen."

---

## 0:35 - 0:55 | Live investigation, grounded in API data

**Do:** Click the **Investigate latest order** chip.

**On screen:** A collapsible tool step appears, then the agent summarizes status, amount
and capture state from the live response.

**Say:**

> "The agent called `GET /v2/checkout/orders/{id}` through a READ tool. Every number and
> status in that answer comes from PayPal, not from the model's memory. Reads run
> automatically and are already in the audit trail."

Expand the tool step for two seconds so the normalized JSON is visible, then collapse it.

---

## 0:55 - 1:20 | The approval gate, then a real decline

**Do:** Type:

> `Capture order <ORDER_ID> with a simulated INSTRUMENT_DECLINED failure.`

**On screen:** The agent proposes `capture_order` with
`mock_application_code: INSTRUMENT_DECLINED`. The proposal appears in the right-hand
approval queue with before/after state. Nothing has been sent to PayPal.

**Say:**

> "The model asked to capture, but the agent loop only stored a proposal. The write tool
> has no execution path inside the agent. The proposal shows the exact arguments and the
> stable PayPal-Request-Id that will be reused if this action is ever retried."

**Do:** Press **Approve and execute**.

**On screen:** PayPal returns `UNPROCESSABLE_ENTITY` / `INSTRUMENT_DECLINED`. The result
card shows the normalized error and `debug_id`. The order remains Approved.

**Say:**

> "I approved, so the server revalidated the live order and called PayPal with the
> sandbox mock header. PayPal declined the instrument. The failure is audited, and the
> action is marked failed, not retried."

---

## 1:20 - 1:50 | Single execution proof + successful capture

**Do:** Type:

> `Capture order <ORDER_ID> again without the mock code.`

**On screen:** A second proposal appears. Point at the new `PayPal-Request-Id`.

**Say:**

> "The order is still Approved, so this is a legitimate retry. Notice the request ID is
> stable and belongs to this proposal, not to the button click."

**Do:** Press **Approve and execute** twice as fast as possible.

**On screen:** Exactly one capture succeeds. The lifecycle stepper advances to Captured
and one capture ID appears. The safety strip increments **executions**, not duplicates.

**Say:**

> "Two clicks, one execution. Approval is an atomic claim in SQLite: the second click
> loses the claim and returns the recorded result. It cannot capture twice."

Pause one second on the capture ID before continuing.

---

## 1:50 - 2:15 | Refund with live validation

**Do:** Type:

> `Refund capture <CAPTURE_ID> for 10.00 USD.`

**On screen:** The proposal appears with the amount and note. Press **Approve and
execute**.

**Say:**

> "Before writing, the server refetches the live capture. It does not trust the local
> refund rows. The capture is Completed, the amount is valid, so the refund goes through
> with the stored request ID."

**On screen:** The refund succeeds and the capture status becomes Partially Refunded.

**Say:**

> "If the capture were already Refunded, this would be blocked. If PayPal reports
> Partially Refunded without a remaining amount, the app asks PayPal to calculate the
> balance instead of inventing one."

---

## 2:15 - 2:35 | Audit trail and honest failure handling

**Do:** Open the **Audit log** view. Point at the sequence: read, proposal, decline,
proposal, capture, refund. Hover one row to show the UTC timestamp, actor,
`request_id`, result and `debug_id`.

**Say:**

> "Every state change is here: reads, proposals, approvals, the decline, the success and
> the refund, each with a request ID and a UTC timestamp. For ambiguous failures such as a
> timeout or a 5xx, the action becomes Unknown and is never auto-retried. A human can
> reconcile it with a live read. The safety counters stay at zero bypassed writes, zero
> duplicate executions and zero silent retries."

---

## 2:35 - 2:45 | Close

**On screen:** Approval queue plus the audit grid and the three safety counters.

**Say:**

> "PayPilot AI makes the agent excellent at reading and proposing, while the approval
> gate, atomic execution, idempotency and live validation keep the money safe. Agentic
> payment operations, with proof."

**End card:** PayPilot AI, GitHub URL, "PayPal sandbox + DeepSeek tool calling".

---

## Fallback notes

- **LLM slow:** keep talking over the loading state; the loop is capped at six steps.
- **Capture after decline fails:** some sandbox accounts require a fresh order after a
  mock decline. Seed a second order, approve it in the buyer tab, then continue from the
  capture step. Do not edit the script live.
- **Invoice demo instead of refund:** propose `create_invoice` to the sandbox buyer email
  for 2 x 75.00 USD. The same approval card and audit flow applies. Invoicing requires a
  sandbox business account with invoicing enabled.
- **Double-click does not visibly race:** the UI may disable the button after the first
  click. Click twice quickly and say "one execution"; the invariant test is the formal
  proof. Never claim a duplicate occurred if the UI prevented it.
- **Ambiguous failure:** if a timeout appears, leave it as Unknown and use the reconcile
  action. Do not retry the write. This is the correct behavior, and you can say so.
- **No credentials:** stop and do not record the live sections. The automated suite and
  `VERIFICATION.md` explicitly mark live sandbox checks `UNVERIFIED`.
- Never switch the app to live credentials during the demo.
