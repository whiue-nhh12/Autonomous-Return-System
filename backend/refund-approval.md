# Return proposal review and refund initiation

Only an authenticated active admin can use these endpoints. A Decision Agent
result is an eligibility proposal; it does not approve the return or create a
refund.

## Inspect proposals

`GET /admin/returns/{return_request_id}/proposals` returns the saved assessment
history, including checks, reasons, missing fields, policy citations, and the
final admin decision if one exists.

## Decide a proposal

`POST /admin/return-proposals/{proposal_id}/decision` requires an
`Idempotency-Key` header (16–128 characters) and a body such as:

```json
{
  "action": "approve",
  "note": "Approved after reviewing the return evidence"
}
```

Only completed proposals can be decided. A proposal with outcome `approve` can
be approved, one with outcome `reject` can be rejected, and `manual_review`
requires an explicit admin choice. Failed or clarification proposals cannot be
decided. The service locks the proposal, request, order, and order items; checks
that the order facts and published policy versions still match the evaluation;
then records the final decision and return status history in one transaction.
Changed order or policy data requires a new evaluation.

`POST /admin/returns/{return_request_id}/reevaluate` starts that new evaluation
for the existing submitted request, saves another proposal, and accepts an
`Idempotency-Key`. It does not create a duplicate return request.

This endpoint does not create a refund.

## Record returned items as received

`POST /admin/returns/{return_request_id}/receive` advances an approved request
to `item_received` after staff inspection. It requires an `Idempotency-Key`
header and a body such as `{"note": "Items received and inspected"}`. This
transition is recorded in `return_status_history` and is the gate for refund
initiation.

## Initiate a refund

`POST /admin/returns/{return_request_id}/initiate-refund` separately creates a
`pending` refund and `refund.requested` outbox event. It requires an
`Idempotency-Key` header and a body such as:

```json
{
  "amount": "125000.00",
  "payment_id": "optional-payment-uuid",
  "note": "Item received and inspected"
}
```

The current policy requires the return status to be `item_received`. There is
no configured immediate-refund exception. The service validates the active
admin, succeeded payment, currency, and remaining refundable amount. If the
order has zero or multiple succeeded payments, `payment_id` must identify one.
The request does not call a payment provider; a worker must consume the outbox
event and update the refund status.

Apply `backend/db/migrations/20261112_refund_approval_outbox.sql` and then
`backend/db/migrations/20261117_return_decision_proposals.sql` to an existing
PostgreSQL database before deploying these endpoints.
