# Order history agent

`OrderAgentGraph` handles order-history text requests as a read-only child agent.
It receives the authenticated `user_id` and request text, searches only records
owned by that user, and returns normalized data. It never receives or mutates the
manager's `AgentSharedState`; orchestration merges the result.

## Database configuration

Configure `ORDER_READ_DATABASE_URL` with a dedicated PostgreSQL principal that
has `SELECT` privileges only for `orders`, `order_items`, `payments`,
`shipments`, `return_requests`, `return_items`, `return_status_history`, and
`refunds`. The query service also marks each PostgreSQL transaction read-only.
There is deliberately no fallback to the general `DATABASE_URL`.

The agent loads at most ten recent orders for history requests, one order for an
explicit order number or an explicit latest-order request, and at most two
orders when the request is ambiguous. It returns a clarification request when
a singular order question could refer to multiple orders. History results set
`has_more` when more records exist beyond the ten-order page.

The response omits payment provider transaction IDs, free-text return notes,
address snapshots, and refund failure details. Monetary values remain
`Decimal` values in the state contract and are serialized as decimal strings.
