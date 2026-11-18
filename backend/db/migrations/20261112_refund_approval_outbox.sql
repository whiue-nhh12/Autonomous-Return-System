-- Additive PostgreSQL migration for reviewer-approved refunds and outbox events.
-- Apply once through the deployment's migration process before running the API.

ALTER TABLE refunds
    ADD COLUMN idempotency_key VARCHAR(128);

ALTER TABLE refunds
    ADD CONSTRAINT uq_refunds_idempotency_key UNIQUE (idempotency_key);

CREATE TABLE outbox_events (
    id UUID PRIMARY KEY,
    event_type VARCHAR(128) NOT NULL,
    aggregate_type VARCHAR(64) NOT NULL,
    aggregate_id UUID NOT NULL,
    idempotency_key VARCHAR(160) NOT NULL,
    payload JSON NOT NULL,
    status VARCHAR(16) NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    available_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    claimed_at TIMESTAMPTZ,
    published_at TIMESTAMPTZ,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ck_outbox_events_status
        CHECK (status IN ('pending', 'processing', 'published', 'failed')),
    CONSTRAINT ck_outbox_events_attempts_nonnegative
        CHECK (attempts >= 0),
    CONSTRAINT uq_outbox_events_idempotency_key
        UNIQUE (idempotency_key)
);

CREATE INDEX ix_outbox_events_pending
    ON outbox_events (status, available_at);
