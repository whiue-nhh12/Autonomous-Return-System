-- Additive PostgreSQL migration for persisted return submissions and decisions.
-- Apply after 20261112_refund_approval_outbox.sql through the deployment migration process.

ALTER TABLE return_requests
    ADD COLUMN submission_request_id UUID,
    ADD COLUMN submission_fingerprint VARCHAR(64);

ALTER TABLE return_requests
    ADD CONSTRAINT uq_return_requests_submission_request_id
        UNIQUE (submission_request_id);

ALTER TABLE return_status_history
    ADD COLUMN idempotency_key VARCHAR(128);

ALTER TABLE return_status_history
    ADD CONSTRAINT uq_return_status_history_idempotency_key
        UNIQUE (idempotency_key);

CREATE TABLE return_decision_proposals (
    id UUID PRIMARY KEY,
    return_request_id UUID NOT NULL
        REFERENCES return_requests(id) ON DELETE RESTRICT,
    evaluation_request_id UUID NOT NULL,
    status VARCHAR(32) NOT NULL,
    outcome VARCHAR(32),
    assessment_message TEXT NOT NULL,
    checks JSON NOT NULL DEFAULT '[]',
    reason_codes JSON NOT NULL DEFAULT '[]',
    missing_fields JSON NOT NULL DEFAULT '[]',
    evidence_refs JSON NOT NULL DEFAULT '[]',
    policy_versions JSON NOT NULL DEFAULT '{}',
    policy_version_ref VARCHAR(255),
    claim_facts JSON NOT NULL DEFAULT '{}',
    evaluated_order_facts JSON NOT NULL DEFAULT '[]',
    evaluated_data_fingerprint VARCHAR(64) NOT NULL,
    rule_set_version VARCHAR(255),
    classifier_version VARCHAR(255),
    decision_engine_version VARCHAR(64) NOT NULL DEFAULT 'return-decision-v1',
    evaluated_at TIMESTAMPTZ NOT NULL,
    decision_action VARCHAR(16),
    decision_note TEXT,
    decision_idempotency_key VARCHAR(128),
    decided_by UUID REFERENCES users(id) ON DELETE SET NULL,
    decided_at TIMESTAMPTZ,
    CONSTRAINT ck_return_decision_proposals_status
        CHECK (status IN ('completed', 'needs_clarification', 'failed')),
    CONSTRAINT ck_return_decision_proposals_outcome
        CHECK (outcome IS NULL OR outcome IN ('approve', 'reject', 'manual_review')),
    CONSTRAINT ck_return_decision_proposals_decision_action
        CHECK (decision_action IS NULL OR decision_action IN ('approve', 'reject')),
    CONSTRAINT uq_return_proposals_evaluation_request
        UNIQUE (evaluation_request_id),
    CONSTRAINT uq_return_proposals_idempotency_key
        UNIQUE (decision_idempotency_key)
);

CREATE INDEX ix_return_decision_proposals_return_request_id
    ON return_decision_proposals (return_request_id);

CREATE INDEX ix_return_proposals_request_evaluated
    ON return_decision_proposals (return_request_id, evaluated_at);
