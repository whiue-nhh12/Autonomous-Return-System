"""HTTP contracts for reviewing and deciding stored return proposals."""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ReturnProposalDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["approve", "reject"]
    note: str = Field(min_length=1, max_length=1000)

    @field_validator("note")
    @classmethod
    def note_must_contain_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("note must contain non-whitespace text")
        return cleaned


class ReturnProposalSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    proposal_id: UUID
    return_request_id: UUID
    status: str
    outcome: str | None
    assessment_message: str
    checks: list[dict[str, Any]]
    reason_codes: list[str]
    missing_fields: list[str]
    policy_version_ref: str | None
    policy_versions: dict[str, int]
    evidence_refs: list[str]
    claim_facts: dict[str, Any]
    evaluated_order_facts: list[dict[str, Any]]
    evaluated_at: datetime
    rule_set_version: str | None
    classifier_version: str | None
    decision_engine_version: str
    decision_action: str | None
    decision_note: str | None
    decided_by: UUID | None
    decided_at: datetime | None


class ReturnProposalDecisionResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    proposal_id: UUID
    return_request_id: UUID
    action: str
    return_status: str
    replayed: bool = False
