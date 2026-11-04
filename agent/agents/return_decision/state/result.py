"""Typed request and result contracts for return eligibility decisions."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field

from agent.core.state.shared import OrderLookupData, PolicyEvidence, StateModel


CheckStatus = Literal["pass", "fail", "unknown"]


class ReturnClaimFacts(StateModel):
    """Facts explicitly supplied by the customer or resolved from their order."""

    order_number: str | None = None
    order_item_id: UUID | None = None
    item_query: str | None = None
    requested_quantity: int | None = Field(default=None, ge=1)
    reason_code: str | None = None
    item_condition: str | None = None


class ReturnDecisionRequest(StateModel):
    request_id: UUID
    user_id: UUID
    message: str = Field(min_length=1)
    language: str = "vi"
    evaluated_at: datetime
    claim: ReturnClaimFacts = Field(default_factory=ReturnClaimFacts)
    orders: list[OrderLookupData] = Field(default_factory=list)
    policy_evidence: list[PolicyEvidence] = Field(default_factory=list)
    policy_lookup_status: Literal["succeeded", "not_found", "failed", "not_started"] = "not_started"


class DecisionCheck(StateModel):
    code: str
    status: CheckStatus
    reason_code: str
    message: str
    evidence_refs: list[str] = Field(default_factory=list)


class ReturnDecisionResult(StateModel):
    status: Literal["completed", "needs_clarification", "failed"]
    outcome: Literal["approve", "reject", "manual_review"] | None = None
    message: str
    checks: list[DecisionCheck] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    rule_set_version: str | None = None
    evaluated_at: datetime
    missing_fields: list[str] = Field(default_factory=list)
    error_code: str | None = None
