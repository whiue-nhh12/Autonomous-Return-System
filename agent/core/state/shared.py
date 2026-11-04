"""Typed shared state for coordinating one request across agents."""

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StateModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ReturnIntent(str, Enum):
    UNCLEAR = "unclear"
    POLICY_QUESTION = "policy_question"
    ORDER_STATUS = "order_status"
    RETURN_ELIGIBILITY = "return_eligibility"
    SUBMIT_RETURN = "submit_return"


StateStatus = Literal["not_started", "pending", "succeeded", "not_found", "failed", "skipped"]
ChildAgentStatus = Literal[
    "succeeded", "not_found", "needs_clarification", "failed", "not_configured", "skipped"
]


class RequestContext(StateModel):
    request_id: UUID = Field(default_factory=uuid4)
    conversation_id: str | None = None
    user_id: UUID | None = None
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RequestInput(StateModel):
    message: str = Field(min_length=1)
    language: str = "vi"

    @field_validator("message")
    @classmethod
    def strip_message(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("message cannot contain only whitespace")
        return normalized


class OrderItemLookup(StateModel):
    order_item_id: UUID
    product_id: UUID
    product_name: str
    sku: str
    purchased_quantity: int = Field(gt=0)
    quantity_already_returned: int = Field(default=0, ge=0)
    unit_price: Decimal = Field(default=Decimal("0"), ge=0)
    line_total: Decimal = Field(default=Decimal("0"), ge=0)


class OrderPaymentLookup(StateModel):
    status: str
    method: str
    amount: Decimal = Field(gt=0)
    currency: str
    paid_at: datetime | None = None


class OrderShipmentLookup(StateModel):
    direction: str
    provider: str
    tracking_number: str | None = None
    tracking_url: str | None = None
    status: str
    shipped_at: datetime | None = None
    delivered_at: datetime | None = None


class ReturnItemLookup(StateModel):
    order_item_id: UUID
    requested_quantity: int = Field(gt=0)
    approved_quantity: int | None = Field(default=None, ge=0)
    reason_code: str | None = None
    item_condition: str | None = None


class ReturnStatusEventLookup(StateModel):
    from_status: str | None = None
    to_status: str
    source: str
    changed_at: datetime


class RefundLookup(StateModel):
    amount: Decimal = Field(gt=0)
    currency: str
    status: str
    initiated_at: datetime | None = None
    completed_at: datetime | None = None


class ReturnHistoryLookup(StateModel):
    return_number: str
    status: str
    reason_code: str
    return_method: str
    requested_at: datetime
    decided_at: datetime | None = None
    completed_at: datetime | None = None
    items: list[ReturnItemLookup] = Field(default_factory=list)
    status_history: list[ReturnStatusEventLookup] = Field(default_factory=list)
    refunds: list[RefundLookup] = Field(default_factory=list)
    shipments: list[OrderShipmentLookup] = Field(default_factory=list)


class OrderLookupData(StateModel):
    order_id: UUID
    order_number: str
    status: str
    currency: str = "VND"
    subtotal: Decimal = Field(default=Decimal("0"), ge=0)
    discount_amount: Decimal = Field(default=Decimal("0"), ge=0)
    shipping_fee: Decimal = Field(default=Decimal("0"), ge=0)
    total_amount: Decimal = Field(default=Decimal("0"), ge=0)
    placed_at: datetime | None = None
    delivered_at: datetime | None = None
    items: list[OrderItemLookup] = Field(default_factory=list)
    payments: list[OrderPaymentLookup] = Field(default_factory=list)
    shipments: list[OrderShipmentLookup] = Field(default_factory=list)
    returns: list[ReturnHistoryLookup] = Field(default_factory=list)


class ClassificationState(StateModel):
    status: Literal["pending", "completed", "failed"] = "pending"
    intent: ReturnIntent | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    entities: dict[str, Any] = Field(default_factory=dict)
    classifier_version: str | None = None


class ClarificationState(StateModel):
    needed: bool = False
    question: str | None = None
    missing_fields: list[str] = Field(default_factory=list)
    awaiting_user: bool = False


class InputValidationState(StateModel):
    status: Literal["not_started", "valid", "invalid"] = "not_started"
    errors: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)


class OrderLookupState(StateModel):
    status: Literal[
        "not_started", "pending", "succeeded", "not_found", "needs_clarification", "failed", "skipped"
    ] = "not_started"
    orders: list[OrderLookupData] = Field(default_factory=list)
    data: OrderLookupData | None = None
    has_more: bool = False
    error: str | None = None


class PolicyEvidence(StateModel):
    doc_id: str
    version: int = Field(ge=1)
    chunk_id: str
    excerpt: str = Field(min_length=1)
    source_title: str | None = None
    source_url: str | None = None
    score: float | None = Field(default=None, ge=0)


class PolicyLookupState(StateModel):
    status: StateStatus = "not_started"
    evidence: list[PolicyEvidence] = Field(default_factory=list)
    error: str | None = None


class BusinessValidationState(StateModel):
    status: Literal["not_started", "valid", "invalid", "needs_review"] = "not_started"
    checks: dict[str, Literal["pass", "fail", "unknown"]] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)


class DecisionProposalState(StateModel):
    status: Literal["not_started", "needs_clarification", "completed", "failed"] = "not_started"
    outcome: Literal["approve", "reject", "manual_review"] | None = None
    reason: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    rule_set_version: str | None = None
    evaluated_at: datetime | None = None


class PersistenceState(StateModel):
    status: Literal["not_started", "committed", "failed"] = "not_started"
    return_request_id: UUID | None = None
    error: str | None = None


class NotificationState(StateModel):
    status: StateStatus = "not_started"
    provider_message_id: str | None = None
    error: str | None = None


class ResponseState(StateModel):
    status: Literal["not_started", "ready", "failed"] = "not_started"
    message: str | None = None


class StateTransition(StateModel):
    component: str
    from_status: str | None = None
    to_status: str
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    detail: str | None = None


class ChildAgentResult(StateModel):
    """Outcome returned by a manager child agent for this request."""

    status: ChildAgentStatus
    message: str | None = None
    error_code: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class OrchestrationState(StateModel):
    """Manager-owned execution summary, separate from business decisions."""

    status: Literal["not_started", "completed", "partial", "failed"] = "not_started"
    child_results: dict[str, ChildAgentResult] = Field(default_factory=dict)


class AgentSharedState(StateModel):
    """Orchestrator-owned state for a single user request."""

    context: RequestContext
    input: RequestInput
    classification: ClassificationState = Field(default_factory=ClassificationState)
    clarification: ClarificationState = Field(default_factory=ClarificationState)
    input_validation: InputValidationState = Field(default_factory=InputValidationState)
    order_lookup: OrderLookupState = Field(default_factory=OrderLookupState)
    policy_lookup: PolicyLookupState = Field(default_factory=PolicyLookupState)
    business_validation: BusinessValidationState = Field(default_factory=BusinessValidationState)
    decision_proposal: DecisionProposalState = Field(default_factory=DecisionProposalState)
    persistence: PersistenceState = Field(default_factory=PersistenceState)
    notification: NotificationState = Field(default_factory=NotificationState)
    response: ResponseState = Field(default_factory=ResponseState)
    orchestration: OrchestrationState = Field(default_factory=OrchestrationState)
    transitions: list[StateTransition] = Field(default_factory=list)
