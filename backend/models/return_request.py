"""Return request, return item, status history, and refund models."""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    JSON,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from backend.models.enums import (
    DecisionSource,
    RefundStatus,
    ReturnItemCondition,
    ReturnMethod,
    ReturnReason,
    ReturnStatus,
    StatusChangeSource,
    enum_column,
)


class ReturnRequest(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "return_requests"

    return_number: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    # The manager request ID makes graph retries idempotent at the submission boundary.
    submission_request_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True, unique=True)
    submission_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    order_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("orders.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    status: Mapped[ReturnStatus] = mapped_column(
        enum_column(ReturnStatus, "return_status"),
        nullable=False,
        default=ReturnStatus.SUBMITTED,
        server_default=ReturnStatus.SUBMITTED.value,
        index=True,
    )
    reason_code: Mapped[ReturnReason] = mapped_column(
        enum_column(ReturnReason, "return_reason"),
        nullable=False,
    )
    reason_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    return_method: Mapped[ReturnMethod] = mapped_column(
        enum_column(ReturnMethod, "return_method"),
        nullable=False,
    )
    pickup_address_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    decision_source: Mapped[DecisionSource | None] = mapped_column(
        enum_column(DecisionSource, "decision_source"),
        nullable=True,
    )
    decision_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    policy_version_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)
    reviewed_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    requested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    order: Mapped["Order"] = relationship(back_populates="return_requests")
    user: Mapped["User"] = relationship(
        back_populates="return_requests",
        foreign_keys=[user_id],
    )
    reviewer: Mapped["User | None"] = relationship(
        back_populates="reviewed_return_requests",
        foreign_keys=[reviewed_by],
    )
    items: Mapped[list["ReturnItem"]] = relationship(back_populates="return_request")
    status_history: Mapped[list["ReturnStatusHistory"]] = relationship(back_populates="return_request")
    decision_proposals: Mapped[list["ReturnDecisionProposal"]] = relationship(
        back_populates="return_request"
    )
    refunds: Mapped[list["Refund"]] = relationship(back_populates="return_request")
    shipments: Mapped[list["Shipment"]] = relationship(back_populates="return_request")


class ReturnItem(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "return_items"
    __table_args__ = (
        UniqueConstraint("return_request_id", "order_item_id", name="uq_return_items_request_order_item"),
        CheckConstraint("requested_quantity > 0", name="ck_return_items_requested_quantity_positive"),
        CheckConstraint(
            "approved_quantity IS NULL OR "
            "(approved_quantity >= 0 AND approved_quantity <= requested_quantity)",
            name="ck_return_items_approved_quantity",
        ),
        Index("ix_return_items_order_item", "order_item_id"),
    )

    return_request_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("return_requests.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    order_item_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("order_items.id", ondelete="RESTRICT"),
        nullable=False,
    )
    requested_quantity: Mapped[int] = mapped_column(nullable=False)
    approved_quantity: Mapped[int | None] = mapped_column(nullable=True)
    reason_code: Mapped[ReturnReason | None] = mapped_column(
        enum_column(ReturnReason, "return_item_reason"),
        nullable=True,
    )
    item_condition: Mapped[ReturnItemCondition | None] = mapped_column(
        enum_column(ReturnItemCondition, "return_item_condition"),
        nullable=True,
    )

    return_request: Mapped[ReturnRequest] = relationship(back_populates="items")
    order_item: Mapped["OrderItem"] = relationship(back_populates="return_items")


class ReturnStatusHistory(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "return_status_history"
    __table_args__ = (Index("ix_return_status_history_request_changed_at", "return_request_id", "changed_at"),)

    return_request_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("return_requests.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    from_status: Mapped[ReturnStatus | None] = mapped_column(
        enum_column(ReturnStatus, "return_history_from_status"),
        nullable=True,
    )
    to_status: Mapped[ReturnStatus] = mapped_column(
        enum_column(ReturnStatus, "return_history_to_status"),
        nullable=False,
    )
    changed_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True, unique=True)
    source: Mapped[StatusChangeSource] = mapped_column(
        enum_column(StatusChangeSource, "return_status_change_source"),
        nullable=False,
    )
    changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    return_request: Mapped[ReturnRequest] = relationship(back_populates="status_history")
    changed_by: Mapped["User | None"] = relationship(foreign_keys=[changed_by_user_id])


class ReturnDecisionProposal(UUIDPrimaryKeyMixin, Base):
    """Immutable Decision Agent assessment and its optional admin decision."""

    __tablename__ = "return_decision_proposals"
    __table_args__ = (
        CheckConstraint(
            "status IN ('completed', 'needs_clarification', 'failed')",
            name="ck_return_decision_proposals_status",
        ),
        CheckConstraint(
            "outcome IS NULL OR outcome IN ('approve', 'reject', 'manual_review')",
            name="ck_return_decision_proposals_outcome",
        ),
        CheckConstraint(
            "decision_action IS NULL OR decision_action IN ('approve', 'reject')",
            name="ck_return_decision_proposals_decision_action",
        ),
        UniqueConstraint("evaluation_request_id", name="uq_return_proposals_evaluation_request"),
        UniqueConstraint("decision_idempotency_key", name="uq_return_proposals_idempotency_key"),
        Index("ix_return_proposals_request_evaluated", "return_request_id", "evaluated_at"),
    )

    return_request_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("return_requests.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    evaluation_request_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    outcome: Mapped[str | None] = mapped_column(String(32), nullable=True)
    assessment_message: Mapped[str] = mapped_column(Text, nullable=False)
    checks: Mapped[list[dict]] = mapped_column(JSON, nullable=False, default=list)
    reason_codes: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    missing_fields: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    evidence_refs: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    policy_versions: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    policy_version_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)
    claim_facts: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    evaluated_order_facts: Mapped[list[dict]] = mapped_column(JSON, nullable=False, default=list)
    evaluated_data_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    rule_set_version: Mapped[str | None] = mapped_column(String(255), nullable=True)
    classifier_version: Mapped[str | None] = mapped_column(String(255), nullable=True)
    decision_engine_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default="return-decision-v1", server_default="return-decision-v1"
    )
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    decision_action: Mapped[str | None] = mapped_column(String(16), nullable=True)
    decision_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    decision_idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    decided_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    return_request: Mapped[ReturnRequest] = relationship(back_populates="decision_proposals")
    decision_maker: Mapped["User | None"] = relationship(foreign_keys=[decided_by])


class Refund(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "refunds"
    __table_args__ = (
        CheckConstraint("amount > 0", name="ck_refunds_amount_positive"),
        Index("ix_refunds_return_status", "return_request_id", "status"),
        UniqueConstraint("idempotency_key", name="uq_refunds_idempotency_key"),
    )

    return_request_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("return_requests.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    payment_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("payments.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    provider_refund_id: Mapped[str | None] = mapped_column(String(255), nullable=True, unique=True)
    # Nullable for legacy/manual refunds; new refund commands always provide a key.
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="VND", server_default="VND")
    status: Mapped[RefundStatus] = mapped_column(
        enum_column(RefundStatus, "refund_status"),
        nullable=False,
        default=RefundStatus.PENDING,
        server_default=RefundStatus.PENDING.value,
        index=True,
    )
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    initiated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    return_request: Mapped[ReturnRequest] = relationship(back_populates="refunds")
    payment: Mapped["Payment | None"] = relationship(back_populates="refunds")


if TYPE_CHECKING:
    from backend.models.order import Order, OrderItem
    from backend.models.payment import Payment
    from backend.models.shipment import Shipment
    from backend.models.user import User
