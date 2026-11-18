"""Create a pending refund command after the returned item is received."""

from __future__ import annotations

from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from backend.models import (
    Order,
    OutboxEvent,
    Payment,
    Refund,
    ReturnRequest,
    ReturnStatusHistory,
    User,
)
from backend.models.enums import (
    DecisionSource,
    RefundStatus,
    ReturnStatus,
    UserRole,
    UserStatus,
)
from backend.repositories.payment import PaymentRepository
from backend.repositories.return_request import RefundRepository, ReturnRequestRepository
from backend.schemas.refund import RefundInitiationResponse


_RESERVING_REFUND_STATUSES = (
    RefundStatus.PENDING,
    RefundStatus.PROCESSING,
    RefundStatus.SUCCEEDED,
)


class RefundInitiationError(Exception):
    pass


class RefundInitiationForbidden(RefundInitiationError):
    pass


class RefundReturnNotFound(RefundInitiationError):
    pass


class RefundNotReady(RefundInitiationError):
    pass


class InvalidRefundPayment(RefundInitiationError):
    pass


class InvalidRefundAmount(RefundInitiationError):
    pass


class RefundInitiationIdempotencyConflict(RefundInitiationError):
    pass


class RefundInitiationService:
    """Queue a refund; it never contacts a payment provider.

    Current policy requires ``ITEM_RECEIVED``. There is no immediate-refund
    policy configuration in this application yet.
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def initiate(
        self,
        *,
        return_request_id: UUID,
        actor_id: UUID,
        idempotency_key: str,
        amount: Decimal,
        payment_id: UUID | None,
        note: str,
    ) -> RefundInitiationResponse:
        key = idempotency_key.strip()
        clean_note = note.strip()
        if not 16 <= len(key) <= 128:
            raise RefundInitiationIdempotencyConflict(
                "Idempotency-Key must contain 16 to 128 characters."
            )
        if not clean_note:
            raise ValueError("Refund note must contain non-whitespace text.")
        if (
            not isinstance(amount, Decimal)
            or not amount.is_finite()
            or amount <= 0
            or amount >= Decimal("10000000000")
            or amount.quantize(Decimal("0.01")) != amount
        ):
            raise InvalidRefundAmount(
                "Refund amount must be positive with at most two decimals."
            )

        try:
            with self._session_factory.begin() as session:
                self._require_active_admin(session, actor_id)
                existing = RefundRepository(session).get_by_idempotency_key(key)
                if existing is not None:
                    return self._replay_or_conflict(
                        existing,
                        return_request_id=return_request_id,
                        amount=amount,
                        payment_id=payment_id,
                    )

                request = ReturnRequestRepository(session).get_for_update(return_request_id)
                if request is None:
                    raise RefundReturnNotFound("Return request was not found.")
                existing = RefundRepository(session).get_by_idempotency_key(key)
                if existing is not None:
                    return self._replay_or_conflict(
                        existing,
                        return_request_id=return_request_id,
                        amount=amount,
                        payment_id=payment_id,
                    )
                if (
                    request.status is not ReturnStatus.ITEM_RECEIVED
                    or request.decision_source is None
                    or request.reviewed_by is None
                ):
                    raise RefundNotReady(
                        "Refund initiation requires an approved return in item_received state."
                    )
                receipt_history = session.scalar(
                    select(ReturnStatusHistory.id)
                    .where(
                        ReturnStatusHistory.return_request_id == request.id,
                        ReturnStatusHistory.from_status == ReturnStatus.APPROVED,
                        ReturnStatusHistory.to_status == ReturnStatus.ITEM_RECEIVED,
                    )
                    .limit(1)
                )
                if receipt_history is None:
                    raise RefundNotReady(
                        "Refund initiation requires a recorded item-received status change."
                    )
                order = session.scalar(
                    select(Order)
                    .where(Order.id == request.order_id)
                    .with_for_update()
                )
                if order is None or order.user_id != request.user_id:
                    raise RefundNotReady("Return request order ownership is inconsistent.")

                payments = PaymentRepository(session).list_succeeded_for_order_for_update(
                    order.id
                )
                payment = self._select_payment(payments, payment_id)
                if payment.currency != order.currency:
                    raise InvalidRefundPayment(
                        "Payment currency does not match the order currency."
                    )
                reserved_amount = session.scalar(
                    select(func.coalesce(func.sum(Refund.amount), 0)).where(
                        Refund.payment_id == payment.id,
                        Refund.status.in_(_RESERVING_REFUND_STATUSES),
                    )
                )
                remaining = payment.amount - Decimal(reserved_amount or 0)
                if amount > remaining:
                    raise InvalidRefundAmount(
                        "Refund amount exceeds the remaining refundable payment amount."
                    )

                refund = Refund(
                    id=uuid4(),
                    return_request_id=request.id,
                    payment_id=payment.id,
                    amount=amount,
                    currency=payment.currency,
                    status=RefundStatus.PENDING,
                    idempotency_key=key,
                )
                session.add(refund)
                session.flush()
                session.add(
                    OutboxEvent(
                        event_type="refund.requested",
                        aggregate_type="refund",
                        aggregate_id=refund.id,
                        idempotency_key=f"refund.requested:{refund.id}",
                        payload={
                            "refund_id": str(refund.id),
                            "return_request_id": str(request.id),
                            "payment_id": str(payment.id),
                            "initiated_by": str(actor_id),
                            "approved_by": str(request.reviewed_by),
                            "amount": format(amount, ".2f"),
                            "currency": payment.currency,
                            "note": clean_note,
                        },
                    )
                )
                session.flush()
                return RefundInitiationResponse(
                    return_request_id=request.id,
                    refund_id=refund.id,
                    return_status=request.status.value,
                    refund_status=refund.status.value,
                    amount=refund.amount,
                    currency=refund.currency,
                )
        except IntegrityError as exc:
            if not _is_refund_key_violation(exc):
                raise
            replay = self._find_replay(
                key,
                return_request_id=return_request_id,
                amount=amount,
                payment_id=payment_id,
            )
            if replay is not None:
                return replay
            raise RefundInitiationIdempotencyConflict(
                "Idempotency-Key has already been used for another refund."
            ) from exc

    @staticmethod
    def _require_active_admin(session: Session, actor_id: UUID) -> None:
        reviewer = session.scalar(
            select(User)
            .where(
                User.id == actor_id,
                User.role == UserRole.ADMIN,
                User.status == UserStatus.ACTIVE,
            )
            .with_for_update(read=True)
        )
        if reviewer is None:
            raise RefundInitiationForbidden("An active admin account is required.")

    @staticmethod
    def _select_payment(payments: list[Payment], payment_id: UUID | None) -> Payment:
        if payment_id is not None:
            selected = next((row for row in payments if row.id == payment_id), None)
            if selected is None:
                raise InvalidRefundPayment(
                    "Selected payment is not a succeeded payment for this order."
                )
            return selected
        if len(payments) != 1:
            raise InvalidRefundPayment(
                "A succeeded payment must be selected explicitly when the order has zero or multiple payments."
            )
        return payments[0]

    @staticmethod
    def _replay_or_conflict(
        refund: Refund,
        *,
        return_request_id: UUID,
        amount: Decimal,
        payment_id: UUID | None,
    ) -> RefundInitiationResponse:
        if (
            refund.return_request_id != return_request_id
            or refund.amount != amount
            or (payment_id is not None and refund.payment_id != payment_id)
        ):
            raise RefundInitiationIdempotencyConflict(
                "Idempotency-Key was already used with different refund details."
            )
        return RefundInitiationResponse(
            return_request_id=refund.return_request_id,
            refund_id=refund.id,
            return_status=refund.return_request.status.value,
            refund_status=refund.status.value,
            amount=refund.amount,
            currency=refund.currency,
            replayed=True,
        )

    def _find_replay(
        self,
        key: str,
        *,
        return_request_id: UUID,
        amount: Decimal,
        payment_id: UUID | None,
    ) -> RefundInitiationResponse | None:
        with self._session_factory() as session:
            existing = RefundRepository(session).get_by_idempotency_key(key)
            if existing is None:
                return None
            return self._replay_or_conflict(
                existing,
                return_request_id=return_request_id,
                amount=amount,
                payment_id=payment_id,
            )


def _is_refund_key_violation(error: IntegrityError) -> bool:
    detail = str(error.orig).casefold()
    return "uq_refunds_idempotency_key" in detail or "refunds.idempotency_key" in detail
