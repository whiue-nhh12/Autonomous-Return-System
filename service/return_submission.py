"""Transactional creation of customer-submitted return requests."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from agent.agents.return_decision.state.result import ReturnClaimFacts
from agent.core.state.shared import OrderLookupData
from backend.models import (
    Order,
    OrderItem,
    ReturnItem,
    ReturnRequest,
    ReturnStatusHistory,
)
from backend.models.enums import (
    ReturnItemCondition,
    ReturnMethod,
    ReturnReason,
    ReturnStatus,
    StatusChangeSource,
)
from service.return_fingerprint import fingerprint_order_lookup, fingerprint_order_rows


class ReturnSubmissionError(Exception):
    """Base class for safe, expected return-submission failures."""


class ReturnSubmissionNotFound(ReturnSubmissionError):
    pass


class ReturnSubmissionConflict(ReturnSubmissionError):
    pass


class ReturnSubmissionStale(ReturnSubmissionError):
    pass


class ReturnSubmissionInvalid(ReturnSubmissionError):
    pass


class ReturnSubmissionNeedsFields(ReturnSubmissionError):
    def __init__(self, fields: list[str]) -> None:
        super().__init__("Required return details are missing.")
        self.fields = fields


@dataclass(frozen=True)
class ReturnSubmissionReceipt:
    return_request_id: UUID
    return_number: str
    status: str
    evaluated_data_fingerprint: str
    replayed: bool = False


class ReturnSubmissionService:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def submit(
        self,
        *,
        request_id: UUID,
        user_id: UUID,
        order_snapshot: OrderLookupData,
        claim: ReturnClaimFacts,
        reason_note: str,
    ) -> ReturnSubmissionReceipt:
        missing = [
            field
            for field, value in (
                ("order_number", claim.order_number or order_snapshot.order_number),
                (
                    "item",
                    claim.order_item_id
                    or claim.item_query
                    or (order_snapshot.items[0].order_item_id if len(order_snapshot.items) == 1 else None),
                ),
                ("requested_quantity", claim.requested_quantity),
                ("reason_code", claim.reason_code),
                ("return_method", claim.return_method),
            )
            if value is None
        ]
        if missing:
            raise ReturnSubmissionNeedsFields(missing)

        try:
            with self._session_factory.begin() as session:
                existing = session.scalar(
                    select(ReturnRequest).where(
                        ReturnRequest.submission_request_id == request_id
                    ).with_for_update()
                )
                if existing is not None:
                    if existing.user_id != user_id:
                        raise ReturnSubmissionConflict("Request ID was already used.")
                    existing_order = session.scalar(
                        select(Order)
                        .where(Order.id == existing.order_id)
                        .with_for_update()
                    )
                    if existing_order is None:
                        raise ReturnSubmissionNotFound("Order was not found for this account.")
                    current = fingerprint_order_rows(
                        session,
                        existing_order,
                        exclude_return_request_id=existing.id,
                    )
                    if current != existing.submission_fingerprint:
                        raise ReturnSubmissionStale(
                            "Order or return history changed since submission. Please reevaluate."
                        )
                    return ReturnSubmissionReceipt(
                        return_request_id=existing.id,
                        return_number=existing.return_number,
                        status=existing.status.value,
                        evaluated_data_fingerprint=existing.submission_fingerprint or "",
                        replayed=True,
                    )

                order = session.scalar(
                    select(Order).where(Order.id == order_snapshot.order_id).with_for_update()
                )
                if order is None or order.user_id != user_id:
                    raise ReturnSubmissionNotFound("Order was not found for this account.")
                # Another retry with the same manager request ID may have committed
                # while this transaction waited on the order lock.
                existing = session.scalar(
                    select(ReturnRequest)
                    .where(ReturnRequest.submission_request_id == request_id)
                    .with_for_update()
                )
                if existing is not None:
                    if existing.user_id != user_id or existing.order_id != order.id:
                        raise ReturnSubmissionConflict("Request ID was already used.")
                    current = fingerprint_order_rows(
                        session,
                        order,
                        exclude_return_request_id=existing.id,
                    )
                    if current != existing.submission_fingerprint:
                        raise ReturnSubmissionStale(
                            "Order or return history changed since submission. Please reevaluate."
                        )
                    return ReturnSubmissionReceipt(
                        return_request_id=existing.id,
                        return_number=existing.return_number,
                        status=existing.status.value,
                        evaluated_data_fingerprint=existing.submission_fingerprint or "",
                        replayed=True,
                    )
                if order.order_number.casefold() != order_snapshot.order_number.casefold():
                    raise ReturnSubmissionInvalid("Order data does not match the selected order.")

                current_fingerprint = fingerprint_order_rows(session, order)
                if current_fingerprint != fingerprint_order_lookup(order_snapshot):
                    raise ReturnSubmissionStale(
                        "Order or return history changed during evaluation. Please submit again."
                    )

                order_items = list(
                    session.scalars(
                        select(OrderItem)
                        .where(OrderItem.order_id == order.id)
                        .order_by(OrderItem.id)
                        .with_for_update()
                    )
                )
                selected_item = self._select_item(order_items, claim)
                reason = self._enum(ReturnReason, claim.reason_code, "return reason")
                method = self._enum(ReturnMethod, claim.return_method, "return method")
                condition = (
                    self._enum(ReturnItemCondition, claim.item_condition, "item condition")
                    if claim.item_condition
                    else None
                )
                requested_quantity = claim.requested_quantity
                if requested_quantity is None or requested_quantity < 1:
                    raise ReturnSubmissionNeedsFields(["requested_quantity"])

                reserved = self._reserved_quantity(session, selected_item.id)
                available = max(0, selected_item.quantity - reserved)
                if requested_quantity > available:
                    raise ReturnSubmissionInvalid(
                        f"Only {available} item(s) remain available for return."
                    )

                request_uuid = uuid4()
                return_number = f"RET-{request_uuid.hex[:20].upper()}"
                return_request = ReturnRequest(
                    id=request_uuid,
                    return_number=return_number,
                    submission_request_id=request_id,
                    submission_fingerprint=current_fingerprint,
                    order_id=order.id,
                    user_id=user_id,
                    status=ReturnStatus.SUBMITTED,
                    reason_code=reason,
                    reason_note=reason_note.strip()[:2000] or None,
                    return_method=method,
                    pickup_address_snapshot=(
                        order.shipping_address_snapshot if method is ReturnMethod.PICKUP else None
                    ),
                )
                session.add(return_request)
                session.add(
                    ReturnItem(
                        return_request_id=request_uuid,
                        order_item_id=selected_item.id,
                        requested_quantity=requested_quantity,
                        reason_code=reason,
                        item_condition=condition,
                    )
                )
                session.add(
                    ReturnStatusHistory(
                        return_request_id=request_uuid,
                        from_status=None,
                        to_status=ReturnStatus.SUBMITTED,
                        changed_by_user_id=user_id,
                        note="Customer submitted return request.",
                        source=StatusChangeSource.USER,
                    )
                )
                session.flush()
                return ReturnSubmissionReceipt(
                    return_request_id=request_uuid,
                    return_number=return_number,
                    status=ReturnStatus.SUBMITTED.value,
                    evaluated_data_fingerprint=current_fingerprint,
                )
        except IntegrityError as exc:
            replay = self._find_replay(request_id, user_id)
            if replay is not None:
                return replay
            raise ReturnSubmissionConflict("Return request could not be created safely.") from exc

    @staticmethod
    def _select_item(items: list[OrderItem], claim: ReturnClaimFacts) -> OrderItem:
        if claim.order_item_id is not None:
            matches = [item for item in items if item.id == claim.order_item_id]
        elif claim.item_query:
            query = claim.item_query.casefold().strip()
            matches = [
                item
                for item in items
                if query in item.product_name_snapshot.casefold()
                or query == item.sku_snapshot.casefold()
            ]
        elif len(items) == 1:
            matches = items
        else:
            raise ReturnSubmissionNeedsFields(["item"])
        if len(matches) != 1:
            raise ReturnSubmissionNeedsFields(["item"])
        return matches[0]

    @staticmethod
    def _reserved_quantity(session: Session, order_item_id: UUID) -> int:
        requests = list(
            session.scalars(
                select(ReturnRequest)
                .join(ReturnItem, ReturnItem.return_request_id == ReturnRequest.id)
                .where(ReturnItem.order_item_id == order_item_id)
                .order_by(ReturnRequest.id)
            )
        )
        active_statuses = {
            ReturnStatus.SUBMITTED,
            ReturnStatus.REVIEWING,
            ReturnStatus.APPROVED,
            ReturnStatus.ITEM_RECEIVED,
        }
        reserved = 0
        for request in requests:
            item = session.scalar(
                select(ReturnItem).where(
                    ReturnItem.return_request_id == request.id,
                    ReturnItem.order_item_id == order_item_id,
                )
            )
            if item is None:
                continue
            if request.status in active_statuses:
                reserved += (
                    item.approved_quantity
                    if item.approved_quantity is not None
                    else item.requested_quantity
                )
            elif request.status is ReturnStatus.COMPLETED:
                reserved += item.approved_quantity or 0
        return reserved

    @staticmethod
    def _enum(enum_type, value: str | None, label: str):
        try:
            return enum_type(value)
        except (TypeError, ValueError) as exc:
            raise ReturnSubmissionInvalid(f"Unsupported {label}.") from exc

    def _find_replay(self, request_id: UUID, user_id: UUID) -> ReturnSubmissionReceipt | None:
        with self._session_factory() as session:
            existing = session.scalar(
                select(ReturnRequest).where(ReturnRequest.submission_request_id == request_id)
            )
            if existing is None or existing.user_id != user_id:
                return None
            return ReturnSubmissionReceipt(
                return_request_id=existing.id,
                return_number=existing.return_number,
                status=existing.status.value,
                evaluated_data_fingerprint=existing.submission_fingerprint or "",
                replayed=True,
            )
