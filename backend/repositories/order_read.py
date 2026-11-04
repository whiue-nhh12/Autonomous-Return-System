"""Read-only, user-scoped queries for order history and related records."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import TypeVar
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.models import (
    Order,
    OrderItem,
    Payment,
    Refund,
    ReturnItem,
    ReturnRequest,
    ReturnStatusHistory,
    Shipment,
)
from backend.models.enums import ReturnStatus, ShipmentDirection

T = TypeVar("T")


@dataclass(frozen=True)
class OrderReadBundle:
    """One order and its explicitly loaded, ownership-scoped related records."""

    order: Order
    items: tuple[OrderItem, ...]
    payments: tuple[Payment, ...]
    shipments: tuple[Shipment, ...]
    returns: tuple[ReturnRequest, ...]
    return_items: dict[UUID, tuple[ReturnItem, ...]]
    return_status_history: dict[UUID, tuple[ReturnStatusHistory, ...]]
    refunds: dict[UUID, tuple[Refund, ...]]
    return_shipments: dict[UUID, tuple[Shipment, ...]]


@dataclass(frozen=True)
class OrderReadPage:
    orders: tuple[OrderReadBundle, ...]
    has_more: bool


class OrderReadRepository:
    """Query orders owned by one authenticated user; this class has no writes."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def search_history(
        self,
        *,
        user_id: UUID,
        order_number: str | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> OrderReadPage:
        if not 1 <= limit <= 20:
            raise ValueError("limit must be between 1 and 20")
        if offset < 0:
            raise ValueError("offset cannot be negative")

        statement = select(Order).where(Order.user_id == user_id)
        if order_number is not None:
            statement = statement.where(Order.order_number == order_number)
        statement = (
            statement.order_by(
                Order.placed_at.desc().nullslast(),
                Order.created_at.desc(),
                Order.id,
            )
            .limit(limit + 1)
            .offset(offset)
        )
        fetched_orders = list(self.session.scalars(statement))
        has_more = len(fetched_orders) > limit
        orders = fetched_orders[:limit]
        if not orders:
            return OrderReadPage(orders=(), has_more=False)

        order_ids = [order.id for order in orders]
        items = list(
            self.session.scalars(
                select(OrderItem)
                .join(Order, Order.id == OrderItem.order_id)
                .where(Order.user_id == user_id, OrderItem.order_id.in_(order_ids))
                .order_by(OrderItem.order_id, OrderItem.id)
            )
        )
        payments = list(
            self.session.scalars(
                select(Payment)
                .join(Order, Order.id == Payment.order_id)
                .where(Order.user_id == user_id, Payment.order_id.in_(order_ids))
                .order_by(Payment.order_id, Payment.created_at, Payment.id)
            )
        )
        shipments = list(
            self.session.scalars(
                select(Shipment)
                .join(Order, Order.id == Shipment.order_id)
                .where(
                    Order.user_id == user_id,
                    Shipment.order_id.in_(order_ids),
                    Shipment.direction == ShipmentDirection.OUTBOUND,
                )
                .order_by(Shipment.order_id, Shipment.created_at, Shipment.id)
            )
        )
        returns = list(
            self.session.scalars(
                select(ReturnRequest)
                .where(
                    ReturnRequest.user_id == user_id,
                    ReturnRequest.order_id.in_(order_ids),
                )
                .order_by(ReturnRequest.order_id, ReturnRequest.requested_at, ReturnRequest.id)
            )
        )
        return_ids = [item.id for item in returns]

        return_items: list[ReturnItem] = []
        status_history: list[ReturnStatusHistory] = []
        refunds: list[Refund] = []
        return_shipments: list[Shipment] = []
        if return_ids:
            return_items = list(
                self.session.scalars(
                    select(ReturnItem)
                    .join(ReturnRequest, ReturnRequest.id == ReturnItem.return_request_id)
                    .where(
                        ReturnRequest.user_id == user_id,
                        ReturnItem.return_request_id.in_(return_ids),
                    )
                    .order_by(ReturnItem.return_request_id, ReturnItem.id)
                )
            )
            status_history = list(
                self.session.scalars(
                    select(ReturnStatusHistory)
                    .join(
                        ReturnRequest,
                        ReturnRequest.id == ReturnStatusHistory.return_request_id,
                    )
                    .where(
                        ReturnRequest.user_id == user_id,
                        ReturnStatusHistory.return_request_id.in_(return_ids),
                    )
                    .order_by(
                        ReturnStatusHistory.return_request_id,
                        ReturnStatusHistory.changed_at,
                        ReturnStatusHistory.id,
                    )
                )
            )
            refunds = list(
                self.session.scalars(
                    select(Refund)
                    .join(ReturnRequest, ReturnRequest.id == Refund.return_request_id)
                    .where(
                        ReturnRequest.user_id == user_id,
                        Refund.return_request_id.in_(return_ids),
                    )
                    .order_by(Refund.return_request_id, Refund.created_at, Refund.id)
                )
            )
            return_shipments = list(
                self.session.scalars(
                    select(Shipment)
                    .join(
                        ReturnRequest,
                        ReturnRequest.id == Shipment.return_request_id,
                    )
                    .where(
                        ReturnRequest.user_id == user_id,
                        Shipment.return_request_id.in_(return_ids),
                        Shipment.direction == ShipmentDirection.RETURN,
                    )
                    .order_by(Shipment.return_request_id, Shipment.created_at, Shipment.id)
                )
            )

        items_by_order = _group(items, lambda item: item.order_id)
        payments_by_order = _group(payments, lambda payment: payment.order_id)
        shipments_by_order = _group(shipments, lambda shipment: shipment.order_id)
        returns_by_order = _group(returns, lambda request: request.order_id)
        return_items_by_request = _group(return_items, lambda item: item.return_request_id)
        history_by_request = _group(status_history, lambda event: event.return_request_id)
        refunds_by_request = _group(refunds, lambda refund: refund.return_request_id)
        return_shipments_by_request = _group(
            return_shipments, lambda shipment: shipment.return_request_id
        )

        bundles = tuple(
            OrderReadBundle(
                order=order,
                items=tuple(items_by_order.get(order.id, ())),
                payments=tuple(payments_by_order.get(order.id, ())),
                shipments=tuple(shipments_by_order.get(order.id, ())),
                returns=tuple(returns_by_order.get(order.id, ())),
                return_items={
                    item.id: tuple(return_items_by_request.get(item.id, ()))
                    for item in returns_by_order.get(order.id, ())
                },
                return_status_history={
                    item.id: tuple(history_by_request.get(item.id, ()))
                    for item in returns_by_order.get(order.id, ())
                },
                refunds={
                    item.id: tuple(refunds_by_request.get(item.id, ()))
                    for item in returns_by_order.get(order.id, ())
                },
                return_shipments={
                    item.id: tuple(return_shipments_by_request.get(item.id, ()))
                    for item in returns_by_order.get(order.id, ())
                },
            )
            for order in orders
        )
        return OrderReadPage(orders=bundles, has_more=has_more)


def _group(values: list[T], key_for: Callable[[T], UUID]) -> dict[UUID, list[T]]:
    grouped: dict[UUID, list[T]] = {}
    for value in values:
        key = key_for(value)
        grouped.setdefault(key, []).append(value)
    return grouped
