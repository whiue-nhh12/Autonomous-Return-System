"""Interpret order requests and normalize read-only database results."""

from __future__ import annotations

import re
import unicodedata
from typing import Protocol
from uuid import UUID

from backend.repositories.order_read import OrderReadPage
from agent.agents.order.state.result import OrderAgentRequest, OrderQueryPlan
from agent.core.state.shared import (
    OrderItemLookup,
    OrderLookupData,
    OrderPaymentLookup,
    OrderShipmentLookup,
    RefundLookup,
    ReturnHistoryLookup,
    ReturnItemLookup,
    ReturnStatusEventLookup,
)
from backend.models.enums import ReturnStatus


class OrderHistorySearch(Protocol):
    def search_history(
        self,
        *,
        user_id: UUID,
        order_number: str | None = None,
        limit: int = 10,
    ) -> OrderReadPage:
        ...


HISTORY_TERMS = (
    "order history", "order list", "all orders", "my orders", "past orders",
    "lich su don", "danh sach don", "cac don hang", "cac don cua toi",
)
LATEST_TERMS = (
    "latest order", "most recent order", "newest order", "last order",
    "don hang gan nhat", "don hang moi nhat", "don moi nhat",
)
ORDER_NUMBER_PATTERNS = (
    re.compile(
        r"\b(?:order\s*(?:number|no\.?|#)|order\s*#)\s*[:#]?\s*([A-Z0-9][A-Z0-9_-]{0,63})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:mã\s*đơn(?:\s*hàng)?|đơn\s*hàng\s*(?:số|#))\s*[:#]?\s*([A-Z0-9][A-Z0-9_-]{0,63})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bđơn\s+(?!hàng\b)([A-Z0-9][A-Z0-9_-]{0,63})\b",
        re.IGNORECASE,
    ),
)


def plan_order_query(request: OrderAgentRequest) -> OrderQueryPlan:
    normalized = _normalize(request.message)
    order_number = request.order_number or _extract_order_number(request.message)
    history = any(term in normalized for term in HISTORY_TERMS)
    latest = any(term in normalized for term in LATEST_TERMS)
    return OrderQueryPlan(
        order_number=order_number,
        wants_history=history and order_number is None,
        wants_latest=latest and order_number is None,
    )


def _extract_order_number(message: str) -> str | None:
    for pattern in ORDER_NUMBER_PATTERNS:
        match = pattern.search(message)
        if match:
            return match.group(1).strip()
    return None


def _normalize(message: str) -> str:
    decomposed = unicodedata.normalize("NFKD", message.casefold())
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    return without_marks.replace("đ", "d")


def normalize_order_page(page: OrderReadPage) -> list[OrderLookupData]:
    """Convert explicitly loaded DB rows into the stable agent result contract."""
    normalized_orders: list[OrderLookupData] = []
    for bundle in page.orders:
        completed_returned: dict[object, int] = {}
        for return_request in bundle.returns:
            if return_request.status is not ReturnStatus.COMPLETED:
                continue
            for return_item in bundle.return_items.get(return_request.id, ()):
                if return_item.approved_quantity is not None:
                    completed_returned[return_item.order_item_id] = (
                        completed_returned.get(return_item.order_item_id, 0)
                        + return_item.approved_quantity
                    )

        items = [
            OrderItemLookup(
                order_item_id=item.id,
                product_id=item.product_id,
                product_name=item.product_name_snapshot,
                sku=item.sku_snapshot,
                purchased_quantity=item.quantity,
                quantity_already_returned=completed_returned.get(item.id, 0),
                unit_price=item.unit_price,
                line_total=item.line_total,
            )
            for item in bundle.items
        ]
        payments = [
            OrderPaymentLookup(
                status=_enum_value(payment.status),
                method=_enum_value(payment.method),
                amount=payment.amount,
                currency=payment.currency,
                paid_at=payment.paid_at,
            )
            for payment in bundle.payments
        ]
        shipments = [_normalize_shipment(shipment) for shipment in bundle.shipments]
        returns = [
            ReturnHistoryLookup(
                return_number=request.return_number,
                status=_enum_value(request.status),
                reason_code=_enum_value(request.reason_code),
                return_method=_enum_value(request.return_method),
                requested_at=request.requested_at,
                decided_at=request.decided_at,
                completed_at=request.completed_at,
                items=[
                    ReturnItemLookup(
                        order_item_id=item.order_item_id,
                        requested_quantity=item.requested_quantity,
                        approved_quantity=item.approved_quantity,
                        reason_code=(
                            _enum_value(item.reason_code)
                            if item.reason_code is not None
                            else None
                        ),
                        item_condition=(
                            _enum_value(item.item_condition)
                            if item.item_condition is not None
                            else None
                        ),
                    )
                    for item in bundle.return_items.get(request.id, ())
                ],
                status_history=[
                    ReturnStatusEventLookup(
                        from_status=(
                            _enum_value(event.from_status)
                            if event.from_status is not None
                            else None
                        ),
                        to_status=_enum_value(event.to_status),
                        source=_enum_value(event.source),
                        changed_at=event.changed_at,
                    )
                    for event in bundle.return_status_history.get(request.id, ())
                ],
                refunds=[
                    RefundLookup(
                        amount=refund.amount,
                        currency=refund.currency,
                        status=_enum_value(refund.status),
                        initiated_at=refund.initiated_at,
                        completed_at=refund.completed_at,
                    )
                    for refund in bundle.refunds.get(request.id, ())
                ],
                shipments=[
                    _normalize_shipment(shipment)
                    for shipment in bundle.return_shipments.get(request.id, ())
                ],
            )
            for request in bundle.returns
        ]
        order = bundle.order
        normalized_orders.append(
            OrderLookupData(
                order_id=order.id,
                order_number=order.order_number,
                status=_enum_value(order.status),
                currency=order.currency,
                subtotal=order.subtotal,
                discount_amount=order.discount_amount,
                shipping_fee=order.shipping_fee,
                total_amount=order.total_amount,
                placed_at=order.placed_at,
                delivered_at=order.delivered_at,
                items=items,
                payments=payments,
                shipments=shipments,
                returns=returns,
            )
        )
    return normalized_orders


def _normalize_shipment(shipment) -> OrderShipmentLookup:
    return OrderShipmentLookup(
        direction=_enum_value(shipment.direction),
        provider=shipment.provider,
        tracking_number=shipment.tracking_number,
        tracking_url=shipment.tracking_url,
        status=_enum_value(shipment.status),
        shipped_at=shipment.shipped_at,
        delivered_at=shipment.delivered_at,
    )


def _enum_value(value) -> str:
    return str(getattr(value, "value", value))
