"""Canonical fingerprints for the order facts used in a return decision."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from agent.core.state.shared import OrderLookupData
from backend.models import Order, OrderItem, ReturnItem, ReturnRequest


def fingerprint_order_lookup(order: OrderLookupData) -> str:
    """Fingerprint order facts as seen by the read-only order agent."""
    payload = {
        "id": str(order.order_id),
        "number": order.order_number,
        "status": order.status,
        "currency": order.currency,
        "subtotal": _money(order.subtotal),
        "discount_amount": _money(order.discount_amount),
        "shipping_fee": _money(order.shipping_fee),
        "total_amount": _money(order.total_amount),
        "placed_at": _timestamp(order.placed_at),
        "delivered_at": _timestamp(order.delivered_at),
        "items": [
            {
                "id": str(item.order_item_id),
                "product_id": str(item.product_id),
                "product_name": item.product_name,
                "sku": item.sku,
                "quantity": item.purchased_quantity,
                "unit_price": _money(item.unit_price),
                "line_total": _money(item.line_total),
                "completed_quantity": item.quantity_already_returned,
            }
            for item in sorted(order.items, key=lambda row: str(row.order_item_id))
        ],
        "returns": [
            {
                "return_number": request.return_number,
                "status": request.status,
                "reason_code": request.reason_code,
                "return_method": request.return_method,
                "items": [
                    {
                        "order_item_id": str(item.order_item_id),
                        "requested_quantity": item.requested_quantity,
                        "approved_quantity": item.approved_quantity,
                        "reason_code": item.reason_code,
                        "item_condition": item.item_condition,
                    }
                    for item in sorted(request.items, key=lambda row: str(row.order_item_id))
                ],
            }
            for request in sorted(order.returns, key=lambda row: row.return_number)
        ],
    }
    return _fingerprint(payload)


def serialize_evaluated_order_facts(order: OrderLookupData) -> dict:
    """Return only the order fields used by return rules for admin audit."""
    return {
        "order_id": str(order.order_id),
        "order_number": order.order_number,
        "status": order.status,
        "delivered_at": _timestamp(order.delivered_at),
        "items": [
            {
                "order_item_id": str(item.order_item_id),
                "product_id": str(item.product_id),
                "product_name": item.product_name,
                "sku": item.sku,
                "purchased_quantity": item.purchased_quantity,
                "quantity_already_returned": item.quantity_already_returned,
            }
            for item in sorted(order.items, key=lambda row: str(row.order_item_id))
        ],
        "returns": [
            {
                "return_number": request.return_number,
                "status": request.status,
                "reason_code": request.reason_code,
                "return_method": request.return_method,
                "items": [
                    {
                        "order_item_id": str(item.order_item_id),
                        "requested_quantity": item.requested_quantity,
                        "approved_quantity": item.approved_quantity,
                        "reason_code": item.reason_code,
                        "item_condition": item.item_condition,
                    }
                    for item in sorted(request.items, key=lambda row: str(row.order_item_id))
                ],
            }
            for request in sorted(order.returns, key=lambda row: row.return_number)
        ],
    }


def fingerprint_order_rows(
    session: Session,
    order: Order,
    *,
    exclude_return_request_id: UUID | None = None,
) -> str:
    """Fingerprint current mutable order facts while the aggregate is locked."""
    order_items = list(
        session.scalars(
            select(OrderItem)
            .where(OrderItem.order_id == order.id)
            .order_by(OrderItem.id)
            .with_for_update()
        )
    )
    requests_query = select(ReturnRequest).where(ReturnRequest.order_id == order.id)
    if exclude_return_request_id is not None:
        requests_query = requests_query.where(ReturnRequest.id != exclude_return_request_id)
    requests = list(session.scalars(requests_query.order_by(ReturnRequest.return_number)))
    request_ids = [request.id for request in requests]
    return_items = (
        list(
            session.scalars(
                select(ReturnItem)
                .where(ReturnItem.return_request_id.in_(request_ids))
                .order_by(ReturnItem.return_request_id, ReturnItem.order_item_id)
            )
        )
        if request_ids
        else []
    )
    return_items_by_request: dict[UUID, list[ReturnItem]] = {}
    for item in return_items:
        return_items_by_request.setdefault(item.return_request_id, []).append(item)

    completed_quantity: dict[UUID, int] = {}
    for request in requests:
        if _value(request.status) != "completed":
            continue
        for item in return_items_by_request.get(request.id, []):
            if item.approved_quantity is not None:
                completed_quantity[item.order_item_id] = (
                    completed_quantity.get(item.order_item_id, 0) + item.approved_quantity
                )

    payload = {
        "id": str(order.id),
        "number": order.order_number,
        "status": _value(order.status),
        "currency": order.currency,
        "subtotal": _money(order.subtotal),
        "discount_amount": _money(order.discount_amount),
        "shipping_fee": _money(order.shipping_fee),
        "total_amount": _money(order.total_amount),
        "placed_at": _timestamp(order.placed_at),
        "delivered_at": _timestamp(order.delivered_at),
        "items": [
            {
                "id": str(item.id),
                "product_id": str(item.product_id),
                "product_name": item.product_name_snapshot,
                "sku": item.sku_snapshot,
                "quantity": item.quantity,
                "unit_price": _money(item.unit_price),
                "line_total": _money(item.line_total),
                "completed_quantity": completed_quantity.get(item.id, 0),
            }
            for item in order_items
        ],
        "returns": [
            {
                "return_number": request.return_number,
                "status": _value(request.status),
                "reason_code": _value(request.reason_code),
                "return_method": _value(request.return_method),
                "items": [
                    {
                        "order_item_id": str(item.order_item_id),
                        "requested_quantity": item.requested_quantity,
                        "approved_quantity": item.approved_quantity,
                        "reason_code": _value(item.reason_code),
                        "item_condition": _value(item.item_condition),
                    }
                    for item in return_items_by_request.get(request.id, [])
                ],
            }
            for request in requests
        ],
    }
    return _fingerprint(payload)


def _fingerprint(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _money(value: Decimal | int | str | None) -> str | None:
    return format(Decimal(value), ".2f") if value is not None else None


def _timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    normalized = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    return normalized.isoformat()


def _value(value):
    return getattr(value, "value", value)
