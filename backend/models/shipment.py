"""Outbound and return shipment model."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from backend.models.enums import ShipmentDirection, ShipmentStatus, enum_column


class Shipment(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "shipments"
    __table_args__ = (
        CheckConstraint(
            "(direction = 'outbound' AND order_id IS NOT NULL AND return_request_id IS NULL) "
            "OR (direction = 'return' AND return_request_id IS NOT NULL AND order_id IS NULL)",
            name="ck_shipments_direction_reference",
        ),
    )

    order_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("orders.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    return_request_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("return_requests.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    direction: Mapped[ShipmentDirection] = mapped_column(
        enum_column(ShipmentDirection, "shipment_direction"),
        nullable=False,
    )
    provider: Mapped[str] = mapped_column(String(100), nullable=False)
    tracking_number: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    tracking_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    status: Mapped[ShipmentStatus] = mapped_column(
        enum_column(ShipmentStatus, "shipment_status"),
        nullable=False,
        default=ShipmentStatus.PENDING,
        server_default=ShipmentStatus.PENDING.value,
        index=True,
    )
    shipped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    order: Mapped["Order | None"] = relationship(
        back_populates="shipments",
        foreign_keys=[order_id],
    )
    return_request: Mapped["ReturnRequest | None"] = relationship(
        back_populates="shipments",
        foreign_keys=[return_request_id],
    )


if TYPE_CHECKING:
    from backend.models.order import Order
    from backend.models.return_request import ReturnRequest
