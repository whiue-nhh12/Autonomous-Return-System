"""SQLAlchemy models for the return-processing backend."""

from backend.models.address import Address
from backend.models.base import Base
from backend.models.catalog import Product
from backend.models.order import Order, OrderItem
from backend.models.outbox import OutboxEvent
from backend.models.payment import Payment
from backend.models.return_request import (
    Refund,
    ReturnDecisionProposal,
    ReturnItem,
    ReturnRequest,
    ReturnStatusHistory,
)
from backend.models.shipment import Shipment
from backend.models.user import User, UserProfile

__all__ = [
    "Address",
    "Base",
    "Order",
    "OrderItem",
    "OutboxEvent",
    "Payment",
    "Product",
    "Refund",
    "ReturnDecisionProposal",
    "ReturnItem",
    "ReturnRequest",
    "ReturnStatusHistory",
    "Shipment",
    "User",
    "UserProfile",
]
