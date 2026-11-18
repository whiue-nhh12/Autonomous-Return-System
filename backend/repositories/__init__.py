"""Repositories for accessing backend models."""

from backend.repositories.base import BaseRepository
from backend.repositories.catalog import ProductRepository
from backend.repositories.order import OrderItemRepository, OrderRepository
from backend.repositories.order_read import OrderReadBundle, OrderReadPage, OrderReadRepository
from backend.repositories.payment import PaymentRepository
from backend.repositories.return_request import (
    OutboxEventRepository,
    RefundRepository,
    ReturnItemRepository,
    ReturnRequestRepository,
    ReturnStatusHistoryRepository,
)
from backend.repositories.shipment import ShipmentRepository
from backend.repositories.user import AddressRepository, UserProfileRepository, UserRepository

__all__ = [
    "AddressRepository",
    "BaseRepository",
    "OrderItemRepository",
    "OrderRepository",
    "OrderReadBundle",
    "OrderReadPage",
    "OrderReadRepository",
    "OutboxEventRepository",
    "PaymentRepository",
    "ProductRepository",
    "RefundRepository",
    "ReturnItemRepository",
    "ReturnRequestRepository",
    "ReturnStatusHistoryRepository",
    "ShipmentRepository",
    "UserProfileRepository",
    "UserRepository",
]
