"""Repositories for orders and order items."""

from backend.models import Order, OrderItem
from backend.repositories.base import BaseRepository


class OrderRepository(BaseRepository[Order]):
    model = Order

    def get_by_order_number(self, order_number: str) -> Order | None:
        return self.find_one_by(order_number=order_number)

    def get_by_external_id(self, source: str, external_order_id: str) -> Order | None:
        return self.find_one_by(source=source, external_order_id=external_order_id)


class OrderItemRepository(BaseRepository[OrderItem]):
    model = OrderItem
