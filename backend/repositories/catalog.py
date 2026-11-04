"""Repository for catalog products."""

from backend.models import Product
from backend.repositories.base import BaseRepository


class ProductRepository(BaseRepository[Product]):
    model = Product

    def get_by_sku(self, sku: str) -> Product | None:
        return self.find_one_by(sku=sku)
