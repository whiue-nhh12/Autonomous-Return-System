"""Repository for payment transactions."""

from backend.models import Payment
from backend.repositories.base import BaseRepository


class PaymentRepository(BaseRepository[Payment]):
    model = Payment

    def get_by_provider_transaction_id(self, transaction_id: str) -> Payment | None:
        return self.find_one_by(provider_transaction_id=transaction_id)
