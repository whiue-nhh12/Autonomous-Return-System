"""Repository for payment transactions."""

from uuid import UUID

from sqlalchemy import select

from backend.models import Payment
from backend.models.enums import PaymentStatus
from backend.repositories.base import BaseRepository


class PaymentRepository(BaseRepository[Payment]):
    model = Payment

    def get_by_provider_transaction_id(self, transaction_id: str) -> Payment | None:
        return self.find_one_by(provider_transaction_id=transaction_id)

    def list_succeeded_for_order_for_update(self, order_id: UUID) -> list[Payment]:
        statement = (
            select(Payment)
            .where(
                Payment.order_id == order_id,
                Payment.status == PaymentStatus.SUCCEEDED,
            )
            .order_by(Payment.id)
            .with_for_update()
        )
        return list(self.session.scalars(statement))
