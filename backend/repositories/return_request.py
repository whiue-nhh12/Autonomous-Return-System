"""Repositories for return requests, returned items, histories, and refunds."""

from backend.models import Refund, ReturnItem, ReturnRequest, ReturnStatusHistory
from backend.repositories.base import BaseRepository


class ReturnRequestRepository(BaseRepository[ReturnRequest]):
    model = ReturnRequest

    def get_by_return_number(self, return_number: str) -> ReturnRequest | None:
        return self.find_one_by(return_number=return_number)


class ReturnItemRepository(BaseRepository[ReturnItem]):
    model = ReturnItem


class ReturnStatusHistoryRepository(BaseRepository[ReturnStatusHistory]):
    model = ReturnStatusHistory


class RefundRepository(BaseRepository[Refund]):
    model = Refund

    def get_by_provider_refund_id(self, refund_id: str) -> Refund | None:
        return self.find_one_by(provider_refund_id=refund_id)
