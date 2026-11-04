"""Read-only application service for user-scoped order history queries."""

from __future__ import annotations

from collections.abc import Callable
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from backend.db.session import get_order_read_session_factory
from backend.repositories.order_read import OrderReadPage, OrderReadRepository


class OrderQueryService:
    """Execute bounded order-history reads in a read-only database transaction."""

    def __init__(
        self,
        session_factory_provider: Callable[[], sessionmaker[Session]] = get_order_read_session_factory,
    ) -> None:
        self._session_factory_provider = session_factory_provider

    def search_history(
        self,
        *,
        user_id: UUID,
        order_number: str | None = None,
        limit: int = 10,
    ) -> OrderReadPage:
        session_factory = self._session_factory_provider()
        with session_factory() as session:
            with session.begin():
                if session.get_bind().dialect.name == "postgresql":
                    session.execute(text("SET TRANSACTION READ ONLY"))
                return OrderReadRepository(session).search_history(
                    user_id=user_id,
                    order_number=order_number,
                    limit=limit,
                )
