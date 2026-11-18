"""Admin transition for recording inspected return items as received."""

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from backend.models import (
    Order,
    ReturnItem,
    ReturnRequest,
    ReturnStatusHistory,
    User,
)
from backend.models.enums import (
    ReturnStatus,
    StatusChangeSource,
    UserRole,
    UserStatus,
)
from backend.schemas.return_receipt import ReturnReceiptResponse


class ReturnReceiptError(Exception):
    pass


class ReturnReceiptForbidden(ReturnReceiptError):
    pass


class ReturnReceiptNotFound(ReturnReceiptError):
    pass


class ReturnReceiptConflict(ReturnReceiptError):
    pass


class ReturnReceiptService:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def mark_received(
        self,
        *,
        return_request_id: UUID,
        actor_id: UUID,
        idempotency_key: str,
        note: str,
    ) -> ReturnReceiptResponse:
        key = idempotency_key.strip()
        clean_note = note.strip()
        if not 16 <= len(key) <= 128:
            raise ReturnReceiptConflict("Idempotency-Key must contain 16 to 128 characters.")
        if not clean_note:
            raise ValueError("Receipt note must contain non-whitespace text.")

        try:
            with self._session_factory.begin() as session:
                self._require_active_admin(session, actor_id)
                replay = self._find_existing(
                    session,
                    key,
                    return_request_id=return_request_id,
                    note=clean_note,
                )
                if replay is not None:
                    return replay

                request = session.scalar(
                    select(ReturnRequest)
                    .where(ReturnRequest.id == return_request_id)
                    .with_for_update()
                )
                if request is None:
                    raise ReturnReceiptNotFound("Return request was not found.")
                # A concurrent retry may have committed while this transaction waited.
                replay = self._find_existing(
                    session,
                    key,
                    return_request_id=return_request_id,
                    note=clean_note,
                )
                if replay is not None:
                    return replay
                if request.status is not ReturnStatus.APPROVED:
                    raise ReturnReceiptConflict(
                        f"Cannot record receipt for a return in '{request.status.value}' state."
                    )
                order = session.scalar(
                    select(Order).where(Order.id == request.order_id).with_for_update()
                )
                if order is None or order.user_id != request.user_id:
                    raise ReturnReceiptConflict("Return request order ownership is inconsistent.")
                items = list(
                    session.scalars(
                        select(ReturnItem)
                        .where(ReturnItem.return_request_id == request.id)
                        .order_by(ReturnItem.order_item_id)
                        .with_for_update()
                    )
                )
                if not items or any(
                    item.approved_quantity is None or item.approved_quantity < 1
                    for item in items
                ):
                    raise ReturnReceiptConflict(
                        "Only approved return items can be marked as received."
                    )

                request.status = ReturnStatus.ITEM_RECEIVED
                session.add(
                    ReturnStatusHistory(
                        return_request_id=request.id,
                        from_status=ReturnStatus.APPROVED,
                        to_status=ReturnStatus.ITEM_RECEIVED,
                        changed_by_user_id=actor_id,
                        note=clean_note,
                        idempotency_key=key,
                        source=StatusChangeSource.STAFF,
                        changed_at=datetime.now(timezone.utc),
                    )
                )
                session.flush()
                return ReturnReceiptResponse(
                    return_request_id=request.id,
                    return_status=request.status.value,
                )
        except IntegrityError as exc:
            replay = self._find_replay(
                key,
                return_request_id=return_request_id,
                note=clean_note,
            )
            if replay is not None:
                return replay
            raise ReturnReceiptConflict(
                "Idempotency-Key has already been used for another return action."
            ) from exc

    @staticmethod
    def _require_active_admin(session: Session, actor_id: UUID) -> None:
        actor = session.scalar(
            select(User)
            .where(
                User.id == actor_id,
                User.role == UserRole.ADMIN,
                User.status == UserStatus.ACTIVE,
            )
            .with_for_update(read=True)
        )
        if actor is None:
            raise ReturnReceiptForbidden("An active admin account is required.")

    @staticmethod
    def _find_existing(
        session: Session,
        key: str,
        *,
        return_request_id: UUID,
        note: str,
    ) -> ReturnReceiptResponse | None:
        history = session.scalar(
            select(ReturnStatusHistory).where(
                ReturnStatusHistory.idempotency_key == key
            )
        )
        if history is None:
            return None
        if (
            history.return_request_id != return_request_id
            or history.from_status is not ReturnStatus.APPROVED
            or history.to_status is not ReturnStatus.ITEM_RECEIVED
            or history.note != note
        ):
            raise ReturnReceiptConflict(
                "Idempotency-Key was already used for another return action."
            )
        return ReturnReceiptResponse(
            return_request_id=return_request_id,
            return_status=ReturnStatus.ITEM_RECEIVED.value,
            replayed=True,
        )

    def _find_replay(
        self,
        key: str,
        *,
        return_request_id: UUID,
        note: str,
    ) -> ReturnReceiptResponse | None:
        with self._session_factory() as session:
            return self._find_existing(
                session,
                key,
                return_request_id=return_request_id,
                note=note,
            )
