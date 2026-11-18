"""Admin review workflow for stored Decision Agent proposals."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from backend.models import (
    Order,
    OrderItem,
    ReturnDecisionProposal,
    ReturnItem,
    ReturnRequest,
    ReturnStatusHistory,
    User,
)
from backend.models.enums import (
    DecisionSource,
    ReturnStatus,
    StatusChangeSource,
    UserRole,
    UserStatus,
)
from backend.schemas.return_proposal import (
    ReturnProposalDecisionResponse,
    ReturnProposalSummary,
)
from service.return_fingerprint import fingerprint_order_rows
from service.policy_versions import PolicyVersionUnavailable, PublishedPolicyVersionSource


class ReturnApprovalError(Exception):
    """Base class for safe, expected proposal-review failures."""


class ReturnApprovalForbidden(ReturnApprovalError):
    pass


class ReturnProposalNotFound(ReturnApprovalError):
    pass


class InvalidProposalState(ReturnApprovalError):
    pass


class StaleReturnProposal(ReturnApprovalError):
    pass


class ReturnDecisionIdempotencyConflict(ReturnApprovalError):
    pass


class ReturnApprovalService:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        policy_version_source: PublishedPolicyVersionSource | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._policy_version_source = policy_version_source or PublishedPolicyVersionSource()

    def decide(
        self,
        *,
        proposal_id: UUID,
        actor_id: UUID,
        action: str,
        idempotency_key: str,
        note: str,
    ) -> ReturnProposalDecisionResponse:
        key = idempotency_key.strip()
        clean_note = note.strip()
        if not 16 <= len(key) <= 128:
            raise ReturnDecisionIdempotencyConflict(
                "Idempotency-Key must contain 16 to 128 characters."
            )
        if action not in {"approve", "reject"}:
            raise InvalidProposalState("Action must be approve or reject.")
        if not clean_note:
            raise ValueError("Decision note must contain non-whitespace text.")

        try:
            with self._session_factory.begin() as session:
                self._require_active_admin(session, actor_id)
                replay = session.scalar(
                    select(ReturnDecisionProposal).where(
                        ReturnDecisionProposal.decision_idempotency_key == key
                    )
                )
                if replay is not None:
                    return self._replay_or_conflict(
                        replay,
                        proposal_id=proposal_id,
                        action=action,
                        note=clean_note,
                    )

                proposal = session.scalar(
                    select(ReturnDecisionProposal)
                    .where(ReturnDecisionProposal.id == proposal_id)
                    .with_for_update()
                )
                if proposal is None:
                    raise ReturnProposalNotFound("Return proposal was not found.")
                if proposal.decision_action is not None:
                    if proposal.decision_idempotency_key == key:
                        return self._replay_or_conflict(
                            proposal,
                            proposal_id=proposal_id,
                            action=action,
                            note=clean_note,
                        )
                    raise InvalidProposalState("This proposal already has a final admin decision.")
                if proposal.status != "completed":
                    raise InvalidProposalState(
                        "Only completed proposals can receive an admin decision."
                    )
                if action == "approve" and proposal.outcome not in {"approve", "manual_review"}:
                    raise InvalidProposalState(
                        "This proposal does not support approval; reevaluate or reject it."
                    )
                if action == "reject" and proposal.outcome not in {"reject", "manual_review"}:
                    raise InvalidProposalState(
                        "This proposal does not support rejection; reevaluate or approve it."
                    )

                request = session.scalar(
                    select(ReturnRequest)
                    .where(ReturnRequest.id == proposal.return_request_id)
                    .with_for_update()
                )
                if request is None:
                    raise ReturnProposalNotFound("Return request was not found.")
                if request.status not in {ReturnStatus.SUBMITTED, ReturnStatus.REVIEWING}:
                    raise InvalidProposalState(
                        f"Cannot decide a return request in '{request.status.value}' state."
                    )
                order = session.scalar(
                    select(Order).where(Order.id == request.order_id).with_for_update()
                )
                if order is None or order.user_id != request.user_id:
                    raise InvalidProposalState("Return request order ownership is inconsistent.")

                current_fingerprint = fingerprint_order_rows(
                    session, order, exclude_return_request_id=request.id
                )
                if current_fingerprint != proposal.evaluated_data_fingerprint:
                    raise StaleReturnProposal(
                        "Order or return data changed after evaluation. Create a new proposal."
                    )
                self._require_current_policy_versions(proposal)

                return_items = list(
                    session.scalars(
                        select(ReturnItem)
                        .where(ReturnItem.return_request_id == request.id)
                        .order_by(ReturnItem.order_item_id)
                        .with_for_update()
                    )
                )
                if not return_items:
                    raise InvalidProposalState("Cannot decide a return request without items.")
                order_item_ids = {item.order_item_id for item in return_items}
                order_items = list(
                    session.scalars(
                        select(OrderItem)
                        .where(OrderItem.id.in_(order_item_ids))
                        .order_by(OrderItem.id)
                        .with_for_update()
                    )
                )
                by_id = {item.id: item for item in order_items}
                if len(by_id) != len(order_item_ids):
                    raise InvalidProposalState("A return item references a missing order item.")
                for return_item in return_items:
                    order_item = by_id[return_item.order_item_id]
                    if order_item.order_id != order.id:
                        raise InvalidProposalState(
                            "A return item does not belong to the request's order."
                        )
                    if return_item.requested_quantity > order_item.quantity:
                        raise InvalidProposalState(
                            "A return item quantity exceeds the quantity purchased."
                        )

                previous_status = request.status
                if action == "approve":
                    self._validate_reserved_quantities(session, order, return_items, by_id)
                    for return_item in return_items:
                        return_item.approved_quantity = return_item.requested_quantity
                    request.status = ReturnStatus.APPROVED
                else:
                    request.status = ReturnStatus.REJECTED

                decided_at = datetime.now(timezone.utc)
                request.decision_source = DecisionSource.MANUAL
                request.reviewed_by = actor_id
                request.decision_note = clean_note
                request.policy_version_ref = proposal.policy_version_ref
                request.decided_at = decided_at
                proposal.decision_action = action
                proposal.decision_note = clean_note
                proposal.decision_idempotency_key = key
                proposal.decided_by = actor_id
                proposal.decided_at = decided_at
                session.add(
                    ReturnStatusHistory(
                        return_request_id=request.id,
                        from_status=previous_status,
                        to_status=request.status,
                        changed_by_user_id=actor_id,
                        note=clean_note,
                        source=StatusChangeSource.STAFF,
                    )
                )
                session.flush()
                return ReturnProposalDecisionResponse(
                    proposal_id=proposal.id,
                    return_request_id=request.id,
                    action=action,
                    return_status=request.status.value,
                )
        except IntegrityError as exc:
            if not _is_decision_key_violation(exc):
                raise
            replay = self._find_replay(
                key,
                proposal_id=proposal_id,
                action=action,
                note=clean_note,
            )
            if replay is not None:
                return replay
            raise ReturnDecisionIdempotencyConflict(
                "Idempotency-Key has already been used for another decision."
            ) from exc

    def list_for_request(
        self, *, return_request_id: UUID, actor_id: UUID
    ) -> list[ReturnProposalSummary]:
        with self._session_factory() as session:
            self._require_active_admin(session, actor_id)
            exists = session.scalar(
                select(ReturnRequest.id).where(ReturnRequest.id == return_request_id)
            )
            if exists is None:
                raise ReturnProposalNotFound("Return request was not found.")
            rows = session.scalars(
                select(ReturnDecisionProposal)
                .where(ReturnDecisionProposal.return_request_id == return_request_id)
                .order_by(ReturnDecisionProposal.evaluated_at.desc(), ReturnDecisionProposal.id)
            )
            return [self._summary(row) for row in rows]

    def _require_current_policy_versions(
        self, proposal: ReturnDecisionProposal
    ) -> None:
        expected = proposal.policy_versions or {}
        if not expected:
            return
        try:
            current = self._policy_version_source.current_versions(list(expected))
        except PolicyVersionUnavailable as exc:
            raise StaleReturnProposal(
                "Current policy versions are unavailable. Retry after policy storage recovers."
            ) from exc
        if any(current.get(doc_id) != version for doc_id, version in expected.items()):
            raise StaleReturnProposal(
                "A cited policy version is no longer current. Create a new proposal."
            )

    @staticmethod
    def _validate_reserved_quantities(
        session: Session,
        order: Order,
        return_items: list[ReturnItem],
        order_items_by_id: dict[UUID, OrderItem],
    ) -> None:
        request_items = list(
            session.scalars(
                select(ReturnItem)
                .join(ReturnRequest, ReturnRequest.id == ReturnItem.return_request_id)
                .where(
                    ReturnRequest.order_id == order.id,
                    ReturnItem.order_item_id.in_(order_items_by_id),
                )
                .order_by(ReturnRequest.id, ReturnItem.order_item_id)
            )
        )
        requests = {
            request.id: request
            for request in session.scalars(
                select(ReturnRequest)
                .where(ReturnRequest.order_id == order.id)
            )
        }
        reserved: dict[UUID, int] = {}
        active = {
            ReturnStatus.SUBMITTED,
            ReturnStatus.REVIEWING,
            ReturnStatus.APPROVED,
            ReturnStatus.ITEM_RECEIVED,
        }
        for item in request_items:
            parent = requests[item.return_request_id]
            quantity = 0
            if parent.status in active:
                quantity = (
                    item.approved_quantity
                    if item.approved_quantity is not None
                    else item.requested_quantity
                )
            elif parent.status is ReturnStatus.COMPLETED:
                quantity = item.approved_quantity or 0
            reserved[item.order_item_id] = reserved.get(item.order_item_id, 0) + quantity
        for order_item_id, quantity in reserved.items():
            if quantity > order_items_by_id[order_item_id].quantity:
                raise InvalidProposalState(
                    "Active return requests exceed the quantity purchased."
                )

    @staticmethod
    def _replay_or_conflict(
        proposal: ReturnDecisionProposal,
        *,
        proposal_id: UUID,
        action: str,
        note: str,
    ) -> ReturnProposalDecisionResponse:
        if (
            proposal.id != proposal_id
            or proposal.decision_action != action
            or proposal.decision_note != note
        ):
            raise ReturnDecisionIdempotencyConflict(
                "Idempotency-Key was already used with different decision details."
            )
        return ReturnProposalDecisionResponse(
            proposal_id=proposal.id,
            return_request_id=proposal.return_request_id,
            action=action,
            return_status=proposal.return_request.status.value,
            replayed=True,
        )

    def _find_replay(
        self, key: str, *, proposal_id: UUID, action: str, note: str
    ) -> ReturnProposalDecisionResponse | None:
        with self._session_factory() as session:
            proposal = session.scalar(
                select(ReturnDecisionProposal).where(
                    ReturnDecisionProposal.decision_idempotency_key == key
                )
            )
            if proposal is None:
                return None
            return self._replay_or_conflict(
                proposal,
                proposal_id=proposal_id,
                action=action,
                note=note,
            )

    @staticmethod
    def _summary(proposal: ReturnDecisionProposal) -> ReturnProposalSummary:
        return ReturnProposalSummary(
            proposal_id=proposal.id,
            return_request_id=proposal.return_request_id,
            status=proposal.status,
            outcome=proposal.outcome,
            assessment_message=proposal.assessment_message,
            checks=proposal.checks,
            reason_codes=proposal.reason_codes,
            missing_fields=proposal.missing_fields,
            policy_version_ref=proposal.policy_version_ref,
            policy_versions=proposal.policy_versions,
            evidence_refs=proposal.evidence_refs,
            claim_facts=proposal.claim_facts,
            evaluated_order_facts=proposal.evaluated_order_facts,
            evaluated_at=proposal.evaluated_at,
            rule_set_version=proposal.rule_set_version,
            classifier_version=proposal.classifier_version,
            decision_engine_version=proposal.decision_engine_version,
            decision_action=proposal.decision_action,
            decision_note=proposal.decision_note,
            decided_by=proposal.decided_by,
            decided_at=proposal.decided_at,
        )


def _is_decision_key_violation(error: IntegrityError) -> bool:
    detail = str(error.orig).casefold()
    return (
        "uq_return_proposals_idempotency_key" in detail
        or "decision_idempotency_key" in detail
    )
