"""Re-evaluate an existing submitted request without creating another request."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from agent.agents.order.graph import OrderAgentGraph
from agent.agents.order.state.result import OrderAgentRequest
from agent.agents.policy.graph import PolicyAgentGraph
from agent.agents.policy.state.result import PolicyAgentRequest
from agent.agents.return_decision.graph import ReturnDecisionAgentGraph
from agent.agents.return_decision.state.result import (
    ReturnClaimFacts,
    ReturnDecisionRequest,
    ReturnDecisionResult,
)
from agent.core.state.shared import OrderLookupData
from backend.models import Order, OrderItem, ReturnItem, ReturnRequest, User
from backend.models.enums import ReturnStatus, UserRole, UserStatus
from backend.schemas.return_reevaluation import ReturnReevaluationResponse
from service.return_decision_proposals import ReturnDecisionProposalService
from service.return_fingerprint import (
    fingerprint_order_lookup,
    fingerprint_order_rows,
    serialize_evaluated_order_facts,
)


_LOGGER = logging.getLogger(__name__)


class ReturnReevaluationError(Exception):
    pass


class ReturnReevaluationForbidden(ReturnReevaluationError):
    pass


class ReturnReevaluationNotFound(ReturnReevaluationError):
    pass


class ReturnReevaluationConflict(ReturnReevaluationError):
    pass


class ReturnReevaluationService:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        order_agent: OrderAgentGraph,
        policy_agent: PolicyAgentGraph | None,
        decision_agent: ReturnDecisionAgentGraph,
        proposal_service: ReturnDecisionProposalService | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._order_agent = order_agent
        self._policy_agent = policy_agent
        self._decision_agent = decision_agent
        self._proposal_service = proposal_service or ReturnDecisionProposalService(
            session_factory
        )

    def reevaluate(
        self,
        *,
        return_request_id: UUID,
        actor_id: UUID,
        idempotency_key: str,
        language: str = "vi",
    ) -> ReturnReevaluationResponse:
        key = idempotency_key.strip()
        if not 16 <= len(key) <= 128:
            raise ReturnReevaluationConflict("Idempotency-Key must contain 16 to 128 characters.")
        evaluation_request_id = uuid5(
            NAMESPACE_URL, f"return-reevaluation:{return_request_id}:{key}"
        )
        with self._session_factory() as session:
            self._require_active_admin(session, actor_id)
        existing = self._proposal_service.get_evaluation(evaluation_request_id)
        if existing is not None:
            proposal_id, existing_request_id, existing_status, existing_outcome = existing
            if existing_request_id != return_request_id:
                raise ReturnReevaluationConflict("Idempotency-Key is already in use.")
            return ReturnReevaluationResponse(
                proposal_id=proposal_id,
                return_request_id=return_request_id,
                status=existing_status,
                outcome=existing_outcome,
                replayed=True,
            )
        request, order, return_items, order_items = self._load_request(
            return_request_id, actor_id
        )
        base_fingerprint = self._current_fingerprint(order.id, request.id)
        order_result = self._lookup_order(
            request.user_id,
            evaluation_request_id,
            order.order_number,
            language,
        )

        policy_evidence = []
        policy_lookup_status = "not_started"
        if self._policy_agent is not None:
            try:
                policy_result = self._policy_agent.invoke(
                    PolicyAgentRequest(
                        message=self._policy_message(order, request),
                        language=language,
                        request_id=str(evaluation_request_id),
                    )
                )
                policy_evidence = policy_result.evidence
                policy_lookup_status = policy_result.lookup_status
            except Exception:
                _LOGGER.exception(
                    "Return reevaluation policy lookup failed",
                    extra={"return_request_id": str(request.id)},
                )
                policy_lookup_status = "failed"

        selected_order = next(
            (
                item
                for item in order_result.orders
                if item.order_id == order.id
            ),
            None,
        )
        if selected_order is not None:
            selected_order = selected_order.model_copy(
                update={
                    "returns": [
                        history
                        for history in selected_order.returns
                        if history.return_number != request.return_number
                    ]
                },
                deep=True,
            )
            evaluated_fingerprint = fingerprint_order_lookup(selected_order)
        else:
            evaluated_fingerprint = base_fingerprint

        claim = self._claim(order, request, return_items, order_items)
        result = self._evaluate(
            request=request,
            order=selected_order,
            order_lookup_status=order_result.status,
            claim=claim,
            policy_evidence=policy_evidence,
            policy_lookup_status=policy_lookup_status,
            evaluation_request_id=evaluation_request_id,
            language=language,
        )
        try:
            proposal_id = self._proposal_service.save(
                return_request_id=request.id,
                evaluation_request_id=evaluation_request_id,
                result=result,
                claim=claim,
                policy_evidence=policy_evidence,
                evaluated_data_fingerprint=evaluated_fingerprint,
                classifier_version="admin-reevaluation-v1",
                evaluated_order_facts=(
                    [serialize_evaluated_order_facts(selected_order)]
                    if selected_order is not None
                    else []
                ),
            )
        except Exception as exc:
            raise ReturnReevaluationConflict(
                "Order or policy data changed during reevaluation. Retry with a new Idempotency-Key."
            ) from exc

        return ReturnReevaluationResponse(
            proposal_id=proposal_id,
            return_request_id=request.id,
            status=result.status,
            outcome=result.outcome,
        )

    def _load_request(self, return_request_id: UUID, actor_id: UUID):
        with self._session_factory() as session:
            self._require_active_admin(session, actor_id)
            request = session.scalar(
                select(ReturnRequest).where(ReturnRequest.id == return_request_id)
            )
            if request is None:
                raise ReturnReevaluationNotFound("Return request was not found.")
            if request.status not in {ReturnStatus.SUBMITTED, ReturnStatus.REVIEWING}:
                raise ReturnReevaluationConflict(
                    f"Cannot reevaluate a return request in '{request.status.value}' state."
                )
            order = session.scalar(select(Order).where(Order.id == request.order_id))
            if order is None or order.user_id != request.user_id:
                raise ReturnReevaluationConflict("Return request order ownership is inconsistent.")
            return_items = list(
                session.scalars(
                    select(ReturnItem)
                    .where(ReturnItem.return_request_id == request.id)
                    .order_by(ReturnItem.order_item_id)
                )
            )
            order_item_ids = {item.order_item_id for item in return_items}
            order_items = list(
                session.scalars(
                    select(OrderItem)
                    .where(OrderItem.id.in_(order_item_ids))
                    .order_by(OrderItem.id)
                )
            ) if order_item_ids else []
            if not return_items or len(order_items) != len(order_item_ids):
                raise ReturnReevaluationConflict("Return request item data is incomplete.")
            return request, order, return_items, order_items

    def _current_fingerprint(self, order_id: UUID, return_request_id: UUID) -> str:
        with self._session_factory() as session:
            order = session.scalar(select(Order).where(Order.id == order_id))
            if order is None:
                raise ReturnReevaluationNotFound("Order was not found.")
            return fingerprint_order_rows(
                session,
                order,
                exclude_return_request_id=return_request_id,
            )

    def _lookup_order(
        self,
        user_id: UUID,
        request_id: UUID,
        order_number: str,
        language: str,
    ):
        try:
            return self._order_agent.invoke(
                OrderAgentRequest(
                    user_id=user_id,
                    request_id=request_id,
                    message=f"Order status and return eligibility for {order_number}.",
                    language=language,
                    order_number=order_number,
                )
            )
        except Exception:
            _LOGGER.exception("Return reevaluation order lookup failed")
            return _failed_order_result(language)

    @staticmethod
    def _claim(
        order: Order,
        request: ReturnRequest,
        return_items: list[ReturnItem],
        order_items: list[OrderItem],
    ) -> ReturnClaimFacts:
        products = {item.id: item for item in order_items}
        if len(return_items) != 1:
            return ReturnClaimFacts(
                order_number=order.order_number,
                reason_code=request.reason_code.value,
                return_method=request.return_method.value,
            )
        return_item = return_items[0]
        order_item = products[return_item.order_item_id]
        return ReturnClaimFacts(
            order_number=order.order_number,
            order_item_id=order_item.id,
            item_query=order_item.product_name_snapshot,
            requested_quantity=return_item.requested_quantity,
            reason_code=request.reason_code.value,
            item_condition=(
                return_item.item_condition.value
                if return_item.item_condition is not None
                else None
            ),
            return_method=request.return_method.value,
        )

    def _evaluate(
        self,
        *,
        request: ReturnRequest,
        order: OrderLookupData | None,
        order_lookup_status: str,
        claim: ReturnClaimFacts,
        policy_evidence,
        policy_lookup_status: str,
        evaluation_request_id: UUID,
        language: str,
    ) -> ReturnDecisionResult:
        now = datetime.now(timezone.utc)
        if order_lookup_status == "failed":
            return _failed_decision(language, "ORDER_LOOKUP_FAILED", now)
        if order_lookup_status == "needs_clarification":
            return ReturnDecisionResult(
                status="needs_clarification",
                message=(
                    "Order facts need clarification before this proposal can be reviewed."
                    if language.casefold().startswith("en")
                    else "Cần làm rõ dữ liệu đơn hàng trước khi xem xét đề xuất này."
                ),
                reason_codes=["ORDER_LOOKUP_NEEDS_CLARIFICATION"],
                missing_fields=["order_number"],
                evaluated_at=now,
            )
        if order is None:
            if order_lookup_status == "not_found":
                english = language.casefold().startswith("en")
                return ReturnDecisionResult(
                    status="completed",
                    outcome="manual_review",
                    message=(
                        "Order facts were unavailable, so staff review is required."
                        if english
                        else "Không truy xuất được dữ liệu đơn hàng nên cần nhân viên xem xét."
                    ),
                    reason_codes=["ORDER_DATA_UNAVAILABLE"],
                    evaluated_at=now,
                )
            return _failed_decision(language, "ORDER_DATA_UNAVAILABLE", now)
        return self._decision_agent.invoke(
            ReturnDecisionRequest(
                request_id=evaluation_request_id,
                user_id=request.user_id,
                message=f"Return request {request.return_number} for order {order.order_number}",
                language=language,
                evaluated_at=now,
                claim=claim,
                orders=[order],
                policy_evidence=policy_evidence,
                policy_lookup_status=policy_lookup_status,
            )
        )

    @staticmethod
    def _policy_message(order: Order, request: ReturnRequest) -> str:
        return (
            f"Return policy eligibility for order status {order.status.value}, "
            f"reason {request.reason_code.value}, and method {request.return_method.value}."
        )

    @staticmethod
    def _require_active_admin(session: Session, actor_id: UUID) -> None:
        actor = session.scalar(
            select(User)
            .where(
                User.id == actor_id,
                User.role == UserRole.ADMIN,
                User.status == UserStatus.ACTIVE,
            )
        )
        if actor is None:
            raise ReturnReevaluationForbidden("An active admin account is required.")


def _failed_order_result(language: str):
    from agent.agents.order.state.result import OrderAgentResult

    english = language.casefold().startswith("en")
    return OrderAgentResult(
        status="failed",
        message=(
            "Order lookup failed during reevaluation."
            if english
            else "Không tra cứu được đơn hàng để đánh giá lại."
        ),
        error_code="RETURN_REEVALUATION_ORDER_LOOKUP_FAILED",
    )


def _failed_decision(language: str, error_code: str, evaluated_at: datetime):
    english = language.casefold().startswith("en")
    return ReturnDecisionResult(
        status="failed",
        message=(
            "Return eligibility could not be reevaluated."
            if english
            else "Chưa thể đánh giá lại yêu cầu trả hàng."
        ),
        reason_codes=[error_code],
        evaluated_at=evaluated_at,
        error_code=error_code,
    )
