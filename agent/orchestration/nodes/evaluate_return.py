"""Adapter that combines order and policy worker results for return decisions."""

from datetime import datetime, timezone
import logging

from agent.agents.return_decision.claims import extract_return_claim
from agent.agents.return_decision.graph import ReturnDecisionAgentGraph
from agent.agents.return_decision.state.result import ReturnDecisionRequest, ReturnDecisionResult
from agent.core.state.shared import ChildAgentResult
from agent.orchestration.state import ManagerGraphState

_LOGGER = logging.getLogger(__name__)


def create_return_decision_node(decision_agent: ReturnDecisionAgentGraph | None = None):
    agent = decision_agent or ReturnDecisionAgentGraph()

    def run(state: ManagerGraphState) -> dict:
        shared = state["shared"]
        order_result = state.get("eligibility_order_worker_state")
        policy_result = state.get("eligibility_policy_state")
        if order_result is None:
            result = _failed(shared.input.language, "ORDER_RESULT_UNAVAILABLE")
        elif order_result.status == "failed":
            result = _failed(shared.input.language, "ORDER_LOOKUP_FAILED")
        elif order_result.status == "not_found":
            english = shared.input.language.casefold().startswith("en")
            result = ReturnDecisionResult(
                status="completed",
                outcome="manual_review",
                message=("I couldn't find the order, so its return eligibility cannot be determined." if english else "Không tìm thấy đơn hàng nên chưa thể xác định điều kiện trả."),
                reason_codes=["ORDER_NOT_FOUND"],
                evaluated_at=datetime.now(timezone.utc),
            )
        elif order_result.status == "needs_clarification":
            result = ReturnDecisionResult(
                status="needs_clarification",
                message=order_result.message,
                reason_codes=["ORDER_LOOKUP_NEEDS_CLARIFICATION"],
                evaluated_at=datetime.now(timezone.utc),
                missing_fields=["order_number"],
            )
        else:
            try:
                result = agent.invoke(ReturnDecisionRequest(
                    request_id=shared.context.request_id,
                    user_id=shared.context.user_id,
                    message=shared.input.message,
                    language=shared.input.language,
                    evaluated_at=datetime.now(timezone.utc),
                    claim=extract_return_claim(shared.input.message, order_result.orders),
                    orders=order_result.orders,
                    policy_evidence=policy_result.evidence if policy_result else [],
                    policy_lookup_status=policy_result.lookup_status if policy_result else "not_started",
                ))
            except Exception:
                _LOGGER.exception("Return decision evaluation failed", extra={"request_id": str(shared.context.request_id)})
                result = _failed(shared.input.language, "RETURN_DECISION_FAILED")

        status = "succeeded" if result.status == "completed" else result.status
        child = ChildAgentResult(
            status=status,
            message=result.message,
            error_code=result.error_code,
            data=result.model_dump(mode="json"),
        )
        return {
            "eligibility_decision_state": result,
            "child_results": {"eligibility": child},
        }

    return run


def _failed(language: str, error_code: str) -> ReturnDecisionResult:
    english = language.casefold().startswith("en")
    return ReturnDecisionResult(
        status="failed",
        message=("I couldn't evaluate return eligibility because order data is unavailable." if english else "Chưa thể đánh giá điều kiện trả hàng vì dữ liệu đơn hàng không khả dụng."),
        reason_codes=[error_code],
        evaluated_at=datetime.now(timezone.utc),
        error_code=error_code,
    )
