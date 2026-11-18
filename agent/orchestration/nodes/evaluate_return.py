"""Adapter that combines order and policy worker results for return decisions."""

from datetime import datetime, timezone
import logging

from agent.agents.return_decision.claims import extract_return_claim
from agent.agents.return_decision.graph import ReturnDecisionAgentGraph
from agent.agents.return_decision.state.result import (
    ReturnClaimFacts,
    ReturnDecisionRequest,
    ReturnDecisionResult,
)
from agent.core.state.shared import ChildAgentResult
from agent.orchestration.state import ManagerGraphState
from backend.db.session import get_session_factory
from service.return_decision_proposals import ReturnDecisionProposalService
from service.return_fingerprint import serialize_evaluated_order_facts

_LOGGER = logging.getLogger(__name__)


def create_return_decision_node(
    decision_agent: ReturnDecisionAgentGraph | None = None,
):
    agent = decision_agent or ReturnDecisionAgentGraph()

    def run(state: ManagerGraphState) -> dict:
        shared = state["shared"]
        order_result = state.get("eligibility_order_worker_state")
        policy_result = state.get("eligibility_policy_state")
        evaluated_claim = state.get("return_submission_claim") or ReturnClaimFacts()
        evaluated_order_facts: list[dict] = []
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
                transcript = "\n".join(
                    [*shared.context.prior_messages, shared.input.message]
                )
                raw_claim = shared.classification.entities.get("return_claim", {})
                try:
                    seed_claim = ReturnClaimFacts.model_validate(raw_claim)
                except (TypeError, ValueError):
                    seed_claim = None
                decision_request = ReturnDecisionRequest(
                    request_id=shared.context.request_id,
                    user_id=shared.context.user_id,
                    message=transcript,
                    language=shared.input.language,
                    evaluated_at=datetime.now(timezone.utc),
                    claim=extract_return_claim(
                        shared.input.message,
                        order_result.orders,
                        seed=seed_claim,
                        item_context=transcript,
                    ),
                    orders=order_result.orders,
                    policy_evidence=policy_result.evidence if policy_result else [],
                    policy_lookup_status=policy_result.lookup_status if policy_result else "not_started",
                )
                evaluated_claim = decision_request.claim
                evaluated_order_facts = [
                    serialize_evaluated_order_facts(order)
                    for order in decision_request.orders
                ]
                result = agent.invoke(decision_request)
            except Exception:
                _LOGGER.exception("Return decision evaluation failed", extra={"request_id": str(shared.context.request_id)})
                result = _failed(shared.input.language, "RETURN_DECISION_FAILED")

        proposal_id = None
        return_request_id = state.get("return_request_id")
        if return_request_id is not None:
            try:
                proposal_id = ReturnDecisionProposalService(get_session_factory()).save(
                    return_request_id=return_request_id,
                    evaluation_request_id=shared.context.request_id,
                    result=result,
                    claim=evaluated_claim,
                    policy_evidence=policy_result.evidence if policy_result else [],
                    evaluated_data_fingerprint=state.get(
                        "return_submission_fingerprint", ""
                    ),
                    classifier_version=shared.classification.classifier_version,
                    evaluated_order_facts=evaluated_order_facts,
                )
            except Exception:
                _LOGGER.exception(
                    "Return decision proposal persistence failed",
                    extra={"request_id": str(shared.context.request_id)},
                )
                result = _failed(shared.input.language, "DECISION_PROPOSAL_PERSIST_FAILED")

        status = "succeeded" if result.status == "completed" else result.status
        child_key = "return_proposal" if return_request_id is not None else "eligibility"
        if return_request_id is not None and proposal_id is not None:
            english = shared.input.language.casefold().startswith("en")
            return_number = state.get("return_request_number", str(return_request_id))
            if result.status == "needs_clarification":
                proposal_message = (
                    f"Return request {return_number} was saved, but more information is needed: {result.message} No approval or refund has been created."
                    if english
                    else f"Yêu cầu trả {return_number} đã được lưu nhưng cần bổ sung thông tin: {result.message} Chưa có chấp thuận hoặc hoàn tiền nào được tạo."
                )
            else:
                proposal_message = (
                    f"Return request {return_number} was submitted. The decision proposal is saved for staff review; it is not an approval and no refund has been created."
                    if english
                    else f"Yêu cầu trả {return_number} đã được ghi nhận. Đề xuất đã lưu để nhân viên xem xét; đây chưa phải là chấp thuận và chưa tạo hoàn tiền."
                )
        elif return_request_id is not None:
            proposal_message = (
                "The return request was saved, but its decision proposal could not be stored. No approval or refund was created."
                if shared.input.language.casefold().startswith("en")
                else "Yêu cầu trả đã được lưu nhưng chưa lưu được đề xuất đánh giá. Chưa có chấp thuận hoặc hoàn tiền nào được tạo."
            )
        else:
            proposal_message = result.message
        child = ChildAgentResult(
            status=status,
            message=proposal_message,
            error_code=result.error_code,
            data={
                **result.model_dump(mode="json"),
                **(
                    {
                        "proposal_id": str(proposal_id),
                        "return_request_id": str(return_request_id),
                        "outcome": result.outcome,
                    }
                    if proposal_id is not None
                    else {}
                ),
            },
        )
        return {
            "eligibility_decision_state": result,
            **({"return_proposal_id": proposal_id} if proposal_id is not None else {}),
            "child_results": {child_key: child},
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
