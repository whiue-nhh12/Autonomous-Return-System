"""Clarification and response-assembly nodes."""

from agent.orchestration.state import ManagerGraphState
from agent.core.state.shared import (
    AgentSharedState,
    BusinessValidationState,
    ChildAgentResult,
    DecisionProposalState,
    PolicyLookupState,
    ReturnIntent,
    StateTransition,
)


def clarify(state: ManagerGraphState) -> dict[str, AgentSharedState]:
    shared = state["shared"]
    question = (
        "Are you asking about a policy, an order, return eligibility, or submitting a return?"
        if shared.input.language.casefold().startswith("en")
        else "Bạn muốn hỏi về chính sách, đơn hàng, điều kiện trả hàng hay gửi yêu cầu trả hàng?"
    )
    shared.clarification.needed = True
    shared.clarification.awaiting_user = True
    shared.clarification.question = question
    shared.response.status = "ready"
    shared.response.message = question
    shared.transitions.append(StateTransition(component="clarify", to_status="awaiting_user"))
    return {"shared": shared}


def finalize_response(state: ManagerGraphState) -> dict[str, AgentSharedState]:
    shared = state["shared"]
    results = state.get("child_results", {})
    shared.orchestration.child_results = results
    for child_name, child_result in sorted(results.items()):
        shared.transitions.append(
            StateTransition(
                component=f"child:{child_name}",
                to_status=child_result.status,
                detail=child_result.error_code,
            )
        )

    if results:
        statuses = {item.status for item in results.values()}
        shared.orchestration.status = (
            "completed"
            if statuses <= {"succeeded", "not_found", "needs_clarification"}
            else "partial"
        )
        intent = shared.classification.intent
        if intent is ReturnIntent.POLICY_QUESTION:
            child_result = state.get("policy_worker_state")
            if child_result is not None:
                shared.policy_lookup = PolicyLookupState(
                    status=child_result.lookup_status,
                    evidence=child_result.evidence,
                    error=child_result.error,
                )
                shared.response.status = (
                    "failed" if child_result.lookup_status == "failed" else "ready"
                )
                shared.response.message = child_result.message
                if child_result.lookup_status == "not_found":
                    shared.clarification.needed = True
                    shared.clarification.awaiting_user = True
                    shared.clarification.question = child_result.message
            else:
                shared.response.status = "ready"
                shared.response.message = results["policy"].message
        elif intent is ReturnIntent.ORDER_STATUS:
            order_result = state.get("order_worker_state")
            if order_result is not None:
                shared.order_lookup.status = order_result.status
                shared.order_lookup.orders = order_result.orders
                shared.order_lookup.data = order_result.orders[0] if order_result.orders else None
                shared.order_lookup.has_more = order_result.has_more
                shared.order_lookup.error = (
                    order_result.message if order_result.status == "failed" else None
                )
                if order_result.status == "needs_clarification":
                    shared.clarification.needed = True
                    shared.clarification.awaiting_user = True
                    shared.clarification.question = order_result.message
                shared.response.status = (
                    "failed" if order_result.status == "failed" else "ready"
                )
                shared.response.message = order_result.message
            else:
                child_result = results["order"]
                lookup_failed = child_result.status in {"failed", "not_configured"}
                shared.order_lookup.status = (
                    "failed" if lookup_failed else "not_found"
                )
                shared.order_lookup.error = child_result.error_code
                shared.response.status = (
                    "failed" if lookup_failed else "ready"
                )
                shared.response.message = child_result.message
        elif intent is ReturnIntent.RETURN_ELIGIBILITY:
            policy_result = state.get("eligibility_policy_state")
            if policy_result is not None:
                shared.policy_lookup = PolicyLookupState(
                    status=policy_result.lookup_status,
                    evidence=policy_result.evidence,
                    error=policy_result.error,
                )
            order_result = state.get("eligibility_order_worker_state")
            if order_result is not None:
                shared.order_lookup.status = order_result.status
                shared.order_lookup.orders = order_result.orders
                shared.order_lookup.data = order_result.orders[0] if order_result.orders else None
                shared.order_lookup.has_more = order_result.has_more
            decision = state.get("eligibility_decision_state")
            if decision is None:
                shared.decision_proposal = DecisionProposalState(
                    status="failed", reason="Return decision worker returned no result."
                )
                shared.response.status = "failed"
                shared.response.message = results.get("eligibility", ChildAgentResult(
                    status="failed", message="Return eligibility could not be evaluated."
                )).message
            else:
                shared.decision_proposal = DecisionProposalState(
                    status=decision.status,
                    outcome=decision.outcome,
                    reason=decision.message,
                    evidence_refs=decision.evidence_refs,
                    reason_codes=decision.reason_codes,
                    rule_set_version=decision.rule_set_version,
                    evaluated_at=decision.evaluated_at,
                )
                if decision.outcome == "approve":
                    validation_status = "valid"
                elif decision.outcome == "reject":
                    validation_status = "invalid"
                elif decision.outcome == "manual_review":
                    validation_status = "needs_review"
                else:
                    validation_status = "not_started"
                shared.business_validation = BusinessValidationState(
                    status=validation_status,
                    checks={check.code: check.status for check in decision.checks},
                    reasons=decision.reason_codes,
                )
                shared.response.status = "failed" if decision.status == "failed" else "ready"
                shared.response.message = decision.message
                if decision.status == "needs_clarification":
                    shared.clarification.needed = True
                    shared.clarification.awaiting_user = True
                    shared.clarification.question = decision.message
                    shared.clarification.missing_fields = decision.missing_fields
        elif intent is ReturnIntent.SUBMIT_RETURN:
            submitted = results.get("submit_return")
            proposal_result = results.get("return_proposal")
            if state.get("return_request_id") is not None:
                shared.persistence.status = "committed"
                shared.persistence.return_request_id = state["return_request_id"]
            if proposal_result is not None:
                decision = state.get("eligibility_decision_state")
                if decision is not None:
                    shared.decision_proposal = DecisionProposalState(
                        status=decision.status,
                        outcome=decision.outcome,
                        reason=decision.message,
                        evidence_refs=decision.evidence_refs,
                        reason_codes=decision.reason_codes,
                        rule_set_version=decision.rule_set_version,
                        evaluated_at=decision.evaluated_at,
                    )
                    shared.business_validation = BusinessValidationState(
                        status=(
                            "valid" if decision.outcome == "approve"
                            else "invalid" if decision.outcome == "reject"
                            else "needs_review" if decision.outcome == "manual_review"
                            else "not_started"
                        ),
                        checks={check.code: check.status for check in decision.checks},
                        reasons=decision.reason_codes,
                    )
                shared.response.status = (
                    "failed" if proposal_result.status == "failed" else "ready"
                )
                shared.response.message = proposal_result.message
                if decision is not None and decision.status == "needs_clarification":
                    shared.clarification.needed = True
                    shared.clarification.awaiting_user = True
                    shared.clarification.question = decision.message
                    shared.clarification.missing_fields = decision.missing_fields
            elif submitted is not None:
                shared.response.status = (
                    "failed" if submitted.status == "failed" else "ready"
                )
                shared.response.message = submitted.message
                if submitted.status == "needs_clarification":
                    shared.clarification.needed = True
                    shared.clarification.awaiting_user = True
                    shared.clarification.question = submitted.message
                    shared.clarification.missing_fields = submitted.data.get(
                        "missing_fields", []
                    )
            else:
                shared.response.status = "failed"
                shared.response.message = "The return request could not be submitted."
    else:
        shared.orchestration.status = "completed"

    shared.transitions.append(
        StateTransition(component="finalize", to_status=shared.orchestration.status)
    )
    return {"shared": shared}
