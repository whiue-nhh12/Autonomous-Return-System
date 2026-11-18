"""LangGraph state contract and reducer for one Manager Agent run."""

from typing import Annotated, TypedDict
from uuid import UUID

from agent.agents.policy.state.result import PolicyAgentResult
from agent.agents.order.state.result import OrderAgentResult
from agent.agents.return_decision.state.result import ReturnClaimFacts, ReturnDecisionResult
from agent.core.state.shared import AgentSharedState, ChildAgentResult


def merge_child_results(
    left: dict[str, ChildAgentResult], right: dict[str, ChildAgentResult]
) -> dict[str, ChildAgentResult]:
    overlap = left.keys() & right.keys()
    if overlap:
        raise ValueError(f"Multiple workers returned duplicate result keys: {sorted(overlap)}")
    return {**left, **right}


class ManagerGraphState(TypedDict, total=False):
    shared: AgentSharedState
    order_worker_state: OrderAgentResult
    policy_worker_state: PolicyAgentResult
    eligibility_policy_state: PolicyAgentResult
    eligibility_order_worker_state: OrderAgentResult
    eligibility_decision_state: ReturnDecisionResult
    return_request_id: UUID
    return_request_number: str
    return_submission_fingerprint: str
    return_submission_claim: ReturnClaimFacts
    return_proposal_id: UUID
    child_results: Annotated[dict[str, ChildAgentResult], merge_child_results]
