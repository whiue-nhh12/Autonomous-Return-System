"""Adapter node for the configured policy worker."""

import logging
from typing import Any

from agent.agents.policy.graph import PolicyAgentGraph
from agent.orchestration.state import ManagerGraphState
from agent.core.state.shared import ChildAgentResult
from agent.agents.policy.state.result import PolicyAgentRequest

_LOGGER = logging.getLogger(__name__)


def create_policy_node(
    policy_agent: PolicyAgentGraph | None,
    *,
    result_key: str,
    state_key: str,
):
    def run_policy(state: ManagerGraphState) -> dict[str, Any]:
        shared = state["shared"]
        if policy_agent is None:
            result = ChildAgentResult(
                status="not_configured",
                message="Policy lookup is not configured.",
                error_code="POLICY_AGENT_NOT_CONFIGURED",
            )
            return {"child_results": {result_key: result}}

        try:
            child_result = policy_agent.invoke(
                PolicyAgentRequest(
                    message=shared.input.message,
                    language=shared.input.language,
                    request_id=str(shared.context.request_id),
                )
            )
        except Exception:
            _LOGGER.exception(
                "Manager child agent failed",
                extra={"request_id": str(shared.context.request_id), "child_agent": result_key},
            )
            result = ChildAgentResult(
                status="failed",
                message="Policy lookup failed. Please try again later.",
                error_code="POLICY_AGENT_FAILED",
            )
            return {"child_results": {result_key: result}}

        lookup_status = child_result.lookup_status
        result = ChildAgentResult(
            status=lookup_status,
            message=child_result.message,
            error_code="POLICY_LOOKUP_FAILED" if lookup_status == "failed" else None,
            data={
                "evidence_count": len(child_result.evidence),
                "citations": [
                    item.model_dump(mode="json") for item in child_result.evidence
                ],
                "lookup_status": lookup_status,
            },
        )
        return {state_key: child_result, "child_results": {result_key: result}}

    return run_policy
