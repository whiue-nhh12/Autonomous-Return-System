"""Return submission node; persistence integration is intentionally pending."""

from agent.orchestration.state import ManagerGraphState
from agent.core.state.shared import ChildAgentResult


def submit_return(state: ManagerGraphState) -> dict:
    return {
        "child_results": {
            "submit_return": ChildAgentResult(
                status="not_configured",
                message="Return submission is not configured.",
                error_code="RETURN_SUBMISSION_NOT_CONFIGURED",
            )
        }
    }
