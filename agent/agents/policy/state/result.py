"""Public request and result contracts for the policy child agent."""

from typing import Literal

from pydantic import Field

from agent.core.state.shared import PolicyEvidence, StateModel


class PolicyAgentRequest(StateModel):
    """Immutable-in-practice input passed from orchestration to policy lookup."""

    message: str = Field(min_length=1)
    language: str = "vi"
    request_id: str | None = None


class PolicyAgentResult(StateModel):
    """Local result returned to orchestration for an explicit state merge."""

    lookup_status: Literal["succeeded", "not_found", "failed"]
    message: str
    evidence: list[PolicyEvidence] = Field(default_factory=list)
    error: str | None = None
