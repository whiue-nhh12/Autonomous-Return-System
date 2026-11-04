"""Request and response schemas for manager-agent API calls."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from agent.core.state.shared import AgentSharedState, ChildAgentResult, OrderLookupState


class ManagerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=8000)
    language: str = Field(default="vi", min_length=2, max_length=10)
    conversation_id: str | None = Field(default=None, max_length=128)


class ManagerResponse(BaseModel):
    request_id: UUID
    conversation_id: str | None
    intent: str | None
    status: str
    message: str | None
    child_results: dict[str, ChildAgentResult]
    order_lookup: OrderLookupState = Field(default_factory=OrderLookupState)

    @classmethod
    def from_state(cls, state: AgentSharedState) -> "ManagerResponse":
        return cls(
            request_id=state.context.request_id,
            conversation_id=state.context.conversation_id,
            intent=state.classification.intent.value if state.classification.intent else None,
            status=state.orchestration.status,
            message=state.response.message,
            child_results=state.orchestration.child_results,
            order_lookup=state.order_lookup,
        )
