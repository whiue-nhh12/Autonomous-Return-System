"""Local request, retrieval plan, and result contracts for order lookup."""

from typing import Literal
from uuid import UUID

from pydantic import Field

from agent.core.state.shared import OrderLookupData, StateModel


class OrderAgentRequest(StateModel):
    user_id: UUID
    request_id: UUID
    message: str = Field(min_length=1)
    language: str = "vi"


class OrderQueryPlan(StateModel):
    order_number: str | None = None
    wants_history: bool = False
    wants_latest: bool = False


class OrderAgentResult(StateModel):
    status: Literal["succeeded", "not_found", "needs_clarification", "failed"]
    message: str
    orders: list[OrderLookupData] = Field(default_factory=list)
    has_more: bool = False
    error_code: str | None = None
