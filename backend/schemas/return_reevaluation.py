"""HTTP response for reevaluating a submitted return request."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ReturnReevaluationResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    proposal_id: UUID
    return_request_id: UUID
    status: str
    outcome: str | None
    replayed: bool = False
