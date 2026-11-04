"""Authenticated application identity returned by the auth service."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict


class AuthenticatedPrincipal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    user_id: UUID
    role: str
