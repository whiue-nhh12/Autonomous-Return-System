"""Schemas for vector database persistence and structured logging."""

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


VectorDBProcess = Literal["FLUSHING", "UPSERTING"]
VectorDBStatus = Literal["SUCCESS", "FAIL", "WARNING"]


class VectorDBLogRecord(BaseModel):
    trace_id: str
    process: VectorDBProcess
    status: VectorDBStatus
    metrics: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    message: str = ""