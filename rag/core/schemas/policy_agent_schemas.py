"""Schemas for policy-agent execution logs."""

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


PolicyAgentProcess = Literal[
    "CLASSIFY",
    "RETRIEVE_POLICY",
    "ANSWER_WITH_EVIDENCE",
    "NO_EVIDENCE",
    "CLARIFY",
]
PolicyAgentLogStatus = Literal["SUCCESS", "WARNING", "FAIL"]


class PolicyAgentLogRecord(BaseModel):
    trace_id: str
    process: PolicyAgentProcess
    status: PolicyAgentLogStatus
    metrics: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    message: str = ""
