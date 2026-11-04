"""Structured JSON logger for policy-agent execution."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from rag.core.schemas.policy_agent_schemas import (
    PolicyAgentLogRecord,
    PolicyAgentLogStatus,
    PolicyAgentProcess,
)


_LOGGER = logging.getLogger("agents.policy_agent")
if not _LOGGER.handlers:
    project_root = Path(__file__).resolve().parents[5]
    log_dir = project_root / "logs" / "agents" / "policy_agent"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "policy_agent.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    _LOGGER.addHandler(file_handler)
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False


class PolicyAgentLogger:
    def emit(
        self,
        *,
        trace_id: str,
        process: PolicyAgentProcess,
        status: PolicyAgentLogStatus,
        metrics: dict[str, Any] | None = None,
        message: str = "",
    ) -> PolicyAgentLogRecord:
        record = PolicyAgentLogRecord(
            trace_id=trace_id,
            process=process,
            status=status,
            metrics=metrics or {},
            message=message,
        )
        payload = json.dumps(record.model_dump(mode="json"), ensure_ascii=False, default=str)
        log_fn = (
            _LOGGER.error
            if status == "FAIL"
            else _LOGGER.warning
            if status == "WARNING"
            else _LOGGER.info
        )
        log_fn(payload)
        return record
