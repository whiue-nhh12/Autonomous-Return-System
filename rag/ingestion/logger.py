"""Logging có cấu trúc cho ingestion."""

from __future__ import annotations

import json
import logging
from typing import Any

from rag.ingestion.schemas import IngestionLogRecord, ProcessName, ProcessStatus

_LOGGER = logging.getLogger("rag.ingestion")
if not _LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False


class IngestionLogger:
    def emit(
        self,
        *,
        trace_id: str,
        process: ProcessName,
        status: ProcessStatus,
        message: str = "",
        metrics: dict[str, Any] | None = None,
    ) -> IngestionLogRecord:
        record = IngestionLogRecord(
            trace_id=trace_id,
            process=process,
            status=status,
            message=message,
            metrics=metrics or {},
        )
        payload = record.model_dump(mode="json")
        log_fn = _LOGGER.error if status == "FAILED" else _LOGGER.warning if status == "WARNING" else _LOGGER.info
        log_fn(json.dumps(payload, ensure_ascii=False, default=str))
        return record
