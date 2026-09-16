"""Logging có cấu trúc cho ingestion."""

from __future__ import annotations
from pathlib import Path

import json
import logging
from typing import Any

from rag.core.schemas.ingestion_schemas import IngestionLogRecord, ProcessName, ProcessStatus

_LOGGER = logging.getLogger("logs.ingestion")
if not _LOGGER.handlers:
    formatter = logging.Formatter("%(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    project_root = Path(__file__).resolve().parents[3]
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    file_handler = logging.FileHandler(
        log_dir / "ingestion.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    _LOGGER.addHandler(console_handler)
    _LOGGER.addHandler(file_handler)
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
        print(payload)
        log_fn = _LOGGER.error if status == "FAILED" else _LOGGER.warning if status == "WARNING" else _LOGGER.info
        log_fn(json.dumps(payload, ensure_ascii=False, default=str))
        return record
