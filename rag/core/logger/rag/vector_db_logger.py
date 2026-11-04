"""Structured JSON logger for vector database operations."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from rag.core.schemas.vector_db_schemas import VectorDBLogRecord, VectorDBProcess, VectorDBStatus


_LOGGER = logging.getLogger("logs.vector_db")
if not _LOGGER.handlers:
    formatter = logging.Formatter("%(message)s")
    project_root = Path(__file__).resolve().parents[4]
    log_dir = project_root / "logs" / "rag"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "vector_db.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    _LOGGER.addHandler(file_handler)
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False


class VectorDBLogger:
    def emit(
        self,
        *,
        trace_id: str,
        process: VectorDBProcess,
        status: VectorDBStatus,
        metrics: dict[str, Any] | None = None,
        message: str = "",
    ) -> VectorDBLogRecord:
        record = VectorDBLogRecord(
            trace_id=trace_id,
            process=process,
            status=status,
            metrics=metrics or {},
            message=message,
        )
        log_fn = _LOGGER.error if status == "FAIL" else _LOGGER.warning if status == "WARNING" else _LOGGER.info
        log_fn(json.dumps(record.model_dump(mode="json"), ensure_ascii=False, default=str))
        return record