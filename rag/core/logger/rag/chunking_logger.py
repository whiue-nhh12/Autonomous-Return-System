"""Structured JSON logger for the chunking stage."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from rag.core.schemas.chunking_schemas import ChunkingLogRecord, ChunkingProcess, ChunkingStatus


_LOGGER = logging.getLogger("logs.chunking")
if not _LOGGER.handlers:
    formatter = logging.Formatter("%(message)s")
    project_root = Path(__file__).resolve().parents[4]
    log_dir = project_root / "logs" / "rag"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "chunking.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    _LOGGER.addHandler(file_handler)
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False


class ChunkingLogger:
    def emit(
        self,
        *,
        trace_id: str,
        process: ChunkingProcess,
        status: ChunkingStatus,
        metrics: dict[str, Any] | None = None,
        message: str = "",
    ) -> ChunkingLogRecord:
        record = ChunkingLogRecord(
            trace_id=trace_id,
            process=process,
            status=status,
            metrics=metrics or {},
            message=message,
        )
        payload = record.model_dump(mode="json")
        log_fn = _LOGGER.error if status == "FAIL" else _LOGGER.warning if status == "WARNING" else _LOGGER.info
        log_fn(json.dumps(payload, ensure_ascii=False, default=str))
        return record