"""Structured JSON logger for GraphRAG stages."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from rag.core.schemas.graphrag_schemas import GraphLogRecord, GraphProcess, GraphProcessStatus


_LOGGER = logging.getLogger("logs.graphrag")
if not _LOGGER.handlers:
    formatter = logging.Formatter("%(message)s")
    project_root = Path(__file__).resolve().parents[4]
    log_dir = project_root / "logs" / "rag"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "graphrag.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    _LOGGER.addHandler(file_handler)
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False


class GraphRagLogger:
    def emit(
        self,
        *,
        trace_id: str,
        process: GraphProcess,
        status: GraphProcessStatus,
        message: str = "",
        metrics: dict[str, Any] | None = None,
    ) -> GraphLogRecord:
        record = GraphLogRecord(
            trace_id=trace_id,
            process=process,
            status=status,
            message=message,
            metrics=metrics or {},
        )
        log_fn = _LOGGER.error if status == "FAIL" else _LOGGER.warning if status == "WARNING" else _LOGGER.info
        log_fn(json.dumps(record.model_dump(mode="json"), ensure_ascii=False, default=str))
        return record
