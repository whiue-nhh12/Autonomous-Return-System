"""Logger và record riêng cho công đoạn cleaning."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from rag.cleaning.schemas import CleaningLogRecord, CleaningProcess, CleaningStatus


_LOGGER = logging.getLogger("logs.cleaning")
if not _LOGGER.handlers:
    formatter = logging.Formatter("%(message)s")
    project_root = Path(__file__).resolve().parents[3]
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "cleaning.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    _LOGGER.addHandler(file_handler)
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False


class CleaningLogger:
    """Ghi JSON log chỉ cho các process của cleaning."""

    def emit(
        self,
        *,
        trace_id: str,
        process: CleaningProcess,
        status: CleaningStatus,
        message: str = "",
        metrics: dict[str, Any] | None = None,
    ) -> CleaningLogRecord:
        record = CleaningLogRecord(
            trace_id=trace_id,
            process=process,
            status=status,
            message=message,
            metrics=metrics or {},
        )
        payload = record.model_dump(mode="json")
        log_fn = _LOGGER.error if status == "FAIL" else _LOGGER.warning if status == "WARNING" else _LOGGER.info
        log_fn(json.dumps(payload, ensure_ascii=False, default=str))
        return record