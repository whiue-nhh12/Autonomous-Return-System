"""Lớp nền cho mọi bước ingestion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any
from uuid import uuid4

from rag.ingestion.config import IngestionSettings, get_settings
from rag.core.exception.ingestion_exceptions import IngestionError
from rag.core.logger.ingestion_logger import IngestionLogger
from rag.core.schemas.ingestion_schemas import IngestionLogRecord, ProcessName, ProcessStatus


class BaseIngestion(ABC):
    """Mọi bước ingestion kế thừa lớp này để dùng chung config, log và xử lý lỗi."""

    process: ProcessName

    def __init__(
        self,
        settings: IngestionSettings | None = None,
        logger: IngestionLogger | None = None,
        trace_id: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.logger = logger or IngestionLogger()
        self.trace_id = trace_id or str(uuid4())

    def emit(
        self,
        status: ProcessStatus,
        message: str = "",
        metrics: dict[str, Any] | None = None,
    ) -> IngestionLogRecord:
        return self.logger.emit(
            trace_id=self.trace_id,
            process=self.process,
            status=status,
            message=message,
            metrics=metrics,
        )

    def timed_metrics(self, started_at: float, **extra: Any) -> dict[str, Any]:
        return {"elapsed_ms": round((perf_counter() - started_at) * 1000, 2), **extra}

    def fail(self, message: str, metrics: dict[str, Any] | None = None, exc_type: type[IngestionError] = IngestionError) -> None:
        self.emit("FAILED", message, metrics)
        raise exc_type(message, process=self.process, metrics=metrics or {})

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Thực thi bước ingestion."""
