"""Pipeline ingestion: VALIDATION -> DOCLING_PARSE -> VLM_INFERENCE."""

from __future__ import annotations

from typing import Any

from rag.ingestion.base import BaseIngestion
from rag.ingestion.parser import DoclingParser
from rag.ingestion.schemas import ParsedDocument
from rag.ingestion.validator import SourceValidator
from rag.ingestion.vlm.vlm import VlmEnricher


class IngestionPipeline(BaseIngestion):
    """Điều phối các bước ingestion. Mỗi bước con kế thừa BaseIngestion."""

    process = "VALIDATION"

    def __init__(
        self,
        *args: Any,
        validator: SourceValidator | None = None,
        parser: DoclingParser | None = None,
        vlm: VlmEnricher | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        shared = {"settings": self.settings, "logger": self.logger, "trace_id": self.trace_id}
        self.validator = validator or SourceValidator(**shared)
        self.parser = parser or DoclingParser(**shared)
        self.vlm = vlm or VlmEnricher(**shared)

    def run(self, source: str) -> ParsedDocument:
        validated = None
        try:
            validated = self.validator.run(source)
            parsed = self.parser.run(validated)
            return self.vlm.run(parsed)
        finally:
            if validated is not None and validated.cleanup_path is not None:
                validated.cleanup_path.unlink(missing_ok=True)
