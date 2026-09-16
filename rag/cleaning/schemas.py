"""Schema đầu ra của công đoạn cleaning."""

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field

from rag.core.schemas.ingestion_schemas import ContentBlock, DocumentMetadata


class Cleaned_Document(BaseModel):
    metadata: DocumentMetadata
    blocks: list[ContentBlock] = Field(default_factory=list)
    total_cleaned_chars: int = 0
    removed_block_count: int = 0

    @computed_field
    @property
    def has_figures(self) -> bool:
        return any(block.block_type.value == "figure" for block in self.blocks)


CleanedDocument = Cleaned_Document


CleaningProcess = Literal["FILTERING_BLOCK", "NORMALIZING", "RE_CLEANING"]
CleaningStatus = Literal["SUCCESS", "FAIL", "WARNING"]


class CleaningLogRecord(BaseModel):
    trace_id: str
    process: CleaningProcess
    status: CleaningStatus
    message: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))