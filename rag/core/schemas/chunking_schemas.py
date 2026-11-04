"""Schemas for the chunking stage."""

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from rag.core.schemas.ingestion_schemas import DocumentMetadata


class Chunking_Block(BaseModel):
    block_id: str = Field(default_factory=lambda: str(uuid4()))
    chunk_index: int = 0
    top_heading: str = ""
    section: str = ""
    subsection: list[str] = Field(default_factory=list)
    topic: list[str] = Field(default_factory=lambda: ["all"])
    raw_text: str = ""
    contextualized_text: str = ""
    hierachical_path: str = ""

    @property
    def hierarchical_path(self) -> str:
        return self.hierachical_path


class Chunking_Document(BaseModel):
    metadata: DocumentMetadata | None = None
    block: list[Chunking_Block] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    block_number: int = 0

    @property
    def blocks(self) -> list[Chunking_Block]:
        return self.block


ChunkingBlock = Chunking_Block
ChunkingDocument = Chunking_Document


ChunkingProcess = Literal["SPLIT_HEADING", "SPLIT_BULLET", "RECURSIVE_SPLIT"]
ChunkingStatus = Literal["SUCCESS", "FAIL", "WARNING"]


class ChunkingLogRecord(BaseModel):
    trace_id: str
    process: ChunkingProcess
    status: ChunkingStatus
    metrics: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    message: str = ""