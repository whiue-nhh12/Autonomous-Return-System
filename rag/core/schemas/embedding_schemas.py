"""Schemas for the embedding stage."""

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from rag.core.schemas.ingestion_schemas import DocumentMetadata


class Embedding_Block(BaseModel):
    block_id: str = Field(default_factory=lambda: str(uuid4()))
    chunk_id: str
    embedding: list[float] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)


class Embedding_Document(BaseModel):
    metadata: DocumentMetadata | None = None
    block_embedding: list[Embedding_Block] = Field(default_factory=list, alias="Block_Embedding")
    number_block: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"populate_by_name": True}

    @property
    def blocks(self) -> list[Embedding_Block]:
        return self.block_embedding

    @property
    def Block_Embedding(self) -> list[Embedding_Block]:
        return self.block_embedding


EmbeddingBlock = Embedding_Block
EmbeddingDocument = Embedding_Document

EmbeddingProcess = Literal["EMBEDDING"]
EmbeddingStatus = Literal["SUCCESS", "FAIL", "WARNING"]


class EmbeddingLogRecord(BaseModel):
    trace_id: str
    process: EmbeddingProcess
    status: EmbeddingStatus
    metrics: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    message: str = ""
