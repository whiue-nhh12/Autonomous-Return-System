from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field


ProcessName = Literal["VALIDATION", "DOCLING_PARSE", "VLM_INFERENCE"]
ProcessStatus = Literal["SUCCESS", "WARNING", "FAILED"]
DocumentVersionStatus = Literal["PROCESSING", "READY", "FAILED", "RETIRED"]
GraphStatus = Literal["PENDING", "NORMAL", "GRAPH_WORTHY", "FAILED"]


class BlockType(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"
    CODE = "code"
    FORMULA = "formula"
    CAPTION = "caption"
    OTHER = "other"


class DocumentMetadata(BaseModel):
    doc_id: str
    source: str
    page_num: int = 0
    length_count: int = 0
    char_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    file_type: str


class DocumentVersion(BaseModel):
    """Business state for one immutable document version."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    doc_id: str
    version_number: int
    source_name: str
    storage_key: str
    content_hash: str
    file_type: str
    status: DocumentVersionStatus
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ContentBlock(BaseModel):
    block_id: str = Field(default_factory=lambda: str(uuid4()))
    section_id: str = "root"
    category_id: str = "uncategorized"
    raw_text: str = ""
    contextual_text: str = ""
    page_num: int | None = None
    block_type: BlockType = BlockType.PARAGRAPH
    hierarchical_path: str = ""
    level: int | None = None
    graph_status: GraphStatus = "PENDING"
    image_data: bytes | None = Field(default=None, exclude=True, repr=False)


class ParsedDocument(BaseModel):
    metadata: DocumentMetadata
    blocks: list[ContentBlock] = Field(default_factory=list)

    @computed_field
    @property
    def has_figures(self) -> bool:
        return any(block.block_type == BlockType.FIGURE for block in self.blocks)


class IngestionLogRecord(BaseModel):
    trace_id: str
    process: ProcessName
    status: ProcessStatus
    message: str = ""
    metrics: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
