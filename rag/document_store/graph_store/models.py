"""PostgreSQL models for graph source chunks and the Neo4j outbox."""

from datetime import datetime, timezone

from sqlalchemy import DateTime, Index, Integer, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class GraphBase(DeclarativeBase):
    pass


class GraphChunkRow(GraphBase):
    __tablename__ = "graph_chunks"
    __table_args__ = (Index("ix_graph_chunks_document_version", "doc_id", "version_number"),)

    id: Mapped[str] = mapped_column(String(36), nullable=False, unique=True)
    chunk_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    contextual_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    hierarchical_path: Mapped[str] = mapped_column(Text, nullable=False, default="")
    sync_status: Mapped[str] = mapped_column(String(16), nullable=False, default="PENDING", index=True)
    sync_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
