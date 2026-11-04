"""PostgreSQL repository for source text and durable Neo4j sync retries."""

from __future__ import annotations

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from rag.core.schemas.graphrag_schemas import GraphChunk, GraphSyncPayload
from rag.document_store.graph_store.models import GraphBase, GraphChunkRow


class PostgresGraphChunkRepository:
    def __init__(self, url: str, *, engine=None) -> None:
        self.engine = engine or create_engine(url, future=True)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)

    def create_schema(self) -> None:
        GraphBase.metadata.create_all(self.engine)

    def enqueue(self, payload: GraphSyncPayload) -> None:
        chunk = payload.chunk
        serialized = payload.model_dump(mode="json")
        serialized["chunk"]["raw_text"] = ""
        serialized["chunk"]["contextual_text"] = ""
        with self.session_factory.begin() as session:
            row = session.get(GraphChunkRow, chunk.chunk_id)
            if row is None:
                row = GraphChunkRow(
                    id=chunk.id,
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    version_number=chunk.version_number,
                    raw_text=chunk.raw_text,
                    contextual_text=chunk.contextual_text,
                    hierarchical_path=chunk.hierarchical_path,
                    sync_status="PENDING",
                    sync_payload=serialized,
                )
                session.add(row)
            else:
                row.id = chunk.id
                row.doc_id = chunk.doc_id
                row.version_number = chunk.version_number
                row.raw_text = chunk.raw_text
                row.contextual_text = chunk.contextual_text
                row.hierarchical_path = chunk.hierarchical_path
                row.sync_status = "PENDING"
                row.sync_payload = serialized
                row.last_error = None

    def get_pending(self, *, limit: int = 100) -> list[GraphSyncPayload]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        with self.session_factory() as session:
            rows = session.scalars(
                select(GraphChunkRow)
                .where(GraphChunkRow.sync_status.in_(("PENDING", "FAILED")))
                .order_by(GraphChunkRow.updated_at, GraphChunkRow.chunk_id)
                .limit(limit)
            )
            payloads = []
            for row in rows:
                payload = GraphSyncPayload.model_validate(row.sync_payload)
                payload.chunk.raw_text = row.raw_text
                payload.chunk.contextual_text = row.contextual_text
                payloads.append(payload)
            return payloads

    def mark_synced(self, chunk_id: str) -> None:
        self._update_status(chunk_id, "SYNCED", None)

    def mark_failed(self, chunk_id: str, error: str) -> None:
        self._update_status(chunk_id, "FAILED", error[:4000])

    def get_chunk(self, chunk_id: str) -> GraphChunk | None:
        with self.session_factory() as session:
            row = session.get(GraphChunkRow, chunk_id)
            if row is None:
                return None
            return GraphChunk(
                id=row.id,
                chunk_id=row.chunk_id,
                doc_id=row.doc_id,
                version_number=row.version_number,
                raw_text=row.raw_text,
                contextual_text=row.contextual_text,
                hierarchical_path=row.hierarchical_path,
            )

    def get_chunks_for_document(self, doc_id: str, version_number: int) -> list[GraphChunk]:
        with self.session_factory() as session:
            rows = session.scalars(
                select(GraphChunkRow)
                .where(
                    GraphChunkRow.doc_id == doc_id,
                    GraphChunkRow.version_number == version_number,
                )
                .order_by(GraphChunkRow.chunk_id)
            )
            return [
                GraphChunk(
                    id=row.id,
                    chunk_id=row.chunk_id,
                    doc_id=row.doc_id,
                    version_number=row.version_number,
                    raw_text=row.raw_text,
                    contextual_text=row.contextual_text,
                    hierarchical_path=row.hierarchical_path,
                )
                for row in rows
            ]

    def _update_status(self, chunk_id: str, status: str, error: str | None) -> None:
        with self.session_factory.begin() as session:
            row = session.get(GraphChunkRow, chunk_id)
            if row is None:
                raise LookupError(f"Graph chunk {chunk_id!r} does not exist")
            row.sync_status = status
            row.last_error = error
