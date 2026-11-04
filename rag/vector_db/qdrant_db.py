"""Qdrant adapter for embedded documents."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from uuid import NAMESPACE_URL, uuid4, uuid5

from rag.core.logger.rag.vector_db_logger import VectorDBLogger
from rag.core.schemas.embedding_schemas import Embedding_Document
from rag.core.schemas.search_schemas import SearchHit
from rag.vector_db.base import Base_VectorDB


@dataclass
class _Point:
    id: str
    vector: list[float]
    payload: dict


class QdrantDB(Base_VectorDB):
    """Store 1024-dimensional embeddings in a cosine-similarity collection."""

    VECTOR_SIZE = 1024
    BATCH_SIZE = 64
    DEFAULT_COLLECTION = "embedding_documents"
    INDEXED_PAYLOAD_FIELDS = ("doc_id", "chunk_id", "version", "topic")

    def __init__(
        self,
        *,
        url: str = "http://localhost:6333",
        collection_name: str = DEFAULT_COLLECTION,
        client=None,
        logger: VectorDBLogger | None = None,
        trace_id: str | None = None,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty")
        self.collection_name = collection_name
        self.trace_id = trace_id or str(uuid4())
        self.logger = logger or VectorDBLogger()
        if client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise ImportError("QdrantDB requires qdrant-client. Install it with: pip install qdrant-client") from exc
            client = QdrantClient(url=url)
        self.client = client

    def create_collection(self) -> None:
        from qdrant_client.http import models

        if self.client.collection_exists(self.collection_name):
            collection = self.client.get_collection(self.collection_name)
            vector_config = collection.config.params.vectors
            vector_size = getattr(vector_config, "size", None)
            if vector_size is None:
                raise ValueError(
                    f"Qdrant collection {self.collection_name!r} does not use a single unnamed vector. "
                    "Recreate it with the configured embedding schema."
                )
            if vector_size != self.VECTOR_SIZE:
                raise ValueError(
                    f"Qdrant collection {self.collection_name!r} uses {vector_size}-dimensional vectors, "
                    f"but this application requires {self.VECTOR_SIZE}. Delete and recreate the collection "
                    "before ingesting documents."
                )
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=self.VECTOR_SIZE, distance=models.Distance.COSINE),
        )

    def create_payload_indexes(self) -> None:
        from qdrant_client.http import models

        for field_name in self.INDEXED_PAYLOAD_FIELDS:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=(
                    models.PayloadSchemaType.INTEGER
                    if field_name == "version"
                    else models.PayloadSchemaType.KEYWORD
                ),
            )

    @staticmethod
    def point_id(*, doc_id: str, chunk_index: int, version: int) -> str:
        value = f"{doc_id} {chunk_index} {version}"
        return str(uuid5(NAMESPACE_URL, value))

    def load(self, document: Embedding_Document, *, version: int = 1) -> int:
        try:
            from qdrant_client.http import models
            point_type = models.PointStruct
        except ImportError:
            point_type = _Point

        if version <= 0:
            raise ValueError("version must be greater than zero")
        doc_id = document.metadata.doc_id if document.metadata else "unknown"
        started = perf_counter()
        points = []
        for chunk_index, block in enumerate(document.blocks):
            if len(block.embedding) != self.VECTOR_SIZE:
                raise ValueError(f"Embedding must have {self.VECTOR_SIZE} dimensions")
            payload = {
                **block.payload,
                "doc_id": doc_id,
                "chunk_id": block.chunk_id,
                "chunk_index": chunk_index,
                "version": version,
            }
            points.append(
                point_type(
                    id=self.point_id(doc_id=doc_id, chunk_index=chunk_index, version=version),
                    vector=block.embedding,
                    payload=payload,
                )
            )

        for offset in range(0, len(points), self.BATCH_SIZE):
            self.client.upload_points(
                collection_name=self.collection_name,
                points=points[offset : offset + self.BATCH_SIZE],
            )
        self.logger.emit(
            trace_id=self.trace_id,
            process="FLUSHING",
            status="SUCCESS",
            message="Đã lưu embedding vào Qdrant.",
            metrics={
                "point_count": len(points),
                "batch_size": self.BATCH_SIZE,
                "elapsed_ms": round((perf_counter() - started) * 1000, 2),
            },
        )
        return len(points)

    def close(self) -> None:
        self.client.close()

    def search(
        self,
        query_vector: list[float],
        *,
        published_versions: dict[str, int],
        limit: int = 20,
    ) -> list[SearchHit]:
        from qdrant_client.http import models

        if len(query_vector) != self.VECTOR_SIZE:
            raise ValueError(f"Query vector must have {self.VECTOR_SIZE} dimensions")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        if not published_versions:
            return []

        published_filters = [
            models.Filter(
                must=[
                    models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
                    models.FieldCondition(key="version", match=models.MatchValue(value=version)),
                ]
            )
            for doc_id, version in published_versions.items()
        ]
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=models.Filter(should=published_filters),
            limit=limit,
            with_payload=True,
        )
        points = response.points if hasattr(response, "points") else response.get("points", [])
        hits = []
        for point in points:
            payload = point.payload or {}
            doc_id = str(payload.get("doc_id", ""))
            version = int(payload.get("version", 0))
            if published_versions.get(doc_id) != version:
                continue
            hits.append(
                SearchHit(
                    doc_id=doc_id,
                    version=version,
                    chunk_id=str(payload.get("chunk_id", point.id)),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    text=str(payload.get("text", "")),
                    score=float(point.score),
                    metadata=payload,
                )
            )
        return hits

    def delete_version(self, *, doc_id: str, version: int) -> None:
        try:
            from qdrant_client.http import models
        except ImportError as exc:
            raise ImportError("QdrantDB requires qdrant-client for version cleanup") from exc
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
                        models.FieldCondition(key="version", match=models.MatchValue(value=version)),
                    ]
                )
            ),
        )