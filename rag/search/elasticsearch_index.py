"""Elasticsearch BM25 index for PyVi-segmented Vietnamese chunks."""

from __future__ import annotations

from time import perf_counter
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from rag.core.logger.rag.vector_db_logger import VectorDBLogger
from rag.core.schemas.embedding_schemas import Embedding_Document
from rag.core.schemas.search_schemas import SearchHit
from rag.search.base_keyword_search import BaseKeywordSearch
from rag.search.vietnamese_tokenizer import VietnameseTokenizer


class ElasticsearchKeywordIndex(BaseKeywordSearch):
    BATCH_SIZE = 64
    DEFAULT_INDEX = "rag-chunks"

    def __init__(
        self,
        *,
        url: str = "http://localhost:9200",
        index_name: str = DEFAULT_INDEX,
        api_key: str | None = None,
        username: str | None = None,
        password: str | None = None,
        request_timeout: float = 30.0,
        client=None,
        tokenizer: VietnameseTokenizer | None = None,
        logger: VectorDBLogger | None = None,
        trace_id: str | None = None,
    ) -> None:
        if not index_name.strip():
            raise ValueError("index_name must not be empty")
        self.index_name = index_name
        self.tokenizer = tokenizer or VietnameseTokenizer()
        self.logger = logger or VectorDBLogger()
        self.trace_id = trace_id
        if (username is None) != (password is None):
            raise ValueError("username and password must be provided together")
        if client is None:
            try:
                from elasticsearch import Elasticsearch
            except ImportError as exc:
                raise ImportError(
                    "ElasticsearchKeywordIndex requires elasticsearch. Install it with: pip install elasticsearch"
                ) from exc
            client_options: dict[str, Any] = {"request_timeout": request_timeout}
            if api_key:
                client_options["api_key"] = api_key
            elif username is not None and password is not None:
                client_options["basic_auth"] = (username, password)
            client = Elasticsearch(url, **client_options)
        self.client = client

    def create_collection(self) -> None:
        if self.client.indices.exists(index=self.index_name):
            return
        self.client.indices.create(
            index=self.index_name,
            settings={
                "analysis": {
                    "analyzer": {
                        "vietnamese_pyvi": {
                            "type": "custom",
                            "tokenizer": "whitespace",
                            "filter": ["lowercase"],
                        }
                    }
                }
            },
            mappings={
                "dynamic": True,
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "version": {"type": "integer"},
                    "chunk_id": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "published_version_key": {"type": "keyword"},
                    "text_vi": {"type": "text", "analyzer": "vietnamese_pyvi"},
                    "text_original": {"type": "text"},
                    "topic": {"type": "keyword"},
                    "hierarchical_path": {"type": "text"},
                    "top_heading": {"type": "text"},
                    "section": {"type": "text"},
                    "subsection": {"type": "keyword"},
                    "payload": {"type": "object", "dynamic": True},
                },
            },
        )

    def index_document(self, document: Embedding_Document, *, version: int) -> int:
        if version <= 0:
            raise ValueError("version must be greater than zero")
        if document.metadata is None:
            raise ValueError("Embedding document must include document metadata")

        doc_id = document.metadata.doc_id
        started = perf_counter()
        indexed_count = 0
        for offset in range(0, len(document.blocks), self.BATCH_SIZE):
            operations: list[dict[str, Any]] = []
            for chunk_index, block in enumerate(document.blocks[offset : offset + self.BATCH_SIZE], start=offset):
                payload = block.payload
                text = str(
                    payload.get("text")
                    or payload.get("contextualized_text")
                    or payload.get("raw_text")
                    or ""
                )
                point_id = str(uuid5(NAMESPACE_URL, f"{doc_id} {chunk_index} {version}"))
                source = {
                    **payload,
                    "doc_id": doc_id,
                    "version": version,
                    "chunk_id": block.chunk_id,
                    "chunk_index": chunk_index,
                    "published_version_key": f"{doc_id}:{version}",
                    "text_original": text,
                    "text_vi": self.tokenizer.tokenize(text),
                    "payload": payload,
                }
                operations.extend(
                    [
                        {"index": {"_index": self.index_name, "_id": point_id}},
                        source,
                    ]
                )
            try:
                response = self.client.bulk(operations=operations, refresh=False)
                body = response.body if hasattr(response, "body") else response
                if body.get("errors"):
                    failures = [
                        item for item in body.get("items", []) if any(action.get("error") for action in item.values())
                    ]
                    raise RuntimeError(f"Elasticsearch bulk indexing failed for {len(failures)} chunks")
            except Exception as exc:
                self.logger.emit(
                    trace_id=self.trace_id or "elasticsearch",
                    process="FLUSHING",
                    status="FAIL",
                    message=f"Elasticsearch bulk indexing failed: {exc}",
                    metrics={"backend": "elasticsearch", "batch_count": len(operations) // 2},
                )
                raise
            indexed_count += len(operations) // 2

        self.logger.emit(
            trace_id=self.trace_id or "elasticsearch",
            process="FLUSHING",
            status="SUCCESS",
            message="Đã lập chỉ mục keyword cho các chunk.",
            metrics={
                "backend": "elasticsearch",
                "point_count": indexed_count,
                "batch_size": self.BATCH_SIZE,
                "elapsed_ms": round((perf_counter() - started) * 1000, 2),
            },
        )
        return indexed_count

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        published_versions: dict[str, int] | None = None,
    ) -> list[SearchHit]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        segmented_query = self.tokenizer.tokenize(query)
        if not segmented_query:
            return []
        es_query: dict[str, Any] = {
            "multi_match": {
                "query": segmented_query,
                "fields": ["text_vi^3", "text_original"],
                "type": "best_fields",
            }
        }
        if published_versions is not None:
            if not published_versions:
                return []
            es_query = {
                "bool": {
                    "must": [es_query],
                    "filter": [
                        {
                            "terms": {
                                "published_version_key": [
                                    f"{doc_id}:{version}" for doc_id, version in published_versions.items()
                                ]
                            }
                        }
                    ],
                }
            }
        response = self.client.search(
            index=self.index_name,
            size=limit,
            query=es_query,
        )
        body = response.body if hasattr(response, "body") else response
        hits = []
        for hit in body.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            hits.append(
                SearchHit(
                    doc_id=source["doc_id"],
                    version=int(source["version"]),
                    chunk_id=source["chunk_id"],
                    chunk_index=int(source["chunk_index"]),
                    text=source.get("text_original", ""),
                    score=float(hit.get("_score", 0.0)),
                    metadata=source.get("payload", {}),
                )
            )
        return hits

    def delete_version(self, *, doc_id: str, version: int) -> None:
        self.client.delete_by_query(
            index=self.index_name,
            query={
                "bool": {
                    "filter": [
                        {"term": {"doc_id": doc_id}},
                        {"term": {"version": version}},
                    ]
                }
            },
            conflicts="proceed",
            refresh=True,
        )

    def close(self) -> None:
        self.client.close()