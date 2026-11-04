"""Combine keyword and vector search results with reciprocal rank fusion."""

from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from rag.core.schemas.search_schemas import SearchHit
from rag.search.base_keyword_search import BaseKeywordSearch
from rag.retrieval.timing import RetrievalTimings

_LOGGER = logging.getLogger(__name__)


class VectorSearchBackend(Protocol):
    def search(
        self,
        query_vector: list[float],
        *,
        published_versions: dict[str, int],
        limit: int = 20,
    ) -> list[SearchHit]:
        ...


@dataclass(frozen=True)
class HybridSearchResult:
    hit: SearchHit
    score: float
    keyword_score: float | None
    vector_score: float | None

    @property
    def rrf_score(self) -> float:
        return self.score


class HybridSearch:
    """Run keyword/vector retrieval and fuse the two rankings with RRF."""

    def __init__(
        self,
        *,
        vector_backend: VectorSearchBackend,
        keyword_backend: BaseKeywordSearch,
        rrf_k: int = 60,
        oversample_factor: int = 4,
        timings: RetrievalTimings | None = None,
    ) -> None:
        if rrf_k < 1 or oversample_factor < 1:
            raise ValueError("rrf_k and oversample_factor must be positive")
        self.vector_backend = vector_backend
        self.keyword_backend = keyword_backend
        self.rrf_k = rrf_k
        self.oversample_factor = oversample_factor
        self.timings = timings or RetrievalTimings()
        self._search_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hybrid-search")

    def search(
        self,
        query: str,
        *,
        query_vector: list[float] | None,
        published_versions: dict[str, int],
        limit: int = 10,
    ) -> list[HybridSearchResult]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        if not published_versions:
            return []

        candidate_limit = limit * self.oversample_factor
        search_futures = [
            (
                "keyword",
                self._search_executor.submit(
                    self._timed_search,
                    "search.keyword",
                    self.keyword_backend.search,
                    query,
                    limit=candidate_limit,
                    published_versions=published_versions,
                ),
            )
        ]
        if query_vector is not None:
            search_futures.append(
                (
                    "vector",
                    self._search_executor.submit(
                        self._timed_search,
                        "search.vector",
                        self.vector_backend.search,
                        query_vector,
                        published_versions=published_versions,
                        limit=candidate_limit,
                    ),
                )
            )

        ranked_lists: list[tuple[str, list[SearchHit]]] = []
        for backend_name, future in search_futures:
            try:
                ranked_lists.append((backend_name, future.result()))
            except Exception:
                _LOGGER.exception("%s retrieval failed", backend_name)
        with self.timings.measure("search.hybrid_fusion"):
            return self.fuse(ranked_lists, published_versions=published_versions, limit=limit)

    def _timed_search(
        self,
        timing_name: str,
        search: Callable[..., list[SearchHit]],
        *args: Any,
        **kwargs: Any,
    ) -> list[SearchHit]:
        with self.timings.measure(timing_name):
            return search(*args, **kwargs)

    def fuse(
        self,
        ranked_lists: list[tuple[str, list[SearchHit]]],
        *,
        published_versions: dict[str, int],
        limit: int,
    ) -> list[HybridSearchResult]:
        fused_scores: dict[tuple[str, int, str], float] = defaultdict(float)
        selected_hits: dict[tuple[str, int, str], SearchHit] = {}
        keyword_scores: dict[tuple[str, int, str], float] = {}
        vector_scores: dict[tuple[str, int, str], float] = {}

        for backend_name, hits in ranked_lists:
            scores = keyword_scores if backend_name == "keyword" else vector_scores
            for rank, hit in enumerate(hits, start=1):
                if published_versions.get(hit.doc_id) != hit.version:
                    continue
                key = (hit.doc_id, hit.version, hit.chunk_id)
                fused_scores[key] += 1.0 / (self.rrf_k + rank)
                selected_hits.setdefault(key, hit)
                scores[key] = max(scores.get(key, hit.score), hit.score)

        keys = sorted(fused_scores, key=fused_scores.get, reverse=True)[:limit]
        return [
            HybridSearchResult(
                hit=selected_hits[key],
                score=fused_scores[key],
                keyword_score=keyword_scores.get(key),
                vector_score=vector_scores.get(key),
            )
            for key in keys
        ]

    def close(self) -> None:
        self._search_executor.shutdown(wait=False, cancel_futures=True)
