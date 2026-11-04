"""Orchestrate query preparation, retrieval, fusion, and reranking."""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter

from rag.core.schemas.search_schemas import SearchHit
from rag.core.schemas.retrieval_schemas import Query, RetrieveResult
from rag.graph.extractor import GraphExtractor
from rag.retrieval.hyde import HyDEGenerator
from rag.retrieval.hybrid_search import HybridSearch, HybridSearchResult, VectorSearchBackend
from rag.retrieval.query_encoder import QueryEncoder
from rag.retrieval.query_classifier import QueryClassifier
from rag.retrieval.query_rewriter import LLMQueryRewriter, QueryRewriter
from rag.retrieval.reranker import Reranker

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalResult:
    hit: SearchHit
    score: float
    keyword_score: float | None
    vector_score: float | None
    rerank_score: float | None = None

    @property
    def rrf_score(self) -> float:
        return self.score


class RetrievalPipeline:
    """Prepare query variants and orchestrate hybrid retrieval components."""

    def __init__(
        self,
        *,
        hybrid_search: HybridSearch,
        vector_backend: VectorSearchBackend,
        query_encoder: QueryEncoder,
        manifest_path: str | Path = "published_versions.json",
        published_versions_loader: Callable[[], Mapping[str, int]] | None = None,
        query_rewriter: QueryRewriter | None = None,
        hyde_generator: HyDEGenerator | None = None,
        reranker: Reranker | None = None,
        query_classifier: QueryClassifier | None = None,
        graph_repository=None,
        graph_extractor: GraphExtractor | None = None,
        enable_query_rewriting: bool = False,
        enable_hyde: bool = False,
        enable_reranker: bool = False,
        max_rewrites: int = 2,
        rerank_candidate_count: int = 40,
        reranker_timeout_seconds: float = 2.0,
    ) -> None:
        if max_rewrites < 0 or max_rewrites > 2 or rerank_candidate_count < 1 or reranker_timeout_seconds <= 0:
            raise ValueError("max_rewrites must be 0-2 and reranker settings must be positive")
        if enable_query_rewriting and query_rewriter is None:
            raise ValueError("Query rewriting requires a query_rewriter")
        if enable_hyde and hyde_generator is None:
            raise ValueError("HyDE requires a hyde_generator")
        if enable_reranker and reranker is None:
            raise ValueError("Reranking requires a reranker")
        self.hybrid_search = hybrid_search
        self.timings = hybrid_search.timings
        self.vector_backend = vector_backend
        self.query_encoder = query_encoder
        self.manifest_path = Path(manifest_path)
        self.published_versions_loader = published_versions_loader
        self.query_rewriter = query_rewriter
        self.hyde_generator = hyde_generator
        self.reranker = reranker
        self.query_classifier = query_classifier or QueryClassifier()
        self.graph_repository = graph_repository
        self.graph_extractor = graph_extractor
        self.enable_query_rewriting = enable_query_rewriting
        self.enable_hyde = enable_hyde
        self.enable_reranker = enable_reranker
        self.max_rewrites = max_rewrites
        self.rerank_candidate_count = rerank_candidate_count
        self.reranker_timeout_seconds = reranker_timeout_seconds
        self._reranker_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="retrieval-reranker")
            if enable_reranker
            else None
        )
        self._reranker_future: Future[list[float]] | None = None
        self._reranker_lock = Lock()

    def search(
        self,
        query: str | Query,
        *,
        limit: int = 10,
        history: list[str] | None = None,
    ) -> list[RetrievalResult]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        search_started = perf_counter()
        query_model = self.prepare_query(query, history=history)
        query_text = query_model.rewritten_query or query_model.query
        with self.timings.measure("manifest.load_published_versions"):
            published_versions = self._load_published_versions()
        if not published_versions:
            self.timings.record("search.total", perf_counter() - search_started)
            return []

        candidate_count = max(limit, self.rerank_candidate_count) if self.enable_reranker else limit
        queries = [query_text]
        if self.enable_query_rewriting:
            try:
                with self.timings.measure("query_rewrite"):
                    if history and isinstance(self.query_rewriter, LLMQueryRewriter):
                        rewrites = self.query_rewriter.rewrite(
                            query_text,
                            max_queries=self.max_rewrites,
                            history=history,
                        )
                    else:
                        rewrites = self.query_rewriter.rewrite(query_text, max_queries=self.max_rewrites)
                if "QueryRewrite" in query_model.status_worthy and rewrites:
                    query_model.rewritten_query = rewrites[0]
                queries.extend(self._unique_queries(query_text, rewrites, self.max_rewrites))
            except Exception:
                _LOGGER.exception("Query rewriting failed; continuing with the original query")

        ranked_lists: list[list[HybridSearchResult]] = []
        for search_query in queries:
            try:
                with self.timings.measure("embedding.query"):
                    query_vector = self.query_encoder.encode(search_query)
            except Exception:
                _LOGGER.exception("Query embedding failed; continuing with keyword retrieval")
                query_vector = None
            with self.timings.measure("search.hybrid_wall"):
                ranked_lists.append(
                    self.hybrid_search.search(
                        search_query,
                        query_vector=query_vector,
                        published_versions=published_versions,
                        limit=candidate_count,
                    )
                )

        if self.enable_hyde or ("HYPE" in query_model.status_worthy and self.hyde_generator is not None):
            try:
                with self.timings.measure("hyde.generate_document"):
                    hypothetical_document = self.hyde_generator.generate(query_text)
                if hypothetical_document.strip():
                    with self.timings.measure("embedding.hyde"):
                        hyde_vector = self.query_encoder.encode(hypothetical_document)
                    with self.timings.measure("hyde.vector_search"):
                        hyde_hits = self.vector_backend.search(
                            hyde_vector,
                            published_versions=published_versions,
                            limit=candidate_count * self.hybrid_search.oversample_factor,
                        )
                    hyde_hits = [
                        hit
                        for hit in hyde_hits
                        if published_versions.get(hit.doc_id) == hit.version
                    ]
                    ranked_lists.append([
                        HybridSearchResult(
                            hit=hit,
                            score=hit.score,
                            keyword_score=None,
                            vector_score=hit.score,
                        )
                        for hit in hyde_hits
                    ])
            except Exception:
                _LOGGER.exception("HyDE generation or retrieval failed; continuing without HyDE")

        with self.timings.measure("search.query_fusion"):
            candidates = self._fuse_candidates(
                ranked_lists,
                limit=candidate_count,
                rrf_k=self.hybrid_search.rrf_k,
            )
        if self.enable_reranker and candidates:
            with self.timings.measure("reranker"):
                rerank_scores = self._score_with_timeout(query_text, candidates)
            if rerank_scores is not None:
                if len(rerank_scores) != len(candidates):
                    _LOGGER.error("Reranker returned an unexpected number of scores; using RRF ranking")
                elif not all(math.isfinite(float(score)) for score in rerank_scores):
                    _LOGGER.error("Reranker returned a non-finite score; using RRF ranking")
                else:
                    candidates = [
                        RetrievalResult(
                            hit=result.hit,
                            score=result.score,
                            keyword_score=result.keyword_score,
                            vector_score=result.vector_score,
                            rerank_score=float(score),
                        )
                        for result, score in zip(candidates, rerank_scores)
                    ]
                    candidates.sort(key=lambda result: result.rerank_score, reverse=True)
        results = candidates[:limit]
        self.timings.record("search.total", perf_counter() - search_started)
        return results

    def retrieve(
        self,
        query: str | Query,
        *,
        limit: int = 10,
        history: list[str] | None = None,
        reset_timings: bool = True,
    ) -> RetrieveResult:
        """Return text retrieval results and, when configured, matching graph rules."""
        if reset_timings:
            self.timings.reset()
        retrieve_started = perf_counter()
        with self.timings.measure("query.prepare"):
            query_model = self.prepare_query(query, history=history)
        with self.timings.measure("search.text_pipeline"):
            rag_results = self.search(query_model, limit=limit, history=history)
        graph_results = []
        if "GraphRag" in query_model.status_worthy and self.graph_repository is not None:
            if self.graph_extractor is None:
                raise ValueError("Graph retrieval requires a graph_extractor")
            try:
                with self.timings.measure("graph.query_extraction"):
                    graph_query = self.graph_extractor.extract_query(
                        query_model.rewritten_query or query_model.query
                    )
                query_model.graph_query = graph_query
                with self.timings.measure("graph.neo4j_query"):
                    graph_results = self.graph_repository.query_policy_rules(graph_query, limit=limit)
            except Exception:
                _LOGGER.exception("Graph retrieval failed; returning text retrieval results")
        score_candidates = [result.score for result in rag_results]
        score_candidates.extend(result.score for result in graph_results)
        result = RetrieveResult(
            graph_doc=graph_results,
            rag_doc=rag_results,
            score=max(score_candidates, default=0.0),
        )
        self.timings.record("retrieval.total", perf_counter() - retrieve_started)
        return result

    def prepare_query(self, query: str | Query, *, history: list[str] | None = None) -> Query:
        """Build the query model and attach tokens using the active BM25 tokenizer."""
        if isinstance(query, Query) and query.status_worthy:
            result = query
        else:
            with self.timings.measure("query.classification"):
                result = self.query_classifier.classify(query, history=history)
        keyword_tokenizer = getattr(self.hybrid_search.keyword_backend, "tokenizer", None)
        if keyword_tokenizer is not None:
            search_text = result.rewritten_query or result.query
            with self.timings.measure("query.tokenization"):
                result.bm25_tokens = keyword_tokenizer.tokenize(search_text).split()
        return result

    @staticmethod
    def _unique_queries(original: str, rewrites: list[str], maximum: int) -> list[str]:
        if maximum <= 0:
            return []
        seen = {original.strip().casefold()}
        unique = []
        for rewritten in rewrites:
            candidate = rewritten.strip()
            if candidate and candidate.casefold() not in seen:
                unique.append(candidate)
                seen.add(candidate.casefold())
            if len(unique) >= maximum:
                break
        return unique

    @staticmethod
    def _fuse_candidates(
        ranked_lists: Sequence[list[HybridSearchResult]],
        *,
        limit: int,
        rrf_k: int = 60,
    ) -> list[RetrievalResult]:
        fused_scores: dict[tuple[str, int, str], float] = defaultdict(float)
        selected_hits: dict[tuple[str, int, str], SearchHit] = {}
        keyword_scores: dict[tuple[str, int, str], float] = {}
        vector_scores: dict[tuple[str, int, str], float] = {}
        for results in ranked_lists:
            for rank, result in enumerate(results, start=1):
                hit = result.hit
                key = (hit.doc_id, hit.version, hit.chunk_id)
                fused_scores[key] += 1.0 / (rrf_k + rank)
                selected_hits.setdefault(key, hit)
                if result.keyword_score is not None:
                    keyword_scores[key] = max(keyword_scores.get(key, result.keyword_score), result.keyword_score)
                if result.vector_score is not None:
                    vector_scores[key] = max(vector_scores.get(key, result.vector_score), result.vector_score)
        keys = sorted(fused_scores, key=fused_scores.get, reverse=True)[:limit]
        return [
            RetrievalResult(
                hit=selected_hits[key],
                score=fused_scores[key],
                keyword_score=keyword_scores.get(key),
                vector_score=vector_scores.get(key),
            )
            for key in keys
        ]

    def _score_with_timeout(self, query: str, results: list[RetrievalResult]) -> list[float] | None:
        executor = self._reranker_executor
        if executor is None:
            return None
        with self._reranker_lock:
            if self._reranker_future is not None and not self._reranker_future.done():
                _LOGGER.warning("Previous reranker call is still running; using RRF ranking")
                return None
            future = executor.submit(self.reranker.score, query, [result.hit for result in results])
            self._reranker_future = future
        try:
            return future.result(timeout=self.reranker_timeout_seconds)
        except FutureTimeoutError:
            _LOGGER.warning("Reranker timed out; using RRF ranking")
            return None
        except Exception:
            _LOGGER.exception("Reranker failed; using RRF ranking")
            return None
        finally:
            if future.done():
                with self._reranker_lock:
                    if self._reranker_future is future:
                        self._reranker_future = None

    def _load_published_versions(self) -> dict[str, int]:
        try:
            with self.manifest_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            if not isinstance(payload, dict):
                raise TypeError("Published version manifest must be a JSON object")
            versions = {}
            for doc_id, value in payload.items():
                raw_version = value.get("version", value.get("version_number")) if isinstance(value, dict) else value
                if isinstance(raw_version, int) and not isinstance(raw_version, bool) and raw_version > 0:
                    versions[str(doc_id)] = raw_version
                else:
                    raise ValueError(f"Invalid published version for {doc_id}")
            return versions
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            if self.published_versions_loader is None:
                _LOGGER.exception("Unable to read published versions manifest: %s", self.manifest_path)
                return {}
            _LOGGER.warning("Rebuilding published-version lookup from PostgreSQL")
            return {str(doc_id): int(version) for doc_id, version in self.published_versions_loader().items()}

    def close(self) -> None:
        self.hybrid_search.close()
        if self._reranker_executor is not None:
            self._reranker_executor.shutdown(wait=False, cancel_futures=True)
