import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.core.schemas.search_schemas import SearchHit
from rag.core.config import StorageSettings
from rag.retrieval.config import RetrievalSettings
from rag.retrieval.factory import _create_text_generator, create_retrieval_pipeline
from rag.retrieval.hybrid_search import HybridSearch
from rag.retrieval.pipeline import RetrievalPipeline
from rag.retrieval.query_rewriter import LLMQueryRewriter
from rag.search.base_keyword_search import BaseKeywordSearch
from rag.vector_db.qdrant_db import QdrantDB


def hit(doc_id, version, chunk_id, score):
    return SearchHit(
        doc_id=doc_id,
        version=version,
        chunk_id=chunk_id,
        chunk_index=0,
        text=chunk_id,
        score=score,
        metadata={},
    )


class FakeVectorBackend:
    def __init__(self, hits):
        self.hits = hits
        self.published_versions = None
        self.queries = []

    def search(self, query_vector, *, published_versions, limit):
        self.published_versions = published_versions
        self.queries.append(query_vector)
        return self.hits[:limit]


class FakeKeywordBackend(BaseKeywordSearch):
    def __init__(self, hits):
        self.hits = hits
        self.published_versions = None
        self.queries = []

    def search(self, query, *, limit, published_versions):
        self.published_versions = published_versions
        self.queries.append(query)
        return self.hits[:limit]

    def create_collection(self):
        pass

    def index_document(self, document, *, version):
        return len(document.blocks)

    def delete_version(self, *, doc_id, version):
        pass

    def close(self):
        pass


class FakeRepository:
    def get_published_versions(self):
        return {}


class FakeQueryEncoder:
    def __init__(self):
        self.queries = []

    def encode(self, text):
        self.queries.append(text)
        return [float(len(self.queries))] + [0.0] * 767


class FakeQueryRewriter:
    def rewrite(self, query, *, max_queries):
        return ["tìm thông tin mã ABC-123"][:max_queries]


class FakeHyDEGenerator:
    def generate(self, query):
        return "Tài liệu giả định nói về mã ABC-123."


class FakeReranker:
    def __init__(self, scores=None, *, delay=0):
        self.scores = scores or {}
        self.delay = delay

    def score(self, query, hits):
        if self.delay:
            time.sleep(self.delay)
        return [self.scores.get(item.chunk_id, 0.0) for item in hits]


class FakeTextGenerator:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def generate(self, prompt, *, max_tokens, timeout_seconds):
        self.calls += 1
        return self.response


class TestHybridSearch(unittest.TestCase):
    def test_text_generator_uses_retrieval_provider_settings(self):
        settings = RetrievalSettings(
            llm_provider="openai",
            llm_model="retrieval-model",
            llm_api_key="retrieval-secret",
            llm_base_url="https://llm.example.test/v1",
        )
        with patch("rag.retrieval.factory.OpenAITextGenerator") as generator_type:
            generator = _create_text_generator(settings, timeout_seconds=3.0)

        self.assertIs(generator, generator_type.return_value)
        generator_type.assert_called_once_with(
            model="retrieval-model",
            api_key="retrieval-secret",
            base_url="https://llm.example.test/v1",
        )

    def test_text_generator_requires_retrieval_api_key(self):
        with self.assertRaisesRegex(ValueError, "RETRIEVAL_LLM_API_KEY"):
            _create_text_generator(RetrievalSettings(), timeout_seconds=3.0)

    def test_factory_accepts_an_alternative_keyword_backend(self):
        backend = FakeKeywordBackend([])
        pipeline = create_retrieval_pipeline(
            StorageSettings(),
            vector_backend=FakeVectorBackend([]),
            repository=FakeRepository(),
            keyword_backend=backend,
            query_encoder=FakeQueryEncoder(),
        )

        self.assertIs(pipeline.hybrid_search.keyword_backend, backend)
        pipeline.close()

    def test_filters_stale_versions_and_fuses_both_rankings(self):
        keyword = FakeKeywordBackend(
            [
                hit("doc_a", 2, "shared", 10),
                hit("doc_a", 2, "keyword-only", 8),
                hit("doc_a", 1, "stale-keyword", 7),
            ]
        )
        vector = FakeVectorBackend(
            [
                hit("doc_a", 2, "vector-only", 0.9),
                hit("doc_a", 2, "shared", 0.8),
                hit("doc_a", 1, "stale-vector", 0.7),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "published_versions.json"
            manifest.write_text(json.dumps({"doc_a": {"version": 2}}), encoding="utf-8")
            search = HybridSearch(vector_backend=vector, keyword_backend=keyword)
            results = search.search(
                "tìm kiếm",
                query_vector=[0.0] * QdrantDB.VECTOR_SIZE,
                published_versions={"doc_a": 2},
                limit=3,
            )
            search.close()

        self.assertEqual(keyword.published_versions, {"doc_a": 2})
        self.assertEqual(vector.published_versions, {"doc_a": 2})
        self.assertEqual(results[0].hit.chunk_id, "shared")
        self.assertEqual(len(results), 3)
        self.assertNotIn("stale-keyword", {item.hit.chunk_id for item in results})
        self.assertNotIn("stale-vector", {item.hit.chunk_id for item in results})
        timings = search.timings.snapshot_ms()
        self.assertIn("search.keyword", timings)
        self.assertIn("search.vector", timings)
        self.assertIn("search.hybrid_fusion", timings)

    def test_uses_postgres_loader_when_manifest_is_missing(self):
        keyword = FakeKeywordBackend([])
        vector = FakeVectorBackend([])
        with tempfile.TemporaryDirectory() as directory:
            search = RetrievalPipeline(
                hybrid_search=HybridSearch(vector_backend=vector, keyword_backend=keyword),
                vector_backend=vector,
                query_encoder=FakeQueryEncoder(),
                manifest_path=Path(directory) / "missing.json",
                published_versions_loader=lambda: {"doc_b": 4},
            )
            self.assertEqual(search.search("query"), [])

        self.assertEqual(keyword.published_versions, {"doc_b": 4})

    def test_rewrite_searches_both_backends_and_hyde_only_adds_vector_query(self):
        keyword = FakeKeywordBackend([hit("doc_a", 2, "chunk-a", 5)])
        vector = FakeVectorBackend([hit("doc_a", 2, "chunk-a", 0.8)])
        encoder = FakeQueryEncoder()
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "published_versions.json"
            manifest.write_text(json.dumps({"doc_a": 2}), encoding="utf-8")
            search = RetrievalPipeline(
                hybrid_search=HybridSearch(vector_backend=vector, keyword_backend=keyword),
                vector_backend=vector,
                query_encoder=encoder,
                manifest_path=manifest,
                query_rewriter=FakeQueryRewriter(),
                hyde_generator=FakeHyDEGenerator(),
                enable_query_rewriting=True,
                enable_hyde=True,
            )

            search.search("mã ABC-123", limit=1)
            timings = search.timings.snapshot_ms()
            search.close()

        self.assertEqual(keyword.queries, ["mã ABC-123", "tìm thông tin mã ABC-123"])
        self.assertEqual(len(vector.queries), 3)
        self.assertEqual(
            encoder.queries,
            [
                "mã ABC-123",
                "tìm thông tin mã ABC-123",
                "Tài liệu giả định nói về mã ABC-123.",
            ],
        )
        self.assertIn("query_rewrite", timings)
        self.assertIn("embedding.query", timings)
        self.assertIn("embedding.hyde", timings)
        self.assertIn("hyde.vector_search", timings)
        self.assertIn("search.query_fusion", timings)

    def test_retrieve_exposes_timings_for_search_bottlenecks(self):
        keyword = FakeKeywordBackend([hit("doc_a", 2, "chunk-a", 5)])
        vector = FakeVectorBackend([hit("doc_a", 2, "chunk-a", 0.8)])
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "published_versions.json"
            manifest.write_text(json.dumps({"doc_a": 2}), encoding="utf-8")
            pipeline = RetrievalPipeline(
                hybrid_search=HybridSearch(vector_backend=vector, keyword_backend=keyword),
                vector_backend=vector,
                query_encoder=FakeQueryEncoder(),
                manifest_path=manifest,
            )

            result = pipeline.retrieve("tai nghe bị lỗi", limit=1)
            timings = pipeline.timings.snapshot_ms()
            pipeline.close()

        self.assertEqual(result.rag_doc[0].hit.chunk_id, "chunk-a")
        self.assertIn("retrieval.total", timings)
        self.assertIn("query.prepare", timings)
        self.assertIn("manifest.load_published_versions", timings)
        self.assertIn("embedding.query", timings)
        self.assertIn("search.keyword", timings)
        self.assertIn("search.vector", timings)
        self.assertIn("search.text_pipeline", timings)

    def test_reranker_orders_results_without_overwriting_rrf_score(self):
        keyword = FakeKeywordBackend(
            [hit("doc_a", 2, "first", 10), hit("doc_a", 2, "second", 8)]
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "published_versions.json"
            manifest.write_text(json.dumps({"doc_a": 2}), encoding="utf-8")
            vector = FakeVectorBackend([])
            search = RetrievalPipeline(
                hybrid_search=HybridSearch(vector_backend=vector, keyword_backend=keyword),
                vector_backend=vector,
                query_encoder=FakeQueryEncoder(),
                manifest_path=manifest,
                reranker=FakeReranker({"first": 0.1, "second": 0.9}),
                enable_reranker=True,
                rerank_candidate_count=2,
            )

            results = search.search("query", limit=2)
            search.close()

        self.assertEqual([item.hit.chunk_id for item in results], ["second", "first"])
        self.assertEqual(results[0].rerank_score, 0.9)
        self.assertEqual(results[0].rrf_score, 1 / 62)

    def test_reranker_timeout_falls_back_to_rrf_order(self):
        keyword = FakeKeywordBackend(
            [hit("doc_a", 2, "first", 10), hit("doc_a", 2, "second", 8)]
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "published_versions.json"
            manifest.write_text(json.dumps({"doc_a": 2}), encoding="utf-8")
            vector = FakeVectorBackend([])
            search = RetrievalPipeline(
                hybrid_search=HybridSearch(vector_backend=vector, keyword_backend=keyword),
                vector_backend=vector,
                query_encoder=FakeQueryEncoder(),
                manifest_path=manifest,
                reranker=FakeReranker({"first": 0.1, "second": 0.9}, delay=0.05),
                enable_reranker=True,
                rerank_candidate_count=2,
                reranker_timeout_seconds=0.001,
            )

            results = search.search("query", limit=2)
            search.close()

        self.assertEqual([item.hit.chunk_id for item in results], ["first", "second"])
        self.assertTrue(all(item.rerank_score is None for item in results))

    def test_query_rewriter_preserves_identifiers_and_caches_results(self):
        generator = FakeTextGenerator('["Tra cứu ABC-123 tại Hà Nội", "Bỏ ABC-123"]')
        rewriter = LLMQueryRewriter(generator)

        first = rewriter.rewrite("ABC-123 ở Hà Nội?")
        second = rewriter.rewrite("ABC-123 ở Hà Nội?")

        self.assertEqual(first, ["Tra cứu ABC-123 tại Hà Nội"])
        self.assertEqual(second, first)
        self.assertEqual(generator.calls, 1)

    def test_factory_wires_opt_in_components(self):
        search = create_retrieval_pipeline(
            StorageSettings(),
            vector_backend=FakeVectorBackend([]),
            repository=FakeRepository(),
            keyword_backend=FakeKeywordBackend([]),
            retrieval_settings=RetrievalSettings(
                enable_query_rewriting=True,
                enable_hyde=True,
                enable_reranker=True,
            ),
            query_rewriter=FakeQueryRewriter(),
            hyde_generator=FakeHyDEGenerator(),
            query_encoder=FakeQueryEncoder(),
            reranker=FakeReranker(),
        )

        self.assertTrue(search.enable_query_rewriting)
        self.assertTrue(search.enable_hyde)
        self.assertTrue(search.enable_reranker)
        search.close()


if __name__ == "__main__":
    unittest.main()