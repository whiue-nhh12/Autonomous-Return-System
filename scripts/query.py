"""Run end-to-end retrieval and answer generation for one query.

Examples:
    python -m scripts.query "Điều kiện hoàn tiền khi sản phẩm bị lỗi là gì?"
    python -m scripts.query --limit 8 --json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any
from time import perf_counter
from dotenv import dotenv_values

from rag.core.config import get_settings
from rag.core.llm.text_generation import GeminiTextGenerator, OpenAITextGenerator, TextGenerator
from rag.core.schemas.retrieval_schemas import Query, RetrieveResult
from rag.document_store.document.repository import PostgresDocumentRepository
from rag.graph.neo4j_repository import Neo4jGraphRepository
from rag.retrieval.config import RetrievalSettings, get_retrieval_settings
from rag.retrieval.factory import create_retrieval_pipeline
from rag.retrieval.query_encoder import QueryEncoder
from rag.retrieval.reranker import Reranker
from rag.search.factory import create_elasticsearch_keyword_index
from rag.search.vietnamese_tokenizer import VietnameseTokenizer
from rag.vector_db.qdrant_db import QdrantDB

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _create_text_generator(
    app_settings,
    retrieval_settings: RetrievalSettings,
    dotenv: dict[str, Any],
) -> TextGenerator:
    """Create the generator used by query routing, graph extraction, and answers."""
    if retrieval_settings.llm_provider == "openai":
        configured_key = retrieval_settings.llm_api_key
        fallback_key = app_settings.vlm_settings.openai.api_key or dotenv.get("OPENAI_API_KEY")
        api_key = configured_key.get_secret_value() if configured_key else str(fallback_key or "")
        if not api_key:
            raise ValueError("Set RETRIEVAL_LLM_API_KEY or OPENAI_API_KEY in .env to generate an answer")
        return OpenAITextGenerator(
            model=retrieval_settings.llm_model,
            api_key=api_key,
            base_url=retrieval_settings.llm_base_url or None,
        )

    configured_key = retrieval_settings.llm_api_key
    fallback_key = app_settings.vlm_settings.genai.api_key or dotenv.get("GEMINI_API_KEY")
    api_key = configured_key.get_secret_value() if configured_key else str(fallback_key or "")
    if not api_key:
        raise ValueError("Set RETRIEVAL_LLM_API_KEY or GEMINI_API_KEY in .env to generate an answer")
    return GeminiTextGenerator(
        model=retrieval_settings.llm_model,
        api_key=api_key,
        base_url=retrieval_settings.llm_base_url or None,
        timeout_seconds=retrieval_settings.generation_timeout_seconds,
    )


def _retrieval_context(result: RetrieveResult) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    context_parts: list[str] = []
    rag_sources: list[dict[str, Any]] = []
    graph_sources: list[dict[str, Any]] = []

    for index, retrieved in enumerate(result.rag_doc, start=1):
        hit = retrieved.hit
        source = {
            "source": f"RAG-{index}",
            "doc_id": hit.doc_id,
            "version": hit.version,
            "chunk_id": hit.chunk_id,
            "score": retrieved.score,
            "text": hit.text,
        }
        rag_sources.append(source)
        context_parts.append(
            f"[RAG-{index}] doc_id={hit.doc_id}, version={hit.version}, chunk_id={hit.chunk_id}\n{hit.text}"
        )

    for index, retrieved in enumerate(result.graph_doc, start=1):
        source = retrieved.model_dump(mode="json")
        source["source"] = f"GRAPH-{index}"
        graph_sources.append(source)
        context_parts.append(
            f"[GRAPH-{index}] Policy rule and supporting evidence:\n"
            f"{json.dumps(source, ensure_ascii=False, indent=2)}"
        )

    # Keep prompts bounded if a user asks for a large number of documents.
    context = "\n\n".join(context_parts)[:30_000]
    return context, rag_sources, graph_sources


def _generate_answer(generator: TextGenerator, query: str, context: str, settings: RetrievalSettings) -> str:
    prompt = (
        "Bạn là trợ lý trả lời câu hỏi về chính sách đổi trả và hoàn tiền. "
        "Chỉ dùng thông tin trong CONTEXT; không tự suy đoán hoặc thêm chính sách ngoài tài liệu. "
        "Nếu tài liệu không đủ để kết luận, hãy nói rõ thông tin còn thiếu. "
        "Trả lời bằng tiếng Việt, súc tích và dẫn mã nguồn như [RAG-1] hoặc [GRAPH-1] khi có thể.\n\n"
        f"QUERY:\n{query}\n\nCONTEXT:\n{context or '(Không tìm thấy tài liệu liên quan.)'}"
    )
    return generator.generate(
        prompt,
        max_tokens=settings.generation_max_tokens,
        timeout_seconds=settings.generation_timeout_seconds,
    )


def run_query(
    query_text: str,
    *,
    limit: int = 5,
    query_encoder: QueryEncoder | None = None,
    reranker: Reranker | None = None,
) -> tuple[dict[str, Any], float]:
    if not query_text.strip():
        raise ValueError("Query must not be empty")
    if limit <= 0:
        raise ValueError("limit must be greater than zero")

    stage = "load application settings"
    repository = None
    vector_backend: QdrantDB | None = None
    graph_repository: Neo4jGraphRepository | None = None
    keyword_backend = None
    pipeline = None
    timings_ms: dict[str, float] = {}
    total_start = perf_counter()
    initialization_start = perf_counter()
    try:
        app_settings = get_settings()
        retrieval_settings = get_retrieval_settings()
        dotenv = dotenv_values(_PROJECT_ROOT / ".env")
        alias_path = _PROJECT_ROOT / "rag" / "data" / "policy_aliases.json"
        with alias_path.open("r", encoding="utf-8") as alias_file:
            aliases = json.load(alias_file)
        if not isinstance(aliases, dict):
            raise ValueError(f"Alias file must contain a JSON object: {alias_path}")

        stage = f"initialize {retrieval_settings.llm_provider} LLM"
        generator = _create_text_generator(app_settings, retrieval_settings, dotenv)

        stage = "initialize PostgreSQL repository"
        repository = PostgresDocumentRepository(app_settings.storage_settings.postgres_url_value)

        qdrant_url = os.getenv("QDRANT_URL") or str(dotenv.get("QDRANT_URL") or "http://localhost:6333")
        qdrant_collection = str(dotenv.get("QDRANT_COLLECTION") or QdrantDB.DEFAULT_COLLECTION)
        stage = f"initialize Qdrant client at {qdrant_url}, collection={qdrant_collection}"
        vector_backend = QdrantDB(
            url=qdrant_url,
            collection_name=qdrant_collection,
        )

        graph_settings = app_settings.graph_settings
        stage = f"initialize Neo4j driver at {graph_settings.neo4j_uri}"
        graph_repository = Neo4jGraphRepository(
            uri=graph_settings.neo4j_uri,
            username=graph_settings.neo4j_username,
            password=graph_settings.neo4j_password.get_secret_value(),
        )

        tokenizer = VietnameseTokenizer(
            stop_words=retrieval_settings.bm25_stop_words,
            remove_stop_words=retrieval_settings.bm25_remove_stop_words,
        )
        stage = (
            "connect to Elasticsearch at "
            f"{app_settings.storage_settings.elasticsearch_url}, "
            f"index={app_settings.storage_settings.elasticsearch_index}"
        )
        keyword_backend = create_elasticsearch_keyword_index(
            app_settings.storage_settings,
            tokenizer=tokenizer,
        )

        stage = "build retrieval pipeline"
        pipeline = create_retrieval_pipeline(
            app_settings.storage_settings,
            vector_backend=vector_backend,
            repository=repository,
            keyword_backend=keyword_backend,
            graph_repository=graph_repository,
            tokenizer=tokenizer,
            retrieval_settings=retrieval_settings,
            text_generator=generator,
            query_encoder=query_encoder,
            reranker=reranker,
            aliases=aliases,
        )
        timings_ms["initialization"] = round((perf_counter() - initialization_start) * 1000, 2)

        query = Query(query=query_text.strip())
        stage = "classify query and tokenize it for BM25"
        pipeline.timings.reset()
        stage_start = perf_counter()
        with pipeline.timings.measure("query.prepare"):
            prepared_query = pipeline.prepare_query(query)
        timings_ms["query_preparation"] = round((perf_counter() - stage_start) * 1000, 2)

        stage = "retrieve from Elasticsearch/Qdrant and, when routed, Neo4j"
        stage_start = perf_counter()
        result = pipeline.retrieve(prepared_query, limit=limit, reset_timings=False)
        timings_ms["retrieval"] = round((perf_counter() - stage_start) * 1000, 2)
        retrieval_timings_ms = pipeline.timings.snapshot_ms()

        stage_start = perf_counter()
        context, rag_sources, graph_sources = _retrieval_context(result)
        timings_ms["context_building"] = round((perf_counter() - stage_start) * 1000, 2)

        stage = f"generate response with {retrieval_settings.llm_provider} model {retrieval_settings.llm_model}"
        stage_start = perf_counter()
        answer = _generate_answer(generator, query_text.strip(), context, retrieval_settings)
        timings_ms["answer_generation"] = round((perf_counter() - stage_start) * 1000, 2)
        timings_ms["total"] = round((perf_counter() - total_start) * 1000, 2)
        return {
            "input": {
                "query": query_text.strip(),
                "status_worthy": prepared_query.status_worthy,
                "bm25_tokens": prepared_query.bm25_tokens,
                "graph_query": (
                    prepared_query.graph_query.model_dump(mode="json")
                    if prepared_query.graph_query is not None
                    else None
                ),
            },
            "output": {
                "response": answer,
                "score": result.score,
                "rag_documents": rag_sources,
                "graph_documents": graph_sources,
            },
            "timings_ms": timings_ms,
            "retrieval_timings_ms": retrieval_timings_ms,
        }, timings_ms["total"]
        
    except Exception as exc:
        raise RuntimeError(f"{stage} failed ({type(exc).__name__}): {exc}") from exc
    finally:
        if pipeline is not None:
            pipeline.close()
        elif keyword_backend is not None:
            keyword_backend.close()
        if vector_backend is not None:
            vector_backend.close()
        if graph_repository is not None:
            graph_repository.close()
        if repository is not None:
            repository.engine.dispose()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieve policy documents and generate an answer.")
    parser.add_argument("query", nargs="?", help="Question to send to the retrieval system")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of results per retrieval source")
    parser.add_argument("--json", action="store_true", help="Print the complete input/output as JSON")
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    args = build_parser().parse_args()
    query = "Tôi mua một chiếc áo sơ mi giảm giá 50% nhưng bị rộng. Tôi có được hoàn tiền không và phí vận chuyển gửi trả tính thế nào?"
    query_text = args.query if args.query is not None else query
    try:
        result, _ = run_query(query_text, limit=args.limit)
    except Exception as exc:
        print(f"Query failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print("=== INPUT ===")
    print(f"Query: {result['input']['query']}")
    print(f"Routing: {', '.join(result['input']['status_worthy']) or 'Normal'}")
    print(f"BM25 tokens: {' '.join(result['input']['bm25_tokens']) or '(none)'}")
    print("\n=== RESPONSE ===")
    print(result["output"]["response"])
    print(f"\nRetrieval score: {result['output']['score']:.4f}")
    timing_labels = {
        "initialization": "Khởi tạo",
        "query_preparation": "Chuẩn bị query",
        "retrieval": "Truy xuất dữ liệu",
        "context_building": "Dựng context",
        "answer_generation": "Sinh câu trả lời",
        "total": "Tổng thời gian query",
    }
    print("\n=== THỜI GIAN CÁC CÔNG ĐOẠN ===")
    for stage, label in timing_labels.items():
        print(f"{label:<24} {result['timings_ms'][stage]:>10.2f} ms")
    print("Sources:")
    for source in result["output"]["rag_documents"]:
        print(f"- [{source['source']}] {source['doc_id']} / {source['chunk_id']} (score={source['score']:.4f})")
        print(f"  Text: {source['text'] or '(không có nội dung)'}")
    for source in result["output"]["graph_documents"]:
        print("Graph document:  \n")
        rule = source.get("rule") or {}
        print(f"- [{source['source']}] rule={rule.get('id', 'unknown')} (score={source.get('score', 0):.4f})")
        print(f"  Policy: {json.dumps(rule, ensure_ascii=False)}")
        evidence = source.get("evidence") or {}
        evidence_texts = evidence.get("evidence_text") or []
        if evidence_texts:
            for evidence_text in evidence_texts:
                print(f"  Evidence: {evidence_text}")
        else:
            print("  Evidence: (không có đoạn trích)")
    if not result["output"]["rag_documents"] and not result["output"]["graph_documents"]:
        print("- Không tìm thấy nguồn phù hợp")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
