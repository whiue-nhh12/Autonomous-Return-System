"""Factories for configured retrieval strategies."""

from rag.document_store.document.repository import PostgresDocumentRepository
from rag.core.config import StorageSettings
from rag.core.llm.text_generation import (
    GeminiTextGenerator,
    OpenAITextGenerator,
    TextGenerator,
)
from rag.graph.extractor import GraphExtractor
from rag.retrieval.config import RetrievalSettings, get_retrieval_settings
from rag.retrieval.hybrid_search import HybridSearch, VectorSearchBackend
from rag.retrieval.hyde import HyDEGenerator, LLMHyDEGenerator
from rag.retrieval.pipeline import RetrievalPipeline
from rag.retrieval.query_encoder import QueryEncoder, SentenceTransformerQueryEncoder
from rag.retrieval.query_classifier import QueryClassifier
from rag.retrieval.query_rewriter import LLMQueryRewriter, QueryRewriter
from rag.retrieval.reranker import CrossEncoderReranker, Reranker
from rag.search.base_keyword_search import BaseKeywordSearch
from rag.search.factory import create_elasticsearch_keyword_index
from rag.search.vietnamese_tokenizer import VietnameseTokenizer


def create_retrieval_pipeline(
    settings: StorageSettings,
    *,
    vector_backend: VectorSearchBackend,
    repository: PostgresDocumentRepository,
    keyword_backend: BaseKeywordSearch | None = None,
    client=None,
    tokenizer: VietnameseTokenizer | None = None,
    retrieval_settings: RetrievalSettings | None = None,
    text_generator: TextGenerator | None = None,
    query_encoder: QueryEncoder | None = None,
    query_rewriter: QueryRewriter | None = None,
    hyde_generator: HyDEGenerator | None = None,
    reranker: Reranker | None = None,
    graph_repository=None,
    graph_extractor: GraphExtractor | None = None,
    aliases: dict | None = None,
) -> RetrievalPipeline:
    retrieval_settings = retrieval_settings or get_retrieval_settings()
    tokenizer = tokenizer or VietnameseTokenizer(
        stop_words=retrieval_settings.bm25_stop_words,
        remove_stop_words=retrieval_settings.bm25_remove_stop_words,
    )
    if keyword_backend is None:
        keyword_backend = create_elasticsearch_keyword_index(
            settings,
            client=client,
            tokenizer=tokenizer,
        )

    needs_text_generator = (
        (retrieval_settings.enable_query_rewriting and query_rewriter is None)
        or (retrieval_settings.enable_hyde and hyde_generator is None)
        or (graph_repository is not None and graph_extractor is None)
    )
    if needs_text_generator and text_generator is None:
        text_generator = _create_text_generator(
            retrieval_settings,
            timeout_seconds=retrieval_settings.generation_timeout_seconds,
        )

    cache_options = {
        "cache_ttl_seconds": retrieval_settings.augmentation_cache_ttl_seconds,
        "cache_size": retrieval_settings.augmentation_cache_size,
    }
    if retrieval_settings.enable_query_rewriting and query_rewriter is None:
        query_rewriter = LLMQueryRewriter(
            text_generator,
            max_tokens=retrieval_settings.generation_max_tokens,
            timeout_seconds=retrieval_settings.generation_timeout_seconds,
            **cache_options,
        )
    query_classifier = None
    if retrieval_settings.enable_query_classification:
        classifier_generator = text_generator or _create_text_generator(
            retrieval_settings,
            timeout_seconds=retrieval_settings.generation_timeout_seconds,
            model_name=retrieval_settings.query_classifier_model,
        )
        query_classifier = QueryClassifier(
            classifier_generator,
            max_tokens=retrieval_settings.generation_max_tokens,
            timeout_seconds=retrieval_settings.generation_timeout_seconds,
        )
    if graph_repository is not None and graph_extractor is None:
        graph_extractor = GraphExtractor(
            generator=text_generator,
            max_tokens=retrieval_settings.generation_max_tokens,
            timeout_seconds=retrieval_settings.generation_timeout_seconds,
            aliases=aliases,
        )
    if retrieval_settings.enable_hyde and hyde_generator is None:
        hyde_generator = LLMHyDEGenerator(
            text_generator,
            max_tokens=retrieval_settings.generation_max_tokens,
            timeout_seconds=retrieval_settings.generation_timeout_seconds,
            **cache_options,
        )

    if query_encoder is None:
        query_encoder = SentenceTransformerQueryEncoder()
    if retrieval_settings.enable_reranker and reranker is None:
        reranker = CrossEncoderReranker(
            model_name=retrieval_settings.reranker_model,
            batch_size=retrieval_settings.reranker_batch_size,
            max_length=retrieval_settings.reranker_max_length,
        )

    hybrid_search = HybridSearch(
        vector_backend=vector_backend,
        keyword_backend=keyword_backend,
    )
    return RetrievalPipeline(
        hybrid_search=hybrid_search,
        vector_backend=vector_backend,
        query_encoder=query_encoder,
        manifest_path=settings.published_versions_manifest,
        published_versions_loader=repository.get_published_versions,
        query_rewriter=query_rewriter,
        hyde_generator=hyde_generator,
        reranker=reranker,
        query_classifier=query_classifier,
        graph_repository=graph_repository,
        graph_extractor=graph_extractor,
        enable_query_rewriting=retrieval_settings.enable_query_rewriting,
        enable_hyde=retrieval_settings.enable_hyde,
        enable_reranker=retrieval_settings.enable_reranker,
        max_rewrites=retrieval_settings.max_rewrites,
        rerank_candidate_count=retrieval_settings.rerank_candidate_count,
        reranker_timeout_seconds=retrieval_settings.reranker_timeout_seconds,
    )


def _create_text_generator(
    settings: RetrievalSettings,
    *,
    timeout_seconds: float,
    model_name: str | None = None,
) -> TextGenerator:
    api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else ""
    if not api_key:
        raise ValueError("Query classification, rewriting, HyDE, or graph retrieval requires RETRIEVAL_LLM_API_KEY")
    if settings.llm_provider == "openai":
        return OpenAITextGenerator(
            model=model_name or settings.llm_model,
            api_key=api_key,
            base_url=settings.llm_base_url or None,
        )
    return GeminiTextGenerator(
        model=model_name or settings.llm_model,
        api_key=api_key,
        base_url=settings.llm_base_url or None,
        timeout_seconds=timeout_seconds,
    )
