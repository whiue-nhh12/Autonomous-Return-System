"""Ingest one local document and publish its chunks to configured storage backends."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from time import sleep
from typing import Protocol
from uuid import uuid4

from rag.chunking.chunker import DocumentChunker
from rag.cleaning.cleaner import DocumentCleaner
from rag.classifier.graph_classifier import GraphClassifier
from rag.core.schemas.chunking_schemas import Chunking_Document
from rag.core.schemas.cleaning_schemas import Cleaned_Document
from rag.core.schemas.embedding_schemas import Embedding_Document
from rag.core.schemas.ingestion_schemas import ParsedDocument
from rag.document_store.document.repository import PostgresDocumentRepository
from rag.document_store.document.service import DocumentVersionService
from rag.embedding.embedding import Sentence_transformer
from rag.graph.pipeline import GraphRagPipeline
from rag.core.config import get_settings
from rag.ingestion.pipeline import IngestionPipeline
from rag.ingestion.validator import SourceValidator
from rag.search.factory import create_elasticsearch_keyword_index
from rag.vector_db.qdrant_db import QdrantDB


class IngestionRunner(Protocol):
    def run(self, source: str) -> ParsedDocument:
        ...


class Cleaner(Protocol):
    def run(self, parsed: ParsedDocument) -> Cleaned_Document:
        ...


class Chunker(Protocol):
    def run(self, cleaned: Cleaned_Document) -> Chunking_Document:
        ...


class Embedder(Protocol):
    def run(self, chunked: Chunking_Document) -> Embedding_Document:
        ...


class _RateLimitRetryingGenerator:
    """Retry Groq TPM rate limits without restarting completed document work."""

    _RETRY_DELAYS_SECONDS = (5, 10, 20, 40, 60)

    def __init__(self, generator) -> None:
        self._generator = generator

    def generate(self, prompt: str, *, max_tokens: int, timeout_seconds: float) -> str:
        for attempt, delay in enumerate((*self._RETRY_DELAYS_SECONDS, None)):
            try:
                return self._generator.generate(
                    prompt,
                    max_tokens=max_tokens,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                message = str(exc).casefold()
                if not _is_groq_rate_limit_error(exc, message) or delay is None:
                    raise
                print(
                    "  Groq rate limit (429); retrying this GraphRAG request "
                    f"in {delay}s ({attempt + 1}/{len(self._RETRY_DELAYS_SECONDS)})."
                )
                sleep(delay)


def _is_groq_rate_limit_error(exc: Exception, message: str | None = None) -> bool:
    message = message if message is not None else str(exc).casefold()
    return (
        "rate_limit_exceeded" in message
        and (getattr(exc, "status_code", None) == 429 or "429" in message)
    )


def process_document_content(
    content: bytes,
    doc_id: str,
    *,
    source_name: str,
    file_suffix: str,
    ingestion: IngestionRunner,
    cleaner: Cleaner,
    chunker: Chunker,
    embedder: Embedder,
    classifier: GraphClassifier | None = None,
    on_cleaned: Callable[[Cleaned_Document], None] | None = None,
) -> Embedding_Document:
    """Run parsing, cleaning, chunking, and embedding for a stored document version."""
    cleaned = _clean_document_content(
        content,
        doc_id,
        source_name=source_name,
        file_suffix=file_suffix,
        ingestion=ingestion,
        cleaner=cleaner,
    )
    if classifier is not None:
        classifier.run(cleaned)
    if on_cleaned is not None:
        on_cleaned(cleaned)
    chunked = chunker.run(cleaned)
    embedded = embedder.run(chunked)
    if embedded.metadata is None or embedded.metadata.doc_id != doc_id:
        raise ValueError("Embedding stage did not preserve the database document ID")
    return embedded


def _clean_document_content(
    content: bytes,
    doc_id: str,
    *,
    source_name: str,
    file_suffix: str,
    ingestion: IngestionRunner,
    cleaner: Cleaner,
) -> Cleaned_Document:
    """Parse and clean source content without running chunking or embedding."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=file_suffix, delete=False) as temporary_file:
            temporary_file.write(content)
            temporary_path = Path(temporary_file.name)

        parsed: ParsedDocument = ingestion.run(str(temporary_path))
        parsed.metadata = parsed.metadata.model_copy(
            update={"doc_id": doc_id, "source": source_name}
        )
        return cleaner.run(parsed)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def collect_sources(source: Path, allowed_extensions: Sequence[str]) -> list[Path]:
    """Return a single supported file or all supported files under a directory."""
    source = source.expanduser().resolve()
    supported_extensions = {extension.lower() for extension in allowed_extensions}
    if source.is_file():
        if source.suffix.lower() not in supported_extensions:
            raise ValueError(f"Unsupported file extension: {source.suffix or '(none)'}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Source path does not exist or is not a file/directory: {source}")

    files = sorted(
        (
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in supported_extensions
        ),
        key=lambda path: path.relative_to(source).as_posix().casefold(),
    )
    if not files:
        raise ValueError(f"No supported documents found in directory: {source}")
    return files


def stable_directory_doc_id(source: Path, root: Path) -> str:
    """Derive a repeatable ID for a file ingested from a directory batch."""
    relative_path = source.resolve().relative_to(root.resolve()).as_posix().casefold()
    identity = f"{root.name.casefold()}/{relative_path}"
    return f"batch_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:32]}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest local documents into PostgreSQL, search indexes, and GraphRAG."
    )
    parser.add_argument("source", type=Path, help="Supported document file or directory to ingest")
    parser.add_argument("--doc-id", help="Existing logical ID (single-file input only)")
    parser.add_argument("--storage-key", help="Storage reference (single-file input only)")
    parser.add_argument(
        "--model",
        default=Sentence_transformer.DEFAULT_MODEL,
        help="Sentence-transformers model name or local model path",
    )
    parser.add_argument("--max-chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=200)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    source = args.source.expanduser().resolve()
    settings = get_settings()
    try:
        sources = collect_sources(source, settings.allowed_file_extensions)
    except (FileNotFoundError, ValueError) as exc:
        _build_parser().error(str(exc))
    if source.is_dir() and (args.doc_id is not None or args.storage_key is not None):
        _build_parser().error("--doc-id and --storage-key can only be used with a single file")

    trace_id = str(uuid4())
    repository = PostgresDocumentRepository(settings.storage_settings.postgres_url_value)
    vector_store: QdrantDB | None = None
    keyword_index = None
    graph_pipeline: GraphRagPipeline | None = None
    failures = 0
    try:
        vector_store = QdrantDB(url=os.getenv("QDRANT_URL", "http://localhost:6333"), trace_id=trace_id)
        repository.create_schema()
        vector_store.create_collection()
        vector_store.create_payload_indexes()
        keyword_index = create_elasticsearch_keyword_index(settings.storage_settings)

        ingestion = IngestionPipeline(settings=settings, trace_id=trace_id)
        cleaner = DocumentCleaner(trace_id=trace_id)
        embedder = Sentence_transformer(model_name=args.model, trace_id=trace_id)
        graph_pipeline = GraphRagPipeline.from_settings(settings, trace_id=trace_id)
        # TPM limits reserve the configured completion budget up front. Keep the
        # extraction response concise, then retry transient Groq 429s in place.
        graph_pipeline.extractor.max_tokens = min(graph_pipeline.extractor.max_tokens, 3000)
        graph_pipeline.extractor.generator = _RateLimitRetryingGenerator(
            graph_pipeline.extractor.generator
        )
        graph_pipeline.classifier.generator = _RateLimitRetryingGenerator(
            graph_pipeline.classifier.generator
        )
        graph_pipeline.create_schema()
        graph_pipeline.flush_pending()
        version_service = DocumentVersionService(
            repository,
            manifest_path=settings.storage_settings.published_versions_manifest,
            vector_store=vector_store,
            keyword_index=keyword_index,
        )

        for index, document_path in enumerate(sources, start=1):
            published_version: int | None = None
            try:
                validated_source = SourceValidator(settings=settings).run(str(document_path))
                content = validated_source.local_path.read_bytes()
                source_name = (
                    document_path.relative_to(source).as_posix()
                    if source.is_dir()
                    else document_path.name
                )
                doc_id = (
                    stable_directory_doc_id(document_path, source)
                    if source.is_dir()
                    else args.doc_id
                )
                cleaned_documents: dict[str, Cleaned_Document] = {}
                result = version_service.update(
                    doc_id=doc_id,
                    content=content,
                    source_name=source_name,
                    storage_key=args.storage_key or str(document_path),
                    file_type=validated_source.file_type,
                    process=lambda document_bytes, assigned_doc_id: process_document_content(
                        document_bytes,
                        assigned_doc_id,
                        source_name=source_name,
                        file_suffix=document_path.suffix.lower(),
                        ingestion=ingestion,
                        cleaner=cleaner,
                        chunker=DocumentChunker(
                            max_chunk_size=args.max_chunk_size,
                            overlap=args.overlap,
                            trace_id=trace_id,
                        ),
                        embedder=embedder,
                        classifier=graph_pipeline.classifier if graph_pipeline else None,
                        on_cleaned=lambda cleaned: cleaned_documents.__setitem__(
                            assigned_doc_id, cleaned
                        ),
                    ),
                )
                published_version = result.version_number
                cleaned = cleaned_documents.pop(result.doc_id, None)
                if result.no_op and cleaned is None and graph_pipeline is not None:
                    cleaned = _clean_document_content(
                        content,
                        result.doc_id,
                        source_name=source_name,
                        file_suffix=document_path.suffix.lower(),
                        ingestion=ingestion,
                        cleaner=cleaner,
                    )
                if cleaned is not None and graph_pipeline is not None:
                    graph_result = graph_pipeline.run(
                        cleaned,
                        version_number=result.version_number,
                    )
                    print(
                        "  GraphRAG: "
                        f"{graph_result.graph_worthy_block_count}/"
                        f"{graph_result.classified_block_count} blocks eligible; "
                        f"{graph_result.persisted_chunk_count} chunks, "
                        f"{graph_result.entity_count} entities, "
                        f"{graph_result.rule_count} rules persisted."
                    )
                state = "Unchanged" if result.no_op else "Published"
                print(f"[{index}/{len(sources)}] {state}: {result.doc_id} version {result.version_number}.")
            except Exception as exc:
                failures += 1
                if published_version is None:
                    print(
                        f"[{index}/{len(sources)}] Failed: "
                        f"{ascii(str(document_path))}: {ascii(str(exc))}"
                    )
                else:
                    print(
                        f"[{index}/{len(sources)}] Published version {published_version}, "
                        f"but GraphRAG processing failed: {ascii(str(exc))}"
                    )
        print(f"Finished: {len(sources) - failures} succeeded, {failures} failed.")
        return 1 if failures else 0
    finally:
        if keyword_index is not None:
            keyword_index.close()
        if graph_pipeline is not None:
            graph_pipeline.close()
        if vector_store is not None:
            vector_store.close()
        repository.engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
