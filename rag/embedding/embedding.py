"""Embedding services built around interchangeable encoder implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any, Protocol, Sequence
from uuid import uuid4

from rag.core.logger.rag.embedding_logger import EmbeddingLogger
from rag.core.schemas.chunking_schemas import Chunking_Document
from rag.core.schemas.embedding_schemas import Embedding_Block, Embedding_Document


class Encoder(Protocol):
    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> Any:
        ...


class Base_Embedding(ABC):
    """Contract for embedding providers."""

    process = "EMBEDDING"

    def __init__(
        self,
        *,
        logger: EmbeddingLogger | None = None,
        trace_id: str | None = None,
        batch_size: int = 64,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.logger = logger or EmbeddingLogger()
        self.trace_id = trace_id or str(uuid4())
        self.batch_size = batch_size

    @abstractmethod
    def embed(self, document: Chunking_Document) -> Embedding_Document:
        """Embed every chunk in a chunking document."""

    def run(self, document: Chunking_Document) -> Embedding_Document:
        """Run the embedding stage."""
        return self.embed(document)

    @abstractmethod
    def load_model(self) -> Encoder:
        """Load and return the provider-specific encoder."""


class Sentence_transformer(Base_Embedding):
    """Sentence-transformers implementation using BGE Vietnamese embeddings."""

    DEFAULT_MODEL = "BAAI/bge-m3"
    EMBEDDING_DIMENSION = 1024

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        model_instance: Encoder | None = None,
        logger: EmbeddingLogger | None = None,
        trace_id: str | None = None,
        batch_size: int = 64,
    ) -> None:
        super().__init__(logger=logger, trace_id=trace_id, batch_size=batch_size)
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        self.model_name = model_name
        self._model = model_instance

    def load_model(self) -> Encoder:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "Sentence_transformer requires sentence-transformers. "
                    "Install it with: pip install sentence-transformers"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, document: Chunking_Document, batch_size: int | None = None) -> Embedding_Document:
        started = perf_counter()
        texts = [block.contextualized_text or block.raw_text for block in document.blocks]
        try:
            vectors = self.load_model().encode(
                texts,
                batch_size=batch_size if batch_size is not None else self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            if len(vectors) != len(document.blocks):
                raise ValueError("Embedding model returned an unexpected number of vectors")
            if any(len(vector) != self.EMBEDDING_DIMENSION for vector in vectors):
                raise ValueError(f"Embedding model must return {self.EMBEDDING_DIMENSION}-dimensional vectors")

            embeddings = [
                Embedding_Block(
                    chunk_id=chunk.block_id,
                    embedding=[float(value) for value in vector],
                    payload={
                        "text": text,
                        "raw_text": chunk.raw_text,
                        "contextualized_text": chunk.contextualized_text,
                        "hierarchical_path": chunk.hierarchical_path,
                        "top_heading": chunk.top_heading,
                        "section": chunk.section,
                        "subsection": chunk.subsection,
                        "topic": chunk.topic,
                    },
                )
                for chunk, text, vector in zip(document.blocks, texts, vectors)
            ]
            result = Embedding_Document(
                metadata=document.metadata,
                Block_Embedding=embeddings,
                number_block=len(embeddings),
            )
            self.logger.emit(
                trace_id=self.trace_id,
                process=self.process,
                status="SUCCESS",
                message="Đã tạo embedding cho tài liệu.",
                metrics={
                    "elapsed_ms": round((perf_counter() - started) * 1000, 2),
                    "block_count": len(embeddings),
                    "dimension": len(embeddings[0].embedding) if embeddings else 0,
                    "batch_size": self.batch_size,
                    "model": self.model_name,
                },
            )
            return result
        except Exception as exc:
            self.logger.emit(
                trace_id=self.trace_id,
                process=self.process,
                status="FAIL",
                message=f"Embedding thất bại: {exc}",
                metrics={"elapsed_ms": round((perf_counter() - started) * 1000, 2), "model": self.model_name},
            )
            raise
