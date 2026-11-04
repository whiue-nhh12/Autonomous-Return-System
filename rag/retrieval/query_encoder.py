"""Query embedding contract and sentence-transformers adapter."""

from __future__ import annotations

from typing import Any, Protocol

from rag.embedding.embedding import Sentence_transformer


class QueryEncoder(Protocol):
    def encode(self, text: str) -> list[float]:
        ...


class SentenceTransformerQueryEncoder:
    """Encode a single query using the configured sentence-transformer model."""

    def __init__(
        self,
        model: Any | None = None,
        *,
        dimension: int = Sentence_transformer.EMBEDDING_DIMENSION,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.model = model
        self.dimension = dimension

    def encode(self, text: str) -> list[float]:
        if self.model is None:
            self.model = Sentence_transformer().load_model()
        vectors = self.model.encode(
            [text],
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        if len(vectors) != 1 or len(vectors[0]) != self.dimension:
            raise ValueError(f"Query embedding must have {self.dimension} dimensions")
        return [float(value) for value in vectors[0]]