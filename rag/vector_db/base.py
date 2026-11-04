"""Abstract contract for vector database adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag.core.schemas.embedding_schemas import Embedding_Document


class Base_VectorDB(ABC):
    """Contract for persisting and querying embedded documents."""

    @abstractmethod
    def create_collection(self) -> None:
        """Create or validate the target collection."""

    @abstractmethod
    def create_payload_indexes(self) -> None:
        """Create indexes for fields used by pre-filtering."""

    @abstractmethod
    def load(self, document: Embedding_Document, *, version: int = 1) -> int:
        """Persist an embedded document and return the number of points written."""

    @abstractmethod
    def close(self) -> None:
        """Release the provider client when it is no longer needed."""