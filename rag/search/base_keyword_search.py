"""Abstract contract for interchangeable keyword-search engines."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag.core.schemas.embedding_schemas import Embedding_Document
from rag.core.schemas.search_schemas import SearchHit


class BaseKeywordSearch(ABC):
    """Contract implemented by Elasticsearch and future keyword-search engines."""

    @abstractmethod
    def create_collection(self) -> None:
        """Create or validate the backend index and its mappings."""

    @abstractmethod
    def index_document(self, document: Embedding_Document, *, version: int) -> int:
        """Index all chunks of a document version and return the indexed count."""

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        published_versions: dict[str, int] | None = None,
    ) -> list[SearchHit]:
        """Return keyword-ranked chunks, optionally filtering published versions."""

    @abstractmethod
    def delete_version(self, *, doc_id: str, version: int) -> None:
        """Delete a document version from the keyword index."""

    @abstractmethod
    def close(self) -> None:
        """Release the backend client."""