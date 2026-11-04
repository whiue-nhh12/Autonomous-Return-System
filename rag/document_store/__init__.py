"""PostgreSQL-backed document and version lifecycle."""

from rag.document_store.document.service import DocumentVersionService, UpdateResult

__all__ = ["DocumentVersionService", "UpdateResult"]