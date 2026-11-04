"""Document update workflow with PostgreSQL as the business source of truth."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from rag.core.schemas.embedding_schemas import Embedding_Document
from rag.document_store.document.repository import PostgresDocumentRepository

_LOGGER = logging.getLogger(__name__)


class VersionVectorStore(Protocol):
    def load(self, document: Embedding_Document, *, version: int) -> int:
        ...

    def delete_version(self, *, doc_id: str, version: int) -> None:
        ...


class VersionKeywordIndex(Protocol):
    def index_document(self, document: Embedding_Document, *, version: int) -> int:
        ...

    def delete_version(self, *, doc_id: str, version: int) -> None:
        ...


@dataclass(frozen=True)
class UpdateResult:
    doc_id: str
    version_number: int
    status: str
    no_op: bool = False
    content_hash: str = ""


class DocumentVersionService:
    """Coordinate version creation, processing, publishing, and cleanup."""

    def __init__(
        self,
        repository: PostgresDocumentRepository,
        *,
        manifest_path: str | Path = "published_versions.json",
        vector_store: VersionVectorStore | None = None,
        keyword_index: VersionKeywordIndex | None = None,
    ) -> None:
        self.repository = repository
        self.manifest_path = Path(manifest_path)
        self.vector_store = vector_store
        self.keyword_index = keyword_index

    @staticmethod
    def calculate_content_hash(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def update(
        self,
        *,
        doc_id: str | None = None,
        content: bytes,
        source_name: str,
        storage_key: str,
        file_type: str,
        process: Callable[[bytes, str], Embedding_Document],
    ) -> UpdateResult:
        """Run one update; the callback owns parse, chunk, embed, and validation."""
        content_hash = self.calculate_content_hash(content)
        if doc_id is None:
            doc_id = self.repository.create_document(file_type=file_type)
        current = self.repository.get_current_version(doc_id)
        if current is not None and current.content_hash == content_hash:
            return UpdateResult(doc_id, current.version_number, current.status, no_op=True, content_hash=content_hash)

        version_number = (current.version_number if current else 0) + 1
        self.repository.create_processing_version(
            doc_id=doc_id,
            version_number=version_number,
            source_name=source_name,
            storage_key=storage_key,
            content_hash=content_hash,
            file_type=file_type,
        )
        try:
            embedded = process(content, doc_id)
            if embedded.metadata is None or embedded.metadata.doc_id != doc_id:
                raise ValueError("Processed embedding must preserve the logical doc_id")
            if self.vector_store is not None:
                self.vector_store.load(embedded, version=version_number)
            if self.keyword_index is not None:
                self.keyword_index.index_document(embedded, version=version_number)
            self.repository.set_status(doc_id, version_number, "READY")
        except Exception:
            self.repository.set_status(doc_id, version_number, "FAILED")
            self._cleanup_failed_version(doc_id, version_number)
            raise

        try:
            self.repository.publish(doc_id, version_number)
        except Exception:
            self.repository.set_status(doc_id, version_number, "FAILED")
            self._cleanup_failed_version(doc_id, version_number)
            raise
        try:
            self._write_manifest(doc_id, version_number, content_hash)
        except (OSError, ValueError):
            _LOGGER.exception("Failed to update published-version manifest; removing stale cache")
            try:
                self.manifest_path.unlink(missing_ok=True)
            except OSError:
                _LOGGER.exception("Failed to remove stale published-version manifest")
        if current is not None:
            self._cleanup_retired_version(doc_id, current.version_number)
        return UpdateResult(doc_id, version_number, "READY", content_hash=content_hash)

    def _cleanup_retired_version(self, doc_id: str, version_number: int) -> None:
        if self.vector_store is not None:
            try:
                self.vector_store.delete_version(doc_id=doc_id, version=version_number)
            except Exception:
                _LOGGER.exception("Failed to clean retired vectors for %s version %s", doc_id, version_number)
        if self.keyword_index is not None:
            try:
                self.keyword_index.delete_version(doc_id=doc_id, version=version_number)
            except Exception:
                _LOGGER.exception("Failed to clean retired keyword index for %s version %s", doc_id, version_number)

    def _cleanup_failed_version(self, doc_id: str, version_number: int) -> None:
        if self.vector_store is not None:
            try:
                self.vector_store.delete_version(doc_id=doc_id, version=version_number)
            except Exception:
                _LOGGER.exception("Failed to clean vectors for %s version %s", doc_id, version_number)
        if self.keyword_index is not None:
            try:
                self.keyword_index.delete_version(doc_id=doc_id, version=version_number)
            except Exception:
                _LOGGER.exception("Failed to clean keyword index for %s version %s", doc_id, version_number)

    def _write_manifest(self, doc_id: str, version_number: int, content_hash: str) -> None:
        manifest: dict[str, Any] = {}
        if self.manifest_path.exists():
            with self.manifest_path.open("r", encoding="utf-8") as file:
                loaded = json.load(file)
                if isinstance(loaded, dict):
                    manifest = loaded
        manifest[doc_id] = {
            "version": version_number,
            "content_hash": content_hash,
        }
        temporary_path = self.manifest_path.with_suffix(f"{self.manifest_path.suffix}.tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary_path.replace(self.manifest_path)