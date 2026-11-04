import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine

from rag.core.schemas.embedding_schemas import Embedding_Document
from rag.core.schemas.ingestion_schemas import DocumentMetadata, DocumentVersion
from rag.document_store.document.models import Base
from rag.document_store.document.repository import PostgresDocumentRepository
from rag.document_store.document.service import DocumentVersionService


class FakeRepository:
    def __init__(self, current=None):
        self.current = current
        self.created = []
        self.statuses = []
        self.published = []

    def get_current_version(self, doc_id):
        return self.current

    def create_document(self, *, file_type):
        return "doc_123"

    def create_processing_version(self, **kwargs):
        self.created.append(kwargs)
        return DocumentVersion(status="PROCESSING", **kwargs)

    def set_status(self, doc_id, version_number, status):
        self.statuses.append((doc_id, version_number, status))

    def publish(self, doc_id, version_number):
        self.published.append((doc_id, version_number))


class FakeVectorStore:
    def __init__(self, *, fail_delete=False):
        self.fail_delete = fail_delete
        self.loaded = []
        self.deleted = []

    def load(self, document, *, version):
        self.loaded.append(version)

    def delete_version(self, *, doc_id, version):
        if self.fail_delete:
            raise RuntimeError("vector cleanup failed")
        self.deleted.append((doc_id, version))


class FakeKeywordIndex:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.indexed = []
        self.deleted = []

    def index_document(self, document, *, version):
        self.indexed.append(version)
        if self.fail:
            raise RuntimeError("keyword indexing failed")
        return len(document.blocks)

    def delete_version(self, *, doc_id, version):
        self.deleted.append((doc_id, version))


class TestDocumentVersionService(unittest.TestCase):
    def test_successful_update_writes_both_backends_before_publish(self):
        repository = FakeRepository()
        vector_store = FakeVectorStore()
        keyword_index = FakeKeywordIndex()
        with tempfile.TemporaryDirectory() as directory:
            service = DocumentVersionService(
                repository,
                manifest_path=Path(directory) / "published_versions.json",
                vector_store=vector_store,
                keyword_index=keyword_index,
            )
            result = service.update(
                doc_id="doc_123",
                content=b"document",
                source_name="doc.md",
                storage_key="doc/1",
                file_type="md",
                process=lambda _content, doc_id: Embedding_Document(
                    metadata=DocumentMetadata(doc_id=doc_id, source="doc.md", file_type="md")
                ),
            )

        self.assertEqual(result.version_number, 1)
        self.assertEqual(vector_store.loaded, [1])
        self.assertEqual(keyword_index.indexed, [1])
        self.assertEqual(repository.published, [("doc_123", 1)])

    def test_keyword_failure_marks_version_failed_and_cleans_new_version(self):
        current = DocumentVersion(
            doc_id="doc_123",
            version_number=1,
            source_name="doc.md",
            storage_key="doc/1",
            content_hash="old",
            file_type="md",
            status="READY",
        )
        repository = FakeRepository(current)
        vector_store = FakeVectorStore()
        keyword_index = FakeKeywordIndex(fail=True)
        with tempfile.TemporaryDirectory() as directory:
            service = DocumentVersionService(
                repository,
                manifest_path=Path(directory) / "published_versions.json",
                vector_store=vector_store,
                keyword_index=keyword_index,
            )
            with self.assertRaisesRegex(RuntimeError, "keyword indexing failed"):
                service.update(
                    doc_id="doc_123",
                    content=b"updated",
                    source_name="doc.md",
                    storage_key="doc/2",
                    file_type="md",
                    process=lambda _content, doc_id: Embedding_Document(
                        metadata=DocumentMetadata(doc_id=doc_id, source="doc.md", file_type="md")
                    ),
                )

        self.assertEqual(repository.statuses, [("doc_123", 2, "FAILED")])
        self.assertEqual(repository.published, [])
        self.assertEqual(vector_store.deleted, [("doc_123", 2)])
        self.assertEqual(keyword_index.deleted, [("doc_123", 2)])

    def test_retired_cleanup_failure_does_not_fail_published_update(self):
        current = DocumentVersion(
            doc_id="doc_123",
            version_number=1,
            source_name="doc.md",
            storage_key="doc/1",
            content_hash="old",
            file_type="md",
            status="READY",
        )
        repository = FakeRepository(current)
        vector_store = FakeVectorStore(fail_delete=True)
        keyword_index = FakeKeywordIndex()
        with tempfile.TemporaryDirectory() as directory:
            service = DocumentVersionService(
                repository,
                manifest_path=Path(directory) / "published_versions.json",
                vector_store=vector_store,
                keyword_index=keyword_index,
            )
            result = service.update(
                doc_id="doc_123",
                content=b"updated",
                source_name="doc.md",
                storage_key="doc/2",
                file_type="md",
                process=lambda _content, doc_id: Embedding_Document(
                    metadata=DocumentMetadata(doc_id=doc_id, source="doc.md", file_type="md")
                ),
            )

        self.assertEqual(result.status, "READY")
        self.assertEqual(repository.published, [("doc_123", 2)])
        self.assertEqual(keyword_index.deleted, [("doc_123", 1)])

    def test_new_document_uses_repository_generated_doc_id(self):
        repository = FakeRepository()
        service = DocumentVersionService(repository)
        processed_doc_ids = []

        with tempfile.TemporaryDirectory() as directory:
            service.manifest_path = Path(directory) / "published_versions.json"
            result = service.update(
                content=b"new document",
                source_name="new.md",
                storage_key="new/1",
                file_type="md",
                process=lambda content, doc_id: (
                    processed_doc_ids.append(doc_id)
                    or Embedding_Document(
                        metadata=DocumentMetadata(doc_id=doc_id, source="new.md", file_type="md")
                    )
                ),
            )

        self.assertEqual(result.doc_id, "doc_123")
        self.assertEqual(processed_doc_ids, ["doc_123"])

    def test_repository_generates_logical_doc_id_from_database_id(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        repository = PostgresDocumentRepository("sqlite:///:memory:", engine=engine)

        doc_id = repository.create_document(file_type="md")
        repository.create_processing_version(
            doc_id=doc_id,
            version_number=1,
            source_name="a.md",
            storage_key="a/1",
            content_hash="a" * 64,
            file_type="md",
        )
        repository.set_status(doc_id, 1, "READY")
        repository.publish(doc_id, 1)

        current = repository.get_current_version(doc_id)
        self.assertRegex(doc_id, r"^doc_\d+$")
        self.assertEqual(current.doc_id, doc_id)
        self.assertEqual(current.version_number, 1)
        engine.dispose()

    def test_same_hash_is_idempotent(self):
        content = b"same content"
        repository = FakeRepository(
            DocumentVersion(
                doc_id="doc_123",
                version_number=2,
                source_name="a.md",
                storage_key="a/2",
                content_hash=DocumentVersionService.calculate_content_hash(content),
                file_type="md",
                status="READY",
            )
        )
        vector_store = FakeVectorStore()
        with tempfile.TemporaryDirectory() as directory:
            result = DocumentVersionService(repository, manifest_path=Path(directory) / "published_versions.json", vector_store=vector_store).update(
                doc_id="doc_123",
                content=content,
                source_name="a.md",
                storage_key="a/3",
                file_type="md",
                process=lambda *_: Embedding_Document(),
            )

        self.assertTrue(result.no_op)
        self.assertEqual(repository.created, [])
        self.assertEqual(vector_store.loaded, [])

    def test_failure_marks_new_version_failed_and_keeps_old_vector(self):
        repository = FakeRepository(
            DocumentVersion(
                doc_id="doc_123",
                version_number=1,
                source_name="a.md",
                storage_key="a/1",
                content_hash="old",
                file_type="md",
                status="READY",
            )
        )
        vector_store = FakeVectorStore()
        with tempfile.TemporaryDirectory() as directory:
            service = DocumentVersionService(repository, manifest_path=Path(directory) / "published_versions.json", vector_store=vector_store)
            with self.assertRaisesRegex(ValueError, "parse failed"):
                service.update(
                    doc_id="doc_123",
                    content=b"new",
                    source_name="a.md",
                    storage_key="a/2",
                    file_type="md",
                    process=lambda *_: (_ for _ in ()).throw(ValueError("parse failed")),
                )

        self.assertEqual(repository.statuses, [("doc_123", 2, "FAILED")])
        self.assertEqual(repository.published, [])
        self.assertEqual(vector_store.deleted, [("doc_123", 2)])


if __name__ == "__main__":
    unittest.main()