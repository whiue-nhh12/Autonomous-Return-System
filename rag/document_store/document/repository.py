"""Repository for document version state in PostgreSQL."""

from __future__ import annotations

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from rag.core.schemas.ingestion_schemas import DocumentVersion, DocumentVersionStatus
from rag.document_store.document.models import Base, DocumentRow, DocumentVersionRow


class PostgresDocumentRepository:
    """All business-state reads and writes go through PostgreSQL."""

    def __init__(self, url: str, *, engine=None) -> None:
        self.engine = engine or create_engine(url, future=True)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def create_document(self, *, file_type: str) -> str:
        """Create a row, then derive doc_id from PostgreSQL's generated ID."""
        with self.session_factory.begin() as session:
            document = DocumentRow(file_type=file_type)
            session.add(document)
            session.flush()
            document.doc_id = f"doc_{document.id}"
            session.flush()
            return document.doc_id

    def get_current_version(self, doc_id: str) -> DocumentVersion | None:
        with self.session_factory() as session:
            document = self._get_document(session, doc_id)
            if document is None or document.published_version_number is None:
                return None
            row = next(
                (
                    version
                    for version in document.versions
                    if version.version_number == document.published_version_number
                ),
                None,
            )
            return self._to_schema(row) if row else None

    def get_published_versions(self) -> dict[str, int]:
        """Return the source-of-truth published version snapshot for manifest recovery."""
        with self.session_factory() as session:
            documents = session.scalars(
                select(DocumentRow).where(DocumentRow.published_version_number.is_not(None))
            )
            return {
                document.doc_id: document.published_version_number
                for document in documents
                if document.doc_id is not None and document.published_version_number is not None
            }

    def create_processing_version(
        self,
        *,
        doc_id: str,
        version_number: int,
        source_name: str,
        storage_key: str,
        content_hash: str,
        file_type: str,
    ) -> DocumentVersion:
        with self.session_factory.begin() as session:
            document = self._get_document(session, doc_id)
            if document is None:
                document = DocumentRow(doc_id=doc_id, file_type=file_type)
                session.add(document)
            else:
                document.file_type = file_type

            row = DocumentVersionRow(
                version_number=version_number,
                source_name=source_name,
                storage_key=storage_key,
                content_hash=content_hash,
                file_type=file_type,
                status="PROCESSING",
            )
            document.versions.append(row)
            session.flush()
            return self._to_schema(row)

    def set_status(self, doc_id: str, version_number: int, status: DocumentVersionStatus) -> None:
        with self.session_factory.begin() as session:
            row = self._get_version(session, doc_id, version_number)
            if row is None:
                raise LookupError(f"Version {doc_id}/v{version_number} does not exist")
            row.status = status

    def publish(self, doc_id: str, version_number: int) -> None:
        with self.session_factory.begin() as session:
            document = self._get_document(session, doc_id)
            row = next(
                (
                    version
                    for version in document.versions
                    if version.version_number == version_number
                ),
                None,
            ) if document is not None else None
            if document is None or row is None or row.status != "READY":
                raise ValueError("Only a READY version can be published")
            old = document.published_version_number
            document.published_version_number = version_number
            if old is not None and old != version_number:
                old_row = next(
                    (version for version in document.versions if version.version_number == old),
                    None,
                )
                if old_row is not None:
                    old_row.status = "RETIRED"

    def _get_version(self, session: Session, doc_id: str, version_number: int) -> DocumentVersionRow | None:
        document = self._get_document(session, doc_id)
        if document is None:
            return None
        return next(
            (version for version in document.versions if version.version_number == version_number),
            None,
        )

    @staticmethod
    def _get_document(session: Session, doc_id: str) -> DocumentRow | None:
        return session.scalar(select(DocumentRow).where(DocumentRow.doc_id == doc_id))

    @staticmethod
    def _to_schema(row: DocumentVersionRow) -> DocumentVersion:
        return DocumentVersion.model_validate(
            {
                "id": str(row.id),
                "doc_id": row.doc_id,
                "version_number": row.version_number,
                "source_name": row.source_name,
                "storage_key": row.storage_key,
                "content_hash": row.content_hash,
                "file_type": row.file_type,
                "status": row.status,
                "created_at": row.created_at,
            }
        )