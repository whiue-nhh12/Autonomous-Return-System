"""Read the currently published policy versions from the RAG document store."""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from backend.db.session import create_engine_from_url
from rag.core.config import get_settings
from rag.document_store.document.models import DocumentRow


class PolicyVersionUnavailable(Exception):
    pass


class PublishedPolicyVersionSource:
    """Resolve citations against the document store's published-version source."""

    def current_versions(self, doc_ids: list[str]) -> dict[str, int | None]:
        if not doc_ids:
            return {}
        try:
            database_url = get_settings().storage_settings.postgres_url_value
            engine = _engine_for(database_url)
            with engine.connect() as connection:
                rows = connection.execute(
                    select(DocumentRow.doc_id, DocumentRow.published_version_number).where(
                        DocumentRow.doc_id.in_(doc_ids)
                    )
                ).all()
        except (SQLAlchemyError, RuntimeError, ValueError) as exc:
            raise PolicyVersionUnavailable(
                "The current published policy versions could not be read."
            ) from exc
        return {doc_id: version for doc_id, version in rows}


@lru_cache(maxsize=2)
def _engine_for(database_url: str):
    return create_engine_from_url(database_url)
