"""Policy document retrieval worker backed by the shared RAG pipeline."""

from __future__ import annotations

from datetime import date, datetime
import logging
import re
from typing import Any, Protocol

from agent.agents.policy.state.retrieval import (
    DocumentCandidate,
    DocumentRetrievalLocalState,
)

_LOGGER = logging.getLogger(__name__)


class PolicySearchPipeline(Protocol):
    """Small part of RetrievalPipeline used by this agent."""

    def search(self, query: str, *, limit: int, history: list[str] | None = None) -> list[Any]:
        ...


class PolicyAgent:
    """Retrieve published policy passages into policy-local state."""

    def __init__(self, retrieval: PolicySearchPipeline) -> None:
        self.retrieval = retrieval

    def retrieve(self, state: DocumentRetrievalLocalState) -> DocumentRetrievalLocalState:
        """Search the RAG index and select policy evidence for ``state``."""
        state.status = "searching"
        state.error = None
        state.candidates = []
        state.selected_evidence = []

        if not state.filters.published_only:
            state.status = "failed"
            state.error = "The retrieval pipeline only exposes published policy versions."
            return state

        query = " ".join(state.original_query.split())
        state.normalized_query = query.casefold()
        queries = [query, *state.query_variants]
        unique_queries = list(dict.fromkeys(" ".join(item.split()) for item in queries if item.strip()))

        try:
            results_by_key: dict[tuple[str, int, str], DocumentCandidate] = {}
            for search_query in unique_queries:
                results = self.retrieval.search(search_query, limit=state.candidate_limit)
                for result in results:
                    hit = result.hit
                    metadata = hit.metadata or {}
                    text = str(hit.text).strip()
                    if not text:
                        continue
                    candidate = DocumentCandidate(
                        doc_id=str(hit.doc_id),
                        version=int(hit.version),
                        chunk_id=str(hit.chunk_id),
                        text=text,
                        score=max(0.0, float(result.score)),
                        source_title=_metadata_value(metadata, "source_title", "title", "source_name"),
                        source_url=_metadata_value(metadata, "source_url", "url"),
                    )
                    if not self._matches_filters(candidate, metadata, state):
                        continue
                    key = (candidate.doc_id, candidate.version, candidate.chunk_id)
                    previous = results_by_key.get(key)
                    if previous is None or candidate.score > previous.score:
                        results_by_key[key] = candidate

            state.candidates = sorted(
                results_by_key.values(), key=lambda item: item.score, reverse=True
            )[: state.candidate_limit]
            state.set_selected_evidence(state.candidates)
            state.status = "completed" if state.selected_evidence else "not_found"
        except Exception as exc:
            _LOGGER.exception("Policy retrieval failed")
            state.status = "failed"
            state.error = str(exc) or exc.__class__.__name__
        return state

    @staticmethod
    def _matches_filters(
        candidate: DocumentCandidate,
        metadata: dict[str, Any],
        state: DocumentRetrievalLocalState,
    ) -> bool:
        filters = state.filters
        if filters.doc_ids and candidate.doc_id not in filters.doc_ids:
            return False
        if filters.policy_version is not None:
            version = str(filters.policy_version).strip().lstrip("vV")
            if version != str(candidate.version):
                return False
        if filters.as_of_date is not None:
            effective_date = _effective_date(metadata)
            if effective_date is None or effective_date > filters.as_of_date:
                return False
        return True

def _metadata_value(metadata: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _effective_date(metadata: dict[str, Any]) -> date | None:
    for key in ("effective_from", "published_at", "published_date"):
        value = metadata.get(key)
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            match = re.match(r"^\s*(\d{4}-\d{2}-\d{2})", value)
            if match:
                try:
                    return date.fromisoformat(match.group(1))
                except ValueError:
                    continue
    return None
