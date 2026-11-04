"""Agent-local state for one policy/document retrieval operation."""

from datetime import date
from typing import Literal

from pydantic import Field

from agent.core.state.shared import PolicyEvidence, StateModel


class RetrievalFilter(StateModel):
    published_only: bool = True
    doc_ids: list[str] = Field(default_factory=list)
    policy_version: str | None = None
    as_of_date: date | None = None


class DocumentCandidate(StateModel):
    doc_id: str
    version: int = Field(ge=1)
    chunk_id: str
    text: str = Field(min_length=1)
    score: float = Field(ge=0)
    source_title: str | None = None
    source_url: str | None = None


class DocumentRetrievalLocalState(StateModel):
    """Temporary search context owned entirely by the policy child agent."""

    status: Literal["pending", "searching", "completed", "not_found", "failed"] = "pending"
    original_query: str = Field(min_length=1)
    normalized_query: str | None = None
    query_variants: list[str] = Field(default_factory=list)
    filters: RetrievalFilter = Field(default_factory=RetrievalFilter)
    candidate_limit: int = Field(default=10, ge=1, le=100)
    candidates: list[DocumentCandidate] = Field(default_factory=list)
    selected_evidence: list[PolicyEvidence] = Field(default_factory=list)
    error: str | None = None

    def set_selected_evidence(self, candidates: list[DocumentCandidate]) -> None:
        """Convert selected passages into the policy result's citation contract."""
        self.selected_evidence = [
            PolicyEvidence(
                doc_id=item.doc_id,
                version=item.version,
                chunk_id=item.chunk_id,
                excerpt=item.text,
                source_title=item.source_title,
                source_url=item.source_url,
                score=item.score,
            )
            for item in candidates
        ]
