"""Schemas for query routing and retrieval results."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from rag.core.schemas.graphrag_schemas import QueryResult, QueryUnderstanding


class Query(BaseModel):
    """Original query plus optional routing and retrieval-specific representations."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    rewritten_query: str | None = None
    status_worthy: list[Literal["HYPE", "QueryRewrite", "GraphRag", "Normal"]] = Field(default_factory=list)
    graph_query: QueryUnderstanding | None = None
    bm25_tokens: list[str] = Field(default_factory=list)


class RetrieveResult(BaseModel):
    """Graph and text retrieval results for one user query."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph_doc: list[QueryResult] = Field(default_factory=list)
    rag_doc: list[Any] = Field(default_factory=list)
    score: float = 0.0
