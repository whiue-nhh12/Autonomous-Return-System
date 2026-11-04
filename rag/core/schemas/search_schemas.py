"""Shared result schema for keyword and vector retrieval backends."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    version: int
    chunk_id: str
    chunk_index: int
    text: str
    score: float
    metadata: dict[str, Any]