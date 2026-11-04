"""Hypothetical document generation used only for dense retrieval."""

from __future__ import annotations

from typing import Protocol

from rag.retrieval.cache import TTLCache
from rag.core.llm.text_generation import TextGenerator


class HyDEGenerator(Protocol):
    def generate(self, query: str) -> str:
        ...


class LLMHyDEGenerator:
    """Generate short hypothetical document text, not a user-facing answer."""

    def __init__(
        self,
        generator: TextGenerator,
        *,
        max_tokens: int = 256,
        timeout_seconds: float = 5.0,
        cache_ttl_seconds: float = 900.0,
        cache_size: int = 1000,
    ) -> None:
        if max_tokens <= 0 or timeout_seconds <= 0:
            raise ValueError("max_tokens and timeout_seconds must be positive")
        self.generator = generator
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self._cache = TTLCache(ttl_seconds=cache_ttl_seconds, max_entries=cache_size)

    def generate(self, query: str) -> str:
        cached, cached_document = self._cache.get(query)
        if cached:
            return cached_document
        prompt = (
            "Write one short hypothetical passage that could appear in a source document answering the query. "
            "Do not answer conversationally, mention that this is hypothetical, or add unsupported specifics. "
            "Preserve identifiers and technical terms exactly. Return only the passage.\n"
            f"Query: {query}"
        )
        document = self.generator.generate(
            prompt,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,
        ).strip()
        if document:
            self._cache.put(query, document)
        return document