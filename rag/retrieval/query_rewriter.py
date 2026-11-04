"""LLM-backed query rewriting for lexical and dense retrieval."""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol

from rag.retrieval.cache import TTLCache
from rag.core.llm.text_generation import TextGenerator

_LOGGER = logging.getLogger(__name__)
_IDENTIFIER_PATTERN = re.compile(r"(?<!\w)[A-Za-z0-9][A-Za-z0-9._/-]*\d[A-Za-z0-9._/-]*(?!\w)")
_ACRONYM_PATTERN = re.compile(r"(?<!\w)[A-ZÀ-Ỹ][A-ZÀ-Ỹ0-9._/-]{1,}(?!\w)")
_QUOTED_PATTERN = re.compile(r"[\"'“‘]([^\"'”’]+)[\"'”’]")
_NAME_PATTERN = re.compile(r"\b[A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)+")


class QueryRewriter(Protocol):
    def rewrite(self, query: str, *, max_queries: int, history: list[str] | None = None) -> list[str]:
        ...


class LLMQueryRewriter:
    """Generate at most two alternate search queries, never answers."""

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

    def rewrite(
        self,
        query: str,
        *,
        max_queries: int = 2,
        history: list[str] | None = None,
    ) -> list[str]:
        if max_queries <= 0:
            return []
        if max_queries > 2:
            raise ValueError("max_queries must not exceed two")
        history = history or []
        history_context = "\n".join(history) if history else "(none)"
        cache_key = (query, max_queries, tuple(history))
        cached, cached_queries = self._cache.get(cache_key)
        if cached:
            return list(cached_queries)
        prompt = (
            "Rewrite the search query into at most "
            f"{max_queries} concise Vietnamese search queries. Do not answer it. "
            "Preserve every identifier, number, quoted phrase, named entity, and technical term exactly; "
            "do not translate or invent facts. Return only a JSON array of strings.\n"
            f"Conversation history:\n{history_context}\n"
            f"Query: {query}"
        )
        response = self.generator.generate(
            prompt,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,
        )
        parsed = json.loads(self._strip_json_fence(response))
        if not isinstance(parsed, list):
            raise TypeError("Query rewriter response must be a JSON array")

        required_terms = set(_IDENTIFIER_PATTERN.findall(query))
        required_terms.update(_ACRONYM_PATTERN.findall(query))
        required_terms.update(_QUOTED_PATTERN.findall(query))
        required_terms.update(_NAME_PATTERN.findall(query))
        results: list[str] = []
        seen = {query.strip().casefold()}
        for value in parsed:
            if not isinstance(value, str):
                continue
            candidate = value.strip()
            if not candidate or candidate.casefold() in seen:
                continue
            candidate_terms = set(_IDENTIFIER_PATTERN.findall(candidate))
            candidate_terms.update(_ACRONYM_PATTERN.findall(candidate))
            candidate_terms.update(_QUOTED_PATTERN.findall(candidate))
            candidate_terms.update(_NAME_PATTERN.findall(candidate))
            if not required_terms.issubset(candidate_terms):
                _LOGGER.warning("Ignoring rewritten query that dropped a protected term")
                continue
            results.append(candidate)
            seen.add(candidate.casefold())
            if len(results) >= max_queries:
                break
        self._cache.put(cache_key, results)
        return results

    @staticmethod
    def _strip_json_fence(response: str) -> str:
        value = response.strip()
        if value.startswith("```"):
            value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.IGNORECASE)
        return value
