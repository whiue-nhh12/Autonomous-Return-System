"""Lightweight two-stage routing for user retrieval queries."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable

from rag.core.schemas.retrieval_schemas import Query


_REWRITE_TERMS = (
    "nó", "chúng", "cái kia", "cái này", "điều đó", "trường hợp đó", "họ", "ở đó",
)
_GRAPH_TERMS = (
    "đối với", "trong trường hợp", "nếu", "trừ khi", "điều kiện", "liên quan đến",
    "áp dụng cho", "được áp dụng", "yêu cầu", "ngoại lệ", "thời hạn", "trong vòng",
    "unless", "provided that", "subject to",
)
_ALLOWED_LABELS = {"HYPE", "QueryRewrite", "GraphRag", "Normal"}
_LOGGER = logging.getLogger(__name__)


class QueryClassifier:
    """Apply lexical routing first, then optionally ask an LLM for additional labels.

    Lexical labels are preserved when the LLM runs. That keeps strong rewrite and
    graph signals from being discarded by a single LLM classification response.
    """

    def __init__(self, generator=None, *, max_tokens: int = 256, timeout_seconds: float = 10.0) -> None:
        if max_tokens <= 0 or timeout_seconds <= 0:
            raise ValueError("LLM limits must be positive")
        self.generator = generator
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    def classify(self, query: str | Query, *, history: Iterable[str] | None = None) -> Query:
        result = query if isinstance(query, Query) else Query(query=query)
        text = result.rewritten_query or result.query
        folded = text.casefold()
        labels = set(result.status_worthy)
        if any(self._contains_term(folded, term) for term in _REWRITE_TERMS):
            labels.add("QueryRewrite")
        if any(self._contains_term(folded, term) for term in _GRAPH_TERMS):
            labels.add("GraphRag")

        if self.generator is not None:
            try:
                labels.update(self._classify_with_llm(text, history=history))
            except Exception:
                _LOGGER.exception("LLM query classification failed; using lexical labels")
        labels.discard("Normal")
        if not labels:
            labels.add("Normal")
        return result.model_copy(update={"status_worthy": [label for label in ("QueryRewrite", "GraphRag", "HYPE", "Normal") if label in labels]})

    @staticmethod
    def _contains_term(text: str, term: str) -> bool:
        return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, flags=re.IGNORECASE) is not None

    def _classify_with_llm(self, query: str, *, history: Iterable[str] | None) -> set[str]:
        context = "\n".join(str(item) for item in (history or []))
        prompt = (
            "Classify the retrieval query. Return JSON only as "
            '{"status_worthy": ["GraphRag"]}. Allowed labels: QueryRewrite, GraphRag, HYPE, Normal. '
            "Use QueryRewrite when resolving a pronoun or reference requires prior conversation context. "
            "Use GraphRag when answering requires conditions, constraints, or relationships among facts. "
            "Use Normal for ordinary document lookup. Use HYPE only when hypothetical-document expansion is useful. "
            "Return one or more labels and do not invent query details.\n"
            f"HISTORY:\n{context or '(none)'}\nQUERY:\n{query}"
        )
        response = self.generator.generate(
            prompt,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,
        )
        payload = self._parse_json(response)
        values = payload.get("status_worthy")
        if not isinstance(values, list) or any(value not in _ALLOWED_LABELS for value in values):
            raise ValueError("Query classifier returned invalid status_worthy labels")
        return set(values)

    @staticmethod
    def _parse_json(response: str) -> dict:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
        payload = json.loads(cleaned)
        if not isinstance(payload, dict):
            raise ValueError("LLM response must be a JSON object")
        return payload
