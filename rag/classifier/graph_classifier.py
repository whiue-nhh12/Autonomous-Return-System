"""Two-stage graph-worthiness classification for cleaned content blocks."""

from __future__ import annotations

import json
import re
from time import perf_counter
from typing import Protocol
from uuid import uuid4

from rag.core.logger.rag.graphrag_logger import GraphRagLogger
from rag.core.schemas.cleaning_schemas import Cleaned_Document
from rag.core.schemas.graphrag_schemas import GraphProcess, GraphProcessStatus


_CONSTRAINT_TERMS = (
    "trong trường hợp",
    "trong khi",
    "nếu",
    "trừ khi",
    "điều kiện",
    "được áp dụng",
    "không áp dụng",
    "phải",
    "bắt buộc",
    "chỉ áp dụng",
    "thời hạn",
    "trong vòng",
    "để được",
    "unless",
    "provided that",
    "subject to",
    "must",
    "eligible",
    "within",
)
_LLM_BATCH_SIZE = 5


class TextGenerator(Protocol):
    def generate(self, prompt: str, *, max_tokens: int, timeout_seconds: float) -> str:
        ...


class GraphLogger(Protocol):
    def emit(
        self,
        *,
        trace_id: str,
        process: GraphProcess,
        status: GraphProcessStatus,
        message: str = "",
        metrics: dict | None = None,
    ) -> object:
        ...


class GraphClassifier:
    def __init__(
        self,
        *,
        generator: TextGenerator,
        logger: GraphLogger | None = None,
        trace_id: str | None = None,
        keyword_density_threshold: float = 0.015,
        minimum_keyword_hits: int = 1,
        max_tokens: int = 256,
        timeout_seconds: float = 10.0,
    ) -> None:
        if not 0.0 <= keyword_density_threshold <= 1.0:
            raise ValueError("keyword_density_threshold must be between 0 and 1")
        if minimum_keyword_hits < 1:
            raise ValueError("minimum_keyword_hits must be at least 1")
        if max_tokens <= 0 or timeout_seconds <= 0:
            raise ValueError("LLM limits must be positive")
        self.generator = generator
        self.logger = logger or GraphRagLogger()
        self.trace_id = trace_id or str(uuid4())
        self.keyword_density_threshold = keyword_density_threshold
        self.minimum_keyword_hits = minimum_keyword_hits
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    def run(self, cleaned: Cleaned_Document) -> Cleaned_Document:
        started = perf_counter()
        lexical_candidates = 0
        llm_calls = 0
        try:
            llm_candidates = []
            for block in cleaned.blocks:
                if block.graph_status in ("NORMAL", "GRAPH_WORTHY"):
                    continue
                text = block.contextual_text or block.raw_text
                if not text.strip():
                    block.graph_status = "NORMAL"
                    continue

                if self._passes_lexical_gate(text):
                    block.graph_status = "GRAPH_WORTHY"
                    lexical_candidates += 1
                    continue

                llm_candidates.append((block, text))

            for batch_start in range(0, len(llm_candidates), _LLM_BATCH_SIZE):
                batch = llm_candidates[batch_start : batch_start + _LLM_BATCH_SIZE]
                llm_calls += 1
                try:
                    decisions = self._classify_batch_with_llm(
                        [text for _, text in batch]
                    )
                except Exception:
                    for block, _ in batch:
                        block.graph_status = "FAILED"
                    raise
                for (block, _), decision in zip(batch, decisions):
                    block.graph_status = "GRAPH_WORTHY" if decision else "NORMAL"

            self.logger.emit(
                trace_id=self.trace_id,
                process="CLASSIFYING",
                status="SUCCESS",
                message="Đã phân loại block theo tiêu chí GraphRAG.",
                metrics={
                    "elapsed_ms": round((perf_counter() - started) * 1000, 2),
                    "block_count": len(cleaned.blocks),
                    "graph_worthy_count": sum(block.graph_status == "GRAPH_WORTHY" for block in cleaned.blocks),
                    "lexical_candidate_count": lexical_candidates,
                    "llm_call_count": llm_calls,
                },
            )
            return cleaned
        except Exception as exc:
            failed_count = sum(block.graph_status == "FAILED" for block in cleaned.blocks)
            self.logger.emit(
                trace_id=self.trace_id,
                process="CLASSIFYING",
                status="FAIL",
                message=f"Graph classification failed: {exc}",
                metrics={
                    "elapsed_ms": round((perf_counter() - started) * 1000, 2),
                    "failed_block_count": failed_count,
                    "llm_call_count": llm_calls,
                },
            )
            raise

    def _classify_batch_with_llm(self, texts: list[str]) -> list[bool]:
        if not texts or len(texts) > _LLM_BATCH_SIZE:
            raise ValueError(f"LLM batch must contain between 1 and {_LLM_BATCH_SIZE} texts")

        items = "\n\n".join(
            f"ITEM {index}:\n{text}" for index, text in enumerate(texts, start=1)
        )
        prompt = (
            "For each item, classify whether it contains an actionable policy, eligibility condition, "
            "requirement, exclusion, deadline, or conditional action useful as a graph fact. "
            "Do not classify general descriptions or headings as graph-worthy. "
            "Return only JSON in this form: "
            '{"results": [{"index": 1, "graph_worthy": true}, ...]}. '
            "Include exactly one result per item and preserve each item index.\n\n"
            f"{items}"
        )
        response = self.generator.generate(
            prompt,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,
        )
        payload = self._parse_json(response)
        results = payload.get("results")
        if not isinstance(results, list) or len(results) != len(texts):
            raise ValueError("LLM classification must return one result per input item")

        decisions: dict[int, bool] = {}
        for result in results:
            if not isinstance(result, dict):
                raise ValueError("Each LLM classification result must be a JSON object")
            index = result.get("index")
            decision = result.get("graph_worthy")
            if not isinstance(index, int) or isinstance(index, bool) or index not in range(1, len(texts) + 1):
                raise ValueError("LLM classification result has an invalid item index")
            if index in decisions:
                raise ValueError("LLM classification returned a duplicate item index")
            if not isinstance(decision, bool):
                raise ValueError("LLM classification must contain a boolean graph_worthy field")
            decisions[index] = decision
        if len(decisions) != len(texts):
            raise ValueError("LLM classification omitted one or more input items")
        return [decisions[index] for index in range(1, len(texts) + 1)]

    def _passes_lexical_gate(self, text: str) -> bool:
        folded = text.casefold()
        hit_count = sum(
            len(re.findall(rf"(?<!\w){re.escape(term)}(?!\w)", folded))
            for term in _CONSTRAINT_TERMS
        )
        word_count = max(len(re.findall(r"\w+", folded, flags=re.UNICODE)), 1)
        return (
            hit_count >= self.minimum_keyword_hits
            and hit_count / word_count >= self.keyword_density_threshold
        )

    @staticmethod
    def _parse_json(response: str) -> dict:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
        payload = json.loads(cleaned)
        if not isinstance(payload, dict):
            raise ValueError("LLM response must be a JSON object")
        return payload
