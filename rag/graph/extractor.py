"""Structured entity and policy-rule extraction."""

from __future__ import annotations

import json
import re
from pathlib import Path
from time import perf_counter
from typing import Protocol
from uuid import uuid4

from rag.core.logger.rag.graphrag_logger import GraphRagLogger
from rag.core.schemas.graphrag_schemas import (
    ExtractionResult,
    GraphProcess,
    GraphProcessStatus,
    ItemCondition,
    PolicyRuleType,
    ProductCategory,
    QueryUnderstanding,
    ReturnReason,
    ShippingFeePayer,
)


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


class GraphExtractor:
    def __init__(
        self,
        *,
        generator,
        max_tokens: int = 2048,
        timeout_seconds: float = 30.0,
        aliases: dict | None = None,
        logger: GraphLogger | None = None,
        trace_id: str | None = None,
    ) -> None:
        if max_tokens <= 0 or timeout_seconds <= 0:
            raise ValueError("LLM limits must be positive")
        self.generator = generator
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.alias = aliases if aliases is not None else self._load_aliases()
        self.logger = logger or GraphRagLogger()
        self.trace_id = trace_id or str(uuid4())

    def extract(self, *, chunk_id: str, text: str) -> ExtractionResult:
        started = perf_counter()
        rule_type_values = self._enum_values("rule_type", [item.value for item in PolicyRuleType])
        return_reason_values = self._enum_values("return_reason", [item.value for item in ReturnReason])
        product_category_values = self._enum_values(
            "product_category", [item.value for item in ProductCategory]
        )
        item_condition_values = self._enum_values("item_condition", [item.value for item in ItemCondition])
        shipping_payer_values = self._enum_values("who_pay_ship", [item.value for item in ShippingFeePayer])
        alias_prompt = self._format_aliases()
        prompt = (
            "Extract only facts explicitly supported by the source text. Return one JSON object with "
            "exactly these top-level fields: chunk_id, entities, rules. Each entity must contain "
            "id (empty string), name, canonical_name (same as name before normalization), "
            "source_chunk_id (use the provided chunk_id), and source_chunks (containing that ID). "
            "Each rule must contain id (empty string), rule_type, return_reason, product_category, product, "
            "item_conditions, condition, customer_type, action, "
            "who_pay_ship, "
            "window_text, window_min, window_max, requirements, exclusions, source_chunks, and source_chunk_id. "
            "window_text must preserve the exact source wording for the time limit, or be null if none is stated. "
            "window_min and window_max must be non-negative JSON integers or null, in the unit stated in window_text. "
            "For a single stated duration, set both bounds to that value; for a range, set its lower and upper bounds; "
            "for an upper-only limit such as 'within 24 hours', set only window_max; for a lower-only limit, set only "
            "window_min. If the wording is non-numeric or its bounds cannot be determined, use null for both bounds. "
            "Never convert hours, business days, or other units to calendar days. Use null "
            "when a scalar field is not stated. requirements, exclusions, item_conditions, "
            "source_chunks must always be JSON arrays, even when they contain only one value; "
            "use empty arrays when there are no list items. "
            f"rule_type must be one of {rule_type_values}; "
            "use other if the type cannot be determined. "
            f"return_reason must be one of {return_reason_values}; "
            "Use null if the source does not state a return reason. "
            f"product_category must be one of {product_category_values}; "
            "Use other when a product category is present but does not fit a listed category; "
            "use null when no product category can be determined. "
            f"item_conditions must be an array containing only {item_condition_values}; "
            "use an empty array when none are stated. "
            "Preserve the original product wording in product and the source condition wording in condition. "
            f"who_pay_ship must be one of {shipping_payer_values}, and must be null unless the source explicitly says who pays shipping. "
            "Map the canonical labels using these aliases:\n"
            f"{alias_prompt}\n"
            "Do not invent missing policy details. Use empty arrays when no entities/rules exist.\n"
            f"chunk_id: {chunk_id}\nSOURCE TEXT:\n{text}"
        )
        stage = "generating_response"
        response_length = 0
        try:
            response = self.generator.generate(
                prompt,
                max_tokens=self.max_tokens,
                timeout_seconds=self.timeout_seconds,
            )
            response_length = len(response)
            stage = "parsing_json"
            payload = self._parse_json(response)
            stage = "validating_payload"
            if not isinstance(payload, dict):
                raise ValueError("LLM extraction response must be a JSON object")
            payload.setdefault("chunk_id", chunk_id)
            if payload["chunk_id"] != chunk_id:
                raise ValueError("LLM extraction returned an unexpected chunk_id")
            for entity in payload.get("entities", []):
                entity.setdefault("id", "")
                entity.setdefault("canonical_name", entity.get("name", ""))
                entity.setdefault("source_chunk_id", chunk_id)
                entity.setdefault("source_chunks", [chunk_id])
                self._normalize_list_fields(entity, "source_chunks")
            for rule in payload.get("rules", []):
                rule.setdefault("id", "")
                rule.setdefault("source_chunk_id", chunk_id)
                rule.setdefault("source_chunks", [chunk_id])
                self._normalize_list_fields(
                    rule,
                    "item_conditions",
                    "requirements",
                    "exclusions",
                    "source_chunks",
                )
            extraction = ExtractionResult.model_validate(payload)
        except Exception as exc:
            self.logger.emit(
                trace_id=self.trace_id,
                process="EXTRACTING",
                status="FAIL",
                message=f"Graph extraction failed during {stage}: {type(exc).__name__}: {exc}",
                metrics={
                    "chunk_id": chunk_id,
                    "elapsed_ms": round((perf_counter() - started) * 1000, 2),
                    "input_char_count": len(text),
                    "response_char_count": response_length,
                },
            )
            raise

        self.logger.emit(
            trace_id=self.trace_id,
            process="EXTRACTING",
            status="SUCCESS",
            message="LLM extraction completed and response validated.",
            metrics={
                "chunk_id": chunk_id,
                "elapsed_ms": round((perf_counter() - started) * 1000, 2),
                "input_char_count": len(text),
                "response_char_count": response_length,
                "entity_count": len(extraction.entities),
                "rule_count": len(extraction.rules),
            },
        )
        return extraction

    def extract_query(self, query: str) -> QueryUnderstanding:
        """Convert a user query into canonical filters for graph retrieval."""
        product_categories = self._enum_values(
            "product_category", [item.value for item in ProductCategory]
        )
        item_conditions = self._enum_values(
            "item_condition", [item.value for item in ItemCondition]
        )
        alias_prompt = self._format_aliases("product_category", "item_condition")
        prompt = (
            "Convert the user question into filters for a policy graph. Return JSON only with fields "
            "product, product_category, item_conditions, condition, "
            "customer_type, purchase_age_days, action. "
            "Use null for any scalar field not explicitly present; do not infer missing values. "
            f"product_category must be one of {product_categories} or null. "
            f"item_conditions must be an array using only {item_conditions}; "
            "use an empty array when no item condition is stated. "
            "Use these alias mappings to map query wording to canonical values:\n"
            f"{alias_prompt}\n"
            "Preserve literal product/condition wording in product and condition. "
            "purchase_age_days must be a non-negative integer or null.\n"
            f"QUESTION:\n{query}"
        )
        response = self.generator.generate(
            prompt,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,
        )
        return QueryUnderstanding.model_validate(self._parse_json(response))

    @staticmethod
    def _load_aliases() -> dict:
        alias_path = Path(__file__).resolve().parents[1] / "data" / "policy_aliases.json"
        with alias_path.open("r", encoding="utf-8") as file:
            aliases = json.load(file)
        if not isinstance(aliases, dict):
            raise ValueError(f"Alias file must contain a JSON object: {alias_path}")
        return aliases

    def _format_aliases(self, *groups: str) -> str:
        lines: list[str] = []
        selected_groups = groups or tuple(self.alias)
        for group in selected_groups:
            mappings = self.alias.get(group)
            if not isinstance(mappings, dict):
                continue
            lines.append(f"{group}:")
            for canonical, aliases in mappings.items():
                variants = [str(alias) for alias in aliases] if isinstance(aliases, list) else [str(aliases)]
                if not variants:
                    lines.append(f"- {canonical}")
                    continue
                lines.append(f"- {canonical}: {variants[0]}")
                lines.extend(f"  - {alias}" for alias in variants[1:])
        return "\n".join(lines) if lines else "(no alias mappings configured)"

    def _enum_values(self, group: str, fallback: list[str]) -> str:
        mappings = self.alias.get(group)
        if isinstance(mappings, dict) and mappings:
            values = list(dict.fromkeys([*(str(value) for value in mappings), *fallback]))
            return ", ".join(values)
        return ", ".join(fallback)

    @staticmethod
    def _normalize_list_fields(payload: dict, *field_names: str) -> None:
        """Repair common LLM shape errors before schema validation."""
        for field_name in field_names:
            value = payload.get(field_name)
            if value is None:
                payload[field_name] = []
            elif isinstance(value, str):
                cleaned = value.strip()
                payload[field_name] = [cleaned] if cleaned else []

    @staticmethod
    def _parse_json(response: str) -> dict:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
        return json.loads(cleaned)
