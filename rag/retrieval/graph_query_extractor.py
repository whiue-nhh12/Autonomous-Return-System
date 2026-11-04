"""Extract structured graph filters from a natural-language query."""

from __future__ import annotations

import json
import re

from rag.core.schemas.graphrag_schemas import QueryUnderstanding


class GraphQueryExtractor:
    def __init__(
        self,
        generator,
        *,
        max_tokens: int = 256,
        timeout_seconds: float = 10.0,
        aliases: dict | None = None,
    ) -> None:
        if max_tokens <= 0 or timeout_seconds <= 0:
            raise ValueError("LLM limits must be positive")
        self.generator = generator
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.aliases = aliases or {}

    def extract(self, query: str) -> QueryUnderstanding:
        product_categories = self._choices(
            "product_category",
            ["general_merchandise", "electronics", "clothing", "cosmetics", "food_and_beverage", "other"],
        )
        item_conditions = self._choices(
            "item_condition",
            ["UNUSED", "INTACT", "ORIGINAL_PACKAGING", "TAGS_ATTACHED", "SEALED"],
        )
        alias_prompt = self._format_aliases("product_category", "item_condition")
        prompt = (
            "Convert the user question into filters for a policy graph. Return JSON only with fields "
            "product, product_category, item_conditions, condition, customer_type, purchase_age_days, action. "
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
        response = self.generator.generate(prompt, max_tokens=self.max_tokens, timeout_seconds=self.timeout_seconds)
        return QueryUnderstanding.model_validate(self._parse_json(response))

    def _choices(self, group: str, fallback: list[str]) -> str:
        mappings = self.aliases.get(group)
        if not isinstance(mappings, dict) or not mappings:
            return ", ".join(fallback)
        return ", ".join(dict.fromkeys([*(str(key) for key in mappings), *fallback]))

    def _format_aliases(self, *groups: str) -> str:
        lines: list[str] = []
        for group in groups:
            mappings = self.aliases.get(group)
            if not isinstance(mappings, dict):
                continue
            lines.append(f"{group}:")
            for canonical, variants in mappings.items():
                values = variants if isinstance(variants, list) else [variants]
                if not values:
                    lines.append(f"- {canonical}")
                    continue
                lines.append(f"- {canonical}: {values[0]}")
                lines.extend(f"  - {value}" for value in values[1:])
        return "\n".join(lines) if lines else "(no alias mappings configured)"

    @staticmethod
    def _parse_json(response: str) -> dict:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
        payload = json.loads(cleaned)
        if not isinstance(payload, dict):
            raise ValueError("LLM response must be a JSON object")
        return payload
