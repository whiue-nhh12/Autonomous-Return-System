"""Canonicalize extracted entities and rules."""

from __future__ import annotations

import hashlib
import re
import unicodedata

from rag.core.schemas.graphrag_schemas import Entity, ExtractionResult, PolicyRule


class GraphNormalizer:
    def __init__(self, aliases: dict[str, str] | None = None) -> None:
        self.aliases = {
            self.normalize_text(alias): self._clean(canonical)
            for alias, canonical in (aliases or {}).items()
        }

    def normalize(self, extraction: ExtractionResult) -> ExtractionResult:
        entities: list[Entity] = []
        for entity in extraction.entities:
            name = self._clean(entity.name)
            canonical = self.aliases.get(self.normalize_text(name), self._clean(entity.canonical_name or name))
            source_chunks = [extraction.chunk_id]
            entities.append(
                entity.model_copy(
                    update={
                        "id": self._stable_id("entity", canonical),
                        "name": name,
                        "canonical_name": canonical,
                        "source_chunk_id": extraction.chunk_id,
                        "source_chunks": source_chunks,
                    }
                )
            )

        rules: list[PolicyRule] = []
        for rule in extraction.rules:
            values = {
                "rule_type": rule.rule_type,
                "return_reason": rule.return_reason,
                "product_category": rule.product_category,
                "product": self._canonical_field(rule.product),
                "item_conditions": rule.item_conditions,
                "condition": self._canonical_field(rule.condition),
                "customer_type": self._canonical_field(rule.customer_type),
                "action": self._canonical_field(rule.action),
                "who_pay_ship": rule.who_pay_ship,
                "window_text": rule.window_text,
                "window_min": rule.window_min,
                "window_max": rule.window_max,
                "requirements": self._unique([self._clean(item) for item in rule.requirements if self._clean(item)]),
                "exclusions": self._unique([self._clean(item) for item in rule.exclusions if self._clean(item)]),
                "source_chunks": [extraction.chunk_id],
                "source_chunk_id": extraction.chunk_id,
            }
            stable_content = "|".join(
                str(getattr(values[key], "value", values[key]) or "")
                for key in (
                    "rule_type",
                    "return_reason",
                    "product_category",
                    "product",
                    "item_conditions",
                    "condition",
                    "customer_type",
                    "action",
                    "who_pay_ship",
                    "window_text",
                    "window_min",
                    "window_max",
                )
            )
            rules.append(
                rule.model_copy(
                    update={
                        **values,
                        "id": self._stable_id("rule", stable_content),
                    }
                )
            )
        return extraction.model_copy(update={"entities": entities, "rules": rules})

    @staticmethod
    def normalize_text(text: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", text).casefold().split())

    def _canonical_field(self, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        cleaned = self._clean(value)
        return self.aliases.get(self.normalize_text(cleaned), cleaned)

    @staticmethod
    def _clean(value: str) -> str:
        return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()

    @staticmethod
    def _stable_id(kind: str, value: str) -> str:
        digest = hashlib.sha256(GraphNormalizer.normalize_text(value).encode("utf-8")).hexdigest()[:32]
        return f"{kind}_{digest}"

    @staticmethod
    def _unique(values: list[str | None]) -> list[str]:
        return list(dict.fromkeys(value for value in values if value))
