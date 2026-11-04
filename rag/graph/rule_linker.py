"""Conflict-aware policy-rule candidate ranking and merging."""

from __future__ import annotations

from difflib import SequenceMatcher

from rag.core.schemas.graphrag_schemas import PolicyRule, RuleLinkCandidate, RuleLinkResult
from rag.graph.normalizer import GraphNormalizer


_WEIGHTS = {
    "condition": 0.55,
    "product": 0.15,
    "customer_type": 0.10,
    "action": 0.15,
    "window_text": 0.05,
}
_CONFLICT_FIELDS = (
    "rule_type",
    "return_reason",
    "product_category",
    "product",
    "item_conditions",
    "customer_type",
    "action",
    "who_pay_ship",
    "window_text",
    "window_min",
    "window_max",
)


class RuleLinker:
    def __init__(self, *, threshold: float = 0.82) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = threshold

    def link(self, new_rule: PolicyRule, candidates: list[PolicyRule]) -> RuleLinkResult:
        ranked = sorted(
            (
                RuleLinkCandidate(
                    rule_id=candidate.id,
                    score=1.0 if candidate.id == new_rule.id else self.score(new_rule, candidate),
                    reasons=(
                        ["stable rule ID matches"]
                        if candidate.id == new_rule.id
                        else self._reasons(new_rule, candidate)
                    ),
                )
                for candidate in candidates
                if not self._conflicts(new_rule, candidate)
            ),
            key=lambda item: (-item.score, item.rule_id),
        )
        if not ranked or ranked[0].score < self.threshold:
            return RuleLinkResult(decision="NEW", reasons=["No non-conflicting candidate met the merge threshold"])
        selected = ranked[0]
        return RuleLinkResult(
            decision="MERGE",
            candidate_rule_id=selected.rule_id,
            score=selected.score,
            reasons=selected.reasons,
        )

    @classmethod
    def score(cls, left: PolicyRule, right: PolicyRule) -> float:
        # PolicyRule.condition is optional; rules without a condition cannot be
        # reliably compared for merging.
        if not left.condition or not right.condition:
            return 0.0
        if not cls._normalize(left.condition) or not cls._normalize(right.condition):
            return 0.0
        numerator = 0.0
        denominator = 0.0
        for field, weight in _WEIGHTS.items():
            left_value = getattr(left, field)
            right_value = getattr(right, field)
            if left_value is None and right_value is None:
                continue
            denominator += weight
            if left_value is None or right_value is None:
                continue
            similarity = SequenceMatcher(
                None,
                GraphNormalizer.normalize_text(str(left_value)),
                GraphNormalizer.normalize_text(str(right_value)),
            ).ratio()
            numerator += weight * similarity
        return numerator / denominator if denominator else 0.0

    @classmethod
    def _conflicts(cls, left: PolicyRule, right: PolicyRule) -> bool:
        for field in _CONFLICT_FIELDS:
            left_value = getattr(left, field)
            right_value = getattr(right, field)
            if left_value is None or right_value is None:
                continue
            if isinstance(left_value, list) and isinstance(right_value, list):
                left_items = {getattr(value, "value", value) for value in left_value}
                right_items = {getattr(value, "value", value) for value in right_value}
                if left_items and right_items and left_items != right_items:
                    return True
                continue
            left_text = getattr(left_value, "value", left_value)
            right_text = getattr(right_value, "value", right_value)
            if cls._normalize(str(left_text)) != cls._normalize(str(right_text)):
                return True
        return False

    @classmethod
    def _reasons(cls, left: PolicyRule, right: PolicyRule) -> list[str]:
        reasons = []
        for field in _WEIGHTS:
            left_value = getattr(left, field)
            right_value = getattr(right, field)
            if left_value is not None and right_value is not None:
                if cls._normalize(str(left_value)) == cls._normalize(str(right_value)):
                    reasons.append(f"{field} matches")
        return reasons or ["Weighted field similarity"]

    @staticmethod
    def merge(existing: PolicyRule, incoming: PolicyRule) -> PolicyRule:
        def merge_list(first: list[str], second: list[str]) -> list[str]:
            return list(dict.fromkeys([*first, *second]))

        return existing.model_copy(
            update={
                "rule_type": existing.rule_type or incoming.rule_type,
                "return_reason": existing.return_reason or incoming.return_reason,
                "product_category": existing.product_category or incoming.product_category,
                "product": existing.product or incoming.product,
                "item_conditions": merge_list(existing.item_conditions, incoming.item_conditions),
                "condition": existing.condition or incoming.condition,
                "customer_type": existing.customer_type or incoming.customer_type,
                "action": existing.action or incoming.action,
                "who_pay_ship": existing.who_pay_ship or incoming.who_pay_ship,
                "window_text": existing.window_text or incoming.window_text,
                "window_min": existing.window_min if existing.window_min is not None else incoming.window_min,
                "window_max": existing.window_max if existing.window_max is not None else incoming.window_max,
                "requirements": merge_list(existing.requirements, incoming.requirements),
                "exclusions": merge_list(existing.exclusions, incoming.exclusions),
                "source_chunks": merge_list(existing.source_chunks, incoming.source_chunks),
                "source_chunk_id": existing.source_chunk_id or incoming.source_chunk_id,
            }
        )

    @staticmethod
    def _normalize(value: str) -> str:
        return GraphNormalizer.normalize_text(value)
