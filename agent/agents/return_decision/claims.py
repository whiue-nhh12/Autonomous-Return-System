"""Conservative extraction of return facts explicitly present in a message."""

from __future__ import annotations

import re
import unicodedata

from agent.agents.return_decision.state.result import ReturnClaimFacts
from agent.core.state.shared import OrderLookupData
from agent.orchestration.understanding import AgentUnderstanding


_ORDER_PATTERNS = (
    re.compile(r"\b(?:order\s*(?:number|no\.?|#)|order\s*#)\s*[:#]?\s*([A-Z0-9][A-Z0-9_-]{0,63})\b", re.I),
    re.compile(r"(?:mã\s*đơn(?:\s*hàng)?|đơn\s*hàng\s*(?:số|#))\s*[:#]?\s*([A-Z0-9][A-Z0-9_-]{0,63})\b", re.I),
    re.compile(r"\bđơn\s+(?!hàng\b)([A-Z0-9][A-Z0-9_-]{0,63})\b", re.I),
)
_QUANTITY_PATTERNS = (
    re.compile(r"(?:số\s*lượng|quantity)\s*[:#]?\s*(\d+)\b", re.I),
    re.compile(r"\b(\d+)\s*(?:sản\s*phẩm|món|cái|items?|units?)\b", re.I),
    re.compile(r"\bx\s*(\d+)\b", re.I),
    re.compile(r"\b(\d+)\s*x\b", re.I),
)
_REASON_TERMS = {
    "damaged": ("bi hong", "hu hong", "bi vo", "damaged", "broken"),
    "wrong_item": ("sai san pham", "nham san pham", "wrong item", "incorrect item"),
    "not_as_described": (
        "khong nhu mo ta",
        "khac mo ta",
        "not as described",
        "misdescribed",
    ),
    "changed_mind": ("doi y", "khong muon nua", "changed my mind", "no longer want"),
    "other": ("ly do khac", "other reason"),
}
_CONDITION_TERMS = {
    "unopened": ("chua mo", "unopened", "sealed"),
    "opened": ("da mo", "opened"),
    "used": ("da su dung", "used"),
    "damaged": ("bi hong", "hu hong", "damaged", "broken"),
}
_METHOD_TERMS = {
    "pickup": ("pickup", "pick up", "nhan hang tai nha", "den lay"),
    "drop_off": ("drop off", "drop-off", "gui tai diem", "mang toi diem"),
    "self_ship": ("self ship", "ship it myself", "tu gui", "tu gui hang"),
}


def extract_return_claim(
    message: str,
    orders: list[OrderLookupData] | None = None,
    *,
    understanding: AgentUnderstanding | None = None,
    seed: ReturnClaimFacts | None = None,
    prior_messages: list[str] | None = None,
    item_context: str | None = None,
) -> ReturnClaimFacts:
    """Extract explicit facts, then fill gaps with high-confidence structured output."""
    normalized = _normalize(message)
    order_match = _last_pattern_match(message, _ORDER_PATTERNS)
    quantity_match = _last_pattern_match(message, _QUANTITY_PATTERNS)
    order_number = (
        order_match.group(1).strip()
        if order_match and not _is_negated(normalized, _normalize(order_match.group(0)))
        else None
    )
    quantity = (
        int(quantity_match.group(1))
        if quantity_match
        and not _is_negated(normalized, _normalize(quantity_match.group(0)))
        else None
    )
    reason_candidates = _matched_codes(normalized, _REASON_TERMS)
    condition_candidates = _matched_codes(normalized, _CONDITION_TERMS)
    method_candidates = _matched_codes(normalized, _METHOD_TERMS)
    reason_code = reason_candidates[0] if len(reason_candidates) == 1 else None
    item_condition = condition_candidates[0] if len(condition_candidates) == 1 else None
    return_method = method_candidates[0] if len(method_candidates) == 1 else None
    current_item_query, current_item_matches = _match_order_item(message, orders or [])
    item_query = current_item_query
    if current_item_matches == 0 and item_context:
        item_query, _ = _match_order_item(item_context, orders or [])
    explicit_clears: set[str] = set()
    if len(reason_candidates) > 1:
        explicit_clears.add("reason_code")
    if len(condition_candidates) > 1:
        explicit_clears.add("item_condition")
    if len(method_candidates) > 1:
        explicit_clears.add("return_method")
    if order_match and order_number is None:
        explicit_clears.add("order_number")
    if quantity_match and quantity is None:
        explicit_clears.add("requested_quantity")
    if current_item_matches > 1:
        explicit_clears.add("item_query")
    parsed = ReturnClaimFacts(
        order_number=order_number,
        item_query=item_query,
        requested_quantity=quantity,
        reason_code=reason_code,
        item_condition=item_condition,
        return_method=return_method,
    )
    if seed is not None:
        cleared_fields = set(explicit_clears)
        if seed.reason_code and _is_explicitly_negated(
            normalized, _terms_for_reason(seed.reason_code)
        ):
            parsed = parsed.model_copy(update={"reason_code": None})
            cleared_fields.add("reason_code")
        if seed.item_condition and _is_explicitly_negated(
            normalized, _terms_for_condition(seed.item_condition)
        ):
            parsed = parsed.model_copy(update={"item_condition": None})
            cleared_fields.add("item_condition")
        if seed.return_method and _is_explicitly_negated(
            normalized, _terms_for_method(seed.return_method)
        ):
            parsed = parsed.model_copy(update={"return_method": None})
            cleared_fields.add("return_method")
        model_derived_fields = set(seed.model_derived_fields)
        for field in (
            "order_number",
            "item_query",
            "requested_quantity",
            "reason_code",
            "item_condition",
            "return_method",
        ):
            if getattr(parsed, field) is not None or field in cleared_fields:
                model_derived_fields.discard(field)
        updates = {
            field: getattr(seed, field)
            for field in (
                "order_number",
                "order_item_id",
                "item_query",
                "requested_quantity",
                "reason_code",
                "item_condition",
                "return_method",
            )
            if (
                getattr(parsed, field) is None
                and getattr(seed, field) is not None
                and field not in cleared_fields
            )
        }
        updates["model_derived_fields"] = model_derived_fields
        parsed = parsed.model_copy(update=updates, deep=True)
    if understanding is not None:
        parsed = understanding.enrich_claim(
            message,
            parsed,
            prior_messages=prior_messages,
        )
    return parsed


def _match_order_item(
    message: str,
    orders: list[OrderLookupData],
) -> tuple[str | None, int]:
    normalized = _normalize(message)
    matches = {}
    for order in orders:
        for item in order.items:
            candidates = (item.product_name, item.sku)
            if any(
                _normalize(candidate) in normalized
                for candidate in candidates
                if candidate.strip()
            ):
                matches[item.order_item_id] = item.product_name
    if len(matches) == 1:
        return next(iter(matches.values())), 1
    return None, len(matches)


def _last_pattern_match(message: str, patterns: tuple[re.Pattern[str], ...]):
    matches = [match for pattern in patterns for match in pattern.finditer(message)]
    return max(matches, key=lambda match: match.start(), default=None)


def _matched_codes(text: str, mapping: dict[str, tuple[str, ...]]) -> list[str]:
    return [
        code
        for code, terms in mapping.items()
        if any(_contains_phrase(text, term) and not _is_negated(text, term) for term in terms)
    ]


def _is_negated(text: str, term: str) -> bool:
    """Avoid turning explicit negations into positive return facts."""
    pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
    for match in re.finditer(pattern, text):
        preceding_words = text[max(0, match.start() - 40) : match.start()]
        if re.search(
            r"(?:\bnot\b|\bno\b|\bkhong\b|\bchua\b)(?:\s+\w+){0,2}\s*$",
            preceding_words,
        ):
            return True
    return False


def _is_explicitly_negated(text: str, terms: tuple[str, ...]) -> bool:
    return any(_contains_phrase(text, term) and _is_negated(text, term) for term in terms)


def _contains_phrase(text: str, phrase: str) -> bool:
    pattern = rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])"
    return re.search(pattern, text) is not None


def _terms_for_reason(reason_code: str) -> tuple[str, ...]:
    return _REASON_TERMS.get(reason_code, ())


def _terms_for_condition(condition: str) -> tuple[str, ...]:
    return _CONDITION_TERMS.get(condition, ())


def _terms_for_method(method: str) -> tuple[str, ...]:
    return _METHOD_TERMS.get(method, ())


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return plain.replace("đ", "d")
