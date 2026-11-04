"""Conservative extraction of return facts explicitly present in a message."""

from __future__ import annotations

import re
import unicodedata

from agent.agents.return_decision.state.result import ReturnClaimFacts
from agent.core.state.shared import OrderLookupData


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


def extract_return_claim(message: str, orders: list[OrderLookupData] | None = None) -> ReturnClaimFacts:
    normalized = _normalize(message)
    order_number = next((match.group(1).strip() for pattern in _ORDER_PATTERNS if (match := pattern.search(message))), None)
    quantity = next((int(match.group(1)) for pattern in _QUANTITY_PATTERNS if (match := pattern.search(message))), None)
    reason_code = _first_reason(normalized)
    item_condition = _first_condition(normalized)
    item_query = None
    for order in orders or []:
        for item in order.items:
            candidates = (item.product_name, item.sku)
            if any(_normalize(candidate) in normalized for candidate in candidates if candidate.strip()):
                if item_query is not None:
                    item_query = None
                    break
                item_query = item.product_name
    return ReturnClaimFacts(
        order_number=order_number,
        item_query=item_query,
        requested_quantity=quantity,
        reason_code=reason_code,
        item_condition=item_condition,
    )


def _first_reason(text: str) -> str | None:
    mapping = (
        ("damaged", ("bi hong", "hu hong", "bi vo", "damaged", "broken")),
        ("wrong_item", ("sai san pham", "nham san pham", "wrong item", "incorrect item")),
        ("not_as_described", ("khong nhu mo ta", "khac mo ta", "not as described", "misdescribed")),
        ("changed_mind", ("doi y", "khong muon nua", "changed my mind", "no longer want")),
    )
    return next((code for code, terms in mapping if any(term in text for term in terms)), None)


def _first_condition(text: str) -> str | None:
    mapping = (
        ("unopened", ("chua mo", "unopened", "sealed")),
        ("opened", ("da mo", "opened")),
        ("used", ("da su dung", "used")),
        ("damaged", ("bi hong", "hu hong", "damaged", "broken")),
    )
    return next((code for code, terms in mapping if any(term in text for term in terms)), None)


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return plain.replace("đ", "d")
