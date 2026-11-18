"""Manager intent routing and fan-out helpers."""

from typing import Literal
import unicodedata
import re

from langgraph.types import Send

from agent.orchestration.state import ManagerGraphState
from agent.core.state.shared import ReturnIntent


RouteName = Literal["policy", "order", "eligibility", "submit", "clarify"]


def classify_intent(message: str) -> tuple[ReturnIntent, float]:
    """Classify clear requests lexically and surface mixed intents as ambiguous."""
    text = _normalize(message)
    submit_terms = (
        "submit return", "create a return", "start a return", "request a return",
        "i want to return", "please return", "send a return request",
        "gui yeu cau tra", "tao yeu cau tra", "dang ky tra hang", "gui tra hang",
        "toi muon tra", "muon tra hang",
    )
    eligibility_terms = (
        "eligible", "eligibility", "can i return", "can this be returned",
        "return eligibility", "would i be able to return", "dieu kien tra hang",
        "du dieu kien", "co duoc tra", "tra duoc khong",
        "duoc doi tra khong", "co duoc doi tra",
    )
    order_terms = (
        "order status", "track my order", "where is my order", "order number",
        "order #", "order no",
        "order history", "order list", "my orders", "all orders", "past orders", "orders",
        "latest order", "most recent order", "newest order", "last order",
        "don hang", "ma don", "kiem tra don", "theo doi don", "lich su don",
        "danh sach don", "cac don", "don moi nhat", "don hang gan nhat",
    )
    policy_terms = (
        "policy", "policies", "policy question", "refund", "return policy", "warranty", "exchange",
        "chinh sach", "hoan tien", "bao hanh", "doi tra", "doi hang",
    )
    matches: list[ReturnIntent] = []
    if _has_term(text, submit_terms):
        matches.append(ReturnIntent.SUBMIT_RETURN)
    if _has_term(text, eligibility_terms):
        matches.append(ReturnIntent.RETURN_ELIGIBILITY)
    if _has_term(text, order_terms):
        matches.append(ReturnIntent.ORDER_STATUS)
    if _has_term(text, policy_terms):
        matches.append(ReturnIntent.POLICY_QUESTION)
    if ReturnIntent.RETURN_ELIGIBILITY in matches:
        # Eligibility requests already fan out to both the order and policy workers.
        return ReturnIntent.RETURN_ELIGIBILITY, 0.85
    if len(matches) == 1:
        intent = matches[0]
        confidence = {
            ReturnIntent.SUBMIT_RETURN: 0.9,
            ReturnIntent.RETURN_ELIGIBILITY: 0.85,
            ReturnIntent.ORDER_STATUS: 0.85,
            ReturnIntent.POLICY_QUESTION: 0.8,
        }[intent]
        return intent, confidence
    if len(matches) > 1:
        return ReturnIntent.UNCLEAR, 0.55
    return ReturnIntent.UNCLEAR, 0.4


def classify_clarification_reply(message: str) -> ReturnIntent | None:
    """Map a short answer to the explicit options in the manager's clarify prompt."""
    normalized = _normalize(message).strip(" .!?;,:")
    options = {
        "policy": ReturnIntent.POLICY_QUESTION,
        "policies": ReturnIntent.POLICY_QUESTION,
        "chinh sach": ReturnIntent.POLICY_QUESTION,
        "order": ReturnIntent.ORDER_STATUS,
        "don hang": ReturnIntent.ORDER_STATUS,
        "eligibility": ReturnIntent.RETURN_ELIGIBILITY,
        "return eligibility": ReturnIntent.RETURN_ELIGIBILITY,
        "du dieu kien": ReturnIntent.RETURN_ELIGIBILITY,
        "submit": ReturnIntent.SUBMIT_RETURN,
        "submit return": ReturnIntent.SUBMIT_RETURN,
        "gui yeu cau tra hang": ReturnIntent.SUBMIT_RETURN,
    }
    return options.get(normalized)


def route_intent(state: ManagerGraphState) -> RouteName:
    intent = state["shared"].classification.intent
    if intent is ReturnIntent.POLICY_QUESTION:
        return "policy"
    if intent is ReturnIntent.ORDER_STATUS:
        return "order"
    if intent is ReturnIntent.RETURN_ELIGIBILITY:
        return "eligibility"
    if intent is ReturnIntent.SUBMIT_RETURN:
        return "submit"
    return "clarify"


def send_eligibility_workers(state: ManagerGraphState) -> list[Send]:
    """Fan out independent policy and order lookups with separate state copies."""
    shared = state["shared"]
    return [
        Send("eligibility_policy_worker", {"shared": shared.model_copy(deep=True)}),
        Send("eligibility_order_worker", {"shared": shared.model_copy(deep=True)}),
    ]


def route_after_return_lookups(state: ManagerGraphState) -> Literal["eligibility", "submit"]:
    """Keep eligibility read-only; only an explicit submit intent persists a request."""
    if state["shared"].classification.intent is ReturnIntent.SUBMIT_RETURN:
        return "submit"
    return "eligibility"


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    return without_marks.replace("đ", "d")


def _has_term(text: str, terms: tuple[str, ...]) -> bool:
    for term in terms:
        normalized_term = _normalize(term)
        end_boundary = r"(?![a-z0-9])" if normalized_term[-1:].isalnum() else ""
        pattern = rf"(?<![a-z0-9]){re.escape(normalized_term)}{end_boundary}"
        if re.search(pattern, text):
            return True
    return False
