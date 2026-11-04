"""Manager intent routing and fan-out helpers."""

from typing import Literal
import unicodedata

from langgraph.types import Send

from agent.orchestration.state import ManagerGraphState
from agent.core.state.shared import ReturnIntent


RouteName = Literal["policy", "order", "eligibility", "submit", "clarify"]


def classify_intent(message: str) -> tuple[ReturnIntent, float]:
    """Classify common Vietnamese and English requests using deterministic terms."""
    text = _normalize(message)
    submit_terms = (
        "submit return", "create a return", "start a return", "return this item",
        "gui yeu cau tra", "tao yeu cau tra", "dang ky tra hang", "gui tra hang",
    )
    eligibility_terms = (
        "eligible", "eligibility", "can i return", "can this be returned",
        "du dieu kien", "co duoc tra", "tra duoc khong", "duoc doi tra khong",
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
        "policy", "policies", "refund", "return policy", "warranty", "exchange",
        "chinh sach", "hoan tien", "bao hanh", "doi tra", "doi hang",
    )
    if any(term in text for term in submit_terms):
        return ReturnIntent.SUBMIT_RETURN, 0.9
    if any(term in text for term in eligibility_terms):
        return ReturnIntent.RETURN_ELIGIBILITY, 0.85
    if any(term in text for term in order_terms):
        return ReturnIntent.ORDER_STATUS, 0.85
    if any(term in text for term in policy_terms):
        return ReturnIntent.POLICY_QUESTION, 0.75
    return ReturnIntent.UNCLEAR, 0.4


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


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    return without_marks.replace("đ", "d")
