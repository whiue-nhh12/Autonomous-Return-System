"""LangGraph workflow for read-only order history lookup."""

from __future__ import annotations

import logging
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from agent.agents.order.agent import (
    OrderHistorySearch,
    normalize_order_page,
    plan_order_query,
)
from agent.agents.order.state.result import OrderAgentRequest, OrderAgentResult, OrderQueryPlan
from backend.repositories.order_read import OrderReadPage

_LOGGER = logging.getLogger(__name__)


class OrderAgentGraphState(TypedDict, total=False):
    request: OrderAgentRequest
    query: OrderQueryPlan
    read_page: OrderReadPage
    error: str
    result: OrderAgentResult


class OrderAgentGraph:
    """Parse a text request, read only the user's orders, and normalize results."""

    def __init__(self, order_search: OrderHistorySearch) -> None:
        self.order_search = order_search
        builder = StateGraph(OrderAgentGraphState)
        builder.add_node("interpret_request", self._interpret_request)
        builder.add_node("retrieve_orders", self._retrieve_orders)
        builder.add_node("normalize_result", self._normalize_result)
        builder.add_node("lookup_failed", self._lookup_failed)
        builder.add_edge(START, "interpret_request")
        builder.add_edge("interpret_request", "retrieve_orders")
        builder.add_conditional_edges(
            "retrieve_orders",
            self._route_after_lookup,
            {"normalize_result": "normalize_result", "lookup_failed": "lookup_failed"},
        )
        builder.add_edge("normalize_result", END)
        builder.add_edge("lookup_failed", END)
        self.graph = builder.compile()

    def invoke(self, request: OrderAgentRequest) -> OrderAgentResult:
        result = self.graph.invoke({"request": request})
        return result["result"]

    @staticmethod
    def _interpret_request(state: OrderAgentGraphState) -> dict[str, OrderQueryPlan]:
        return {"query": plan_order_query(state["request"])}

    def _retrieve_orders(self, state: OrderAgentGraphState) -> dict:
        request = state["request"]
        query = state["query"]
        if query.order_number is not None or query.wants_latest:
            limit = 1
        elif query.wants_history:
            limit = 10
        else:
            # Inspect at most two records before deciding whether clarification is needed.
            limit = 2
        try:
            page = self.order_search.search_history(
                user_id=request.user_id,
                order_number=query.order_number,
                limit=limit,
            )
            _LOGGER.info(
                "Order history lookup completed",
                extra={
                    "request_id": str(request.request_id),
                    "user_id": str(request.user_id),
                    "result_count": len(page.orders),
                    "has_more": page.has_more,
                },
            )
            return {"read_page": page}
        except Exception as exc:
            _LOGGER.exception(
                "Order history lookup failed",
                extra={
                    "request_id": str(request.request_id),
                    "user_id": str(request.user_id),
                },
            )
            return {"error": exc.__class__.__name__}

    @staticmethod
    def _route_after_lookup(state: OrderAgentGraphState) -> Literal["normalize_result", "lookup_failed"]:
        return "lookup_failed" if state.get("error") else "normalize_result"

    @staticmethod
    def _normalize_result(state: OrderAgentGraphState) -> dict[str, OrderAgentResult]:
        request = state["request"]
        query = state["query"]
        page = state["read_page"]
        orders = normalize_order_page(page)
        english = request.language.casefold().startswith("en")

        if not orders:
            message = (
                "I couldn't find an order matching that request."
                if english
                else "Không tìm thấy đơn hàng phù hợp với yêu cầu."
            )
            return {"result": OrderAgentResult(status="not_found", message=message)}

        if not query.order_number and not query.wants_history and not query.wants_latest and len(orders) > 1:
            message = (
                "I found multiple recent orders. Please include the order number or ask for your latest order."
                if english
                else "Mình tìm thấy nhiều đơn hàng gần đây. Vui lòng cung cấp mã đơn hoặc yêu cầu đơn gần nhất."
            )
            return {
                "result": OrderAgentResult(
                    status="needs_clarification",
                    message=message,
                )
            }

        message = _format_orders(orders, language=request.language, has_more=page.has_more)
        return {
            "result": OrderAgentResult(
                status="succeeded",
                message=message,
                orders=orders,
                has_more=page.has_more,
            )
        }

    @staticmethod
    def _lookup_failed(state: OrderAgentGraphState) -> dict[str, OrderAgentResult]:
        english = state["request"].language.casefold().startswith("en")
        message = (
            "Order history is temporarily unavailable. Please try again later."
            if english
            else "Hiện chưa thể tra cứu lịch sử đơn hàng. Vui lòng thử lại sau."
        )
        return {
            "result": OrderAgentResult(
                status="failed",
                message=message,
                error_code="ORDER_LOOKUP_FAILED",
            )
        }


def _format_orders(orders, *, language: str, has_more: bool) -> str:
    english = language.casefold().startswith("en")
    lines = []
    for order in orders:
        total = f"{order.total_amount:f} {order.currency}"
        placed = order.placed_at.isoformat() if order.placed_at else "unknown date"
        if english:
            lines.append(
                f"Order {order.order_number}: {order.status}; total {total}; placed {placed}."
            )
        else:
            lines.append(
                f"Đơn {order.order_number}: trạng thái {order.status}; tổng {total}; đặt lúc {placed}."
            )
    if has_more:
        lines.append(
            "Showing the 10 most recent orders."
            if english
            else "Đang hiển thị 10 đơn hàng gần nhất."
        )
    return "\n".join(lines)
