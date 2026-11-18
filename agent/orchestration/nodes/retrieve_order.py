"""Adapters between the manager graph and order-agent results."""

from __future__ import annotations

import logging
from typing import Any

from agent.agents.order.graph import OrderAgentGraph
from agent.agents.order.state.result import OrderAgentRequest
from agent.core.state.shared import ChildAgentResult
from agent.orchestration.state import ManagerGraphState

_LOGGER = logging.getLogger(__name__)


def create_order_node(order_agent: OrderAgentGraph | None):
    def run_order(state: ManagerGraphState) -> dict[str, Any]:
        shared = state["shared"]
        if order_agent is None:
            child_result = ChildAgentResult(
                status="not_configured",
                message="Order lookup is not configured.",
                error_code="ORDER_AGENT_NOT_CONFIGURED",
            )
            return {"child_results": {"order": child_result}}

        try:
            message = "\n".join([*shared.context.prior_messages, shared.input.message])
            order_number = shared.classification.entities.get("order_number")
            result = order_agent.invoke(
                OrderAgentRequest(
                    user_id=shared.context.user_id,
                    request_id=shared.context.request_id,
                    message=message,
                    language=shared.input.language,
                    order_number=order_number if isinstance(order_number, str) else None,
                )
            )
        except Exception:
            _LOGGER.exception(
                "Order child agent failed",
                extra={"request_id": str(shared.context.request_id)},
            )
            child_result = ChildAgentResult(
                status="failed",
                message="Order history is temporarily unavailable. Please try again later.",
                error_code="ORDER_AGENT_FAILED",
            )
            return {"child_results": {"order": child_result}}

        child_result = ChildAgentResult(
            status=result.status,
            message=result.message,
            error_code=result.error_code,
            data={
                "orders": [order.model_dump(mode="json") for order in result.orders],
                "has_more": result.has_more,
            },
        )
        return {
            "order_worker_state": result,
            "child_results": {"order": child_result},
        }

    return run_order


def create_eligibility_order_node(order_agent: OrderAgentGraph | None):
    """Read the authenticated user's order facts for eligibility evaluation."""
    from agent.agents.order.state.result import OrderAgentRequest

    def run(state: ManagerGraphState) -> dict:
        shared = state["shared"]
        if order_agent is None:
            return {"child_results": {"eligibility_order": ChildAgentResult(
                status="not_configured",
                message="Order lookup is not configured.",
                error_code="ELIGIBILITY_ORDER_AGENT_NOT_CONFIGURED",
            )}}
        try:
            message = "\n".join([*shared.context.prior_messages, shared.input.message])
            order_number = shared.classification.entities.get("order_number")
            result = order_agent.invoke(
                OrderAgentRequest(
                    user_id=shared.context.user_id,
                    request_id=shared.context.request_id,
                    message=message,
                    language=shared.input.language,
                    order_number=order_number if isinstance(order_number, str) else None,
                )
            )
        except Exception:
            _LOGGER.exception("Eligibility order lookup failed", extra={"request_id": str(shared.context.request_id)})
            return {"child_results": {"eligibility_order": ChildAgentResult(
                status="failed",
                message="Order history is temporarily unavailable. Please try again later.",
                error_code="ELIGIBILITY_ORDER_LOOKUP_FAILED",
            )}}

        return {
            "eligibility_order_worker_state": result,
            "child_results": {"eligibility_order": ChildAgentResult(
                status=result.status,
                message=result.message,
                error_code=result.error_code,
                data={"orders": [order.model_dump(mode="json") for order in result.orders]},
            )},
        }

    return run
