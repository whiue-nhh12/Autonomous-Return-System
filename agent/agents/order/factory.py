"""Composition helpers for the production order agent."""

from agent.agents.order.graph import OrderAgentGraph
from service.order_query import OrderQueryService


def create_order_agent_graph() -> OrderAgentGraph:
    """Create the DB-backed graph; database configuration is resolved on lookup."""
    return OrderAgentGraph(OrderQueryService())
