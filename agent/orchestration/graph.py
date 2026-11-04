"""Manager Agent graph that routes authenticated requests to worker agents."""

from __future__ import annotations

from uuid import UUID

from langgraph.graph import END, START, StateGraph

from agent.agents.order.factory import create_order_agent_graph
from agent.agents.order.graph import OrderAgentGraph
from agent.agents.policy.graph import PolicyAgentGraph
from agent.agents.return_decision.graph import ReturnDecisionAgentGraph
from agent.orchestration.nodes.classify import classify_request
from agent.orchestration.nodes.generate_response import clarify, finalize_response
from agent.orchestration.nodes.retrieve_order import (
    create_order_node,
    create_eligibility_order_node,
)
from agent.orchestration.nodes.retrieve_policy import create_policy_node
from agent.orchestration.nodes.submit_return import submit_return
from agent.orchestration.nodes.evaluate_return import create_return_decision_node
from agent.orchestration.nodes.validate_request import validate_request
from agent.orchestration.routing import route_intent, send_eligibility_workers
from agent.orchestration.state import ManagerGraphState
from agent.core.state.shared import AgentSharedState, RequestContext, RequestInput


class ManagerAgentGraph:
    """Start a fresh orchestration run for each authenticated user turn.

    When no order agent is injected, the graph composes the DB-backed,
    read-only order agent. It connects only when an order lookup is routed.
    """

    def __init__(
        self,
        policy_agent: PolicyAgentGraph | None = None,
        order_agent: OrderAgentGraph | None = None,
        return_decision_agent: ReturnDecisionAgentGraph | None = None,
    ) -> None:
        order_agent = order_agent or create_order_agent_graph()
        builder = StateGraph(ManagerGraphState)
        builder.add_node("validate_request", validate_request)
        builder.add_node("classify_intent", classify_request)
        builder.add_node(
            "policy_worker",
            create_policy_node(policy_agent, result_key="policy", state_key="policy_worker_state"),
        )
        builder.add_node("order_worker", create_order_node(order_agent))
        builder.add_node(
            "eligibility_policy_worker",
            create_policy_node(
                policy_agent,
                result_key="eligibility_policy",
                state_key="eligibility_policy_state",
            ),
        )
        builder.add_node("eligibility_order_worker", create_eligibility_order_node(order_agent))
        builder.add_node("evaluate_return", create_return_decision_node(return_decision_agent))
        builder.add_node("submit_return", submit_return)
        builder.add_node("clarify", clarify)
        builder.add_node("finalize_response", finalize_response)
        builder.add_node("eligibility_fanout", lambda state: {"shared": state["shared"]})

        builder.add_edge(START, "validate_request")
        builder.add_edge("validate_request", "classify_intent")
        builder.add_conditional_edges(
            "classify_intent",
            route_intent,
            {
                "policy": "policy_worker",
                "order": "order_worker",
                "eligibility": "eligibility_fanout",
                "submit": "submit_return",
                "clarify": "clarify",
            },
        )
        builder.add_conditional_edges("eligibility_fanout", send_eligibility_workers)
        builder.add_edge("policy_worker", "finalize_response")
        builder.add_edge("order_worker", "finalize_response")
        builder.add_edge(
            ["eligibility_policy_worker", "eligibility_order_worker"],
            "evaluate_return",
        )
        builder.add_edge("evaluate_return", "finalize_response")
        builder.add_edge("submit_return", "finalize_response")
        builder.add_edge("clarify", "finalize_response")
        builder.add_edge("finalize_response", END)
        self.graph = builder.compile()

    def invoke(
        self,
        message: str,
        *,
        user_id: UUID,
        language: str = "vi",
        conversation_id: str | None = None,
    ) -> AgentSharedState:
        """Invoke the graph with a trusted user identity and fresh request ID."""
        shared = AgentSharedState(
            context=RequestContext(user_id=user_id, conversation_id=conversation_id),
            input=RequestInput(message=message, language=language),
        )
        result = self.graph.invoke({"shared": shared, "child_results": {}})
        return result["shared"]
