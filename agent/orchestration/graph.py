"""Manager Agent graph that routes authenticated requests to worker agents."""

from __future__ import annotations

from contextlib import nullcontext
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
from agent.orchestration.nodes.submit_return import create_submit_return_node
from agent.orchestration.nodes.evaluate_return import create_return_decision_node
from agent.orchestration.nodes.validate_request import validate_request
from agent.orchestration.conversation import (
    ConversationSnapshot,
    ConversationStore,
    InMemoryConversationStore,
)
from agent.orchestration.routing import (
    route_after_return_lookups,
    route_intent,
    send_eligibility_workers,
)
from agent.orchestration.state import ManagerGraphState
from agent.core.state.shared import AgentSharedState, RequestContext, RequestInput
from agent.orchestration.understanding import AgentUnderstanding


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
        *,
        understanding: AgentUnderstanding | None = None,
        conversation_store: ConversationStore | None = None,
    ) -> None:
        self.understanding = (
            understanding
            if understanding is not None
            else AgentUnderstanding.from_environment()
        )
        self.conversation_store = (
            conversation_store
            if conversation_store is not None
            else InMemoryConversationStore()
        )
        self.order_agent = order_agent or create_order_agent_graph()
        self.policy_agent = policy_agent
        self.return_decision_agent = return_decision_agent or ReturnDecisionAgentGraph()
        builder = StateGraph(ManagerGraphState)
        builder.add_node("validate_request", validate_request)
        builder.add_node(
            "classify_intent",
            lambda state: classify_request(state, self.understanding),
        )
        builder.add_node(
            "policy_worker",
            create_policy_node(self.policy_agent, result_key="policy", state_key="policy_worker_state"),
        )
        builder.add_node("order_worker", create_order_node(self.order_agent))
        builder.add_node(
            "eligibility_policy_worker",
            create_policy_node(
                self.policy_agent,
                result_key="eligibility_policy",
                state_key="eligibility_policy_state",
            ),
        )
        builder.add_node("eligibility_order_worker", create_eligibility_order_node(self.order_agent))
        builder.add_node(
            "evaluate_return",
            create_return_decision_node(self.return_decision_agent),
        )
        builder.add_node("submit_return", create_submit_return_node())
        builder.add_node("return_flow_route", lambda state: {"shared": state["shared"]})
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
                "submit": "eligibility_fanout",
                "clarify": "clarify",
            },
        )
        builder.add_conditional_edges("eligibility_fanout", send_eligibility_workers)
        builder.add_edge("policy_worker", "finalize_response")
        builder.add_edge("order_worker", "finalize_response")
        builder.add_edge(
            ["eligibility_policy_worker", "eligibility_order_worker"],
            "return_flow_route",
        )
        builder.add_conditional_edges(
            "return_flow_route",
            route_after_return_lookups,
            {"eligibility": "evaluate_return", "submit": "submit_return"},
        )
        builder.add_conditional_edges(
            "submit_return",
            lambda state: "evaluate" if state.get("return_request_id") else "finish",
            {"evaluate": "evaluate_return", "finish": "finalize_response"},
        )
        builder.add_edge("evaluate_return", "finalize_response")
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
        """Invoke one turn and retain context only while clarification is pending."""
        normalized_conversation_id = (
            conversation_id.strip() if conversation_id and conversation_id.strip() else None
        )
        lock = (
            self.conversation_store.turn_lock(user_id, normalized_conversation_id)
            if normalized_conversation_id
            else nullcontext()
        )
        with lock:
            snapshot = (
                self.conversation_store.get(user_id, normalized_conversation_id)
                if normalized_conversation_id
                else None
            )
            prior_messages = snapshot.messages if snapshot else []
            shared = AgentSharedState(
                context=RequestContext(
                    user_id=user_id,
                    conversation_id=normalized_conversation_id,
                    prior_messages=prior_messages,
                    pending_intent=snapshot.pending_intent if snapshot else None,
                    conversation_claim=snapshot.claim if snapshot else {},
                ),
                input=RequestInput(message=message, language=language),
            )
            result = self.graph.invoke({"shared": shared, "child_results": {}})["shared"]
            if normalized_conversation_id:
                if result.clarification.awaiting_user:
                    messages = [*prior_messages, result.input.message]
                    pending_intent = result.classification.intent
                    if pending_intent is None or pending_intent.value == "unclear":
                        pending_intent = snapshot.pending_intent if snapshot else None
                    claim = result.classification.entities.get("return_claim")
                    if not isinstance(claim, dict):
                        claim = snapshot.claim if snapshot else {}
                    self.conversation_store.save(
                        user_id,
                        normalized_conversation_id,
                        ConversationSnapshot(
                            messages=[item[-2000:] for item in messages[-8:]],
                            pending_intent=pending_intent,
                            claim=claim,
                        ),
                    )
                else:
                    self.conversation_store.delete(user_id, normalized_conversation_id)
            return result
