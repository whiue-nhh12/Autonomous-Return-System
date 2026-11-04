"""LangGraph wrapper for deterministic return eligibility evaluation."""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agent.agents.return_decision.agent import ReturnDecisionAgent
from agent.agents.return_decision.state.result import ReturnDecisionRequest, ReturnDecisionResult


class ReturnDecisionGraphState(TypedDict, total=False):
    request: ReturnDecisionRequest
    result: ReturnDecisionResult


class ReturnDecisionAgentGraph:
    def __init__(self, agent: ReturnDecisionAgent | None = None) -> None:
        self.agent = agent or ReturnDecisionAgent()
        builder = StateGraph(ReturnDecisionGraphState)
        builder.add_node("evaluate", self._evaluate)
        builder.add_edge(START, "evaluate")
        builder.add_edge("evaluate", END)
        self.graph = builder.compile()

    def invoke(self, request: ReturnDecisionRequest) -> ReturnDecisionResult:
        return self.graph.invoke({"request": request})["result"]

    def _evaluate(self, state: ReturnDecisionGraphState) -> dict[str, ReturnDecisionResult]:
        return {"result": self.agent.evaluate(state["request"])}
