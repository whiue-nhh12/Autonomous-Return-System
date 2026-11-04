"""Policy retrieval agent and its LangGraph workflow."""

from agent.agents.policy.agent import PolicyAgent, PolicySearchPipeline

__all__ = ["PolicyAgent", "PolicyAgentGraph", "PolicyAgentGraphState", "PolicySearchPipeline"]


def __getattr__(name: str):
    if name in {"PolicyAgentGraph", "PolicyAgentGraphState"}:
        from agent.agents.policy.graph import PolicyAgentGraph, PolicyAgentGraphState

        return {
            "PolicyAgentGraph": PolicyAgentGraph,
            "PolicyAgentGraphState": PolicyAgentGraphState,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
