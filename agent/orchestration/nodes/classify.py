"""Intent-classification node."""

from agent.core.state.shared import AgentSharedState, ClassificationState, StateTransition
from agent.orchestration.routing import classify_intent


def classify_request(state: dict) -> dict[str, AgentSharedState]:
    shared = state["shared"]
    intent, confidence = classify_intent(shared.input.message)
    shared.classification = ClassificationState(
        status="completed",
        intent=intent,
        confidence=confidence,
        classifier_version="manager-lexical-v1",
    )
    shared.transitions.append(
        StateTransition(component="classify_intent", to_status="completed", detail=intent.value)
    )
    return {"shared": shared}
