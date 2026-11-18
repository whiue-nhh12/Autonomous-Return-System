"""Hybrid intent and entity understanding node."""

from agent.agents.return_decision.claims import extract_return_claim
from agent.agents.return_decision.state.result import ReturnClaimFacts
from agent.core.state.shared import (
    AgentSharedState,
    ClassificationState,
    ReturnIntent,
    StateTransition,
)
from agent.orchestration.understanding import AgentUnderstanding


def classify_request(
    state: dict,
    understanding: AgentUnderstanding | None = None,
) -> dict[str, AgentSharedState]:
    shared = state["shared"]
    service = understanding or AgentUnderstanding()
    intent, confidence, classifier_version = service.classify(
        shared.input.message,
        prior_messages=shared.context.prior_messages,
        pending_intent=shared.context.pending_intent,
    )
    entities = {}
    if intent in {
        ReturnIntent.ORDER_STATUS,
        ReturnIntent.RETURN_ELIGIBILITY,
        ReturnIntent.SUBMIT_RETURN,
    }:
        try:
            prior_claim = ReturnClaimFacts.model_validate(
                shared.context.conversation_claim
            )
        except (TypeError, ValueError):
            prior_claim = None
        claim = extract_return_claim(
            shared.input.message,
            understanding=service,
            seed=prior_claim,
            prior_messages=shared.context.prior_messages,
        )
        entities = {
            "order_number": claim.order_number,
            "return_claim": claim.model_dump(mode="json", exclude_none=True),
        }
    shared.classification = ClassificationState(
        status="completed",
        intent=intent,
        confidence=confidence,
        entities=entities,
        classifier_version=classifier_version,
    )
    shared.transitions.append(
        StateTransition(
            component="classify_intent",
            to_status="completed",
            detail=f"{intent.value}:{classifier_version}:{confidence:.2f}",
        )
    )
    return {"shared": shared}
