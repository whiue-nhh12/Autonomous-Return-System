"""Validate request state before intent routing."""

from agent.core.state.shared import AgentSharedState, RequestInput, StateTransition


def validate_request(state: dict) -> dict[str, AgentSharedState]:
    shared = state["shared"]
    shared.input = RequestInput.model_validate(shared.input.model_dump())
    shared.transitions.append(StateTransition(component="validate_request", to_status="valid"))
    return {"shared": shared}
