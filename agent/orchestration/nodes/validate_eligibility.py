"""Compatibility entry point for the return decision adapter."""

from agent.orchestration.nodes.evaluate_return import create_return_decision_node

validate_eligibility = create_return_decision_node()
