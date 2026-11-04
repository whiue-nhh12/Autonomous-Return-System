"""Application service for proposing return eligibility decisions."""

from __future__ import annotations

from typing import Protocol

from agent.agents.return_decision.rules import (
    NoStructuredPolicyRules,
    ReturnPolicyRuleProvider,
    ReturnPolicyRuleSet,
    evaluate_return_request,
)
from agent.agents.return_decision.state.result import ReturnDecisionRequest, ReturnDecisionResult


class ReturnDecisionEvaluator(Protocol):
    def evaluate(self, request: ReturnDecisionRequest) -> ReturnDecisionResult: ...


class ReturnDecisionAgent:
    def __init__(self, rule_provider: ReturnPolicyRuleProvider | None = None) -> None:
        self.rule_provider = rule_provider or NoStructuredPolicyRules()

    def evaluate(self, request: ReturnDecisionRequest) -> ReturnDecisionResult:
        rule_set: ReturnPolicyRuleSet | None = None
        if request.policy_evidence:
            rule_set = self.rule_provider.get_rule_set(
                request.policy_evidence, evaluated_at=request.evaluated_at
            )
        return evaluate_return_request(request, rule_set)
