"""Persistence for immutable Decision Agent proposals."""

from __future__ import annotations

import hashlib
import json
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from agent.agents.return_decision.state.result import (
    ReturnClaimFacts,
    ReturnDecisionResult,
)
from agent.core.state.shared import PolicyEvidence
from backend.models import Order, ReturnDecisionProposal, ReturnRequest
from backend.models.enums import ReturnStatus
from service.return_fingerprint import fingerprint_order_rows


class ReturnDecisionProposalError(Exception):
    pass


class ReturnDecisionProposalNotFound(ReturnDecisionProposalError):
    pass


class ReturnDecisionProposalStaleInput(ReturnDecisionProposalError):
    pass


class ReturnDecisionProposalService:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def get_evaluation(
        self, evaluation_request_id: UUID
    ) -> tuple[UUID, UUID, str, str | None] | None:
        with self._session_factory() as session:
            proposal = session.scalar(
                select(ReturnDecisionProposal).where(
                    ReturnDecisionProposal.evaluation_request_id == evaluation_request_id
                )
            )
            if proposal is None:
                return None
            return (
                proposal.id,
                proposal.return_request_id,
                proposal.status,
                proposal.outcome,
            )

    def save(
        self,
        *,
        return_request_id: UUID,
        evaluation_request_id: UUID,
        result: ReturnDecisionResult,
        claim: ReturnClaimFacts,
        policy_evidence: list[PolicyEvidence],
        evaluated_data_fingerprint: str,
        classifier_version: str | None,
        evaluated_order_facts: list[dict] | None = None,
    ) -> UUID:
        if len(evaluated_data_fingerprint) != 64 or any(
            character not in "0123456789abcdef"
            for character in evaluated_data_fingerprint.casefold()
        ):
            raise ReturnDecisionProposalError("A valid order-data fingerprint is required.")
        versions = {
            item.doc_id: item.version
            for item in sorted(policy_evidence, key=lambda row: (row.doc_id, row.version))
        }
        claim_facts = claim.model_dump(mode="json", exclude_none=True)
        claim_facts["model_derived_fields"] = sorted(
            claim_facts.get("model_derived_fields", [])
        )
        checks = [item.model_dump(mode="json") for item in result.checks]
        order_facts = evaluated_order_facts or []
        version_ref = _policy_version_ref(versions)
        with self._session_factory.begin() as session:
            existing = session.scalar(
                select(ReturnDecisionProposal).where(
                    ReturnDecisionProposal.evaluation_request_id == evaluation_request_id
                )
            )
            if existing is not None:
                self._assert_same_evaluation(
                    existing,
                    return_request_id=return_request_id,
                    result=result,
                    claim_facts=claim_facts,
                    policy_versions=versions,
                    evaluated_data_fingerprint=evaluated_data_fingerprint,
                    classifier_version=classifier_version,
                    checks=checks,
                    evaluated_order_facts=order_facts,
                )
                return existing.id

            request = session.scalar(
                select(ReturnRequest)
                .where(ReturnRequest.id == return_request_id)
                .with_for_update()
            )
            if request is None:
                raise ReturnDecisionProposalNotFound("Return request was not found.")
            if request.status not in {ReturnStatus.SUBMITTED, ReturnStatus.REVIEWING}:
                raise ReturnDecisionProposalError(
                    "Only submitted or reviewing return requests can be evaluated."
                )
            order = session.scalar(
                select(Order).where(Order.id == request.order_id).with_for_update()
            )
            if order is None or order.user_id != request.user_id:
                raise ReturnDecisionProposalError(
                    "Return request order ownership is inconsistent."
                )
            current_fingerprint = fingerprint_order_rows(
                session, order, exclude_return_request_id=request.id
            )
            if current_fingerprint != evaluated_data_fingerprint:
                raise ReturnDecisionProposalStaleInput(
                    "Order or return data changed during evaluation. Reevaluate before review."
                )
            # Recheck after the request lock to make concurrent graph retries idempotent.
            existing = session.scalar(
                select(ReturnDecisionProposal)
                .where(ReturnDecisionProposal.evaluation_request_id == evaluation_request_id)
                .with_for_update()
            )
            if existing is not None:
                self._assert_same_evaluation(
                    existing,
                    return_request_id=return_request_id,
                    result=result,
                    claim_facts=claim_facts,
                    policy_versions=versions,
                    evaluated_data_fingerprint=evaluated_data_fingerprint,
                    classifier_version=classifier_version,
                    checks=checks,
                    evaluated_order_facts=order_facts,
                )
                return existing.id
            proposal = ReturnDecisionProposal(
                id=uuid4(),
                return_request_id=return_request_id,
                evaluation_request_id=evaluation_request_id,
                status=result.status,
                outcome=result.outcome,
                assessment_message=result.message,
                checks=checks,
                reason_codes=list(result.reason_codes),
                missing_fields=list(result.missing_fields),
                evidence_refs=list(result.evidence_refs),
                policy_versions=versions,
                policy_version_ref=version_ref,
                claim_facts=claim_facts,
                evaluated_order_facts=order_facts,
                evaluated_data_fingerprint=evaluated_data_fingerprint,
                rule_set_version=result.rule_set_version,
                classifier_version=classifier_version,
                evaluated_at=result.evaluated_at,
            )
            session.add(proposal)
            session.flush()
            return proposal.id

    @staticmethod
    def _assert_same_evaluation(
        proposal: ReturnDecisionProposal,
        *,
        return_request_id: UUID,
        result: ReturnDecisionResult,
        claim_facts: dict,
        policy_versions: dict[str, int],
        evaluated_data_fingerprint: str,
        classifier_version: str | None,
        checks: list[dict],
        evaluated_order_facts: list[dict],
    ) -> None:
        if (
            proposal.return_request_id != return_request_id
            or proposal.status != result.status
            or proposal.outcome != result.outcome
            or proposal.assessment_message != result.message
            or proposal.checks != checks
            or proposal.reason_codes != result.reason_codes
            or proposal.missing_fields != result.missing_fields
            or proposal.evidence_refs != result.evidence_refs
            or proposal.evaluated_data_fingerprint != evaluated_data_fingerprint
            or proposal.policy_versions != policy_versions
            or proposal.rule_set_version != result.rule_set_version
            or proposal.classifier_version != classifier_version
            or proposal.claim_facts != claim_facts
            or proposal.evaluated_order_facts != evaluated_order_facts
        ):
            raise ReturnDecisionProposalError(
                "Evaluation request ID was already used with different decision inputs."
            )


def _policy_version_ref(versions: dict[str, int]) -> str | None:
    if not versions:
        return None
    encoded = json.dumps(versions, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()
