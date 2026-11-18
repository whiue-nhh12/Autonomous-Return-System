"""Admin endpoints for return proposal decisions and later refund initiation."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, status

from agent.orchestration.graph import ManagerAgentGraph
from backend.db.session import get_session_factory
from backend.schemas.auth import AuthenticatedPrincipal
from backend.schemas.refund import RefundInitiationRequest, RefundInitiationResponse
from backend.schemas.return_reevaluation import ReturnReevaluationResponse
from backend.schemas.return_receipt import ReturnReceiptRequest, ReturnReceiptResponse
from backend.schemas.return_proposal import (
    ReturnProposalDecisionRequest,
    ReturnProposalDecisionResponse,
    ReturnProposalSummary,
)
from service.auth.dependencies import get_current_user
from service.refund_initiation import (
    InvalidRefundAmount,
    InvalidRefundPayment,
    RefundInitiationForbidden,
    RefundInitiationIdempotencyConflict,
    RefundInitiationService,
    RefundNotReady,
    RefundReturnNotFound,
)
from service.return_approval import (
    InvalidProposalState,
    ReturnApprovalForbidden,
    ReturnApprovalService,
    ReturnDecisionIdempotencyConflict,
    ReturnProposalNotFound,
    StaleReturnProposal,
)
from service.return_receipt import (
    ReturnReceiptConflict,
    ReturnReceiptForbidden,
    ReturnReceiptNotFound,
    ReturnReceiptService,
)
from service.return_reevaluation import (
    ReturnReevaluationConflict,
    ReturnReevaluationForbidden,
    ReturnReevaluationNotFound,
    ReturnReevaluationService,
)


def get_return_approval_service() -> ReturnApprovalService:
    return ReturnApprovalService(get_session_factory())


def get_refund_initiation_service() -> RefundInitiationService:
    return RefundInitiationService(get_session_factory())


def get_return_receipt_service() -> ReturnReceiptService:
    return ReturnReceiptService(get_session_factory())


def get_return_reevaluation_service(
    manager: ManagerAgentGraph,
) -> ReturnReevaluationService:
    return ReturnReevaluationService(
        get_session_factory(),
        order_agent=manager.order_agent,
        policy_agent=manager.policy_agent,
        decision_agent=manager.return_decision_agent,
    )


def create_return_admin_router(manager: ManagerAgentGraph) -> APIRouter:
    router = APIRouter(tags=["admin-returns"])

    @router.get(
        "/admin/returns/{return_request_id}/proposals",
        response_model=list[ReturnProposalSummary],
    )
    def list_return_proposals(
        return_request_id: UUID,
        principal: AuthenticatedPrincipal = Depends(get_current_user),
        service: ReturnApprovalService = Depends(get_return_approval_service),
    ) -> list[ReturnProposalSummary]:
        try:
            return service.list_for_request(
                return_request_id=return_request_id,
                actor_id=principal.user_id,
            )
        except ReturnApprovalForbidden as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ReturnProposalNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post(
        "/admin/returns/{return_request_id}/reevaluate",
        response_model=ReturnReevaluationResponse,
    )
    def reevaluate_return_request(
        return_request_id: UUID,
        idempotency_key: Annotated[
            str,
            Header(alias="Idempotency-Key", min_length=16, max_length=128),
        ],
        principal: AuthenticatedPrincipal = Depends(get_current_user),
        service: ReturnReevaluationService = Depends(
            lambda: get_return_reevaluation_service(manager)
        ),
    ) -> ReturnReevaluationResponse:
        try:
            return service.reevaluate(
                return_request_id=return_request_id,
                actor_id=principal.user_id,
                idempotency_key=idempotency_key,
            )
        except ReturnReevaluationForbidden as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ReturnReevaluationNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ReturnReevaluationConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post(
        "/admin/return-proposals/{proposal_id}/decision",
        response_model=ReturnProposalDecisionResponse,
    )
    def decide_return_proposal(
        proposal_id: UUID,
        request: ReturnProposalDecisionRequest,
        idempotency_key: Annotated[
            str,
            Header(alias="Idempotency-Key", min_length=16, max_length=128),
        ],
        principal: AuthenticatedPrincipal = Depends(get_current_user),
        service: ReturnApprovalService = Depends(get_return_approval_service),
    ) -> ReturnProposalDecisionResponse:
        try:
            return service.decide(
                proposal_id=proposal_id,
                actor_id=principal.user_id,
                action=request.action,
                idempotency_key=idempotency_key,
                note=request.note,
            )
        except ReturnApprovalForbidden as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ReturnProposalNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (
            InvalidProposalState,
            StaleReturnProposal,
            ReturnDecisionIdempotencyConflict,
        ) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post(
        "/admin/returns/{return_request_id}/receive",
        response_model=ReturnReceiptResponse,
    )
    def mark_return_received(
        return_request_id: UUID,
        request: ReturnReceiptRequest,
        idempotency_key: Annotated[
            str,
            Header(alias="Idempotency-Key", min_length=16, max_length=128),
        ],
        principal: AuthenticatedPrincipal = Depends(get_current_user),
        service: ReturnReceiptService = Depends(get_return_receipt_service),
    ) -> ReturnReceiptResponse:
        try:
            return service.mark_received(
                return_request_id=return_request_id,
                actor_id=principal.user_id,
                idempotency_key=idempotency_key,
                note=request.note,
            )
        except ReturnReceiptForbidden as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ReturnReceiptNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ReturnReceiptConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post(
        "/admin/returns/{return_request_id}/initiate-refund",
        response_model=RefundInitiationResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def initiate_refund(
        return_request_id: UUID,
        request: RefundInitiationRequest,
        idempotency_key: Annotated[
            str,
            Header(alias="Idempotency-Key", min_length=16, max_length=128),
        ],
        principal: AuthenticatedPrincipal = Depends(get_current_user),
        service: RefundInitiationService = Depends(get_refund_initiation_service),
    ) -> RefundInitiationResponse:
        try:
            return service.initiate(
                return_request_id=return_request_id,
                actor_id=principal.user_id,
                idempotency_key=idempotency_key,
                amount=request.amount,
                payment_id=request.payment_id,
                note=request.note,
            )
        except RefundInitiationForbidden as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except RefundReturnNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (
            RefundNotReady,
            RefundInitiationIdempotencyConflict,
        ) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (InvalidRefundAmount, InvalidRefundPayment) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return router
