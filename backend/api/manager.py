"""Authenticated HTTP endpoint for the manager-agent graph."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from agent.orchestration.graph import ManagerAgentGraph
from backend.schemas.auth import AuthenticatedPrincipal
from schemas.manager import ManagerRequest, ManagerResponse
from service.auth.dependencies import get_current_user


def create_manager_router(manager: ManagerAgentGraph) -> APIRouter:
    """Create the manager route with authentication enforced as a dependency."""
    router = APIRouter(prefix="/agent", tags=["agent"])

    @router.post("/requests", response_model=ManagerResponse)
    def run_manager(
        request: ManagerRequest,
        principal: AuthenticatedPrincipal = Depends(get_current_user),
    ) -> ManagerResponse:
        result = manager.invoke(
            request.message,
            user_id=principal.user_id,
            language=request.language,
            conversation_id=request.conversation_id,
        )
        return ManagerResponse.from_state(result)

    return router
