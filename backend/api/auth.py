"""Thin API-level exports for the authentication service dependency."""

from backend.schemas.auth import AuthenticatedPrincipal
from service.auth.dependencies import get_current_user

__all__ = ["AuthenticatedPrincipal", "get_current_user"]
