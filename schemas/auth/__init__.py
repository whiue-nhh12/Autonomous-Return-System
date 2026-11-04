"""Compatibility exports; canonical auth schemas live in ``backend.schemas.auth``."""

from backend.schemas.auth import AuthenticatedPrincipal, JWTClaims, JWTSettings

__all__ = ["AuthenticatedPrincipal", "JWTClaims", "JWTSettings"]
