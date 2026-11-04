"""Injectable HTTP dependencies for authenticating requests from bearer tokens."""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.db.session import get_db
from backend.repositories.user import UserRepository
from backend.core.security.jwt import (
    JWTTokenVerifier,
    SecurityConfigurationError,
    get_jwt_verifier,
)
from backend.schemas.auth import AuthenticatedPrincipal
from service.auth.exceptions import InvalidCredentialsError
from service.auth.service import AuthService


_bearer_scheme = HTTPBearer(auto_error=False)


def _get_configured_token_verifier() -> JWTTokenVerifier:
    try:
        return get_jwt_verifier()
    except SecurityConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured.",
        ) from exc


def get_auth_service(
    session: Session = Depends(get_db),
    token_verifier: JWTTokenVerifier = Depends(_get_configured_token_verifier),
) -> AuthService:
    """Build the auth service; both dependencies can be overridden by a host app."""
    return AuthService(UserRepository(session), token_verifier)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    auth_service: AuthService = Depends(get_auth_service),
) -> AuthenticatedPrincipal:
    """Resolve the active application user from a verified bearer access token."""
    if credentials is None or credentials.scheme.casefold() != "bearer":
        raise _unauthorized()
    try:
        return auth_service.authenticate_token(credentials.credentials)
    except InvalidCredentialsError:
        raise _unauthorized() from None


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or inactive account credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )
