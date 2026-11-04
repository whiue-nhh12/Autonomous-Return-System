"""Application authentication logic, independent of HTTP request handling."""

from uuid import UUID

from backend.core.security.jwt import JWTTokenVerifier, TokenVerificationError
from backend.models.enums import UserStatus
from backend.repositories.user import UserRepository
from backend.schemas.auth import AuthenticatedPrincipal
from service.auth.exceptions import InvalidCredentialsError


class AuthService:
    """Verify an access token and resolve it to an active local user."""

    def __init__(self, users: UserRepository, token_verifier: JWTTokenVerifier) -> None:
        self.users = users
        self.token_verifier = token_verifier

    def authenticate_token(self, token: str) -> AuthenticatedPrincipal:
        try:
            claims = self.token_verifier.verify(token)
            user_id = UUID(str(claims[self.token_verifier.settings.user_id_claim]))
        except (TokenVerificationError, KeyError, TypeError, ValueError) as exc:
            raise InvalidCredentialsError("Invalid or inactive account credentials") from exc

        user = self.users.get(user_id)
        if user is None or user.status is not UserStatus.ACTIVE:
            raise InvalidCredentialsError("Invalid or inactive account credentials")
        return AuthenticatedPrincipal(user_id=user.id, role=user.role.value)
