"""JWT/JWKS token verification adapter backed by PyJWT."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import jwt
from pydantic import ValidationError
from jwt import PyJWKClient, PyJWTError

from backend.schemas.auth import JWTClaims, JWTSettings


class SecurityConfigurationError(RuntimeError):
    """Raised when the token verification provider is not configured."""


class TokenVerificationError(ValueError):
    """Raised when an access token cannot be verified."""


class JWTTokenVerifier:
    """Verify signed JWT access tokens against configured issuer keys."""

    def __init__(self, settings: JWTSettings) -> None:
        self.settings = settings
        self._jwks_client = PyJWKClient(settings.jwks_url, cache_keys=True, timeout=5)

    def verify(self, token: str) -> dict[str, Any]:
        try:
            signing_key = self._jwks_client.get_signing_key_from_jwt(token)
            decoded_claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                issuer=self.settings.issuer,
                audience=self.settings.audience,
                options={
                    "require": [
                        "exp",
                        "iat",
                        "iss",
                        "aud",
                        "sub",
                        self.settings.user_id_claim,
                    ]
                },
            )
            claims = JWTClaims.model_validate(decoded_claims)
        except (PyJWTError, ValidationError) as exc:
            raise TokenVerificationError("Token verification failed") from exc
        return claims.model_dump()


@lru_cache(maxsize=1)
def get_jwt_verifier() -> JWTTokenVerifier:
    """Return the process-wide verifier; override this dependency in tests/apps."""
    try:
        settings = JWTSettings.from_environment()
    except ValueError as exc:
        raise SecurityConfigurationError(str(exc)) from exc
    return JWTTokenVerifier(settings)
