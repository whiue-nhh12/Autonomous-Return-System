"""Configuration schema for validating JWT access tokens."""

from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict


class JWTClaims(BaseModel):
    """Standard required claims plus provider-specific claims such as user_id."""

    model_config = ConfigDict(extra="allow")

    exp: int | float
    iat: int | float
    iss: str
    aud: str | list[str]
    sub: str


class JWTSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    jwks_url: str
    issuer: str
    audience: str
    user_id_claim: str = "user_id"

    @classmethod
    def from_environment(cls) -> "JWTSettings":
        values = {
            "jwks_url": os.getenv("AUTH_JWKS_URL", "").strip(),
            "issuer": os.getenv("AUTH_JWT_ISSUER", "").strip(),
            "audience": os.getenv("AUTH_JWT_AUDIENCE", "").strip(),
            "user_id_claim": os.getenv("AUTH_USER_ID_CLAIM", "user_id").strip(),
        }
        missing = [
            env_name
            for field, env_name in (
                ("jwks_url", "AUTH_JWKS_URL"),
                ("issuer", "AUTH_JWT_ISSUER"),
                ("audience", "AUTH_JWT_AUDIENCE"),
            )
            if not values[field]
        ]
        if missing:
            raise ValueError(f"Missing authentication configuration: {', '.join(missing)}")
        if not values["user_id_claim"]:
            raise ValueError("AUTH_USER_ID_CLAIM cannot be empty")
        return cls(**values)
