"""Application-level authentication errors."""


class InvalidCredentialsError(ValueError):
    """Raised when a token does not map to an active application user."""
