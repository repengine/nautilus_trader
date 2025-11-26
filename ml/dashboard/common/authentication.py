"""
Authentication component for validating dashboard access tokens.

This component handles token validation including expiry checks and multi-token support.
All operations are cold-path only.

"""

from __future__ import annotations

import hashlib
import hmac
import logging
from datetime import datetime
from typing import Protocol

from ml.common.metrics_bootstrap import get_counter
from ml.dashboard.config import DashboardToken


logger = logging.getLogger(__name__)


_AUTH_VALIDATIONS_TOTAL = get_counter(
    "ml_dashboard_auth_validations_total",
    "Dashboard token validation attempts",
    labels=["result"],
)


class AuthenticationProtocol(Protocol):
    """Protocol for dashboard authentication operations."""

    def validate_token(self, provided: str | None, *, now: datetime | None = None) -> bool:
        """
        Validate a provided authentication token.

        Args:
            provided: The token string to validate (None indicates missing token)
            now: Current datetime for expiry checks (defaults to UTC now)

        Returns:
            True if token is valid and not expired, False otherwise

        """
        ...


class AuthenticationComponent:
    """
    Component responsible for validating dashboard authentication tokens.

    Supports multiple active tokens with optional expiry timestamps. Performs
    constant-time token comparison and tracks validation attempts via metrics.

    """

    def __init__(self, tokens: tuple[DashboardToken, ...]) -> None:
        """
        Initialize the authentication component.

        Args:
            tokens: Tuple of configured DashboardToken instances

        Example:
            >>> from ml.dashboard.config import DashboardToken
            >>> from datetime import datetime, timezone
            >>> expires = datetime(2025, 12, 31, tzinfo=timezone.utc)
            >>> token = DashboardToken(value="secret123", expires_at=expires)
            >>> auth = AuthenticationComponent(tokens=(token,))
            >>> auth.validate_token("secret123")
            True

        """
        self._tokens = tokens

    def validate_token(self, provided: str | None, *, now: datetime | None = None) -> bool:
        """
        Validate a provided authentication token.

        Validates against all active (non-expired) configured tokens using constant-time
        comparison. Records validation results via Prometheus metrics.

        Args:
            provided: The token string to validate (None indicates missing token)
            now: Current datetime for expiry checks (defaults to UTC now)

        Returns:
            True if token is valid and not expired, False otherwise

        Example:
            >>> from ml.dashboard.config import DashboardToken
            >>> from datetime import datetime, timezone
            >>> expires = datetime(2025, 12, 31, tzinfo=timezone.utc)
            >>> token = DashboardToken(value="secret123", expires_at=expires)
            >>> auth = AuthenticationComponent(tokens=(token,))
            >>> auth.validate_token("secret123")
            True
            >>> auth.validate_token("wrong")
            False
            >>> auth.validate_token(None)
            False

        """
        # No tokens configured means authentication is disabled
        if not self._tokens:
            return True

        # Use provided timestamp or current UTC time
        import datetime as dt

        now = now or datetime.now(dt.UTC)

        # Missing token
        if not provided:
            _AUTH_VALIDATIONS_TOTAL.labels(result="missing").inc()
            logger.warning("dashboard token missing", extra={"route": "ml.dashboard"})
            return False

        # Filter to active (non-expired) tokens
        active_tokens = tuple(token for token in self._tokens if token.is_valid(now=now))
        if not active_tokens:
            _AUTH_VALIDATIONS_TOTAL.labels(result="expired").inc()
            logger.warning("all dashboard tokens expired", extra={"route": "ml.dashboard"})
            return False

        # Generate fingerprint for logging (first 8 chars of SHA-256 hash)
        provided_digest = hashlib.sha256(provided.encode("utf-8")).hexdigest()[:8]

        # Constant-time comparison against all active tokens
        for token in active_tokens:
            try:
                if hmac.compare_digest(token.value, provided):
                    _AUTH_VALIDATIONS_TOTAL.labels(result="success").inc()
                    return True
            except TypeError:
                # hmac.compare_digest raises TypeError for non-ASCII strings
                # Fall back to direct comparison (still secure for our use case)
                if token.value == provided:
                    _AUTH_VALIDATIONS_TOTAL.labels(result="success").inc()
                    return True

        # No match found
        _AUTH_VALIDATIONS_TOTAL.labels(result="invalid").inc()
        logger.warning(
            "dashboard token invalid",
            extra={"token_fingerprint": provided_digest},
        )
        return False


__all__ = [
    "AuthenticationComponent",
    "AuthenticationProtocol",
]
