"""
Unit tests for AuthenticationComponent.

Tests cover:
- Valid token validation
- Expired token rejection
- Missing token handling
- Multiple tokens support
- Edge cases (empty string, None)
- Protocol conformance

"""

from __future__ import annotations

import datetime as dt
from datetime import datetime
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from ml.dashboard.common.authentication import AuthenticationComponent
from ml.dashboard.common.authentication import AuthenticationProtocol
from ml.dashboard.config import DashboardToken


# -----------------
# Fixtures
# -----------------


@pytest.fixture
def fixed_now() -> datetime:
    """Return a fixed datetime for consistent testing."""
    return datetime(2025, 6, 15, 12, 0, 0, tzinfo=dt.UTC)


@pytest.fixture
def valid_token() -> DashboardToken:
    """Return a token that never expires."""
    return DashboardToken(value="secret123", expires_at=None)


@pytest.fixture
def expired_token(fixed_now: datetime) -> DashboardToken:
    """Return a token that expired yesterday."""
    expires = fixed_now - dt.timedelta(days=1)
    return DashboardToken(value="expired_token", expires_at=expires)


@pytest.fixture
def future_token(fixed_now: datetime) -> DashboardToken:
    """Return a token that expires in the future."""
    expires = fixed_now + dt.timedelta(days=30)
    return DashboardToken(value="future_secret", expires_at=expires)


@pytest.fixture
def auth_no_tokens() -> AuthenticationComponent:
    """Return an auth component with no tokens configured."""
    return AuthenticationComponent(tokens=())


@pytest.fixture
def auth_single_valid(valid_token: DashboardToken) -> AuthenticationComponent:
    """Return an auth component with a single valid token."""
    return AuthenticationComponent(tokens=(valid_token,))


@pytest.fixture
def auth_multiple(
    valid_token: DashboardToken,
    future_token: DashboardToken,
) -> AuthenticationComponent:
    """Return an auth component with multiple valid tokens."""
    return AuthenticationComponent(tokens=(valid_token, future_token))


@pytest.fixture
def auth_only_expired(expired_token: DashboardToken) -> AuthenticationComponent:
    """Return an auth component with only expired tokens."""
    return AuthenticationComponent(tokens=(expired_token,))


@pytest.fixture
def auth_mixed(
    valid_token: DashboardToken,
    expired_token: DashboardToken,
    future_token: DashboardToken,
) -> AuthenticationComponent:
    """Return an auth component with mixed valid/expired tokens."""
    return AuthenticationComponent(tokens=(expired_token, valid_token, future_token))


# -----------------
# Protocol Tests
# -----------------


def test_authentication_component_implements_protocol() -> None:
    """AuthenticationComponent implements AuthenticationProtocol."""
    # Protocol structural typing check - verify method exists with correct signature
    component = AuthenticationComponent(tokens=())
    assert hasattr(component, "validate_token")
    assert callable(component.validate_token)


def test_protocol_signature_matches() -> None:
    """Protocol method signature matches component implementation."""
    import inspect

    protocol_sig = inspect.signature(AuthenticationProtocol.validate_token)
    component_sig = inspect.signature(AuthenticationComponent.validate_token)

    # Both should have 'self', 'provided', and 'now' parameters
    assert "provided" in protocol_sig.parameters
    assert "now" in protocol_sig.parameters
    assert "provided" in component_sig.parameters
    assert "now" in component_sig.parameters


# -----------------
# No Tokens Configured
# -----------------


def test_no_tokens_configured_allows_all(auth_no_tokens: AuthenticationComponent) -> None:
    """When no tokens configured, all requests are allowed."""
    assert auth_no_tokens.validate_token("any_token") is True
    assert auth_no_tokens.validate_token("") is True
    assert auth_no_tokens.validate_token(None) is True


@patch("ml.dashboard.common.authentication._AUTH_VALIDATIONS_TOTAL")
def test_no_tokens_emits_no_metrics(
    mock_counter: MagicMock,
    auth_no_tokens: AuthenticationComponent,
) -> None:
    """When no tokens configured, no validation metrics are emitted."""
    auth_no_tokens.validate_token("any_token")
    mock_counter.labels.assert_not_called()


# -----------------
# Valid Token Validation
# -----------------


def test_valid_token_accepts(
    auth_single_valid: AuthenticationComponent,
    valid_token: DashboardToken,
) -> None:
    """Valid token is accepted."""
    assert auth_single_valid.validate_token(valid_token.value) is True


def test_valid_token_accepts_with_explicit_now(
    auth_single_valid: AuthenticationComponent,
    valid_token: DashboardToken,
    fixed_now: datetime,
) -> None:
    """Valid token is accepted when explicit now is provided."""
    assert auth_single_valid.validate_token(valid_token.value, now=fixed_now) is True


@patch("ml.dashboard.common.authentication._AUTH_VALIDATIONS_TOTAL")
def test_valid_token_emits_success_metric(
    mock_counter: MagicMock,
    auth_single_valid: AuthenticationComponent,
    valid_token: DashboardToken,
) -> None:
    """Valid token validation emits success metric."""
    auth_single_valid.validate_token(valid_token.value)
    mock_counter.labels.assert_called_once_with(result="success")
    mock_counter.labels.return_value.inc.assert_called_once()


# -----------------
# Invalid Token Validation
# -----------------


def test_invalid_token_rejects(auth_single_valid: AuthenticationComponent) -> None:
    """Invalid token is rejected."""
    assert auth_single_valid.validate_token("wrong_token") is False


@patch("ml.dashboard.common.authentication._AUTH_VALIDATIONS_TOTAL")
def test_invalid_token_emits_invalid_metric(
    mock_counter: MagicMock,
    auth_single_valid: AuthenticationComponent,
) -> None:
    """Invalid token validation emits invalid metric."""
    auth_single_valid.validate_token("wrong_token")
    mock_counter.labels.assert_called_once_with(result="invalid")
    mock_counter.labels.return_value.inc.assert_called_once()


@patch("ml.dashboard.common.authentication.logger")
def test_invalid_token_logs_warning(
    mock_logger: MagicMock,
    auth_single_valid: AuthenticationComponent,
) -> None:
    """Invalid token validation logs warning with fingerprint."""
    auth_single_valid.validate_token("wrong_token")
    mock_logger.warning.assert_called_once()
    call_args = mock_logger.warning.call_args
    assert "dashboard token invalid" in call_args[0][0]
    assert "token_fingerprint" in call_args[1]["extra"]


# -----------------
# Missing Token Validation
# -----------------


def test_missing_token_none_rejects(auth_single_valid: AuthenticationComponent) -> None:
    """Missing token (None) is rejected."""
    assert auth_single_valid.validate_token(None) is False


def test_missing_token_empty_string_rejects(auth_single_valid: AuthenticationComponent) -> None:
    """Empty string token is rejected."""
    # Empty string is NOT None, so it will be checked and fail validation
    assert auth_single_valid.validate_token("") is False


@patch("ml.dashboard.common.authentication._AUTH_VALIDATIONS_TOTAL")
def test_missing_token_emits_missing_metric(
    mock_counter: MagicMock,
    auth_single_valid: AuthenticationComponent,
) -> None:
    """Missing token validation emits missing metric."""
    auth_single_valid.validate_token(None)
    mock_counter.labels.assert_called_once_with(result="missing")
    mock_counter.labels.return_value.inc.assert_called_once()


@patch("ml.dashboard.common.authentication.logger")
def test_missing_token_logs_warning(
    mock_logger: MagicMock,
    auth_single_valid: AuthenticationComponent,
) -> None:
    """Missing token validation logs warning."""
    auth_single_valid.validate_token(None)
    mock_logger.warning.assert_called_once()
    call_args = mock_logger.warning.call_args
    assert "dashboard token missing" in call_args[0][0]


# -----------------
# Expired Token Validation
# -----------------


def test_expired_token_rejects(
    auth_only_expired: AuthenticationComponent,
    expired_token: DashboardToken,
    fixed_now: datetime,
) -> None:
    """Expired token is rejected."""
    assert auth_only_expired.validate_token(expired_token.value, now=fixed_now) is False


@patch("ml.dashboard.common.authentication._AUTH_VALIDATIONS_TOTAL")
def test_expired_token_emits_expired_metric(
    mock_counter: MagicMock,
    auth_only_expired: AuthenticationComponent,
    expired_token: DashboardToken,
    fixed_now: datetime,
) -> None:
    """Expired token validation emits expired metric."""
    auth_only_expired.validate_token(expired_token.value, now=fixed_now)
    mock_counter.labels.assert_called_once_with(result="expired")
    mock_counter.labels.return_value.inc.assert_called_once()


@patch("ml.dashboard.common.authentication.logger")
def test_expired_token_logs_warning(
    mock_logger: MagicMock,
    auth_only_expired: AuthenticationComponent,
    expired_token: DashboardToken,
    fixed_now: datetime,
) -> None:
    """Expired token validation logs warning."""
    auth_only_expired.validate_token(expired_token.value, now=fixed_now)
    mock_logger.warning.assert_called_once()
    call_args = mock_logger.warning.call_args
    assert "all dashboard tokens expired" in call_args[0][0]


# -----------------
# Multiple Tokens Support
# -----------------


def test_multiple_tokens_first_valid_accepts(
    auth_multiple: AuthenticationComponent,
    valid_token: DashboardToken,
) -> None:
    """First valid token in multiple tokens is accepted."""
    assert auth_multiple.validate_token(valid_token.value) is True


def test_multiple_tokens_second_valid_accepts(
    auth_multiple: AuthenticationComponent,
    future_token: DashboardToken,
    fixed_now: datetime,
) -> None:
    """Second valid token in multiple tokens is accepted."""
    assert auth_multiple.validate_token(future_token.value, now=fixed_now) is True


def test_multiple_tokens_invalid_rejects(auth_multiple: AuthenticationComponent) -> None:
    """Invalid token is rejected even with multiple valid tokens."""
    assert auth_multiple.validate_token("wrong_token") is False


def test_mixed_tokens_valid_accepts(
    auth_mixed: AuthenticationComponent,
    valid_token: DashboardToken,
    fixed_now: datetime,
) -> None:
    """Valid token is accepted even when mixed with expired tokens."""
    assert auth_mixed.validate_token(valid_token.value, now=fixed_now) is True


def test_mixed_tokens_future_accepts(
    auth_mixed: AuthenticationComponent,
    future_token: DashboardToken,
    fixed_now: datetime,
) -> None:
    """Future-expiring token is accepted when mixed with expired tokens."""
    assert auth_mixed.validate_token(future_token.value, now=fixed_now) is True


def test_mixed_tokens_expired_rejects(
    auth_mixed: AuthenticationComponent,
    expired_token: DashboardToken,
    fixed_now: datetime,
) -> None:
    """Expired token is rejected even when mixed with valid tokens."""
    # The expired token value won't match because it's filtered out
    assert auth_mixed.validate_token(expired_token.value, now=fixed_now) is False


# -----------------
# Edge Cases
# -----------------


def test_whitespace_token_rejects(auth_single_valid: AuthenticationComponent) -> None:
    """Token with only whitespace is rejected."""
    assert auth_single_valid.validate_token("   ") is False


def test_token_with_unicode_handled(auth_single_valid: AuthenticationComponent) -> None:
    """Token with unicode characters can be validated (UTF-8 encoding)."""
    # Create a token with unicode and validate it
    unicode_token = DashboardToken(value="secret🔑123")
    auth_unicode = AuthenticationComponent(tokens=(unicode_token,))
    assert auth_unicode.validate_token("secret🔑123") is True
    assert auth_unicode.validate_token("wrong🔑token") is False


def test_token_case_sensitive(auth_single_valid: AuthenticationComponent) -> None:
    """Token validation is case-sensitive."""
    assert auth_single_valid.validate_token("SECRET123") is False
    assert auth_single_valid.validate_token("secret123") is True


def test_token_prefix_not_accepted(auth_single_valid: AuthenticationComponent) -> None:
    """Token prefix without full match is rejected."""
    assert auth_single_valid.validate_token("secret") is False


def test_token_suffix_not_accepted(auth_single_valid: AuthenticationComponent) -> None:
    """Token suffix without full match is rejected."""
    assert auth_single_valid.validate_token("123") is False


# -----------------
# Security Properties
# -----------------


def test_constant_time_comparison_used() -> None:
    """Validation uses hmac.compare_digest for constant-time comparison."""
    import hmac
    from unittest.mock import patch

    token = DashboardToken(value="secret123")
    auth = AuthenticationComponent(tokens=(token,))

    with patch("hmac.compare_digest", wraps=hmac.compare_digest) as mock_compare:
        auth.validate_token("secret123")
        # Should have been called at least once
        assert mock_compare.call_count >= 1


def test_token_fingerprint_generated() -> None:
    """Token fingerprint is generated using SHA-256 hash."""
    import hashlib
    from unittest.mock import patch

    token = DashboardToken(value="secret123")
    auth = AuthenticationComponent(tokens=(token,))

    with patch("hashlib.sha256", wraps=hashlib.sha256) as mock_hash:
        auth.validate_token("wrong_token")
        # Should have been called for fingerprinting
        assert mock_hash.call_count >= 1


# -----------------
# Default Parameters
# -----------------


def test_validate_token_uses_utc_now_by_default(
    auth_single_valid: AuthenticationComponent,
    valid_token: DashboardToken,
) -> None:
    """validate_token uses current UTC time when now is not provided."""
    with patch("ml.dashboard.common.authentication.datetime") as mock_dt:
        mock_now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=dt.UTC)
        mock_dt.now.return_value = mock_now

        auth_single_valid.validate_token(valid_token.value)
        mock_dt.now.assert_called_once()


# -----------------
# Initialization
# -----------------


def test_initialization_with_empty_tuple() -> None:
    """Component can be initialized with empty token tuple."""
    auth = AuthenticationComponent(tokens=())
    assert auth._tokens == ()


def test_initialization_with_single_token(valid_token: DashboardToken) -> None:
    """Component can be initialized with single token."""
    auth = AuthenticationComponent(tokens=(valid_token,))
    assert len(auth._tokens) == 1
    assert auth._tokens[0] == valid_token


def test_initialization_with_multiple_tokens(
    valid_token: DashboardToken,
    future_token: DashboardToken,
) -> None:
    """Component can be initialized with multiple tokens."""
    auth = AuthenticationComponent(tokens=(valid_token, future_token))
    assert len(auth._tokens) == 2
    assert valid_token in auth._tokens
    assert future_token in auth._tokens
