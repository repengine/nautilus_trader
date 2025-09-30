"""Databento credential resolution utilities."""

from __future__ import annotations

import os
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import structlog


LOGGER = structlog.get_logger(__name__)

_PRIMARY_ENV_KEYS: tuple[str, ...] = ("DATABENTO_API_KEY", "ML_DATABENTO_API_KEY")
_SECRET_KEYS: tuple[str, ...] = ("databento_api_key", "DATABENTO_API_KEY")


class CredentialSource(StrEnum):
    """Sources that may supply the Databento API key."""

    NONE = "none"
    EXPLICIT = "explicit"
    ENVIRONMENT = "environment"
    SECRETS_MAPPING = "secrets_mapping"
    CALLBACK = "callback"


@dataclass(slots=True, frozen=True)
class CredentialResolution:
    """Result of Databento credential resolution."""

    value: str | None
    source: CredentialSource
    injected: bool

    @property
    def available(self) -> bool:
        """Return ``True`` when a credential value was located."""
        return bool(self.value)


def resolve_databento_api_key(
    *,
    explicit: str | None = None,
    environ: Mapping[str, str] | None = None,
    secrets: Mapping[str, str] | None = None,
    fetch: Callable[[], str | None] | None = None,
    inject: bool = True,
) -> CredentialResolution:
    """Resolve the Databento API key from multiple sources."""
    environ_map: Mapping[str, str] = environ or os.environ

    def _normalize(raw: str | None) -> str | None:
        if raw is None:
            return None
        trimmed = raw.strip()
        return trimmed if trimmed else None

    candidate = _normalize(explicit)
    if candidate is not None:
        injected = _inject_if_required(candidate, environ=environ_map, inject=inject)
        if injected:
            LOGGER.info("databento_api_key_injected", source=CredentialSource.EXPLICIT.value)
        return CredentialResolution(candidate, CredentialSource.EXPLICIT, injected)

    for env_key in _PRIMARY_ENV_KEYS:
        candidate = _normalize(environ_map.get(env_key))
        if candidate is not None:
            return CredentialResolution(candidate, CredentialSource.ENVIRONMENT, False)

    if secrets is not None:
        for secret_key in _SECRET_KEYS:
            candidate = _normalize(secrets.get(secret_key))
            if candidate is not None:
                injected = _inject_if_required(candidate, environ=environ_map, inject=inject)
                LOGGER.info(
                    "databento_api_key_injected",
                    source=CredentialSource.SECRETS_MAPPING.value,
                )
                return CredentialResolution(candidate, CredentialSource.SECRETS_MAPPING, injected)

    if fetch is not None:
        try:
            candidate = _normalize(fetch())
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.warning("databento_api_key_fetch_failed", error=str(exc))
        else:
            if candidate is not None:
                injected = _inject_if_required(candidate, environ=environ_map, inject=inject)
                LOGGER.info(
                    "databento_api_key_injected",
                    source=CredentialSource.CALLBACK.value,
                )
                return CredentialResolution(candidate, CredentialSource.CALLBACK, injected)

    return CredentialResolution(None, CredentialSource.NONE, False)


def _inject_if_required(
    value: str,
    *,
    environ: Mapping[str, str],
    inject: bool,
) -> bool:
    if not inject:
        return False

    existing = environ.get("DATABENTO_API_KEY")
    if existing == value:
        return False
    os.environ["DATABENTO_API_KEY"] = value
    return True


__all__ = [
    "CredentialResolution",
    "CredentialSource",
    "resolve_databento_api_key",
]
