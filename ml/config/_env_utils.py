"""
Utility helpers for environment-driven configuration parsing.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Final

from ml.common.db_connections import ConnectionRole
from ml.common.db_connections import collect_postgres_candidates


LOGGER: Final[logging.Logger] = logging.getLogger(__name__)


def ensure_env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    """
    Return ``env`` when provided, otherwise ``os.environ``.
    """
    return env if env is not None else os.environ


def env_truthy(source: Mapping[str, str], key: str, default: bool = False) -> bool:
    """
    Interpret an environment variable as a boolean.
    """
    raw = source.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_float(source: Mapping[str, str], key: str, default: float) -> float:
    """
    Parse a float value from environment variables.
    """
    raw = source.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        LOGGER.debug("invalid_float_env_override", extra={"key": key, "value": raw})
        return default


def env_positive_float(source: Mapping[str, str], key: str, default: float) -> float:
    """
    Parse a positive float value, falling back to ``default`` when invalid.
    """
    value = env_float(source, key, default)
    if value < 0.0:
        LOGGER.debug("negative_float_env_override", extra={"key": key, "value": value})
        return default
    return value


def env_positive_int(source: Mapping[str, str], key: str, default: int) -> int:
    """
    Parse a positive integer value from environment variables.
    """
    raw = source.get(key)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        LOGGER.debug("invalid_int_env_override", extra={"key": key, "value": raw})
        return default
    if value <= 0:
        LOGGER.debug("non_positive_int_env_override", extra={"key": key, "value": value})
        return default
    return value


def env_non_negative_int(source: Mapping[str, str], key: str, default: int) -> int:
    """
    Parse a non-negative integer value from environment variables.
    """
    raw = source.get(key)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        LOGGER.debug("invalid_int_env_override", extra={"key": key, "value": raw})
        return default
    if value < 0:
        LOGGER.debug("negative_int_env_override", extra={"key": key, "value": value})
        return default
    return value


def resolve_db_connection(env: Mapping[str, str]) -> str | None:
    """
    Resolve the primary database connection string from environment overrides.
    """
    explicit = (
        env.get("ML_DB_CONNECTION")
        or env.get("DB_CONNECTION")
        or env.get("NAUTILUS_DB")
        or env.get("DATABASE_URL")
    )
    candidates = collect_postgres_candidates(
        ConnectionRole.PRIMARY,
        explicit=explicit,
    ).urls
    if not candidates:
        return None
    return candidates[0]
