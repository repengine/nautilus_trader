"""
Utilities for resolving PostgreSQL connection strings.

The ML stack is routinely executed in multiple environments:

* Local development (typically Docker Compose forwarding to ``localhost:5433``)
* CI runners (often ``localhost:5432``)
* Deployed services (custom hosts provided via environment variables)

Historically each CLI or service constructed its own default connection string
and therefore disagreed on which port to probe first.  When a developer had
both a legacy local PostgreSQL instance on ``5432`` *and* the ML Compose stack
on ``5433`` the orchestrator would frequently attach to the wrong instance,
exhaust the connection pool, and fall back to file-backed stores.

This module centralises connection resolution.  Callers obtain an ordered
candidate list using :func:`collect_postgres_candidates` and may optionally
select the first working connection via :func:`select_first_working_connection`.
All helpers are cold-path safe and fully typed.

Example
-------
>>> from ml.common.db_connections import ConnectionRole, collect_postgres_candidates
>>> collect_postgres_candidates(ConnectionRole.PRIMARY)[0]
'postgresql://postgres:postgres@localhost:5433/nautilus'

When the Compose stack is not running the helper automatically falls back to
``localhost:5432`` or any user-provided override.

"""

from __future__ import annotations

import os
from collections.abc import Iterator
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import structlog
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from ml.core.db_engine import EngineManager


LOGGER = structlog.get_logger(__name__)


class ConnectionRole(StrEnum):
    """
    Logical connection roles used throughout the ML stack.
    """

    PRIMARY = "primary"
    MIGRATION = "migration"
    REGISTRY = "registry"
    PARTITION = "partition"


_ROLE_ENV_KEYS: dict[ConnectionRole, tuple[str, ...]] = {
    ConnectionRole.PRIMARY: (
        "NAUTILUS_DB",
        "DATABASE_URL",
        "ML_DB_CONNECTION",
        "NAUTILUS_DB_CONNECTION",
        "DB_CONNECTION",
    ),
    ConnectionRole.MIGRATION: (
        "DATABASE_URL",
        "ML_DB_CONNECTION",
        "NAUTILUS_DB",
    ),
    ConnectionRole.REGISTRY: (
        "NAUTILUS_REGISTRY_DB_URL",
        "DATABASE_URL",
        "ML_DB_CONNECTION",
        "NAUTILUS_DB",
    ),
    ConnectionRole.PARTITION: (
        "ML_DB_CONNECTION",
        "DATABASE_URL",
        "NAUTILUS_DB",
    ),
}


def connection_env_priority(role: ConnectionRole) -> tuple[str, ...]:
    """
    Return the ordered environment variable precedence for ``role``.
    """
    return _ROLE_ENV_KEYS.get(role, ())


@dataclass(slots=True, frozen=True)
class ConnectionCandidates:
    """
    Ordered list of candidate PostgreSQL URLs for a given role.
    """

    role: ConnectionRole
    urls: tuple[str, ...]

    def __iter__(self) -> Iterator[str]:  # pragma: no cover - convenience only
        return iter(self.urls)


def collect_postgres_candidates(
    role: ConnectionRole,
    *,
    explicit: str | None = None,
) -> ConnectionCandidates:
    """
    Collect ordered PostgreSQL connection string candidates for ``role``.

    Parameters
    ----------
    role:
        Logical connection consumer (primary runtime, migrations, registry, …).
    explicit:
        Optional caller provided override.  When supplied this URL is placed at
        the front of the candidate list.

    Returns
    -------
    ConnectionCandidates
        Ordered, deduplicated candidate URLs.

    """
    env_keys = _ROLE_ENV_KEYS.get(role, ())
    seen: set[str] = set()
    ordered: list[str] = []

    def _try_add(url: str | None) -> None:
        if not url:
            return
        trimmed = url.strip()
        if not trimmed or trimmed in seen:
            return
        seen.add(trimmed)
        ordered.append(trimmed)

    _try_add(explicit)
    for key in env_keys:
        _try_add(os.getenv(key))

    user = os.getenv("ML_DB_USER") or os.getenv("POSTGRES_USER") or "postgres"
    password = os.getenv("ML_DB_PASSWORD") or os.getenv("POSTGRES_PASSWORD") or "postgres"
    database = os.getenv("ML_DB_NAME") or os.getenv("POSTGRES_DB") or "nautilus"

    host_candidates = _dedupe_preserve_order(
        (
            os.getenv("ML_DB_HOST"),
            os.getenv("POSTGRES_HOST"),
            os.getenv("NAUTILUS_DB_HOST"),
            "localhost",
            "postgres",
        ),
    )
    port_candidates = _dedupe_preserve_order(
        (
            os.getenv("ML_DB_PORT"),
            os.getenv("POSTGRES_HOST_PORT"),
            os.getenv("NAUTILUS_DB_PORT"),
            "5433",
            "5432",
        ),
    )

    for host in host_candidates:
        if host is None:
            continue
        normalized_host = host.strip()
        if not normalized_host:
            continue
        for port in port_candidates:
            if port is None:
                continue
            normalized_port = port.strip()
            if not normalized_port:
                continue
            candidate = (
                f"postgresql://{user}:{password}@{normalized_host}:{normalized_port}/{database}"
            )
            _try_add(candidate)

    return ConnectionCandidates(role=role, urls=tuple(ordered))


def select_first_working_connection(candidates: Sequence[str]) -> str:
    """
    Return the first connection string that responds to ``SELECT 1``.

    Raises
    ------
    RuntimeError
        Raised when none of the provided candidates are reachable.

    """
    last_error: Exception | None = None

    for url in candidates:
        try:
            engine = EngineManager.get_engine(url)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            LOGGER.debug("postgres_candidate_selected", connection=url)
            return url
        except OperationalError as exc:
            EngineManager.dispose_engine(url)
            last_error = exc
            LOGGER.debug("postgres_candidate_unavailable", connection=url, error=str(exc))
            continue
        except Exception as exc:  # pragma: no cover - defensive guard
            EngineManager.dispose_engine(url)
            last_error = exc
            LOGGER.debug("postgres_candidate_probe_failed", connection=url, error=str(exc))
            continue

    raise RuntimeError("No PostgreSQL candidates are reachable") from last_error


def _dedupe_preserve_order(values: Sequence[str | None]) -> tuple[str | None, ...]:
    seen: set[str | None] = set()
    ordered: list[str | None] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


__all__ = [
    "ConnectionCandidates",
    "ConnectionRole",
    "collect_postgres_candidates",
    "select_first_working_connection",
]
