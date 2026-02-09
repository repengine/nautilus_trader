"""
Database Lifecycle Management Component.

This module provides database lifecycle management extracted from MLIntegrationManager
as part of the god-class decomposition effort (Phase 3.6.1). The component handles:

- PostgreSQL connection probing with multi-candidate fallback
- Docker container startup for local development
- Database migration execution (CLI-based and fallback inline)
- Connection candidate management and progressive fallback

The component follows Protocol-First Interface Design and can be used independently
or composed via the MLIntegrationManagerFacade.

Example
-------
>>> from ml.core.common.database_lifecycle import DatabaseLifecycleComponent
>>> component = DatabaseLifecycleComponent(
...     connection_candidates=("postgresql://postgres:postgres@localhost:5432/nautilus",),
...     auto_migrate=True,
... )
>>> if component.is_postgres_running():
...     component.init_database()

"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import text
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import OperationalError

from ml.core.db_engine import EngineManager
from ml.stores.migrations_runner import MigrationSchema


if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


logger = logging.getLogger(__name__)


@dataclass
class DatabaseLifecycleComponent:
    """
    Manages PostgreSQL database lifecycle including connection probing,
    container startup, and migration execution.

    This component implements the database management responsibilities
    extracted from MLIntegrationManager. It follows the Progressive Fallback
    Chain pattern (PRIMARY -> CACHED -> FILE -> DUMMY) for resilient
    database connectivity.

    Attributes
    ----------
    connection_candidates : tuple[str, ...]
        Ordered list of PostgreSQL connection URLs to probe.
    auto_start_postgres : bool
        Whether to automatically start PostgreSQL via Docker if not running.
    auto_migrate : bool
        Whether to automatically run database migrations on init.
    allow_dummy : bool
        Whether to allow dummy mode when PostgreSQL is unavailable.
    db_connection : str
        The currently selected database connection string.

    Example
    -------
    >>> component = DatabaseLifecycleComponent(
    ...     connection_candidates=(
    ...         "postgresql://postgres:postgres@localhost:5433/nautilus",
    ...         "postgresql://postgres:postgres@localhost:5432/nautilus",
    ...     ),
    ...     auto_start_postgres=False,
    ...     auto_migrate=True,
    ... )
    >>> if component.is_postgres_running():
    ...     print(f"Connected to: {component.db_connection}")
    ...     component.init_database()

    """

    connection_candidates: tuple[str, ...]
    auto_start_postgres: bool = False
    auto_migrate: bool = False
    allow_dummy: bool = False

    # State (initialized in __post_init__)
    db_connection: str = field(init=False)

    def __post_init__(self) -> None:
        """
        Initialize with first candidate as default connection.

        Raises
        ------
        ValueError
            If connection_candidates is empty.

        """
        if not self.connection_candidates:
            raise ValueError("At least one connection candidate required")
        self.db_connection = self.connection_candidates[0]

    def init_database(self) -> None:
        """
        Initialize database connection and run migrations.

        This method checks if PostgreSQL is running (or in dummy mode) and
        runs migrations if auto_migrate is enabled. Safe to call multiple times.

        Example
        -------
        >>> component = DatabaseLifecycleComponent(
        ...     connection_candidates=("postgresql://...",),
        ...     auto_migrate=True,
        ... )
        >>> component.init_database()  # Runs migrations if connected

        """
        # At this point either PostgreSQL is running or we are in dummy mode.
        if self.allow_dummy and not self.is_postgres_running():
            # Dummy mode: nothing to do
            return
        # Run migrations if needed
        if self.auto_migrate:
            self.run_migrations()

    def is_postgres_running(self) -> bool:
        """
        Check whether any candidate PostgreSQL connection is reachable.

        Iterates through connection_candidates in order, probing each with
        a SELECT 1 query. If a non-primary candidate is reachable, updates
        db_connection and disposes the stale engine.

        Returns
        -------
        bool
            True if any PostgreSQL candidate is reachable, False otherwise.

        Example
        -------
        >>> component = DatabaseLifecycleComponent(
        ...     connection_candidates=(
        ...         "postgresql://postgres:postgres@localhost:5432/nautilus",
        ...         "postgresql://postgres:postgres@localhost:5433/nautilus",
        ...     ),
        ... )
        >>> if component.is_postgres_running():
        ...     print(f"PostgreSQL reachable at: {component.db_connection}")

        """
        for candidate in self.connection_candidates:
            if self.can_connect(candidate):
                if candidate != self.db_connection:
                    try:  # pragma: no cover - structlog guard
                        alt_url = make_url(candidate)
                        host = alt_url.host or "localhost"
                        port = alt_url.port or "?"
                    except Exception:  # pragma: no cover - defensive guard
                        host = "localhost"
                        port = "?"
                    logger.info(
                        "PostgreSQL reachable - updating integration connection (host=%s port=%s)",
                        host,
                        port,
                    )
                    EngineManager.dispose_engine(self.db_connection)
                    self.db_connection = candidate
                return True

        logger.debug(
            "postgres_unreachable candidates=%s",
            list(self.connection_candidates),
        )
        return False

    def can_connect(self, connection_string: str) -> bool:
        """
        Probe whether a database connection string is usable.

        Attempts to execute a SELECT 1 query on the connection. On failure,
        disposes the engine and returns False.

        Parameters
        ----------
        connection_string : str
            PostgreSQL connection URL to probe.

        Returns
        -------
        bool
            True if connection succeeds, False otherwise.

        Example
        -------
        >>> component = DatabaseLifecycleComponent(connection_candidates=(...,))
        >>> if component.can_connect("postgresql://localhost:5432/nautilus"):
        ...     print("Connection successful!")

        """
        try:
            engine = EngineManager.get_engine(connection_string)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except OperationalError:
            EngineManager.dispose_engine(connection_string)
            return False
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("postgres_connect_probe_failed", exc_info=True)
            EngineManager.dispose_engine(connection_string)
            return False

    def start_postgres_container(self) -> None:
        """
        Start PostgreSQL using Docker Compose if available, else docker run.

        Searches for Docker Compose files in order:
        1. ML_COMPOSE_FILE environment variable
        2. ml/deployment/docker-compose.yml
        3. ml/docker-compose.dev.yml
        4. docker-compose.yml

        Falls back to plain `docker run` if no compose file found.

        Raises
        ------
        RuntimeError
            If docker executable is not found in PATH.
        RuntimeError
            If PostgreSQL fails to start within 30 seconds.

        Example
        -------
        >>> component = DatabaseLifecycleComponent(
        ...     connection_candidates=(...,),
        ...     auto_start_postgres=True,
        ... )
        >>> component.start_postgres_container()

        """
        logger.info("Starting PostgreSQL (preferring Docker Compose if available)...")

        compose_file = None
        docker_path = shutil.which("docker")
        if docker_path is None:
            raise RuntimeError("docker executable not found in PATH")

        # Prefer explicit env override, then deployment compose, then dev compose, then root
        candidates: list[object] = []
        env_compose = os.getenv("ML_COMPOSE_FILE")
        if env_compose:
            candidates.append(Path(env_compose))
        candidates.extend(
            [
                Path("ml/deployment/docker-compose.yml"),
                Path("ml/docker-compose.dev.yml"),
                Path("docker-compose.yml"),
            ],
        )
        for candidate in candidates:
            try:
                if isinstance(candidate, Path) and candidate.exists():
                    compose_file = candidate
                    break
            except Exception:
                # Fallback silently to next candidate
                continue

        if compose_file is not None:
            try:
                subprocess.run(
                    [docker_path, "compose", "-f", str(compose_file), "up", "-d", "postgres"],
                    check=True,
                )
            except Exception:
                compose_file = None

        if compose_file is None:
            result = subprocess.run(
                [
                    docker_path,
                    "ps",
                    "-a",
                    "--filter",
                    "name=nautilus-postgres",
                    "--format",
                    "{{.Names}}",
                ],
                capture_output=True,
                text=True,
            )

            if "nautilus-postgres" in result.stdout:
                subprocess.run([docker_path, "start", "nautilus-postgres"], check=True)
            else:
                subprocess.run(
                    [
                        docker_path,
                        "run",
                        "-d",
                        "--name",
                        "nautilus-postgres",
                        "-e",
                        "POSTGRES_PASSWORD=postgres",
                        "-e",
                        "POSTGRES_DB=nautilus",
                        "-p",
                        "5432:5432",
                        "postgres:15",
                    ],
                    check=True,
                )

        # Wait for PostgreSQL to be ready
        for _ in range(30):
            if self.is_postgres_running():
                logger.info("PostgreSQL is ready!")
                return
            time.sleep(1)

        raise RuntimeError("PostgreSQL failed to start within 30 seconds")

    def run_migrations(self) -> None:
        """
        Run database migrations using the shared migration plan builder.

        Attempts to use the shared migration helpers first. If unavailable,
        falls back to applying a hardcoded list of base migrations inline.

        Handles "already exists" and similar idempotent errors gracefully
        with debug logging rather than failing.

        Example
        -------
        >>> component = DatabaseLifecycleComponent(
        ...     connection_candidates=("postgresql://localhost:5432/nautilus",),
        ...     auto_migrate=True,
        ... )
        >>> component.run_migrations()  # Safe to call multiple times

        """
        logger.info("Running database migrations...")

        # Decide plan from environment
        env_full = os.getenv("ML_MIGRATIONS_FULL", "").lower() in {"1", "true", "yes"}
        # Prefer full migrations in production-like environments
        env_mode = os.getenv("ML_ENV", "").lower()
        full = env_full or env_mode in {"prod", "production"}
        schema_env = os.getenv("ML_MIGRATIONS_SCHEMA", MigrationSchema.BOTH.value)
        try:
            schema_enum = MigrationSchema(schema_env)
        except ValueError:
            schema_enum = MigrationSchema.BOTH

        engine = EngineManager.get_engine(self.db_connection)

        # Prefer the shared migration helpers to keep in sync
        try:
            from ml.stores.migrations_runner import apply_migration_files as _apply
            from ml.stores.migrations_runner import build_migration_plan as _build

            plan = _build(include_optional=full, schema=schema_enum)
            result = _apply(engine, plan, dry_run=False)
            logger.info(
                "Migrations applied=%d skipped=%d warnings=%d errors=%d",
                result.applied,
                result.skipped,
                result.warnings,
                result.errors,
            )
        except Exception as exc:
            # Fallback to the former inlined list with simple splitting
            logger.warning("Migration helpers unavailable (%s); using fallback plan", exc)
            migrations = [
                "ml/registry/migrations/001_initial_schema.sql",
                "ml/registry/migrations/002_add_cold_path_fields.sql",
                "ml/registry/migrations/003_add_artifact_digest.sql",
                "ml/stores/migrations_bootstrap/001_bootstrap.sql",
            ]

            # Use the same splitter as the shared migration helper when available
            try:
                from ml.stores.common.sql_splitter import split_sql_statements as _splitter
            except Exception:

                def _splitter(sql: str) -> Iterable[str]:
                    return [s for s in sql.split(";") if s.strip()]

            for migration_path in migrations:
                migration_file = Path(migration_path)
                if not migration_file.exists():
                    continue

                logger.info("Applying migration: %s", migration_path)
                sql = migration_file.read_text(encoding="utf-8")
                try:
                    with engine.begin() as conn:
                        for statement in _splitter(sql):
                            try:
                                conn.execute(text(statement))
                            except Exception as e:
                                msg = str(e).lower()
                                if (
                                    "already exists" in msg
                                    or "does not exist" in msg
                                    or "duplicate" in msg
                                ):
                                    logger.debug("Migration notice (%s): %s", migration_path, e)
                                else:
                                    logger.warning(
                                        "Warning in migration %s: %s", migration_path, e
                                    )
                except Exception as exc2:
                    logger.error("Migration failed for %s: %s", migration_path, exc2)


__all__ = ["DatabaseLifecycleComponent"]
