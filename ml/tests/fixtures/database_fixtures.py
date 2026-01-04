#!/usr/bin/env python3

"""
Database fixtures for ML module testing.

This module provides:
- Test database initialization
- Schema creation utilities
- Test data seeding
- Database cleanup utilities
- Transaction management for test isolation

"""

from __future__ import annotations

import errno
import fcntl
import logging
import os
import subprocess
import time
from collections.abc import Generator
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path
from typing import Any, IO, Callable, ContextManager

import pytest
from unittest.mock import MagicMock, patch


# Track schema initialization per connection URL to avoid re-running migrations
_SCHEMA_INITIALIZED: dict[str, bool] = {}

import pandas as pd
from sqlalchemy import MetaData
from sqlalchemy import create_engine
from sqlalchemy import event
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ml.core.db_engine import EngineManager


_DEFAULT_TEST_DB_PORT = os.getenv("TEST_DB_PORT", "5434")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://postgres:postgres@localhost:{_DEFAULT_TEST_DB_PORT}/nautilus_test",
)
_TEMPLATE_DB_NAME = os.getenv("TEST_DB_TEMPLATE_NAME", "nautilus_template")

_ENGINE_MANAGER_PATCH_TARGETS: tuple[str, ...] = (
    "ml.common.db_utils.EngineManager.get_engine",
    "ml.core.db_engine.EngineManager.get_engine",
    "ml.dashboard.service.EngineManager.get_engine",
    "ml.dashboard.services.metrics_service.EngineManager.get_engine",
    "ml.dashboard.services.trading_service.EngineManager.get_engine",
    "ml.observability.db_persistence.EngineManager.get_engine",
    "ml.stores.data_processor.EngineManager.get_engine",
    "ml.stores.data_store.EngineManager.get_engine",
    "ml.stores.feature_store.EngineManager.get_engine",
    "ml.stores.model_store.EngineManager.get_engine",
    "ml.stores.strategy_store.EngineManager.get_engine",
)


def _template_engine_url() -> str:
    """
    Build a connection string pointing at the template database.
    """
    try:
        from sqlalchemy.engine import make_url

        url = make_url(DATABASE_URL)
        template_url = url.set(database=_TEMPLATE_DB_NAME)
        return template_url.render_as_string(hide_password=False)
    except Exception:
        prefix, _, _ = DATABASE_URL.rpartition("/")
        if not prefix:
            return DATABASE_URL
        return f"{prefix}/{_TEMPLATE_DB_NAME}"


@contextmanager
def patch_engine_manager(
    *,
    engine: Engine | MagicMock | None = None,
    record_calls: bool = False,
    side_effect: Exception | None = None,
) -> Generator[MagicMock, None, None]:
    """
    Patch EngineManager lookups to return a deterministic engine-like object.

    Ensures patched engines do not leak into subsequent tests by disposing the
    EngineManager cache after use.
    """

    mock_engine = engine if engine is not None else MagicMock(name="engine")
    call_log: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _patched_get_engine(*args: object, **kwargs: object) -> MagicMock:
        if record_calls:
            call_log.append((args, kwargs))
        if side_effect is not None:
            raise side_effect
        return mock_engine

    setattr(mock_engine, "_engine_manager_calls", call_log)

    with ExitStack() as stack:
        for target in _ENGINE_MANAGER_PATCH_TARGETS:
            with suppress(AttributeError, ModuleNotFoundError):
                stack.enter_context(patch(target, _patched_get_engine))
        try:
            yield mock_engine
        finally:
            EngineManager.dispose_all()


_PATCH_ENGINE_MANAGER_FN = patch_engine_manager


@pytest.fixture(name="patch_engine_manager")
def _patch_engine_manager_fixture() -> Callable[..., ContextManager[MagicMock]]:
    """
    Provide the patch_engine_manager context manager via fixture injection.
    """

    def _factory(
        *,
        engine: Engine | MagicMock | None = None,
        record_calls: bool = False,
        side_effect: Exception | None = None,
    ) -> ContextManager[MagicMock]:
        return _PATCH_ENGINE_MANAGER_FN(
            engine=engine,
            record_calls=record_calls,
            side_effect=side_effect,
        )

    return _factory


@pytest.fixture
def mock_engine_manager() -> Generator[MagicMock, None, None]:
    """
    Provide a MagicMock-backed engine via EngineManager while ensuring cleanup.
    """

    with patch_engine_manager() as engine_mock:
        yield engine_mock


@pytest.fixture
def real_engine_manager() -> Generator[None, None, None]:
    """
    Ensure EngineManager cache is clear before and after a test uses real engines.
    """

    EngineManager.dispose_all()
    try:
        yield
    finally:
        EngineManager.dispose_all()


@pytest.fixture(scope="session")
def template_database() -> Generator[str, None, None]:
    """
    Create a session-scoped template database for clones (PostgreSQL only).

    Returns the connection string pointing to the template DB.
    """
    if "sqlite" in DATABASE_URL:
        # SQLite path uses the base URL directly
        yield DATABASE_URL
        return

    template_url = _template_engine_url()
    # Ensure template database exists (create if missing)
    from sqlalchemy.engine import make_url

    base_url = make_url(DATABASE_URL)
    admin_engine = EngineManager.get_engine(
        base_url.render_as_string(hide_password=False),
        pool_size=1,
        max_overflow=0,
    )
    with admin_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f"DROP DATABASE IF EXISTS {_TEMPLATE_DB_NAME}"))
        conn.execute(text(f"CREATE DATABASE {_TEMPLATE_DB_NAME}"))

    template_engine = EngineManager.get_engine(template_url, pool_size=2, max_overflow=0)
    # Initialize schema once
    td = TestDatabase(engine=template_engine, connection_string=template_url)
    td.init_schema()
    try:
        yield template_url
    finally:
        # Keep template alive for the session; cleanup handled by EngineManager
        EngineManager.dispose_all()


@pytest.fixture
def cloned_test_database(template_database: str) -> Generator[str, None, None]:
    """
    Provide a per-test clone of the session template database.

    Yields a connection string pointing to an isolated clone. Drops the clone and
    disposes EngineManager caches on teardown.
    """
    if "sqlite" in template_database:
        yield template_database
        return

    # Build clone name and clone DB from template
    from uuid import uuid4

    clone_name = f"nautilus_clone_{uuid4().hex}"
    template_engine = EngineManager.get_engine(template_database, pool_size=2, max_overflow=0)
    clone_url = _clone_database(
        source_db=_TEMPLATE_DB_NAME,
        clone_db=clone_name,
        template_engine=template_engine,
    )

    try:
        yield clone_url
    finally:
        # Drop clone and clear engine caches
        _drop_database(clone_name, template_engine)
        EngineManager.dispose_all()


def _connect_timeout_seconds() -> int:
    try:
        return max(1, int(os.getenv("TEST_DB_CONNECT_TIMEOUT", "15")))
    except ValueError:
        return 15


def is_postgresql_running() -> bool:
    """
    Check if PostgreSQL is running and accessible.
    """
    try:
        import psycopg2  # Local import to avoid hard dependency for unit tests

        conn = psycopg2.connect(
            DATABASE_URL,
            connect_timeout=_connect_timeout_seconds(),
        )
        conn.close()
        return True
    except Exception:
        return False


def _clone_database(
    *,
    source_db: str,
    clone_db: str,
    template_engine: Engine,
) -> str:
    """
    Clone a PostgreSQL database from a template into a new database.

    Returns the connection string for the clone.
    """
    with template_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        # Ensure no active sessions on the template before cloning
        try:
            conn.execute(
                text(
                    """
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = :db_name
                      AND pid <> pg_backend_pid()
                    """,
                ),
                {"db_name": source_db},
            )
        except Exception:
            pass
        # Drop if exists to keep clones idempotent in test teardown reruns
        conn.execute(text(f"DROP DATABASE IF EXISTS {clone_db}"))
        conn.execute(text(f"CREATE DATABASE {clone_db} TEMPLATE {source_db}"))

    url = template_engine.url
    try:
        clone_url = url.set(database=clone_db).render_as_string(hide_password=False)
    except Exception:
        clone_url = f"postgresql://{url.username}:{url.password}@{url.host}:{url.port}/{clone_db}"
    return clone_url


def _drop_database(db_name: str, engine: Engine) -> None:
    """
    Drop a PostgreSQL database safely (best-effort).
    """
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        try:
            conn.execute(
                text(
                    """
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = :db_name
""",
                ),
                {"db_name": db_name},
            )
        except Exception:
            pass
        try:
            conn.execute(text(f"DROP DATABASE IF EXISTS {db_name}"))
        except Exception:
            pass


def start_postgresql() -> None:
    """
    Attempt to start PostgreSQL if not running.
    """
    if is_postgresql_running():
        print("PostgreSQL is already running")
        return

    print("Starting PostgreSQL...")
    try:
        if os.path.exists("/usr/local/bin/pg_ctl"):
            subprocess.run(
                ["pg_ctl", "start", "-D", "/usr/local/var/postgres"],
                capture_output=True,
                check=False,
            )
        elif os.path.exists("/usr/bin/systemctl"):
            subprocess.run(
                ["sudo", "systemctl", "start", "postgresql"],
                capture_output=True,
                check=False,
            )

        for _ in range(10):
            if is_postgresql_running():
                print("PostgreSQL started successfully")
                return
            time.sleep(1)
    except Exception as exc:
        print(f"Could not start PostgreSQL: {exc}")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_db_lock(name: str = "db") -> IO[str] | None:
    """
    Acquire an interprocess file lock to serialize DB/serial-marked tests across workers.
    """
    lock_dir = Path.home() / ".nautilus" / "ml"
    try:
        lock_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        import logging

        logging.getLogger(__name__).debug(
            "Creating lock directory failed: %s",
            exc,
            exc_info=True,
        )
    lock_path = lock_dir / f"pytest_{name}.lock"
    fh = open(lock_path, "a+")

    timeout_env = os.getenv("ML_TEST_DB_LOCK_TIMEOUT_SEC", "15")
    try:
        timeout_sec = max(1.0, float(timeout_env))
    except Exception:
        timeout_sec = 15.0
    deadline = time.monotonic() + timeout_sec

    while True:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                fh.seek(0)
                fh.truncate()
                fh.write(str(os.getpid()))
                fh.flush()
            except Exception as exc:
                import logging

                logging.getLogger(__name__).debug(
                    "Recording DB lock ownership failed: %s",
                    exc,
                    exc_info=True,
                )
            return fh
        except OSError as exc:  # pragma: no cover - contention path
            if exc.errno not in (errno.EAGAIN, errno.EACCES):
                raise

            try:
                owner_pid_str = lock_path.read_text().strip()
                owner_pid = int(owner_pid_str) if owner_pid_str else -1
            except Exception:
                owner_pid = -1

            if owner_pid > 0 and not _pid_alive(owner_pid):
                try:
                    fh.seek(0)
                    fh.truncate()
                    fh.flush()
                except Exception as inner_exc:
                    import logging

                    logging.getLogger(__name__).debug(
                        "Clearing stale DB lock ownership failed: %s",
                        inner_exc,
                        exc_info=True,
                    )

            if time.monotonic() >= deadline:
                try:
                    fh.close()
                except Exception as inner_exc:
                    import logging

                    logging.getLogger(__name__).debug(
                        "Closing DB lock file failed: %s",
                        inner_exc,
                        exc_info=True,
                    )
                return None

            time.sleep(0.1)


def release_db_lock(fh: IO[str]) -> None:
    """Release an acquired DB file lock."""

    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        try:
            fh.close()
        except Exception as exc:
            import logging

            logging.getLogger(__name__).debug(
                "Closing DB lock handle failed: %s",
                exc,
                exc_info=True,
            )


class TestDatabase:
    """
    Test database manager for ML tests.
    """

    def __init__(
        self,
        engine: Engine | None = None,
        connection_string: str | None = None,
        use_in_memory: bool = False,  # Default to False, prefer PostgreSQL
        auto_rollback: bool = True,
        echo: bool = False,
    ):
        """
        Initialize test database.

        Parameters
        ----------
        engine : Engine, optional
            Pre-configured SQLAlchemy engine (takes precedence)
        connection_string : str, optional
            Database connection string (auto-generated if not provided)
        use_in_memory : bool, default False
            Use in-memory database for unit tests (deprecated, use PostgreSQL)
        auto_rollback : bool, default True
            Automatically rollback transactions after each test
        echo : bool, default False
            Echo SQL statements

        """
        self.use_in_memory = use_in_memory
        self.auto_rollback = auto_rollback
        self.echo = echo

        # Use provided engine if available
        if engine:
            self.engine = engine
            # Use provided connection_string if available, otherwise extract from engine
            # Note: str(engine.url) masks the password with *** which breaks subsequent connections
            if connection_string:
                self.connection_string = connection_string
            else:
                # Fallback - this will have masked password
                self.connection_string = str(engine.url)
            if "sqlite" not in self.connection_string:
                try:
                    if not event.contains(
                        self.engine,
                        "connect",
                        TestDatabase._enforce_search_path,
                    ):
                        event.listen(
                            self.engine,
                            "connect",
                            TestDatabase._enforce_search_path,
                        )
                    if not event.contains(
                        self.engine,
                        "checkout",
                        TestDatabase._enforce_search_path_on_checkout,
                    ):
                        event.listen(
                            self.engine,
                            "checkout",
                            TestDatabase._enforce_search_path_on_checkout,
                        )
                except Exception:
                    logging.getLogger(__name__).debug(
                        "Registering search_path hooks failed",
                        exc_info=True,
                    )
        else:
            # Set up connection string
            if connection_string:
                self.connection_string = connection_string
            else:
                # Default to PostgreSQL from environment
                import os

                self.connection_string = os.getenv(
                    "DATABASE_URL",
                    "postgresql://postgres:postgres@localhost:5434/nautilus_test",
                )

            # Use EngineManager to get or create engine
            # This ensures proper connection pooling and prevents "too many clients" errors
            if "sqlite" in self.connection_string:
                # SQLite-specific settings
                connect_args = (
                    {"check_same_thread": False} if "memory" in self.connection_string else {}
                )
                self.engine = EngineManager.get_engine(
                    self.connection_string,
                    echo=echo,
                    connect_args=connect_args,
                )
                # Enable foreign keys for SQLite
                event.listen(self.engine, "connect", self._enable_sqlite_foreign_keys)
            else:
                # PostgreSQL or other databases
                # Use conservative pool settings for tests
                self.engine = EngineManager.get_engine(
                    self.connection_string,
                    echo=echo,
                    pool_size=2,  # Conservative for tests
                    max_overflow=3,  # Conservative for tests
                )
                event.listen(self.engine, "connect", self._enforce_search_path)
                event.listen(self.engine, "checkout", self._enforce_search_path_on_checkout)

        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)

        # Track if schema is initialized
        self._schema_initialized = False

    @staticmethod
    def _enable_sqlite_foreign_keys(dbapi_conn: Any, connection_record: Any) -> None:
        """
        Enable foreign key constraints for SQLite.
        """
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    @staticmethod
    def _enforce_search_path(dbapi_conn: Any, connection_record: Any) -> None:  # noqa: ARG002
        """
        Ensure PostgreSQL sessions default to the public schema first.
        """
        TestDatabase._set_search_path(dbapi_conn)

    @staticmethod
    def _enforce_search_path_on_checkout(
        dbapi_conn: Any,
        connection_record: Any,  # noqa: ARG002 - required signature for SQLAlchemy
        connection_proxy: Any,  # noqa: ARG002 - required signature for SQLAlchemy
    ) -> None:
        """
        Reset pooled PostgreSQL connections to the canonical search path.
        """
        TestDatabase._set_search_path(dbapi_conn)

    @staticmethod
    def _set_search_path(dbapi_conn: Any) -> None:
        """
        Set the PostgreSQL search path so public objects shadow ml_registry.
        """
        try:
            cursor = dbapi_conn.cursor()
        except Exception:
            return
        try:
            cursor.execute("SET search_path TO public, pg_catalog, ml_registry")
        finally:
            try:
                cursor.close()
            except Exception:
                pass

    def init_schema(self, schema_files: list[Path] | None = None) -> None:
        """
        Initialize database schema.

        Parameters
        ----------
        schema_files : list[Path], optional
            SQL schema files to execute (uses default ML schema if not provided)

        """
        # Short-circuit when schema already initialized for this engine URL
        try:
            engine_key = str(self.engine.url)
        except Exception:
            engine_key = "default"

        # Detect stale/partial initialization for PostgreSQL and force re-run if needed
        needs_init = True
        try:
            with self.engine.connect() as _conn:
                probe = _conn.execute(
                    text(
                        "SELECT EXISTS (SELECT 1 FROM information_schema.routines WHERE routine_name = 'create_monthly_partitions')",
                    ),
                ).scalar()
                # Presence of helper function indicates bootstrap schema landed
                needs_init = not bool(probe)
        except Exception:
            # On any error, fall back to running initialization
            needs_init = True

        if (
            self._schema_initialized or _SCHEMA_INITIALIZED.get(engine_key, False)
        ) and not needs_init:
            self._schema_initialized = True
            return

        # Get default schema files if not provided
        if schema_files is None:
            schema_files = self._get_default_schema_files()

        if "sqlite" in self.connection_string:
            with self.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                for schema_file in schema_files:
                    if not schema_file.exists():
                        continue

                    sql = schema_file.read_text()
                    sql = self._adapt_sql_for_sqlite(sql)

                    from typing import Callable, Iterable, cast as _cast

                    try:
                        from ml.cli.apply_migrations import split_statements as _split

                        splitter: Callable[[str], Iterable[str]] = _split
                    except Exception:
                        def _fallback_splitter(x: str) -> list[str]:
                            return [s for s in x.split(";") if s.strip()]

                        splitter = _fallback_splitter

                    for statement in splitter(sql):
                        statement = statement.strip()
                        if not statement or statement.startswith("--"):
                            continue
                        try:
                            conn.execute(text(statement))
                        except Exception as exc:
                            if any(
                                keyword in str(exc).lower()
                                for keyword in ("partition", "inherit", "tablespace")
                            ):
                                continue
                            raise

                try:
                    conn.execute(
                        text(
                            """
CREATE OR REPLACE FUNCTION attach_partition_with_data(
    target_table TEXT,
    partition_name TEXT,
    start_ns BIGINT,
    end_ns BIGINT
)
RETURNS VOID AS $$
DECLARE
    default_partition TEXT := target_table || '_default';
    column_list TEXT;
BEGIN
    SELECT string_agg(format('%I', column_name), ', ')
    INTO column_list
    FROM information_schema.columns
    WHERE table_schema = current_schema()
      AND table_name = target_table
      AND (is_generated = 'NEVER' OR is_generated IS NULL);

    IF column_list IS NULL THEN
        RAISE EXCEPTION 'Unable to resolve column list for %', target_table;
    END IF;

    EXECUTE format(
        'CREATE TABLE IF NOT EXISTS %I (LIKE %I INCLUDING ALL)',
        partition_name,
        target_table
    );
    EXECUTE format(
        'INSERT INTO %I (%s) SELECT %s FROM %I WHERE ts_event >= %L AND ts_event < %L',
        partition_name,
        column_list,
        column_list,
        default_partition,
        start_ns,
        end_ns
    );
    EXECUTE format(
        'DELETE FROM %I WHERE ts_event >= %L AND ts_event < %L',
        default_partition,
        start_ns,
        end_ns
    );
    EXECUTE format(
        'ALTER TABLE %I ATTACH PARTITION %I FOR VALUES FROM (%L) TO (%L)',
        target_table,
        partition_name,
        start_ns,
        end_ns
    );
EXCEPTION
    WHEN duplicate_object THEN
        NULL;
    WHEN object_in_use THEN
        NULL;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION create_monthly_partitions(
    table_name TEXT,
    start_date DATE,
    num_months INTEGER
)
RETURNS VOID AS $$
DECLARE
    partition_date DATE;
    partition_name TEXT;
    start_ns BIGINT;
    end_ns BIGINT;
BEGIN
    FOR i IN 0..num_months-1 LOOP
        partition_date := start_date + (i || ' months')::INTERVAL;
        partition_name := table_name || '_' || TO_CHAR(partition_date, 'YYYY_MM');
        start_ns := EXTRACT(EPOCH FROM partition_date) * 1000000000;
        end_ns := EXTRACT(EPOCH FROM partition_date + '1 month'::INTERVAL) * 1000000000;
        BEGIN
            EXECUTE format(
                'CREATE TABLE %I PARTITION OF %I FOR VALUES FROM (%L) TO (%L)',
                partition_name, table_name, start_ns, end_ns
            );
        EXCEPTION
            WHEN check_violation THEN
                PERFORM attach_partition_with_data(table_name, partition_name, start_ns, end_ns);
            WHEN duplicate_table THEN
                NULL;
            WHEN object_in_use THEN
                NULL;
            WHEN invalid_object_definition THEN
                NULL;
            WHEN SQLSTATE '42P17' THEN
                NULL;
        END;
    END LOOP;
END;
$$ LANGUAGE plpgsql;
                                """,
                        ),
                    )
                except Exception:
                    pass

        # Apply canonical baseline via migration runner (idempotent, best-effort)
        try:
            from ml.cli.apply_migrations import apply_files as _apply_files
            from ml.cli.apply_migrations import build_plan as _build_plan
            from ml.tasks.db import MigrationSchema

            plan = _build_plan(include_optional=True, schema=MigrationSchema.BOTH)
            _apply_files(self.engine, plan, dry_run=False)
        except Exception:
            # Ignore if migration helpers are unavailable in the environment
            pass
        else:
            # Ensure subsequent statements run with public first in search_path.
            try:
                with self.engine.begin() as _conn:
                    _conn.execute(
                        text("SET search_path TO public, pg_catalog, ml_registry"),
                    )
            except Exception:
                pass

        self._schema_initialized = True
        _SCHEMA_INITIALIZED[engine_key] = True

        # Re-run database hygiene to disable legacy triggers and ensure partitions.
        try:
            from ml.tests.fix_database_issues import _ensure_functions_and_partitions

            _ensure_functions_and_partitions(self.engine)
        except Exception:
            # Non-fatal: tests depending on the helper will surface issues explicitly.
            pass

    def _get_default_schema_files(self) -> list[Path]:
        """
        Get default ML schema files.
        """
        # Resolve against the ml/ package so we stay within the repository tree even
        # when tests run from an arbitrary working directory. The previous relative
        # walk jumped to the project root and looked for ``stores/migrations`` which
        # does not exist (the migrations live under ``ml/stores``).
        ml_root = Path(__file__).resolve().parents[2]
        migrations_dir = ml_root / "stores" / "migrations"

        # Consolidated bootstrap schema (2025-10-01: merged 18 migrations)
        schema_files = [
            migrations_dir / "001_bootstrap_schema.sql",
        ]

        # Filter to existing files
        return [f for f in schema_files if f.exists()]

    def _adapt_sql_for_sqlite(self, sql: str) -> str:
        """
        Adapt PostgreSQL SQL for SQLite compatibility.
        """
        # Remove PostgreSQL-specific features
        replacements = [
            ("SERIAL", "INTEGER"),
            ("BIGSERIAL", "INTEGER"),
            ("JSONB", "TEXT"),
            ("JSON", "TEXT"),
            ("UUID", "TEXT"),
            ("BYTEA", "BLOB"),
            ("TIMESTAMP WITH TIME ZONE", "TIMESTAMP"),
            ("TIMESTAMPTZ", "TIMESTAMP"),
            ("::BIGINT", ""),
            ("::INTEGER", ""),
            ("::TEXT", ""),
            ("ON CONFLICT DO NOTHING", "OR IGNORE"),
            ("PARTITION BY", "-- PARTITION BY"),  # Comment out partitioning
            ("INHERITS", "-- INHERITS"),  # Comment out inheritance
            ("IF NOT EXISTS", ""),  # SQLite doesn't support for all statements
        ]

        for old, new in replacements:
            sql = sql.replace(old, new)

        # Remove CREATE EXTENSION statements
        lines = sql.split("\n")
        filtered_lines = []
        for line in lines:
            if not any(
                keyword in line.upper()
                for keyword in ["CREATE EXTENSION", "CREATE INDEX CONCURRENTLY"]
            ):
                filtered_lines.append(line)

        return "\n".join(filtered_lines)

    def seed_test_data(self, data_type: str = "basic") -> None:
        """
        Seed database with test data.

        Parameters
        ----------
        data_type : str, default "basic"
            Type of test data to seed ("basic", "full", "minimal")

        """
        with self.get_session() as session:
            if data_type == "basic":
                self._seed_basic_data(session)
            elif data_type == "full":
                self._seed_full_data(session)
            elif data_type == "minimal":
                self._seed_minimal_data(session)

            session.commit()

    def _seed_basic_data(self, session: Session) -> None:
        """
        Seed basic test data.
        """
        # Add instruments
        session.execute(
            text(
                """
                INSERT INTO ml_instruments (instrument_id, symbol, asset_type, tick_size, lot_size)
                VALUES
                    ('EURUSD.SIM', 'EURUSD', 'FX', 0.00001, 1000),
                    ('SPY.XNAS', 'SPY', 'EQUITY', 0.01, 1),
                    ('BTCUSD.COINBASE', 'BTCUSD', 'CRYPTO', 0.01, 0.001)
                ON CONFLICT DO NOTHING
            """,
            ),
        )

        # Add sample feature values
        import time

        current_ns = int(time.time() * 1e9)

        session.execute(
            text(
                """
                INSERT INTO ml_feature_values (
                    feature_set_id, instrument_id, ts_event, ts_init, values, is_live
                )
                VALUES (
                    'test_features_v1',
                    'EURUSD.SIM',
                    :ts_event,
                    :ts_init,
                    '{"sma_20": 1.0900, "rsi": 55.5, "volume": 12345}'::text,
                    false
                )
                ON CONFLICT DO NOTHING
            """,
            ),
            {"ts_event": current_ns, "ts_init": current_ns + 1000},
        )

    def _seed_full_data(self, session: Session) -> None:
        """
        Seed comprehensive test data.
        """
        # Start with basic data
        self._seed_basic_data(session)

        # Add more instruments
        instruments = [
            ("GBPUSD.SIM", "GBPUSD", "FX", 0.00001, 1000),
            ("QQQ.XNAS", "QQQ", "EQUITY", 0.01, 1),
            ("AAPL.XNAS", "AAPL", "EQUITY", 0.01, 1),
        ]

        for inst_id, symbol, asset_type, tick_size, lot_size in instruments:
            session.execute(
                text(
                    """
                    INSERT INTO ml_instruments (instrument_id, symbol, asset_type, tick_size, lot_size)
                    VALUES (:inst_id, :symbol, :asset_type, :tick_size, :lot_size)
                    ON CONFLICT DO NOTHING
                """,
                ),
                {
                    "inst_id": inst_id,
                    "symbol": symbol,
                    "asset_type": asset_type,
                    "tick_size": tick_size,
                    "lot_size": lot_size,
                },
            )

        # Add historical feature values
        import time

        base_ts = int(time.time() * 1e9)

        for i in range(100):
            ts = base_ts - i * 60 * 1e9  # 1-minute intervals

            for inst_id in ["EURUSD.SIM", "GBPUSD.SIM", "SPY.XNAS"]:
                session.execute(
                    text(
                        """
                        INSERT INTO ml_feature_values (
                            feature_set_id, instrument_id, ts_event, ts_init, values, is_live
                        )
                        VALUES (
                            'test_features_v1',
                            :inst_id,
                            :ts_event,
                            :ts_init,
                            :values,
                            false
                        )
                        ON CONFLICT DO NOTHING
                    """,
                    ),
                    {
                        "inst_id": inst_id,
                        "ts_event": ts,
                        "ts_init": ts + 1000,
                        "values": f'{{"sma_20": {1.09 + i * 0.0001}, "rsi": {50 + i % 30}, "volume": {10000 + i * 100}}}',
                    },
                )

    def _seed_minimal_data(self, session: Session) -> None:
        """
        Seed minimal test data.
        """
        # Just one instrument
        session.execute(
            text(
                """
                INSERT INTO ml_instruments (instrument_id, symbol, asset_type, tick_size, lot_size)
                VALUES ('TEST.SIM', 'TEST', 'EQUITY', 0.01, 1)
                ON CONFLICT DO NOTHING
            """,
            ),
        )

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get database session with automatic cleanup.

        Yields
        ------
        Session
            Database session

        """
        session = self.SessionLocal()
        try:
            yield session
            if not self.auto_rollback:
                session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            if self.auto_rollback:
                session.rollback()
            session.close()

    def clear_all_data(self) -> None:
        """
        Clear all data from database (preserves schema).
        """
        with self.engine.connect() as conn:
            # Get all tables
            metadata = MetaData()
            metadata.reflect(bind=self.engine)

            # Delete data from all tables (in reverse dependency order)
            for table in reversed(metadata.sorted_tables):
                conn.execute(
                    text(f"DELETE FROM {table.name}"),
                )

            conn.commit()

    def drop_all(self) -> None:
        """
        Drop all tables and schema.
        """
        with self.engine.connect() as conn:
            # Get all tables
            metadata = MetaData()
            metadata.reflect(bind=self.engine)

            # Drop all tables
            metadata.drop_all(bind=self.engine)

            conn.commit()

        self._schema_initialized = False

    def cleanup(self) -> None:
        """
        Clean up test database resources.
        """
        # Note: We don't dispose the engine here anymore as it's managed by EngineManager
        # The engine will be disposed by the cleanup_engines fixture after each test

        # Remove temporary file if using file-based SQLite
        if hasattr(self, "db_path") and self.db_path.exists():
            try:
                self.db_path.unlink()
            except Exception:
                import logging as _logging

                _logging.getLogger(__name__).debug(
                    "Failed to unlink temp DB path; ignoring",
                    exc_info=True,
                )

    def execute_sql(self, sql: str, params: dict[str, Any] | None = None) -> Any:
        """
        Execute arbitrary SQL statement.

        Parameters
        ----------
        sql : str
            SQL statement to execute
        params : dict, optional
            Parameters for SQL statement

        Returns
        -------
        Any
            Query result

        """
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            conn.commit()
            return result

    def get_table_data(self, table_name: str) -> pd.DataFrame:
        """
        Get all data from a table as DataFrame.

        Parameters
        ----------
        table_name : str
            Name of table to query

        Returns
        -------
        pd.DataFrame
            Table data

        """
        return pd.read_sql(
            f"SELECT * FROM {table_name}",
            self.engine,
        )


def create_test_database(**kwargs: Any) -> TestDatabase:
    """
    Factory function to create test database.

    Parameters
    ----------
    **kwargs
        Arguments passed to TestDatabase constructor

    Returns
    -------
    TestDatabase
        Configured test database instance

    """
    return TestDatabase(**kwargs)


@contextmanager
def temp_database(
    use_in_memory: bool = True,
    init_schema: bool = True,
    seed_data: str | None = None,
) -> Generator[TestDatabase, None, None]:
    """
    Context manager for temporary test database.

    Parameters
    ----------
    use_in_memory : bool, default True
        Use in-memory database
    init_schema : bool, default True
        Initialize database schema
    seed_data : str, optional
        Type of test data to seed

    Yields
    ------
    TestDatabase
        Temporary test database

    """
    db = TestDatabase(use_in_memory=use_in_memory)

    try:
        if init_schema:
            db.init_schema()

        if seed_data:
            db.seed_test_data(seed_data)

        yield db
    finally:
        db.cleanup()


class DatabaseSnapshot:
    """
    Utility for database state snapshots and restoration.
    """

    def __init__(self, database: TestDatabase):
        """
        Initialize snapshot utility.
        """
        self.database = database
        self.snapshots: dict[str, dict[str, pd.DataFrame]] = {}

    def take_snapshot(self, name: str = "default") -> None:
        """
        Take snapshot of current database state.
        """
        snapshot: dict[str, pd.DataFrame] = {}

        # Get all tables
        metadata = MetaData()
        metadata.reflect(bind=self.database.engine)

        # Save data from each table
        for table in metadata.tables:
            table_frame = self.database.get_table_data(table)
            snapshot[table] = table_frame

        self.snapshots[name] = snapshot

    def restore_snapshot(self, name: str = "default") -> None:
        """
        Restore database to snapshot state.
        """
        if name not in self.snapshots:
            raise ValueError(f"Snapshot '{name}' not found")

        snapshot = self.snapshots[name]

        # Clear all data
        self.database.clear_all_data()

        # Restore data for each table
        with self.database.get_session() as session:
            for table_name, df in snapshot.items():
                if not df.empty:
                    df.to_sql(table_name, self.database.engine, if_exists="append", index=False)
            session.commit()

    def has_snapshot(self, name: str = "default") -> bool:
        """
        Check if snapshot exists.
        """
        return name in self.snapshots


# ============================================================================
# Pytest fixtures
# ============================================================================


@pytest.fixture(scope="session")
def database_engine() -> Generator[Engine, None, None]:
    """Create a single database engine for the entire test session."""

    logger = logging.getLogger(__name__)

    logger.info(
        "Creating session-scoped database engine with pool_size=5, max_overflow=10",
    )

    engine = EngineManager.get_engine(
        DATABASE_URL,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=300,
    )

    try:
        yield engine
    finally:
        try:
            pool = engine.pool
            logger.info(
                "Database engine pool stats before disposal: size=%s, checked_in=%s, "
                "checked_out=%s, overflow=%s",
                pool.size(),
                pool.checkedin(),
                pool.checkedout(),
                pool.overflow(),
            )
        except Exception:
            pass

        EngineManager.dispose_all()


@pytest.fixture(scope="session")
def database_session_factory(database_engine: Engine) -> sessionmaker:
    """Create a session factory bound to the shared session-scoped engine."""

    return sessionmaker(bind=database_engine)


@pytest.fixture
def database_session(database_session_factory: sessionmaker) -> Generator[Session, None, None]:
    """Create an isolated database session for each test using nested transactions."""

    connection = database_session_factory.bind.connect()
    transaction = connection.begin()
    session = database_session_factory(bind=connection)
    connection.begin_nested()

    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def isolated_engine() -> Generator[Engine, None, None]:
    """Create an isolated in-memory SQLite engine for unit tests."""

    engine = create_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def postgres_connection() -> str:
    """Expose PostgreSQL connection string for legacy tests."""

    return DATABASE_URL


@pytest.fixture(scope="function")
def clean_postgres_db() -> Generator[None, None, None]:
    """Ensure a clean PostgreSQL state before and after each test."""

    if os.getenv("TEST_DB_SKIP_TRUNCATE") == "1":
        yield
        return

    engine = EngineManager.get_engine(
        DATABASE_URL,
        pool_size=2,
        max_overflow=3,
        pool_pre_ping=True,
    )

    timeout_ms = int(os.getenv("TEST_DB_TRUNCATE_TIMEOUT_MS", "2000"))

    def _set_timeout(connection: Any) -> None:
        try:
            connection.execute(text(f"SET LOCAL statement_timeout = '{timeout_ms}ms'"))
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "Set statement timeout failed: %s",
                exc,
                exc_info=True,
            )

    def _truncate_and_verify(connection: Any) -> None:
        try:
            connection.rollback()
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "Transaction rollback failed: %s",
                exc,
                exc_info=True,
            )

        result = connection.execute(
            text(
                """
                SELECT tablename FROM pg_tables
                WHERE schemaname = 'public'
                  AND tablename LIKE 'ml_%'
                """,
            ),
        )
        tables = [row[0] for row in result]

        for table in tables:
            try:
                connection.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
            except Exception as exc:
                print(f"Warning during cleanup of {table}: {exc}")
                try:
                    connection.rollback()
                except Exception as inner_exc:
                    logging.getLogger(__name__).debug(
                        "Transaction rollback failed: %s",
                        inner_exc,
                        exc_info=True,
                    )

        core_tables = ("ml_model_predictions", "ml_strategy_signals", "ml_feature_values")
        retry_needed = False
        for table in core_tables:
            try:
                count = connection.execute(text(f"SELECT COUNT(*) FROM {table}"))
            except Exception:
                continue
            rows = count.scalar_one_or_none()
            if rows:
                retry_needed = True

        if retry_needed:
            for table in tables:
                try:
                    connection.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
                except Exception as exc:
                    print(f"Retry cleanup warning for {table}: {exc}")
                    try:
                        connection.rollback()
                    except Exception as inner_exc:
                        logging.getLogger(__name__).debug(
                            "Transaction rollback failed: %s",
                            inner_exc,
                            exc_info=True,
                        )

    try:
        with engine.connect() as conn:
            conn.execute(text("SET CONSTRAINTS ALL DEFERRED"))
            _set_timeout(conn)
            _truncate_and_verify(conn)
            conn.commit()
    except Exception as exc:
        print(f"clean_postgres_db pre-test cleanup skipped: {exc}")

    try:
        yield
    finally:
        try:
            with engine.connect() as conn:
                _set_timeout(conn)
                _truncate_and_verify(conn)
                conn.commit()
        except Exception as exc:
            print(f"clean_postgres_db post-test cleanup skipped: {exc}")


@pytest.fixture(scope="class")
def clean_postgres_db_class() -> Generator[None, None, None]:
    """Class-scoped PostgreSQL cleanup fixture."""

    engine = EngineManager.get_engine(
        DATABASE_URL,
        pool_size=2,
        max_overflow=3,
        pool_pre_ping=True,
    )
    timeout_ms = int(os.getenv("TEST_DB_TRUNCATE_TIMEOUT_MS", "2000"))

    def _truncate_all() -> None:
        try:
            with engine.connect() as conn:
                try:
                    conn.rollback()
                except Exception as exc:
                    logging.getLogger(__name__).debug(
                        "Transaction rollback failed: %s",
                        exc,
                        exc_info=True,
                    )
                conn.execute(text("SET CONSTRAINTS ALL DEFERRED"))
                try:
                    conn.execute(text(f"SET LOCAL statement_timeout = '{timeout_ms}ms'"))
                except Exception as exc:
                    logging.getLogger(__name__).debug(
                        "Set statement timeout failed: %s",
                        exc,
                        exc_info=True,
                    )
                result = conn.execute(
                    text(
                        """
                        SELECT tablename FROM pg_tables
                        WHERE schemaname = 'public'
                          AND tablename LIKE 'ml_%'
                        """,
                    ),
                )
                tables = [row[0] for row in result]
                for table in tables:
                    try:
                        conn.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
                    except Exception as exc:
                        print(f"Warning during class-scope cleanup of {table}: {exc}")
                for table in ("ml_model_predictions", "ml_strategy_signals", "ml_feature_values"):
                    try:
                        count = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    except Exception:
                        continue
                    rows = count.scalar_one_or_none()
                    if rows:
                        try:
                            conn.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
                        except Exception as exc:
                            print(f"Retry cleanup warning for {table}: {exc}")
                conn.commit()
        except Exception as exc:
            print(f"clean_postgres_db_class cleanup skipped: {exc}")

    fh = acquire_db_lock("db")
    try:
        if fh is None:
            print("clean_postgres_db_class: DB lock not acquired; proceeding without lock")
    except Exception:
        fh = None

    previous = os.getenv("TEST_DB_SKIP_TRUNCATE")
    os.environ["TEST_DB_SKIP_TRUNCATE"] = "1"
    _truncate_all()
    try:
        yield
    finally:
        _truncate_all()
        if previous is None:
            os.environ.pop("TEST_DB_SKIP_TRUNCATE", None)
        else:
            os.environ["TEST_DB_SKIP_TRUNCATE"] = previous
        if fh is not None:
            release_db_lock(fh)


@pytest.fixture(scope="module")
def clean_postgres_db_module() -> Generator[None, None, None]:
    """Module-scoped PostgreSQL cleanup fixture."""

    engine = EngineManager.get_engine(
        DATABASE_URL,
        pool_size=2,
        max_overflow=3,
        pool_pre_ping=True,
    )
    timeout_ms = int(os.getenv("TEST_DB_TRUNCATE_TIMEOUT_MS", "2000"))

    def _truncate_all() -> None:
        try:
            with engine.connect() as conn:
                conn.execute(text("SET CONSTRAINTS ALL DEFERRED"))
                try:
                    conn.execute(text(f"SET LOCAL statement_timeout = '{timeout_ms}ms'"))
                except Exception as exc:
                    logging.getLogger(__name__).debug(
                        "Set statement timeout failed: %s",
                        exc,
                        exc_info=True,
                    )
                result = conn.execute(
                    text(
                        """
                        SELECT tablename FROM pg_tables
                        WHERE schemaname = 'public'
                          AND tablename LIKE 'ml_%'
                        """,
                    ),
                )
                tables = [row[0] for row in result]
                for table in tables:
                    try:
                        conn.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
                    except Exception as exc:
                        print(f"Warning during module-scope cleanup of {table}: {exc}")
                        try:
                            conn.rollback()
                        except Exception as inner_exc:
                            logging.getLogger(__name__).debug(
                                "Transaction rollback failed: %s",
                                inner_exc,
                                exc_info=True,
                            )
                for table in ("ml_model_predictions", "ml_strategy_signals", "ml_feature_values"):
                    try:
                        count = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    except Exception:
                        continue
                    rows = count.scalar_one_or_none()
                    if rows:
                        try:
                            conn.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
                        except Exception as exc:
                            print(f"Retry cleanup warning for {table}: {exc}")
                            try:
                                conn.rollback()
                            except Exception as inner_exc:
                                logging.getLogger(__name__).debug(
                                    "Transaction rollback failed: %s",
                                    inner_exc,
                                    exc_info=True,
                                )
                conn.commit()
        except Exception as exc:
            print(f"clean_postgres_db_module cleanup skipped: {exc}")

    print("clean_postgres_db_module: attempting DB lock")
    fh = acquire_db_lock("db")
    try:
        if fh is None:
            print("clean_postgres_db_module: DB lock not acquired; proceeding without lock")
        else:
            print("clean_postgres_db_module: acquired DB lock")
    except Exception:
        fh = None

    previous = os.getenv("TEST_DB_SKIP_TRUNCATE")
    os.environ["TEST_DB_SKIP_TRUNCATE"] = "1"
    _truncate_all()
    try:
        yield
    finally:
        _truncate_all()
        if previous is None:
            os.environ.pop("TEST_DB_SKIP_TRUNCATE", None)
        else:
            os.environ["TEST_DB_SKIP_TRUNCATE"] = previous
        if fh is not None:
            release_db_lock(fh)


@pytest.fixture(scope="session")
def module_test_database() -> Generator[TestDatabase, None, None]:
    """Session-scoped PostgreSQL database with schema initialized once."""

    if not is_postgresql_running():
        pytest.skip(f"PostgreSQL not reachable at {DATABASE_URL}")

    _SCHEMA_INITIALIZED.clear()

    engine = EngineManager.get_engine(
        DATABASE_URL,
        pool_size=2,
        max_overflow=3,
        pool_pre_ping=True,
    )

    db = TestDatabase(engine=engine, connection_string=DATABASE_URL, auto_rollback=False)

    try:
        try:
            db.init_schema()
        except Exception:
            logging.getLogger(__name__).debug(
                "module_test_database init_schema failed; continuing",
                exc_info=True,
            )
        yield db
    finally:
        db.cleanup()


@pytest.fixture(scope="function")
def test_database() -> Generator[TestDatabase, None, None]:
    """Create a TestDatabase bound to PostgreSQL with schema initialized."""

    logger = logging.getLogger(__name__)

    if not is_postgresql_running():
        pytest.skip(f"PostgreSQL not reachable at {DATABASE_URL}")

    try:
        EngineManager.dispose_all()
        logger.debug("EngineManager cache cleared before test")
    except Exception as exc:
        logger.warning(
            "Failed to dispose EngineManager engines before test: %s",
            exc,
            exc_info=True,
        )

    _SCHEMA_INITIALIZED.clear()
    logger.debug("Schema initialization tracking cleared")

    engine = EngineManager.get_engine(
        DATABASE_URL,
        pool_size=2,
        max_overflow=3,
        pool_pre_ping=True,
    )

    if os.getenv("TEST_DB_SKIP_TRUNCATE") != "1":
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT tablename FROM pg_tables
                    WHERE schemaname = 'public'
                      AND tablename NOT LIKE 'pg_%'
                      AND tablename NOT LIKE 'sql_%'
                    """,
                ),
            )
            for row in result:
                table_name = row[0]
                try:
                    conn.execute(text(f"TRUNCATE TABLE {table_name} CASCADE"))
                except Exception:
                    logging.getLogger(__name__).debug(
                        "TRUNCATE failed for table %s; ignoring in test cleanup",
                        table_name,
                        exc_info=True,
                    )
            conn.commit()

    try:
        with engine.connect() as conn:
            for table in ("ml_feature_values", "ml_model_predictions", "ml_strategy_signals"):
                try:
                    conn.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
                except Exception:
                    pass
            conn.commit()
    except Exception:
        pass

    db = TestDatabase(engine=engine, connection_string=DATABASE_URL, auto_rollback=False)
    try:
        db.init_schema()
    except Exception:
        logging.getLogger(__name__).debug(
            "Database init_schema failed; continuing",
            exc_info=True,
        )

    try:
        yield db
    finally:
        try:
            db.cleanup()
        except Exception as exc:
            logger.warning(
                "TestDatabase cleanup failed: %s",
                exc,
                exc_info=True,
            )

        try:
            EngineManager.dispose_all()
            logger.debug("EngineManager cache cleared after test")
        except Exception as exc:
            logger.warning(
                "Failed to dispose EngineManager engines after test: %s",
                exc,
                exc_info=True,
            )


@pytest.fixture
def test_db_engine(test_database: TestDatabase) -> Engine:
    """Expose the SQLAlchemy engine from TestDatabase."""

    return test_database.engine


@pytest.fixture
def test_db_session(test_database: TestDatabase) -> Generator[Session, None, None]:
    """Yield a session with automatic rollback from TestDatabase."""

    with test_database.get_session() as session:
        yield session


@pytest.fixture
def seeded_database(test_database: TestDatabase) -> TestDatabase:
    """Seed basic data for tests requiring pre-populated state."""

    test_database.seed_test_data("basic")
    return test_database


@pytest.fixture
def database_snapshot(test_database: TestDatabase) -> DatabaseSnapshot:
    """Provide DatabaseSnapshot helper."""

    return DatabaseSnapshot(test_database)


@pytest.fixture
def connection_monitor(database_engine: Engine) -> Generator[None, None, None]:
    """Monitor database connections during test execution."""

    logger = logging.getLogger(__name__)

    with database_engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()",
            ),
        )
        initial_count = result.scalar()

    try:
        yield
    finally:
        with database_engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()",
                ),
            )
            final_count = result.scalar()

        if final_count > initial_count + 2:
            logger.warning(
                "Potential connection leak detected: Initial=%s, Final=%s",
                initial_count,
                final_count,
            )


__all__ = sorted(
    [
        "_SCHEMA_INITIALIZED",
        "DATABASE_URL",
        "DatabaseSnapshot",
        "TestDatabase",
        "acquire_db_lock",
        "clean_postgres_db",
        "clean_postgres_db_class",
        "clean_postgres_db_module",
        "cloned_test_database",
        "connection_monitor",
        "database_engine",
        "database_session",
        "database_session_factory",
        "database_snapshot",
        "is_postgresql_running",
        "mock_engine_manager",
        "isolated_engine",
        "module_test_database",
        "patch_engine_manager",
        "postgres_connection",
        "release_db_lock",
        "real_engine_manager",
        "seeded_database",
        "start_postgresql",
        "template_database",
        "temp_database",
        "test_database",
        "test_db_engine",
        "test_db_session",
    ],
)
