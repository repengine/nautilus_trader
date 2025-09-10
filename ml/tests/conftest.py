#!/usr/bin/env python3
"""
Consolidated pytest fixtures and configuration for ML module tests.

This module provides:
- Database connection management with proper pooling
- Test isolation using transactions
- Mock services for external dependencies
- Hypothesis testing profiles
- Performance monitoring fixtures

"""


import os
import subprocess
import tempfile
import time
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock
from unittest.mock import patch

import psycopg2
import pytest
from hypothesis import settings
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy.pool import StaticPool


# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv

    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass  # dotenv not installed, use system environment

from ml.core.db_engine import EngineManager
from ml.tests.data import get_model_registry_dir
from ml.tests.data import get_model_registry_rollout_dir
from ml.tests.data import get_test_data_dir
from ml.tests.fixtures.database_fixtures import DatabaseSnapshot
from ml.tests.fixtures.database_fixtures import TestDatabase
from ml.tests.fixtures.database_fixtures import create_test_database
from ml.tests.fixtures.database_fixtures import temp_database
from ml.tests.fixtures.mock_services import create_mock_databento_client
from ml.tests.fixtures.mock_services import create_mock_fred_client
from ml.tests.fixtures.mock_services import create_mock_postgresql
from ml.tests.fixtures.mock_services import create_mock_redis
from ml.tests.fixtures.mock_services import create_mock_yahoo_client
from ml.tests.unit.config.test_config import TestConfig
from ml.tests.unit.config.test_config import TestEnvironment
from ml.tests.unit.config.test_config import get_test_config


if TYPE_CHECKING:
    from collections.abc import Generator

# ============================================================================
# Constants and Configuration
# ============================================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/nautilus",
)

# ============================================================================
# Mark TDD prototype suites for default exclusion
# ============================================================================


_PROTOTYPE_PATH_SUFFIXES = [
    # Domain bookkeeping prototypes (Phase 1 & 2)
    "ml/tests/property/test_domain_bookkeeping_phase1.py",
    "ml/tests/contracts/test_domain_bookkeeping_schemas.py",
    "ml/tests/metamorphic/test_domain_bookkeeping_event_flow.py",
    "ml/tests/property/test_domain_bookkeeping_phase2.py",
    "ml/tests/contracts/test_observability_pipeline_schemas.py",
    "ml/tests/metamorphic/test_observability_correlation.py",
    "ml/tests/combinatorial/test_domain_bookkeeping_configs.py",
    "ml/tests/property/test_domain_bookkeeping_stateful.py",
]


def _mark_prototypes(items: list[pytest.Item]) -> None:
    """
    Mark TDD prototype tests so they don't block by default.

    Adds the `prototype` marker to tests whose path matches known TDD prototype files.
    The root config excludes `prototype` by default via `-m 'not prototype'`.

    Parameters
    ----------
    items : list[pytest.Item]
        Collected pytest items to inspect and mark.

    """
    for item in items:
        nodeid = item.nodeid.replace("::", "/")
        for suffix in _PROTOTYPE_PATH_SUFFIXES:
            if nodeid.endswith(suffix):
                item.add_marker(pytest.mark.prototype)
                break


# ============================================================================
# Hypothesis Configuration
# ============================================================================

# Register CI profile for faster tests in CI
settings.register_profile(
    "ci",
    max_examples=50,  # Reduced from default 100
    deadline=5000,  # 5 seconds
    print_blob=True,
    report_multiple_bugs=True,
    derandomize=True,  # Reproducible in CI
)

# Register dev profile for thorough local testing
settings.register_profile(
    "dev",
    max_examples=200,  # More thorough than default
    deadline=None,  # No deadline for debugging
    print_blob=True,
    report_multiple_bugs=True,
)

# Register debug profile
settings.register_profile(
    "debug",
    max_examples=10,
    deadline=None,
    print_blob=True,
    verbosity=2,  # Verbose output
)

# Use CI profile if running in CI environment
if os.getenv("CI"):
    settings.load_profile("ci")
else:
    # Default to fast profile locally unless explicitly overridden
    try:
        settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "ci"))
    except Exception:
        settings.load_profile("ci")

# ============================================================================
# Session-scoped fixtures for expensive resources
# ============================================================================


@pytest.fixture(scope="session")
def database_engine() -> Generator[Engine, None, None]:
    """
    Create a single database engine for the entire test session.

    This prevents connection exhaustion by reusing the same engine
    across all tests. The engine is properly disposed at session end.

    Following Martin Fowler's Test Pyramid principle:
    - Use a single shared connection for fast tests
    - Only integration tests get separate connections

    """
    # Use conservative pooling for tests (prevents exhaustion)
    engine = EngineManager.get_engine(
        DATABASE_URL,
        pool_size=2,  # Small pool for tests
        max_overflow=3,  # Limited overflow
        pool_pre_ping=True,  # Test connections before use
        pool_recycle=300,  # Recycle connections every 5 minutes
    )

    yield engine

    # Clean up at session end
    EngineManager.dispose_all()


@pytest.fixture(scope="session")
def database_session_factory(database_engine: Engine) -> sessionmaker:
    """
    Create a session factory for the test session.

    This factory creates sessions that share the same connection pool, preventing
    connection exhaustion.

    """
    return sessionmaker(bind=database_engine)


# ============================================================================
# Function-scoped fixtures with proper isolation
# ============================================================================


@pytest.fixture
def database_session(database_session_factory: sessionmaker) -> Generator[Session, None, None]:
    """
    Create an isolated database session for each test.

    Uses transactions with automatic rollback to ensure test isolation
    without creating new connections.

    Based on SQLAlchemy testing best practices:
    https://docs.sqlalchemy.org/en/14/orm/session_transaction.html

    """
    connection = database_session_factory.bind.connect()
    transaction = connection.begin()

    # Create session bound to the connection
    session = database_session_factory(bind=connection)

    # Begin nested transaction for test isolation
    nested = connection.begin_nested()

    yield session

    # Rollback the transaction to undo test changes
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def isolated_engine() -> Generator[Engine, None, None]:
    """
    Create an isolated in-memory SQLite engine for unit tests.

    This is perfect for tests that don't need PostgreSQL-specific features and provides
    complete isolation with zero connection overhead.

    """
    # Use in-memory SQLite with shared cache for speed
    engine = create_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,  # Keep connection alive
        connect_args={"check_same_thread": False},
    )

    # Tables will be created by stores as needed

    yield engine

    engine.dispose()


@pytest.fixture
def postgres_connection() -> str:
    """
    Get PostgreSQL connection string from environment.
    """
    return DATABASE_URL


# ============================================================================
# PostgreSQL Management Fixtures
# ============================================================================


def is_postgresql_running() -> bool:
    """
    Check if PostgreSQL is running and accessible.
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.close()
        return True
    except psycopg2.OperationalError:
        return False


def start_postgresql() -> None:
    """
    Attempt to start PostgreSQL if not running.
    """
    if is_postgresql_running():
        print("PostgreSQL is already running")
        return

    print("Starting PostgreSQL...")
    try:
        # Try to start PostgreSQL (platform-specific)
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

        # Wait for PostgreSQL to be ready
        for _ in range(10):
            if is_postgresql_running():
                print("PostgreSQL started successfully")
                return
            time.sleep(1)
    except Exception as e:
        print(f"Could not start PostgreSQL: {e}")


# ============================================================================
# Compatibility fixture: clean Postgres DB pre/post test
# ============================================================================


@pytest.fixture(scope="function")
def clean_postgres_db() -> Generator[None, None, None]:
    """
    Ensure a clean PostgreSQL state before and after each test.

    - Uses `EngineManager.get_engine(DATABASE_URL)` to respect pooling
    - Defers constraints to allow TRUNCATE order-agnostically
    - TRUNCATEs all user tables in the `public` schema pre/post test

    This fixture exists for compatibility with legacy tests which
    assume a clean database. Prefer transaction-scoped isolation
    where possible, but this keeps existing tests unblocked.

    """
    engine = EngineManager.get_engine(
        DATABASE_URL,
        pool_size=2,
        max_overflow=3,
        pool_pre_ping=True,
    )

    # Clean before test
    try:
        with engine.connect() as conn:
            conn.execute(text("SET CONSTRAINTS ALL DEFERRED"))
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
                table = row[0]
                try:
                    conn.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
                except Exception as e:  # Best-effort cleanup
                    print(f"Warning during pre-test cleanup of {table}: {e}")
            conn.commit()
    except Exception as e:
        # If cleanup cannot run (e.g., DB down), proceed; tests may be skipped by gate
        print(f"clean_postgres_db pre-test cleanup skipped: {e}")

    yield

    # Clean after test as well
    try:
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
                table = row[0]
                try:
                    conn.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
                except Exception as e:
                    print(f"Warning during post-test cleanup of {table}: {e}")
            conn.commit()
    except Exception as e:
        print(f"clean_postgres_db post-test cleanup skipped: {e}")


@pytest.fixture(scope="class")
def clean_postgres_db_class() -> Generator[None, None, None]:
    """
    Class-scoped PostgreSQL cleanup fixture.

    Performs a best-effort TRUNCATE of user tables in the `public` schema once
    before the first test in a class and once after the last test. This reduces
    per-test overhead for integration/benchmark suites which otherwise call the
    function-scoped `clean_postgres_db` for every test method.

    Notes
    -----
    - Uses `EngineManager.get_engine(DATABASE_URL)` to respect the shared pool.
    - Defers constraints to allow order-agnostic TRUNCATE.
    - Gracefully degrades if the database is not available (tests should already
      be gated via markers when Postgres is down).

    """
    engine = EngineManager.get_engine(
        DATABASE_URL,
        pool_size=2,
        max_overflow=3,
        pool_pre_ping=True,
    )

    def _truncate_all() -> None:
        from sqlalchemy import text as _text

        try:
            with engine.connect() as conn:
                conn.execute(_text("SET CONSTRAINTS ALL DEFERRED"))
                result = conn.execute(
                    _text(
                        """
                        SELECT tablename FROM pg_tables
                        WHERE schemaname = 'public'
                          AND tablename NOT LIKE 'pg_%'
                          AND tablename NOT LIKE 'sql_%'
                        """,
                    ),
                )
                for row in result:
                    table = row[0]
                    try:
                        conn.execute(_text(f"TRUNCATE TABLE {table} CASCADE"))
                    except Exception as exc:
                        print(f"Warning during class-scope cleanup of {table}: {exc}")
                conn.commit()
        except Exception as exc:
            print(f"clean_postgres_db_class cleanup skipped: {exc}")

    # Clean before class; disable per-test TRUNCATE while this fixture is active
    import os as _os

    _prev = _os.getenv("TEST_DB_SKIP_TRUNCATE")
    _os.environ["TEST_DB_SKIP_TRUNCATE"] = "1"
    # Clean before class
    _truncate_all()
    yield
    # Clean after class
    _truncate_all()
    # Restore environment
    if _prev is None:
        _os.environ.pop("TEST_DB_SKIP_TRUNCATE", None)
    else:
        _os.environ["TEST_DB_SKIP_TRUNCATE"] = _prev


@pytest.fixture(scope="module")
def clean_postgres_db_module() -> Generator[None, None, None]:
    """
    Module-scoped PostgreSQL cleanup fixture.

    Performs a best-effort TRUNCATE of user tables in the `public` schema once
    before the first test in a module and once after the last test.

    Returns
    -------
    Generator[None, None, None]
        Yields control to the test module between pre/post cleanups.

    """
    engine = EngineManager.get_engine(
        DATABASE_URL,
        pool_size=2,
        max_overflow=3,
        pool_pre_ping=True,
    )

    def _truncate_all() -> None:
        from sqlalchemy import text as _text

        try:
            with engine.connect() as conn:
                conn.execute(_text("SET CONSTRAINTS ALL DEFERRED"))
                result = conn.execute(
                    _text(
                        """
                        SELECT tablename FROM pg_tables
                        WHERE schemaname = 'public'
                          AND tablename NOT LIKE 'pg_%'
                          AND tablename NOT LIKE 'sql_%'
                        """,
                    ),
                )
                for row in result:
                    table = row[0]
                    try:
                        conn.execute(_text(f"TRUNCATE TABLE {table} CASCADE"))
                    except Exception as exc:
                        print(f"Warning during module-scope cleanup of {table}: {exc}")
                conn.commit()
        except Exception as exc:
            print(f"clean_postgres_db_module cleanup skipped: {exc}")

    import os as _os

    _prev = _os.getenv("TEST_DB_SKIP_TRUNCATE")
    _os.environ["TEST_DB_SKIP_TRUNCATE"] = "1"
    _truncate_all()
    yield
    _truncate_all()
    if _prev is None:
        _os.environ.pop("TEST_DB_SKIP_TRUNCATE", None)
    else:
        _os.environ["TEST_DB_SKIP_TRUNCATE"] = _prev


# ============================================================================
# Compatibility database fixtures (legacy names expected by tests)
# ============================================================================


@pytest.fixture(scope="function")
def test_database() -> Generator[TestDatabase, None, None]:
    """
    Create a TestDatabase bound to PostgreSQL with schema initialized.

    Provides a connection string and engine consistent with production usage while
    ensuring each test starts from a clean state and has required tables.

    """
    if not is_postgresql_running():
        pytest.skip(f"PostgreSQL not reachable at {DATABASE_URL}")

    engine = EngineManager.get_engine(
        DATABASE_URL,
        pool_size=2,
        max_overflow=3,
        pool_pre_ping=True,
    )

    # Best-effort clean before creating wrapper, unless class/module cleanup handles it
    import os as _os

    if not _os.getenv("TEST_DB_SKIP_TRUNCATE"):
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
                    # Ignore per-table errors; migrations may not be applied yet
                    import logging as _logging

                    _logging.getLogger(__name__).debug(
                        "TRUNCATE failed for table %s; ignoring in test cleanup",
                        table_name,
                        exc_info=True,
                    )
            conn.commit()

    db = TestDatabase(engine=engine, connection_string=DATABASE_URL, auto_rollback=False)
    # Ensure minimal schema exists for tests expecting migrations
    try:
        db.init_schema()
    except Exception:
        # If migrations fail due to environment, let tests surface specifics
        import logging as _logging

        _logging.getLogger(__name__).debug(
            "Database init_schema failed; continuing",
            exc_info=True,
        )

    try:
        yield db
    finally:
        db.cleanup()


@pytest.fixture
def test_db_engine(test_database: TestDatabase) -> Engine:
    """
    Expose the SQLAlchemy engine from TestDatabase (compat fixture).
    """
    return test_database.engine


@pytest.fixture
def test_db_session(test_database: TestDatabase) -> Generator[Session, None, None]:
    """
    Yield a session with automatic rollback from TestDatabase (compat).
    """
    with test_database.get_session() as session:
        yield session


@pytest.fixture
def seeded_database(test_database: TestDatabase) -> TestDatabase:
    """
    Seed basic data for tests requiring pre-populated state (compat).
    """
    test_database.seed_test_data("basic")
    return test_database


@pytest.fixture
def database_snapshot(test_database: TestDatabase) -> DatabaseSnapshot:
    """
    Provide DatabaseSnapshot helper (compat).
    """
    return DatabaseSnapshot(test_database)


# ============================================================================
# Store Fixtures with Proper Mocking
# ============================================================================


@pytest.fixture
def mock_feature_store() -> MagicMock:
    """
    Create a mock FeatureStore for unit tests.

    This avoids database connections entirely, following the test pyramid principle of
    using mocks for unit tests.

    """
    mock_store = MagicMock()
    mock_store.write_features = MagicMock(return_value=True)
    mock_store.read_features = MagicMock(return_value={})
    mock_store.get_latest_features = MagicMock(return_value={})
    mock_store.compute_features = MagicMock(return_value={"feature_1": 0.5})
    return mock_store


@pytest.fixture
def mock_model_store() -> MagicMock:
    """
    Create a mock ModelStore for unit tests.
    """
    mock_store = MagicMock()
    mock_store.write_predictions = MagicMock(return_value=True)
    mock_store.get_latest_predictions = MagicMock(return_value=[])
    return mock_store


@pytest.fixture
def mock_strategy_store() -> MagicMock:
    """
    Create a mock StrategyStore for unit tests.
    """
    mock_store = MagicMock()
    mock_store.write_signals = MagicMock(return_value=True)
    mock_store.get_active_signals = MagicMock(return_value=[])
    return mock_store


# ============================================================================
# Connection Monitoring Fixtures
# ============================================================================


@pytest.fixture
def connection_monitor(database_engine: Engine):
    """
    Monitor database connections during test execution.

    This helps identify connection leaks and exhaustion issues. Logs warnings if
    connection usage exceeds thresholds.

    """
    import logging

    logger = logging.getLogger(__name__)

    # Get initial connection count
    with database_engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()",
            ),
        )
        initial_count = result.scalar()

    # Monitor during test
    yield

    # Check final connection count
    with database_engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()",
            ),
        )
        final_count = result.scalar()

    # Warn if connections leaked
    if final_count > initial_count + 2:  # Allow small variance
        logger.warning(
            f"Potential connection leak detected: " f"Initial={initial_count}, Final={final_count}",
        )


# ============================================================================
# Hypothesis-specific Fixtures
# ============================================================================


@pytest.fixture
def hypothesis_database_session():
    """
    Special fixture for Hypothesis property tests.

    Uses in-memory SQLite to avoid connection exhaustion
    when Hypothesis generates many test cases.

    Reference: Hypothesis docs on database testing
    https://hypothesis.readthedocs.io/en/latest/database.html

    """
    engine = create_engine(
        "sqlite:///:memory:",
        poolclass=NullPool,  # No pooling needed
    )

    # Tables will be created by stores as needed

    Session = sessionmaker(bind=engine)
    session = Session()

    yield session

    session.close()
    engine.dispose()


# ============================================================================
# Cleanup Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """
    Automatic cleanup after each test.

    This ensures no test leaves behind state that could affect subsequent tests.

    """
    yield

    # Clear any caches (if cache module exists)
    try:
        from ml.core.cache import clear_all_caches

        clear_all_caches()
    except ImportError:
        pass

    # Reset any global state (if config module exists)
    try:
        from ml.config import reset_global_config

        reset_global_config()
    except ImportError:
        pass

    # Garbage collect to free memory
    import gc

    gc.collect()


@pytest.fixture(autouse=False)
def cleanup_engines() -> None:
    """
    Deprecated per-test engine cleanup (use session finish cleanup).

    Left as a no-op to preserve import references in legacy tests.

    """
    return None


@pytest.fixture(autouse=True, scope="session")
def configure_test_logging():
    """
    Configure logging for tests.

    Reduces log noise during test runs while preserving important error information.

    """
    import logging

    # Set appropriate log levels
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("ml").setLevel(logging.INFO)

    # Add test run identifier to logs
    import uuid

    test_run_id = str(uuid.uuid4())[:8]
    logging.basicConfig(
        format=f"[{test_run_id}] %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )


# ============================================================================
# Parallel Test Execution Support
# ============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """
    Configure pytest for optimal parallel execution.

    Uses pytest-xdist for parallel test execution to reduce total test time while
    preventing connection exhaustion.

    """
    # Register known markers to silence PytestUnknownMarkWarning
    config.addinivalue_line("markers", "database: requires PostgreSQL; may run serially")
    config.addinivalue_line("markers", "serial: run test in isolation (no xdist)")
    config.addinivalue_line("markers", "integration: integration test category")

    # Check if xdist is available
    try:
        # Set optimal worker count based on CPU cores
        import multiprocessing

        import xdist

        cpu_count = multiprocessing.cpu_count()

        # Use half the CPUs to avoid overwhelming the database
        optimal_workers = max(1, cpu_count // 2)

        # Only set if not already specified
        if not config.getoption("--numprocesses", default=None):
            config.option.numprocesses = optimal_workers

    except ImportError:
        pass  # xdist not installed


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """
    Apply prototype marks and gate DB tests on collection.

    Parameters
    ----------
    config : pytest.Config
        Pytest configuration object.
    items : list[pytest.Item]
        Collected test items.

    """
    # Mark prototypes first
    _mark_prototypes(items)

    # Gate database-marked tests when PostgreSQL is not reachable.
    if not is_postgresql_running():
        skip_reason = (
            f"PostgreSQL not reachable at {DATABASE_URL}; skipping @pytest.mark.database tests"
        )
        skip_db = pytest.mark.skip(reason=skip_reason)
        for item in items:
            if "database" in item.keywords:
                item.add_marker(skip_db)

    # When xdist is active, group database tests to run on a single worker to prevent
    # cross-worker DDL/DML interference and deadlocks.
    try:
        import xdist

        for item in items:
            if "database" in item.keywords or "serial" in item.keywords:
                try:
                    # Group all DB/serial tests in a single worker named "db"
                    item.add_marker(pytest.mark.xdist_group("db"))  # type: ignore[attr-defined]
                except Exception:
                    # If the marker is unavailable, tests will still run; just without grouping
                    pass
    except Exception:
        # xdist not installed or import failed; no grouping needed
        pass


def pytest_sessionstart(session):
    """
    Set up test database at session start.
    """
    # Ensure test mode permits non-ONNX models where needed
    os.environ.setdefault("ML_TEST_ALLOW_NON_ONNX", "1")

    # Proactively clear pytest cache to avoid stale collection/results
    try:
        from shutil import rmtree

        cache_dir = Path.cwd() / ".pytest_cache"
        if cache_dir.exists():
            rmtree(cache_dir)
    except Exception:
        import logging as _logging

        _logging.getLogger(__name__).debug(
            "Failed to clear pytest cache; continuing",
            exc_info=True,
        )

    # Ensure PostgreSQL is running
    start_postgresql()

    # Initialize test database
    if is_postgresql_running():
        from ml.core.db_engine import EngineManager

        engine = EngineManager.get_engine(DATABASE_URL)

        # Tables will be created by stores as needed
        print("Database initialized, stores will create tables as needed...")
        engine.dispose()

        # Apply known DB fixes for tests (partitions, functions, relaxed constraints)
        try:
            import ml.tests.fix_database_issues as _dbfix

            _dbfix.main()
        except Exception as e:  # Best-effort; tests may gate DB usage
            print(f"Warning: database fixes could not be applied: {e}")

        # Run preflight to validate required functions/partitions exist
        try:
            from ml.stores.db_preflight import check_db_prereqs

            status = check_db_prereqs(DATABASE_URL)
            ok = bool(status.get("ok", False))
            if not ok:
                print(f"Warning: DB preflight failed: {status}")
        except Exception as e:
            print(f"Warning: DB preflight error: {e}")


def pytest_sessionfinish(session, exitstatus):
    """
    Clean up after test session completes.

    Ensures all resources are properly released.

    """
    # Final cleanup of all database connections
    EngineManager.dispose_all()

    # Log session summary
    import logging

    logger = logging.getLogger(__name__)
    logger.info(f"Test session completed with exit status: {exitstatus}")


# ============================================================================
# Legacy Fixtures (kept for backwards compatibility)
# ============================================================================

# Import existing fixtures to maintain compatibility
from ml.tests.fixtures.database_fixtures import *  # noqa: E402, F403 - test re-exports
from ml.tests.fixtures.mock_services import *  # noqa: E402, F403 - test re-exports


# Re-export test utilities
__all__ = [
    "DatabaseSnapshot",
    "TestDatabase",
    "clean_postgres_db",
    "connection_monitor",
    "create_mock_databento_client",
    "create_mock_fred_client",
    "create_mock_postgresql",
    "create_mock_redis",
    "create_mock_yahoo_client",
    "create_test_database",
    "database_engine",
    "database_session",
    "database_session_factory",
    "database_snapshot",
    "hypothesis_database_session",
    "isolated_engine",
    "mock_feature_store",
    "mock_model_store",
    "mock_strategy_store",
    "postgres_connection",
    "seeded_database",
    "temp_database",
    "test_database",
    "test_db_engine",
    "test_db_session",
]


# ============================================================================
# Deployment fixtures used by entrypoint tests
# ============================================================================


@pytest.fixture
def valid_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Provide a minimal valid environment for deployment entrypoint tests.

    Mirrors the per-test fixture to make it available module-wide.

    """
    monkeypatch.setenv("DB_CONNECTION", "postgresql://test:test@localhost:5432/test")
    monkeypatch.setenv("STRATEGY_ID", "MLStrategy-TEST-001")
    monkeypatch.setenv("ML_SIGNAL_SOURCE", "MLSignalActor-001")
    monkeypatch.setenv("INSTRUMENT_ID", "BTC-USDT.DATABENTO")
    monkeypatch.setenv("EXECUTE_TRADES", "false")
    monkeypatch.setenv("POSITION_SIZE_PCT", "0.02")
    monkeypatch.setenv("MIN_CONFIDENCE", "0.6")
    monkeypatch.setenv("MAX_POSITIONS", "3")
    monkeypatch.setenv("STOP_LOSS_PCT", "0.02")
    monkeypatch.setenv("TAKE_PROFIT_PCT", "0.04")
    monkeypatch.setenv("USE_STRATEGY_STORE", "true")
    monkeypatch.setenv("PERSIST_ALL_SIGNALS", "true")
