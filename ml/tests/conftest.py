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
    "postgresql://postgres:postgres@localhost:5432/nautilus_test",
)

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
    # Default profile for local development
    settings.load_profile("dev")

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

    This factory creates sessions that share the same connection pool,
    preventing connection exhaustion.
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

    This is perfect for tests that don't need PostgreSQL-specific features
    and provides complete isolation with zero connection overhead.
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
    """Get PostgreSQL connection string from environment."""
    return DATABASE_URL


# ============================================================================
# PostgreSQL Management Fixtures
# ============================================================================

def is_postgresql_running() -> bool:
    """Check if PostgreSQL is running and accessible."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.close()
        return True
    except psycopg2.OperationalError:
        return False


def start_postgresql() -> None:
    """Attempt to start PostgreSQL if not running."""
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
# Store Fixtures with Proper Mocking
# ============================================================================

@pytest.fixture
def mock_feature_store() -> MagicMock:
    """
    Create a mock FeatureStore for unit tests.

    This avoids database connections entirely, following the
    test pyramid principle of using mocks for unit tests.
    """
    mock_store = MagicMock()
    mock_store.write_features = MagicMock(return_value=True)
    mock_store.read_features = MagicMock(return_value={})
    mock_store.get_latest_features = MagicMock(return_value={})
    mock_store.compute_features = MagicMock(return_value={"feature_1": 0.5})
    return mock_store


@pytest.fixture
def mock_model_store() -> MagicMock:
    """Create a mock ModelStore for unit tests."""
    mock_store = MagicMock()
    mock_store.write_predictions = MagicMock(return_value=True)
    mock_store.get_latest_predictions = MagicMock(return_value=[])
    return mock_store


@pytest.fixture
def mock_strategy_store() -> MagicMock:
    """Create a mock StrategyStore for unit tests."""
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

    This helps identify connection leaks and exhaustion issues.
    Logs warnings if connection usage exceeds thresholds.
    """
    import logging
    logger = logging.getLogger(__name__)

    # Get initial connection count
    with database_engine.connect() as conn:
        result = conn.execute(text(
            "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()"
        ))
        initial_count = result.scalar()

    # Monitor during test
    yield

    # Check final connection count
    with database_engine.connect() as conn:
        result = conn.execute(text(
            "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()"
        ))
        final_count = result.scalar()

    # Warn if connections leaked
    if final_count > initial_count + 2:  # Allow small variance
        logger.warning(
            f"Potential connection leak detected: "
            f"Initial={initial_count}, Final={final_count}"
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

    This ensures no test leaves behind state that could
    affect subsequent tests.
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


@pytest.fixture(autouse=True)
def cleanup_engines():
    """Clean up database engines after each test to prevent leaks."""
    yield
    EngineManager.dispose_all()


@pytest.fixture(autouse=True, scope="session")
def configure_test_logging():
    """
    Configure logging for tests.

    Reduces log noise during test runs while preserving
    important error information.
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
        level=logging.INFO
    )


# ============================================================================
# Parallel Test Execution Support
# ============================================================================

def pytest_configure(config):
    """
    Configure pytest for optimal parallel execution.

    Uses pytest-xdist for parallel test execution to reduce
    total test time while preventing connection exhaustion.
    """
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


def pytest_sessionstart(session):
    """Set up test database at session start."""
    # Ensure PostgreSQL is running
    start_postgresql()

    # Initialize test database
    if is_postgresql_running():
        from ml.core.db_engine import EngineManager
        engine = EngineManager.get_engine(DATABASE_URL)

        # Tables will be created by stores as needed
        print("Database initialized, stores will create tables as needed...")
        engine.dispose()


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
from ml.tests.fixtures.database_fixtures import *
from ml.tests.fixtures.mock_services import *


# Re-export test utilities
__all__ = [
    "DatabaseSnapshot",
    # Legacy
    "TestDatabase",
    # Monitoring
    "connection_monitor",
    "create_mock_databento_client",
    "create_mock_fred_client",
    "create_mock_postgresql",
    "create_mock_redis",
    "create_mock_yahoo_client",
    "create_test_database",
    # Database
    "database_engine",
    "database_session",
    "database_session_factory",
    "hypothesis_database_session",
    "isolated_engine",
    # Mocks
    "mock_feature_store",
    "mock_model_store",
    "mock_strategy_store",
    "postgres_connection",
    "temp_database",
]
