#!/usr/bin/env python3

"""Common pytest fixtures and configuration for ML module tests."""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path


# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass  # dotenv not installed, use system environment
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock
from unittest.mock import patch

import psycopg2
import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker

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

# PostgreSQL configuration
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_DB = os.getenv("POSTGRES_DB", "nautilus_test")
DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"


def pytest_configure(config):
    """Configure pytest for optimal parallel execution."""
    try:
        import multiprocessing

        import xdist

        # Use conservative parallelism to avoid overwhelming DB
        cpu_count = multiprocessing.cpu_count()
        optimal_workers = min(4, max(1, cpu_count // 2))

        if not config.getoption("--numprocesses", default=None):
            config.option.numprocesses = optimal_workers
    except ImportError:
        pass  # xdist not installed


def pytest_sessionstart(session):
    """Start PostgreSQL before any tests run."""
    # Check if PostgreSQL is already running
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            database="postgres",
            connect_timeout=3
        )
        conn.close()
        print("PostgreSQL is already running")
    except (psycopg2.OperationalError, psycopg2.Error):
        # Start PostgreSQL using docker-compose
        print("Starting PostgreSQL with docker-compose...")
        ml_dir = Path(__file__).parent.parent  # ml/ directory
        result = subprocess.run(
            ["docker-compose", "up", "-d", "postgres"],
            cwd=ml_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            pytest.exit(f"Failed to start PostgreSQL: {result.stderr}")

        # Wait for PostgreSQL to be ready
        print("Waiting for PostgreSQL to be ready...")
        max_retries = 30
        for i in range(max_retries):
            try:
                conn = psycopg2.connect(
                    host=POSTGRES_HOST,
                    port=POSTGRES_PORT,
                    user=POSTGRES_USER,
                    password=POSTGRES_PASSWORD,
                    database="postgres",
                    connect_timeout=3
                )
                conn.close()
                print("PostgreSQL is ready")
                break
            except (psycopg2.OperationalError, psycopg2.Error):
                if i == max_retries - 1:
                    pytest.exit("PostgreSQL failed to start after 30 seconds")
                time.sleep(1)

    # Create test database if it doesn't exist
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            database="postgres"
        )
        conn.set_isolation_level(0)  # Set autocommit mode
        cursor = conn.cursor()

        # Check if test database exists
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (POSTGRES_DB,)
        )
        if not cursor.fetchone():
            print(f"Creating test database '{POSTGRES_DB}'...")
            cursor.execute(f"CREATE DATABASE {POSTGRES_DB}")

        cursor.close()
        conn.close()

        # Apply migrations to test database
        from ml.core.db_engine import EngineManager
        engine = EngineManager.get_engine(
            DATABASE_URL,
            pool_size=2,  # Conservative for tests
            max_overflow=3,  # Conservative for tests
        )

        # Check if migrations are already applied
        if os.getenv("SKIP_MIGRATIONS"):
            print("Skipping migrations (SKIP_MIGRATIONS=1)")
            return

        with engine.connect() as conn:
            # Check if core tables exist
            result = conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema = 'public' "
                "AND table_name IN ('ml_feature_values', 'ml_model_predictions', 'ml_strategy_signals')"
            ))
            table_count = result.scalar()
            if table_count >= 3:
                print(f"Core tables exist ({table_count} found), skipping migrations...")
                return

            ml_dir = Path(__file__).parent.parent
            migration_files = sorted(
                (ml_dir / "stores" / "migrations").glob("*.sql")
            )

            for migration_file in migration_files:
                print(f"Applying migration: {migration_file.name}")
                with open(migration_file) as f:
                    migration_sql = f.read()
                    # Execute the entire migration file as one block
                    # PostgreSQL functions with $$ delimiters need to be executed as a whole
                    try:
                        conn.execute(text(migration_sql))
                        conn.commit()
                    except Exception as e:
                        print(f"Warning: Migration {migration_file.name} failed: {e}")
                        conn.rollback()
                        # Try simpler approach - just create tables without functions
                        if "001_stores_schema.sql" in str(migration_file):
                            # Create basic tables without partitioning
                            print("Falling back to basic table creation...")
                            basic_tables = """
                            CREATE TABLE IF NOT EXISTS ml_feature_values (
                                id BIGSERIAL PRIMARY KEY,
                                feature_set_id VARCHAR(255) NOT NULL,
                                instrument_id VARCHAR(100) NOT NULL,
                                ts_event BIGINT NOT NULL,
                                ts_init BIGINT NOT NULL,
                                values JSONB NOT NULL,
                                is_live BOOLEAN DEFAULT FALSE,
                                source VARCHAR(50),
                                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                            );

                            CREATE TABLE IF NOT EXISTS ml_model_predictions (
                                id BIGSERIAL PRIMARY KEY,
                                model_id VARCHAR(255) NOT NULL,
                                instrument_id VARCHAR(100) NOT NULL,
                                ts_event BIGINT NOT NULL,
                                ts_init BIGINT NOT NULL,
                                prediction FLOAT NOT NULL,
                                confidence FLOAT,
                                features_used JSONB,
                                inference_time_ms FLOAT,
                                is_live BOOLEAN DEFAULT FALSE,
                                created_at BIGINT
                            );

                            CREATE TABLE IF NOT EXISTS ml_strategy_signals (
                                id BIGSERIAL PRIMARY KEY,
                                strategy_id VARCHAR(255) NOT NULL,
                                instrument_id VARCHAR(100) NOT NULL,
                                ts_event BIGINT NOT NULL,
                                ts_init BIGINT NOT NULL,
                                signal_type VARCHAR(20) NOT NULL,
                                strength FLOAT NOT NULL,
                                model_predictions JSONB,
                                risk_metrics JSONB,
                                execution_params JSONB,
                                is_live BOOLEAN DEFAULT FALSE,
                                created_at BIGINT
                            );
                            """
                            try:
                                conn.execute(text(basic_tables))
                                conn.commit()
                                print("Created basic tables without partitioning")
                            except Exception as e2:
                                print(f"Failed to create basic tables: {e2}")
                                conn.rollback()

        # Create test partitions after migrations
        print("Creating test partitions...")
        try:
            from ml.stores.partition_manager import PartitionManager

            # Create partition manager
            partition_manager = PartitionManager(
                connection_string=DATABASE_URL,
                tables=["ml_feature_values", "ml_model_predictions", "ml_strategy_signals"],
                months_ahead=6,  # Create extra partitions for future
            )

            # Create partitions for test data (2023-2026)
            # The common test timestamp 1700000000000000000 is in November 2023
            created = partition_manager.create_test_partitions(
                start_year=2023,
                start_month=1,
                end_year=2026,
                end_month=12
            )
            print(f"Created {created} test partitions")

            # Also ensure current month partitions exist
            for table in partition_manager.tables:
                partition_manager.ensure_current_partition(table)

        except Exception as e:
            print(f"Warning: Failed to create test partitions: {e}")
            # Don't fail tests if partition creation fails, as they may already exist

        print("Test database ready")

    except Exception as e:
        print(f"Failed to setup test database: {e}")
        pytest.exit(f"Database setup failed: {e}")


# Auto-marking configuration based on directory structure
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-apply markers based on test file location."""
    for item in items:
        # Get the relative path from ml/tests/
        test_path = Path(item.fspath)
        tests_parent = Path(__file__).parent

        # Check if the test is within the tests directory
        try:
            relative_parts = test_path.relative_to(tests_parent).parts
        except ValueError:
            # Test is outside tests directory, skip auto-marking
            continue

        if not relative_parts:
            continue

        # Apply test type markers based on directory
        test_type = relative_parts[0] if relative_parts else ""

        # Test type markers
        if test_type == "unit":
            item.add_marker(pytest.mark.unit)
        elif test_type == "integration":
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.requires_data)
        elif test_type == "e2e":
            item.add_marker(pytest.mark.e2e)
            item.add_marker(pytest.mark.requires_data)
            item.add_marker(pytest.mark.slow)
        elif test_type == "system":
            item.add_marker(pytest.mark.system)
            item.add_marker(pytest.mark.requires_data)
        elif test_type == "property":
            item.add_marker(pytest.mark.property)
        elif test_type == "contracts":
            item.add_marker(pytest.mark.contract)
        elif test_type == "benchmarks" or test_type == "performance":
            item.add_marker(pytest.mark.benchmark)
            item.add_marker(pytest.mark.slow)

        # Component markers based on subdirectory
        if len(relative_parts) > 1:
            component = relative_parts[1] if test_type in ["unit", "integration"] else relative_parts[0]

            component_markers = {
                "stores": pytest.mark.stores,
                "registry": pytest.mark.registry,
                "actors": pytest.mark.actors,
                "strategies": pytest.mark.strategies,
                "features": pytest.mark.features,
                "models": pytest.mark.models,
                "data": pytest.mark.data,
                "monitoring": pytest.mark.monitoring,
                "deployment": pytest.mark.deployment,
                "training": pytest.mark.training,
                "preprocessing": pytest.mark.preprocessing,
            }

            if component in component_markers:
                item.add_marker(component_markers[component])

        # Apply markers based on test name patterns
        test_name = item.name.lower()

        if "hypothesis" in test_name or "property" in test_name:
            item.add_marker(pytest.mark.property)

        if "slow" in test_name or "benchmark" in test_name:
            item.add_marker(pytest.mark.slow)

        if "databento" in test_name:
            item.add_marker(pytest.mark.requires_databento)

        if "fred" in test_name:
            item.add_marker(pytest.mark.requires_fred)

        if any(ml_dep in test_name for ml_dep in ["xgboost", "lightgbm", "torch", "tensorflow"]):
            item.add_marker(pytest.mark.requires_ml_deps)

        if "training" in test_name or "train" in test_name:
            item.add_marker(pytest.mark.training)

        if "inference" in test_name or "predict" in test_name:
            item.add_marker(pytest.mark.inference)

        if "distill" in test_name:
            item.add_marker(pytest.mark.distillation)


# Common Fixtures
@pytest.fixture
def test_data_dir() -> Path:
    """Provide path to test data directory."""
    return get_test_data_dir()


@pytest.fixture
def model_registry_dir() -> Path:
    """Provide path to test model registry."""
    return get_model_registry_dir()


@pytest.fixture
def model_registry_rollout_dir() -> Path:
    """Provide path to test model registry for rollout testing."""
    return get_model_registry_rollout_dir()


@pytest.fixture
def xgb_v1_model_path(model_registry_dir: Path) -> Path:
    """Provide path to XGBoost v1 test model."""
    return model_registry_dir / "models" / "xgb_v1.json"


@pytest.fixture
def xgb_v2_model_path(model_registry_dir: Path) -> Path:
    """Provide path to XGBoost v2 test model."""
    return model_registry_dir / "models" / "xgb_v2.json"


@pytest.fixture
def prod_onnx_model_path(model_registry_rollout_dir: Path) -> Path:
    """Provide path to production ONNX test model."""
    return model_registry_rollout_dir / "models" / "prod.onnx"


@pytest.fixture
def new_onnx_model_path(model_registry_rollout_dir: Path) -> Path:
    """Provide path to new ONNX test model."""
    return model_registry_rollout_dir / "models" / "new.onnx"


# Test configuration fixtures
@pytest.fixture(scope="session")
def test_config() -> TestConfig:
    """Get test configuration for current environment."""
    return get_test_config()


# Database fixtures for testing stores
@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create a temporary database path."""
    return tmp_path / "test.db"


@pytest.fixture(scope="function")
def test_database(test_config: TestConfig) -> Generator[TestDatabase, None, None]:
    """
    Create test database with automatic cleanup.

    Always uses PostgreSQL for consistency with production.
    Each test gets a clean database state.
    """
    # Import EngineManager locally to avoid circular dependencies
    from ml.core.db_engine import EngineManager

    # Always use PostgreSQL
    connection_string = DATABASE_URL

    # Get engine from EngineManager (creates if not exists)
    engine = EngineManager.get_engine(
        connection_string,
        pool_size=2,  # Conservative for tests
        max_overflow=3,  # Conservative for tests
        pool_pre_ping=True,
        echo=test_config.database.echo
    )

    # Clean all test data before each test
    with engine.connect() as conn:
        # Truncate all tables except schema management
        result = conn.execute(text("""
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public'
            AND tablename NOT LIKE 'pg_%'
            AND tablename NOT LIKE 'sql_%'
        """))

        for row in result:
            table_name = row[0]
            try:
                conn.execute(text(f"TRUNCATE TABLE {table_name} CASCADE"))
            except Exception as e:
                print(f"Warning: Could not truncate {table_name}: {e}")

    # Create test database wrapper
    db = TestDatabase(
        engine=engine,
        connection_string=connection_string,
        auto_rollback=False  # Use transactions for each operation
    )

    try:
        yield db
    finally:
        # Clean up after test (engine disposal handled by cleanup_engines fixture)
        db.cleanup()


@pytest.fixture
def test_db_engine(test_database: TestDatabase):
    """Create a test database engine (compatibility fixture)."""
    return test_database.engine


@pytest.fixture
def test_db_session(test_database: TestDatabase) -> Generator[Session, None, None]:
    """Create a test database session with automatic rollback."""
    with test_database.get_session() as session:
        yield session


@pytest.fixture
def seeded_database(test_database: TestDatabase) -> TestDatabase:
    """Create test database with seed data."""
    test_database.seed_test_data("basic")
    return test_database


@pytest.fixture
def database_snapshot(test_database: TestDatabase) -> DatabaseSnapshot:
    """Create database snapshot utility for state management."""
    return DatabaseSnapshot(test_database)


# Mock fixtures for external dependencies
@pytest.fixture
def mock_databento_client(test_config: TestConfig) -> Any:
    """Mock Databento client for testing."""
    if test_config.use_real_databento:
        # Return real client if configured
        pytest.skip("Real Databento client not implemented in test mode")

    return create_mock_databento_client(
        api_key=test_config.external_services.databento_api_key,
        fail_on_request=False,
    )


@pytest.fixture
def mock_fred_client(test_config: TestConfig) -> Any:
    """Mock FRED API client for testing."""
    if test_config.use_real_fred:
        # Return real client if configured
        pytest.skip("Real FRED client not implemented in test mode")

    return create_mock_fred_client(
        api_key=test_config.external_services.fred_api_key,
    )


@pytest.fixture
def mock_yahoo_client() -> Any:
    """Mock Yahoo Finance client for testing."""
    return create_mock_yahoo_client()


@pytest.fixture
def mock_redis(test_config: TestConfig) -> Any:
    """Mock Redis client for testing."""
    # For unit tests, always use mock
    if test_config.environment == TestEnvironment.UNIT:
        return create_mock_redis()

    # For integration/E2E, try real Redis first
    try:
        import redis
        client = redis.Redis(
            host=test_config.external_services.redis_host,
            port=test_config.external_services.redis_port,
            db=test_config.external_services.redis_db,
            password=test_config.external_services.redis_password,
            decode_responses=True,
        )
        client.ping()
        # Clean test database
        client.flushdb()
        return client
    except Exception:
        # Fall back to mock if Redis not available
        return create_mock_redis()


@pytest.fixture
def postgres_connection() -> str:
    """Get PostgreSQL connection string for tests."""
    return DATABASE_URL


@pytest.fixture(scope="function")
def clean_postgres_db():
    """Ensure clean PostgreSQL database state for each test."""
    from ml.core.db_engine import EngineManager

    # Get engine from EngineManager (reuses existing or creates new)
    engine = EngineManager.get_engine(
        DATABASE_URL,
        pool_size=2,  # Conservative for tests
        max_overflow=3,  # Conservative for tests
    )

    # Clean before test
    with engine.connect() as conn:
        conn.execute(text("SET CONSTRAINTS ALL DEFERRED"))

        # Get all tables
        result = conn.execute(text("""
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public'
            AND tablename NOT LIKE 'pg_%'
            AND tablename NOT LIKE 'sql_%'
        """))

        # Truncate all data tables
        for row in result:
            table_name = row[0]
            try:
                conn.execute(text(f"TRUNCATE TABLE {table_name} CASCADE"))
                conn.commit()
            except Exception as e:
                print(f"Warning during cleanup: {e}")
                conn.rollback()

    yield

    # Clean after test as well
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public'
            AND tablename NOT LIKE 'pg_%'
            AND tablename NOT LIKE 'sql_%'
        """))

        for row in result:
            table_name = row[0]
            try:
                conn.execute(text(f"TRUNCATE TABLE {table_name} CASCADE"))
                conn.commit()
            except Exception:
                conn.rollback()

    # Note: engine disposal is handled by cleanup_engines fixture


@pytest.fixture
def mock_mlflow_client() -> MagicMock:
    """Mock MLflow client for testing."""
    mock = MagicMock()
    mock.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
    mock.create_run.return_value = MagicMock(info=MagicMock(run_id="test_run_id"))
    return mock


# Patching fixtures for automatic mocking
@pytest.fixture(autouse=True)
def auto_mock_external_services(request, monkeypatch, test_config: TestConfig):
    """Automatically mock external services based on test configuration."""
    # Skip for tests that explicitly want real services
    if "no_mock" in request.keywords:
        return

    # Mock Databento if not using real API
    if not test_config.use_real_databento:
        mock_client = create_mock_databento_client()
        monkeypatch.setattr("databento.DBNStore", lambda *args, **kwargs: mock_client)
        monkeypatch.setattr("databento.Historical", lambda *args, **kwargs: mock_client)

    # Mock FRED if not using real API
    if not test_config.use_real_fred:
        mock_client = create_mock_fred_client()
        try:
            import fredapi
            monkeypatch.setattr("fredapi.Fred", lambda api_key: mock_client)
        except (ImportError, AttributeError):
            pass  # fredapi not installed

    # Always mock expensive ML operations in unit tests
    if test_config.environment == TestEnvironment.UNIT:
        # Mock heavy ML libraries to speed up imports
        monkeypatch.setenv("ML_DISABLE_GPU", "true")
        monkeypatch.setenv("TF_CPP_MIN_LOG_LEVEL", "3")  # Suppress TensorFlow logs


# Environment fixtures
@pytest.fixture(autouse=True)
def set_test_environment(monkeypatch) -> None:
    """Set environment variables for testing."""
    monkeypatch.setenv("ML_ENV", "test")
    monkeypatch.setenv("ML_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("ML_DISABLE_METRICS", "true")
    # Always use PostgreSQL for tests
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    monkeypatch.setenv("ML_DATABASE_URL", DATABASE_URL)
    # Ensure registry code paths using env fallbacks pick up the same DB
    monkeypatch.setenv("NAUTILUS_REGISTRY_DB_URL", DATABASE_URL)


@pytest.fixture
def clean_environment(monkeypatch) -> None:
    """Clean environment for isolated testing."""
    # Remove any ML-related environment variables
    for key in list(os.environ.keys()):
        if key.startswith(("ML_", "NAUTILUS_")):
            monkeypatch.delenv(key, raising=False)


# Performance testing fixtures
@pytest.fixture
def benchmark_timer():
    """Simple timer for benchmark tests."""
    import time

    class Timer:
        def __init__(self):
            self.times = []

        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.end = time.perf_counter()
            self.times.append(self.end - self.start)

        @property
        def elapsed(self):
            return self.times[-1] if self.times else 0

        @property
        def mean(self):
            return sum(self.times) / len(self.times) if self.times else 0

    return Timer()


# Temporary directory fixtures with cleanup
@pytest.fixture
def temp_model_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for model storage."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    return model_dir


@pytest.fixture
def temp_feature_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for feature storage."""
    feature_dir = tmp_path / "features"
    feature_dir.mkdir(exist_ok=True)
    return feature_dir


# Store initialization fixtures
@pytest.fixture
def feature_store_connection(test_database: TestDatabase) -> str:
    """Get connection string for FeatureStore."""
    return test_database.connection_string


@pytest.fixture
def model_store_connection(test_database: TestDatabase) -> str:
    """Get connection string for ModelStore."""
    return test_database.connection_string


@pytest.fixture
def strategy_store_connection(test_database: TestDatabase) -> str:
    """Get connection string for StrategyStore."""
    return test_database.connection_string


# Test isolation helpers
@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    # Reset test config singleton
    from ml.tests.unit.config.test_config import reset_test_config
    reset_test_config()

    # Reset any other singletons here
    yield

    # Cleanup after test
    reset_test_config()


@pytest.fixture
def capture_metrics():
    """Capture Prometheus metrics for testing."""
    metrics = {}

    def capture(metric_name: str, value: float, labels: dict[str, str] | None = None):
        key = (metric_name, tuple(labels.items()) if labels else ())
        if key not in metrics:
            metrics[key] = []
        metrics[key].append(value)

    # Return both the capture function and the metrics dict
    return capture, metrics


# Session-scoped engine to prevent connection exhaustion
@pytest.fixture(scope="session")
def session_engine():
    """Single engine for entire test session to prevent connection exhaustion."""
    from ml.core.db_engine import EngineManager

    engine = EngineManager.get_engine(
        DATABASE_URL,
        pool_size=2,  # Conservative for tests
        max_overflow=3,  # Limited overflow
        pool_pre_ping=True,  # Test connections
    )
    yield engine
    EngineManager.dispose_all()


@pytest.fixture(autouse=True)
def cleanup_engines():
    """
    Clean up all database engines after each test.

    This fixture automatically runs after each test to dispose all cached
    database engines from the EngineManager, preventing connection leaks
    and "too many clients" errors in PostgreSQL.

    This is particularly important for:
    - Hypothesis property-based tests that create many instances
    - Integration tests that create multiple stores
    - Tests that don't properly clean up their database connections
    """
    yield
    # Clean up all engines after the test completes
    from ml.core.db_engine import EngineManager
    EngineManager.dispose_all()


@pytest.fixture(scope="function")
def connection_monitor():
    """
    Monitor database connections during test execution.

    This fixture tracks the number of database engines created and disposed
    during a test, helping to identify connection leaks.

    Returns
    -------
    dict
        Dictionary with 'initial_count', 'peak_count', and 'final_count' keys
    """
    from ml.core.db_engine import EngineManager

    monitor = {
        "initial_count": EngineManager.get_engine_count(),
        "peak_count": 0,
        "final_count": 0,
    }

    def update_peak():
        current = EngineManager.get_engine_count()
        if current > monitor["peak_count"]:
            monitor["peak_count"] = current

    monitor["update_peak"] = update_peak

    yield monitor

    monitor["final_count"] = EngineManager.get_engine_count()

    # Log if there's a potential leak
    if monitor["final_count"] > monitor["initial_count"]:
        import warnings
        warnings.warn(
            f"Potential connection leak detected: "
            f"Started with {monitor['initial_count']} engines, "
            f"ended with {monitor['final_count']} engines"
        )
