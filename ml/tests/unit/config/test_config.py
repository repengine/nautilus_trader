#!/usr/bin/env python3
"""
Central test configuration for ML module.

This module provides:
- Database connection configuration for tests
- Mock service configurations
- Environment variable handling
- Test isolation utilities

"""


import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pytest

from ml.tests.utils.db import build_postgres_url


class TestEnvironment(Enum):
    """
    Test environment types.
    """

    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    CI = "ci"


@dataclass(frozen=True)
class DatabaseConfig:
    """
    Database configuration for tests.
    """

    backend: str
    connection_string: str
    use_in_memory: bool
    auto_rollback: bool
    isolation_level: str = "READ_COMMITTED"
    pool_size: int = 1
    echo: bool = False

    @classmethod
    def for_unit_tests(cls) -> "DatabaseConfig":
        """
        Create config for unit tests (DB-free by default).
        """
        if os.environ.get("ML_FORCE_DB_INIT", "0") == "1":
            pg_connection = os.environ.get("DATABASE_URL", build_postgres_url())
            return cls(
                backend="postgresql",
                connection_string=pg_connection,
                use_in_memory=False,
                auto_rollback=True,
                isolation_level="READ_COMMITTED",
                pool_size=1,
                echo=False,
            )

        return cls(
            backend="sqlite",
            connection_string="sqlite:///:memory:",
            use_in_memory=True,
            auto_rollback=True,
            isolation_level="SERIALIZABLE",
            pool_size=1,
            echo=False,
        )

    @classmethod
    def for_integration_tests(cls) -> "DatabaseConfig":
        """
        Create config for integration tests (PostgreSQL only).
        """
        # Always use PostgreSQL - required for ML stores
        pg_connection = os.environ.get("DATABASE_URL", build_postgres_url())
        return cls(
            backend="postgresql",
            connection_string=pg_connection,
            use_in_memory=False,
            auto_rollback=True,
            isolation_level="READ_COMMITTED",
            pool_size=5,
            echo=False,
        )

    @classmethod
    def for_e2e_tests(cls) -> "DatabaseConfig":
        """
        Create config for E2E tests (persistent PostgreSQL database).
        """
        # Always use PostgreSQL for E2E tests
        pg_connection = os.environ.get("DATABASE_URL", build_postgres_url())
        return cls(
            backend="postgresql",
            connection_string=pg_connection,
            use_in_memory=False,
            auto_rollback=False,  # E2E tests manage their own transactions
            isolation_level="READ_COMMITTED",
            pool_size=10,
            echo=False,
        )


@dataclass(frozen=True)
class ExternalServiceConfig:
    """
    Configuration for external service mocks.
    """

    databento_api_key: str = "test_key_123"
    databento_datasets: list[str] | None = None
    databento_base_url: str = "http://mock-databento:8080"
    databento_rate_limit: int = 100  # requests per second

    fred_api_key: str = "test_fred_key"
    fred_base_url: str = "http://mock-fred:8081"
    fred_series: dict[str, str] | None = None

    yahoo_base_url: str = "http://mock-yahoo:8082"
    yahoo_symbols: list[str] | None = None

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 15  # Use DB 15 for tests
    redis_password: str | None = None

    def __post_init__(self) -> None:
        """
        Initialize default values for mutable fields.
        """
        if self.databento_datasets is None:
            object.__setattr__(self, "databento_datasets", ["XNAS.ITCH", "GLBX.MDP3"])
        if self.fred_series is None:
            object.__setattr__(
                self,
                "fred_series",
                {
                    "DGS10": "10-Year Treasury Rate",
                    "DEXUSEU": "USD/EUR Exchange Rate",
                    "VIXCLS": "VIX Volatility Index",
                },
            )
        if self.yahoo_symbols is None:
            object.__setattr__(self, "yahoo_symbols", ["SPY", "QQQ", "IWM", "AAPL", "MSFT"])


@dataclass(frozen=True)
class MockDataConfig:
    """
    Configuration for mock data generation.
    """

    n_bars: int = 1000
    n_instruments: int = 3
    start_date: str = "2024-01-01"
    end_date: str = "2024-01-31"
    bar_interval_minutes: int = 1

    # Market data characteristics
    base_prices: dict[str, float] | None = None
    volatility: float = 0.0002
    drift: float = 0.00001
    correlation_matrix: list[list[float]] | None = None

    # L2 data configuration
    n_levels: int = 5
    tick_size: float = 0.00001
    lot_size: float = 1000

    def __post_init__(self) -> None:
        """
        Initialize default values.
        """
        if self.base_prices is None:
            object.__setattr__(
                self,
                "base_prices",
                {
                    "EURUSD": 1.0900,
                    "GBPUSD": 1.2700,
                    "USDJPY": 148.50,
                },
            )
        if self.correlation_matrix is None:
            object.__setattr__(
                self,
                "correlation_matrix",
                [
                    [1.0, 0.7, -0.3],  # EURUSD
                    [0.7, 1.0, -0.2],  # GBPUSD
                    [-0.3, -0.2, 1.0],  # USDJPY
                ],
            )


class TestConfig:
    """
    Central test configuration manager.
    """

    def __init__(self, environment: TestEnvironment | None = None):
        """
        Initialize test configuration.

        Parameters
        ----------
        environment : TestEnvironment, optional
            Test environment type (auto-detected if not provided)

        """
        self.environment = environment or self._detect_environment()
        self.database = self._get_database_config()
        self.external_services = self._get_external_services_config()
        self.mock_data = self._get_mock_data_config()

        # Test execution settings
        self.timeout_seconds = self._get_timeout()
        self.retry_attempts = 3 if self.environment == TestEnvironment.CI else 1
        self.parallel_workers = self._get_parallel_workers()

        # Feature flags
        self.use_real_databento = os.environ.get("ML_USE_REAL_DATABENTO", "false").lower() == "true"
        self.use_real_fred = os.environ.get("ML_USE_REAL_FRED", "false").lower() == "true"
        self.enable_slow_tests = os.environ.get("ML_ENABLE_SLOW_TESTS", "false").lower() == "true"
        self.enable_ml_deps_tests = (
            os.environ.get("ML_ENABLE_ML_DEPS_TESTS", "true").lower() == "true"
        )

    def _detect_environment(self) -> TestEnvironment:
        """
        Auto-detect test environment.
        """
        # Check CI environment variables
        if any(os.environ.get(var) for var in ["CI", "GITHUB_ACTIONS", "JENKINS", "GITLAB_CI"]):
            return TestEnvironment.CI

        # Check pytest markers (would be set by test runner)
        pytest_current = os.environ.get("PYTEST_CURRENT_TEST", "")
        if "integration" in pytest_current:
            return TestEnvironment.INTEGRATION
        elif "e2e" in pytest_current:
            return TestEnvironment.E2E

        # Default to unit tests
        return TestEnvironment.UNIT

    def _get_database_config(self) -> DatabaseConfig:
        """
        Get database config for current environment.
        """
        if self.environment == TestEnvironment.UNIT:
            return DatabaseConfig.for_unit_tests()
        elif self.environment == TestEnvironment.INTEGRATION:
            return DatabaseConfig.for_integration_tests()
        elif self.environment == TestEnvironment.E2E:
            return DatabaseConfig.for_e2e_tests()
        elif self.environment == TestEnvironment.CI:
            # CI uses integration config by default
            return DatabaseConfig.for_integration_tests()
        else:
            return DatabaseConfig.for_unit_tests()

    def _get_external_services_config(self) -> ExternalServiceConfig:
        """
        Get external services config, respecting environment overrides.
        """
        return ExternalServiceConfig(
            databento_api_key=os.environ.get("DATABENTO_API_KEY", "test_key_123"),
            fred_api_key=os.environ.get("FRED_API_KEY", "test_fred_key"),
            redis_host=os.environ.get("REDIS_HOST", "localhost"),
            redis_port=int(os.environ.get("REDIS_PORT", "6379")),
            redis_db=int(os.environ.get("REDIS_TEST_DB", "15")),
        )

    def _get_mock_data_config(self) -> MockDataConfig:
        """
        Get mock data generation config.
        """
        return MockDataConfig(
            n_bars=int(os.environ.get("ML_TEST_N_BARS", "1000")),
            n_instruments=int(os.environ.get("ML_TEST_N_INSTRUMENTS", "3")),
        )

    def _get_timeout(self) -> int:
        """
        Get test timeout based on environment.
        """
        timeouts = {
            TestEnvironment.UNIT: 10,
            TestEnvironment.INTEGRATION: 60,
            TestEnvironment.E2E: 300,
            TestEnvironment.CI: 120,
        }
        default = timeouts.get(self.environment, 30)
        return int(os.environ.get("ML_TEST_TIMEOUT", str(default)))

    def _get_parallel_workers(self) -> int:
        """
        Get number of parallel workers for test execution.
        """
        if self.environment == TestEnvironment.CI:
            # Use fewer workers in CI to avoid resource contention
            return min(2, os.cpu_count() or 1)

        # For local testing, use more workers
        return int(os.environ.get("ML_TEST_WORKERS", str((os.cpu_count() or 1) // 2)))

    def get_temp_dir(self) -> Path:
        """
        Get temporary directory for test artifacts.
        """
        base_temp = Path(tempfile.gettempdir())
        test_dir = base_temp / "nautilus_ml_tests" / self.environment.value
        test_dir.mkdir(parents=True, exist_ok=True)
        return test_dir

    def get_schema_files(self) -> list[Path]:
        """
        Get SQL schema files for database initialization.
        """
        migrations_dir = Path(__file__).parent.parent.parent / "stores" / "migrations"

        # Return files in order
        schema_files = [
            migrations_dir / "002_stores_schema.sql",
            migrations_dir / "003_auto_partitioning.sql",
            migrations_dir / "004_market_data.sql",
            migrations_dir / "005_data_registry.sql",
            migrations_dir / "007_schema_hardening.sql",
            migrations_dir / "008_views.sql",
            migrations_dir / "006_feature_values_dedupe.sql",
            migrations_dir / "015_macro_release_calendar.sql",
            migrations_dir / "016_macro_observations.sql",
            migrations_dir / "017_events_calendar.sql",
            migrations_dir / "018_microstructure_minute.sql",
            migrations_dir / "019_l2_minute.sql",
        ]

        # Filter to only existing files
        return [f for f in schema_files if f.exists()]

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionary.
        """
        return {
            "environment": self.environment.value,
            "database": {
                "backend": self.database.backend,
                "connection_string": self.database.connection_string,
                "use_in_memory": self.database.use_in_memory,
                "auto_rollback": self.database.auto_rollback,
            },
            "external_services": {
                "databento_api_key": self.external_services.databento_api_key,
                "fred_api_key": self.external_services.fred_api_key,
                "redis_host": self.external_services.redis_host,
                "redis_port": self.external_services.redis_port,
            },
            "mock_data": {
                "n_bars": self.mock_data.n_bars,
                "n_instruments": self.mock_data.n_instruments,
            },
            "execution": {
                "timeout_seconds": self.timeout_seconds,
                "retry_attempts": self.retry_attempts,
                "parallel_workers": self.parallel_workers,
            },
            "feature_flags": {
                "use_real_databento": self.use_real_databento,
                "use_real_fred": self.use_real_fred,
                "enable_slow_tests": self.enable_slow_tests,
                "enable_ml_deps_tests": self.enable_ml_deps_tests,
            },
        }


# Global instance for easy access
_test_config: TestConfig | None = None


def get_test_config() -> TestConfig:
    """
    Get or create global test configuration.
    """
    global _test_config
    if _test_config is None:
        _test_config = TestConfig()
    return _test_config


def reset_test_config() -> None:
    """
    Reset global test configuration (useful for testing).
    """
    global _test_config
    _test_config = None
