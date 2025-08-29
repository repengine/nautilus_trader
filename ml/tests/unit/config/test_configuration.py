#!/usr/bin/env python3
"""
Test the test configuration infrastructure itself.

This ensures our test configuration, mocks, and database fixtures work correctly.
"""

import pytest
from sqlalchemy import text

from ml.tests.fixtures.database_fixtures import TestDatabase
from ml.tests.fixtures.mock_services import MockDatabentoClient
from ml.tests.fixtures.mock_services import MockFredClient
from ml.tests.fixtures.mock_services import MockRedis
from ml.tests.fixtures.mock_services import MockYahooClient
from ml.tests.test_config import TestConfig
from ml.tests.test_config import TestEnvironment


@pytest.mark.database
@pytest.mark.serial
@pytest.mark.redis
@pytest.mark.parallel_safe
@pytest.mark.unit
class TestTestConfiguration:
    """Test the test configuration system."""

    @pytest.mark.database
    @pytest.mark.serial
    def test_config_initialization(self, test_config: TestConfig):
        """Test that test configuration initializes correctly."""
        assert test_config is not None
        assert test_config.environment in TestEnvironment
        assert test_config.database is not None
        assert test_config.external_services is not None
        assert test_config.mock_data is not None

    @pytest.mark.database
    @pytest.mark.serial
    def test_database_config_by_environment(self):
        """Test database configuration for different environments."""
        # Unit test config
        unit_config = TestConfig(TestEnvironment.UNIT)
        assert unit_config.database.use_in_memory is True
        assert unit_config.database.auto_rollback is True
        assert "memory" in unit_config.database.connection_string

        # Integration test config
        integration_config = TestConfig(TestEnvironment.INTEGRATION)
        assert integration_config.database.auto_rollback is True

        # E2E test config
        e2e_config = TestConfig(TestEnvironment.E2E)
        assert e2e_config.database.auto_rollback is False

    @pytest.mark.database
    @pytest.mark.serial
    def test_test_database_initialization(self, test_database: TestDatabase):
        """Test that test database initializes and works."""
        assert test_database is not None
        assert test_database.engine is not None
        assert test_database._schema_initialized is True

        # Test basic query
        with test_database.get_session() as session:
            result = session.execute(text("SELECT 1"))
            assert result.scalar() == 1

    @pytest.mark.database
    @pytest.mark.serial
    def test_database_seed_data(self, test_database: TestDatabase):
        """Test database seeding functionality."""
        test_database.seed_test_data("basic")

        # Check that data was seeded
        with test_database.get_session() as session:
            # Check instruments table
            result = session.execute(
                text("SELECT COUNT(*) FROM ml_instruments")
            )
            count = result.scalar()
            assert count >= 3  # Basic seed should have at least 3 instruments

    @pytest.mark.database
    @pytest.mark.serial
    def test_database_rollback(self, test_database: TestDatabase):
        """Test that database rollback works for test isolation."""
        # Insert data in a session
        with test_database.get_session() as session:
            session.execute(
                text("""
                    INSERT INTO ml_instruments (instrument_id, symbol, asset_type, tick_size, lot_size)
                    VALUES ('TEST.ROLLBACK', 'TEST', 'EQUITY', 0.01, 1)
                """)
            )
            # No explicit commit due to auto_rollback

        # Verify data was rolled back
        with test_database.get_session() as session:
            result = session.execute(
                text("SELECT COUNT(*) FROM ml_instruments WHERE instrument_id = 'TEST.ROLLBACK'")
            )
            assert result.scalar() == 0


@pytest.mark.database
@pytest.mark.serial
class TestMockServices:
    """Test mock service implementations."""

    @pytest.mark.database
    @pytest.mark.serial
    def test_mock_databento_client(self, mock_databento_client):
        """Test mock Databento client functionality."""
        assert isinstance(mock_databento_client, MockDatabentoClient)

        # Test list datasets
        datasets = mock_databento_client.list_datasets()
        assert len(datasets) > 0
        assert "XNAS.ITCH" in datasets

        # Test get data
        import pandas as pd
        data = mock_databento_client.get_data(
            dataset="XNAS.ITCH",
            symbols=["SPY"],
            schema="ohlcv-1m",
            start="2024-01-01",
            end="2024-01-02",
        )
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert "symbol" in data.columns
        assert "open" in data.columns

    @pytest.mark.database
    @pytest.mark.serial
    def test_mock_fred_client(self, mock_fred_client):
        """Test mock FRED client functionality."""
        assert isinstance(mock_fred_client, MockFredClient)

        # Test get series
        import pandas as pd
        data = mock_fred_client.get_series("DGS10")
        assert isinstance(data, pd.DataFrame)
        assert "value" in data.columns
        assert len(data) > 0

    @pytest.mark.database
    @pytest.mark.serial
    def test_mock_yahoo_client(self, mock_yahoo_client):
        """Test mock Yahoo client functionality."""
        assert isinstance(mock_yahoo_client, MockYahooClient)

        # Test get history
        import pandas as pd
        data = mock_yahoo_client.get_history("SPY")
        assert isinstance(data, pd.DataFrame)
        assert "Open" in data.columns
        assert "Close" in data.columns
        assert len(data) > 0

    @pytest.mark.database
    @pytest.mark.serial
    def test_mock_redis(self, mock_redis):
        """Test mock Redis functionality."""
        # Could be real or mock depending on environment
        assert mock_redis is not None

        # Test basic operations
        mock_redis.set("test_key", "test_value")
        value = mock_redis.get("test_key")
        assert value == b"test_value" or value == "test_value"

        # Test delete
        deleted = mock_redis.delete("test_key")
        assert deleted == 1
        assert mock_redis.get("test_key") is None


@pytest.mark.database
@pytest.mark.serial
class TestFixtureIntegration:
    """Test that fixtures work together correctly."""

    @pytest.mark.database
    @pytest.mark.serial
    def test_database_with_mocks(self, test_database, mock_databento_client):
        """Test database and mock services work together."""
        # Both should be available
        assert test_database is not None
        assert mock_databento_client is not None

        # Should be able to use both
        with test_database.get_session() as session:
            result = session.execute(text("SELECT 1"))
            assert result.scalar() == 1

        datasets = mock_databento_client.list_datasets()
        assert len(datasets) > 0

    @pytest.mark.database
    @pytest.mark.serial
    def test_seeded_database_fixture(self, seeded_database):
        """Test the seeded database fixture."""
        assert seeded_database is not None

        # Should have seed data
        with seeded_database.get_session() as session:
            result = session.execute(
                text("SELECT COUNT(*) FROM ml_instruments")
            )
            assert result.scalar() >= 3

    @pytest.mark.database
    @pytest.mark.serial
    def test_store_connection_fixtures(
        self,
        feature_store_connection,
        model_store_connection,
        strategy_store_connection,
    ):
        """Test store connection string fixtures."""
        assert feature_store_connection is not None
        assert model_store_connection is not None
        assert strategy_store_connection is not None

        # All should be valid connection strings
        assert "://" in feature_store_connection
        assert "://" in model_store_connection
        assert "://" in strategy_store_connection


@pytest.mark.database
@pytest.mark.serial
class TestEnvironmentDetection:
    """Test automatic environment detection."""

    @pytest.mark.database
    @pytest.mark.serial
    def test_environment_detection(self, test_config):
        """Test that environment is detected correctly."""
        # In pytest context, should detect unit tests by default
        assert test_config.environment in TestEnvironment

        # Timeout should be appropriate for environment
        assert test_config.timeout_seconds > 0
        assert test_config.timeout_seconds <= 300  # Max 5 minutes

    @pytest.mark.database
    @pytest.mark.serial
    def test_feature_flags(self, test_config):
        """Test feature flags configuration."""
        # By default, should use mocks
        assert test_config.use_real_databento is False
        assert test_config.use_real_fred is False

        # ML deps should be enabled by default
        assert test_config.enable_ml_deps_tests is True
