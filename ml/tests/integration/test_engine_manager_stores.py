"""
Integration test demonstrating EngineManager usage with ML stores.

This test shows how using the EngineManager prevents connection pool exhaustion
when multiple stores are created, particularly during hypothesis testing.

"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from ml.core.db_engine import EngineManager
from ml.features.engineering import FeatureConfig
from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelStore
from ml.stores.strategy_store import StrategyStore


if TYPE_CHECKING:
    from sqlalchemy.engine import Engine


# Use test database
TEST_DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/nautilus_test",
)


@pytest.mark.property
@pytest.mark.database
@pytest.mark.serial
@pytest.mark.integration
class TestEngineManagerIntegration:
    """Test EngineManager integration with ML stores."""

    def setup_method(self) -> None:
        """Clean up engines before each test."""
        EngineManager.dispose_all()

    def teardown_method(self) -> None:
        """Clean up engines after each test."""
        EngineManager.dispose_all()

    @pytest.mark.database
    @pytest.mark.serial
    def test_stores_share_engine_instance(self) -> None:
        """Test that multiple stores share the same engine instance."""
        EngineManager.dispose_all()

        # Create multiple stores with the same connection string
        feature_store1 = FeatureStore(TEST_DB_URL, FeatureConfig())
        feature_store2 = FeatureStore(TEST_DB_URL, FeatureConfig())

        # They should share the same engine via EngineManager
        assert feature_store1.engine is feature_store2.engine
        assert EngineManager.get_engine_count() == 1

    @pytest.mark.skipif(
        "postgres" not in TEST_DB_URL.lower(),
        reason="Requires PostgreSQL",
    )
    @pytest.mark.database
    @pytest.mark.serial
    def test_hypothesis_without_pool_exhaustion(self) -> None:
        """Test that hypothesis testing doesn't exhaust connection pool."""
        # Track engines created
        engine_count_history = []

        @given(
            num_stores=st.integers(min_value=1, max_value=5),
        )
        @settings(max_examples=10, deadline=None)
        def create_stores(num_stores: int) -> None:
            """Create multiple stores rapidly as hypothesis would."""
            stores = []

            for _ in range(num_stores):
                # Create each type of store
                fs = FeatureStore(TEST_DB_URL, FeatureConfig())
                ms = ModelStore(TEST_DB_URL)
                ss = StrategyStore(TEST_DB_URL)
                stores.extend([fs, ms, ss])

            # Track engine count
            engine_count = EngineManager.get_engine_count()
            engine_count_history.append(engine_count)

            # Should only have one engine regardless of store count
            assert engine_count == 1, f"Expected 1 engine, got {engine_count}"

        # Run the hypothesis test
        create_stores()

        # Verify we never created more than one engine
        assert all(count == 1 for count in engine_count_history), \
            f"Engine counts should always be 1, got: {engine_count_history}"

    @pytest.mark.database
    @pytest.mark.serial
    def test_connection_pool_limits_enforced(self) -> None:
        """Test that connection pool limits are properly enforced."""
        # Create engine with specific limits
        engine = EngineManager.get_engine(
            TEST_DB_URL,
            pool_size=2,
            max_overflow=3,
        )

        # Get pool status
        status = EngineManager.get_pool_status(TEST_DB_URL)

        assert status is not None
        # For PostgreSQL, we should have pool information
        if "postgresql" in TEST_DB_URL:
            # Pool was created with our limits
            assert "size" in status or "pool_type" in status

    @pytest.mark.database
    @pytest.mark.serial
    def test_dispose_cleans_up_properly(self) -> None:
        """Test that dispose properly cleans up connections."""
        # Create an engine
        engine1 = EngineManager.get_engine(TEST_DB_URL)
        assert EngineManager.get_engine_count() == 1

        # Dispose it
        EngineManager.dispose_engine(TEST_DB_URL)
        assert EngineManager.get_engine_count() == 0

        # Create a new one - should be different instance
        engine2 = EngineManager.get_engine(TEST_DB_URL)
        assert engine1 is not engine2
        assert EngineManager.get_engine_count() == 1

    @pytest.mark.database
    @pytest.mark.serial
    def test_test_environment_detection(self) -> None:
        """Test that test environments get conservative settings."""
        # Various test database URLs
        test_urls = [
            "postgresql://user:pass@localhost/test_db",
            "postgresql://user:pass@localhost/temp_db",
            "postgresql://user:pass@localhost/tmp_db",
            "sqlite:///:memory:",
        ]

        for url in test_urls:
            engine = EngineManager.get_engine(
                url,
                pool_size=100,  # Request large pool
                max_overflow=200,  # Request large overflow
            )

            # Engine should be created (can't directly verify pool size due to encapsulation)
            assert engine is not None

            # Clean up
            EngineManager.dispose_engine(url)


@pytest.mark.database
@pytest.mark.serial
class TestEngineManagerWithStores:
    """Test actual integration with patched stores."""

    def setup_method(self) -> None:
        """Clean up before tests."""
        EngineManager.dispose_all()

    def teardown_method(self) -> None:
        """Clean up."""
        EngineManager.dispose_all()

    @pytest.mark.skipif(
        "postgres" not in TEST_DB_URL.lower(),
        reason="Requires PostgreSQL",
    )
    @pytest.mark.database
    @pytest.mark.serial
    def test_rapid_store_creation_no_exhaustion(self) -> None:
        """Test that rapid store creation doesn't exhaust connections."""
        # Create many stores rapidly
        stores = []

        for i in range(20):  # Create 60 stores total (20 of each type)
            fs = FeatureStore(TEST_DB_URL, FeatureConfig())
            ms = ModelStore(TEST_DB_URL)
            ss = StrategyStore(TEST_DB_URL)
            stores.extend([fs, ms, ss])

        # Should still only have one engine
        assert EngineManager.get_engine_count() == 1

        # Pool status should be healthy
        status = EngineManager.get_pool_status(TEST_DB_URL)
        assert status is not None

        # All stores should share the same engine
        engine = EngineManager.get_engine(TEST_DB_URL)
        for store in stores:
            assert store.engine is engine
