#!/usr/bin/env python3

"""
Unit tests for FeatureStoreFacade (Phase 3.7.7).

These tests verify that the facade:
1. Correctly wires all 6 components together
2. Preserves the exact public API of legacy FeatureStore
3. Delegates operations to the appropriate components
4. Accepts all legacy init parameters

Test Categories:
- Component wiring and initialization
- Public API preservation
- Delegation to components
- Feature flag switching

"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml.features import FeatureConfig
from ml.stores.feature_store_facade import FeatureStoreFacade


if TYPE_CHECKING:
    from collections.abc import Generator


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_engine() -> MagicMock:
    """Create a mock SQLAlchemy engine."""
    engine = MagicMock()
    engine.dialect.name = "postgresql"
    engine.begin.return_value.__enter__ = MagicMock()
    engine.begin.return_value.__exit__ = MagicMock()
    engine.connect.return_value.__enter__ = MagicMock()
    engine.connect.return_value.__exit__ = MagicMock()
    return engine


@pytest.fixture
def mock_table() -> MagicMock:
    """Create a mock SQLAlchemy table."""
    table = MagicMock()
    table.c = MagicMock()
    table.c.feature_set_id = MagicMock()
    table.c.instrument_id = MagicMock()
    table.c.ts_event = MagicMock()
    return table


@pytest.fixture
def mock_feature_config() -> FeatureConfig:
    """Create a test feature config."""
    return FeatureConfig(
        lookback_window=50,
    )


@pytest.fixture
def connection_string(cloned_test_database: str) -> str:
    """Provide isolated Postgres connection string for facade initialization."""
    return cloned_test_database


@pytest.fixture
def facade_with_mocks(
    mock_engine: MagicMock,
    mock_table: MagicMock,
    mock_feature_config: FeatureConfig,
    connection_string: str,
) -> Generator[FeatureStoreFacade, None, None]:
    """Create a FeatureStoreFacade with mocked components."""
    with (
        patch("ml.stores.feature_store_facade.get_or_create_engine") as mock_get_engine,
        patch("ml.stores.feature_store_facade.FeatureSchemaComponent") as mock_schema_cls,
        patch("ml.stores.feature_store_facade.FeatureWriterComponent") as mock_writer_cls,
        patch("ml.stores.feature_store_facade.FeatureReaderComponent") as mock_reader_cls,
        patch("ml.stores.feature_store_facade.FeatureHealthComponent") as mock_health_cls,
        patch("ml.stores.feature_store_facade.FeatureEventComponent") as mock_event_cls,
        patch("ml.stores.feature_store_facade.FeatureComputationComponent") as mock_comp_cls,
    ):
        mock_get_engine.return_value = mock_engine

        # Setup schema component
        mock_schema = MagicMock()
        mock_schema.setup_tables.return_value = mock_table
        mock_schema.metadata = MagicMock()
        mock_schema.pipeline_hash = "abc123def456"
        mock_schema.get_feature_set_id.return_value = "fs_abc123def4"
        mock_schema.get_feature_names.return_value = ["close_return", "volume_ratio"]
        mock_schema.get_feature_names_online.return_value = ["close_return"]
        mock_schema.compute_config_hash.return_value = "1234567890abcdef"
        mock_schema_cls.return_value = mock_schema

        # Setup writer component
        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer

        # Setup reader component
        mock_reader = MagicMock()
        mock_reader.get_training_data.return_value = (
            np.array([[0.01, 1.5]]),
            np.array([1700000000000000000]),
            ["close_return", "volume_ratio"],
        )
        mock_reader.get_latest_at_or_before.return_value = {"close_return": 0.01}
        mock_reader_cls.return_value = mock_reader

        # Setup health component
        mock_health = MagicMock()
        mock_health.is_healthy.return_value = True
        mock_health_cls.return_value = mock_health

        # Setup event component
        mock_event = MagicMock()
        mock_event_cls.return_value = mock_event

        # Setup computation component
        mock_computation = MagicMock()
        mock_computation.compute_realtime.return_value = np.array([0.01], dtype=np.float32)
        mock_computation.compute_and_store_historical.return_value = 100
        mock_computation.compute_historical_parallel.return_value = {"SPY": 100}
        mock_comp_cls.return_value = mock_computation

        facade = FeatureStoreFacade(
            connection_string=connection_string,
            feature_config=mock_feature_config,
        )

        yield facade


# =============================================================================
# Component Wiring Tests
# =============================================================================


class TestFacadeComponentWiring:
    """Tests for facade component wiring."""

    def test_facade_wires_all_components(
        self,
        facade_with_mocks: FeatureStoreFacade,
    ) -> None:
        """Verify facade initializes all 6 components."""
        facade = facade_with_mocks

        # Verify all components are present
        assert facade._schema_component is not None
        assert facade._writer_component is not None
        assert facade._reader_component is not None
        assert facade._health_component is not None
        assert facade._event_component is not None
        assert facade._computation_component is not None

    def test_facade_initializes_engine(
        self,
        facade_with_mocks: FeatureStoreFacade,
        connection_string: str,
    ) -> None:
        """Verify facade creates SQLAlchemy engine."""
        facade = facade_with_mocks

        assert facade.engine is not None
        assert facade.connection_string == connection_string

    def test_facade_initializes_feature_engineer(
        self,
        facade_with_mocks: FeatureStoreFacade,
    ) -> None:
        """Verify facade creates FeatureEngineer."""
        facade = facade_with_mocks

        assert facade.feature_engineer is not None
        assert facade.feature_config is not None


# =============================================================================
# Public API Preservation Tests
# =============================================================================


class TestFacadePublicAPI:
    """Tests for public API preservation."""

    def test_facade_preserves_public_api(self) -> None:
        """Verify facade has all required public methods and attributes."""
        # Check methods exist
        assert hasattr(FeatureStoreFacade, "__init__")
        assert hasattr(FeatureStoreFacade, "set_data_registry")
        assert hasattr(FeatureStoreFacade, "compute_historical_parallel")
        assert hasattr(FeatureStoreFacade, "compute_and_store_historical")
        assert hasattr(FeatureStoreFacade, "compute_realtime")
        assert hasattr(FeatureStoreFacade, "get_training_data")
        assert hasattr(FeatureStoreFacade, "get_latest_at_or_before")
        assert hasattr(FeatureStoreFacade, "clear_features")
        assert hasattr(FeatureStoreFacade, "write_features")
        assert hasattr(FeatureStoreFacade, "flush")
        assert hasattr(FeatureStoreFacade, "write_batch")
        assert hasattr(FeatureStoreFacade, "is_healthy")
        assert hasattr(FeatureStoreFacade, "read_range")
        assert hasattr(FeatureStoreFacade, "store_features")

    def test_facade_accepts_all_legacy_init_params(self) -> None:
        """Verify facade accepts all legacy __init__ parameters."""
        import inspect

        sig = inspect.signature(FeatureStoreFacade.__init__)
        params = set(sig.parameters.keys())

        # Required parameters from legacy FeatureStore
        expected_params = {
            "self",
            "connection_string",
            "feature_config",
            "pipeline_spec",
            "persistence_manager",
            "enable_publishing",
            "publisher",
            "publish_mode",
        }

        # All expected params should be present
        assert expected_params.issubset(params), (
            f"Missing params: {expected_params - params}"
        )

    def test_facade_exposes_required_attributes(
        self,
        facade_with_mocks: FeatureStoreFacade,
    ) -> None:
        """Verify facade exposes required attributes."""
        facade = facade_with_mocks

        # Check attributes exist
        assert hasattr(facade, "connection_string")
        assert hasattr(facade, "feature_config")
        assert hasattr(facade, "pipeline_spec")
        assert hasattr(facade, "engine")
        assert hasattr(facade, "metadata")
        assert hasattr(facade, "feature_engineer")
        assert hasattr(facade, "_indicator_managers")
        assert hasattr(facade, "pipeline_runner_offline")
        assert hasattr(facade, "pipeline_runner_online")
        assert hasattr(facade, "pipeline_hash")
        assert hasattr(facade, "_write_buffer")
        assert hasattr(facade, "_buffer")
        assert hasattr(facade, "_circuit_breaker")
        assert hasattr(facade, "feature_values_table")


# =============================================================================
# Delegation Tests
# =============================================================================


class TestFacadeDelegation:
    """Tests for delegation to components."""

    def test_facade_delegates_write_features_to_writer(
        self,
        facade_with_mocks: FeatureStoreFacade,
    ) -> None:
        """Verify write_features delegates to writer component."""
        facade = facade_with_mocks

        facade.write_features(
            feature_set_id="fs_test",
            instrument_id="SPY.DATABENTO",
            features={"close_return": 0.01},
            ts_event=1700000000000000000,
        )

        facade._writer_component.write_features.assert_called_once_with(
            feature_set_id="fs_test",
            instrument_id="SPY.DATABENTO",
            features={"close_return": 0.01},
            ts_event=1700000000000000000,
            ts_init=None,
            data=None,
            publish_bus=True,
        )

    def test_facade_delegates_get_training_data_to_reader(
        self,
        facade_with_mocks: FeatureStoreFacade,
    ) -> None:
        """Verify get_training_data delegates to reader component."""
        facade = facade_with_mocks

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 2, tzinfo=UTC)

        result = facade.get_training_data(
            instrument_id="SPY.DATABENTO",
            start=start,
            end=end,
        )

        facade._reader_component.get_training_data.assert_called_once_with(
            instrument_id="SPY.DATABENTO",
            start=start,
            end=end,
            include_bars=True,
        )
        assert result is not None
        assert len(result) == 3  # features, timestamps, names

    def test_facade_delegates_compute_realtime_to_computation(
        self,
        facade_with_mocks: FeatureStoreFacade,
    ) -> None:
        """Verify compute_realtime delegates to computation component."""
        facade = facade_with_mocks

        mock_bar = MagicMock()
        mock_bar.ts_event = 1700000000000000000
        mock_bar.ts_init = 1700000000000000000
        mock_bar.close = 100.0
        mock_bar.high = 101.0
        mock_bar.low = 99.0
        mock_bar.volume = 1000.0

        result = facade.compute_realtime(bar=mock_bar, store=True)

        facade._computation_component.compute_realtime.assert_called_once_with(
            bar=mock_bar,
            store=True,
            indicator_manager=None,
        )
        assert isinstance(result, np.ndarray)

    def test_facade_delegates_is_healthy_to_health(
        self,
        facade_with_mocks: FeatureStoreFacade,
    ) -> None:
        """Verify is_healthy delegates to health component."""
        facade = facade_with_mocks

        result = facade.is_healthy()

        facade._health_component.is_healthy.assert_called_once()
        assert result is True

    def test_facade_delegates_clear_features_to_health(
        self,
        facade_with_mocks: FeatureStoreFacade,
    ) -> None:
        """Verify clear_features delegates to health component."""
        facade = facade_with_mocks

        facade.clear_features(instrument_id="SPY.DATABENTO")

        facade._health_component.clear_features.assert_called_once_with(
            instrument_id="SPY.DATABENTO",
            feature_version=None,
        )

    def test_facade_delegates_flush_to_health(
        self,
        facade_with_mocks: FeatureStoreFacade,
    ) -> None:
        """Verify flush delegates to health component."""
        facade = facade_with_mocks

        facade.flush()

        facade._health_component.flush.assert_called_once()

    def test_facade_delegates_write_batch_to_writer(
        self,
        facade_with_mocks: FeatureStoreFacade,
    ) -> None:
        """Verify write_batch delegates to writer component."""
        facade = facade_with_mocks

        mock_data = [MagicMock(), MagicMock()]
        facade.write_batch(mock_data)

        facade._writer_component.write_batch.assert_called_once_with(mock_data)

    def test_facade_delegates_store_features_to_writer(
        self,
        facade_with_mocks: FeatureStoreFacade,
    ) -> None:
        """Verify store_features delegates to writer component."""
        facade = facade_with_mocks

        facade.store_features(
            instrument_id="SPY.DATABENTO",
            ts_event=1700000000000000000,
            features={"close_return": 0.01},
        )

        facade._writer_component.store_features.assert_called_once()

    def test_facade_delegates_get_latest_at_or_before_to_reader(
        self,
        facade_with_mocks: FeatureStoreFacade,
    ) -> None:
        """Verify get_latest_at_or_before delegates to reader component."""
        facade = facade_with_mocks

        result = facade.get_latest_at_or_before(
            instrument_id="SPY.DATABENTO",
            ts_event=1700000000000000000,
        )

        facade._reader_component.get_latest_at_or_before.assert_called_once_with(
            instrument_id="SPY.DATABENTO",
            ts_event=1700000000000000000,
        )
        assert result == {"close_return": 0.01}

    def test_facade_delegates_compute_and_store_historical_to_computation(
        self,
        facade_with_mocks: FeatureStoreFacade,
    ) -> None:
        """Verify compute_and_store_historical delegates to computation component."""
        facade = facade_with_mocks

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 2, tzinfo=UTC)

        result = facade.compute_and_store_historical(
            instrument_id="SPY.DATABENTO",
            start=start,
            end=end,
            force_recompute=False,
        )

        facade._computation_component.compute_and_store_historical.assert_called_once_with(
            instrument_id="SPY.DATABENTO",
            start=start,
            end=end,
            force_recompute=False,
        )
        assert result == 100

    def test_facade_delegates_compute_historical_parallel_to_computation(
        self,
        facade_with_mocks: FeatureStoreFacade,
    ) -> None:
        """Verify compute_historical_parallel delegates to computation component."""
        facade = facade_with_mocks

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 2, tzinfo=UTC)

        result = facade.compute_historical_parallel(
            instrument_ids=["SPY.DATABENTO"],
            start=start,
            end=end,
        )

        facade._computation_component.compute_historical_parallel.assert_called_once_with(
            instrument_ids=["SPY.DATABENTO"],
            start=start,
            end=end,
            force_recompute=False,
            max_workers=4,
        )
        assert result == {"SPY": 100}


# =============================================================================
# Facade Export Tests
# =============================================================================


def test_feature_store_export_is_facade() -> None:
    """Verify FeatureStore export resolves to FeatureStoreFacade."""
    from ml.stores import FeatureStore

    assert FeatureStore is FeatureStoreFacade, f"Expected FeatureStoreFacade, got {FeatureStore}"


# =============================================================================
# Registry Integration Tests
# =============================================================================


class TestRegistryIntegration:
    """Tests for DataRegistry integration."""

    def test_facade_set_data_registry(
        self,
        facade_with_mocks: FeatureStoreFacade,
    ) -> None:
        """Verify set_data_registry propagates to components."""
        facade = facade_with_mocks

        mock_registry = MagicMock()
        facade.set_data_registry(mock_registry)

        assert facade._data_registry is mock_registry
        facade._computation_component.set_data_registry.assert_called_once_with(
            mock_registry
        )

    def test_facade_get_data_registry(
        self,
        facade_with_mocks: FeatureStoreFacade,
    ) -> None:
        """Verify _get_data_registry returns set registry."""
        facade = facade_with_mocks

        mock_registry = MagicMock()
        facade._data_registry = mock_registry

        result = facade._get_data_registry()
        assert result is mock_registry


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_facade_with_none_feature_config(self, connection_string: str) -> None:
        """Verify facade handles None feature_config."""
        with (
            patch("ml.stores.feature_store_facade.get_or_create_engine") as mock_get_engine,
            patch("ml.stores.feature_store_facade.FeatureSchemaComponent") as mock_schema_cls,
            patch("ml.stores.feature_store_facade.FeatureWriterComponent"),
            patch("ml.stores.feature_store_facade.FeatureReaderComponent"),
            patch("ml.stores.feature_store_facade.FeatureHealthComponent"),
            patch("ml.stores.feature_store_facade.FeatureEventComponent"),
            patch("ml.stores.feature_store_facade.FeatureComputationComponent"),
        ):
            mock_engine = MagicMock()
            mock_engine.dialect.name = "postgresql"
            mock_get_engine.return_value = mock_engine

            mock_schema = MagicMock()
            mock_schema.setup_tables.return_value = MagicMock()
            mock_schema.metadata = MagicMock()
            mock_schema.pipeline_hash = "abc123"
            mock_schema_cls.return_value = mock_schema

            facade = FeatureStoreFacade(
                connection_string=connection_string,
                feature_config=None,
            )

            # Should use default FeatureConfig
            assert facade.feature_config is not None
            assert isinstance(facade.feature_config, FeatureConfig)

    def test_facade_with_ml_feature_config(self, connection_string: str) -> None:
        """Verify facade handles MLFeatureConfig."""
        from ml.config.base import MLFeatureConfig

        with (
            patch("ml.stores.feature_store_facade.get_or_create_engine") as mock_get_engine,
            patch("ml.stores.feature_store_facade.FeatureSchemaComponent") as mock_schema_cls,
            patch("ml.stores.feature_store_facade.FeatureWriterComponent"),
            patch("ml.stores.feature_store_facade.FeatureReaderComponent"),
            patch("ml.stores.feature_store_facade.FeatureHealthComponent"),
            patch("ml.stores.feature_store_facade.FeatureEventComponent"),
            patch("ml.stores.feature_store_facade.FeatureComputationComponent"),
        ):
            mock_engine = MagicMock()
            mock_engine.dialect.name = "postgresql"
            mock_get_engine.return_value = mock_engine

            mock_schema = MagicMock()
            mock_schema.setup_tables.return_value = MagicMock()
            mock_schema.metadata = MagicMock()
            mock_schema.pipeline_hash = "abc123"
            mock_schema_cls.return_value = mock_schema

            ml_config = MLFeatureConfig(lookback_window=100)

            facade = FeatureStoreFacade(
                connection_string=connection_string,
                feature_config=ml_config,
            )

            # Should convert to FeatureConfig
            assert facade.feature_config is not None
            assert isinstance(facade.feature_config, FeatureConfig)

    def test_facade_buffer_alias(
        self,
        facade_with_mocks: FeatureStoreFacade,
    ) -> None:
        """Verify _buffer is an alias for _write_buffer."""
        facade = facade_with_mocks

        # Both should reference the same list
        assert facade._buffer is facade._write_buffer

        # Modifying one should affect the other
        facade._write_buffer.append("test")
        assert "test" in facade._buffer

        facade._buffer.clear()
        assert len(facade._write_buffer) == 0
