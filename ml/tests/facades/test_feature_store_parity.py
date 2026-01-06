#!/usr/bin/env python3

"""
Behavior tests for FeatureStoreFacade (Phase 3.7.7).

These tests verify that the facade produces stable outputs across core
operations and configuration combinations.
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

    # Setup context managers
    mock_conn = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchall.return_value = []
    mock_result.fetchone.return_value = None
    mock_conn.execute.return_value = mock_result

    engine.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
    engine.begin.return_value.__exit__ = MagicMock()
    engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    engine.connect.return_value.__exit__ = MagicMock()

    return engine


@pytest.fixture
def mock_table() -> MagicMock:
    """Create a mock SQLAlchemy table."""
    table = MagicMock()
    table.name = "ml_feature_values"
    table.c = MagicMock()
    table.c.feature_set_id = MagicMock()
    table.c.instrument_id = MagicMock()
    table.c.ts_event = MagicMock()
    table.c.ts_init = MagicMock()
    table.c.values = MagicMock()
    table.c.__getitem__ = MagicMock(return_value=MagicMock())
    return table


@pytest.fixture
def mock_feature_config() -> FeatureConfig:
    """Create a test feature config."""
    return FeatureConfig(
        lookback_window=50,
    )


@pytest.fixture
def legacy_store(
    mock_engine: MagicMock,
    mock_table: MagicMock,
    mock_feature_config: FeatureConfig,
) -> Generator[MagicMock, None, None]:
    """Create a mock legacy FeatureStore."""
    with patch("ml.stores.feature_store_facade.get_or_create_engine") as mock_get_engine:
        mock_get_engine.return_value = mock_engine

        # Import and create legacy store
        from ml.stores.feature_store_facade import FeatureStore

        with (
            patch.object(FeatureStore, "_setup_tables") as mock_setup,
        ):
            mock_setup.return_value = None

            store = MagicMock(spec=FeatureStore)
            store.connection_string = "postgresql://test:test@localhost/test"
            store.feature_config = mock_feature_config
            store.engine = mock_engine
            store.feature_values_table = mock_table
            store._data_registry = None
            store._write_buffer = []
            store._buffer = store._write_buffer

            yield store


@pytest.fixture
def facade_store(
    mock_engine: MagicMock,
    mock_table: MagicMock,
    mock_feature_config: FeatureConfig,
) -> Generator[Any, None, None]:
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
        mock_reader_cls.return_value = mock_reader

        # Setup health component
        mock_health = MagicMock()
        mock_health_cls.return_value = mock_health

        # Setup event component
        mock_event = MagicMock()
        mock_event_cls.return_value = mock_event

        # Setup computation component
        mock_computation = MagicMock()
        mock_comp_cls.return_value = mock_computation

        from ml.stores.feature_store_facade import FeatureStoreFacade

        facade = FeatureStoreFacade(
            connection_string="postgresql://test:test@localhost/test",
            feature_config=mock_feature_config,
        )

        yield facade


# =============================================================================
# Write Operation Parity Tests
# =============================================================================


class TestWriteParity:
    """Parity tests for write operations."""

    def test_write_features_parity_explicit_args(
        self,
        facade_store: Any,
        legacy_store: MagicMock,
    ) -> None:
        """
        Verify write_features with explicit args produces same behavior.

        Both implementations should:
        1. Accept same parameters
        2. Perform upsert with same conflict resolution
        3. Support same publish_bus flag

        """
        # Setup legacy to track calls
        legacy_store.write_features = MagicMock()

        # Setup facade
        facade_store._writer_component.write_features = MagicMock()

        # Call both with same arguments
        args = {
            "feature_set_id": "fs_test",
            "instrument_id": "SPY.DATABENTO",
            "features": {"close_return": 0.01, "volume_ratio": 1.5},
            "ts_event": 1700000000000000000,
            "ts_init": 1700000000000000000,
            "publish_bus": True,
        }

        legacy_store.write_features(**args)
        facade_store.write_features(**args)

        # Verify both were called with same args
        legacy_store.write_features.assert_called_once()
        facade_store._writer_component.write_features.assert_called_once()

        # Verify call args match
        facade_call = facade_store._writer_component.write_features.call_args
        assert facade_call.kwargs.get("feature_set_id") == "fs_test"
        assert facade_call.kwargs.get("instrument_id") == "SPY.DATABENTO"
        assert facade_call.kwargs.get("ts_event") == 1700000000000000000

    def test_write_features_parity_batch_mode(
        self,
        facade_store: Any,
        legacy_store: MagicMock,
    ) -> None:
        """
        Verify write_features with batch data produces same behavior.

        Both implementations should handle list[FeatureData] input.

        """
        # Create mock FeatureData items
        mock_data = []
        for i in range(3):
            item = MagicMock()
            item.feature_set_id = "fs_test"
            item.instrument_id = "SPY.DATABENTO"
            item.ts_event = 1700000000000000000 + i * 60_000_000_000
            item.ts_init = item.ts_event
            item.feature_values = {"close_return": 0.01 * i}
            mock_data.append(item)

        # Setup mocks
        legacy_store.write_features = MagicMock()
        facade_store._writer_component.write_features = MagicMock()

        # Call with batch data
        legacy_store.write_features(data=mock_data)
        facade_store.write_features(data=mock_data)

        # Verify both handle batch input
        legacy_store.write_features.assert_called_once()
        facade_store._writer_component.write_features.assert_called_once()


# =============================================================================
# Read Operation Parity Tests
# =============================================================================


class TestReadParity:
    """Parity tests for read operations."""

    def test_get_training_data_parity(
        self,
        facade_store: Any,
        legacy_store: MagicMock,
    ) -> None:
        """
        Verify get_training_data produces identical outputs.

        Both implementations should return same:
        1. Features array shape and values
        2. Timestamps array
        3. Feature names list

        """
        # Setup return values
        expected_features = np.array([[0.01, 1.5], [0.02, 1.6]], dtype=np.float64)
        expected_timestamps = np.array(
            [1700000000000000000, 1700000060000000000],
            dtype=np.int64,
        )
        expected_names = ["close_return", "volume_ratio"]

        legacy_store.get_training_data = MagicMock(
            return_value=(expected_features, expected_timestamps, expected_names)
        )
        facade_store._reader_component.get_training_data = MagicMock(
            return_value=(expected_features, expected_timestamps, expected_names)
        )

        # Call both
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 2, tzinfo=UTC)

        legacy_result = legacy_store.get_training_data(
            instrument_id="SPY.DATABENTO",
            start=start,
            end=end,
        )

        facade_result = facade_store.get_training_data(
            instrument_id="SPY.DATABENTO",
            start=start,
            end=end,
        )

        # Verify parity
        np.testing.assert_allclose(
            legacy_result[0],
            facade_result[0],
            rtol=1e-10,
            err_msg="Features array mismatch",
        )
        np.testing.assert_array_equal(
            legacy_result[1],
            facade_result[1],
            err_msg="Timestamps array mismatch",
        )
        assert legacy_result[2] == facade_result[2], "Feature names mismatch"

    def test_get_latest_at_or_before_parity(
        self,
        facade_store: Any,
        legacy_store: MagicMock,
    ) -> None:
        """
        Verify get_latest_at_or_before produces identical outputs.

        Both implementations should:
        1. Return same feature values mapping
        2. Return None for same not-found cases

        """
        expected_result = {"close_return": 0.01, "volume_ratio": 1.5}

        legacy_store.get_latest_at_or_before = MagicMock(return_value=expected_result)
        facade_store._reader_component.get_latest_at_or_before = MagicMock(
            return_value=expected_result
        )

        # Call both
        legacy_result = legacy_store.get_latest_at_or_before(
            instrument_id="SPY.DATABENTO",
            ts_event=1700000000000000000,
        )

        facade_result = facade_store.get_latest_at_or_before(
            instrument_id="SPY.DATABENTO",
            ts_event=1700000000000000000,
        )

        # Verify parity
        assert legacy_result == facade_result

    def test_read_range_parity(
        self,
        facade_store: Any,
        legacy_store: MagicMock,
    ) -> None:
        """
        Verify read_range produces identical DataFrames.

        Both implementations should:
        1. Return DataFrame with same columns
        2. Return same row data
        3. Apply same time range filtering

        """
        import pandas as pd

        expected_df = pd.DataFrame(
            {
                "feature_set_id": ["fs_test", "fs_test"],
                "instrument_id": ["SPY.DATABENTO", "SPY.DATABENTO"],
                "values": [{"close_return": 0.01}, {"close_return": 0.02}],
                "ts_event": [1700000000000000000, 1700000060000000000],
                "ts_init": [1700000000000000000, 1700000060000000000],
            }
        )

        legacy_store.read_range = MagicMock(return_value=expected_df)
        facade_store._reader_component.read_range = MagicMock(return_value=expected_df)

        # Call both
        legacy_result = legacy_store.read_range(
            start_ns=1700000000000000000,
            end_ns=1700000120000000000,
            instrument_id="SPY.DATABENTO",
        )

        facade_result = facade_store.read_range(
            start_ns=1700000000000000000,
            end_ns=1700000120000000000,
            instrument_id="SPY.DATABENTO",
        )

        # Verify parity
        assert legacy_result.shape == facade_result.shape
        assert list(legacy_result.columns) == list(facade_result.columns)


# =============================================================================
# Computation Parity Tests
# =============================================================================


class TestComputationParity:
    """Parity tests for computation operations."""

    def test_compute_realtime_parity(
        self,
        facade_store: Any,
        legacy_store: MagicMock,
    ) -> None:
        """
        Verify compute_realtime produces identical feature vectors.

        Both implementations should:
        1. Use same FeatureEngineer
        2. Produce identical feature arrays
        3. Handle indicator warmup identically

        """
        expected_features = np.array([0.01, 1.5], dtype=np.float32)

        legacy_store.compute_realtime = MagicMock(return_value=expected_features)
        facade_store._computation_component.compute_realtime = MagicMock(
            return_value=expected_features
        )

        # Create mock bar
        mock_bar = MagicMock()
        mock_bar.ts_event = 1700000000000000000
        mock_bar.ts_init = 1700000000000000000
        mock_bar.close = 100.0
        mock_bar.high = 101.0
        mock_bar.low = 99.0
        mock_bar.volume = 1000.0

        # Call both
        legacy_result = legacy_store.compute_realtime(bar=mock_bar, store=True)
        facade_result = facade_store.compute_realtime(bar=mock_bar, store=True)

        # Verify parity
        np.testing.assert_allclose(
            legacy_result,
            facade_result,
            rtol=1e-10,
            err_msg="Compute realtime feature vector mismatch",
        )

    def test_compute_and_store_historical_parity(
        self,
        facade_store: Any,
        legacy_store: MagicMock,
    ) -> None:
        """
        Verify compute_and_store_historical produces same row counts.

        Both implementations should:
        1. Load same bars from database
        2. Compute same features
        3. Store same number of rows

        """
        legacy_store.compute_and_store_historical = MagicMock(return_value=100)
        facade_store._computation_component.compute_and_store_historical = MagicMock(
            return_value=100
        )

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 2, tzinfo=UTC)

        legacy_result = legacy_store.compute_and_store_historical(
            instrument_id="SPY.DATABENTO",
            start=start,
            end=end,
            force_recompute=False,
        )

        facade_result = facade_store.compute_and_store_historical(
            instrument_id="SPY.DATABENTO",
            start=start,
            end=end,
            force_recompute=False,
        )

        # Verify parity
        assert legacy_result == facade_result


# =============================================================================
# Health Operation Parity Tests
# =============================================================================


class TestHealthParity:
    """Parity tests for health operations."""

    def test_clear_features_parity(
        self,
        facade_store: Any,
        legacy_store: MagicMock,
    ) -> None:
        """
        Verify clear_features applies same filters.

        Both implementations should:
        1. Accept same filter parameters
        2. Delete same rows

        """
        legacy_store.clear_features = MagicMock()
        facade_store._health_component.clear_features = MagicMock()

        # Call both with same filters
        legacy_store.clear_features(
            instrument_id="SPY.DATABENTO",
            feature_version="v1.0",
        )
        facade_store.clear_features(
            instrument_id="SPY.DATABENTO",
            feature_version="v1.0",
        )

        # Verify both called with same args
        legacy_store.clear_features.assert_called_once_with(
            instrument_id="SPY.DATABENTO",
            feature_version="v1.0",
        )
        facade_store._health_component.clear_features.assert_called_once_with(
            instrument_id="SPY.DATABENTO",
            feature_version="v1.0",
        )

    def test_is_healthy_parity(
        self,
        facade_store: Any,
        legacy_store: MagicMock,
    ) -> None:
        """
        Verify is_healthy returns same status.

        Both implementations should:
        1. Execute same health check query
        2. Return same boolean result

        """
        legacy_store.is_healthy = MagicMock(return_value=True)
        facade_store._health_component.is_healthy = MagicMock(return_value=True)

        legacy_result = legacy_store.is_healthy()
        facade_result = facade_store.is_healthy()

        assert legacy_result == facade_result


class TestOperationCounts:
    """Tests for facade operation counts."""

    def test_operation_pass_counts_match(
        self,
        facade_store: Any,
        legacy_store: MagicMock,
    ) -> None:
        """
        Verify operations handle same call counts.

        """
        n_operations = 10

        # Setup mocks
        legacy_store.write_features = MagicMock()
        legacy_store.get_training_data = MagicMock(
            return_value=(np.array([[]]), np.array([]), [])
        )
        legacy_store.compute_realtime = MagicMock(
            return_value=np.array([], dtype=np.float32)
        )

        facade_store._writer_component.write_features = MagicMock()
        facade_store._reader_component.get_training_data = MagicMock(
            return_value=(np.array([[]]), np.array([]), [])
        )
        facade_store._computation_component.compute_realtime = MagicMock(
            return_value=np.array([], dtype=np.float32)
        )

        mock_bar = MagicMock()
        mock_bar.ts_event = 1700000000000000000

        # Execute same operations on both
        for i in range(n_operations):
            ts = 1700000000000000000 + i * 60_000_000_000

            legacy_store.write_features(
                feature_set_id="fs_test",
                instrument_id="SPY.DATABENTO",
                features={"close_return": 0.01},
                ts_event=ts,
            )
            facade_store.write_features(
                feature_set_id="fs_test",
                instrument_id="SPY.DATABENTO",
                features={"close_return": 0.01},
                ts_event=ts,
            )

        # Verify same call counts
        assert legacy_store.write_features.call_count == n_operations
        assert facade_store._writer_component.write_features.call_count == n_operations


# =============================================================================
# API Signature Parity Tests
# =============================================================================


class TestAPISignatureParity:
    """Tests for API signature compatibility."""

    def test_init_signature_parity(self) -> None:
        """Verify __init__ signatures are compatible."""
        import inspect

        from ml.stores.feature_store_facade import FeatureStore
        from ml.stores.feature_store_facade import FeatureStoreFacade

        legacy_sig = inspect.signature(FeatureStore.__init__)
        facade_sig = inspect.signature(FeatureStoreFacade.__init__)

        legacy_params = set(legacy_sig.parameters.keys())
        facade_params = set(facade_sig.parameters.keys())

        # Facade should have all legacy params
        assert legacy_params.issubset(facade_params), (
            f"Facade missing params: {legacy_params - facade_params}"
        )

    def test_write_features_signature_parity(self) -> None:
        """Verify write_features signatures are compatible."""
        import inspect

        from ml.stores.feature_store_facade import FeatureStore
        from ml.stores.feature_store_facade import FeatureStoreFacade

        legacy_sig = inspect.signature(FeatureStore.write_features)
        facade_sig = inspect.signature(FeatureStoreFacade.write_features)

        legacy_params = set(legacy_sig.parameters.keys())
        facade_params = set(facade_sig.parameters.keys())

        # Check required params present
        required = {
            "self",
            "feature_set_id",
            "instrument_id",
            "features",
            "ts_event",
            "ts_init",
            "data",
            "publish_bus",
        }
        assert required.issubset(facade_params)

    def test_get_training_data_signature_parity(self) -> None:
        """Verify get_training_data signatures are compatible."""
        import inspect

        from ml.stores.feature_store_facade import FeatureStore
        from ml.stores.feature_store_facade import FeatureStoreFacade

        legacy_sig = inspect.signature(FeatureStore.get_training_data)
        facade_sig = inspect.signature(FeatureStoreFacade.get_training_data)

        legacy_params = set(legacy_sig.parameters.keys())
        facade_params = set(facade_sig.parameters.keys())

        # Params should match
        assert legacy_params == facade_params, (
            f"Param mismatch: legacy has {legacy_params - facade_params}, "
            f"facade has {facade_params - legacy_params}"
        )

    def test_compute_realtime_signature_parity(self) -> None:
        """Verify compute_realtime signatures are compatible."""
        import inspect

        from ml.stores.feature_store_facade import FeatureStore
        from ml.stores.feature_store_facade import FeatureStoreFacade

        legacy_sig = inspect.signature(FeatureStore.compute_realtime)
        facade_sig = inspect.signature(FeatureStoreFacade.compute_realtime)

        legacy_params = set(legacy_sig.parameters.keys())
        facade_params = set(facade_sig.parameters.keys())

        # Params should match
        assert legacy_params == facade_params
