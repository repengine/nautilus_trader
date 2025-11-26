#!/usr/bin/env python3

"""
Unit tests for FeatureComputationComponent (Phase 3.7.3).

Tests feature computation operations including real-time (hot path) and
historical batch computation, indicator manager integration, and parallel
processing.

Coverage target: 90%

Test Cases (from test design report):
- test_compute_realtime_returns_feature_array
- test_compute_realtime_returns_empty_when_not_warmed
- test_compute_realtime_stores_when_enabled
- test_compute_realtime_uses_indicator_manager
- test_compute_realtime_creates_internal_indicator_manager
- test_compute_historical_computes_batch_features
- test_compute_historical_skips_when_features_exist
- test_compute_historical_force_recompute_overwrites
- test_compute_historical_parallel_distributes_work
- test_compute_historical_parallel_handles_failures

"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml.stores.common.feature_computation import (
    FeatureComputationComponent,
    FeatureComputationConfig,
    FeatureComputationProtocol,
    FeatureSchemaProtocol,
)


# =========================================================================
# Mock Classes and Helpers
# =========================================================================


class MockBar:
    """
    Mock Nautilus Bar for testing.
    """

    def __init__(
        self,
        close: float = 100.0,
        high: float = 101.0,
        low: float = 99.0,
        volume: float = 1000.0,
        ts_event: int = 1700000000000000000,
        ts_init: int = 1700000000000000000,
        instrument_id: str = "SPY.DATABENTO",
    ) -> None:
        self.close = close
        self.high = high
        self.low = low
        self.volume = volume
        self.ts_event = ts_event
        self.ts_init = ts_init
        # Create bar_type with instrument_id
        self.bar_type = MagicMock()
        self.bar_type.instrument_id = instrument_id


class MockIndicatorManager:
    """
    Mock IndicatorManager for testing.
    """

    def __init__(self, initialized: bool = True) -> None:
        self._initialized = initialized
        self.update_calls: list[Any] = []

    def update_from_bar(self, bar: Any) -> None:
        self.update_calls.append(bar)

    def all_initialized(self) -> bool:
        return self._initialized


class MockPolarsDataFrame:
    """
    Mock Polars DataFrame for testing.
    """

    def __init__(
        self,
        data: dict[str, list[Any]] | None = None,
        empty: bool = False,
    ) -> None:
        self._data = data or {}
        self._empty = empty

    def is_empty(self) -> bool:
        return self._empty

    def __getitem__(self, key: str) -> Any:
        col = MagicMock()
        col.to_numpy.return_value = np.array(self._data.get(key, []))
        return col

    def iter_rows(self) -> list[tuple[Any, ...]]:
        if not self._data:
            return []
        # Return list of row tuples
        keys = list(self._data.keys())
        n_rows = len(self._data[keys[0]]) if keys else 0
        return [tuple(self._data[k][i] for k in keys) for i in range(n_rows)]


def create_mock_feature_engineer(
    online_features: np.ndarray | None = None,
    batch_features: Any | None = None,
) -> MagicMock:
    """
    Create a mock FeatureEngineer.
    """
    engineer = MagicMock()
    engineer.config = MagicMock()

    # Online computation
    if online_features is None:
        online_features = np.array([0.01, 1.5], dtype=np.float32)
    engineer.calculate_features_online.return_value = online_features

    # Batch computation
    if batch_features is None:
        batch_features = MockPolarsDataFrame(
            data={
                "close_return": [0.01, 0.02],
                "volume_ratio": [1.5, 1.6],
            },
        )
    engineer.calculate_features_batch.return_value = (batch_features, None)

    return engineer


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def mock_engine() -> MagicMock:
    """
    Create a mock SQLAlchemy engine.
    """
    engine = MagicMock()
    engine.dialect.name = "postgresql"
    conn_mock = MagicMock()
    engine.connect.return_value.__enter__ = MagicMock(return_value=conn_mock)
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    engine.begin.return_value.__enter__ = MagicMock(return_value=conn_mock)
    engine.begin.return_value.__exit__ = MagicMock(return_value=False)
    return engine


@pytest.fixture
def mock_table() -> MagicMock:
    """
    Create a mock SQLAlchemy table.
    """
    table = MagicMock()
    table.c = MagicMock()
    table.c.ts_event = MagicMock()
    table.c.feature_set_id = MagicMock()
    table.c.instrument_id = MagicMock()
    table.c.__getitem__ = MagicMock(return_value=MagicMock())
    return table


@pytest.fixture
def mock_feature_writer() -> MagicMock:
    """
    Create a mock FeatureWriterComponent.
    """
    return MagicMock()


@pytest.fixture
def mock_feature_reader() -> MagicMock:
    """
    Create a mock FeatureReaderComponent.
    """
    reader = MagicMock()
    reader.features_exist.return_value = False
    return reader


@pytest.fixture
def mock_feature_engineer() -> MagicMock:
    """
    Create a mock FeatureEngineer.
    """
    return create_mock_feature_engineer()


@pytest.fixture
def feature_computation(
    mock_engine: MagicMock,
    mock_table: MagicMock,
    mock_feature_writer: MagicMock,
    mock_feature_reader: MagicMock,
    mock_feature_engineer: MagicMock,
) -> FeatureComputationComponent:
    """
    Create a FeatureComputationComponent for testing.
    """
    return FeatureComputationComponent(
        engine=mock_engine,
        table=mock_table,
        feature_engineer=mock_feature_engineer,
        feature_writer=mock_feature_writer,
        feature_reader=mock_feature_reader,
        get_feature_set_id=lambda: "fs_test_001",
        get_feature_names=lambda: ["close_return", "volume_ratio"],
        get_feature_names_online=lambda: ["close_return", "volume_ratio"],
    )


# =========================================================================
# Protocol Compliance Tests
# =========================================================================


class TestFeatureComputationProtocol:
    """
    Test protocol compliance.
    """

    def test_component_satisfies_computation_protocol(
        self,
        feature_computation: FeatureComputationComponent,
    ) -> None:
        """
        Verify FeatureComputationComponent satisfies FeatureComputationProtocol.
        """
        assert isinstance(feature_computation, FeatureComputationProtocol)

    def test_schema_protocol_has_required_methods(self) -> None:
        """
        Verify FeatureSchemaProtocol has required methods.
        """
        # Verify the protocol defines the expected methods
        assert hasattr(FeatureSchemaProtocol, "get_feature_names")
        assert hasattr(FeatureSchemaProtocol, "get_feature_names_online")
        assert hasattr(FeatureSchemaProtocol, "get_feature_set_id")


# =========================================================================
# Configuration Tests
# =========================================================================


class TestFeatureComputationConfig:
    """
    Test configuration.
    """

    def test_default_config(self) -> None:
        """
        Test default configuration values.
        """
        config = FeatureComputationConfig()
        assert config.max_parallel_workers == 4
        assert config.default_lookback_days == 1

    def test_custom_config(self) -> None:
        """
        Test custom configuration values.
        """
        config = FeatureComputationConfig(
            max_parallel_workers=8,
            default_lookback_days=7,
        )
        assert config.max_parallel_workers == 8
        assert config.default_lookback_days == 7

    def test_config_validation_max_workers_too_low(self) -> None:
        """
        Test validation fails for max_parallel_workers < 1.
        """
        with pytest.raises(ValueError, match="max_parallel_workers must be >= 1"):
            FeatureComputationConfig(max_parallel_workers=0)

    def test_config_validation_max_workers_too_high(self) -> None:
        """
        Test validation fails for max_parallel_workers > 8.
        """
        with pytest.raises(ValueError, match="max_parallel_workers capped at 8"):
            FeatureComputationConfig(max_parallel_workers=16)

    def test_config_validation_lookback_days_too_low(self) -> None:
        """
        Test validation fails for default_lookback_days < 1.
        """
        with pytest.raises(ValueError, match="default_lookback_days must be >= 1"):
            FeatureComputationConfig(default_lookback_days=0)


# =========================================================================
# Happy Path Tests - compute_realtime
# =========================================================================


class TestComputeRealtime:
    """
    Test compute_realtime method (HOT PATH).
    """

    def test_compute_realtime_returns_feature_array(
        self,
        feature_computation: FeatureComputationComponent,
    ) -> None:
        """
        Verify real-time computation returns valid feature array.
        """
        bar = MockBar()

        # Create initialized indicator manager
        indicator_manager = MockIndicatorManager(initialized=True)

        features = feature_computation.compute_realtime(
            bar=bar,
            store=False,
            indicator_manager=indicator_manager,
        )

        # Verify return type and shape
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32
        assert features.size > 0

        # Verify indicator manager was updated
        assert len(indicator_manager.update_calls) == 1

    def test_compute_realtime_returns_empty_when_not_warmed(
        self,
        feature_computation: FeatureComputationComponent,
    ) -> None:
        """
        Verify empty array returned when indicators not warmed up.
        """
        bar = MockBar()

        # Create uninitialized indicator manager
        indicator_manager = MockIndicatorManager(initialized=False)

        features = feature_computation.compute_realtime(
            bar=bar,
            store=False,
            indicator_manager=indicator_manager,
        )

        # Should return empty array
        assert features.size == 0
        assert features.dtype == np.float32

    def test_compute_realtime_stores_when_enabled(
        self,
        feature_computation: FeatureComputationComponent,
        mock_engine: MagicMock,
    ) -> None:
        """
        Verify features are stored when store=True.
        """
        bar = MockBar()
        indicator_manager = MockIndicatorManager(initialized=True)

        # Patch the upsert method
        feature_computation._execute_realtime_upsert = MagicMock()

        features = feature_computation.compute_realtime(
            bar=bar,
            store=True,
            indicator_manager=indicator_manager,
        )

        # Verify store was called
        assert features.size > 0
        feature_computation._execute_realtime_upsert.assert_called_once()

        # Verify row contents
        call_args = feature_computation._execute_realtime_upsert.call_args[0]
        row = call_args[0]
        assert row["feature_set_id"] == "fs_test_001"
        assert row["is_live"] is True
        assert row["source"] == "live"

    def test_compute_realtime_does_not_store_when_disabled(
        self,
        feature_computation: FeatureComputationComponent,
    ) -> None:
        """
        Verify features are not stored when store=False.
        """
        bar = MockBar()
        indicator_manager = MockIndicatorManager(initialized=True)

        # Patch the upsert method
        feature_computation._execute_realtime_upsert = MagicMock()

        features = feature_computation.compute_realtime(
            bar=bar,
            store=False,
            indicator_manager=indicator_manager,
        )

        # Verify store was NOT called
        assert features.size > 0
        feature_computation._execute_realtime_upsert.assert_not_called()

    def test_compute_realtime_uses_indicator_manager(
        self,
        feature_computation: FeatureComputationComponent,
    ) -> None:
        """
        Verify provided indicator manager is used.
        """
        bar = MockBar()
        indicator_manager = MockIndicatorManager(initialized=True)

        feature_computation.compute_realtime(
            bar=bar,
            store=False,
            indicator_manager=indicator_manager,
        )

        # Verify indicator manager was updated with the bar
        assert len(indicator_manager.update_calls) == 1
        assert indicator_manager.update_calls[0] is bar

    def test_compute_realtime_creates_internal_indicator_manager(
        self,
        feature_computation: FeatureComputationComponent,
    ) -> None:
        """
        Verify internal indicator manager is created when not provided.
        """
        bar = MockBar()

        # Mock IndicatorManager creation
        with patch(
            "ml.stores.common.feature_computation.FeatureComputationComponent._get_or_create_indicator_manager",
        ) as mock_get_im:
            mock_im = MockIndicatorManager(initialized=True)
            mock_get_im.return_value = mock_im

            feature_computation.compute_realtime(
                bar=bar,
                store=False,
                indicator_manager=None,
            )

            # Verify internal manager was requested
            mock_get_im.assert_called_once()

    def test_compute_realtime_respects_circuit_breaker_on_store(
        self,
        feature_computation: FeatureComputationComponent,
    ) -> None:
        """
        Verify storage is skipped when circuit breaker is open.
        """
        bar = MockBar()
        indicator_manager = MockIndicatorManager(initialized=True)

        # Set up circuit breaker to block execution
        cb = MagicMock()
        cb.can_execute.return_value = False
        feature_computation.circuit_breaker = cb

        # Patch the upsert method
        feature_computation._execute_realtime_upsert = MagicMock()

        features = feature_computation.compute_realtime(
            bar=bar,
            store=True,
            indicator_manager=indicator_manager,
        )

        # Features should still be computed
        assert features.size > 0

        # But storage should be skipped
        feature_computation._execute_realtime_upsert.assert_not_called()


# =========================================================================
# Tests - compute_and_store_historical
# =========================================================================


class TestComputeAndStoreHistorical:
    """
    Test compute_and_store_historical method (COLD PATH).
    """

    def test_compute_historical_computes_batch_features(
        self,
        feature_computation: FeatureComputationComponent,
        mock_feature_reader: MagicMock,
        mock_feature_engineer: MagicMock,
    ) -> None:
        """
        Verify historical computation processes batch features.
        """
        # Setup mocks
        mock_feature_reader.features_exist.return_value = False

        bars_df = MockPolarsDataFrame(
            data={
                "ts_event": [1700000000000000000, 1700000001000000000],
                "close": [100.0, 101.0],
            },
        )
        feature_computation._load_bars_from_nautilus = MagicMock(return_value=bars_df)
        feature_computation._execute_historical_bulk_upsert = MagicMock()
        feature_computation._emit_historical_event = MagicMock()

        row_count = feature_computation.compute_and_store_historical(
            instrument_id="SPY.DATABENTO",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
        )

        # Verify features were computed
        mock_feature_engineer.calculate_features_batch.assert_called_once()

        # Verify bulk upsert was called
        feature_computation._execute_historical_bulk_upsert.assert_called_once()

        # Row count should match number of bars
        assert row_count == 2

    def test_compute_historical_skips_when_features_exist(
        self,
        feature_computation: FeatureComputationComponent,
        mock_feature_reader: MagicMock,
    ) -> None:
        """
        Verify computation is skipped when features already exist.
        """
        mock_feature_reader.features_exist.return_value = True

        row_count = feature_computation.compute_and_store_historical(
            instrument_id="SPY.DATABENTO",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
            force_recompute=False,
        )

        # Should return 0 since features exist
        assert row_count == 0

    def test_compute_historical_force_recompute_overwrites(
        self,
        feature_computation: FeatureComputationComponent,
        mock_feature_reader: MagicMock,
        mock_feature_engineer: MagicMock,
    ) -> None:
        """
        Verify force_recompute=True ignores existing features.
        """
        mock_feature_reader.features_exist.return_value = True

        bars_df = MockPolarsDataFrame(
            data={
                "ts_event": [1700000000000000000, 1700000001000000000],
                "close": [100.0, 101.0],
            },
        )
        feature_computation._load_bars_from_nautilus = MagicMock(return_value=bars_df)
        feature_computation._execute_historical_bulk_upsert = MagicMock()
        feature_computation._emit_historical_event = MagicMock()

        row_count = feature_computation.compute_and_store_historical(
            instrument_id="SPY.DATABENTO",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
            force_recompute=True,
        )

        # Features should still be computed despite features_exist returning True
        mock_feature_engineer.calculate_features_batch.assert_called_once()
        # Row count should match: 2 bars -> 2 feature rows
        assert row_count == 2

    def test_compute_historical_returns_zero_for_empty_bars(
        self,
        feature_computation: FeatureComputationComponent,
        mock_feature_reader: MagicMock,
    ) -> None:
        """
        Verify zero returned when no bars are found.
        """
        mock_feature_reader.features_exist.return_value = False

        empty_df = MockPolarsDataFrame(empty=True)
        feature_computation._load_bars_from_nautilus = MagicMock(return_value=empty_df)

        row_count = feature_computation.compute_and_store_historical(
            instrument_id="SPY.DATABENTO",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
        )

        assert row_count == 0


# =========================================================================
# Tests - compute_historical_parallel
# =========================================================================


class TestComputeHistoricalParallel:
    """
    Test compute_historical_parallel method.
    """

    def test_compute_historical_parallel_distributes_work(
        self,
        feature_computation: FeatureComputationComponent,
    ) -> None:
        """
        Verify parallel computation distributes work across instruments.
        """
        instruments = ["SPY.DATABENTO", "AAPL.DATABENTO"]

        # Mock compute_and_store_historical to return predictable results
        call_count = 0

        def mock_compute(*args: Any, **kwargs: Any) -> int:
            nonlocal call_count
            call_count += 1
            return 100

        feature_computation.compute_and_store_historical = mock_compute  # type: ignore[method-assign]

        results = feature_computation.compute_historical_parallel(
            instrument_ids=instruments,
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
            max_workers=2,
        )

        # Verify all instruments were processed
        assert len(results) == 2
        assert "SPY.DATABENTO" in results
        assert "AAPL.DATABENTO" in results
        assert call_count == 2

    def test_compute_historical_parallel_handles_failures(
        self,
        feature_computation: FeatureComputationComponent,
    ) -> None:
        """
        Verify parallel computation handles individual failures gracefully.
        """
        instruments = ["VALID.DATABENTO", "INVALID.DATABENTO"]

        def mock_compute(
            instrument_id: str,
            *args: Any,
            **kwargs: Any,
        ) -> int:
            if instrument_id == "INVALID.DATABENTO":
                raise ValueError("Simulated failure")
            return 100

        feature_computation.compute_and_store_historical = mock_compute  # type: ignore[method-assign]

        results = feature_computation.compute_historical_parallel(
            instrument_ids=instruments,
            max_workers=2,
        )

        # Valid instrument should succeed
        assert results["VALID.DATABENTO"] == 100

        # Invalid instrument should return 0 (not raise)
        assert results["INVALID.DATABENTO"] == 0

    def test_compute_historical_parallel_empty_list_returns_empty_dict(
        self,
        feature_computation: FeatureComputationComponent,
    ) -> None:
        """
        Verify empty instrument list returns empty dict.
        """
        results = feature_computation.compute_historical_parallel(
            instrument_ids=[],
        )
        assert results == {}

    def test_compute_historical_parallel_uses_default_dates(
        self,
        feature_computation: FeatureComputationComponent,
    ) -> None:
        """
        Verify default start/end dates are applied when not provided.
        """
        call_args: list[dict[str, Any]] = []

        def mock_compute(
            instrument_id: str,
            start: datetime,
            end: datetime,
            **kwargs: Any,
        ) -> int:
            call_args.append({"start": start, "end": end})
            return 1

        feature_computation.compute_and_store_historical = mock_compute  # type: ignore[method-assign]

        feature_computation.compute_historical_parallel(
            instrument_ids=["TEST.DATABENTO"],
            start=None,
            end=None,
        )

        # Verify defaults were applied
        assert len(call_args) == 1
        # End should be recent (within last minute)
        assert (datetime.now(UTC) - call_args[0]["end"]).total_seconds() < 60
        # Start should be default_lookback_days before end
        delta = call_args[0]["end"] - call_args[0]["start"]
        assert delta == timedelta(days=feature_computation.config.default_lookback_days)

    def test_compute_historical_parallel_caps_workers(
        self,
        feature_computation: FeatureComputationComponent,
    ) -> None:
        """
        Verify max_workers is capped to config maximum.
        """
        # Set config with max_parallel_workers=4
        from dataclasses import replace

        feature_computation.config = FeatureComputationConfig(max_parallel_workers=4)

        # Track what max_workers value was actually used
        workers_used: list[int] = []

        def mock_compute(*args: Any, **kwargs: Any) -> int:
            return 1

        feature_computation.compute_and_store_historical = mock_compute  # type: ignore[method-assign]

        # Patch ThreadPoolExecutor to capture the max_workers argument
        original_executor = __import__("concurrent.futures").futures.ThreadPoolExecutor

        class RecordingExecutor:
            def __init__(self, max_workers: int = 4):
                workers_used.append(max_workers)
                self._executor = original_executor(max_workers=max_workers)

            def __enter__(self) -> Any:
                return self._executor.__enter__()

            def __exit__(self, *args: Any) -> Any:
                return self._executor.__exit__(*args)

        with patch(
            "ml.stores.common.feature_computation.ThreadPoolExecutor",
            RecordingExecutor,
        ):
            feature_computation.compute_historical_parallel(
                instrument_ids=["TEST.DATABENTO"],
                max_workers=100,  # Request way more than allowed
            )

        # Should be capped at config max (4)
        assert len(workers_used) == 1
        assert workers_used[0] == 4


# =========================================================================
# Helper Method Tests
# =========================================================================


class TestHelperMethods:
    """
    Test helper methods.
    """

    def test_get_instrument_key_from_bar_type(
        self,
        feature_computation: FeatureComputationComponent,
    ) -> None:
        """
        Verify instrument key extraction from bar_type.
        """
        bar = MockBar(instrument_id="SPY.DATABENTO")

        key = feature_computation._get_instrument_key(bar)

        assert key == "SPY.DATABENTO"

    def test_get_instrument_key_fallback(
        self,
        feature_computation: FeatureComputationComponent,
    ) -> None:
        """
        Verify instrument key fallback when bar_type not available.
        """
        bar = MagicMock()
        del bar.bar_type  # Remove bar_type
        bar.instrument_id = "AAPL.DATABENTO"

        key = feature_computation._get_instrument_key(bar)

        assert key == "AAPL.DATABENTO"

    def test_get_or_create_indicator_manager_creates_new(
        self,
        feature_computation: FeatureComputationComponent,
    ) -> None:
        """
        Verify new indicator manager is created when not exists.
        """
        with patch(
            "ml.features.engineering.IndicatorManager",
        ) as mock_im_class:
            mock_im = MagicMock()
            mock_im_class.return_value = mock_im

            result = feature_computation._get_or_create_indicator_manager(
                "NEW_INSTRUMENT",
            )

            # Should create new manager
            mock_im_class.assert_called_once()
            assert feature_computation._indicator_managers["NEW_INSTRUMENT"] is result

    def test_get_or_create_indicator_manager_reuses_existing(
        self,
        feature_computation: FeatureComputationComponent,
    ) -> None:
        """
        Verify existing indicator manager is reused.
        """
        existing_im = MagicMock()
        feature_computation._indicator_managers["EXISTING"] = existing_im

        result = feature_computation._get_or_create_indicator_manager("EXISTING")

        assert result is existing_im

    def test_get_indicator_manager_returns_none_for_unknown(
        self,
        feature_computation: FeatureComputationComponent,
    ) -> None:
        """
        Verify None returned for unknown instrument.
        """
        result = feature_computation.get_indicator_manager("UNKNOWN")
        assert result is None

    def test_set_data_registry(
        self,
        feature_computation: FeatureComputationComponent,
    ) -> None:
        """
        Verify data registry can be set.
        """
        registry = MagicMock()

        feature_computation.set_data_registry(registry)

        assert feature_computation.data_registry is registry


# =========================================================================
# Event Emission Tests
# =========================================================================


class TestEventEmission:
    """
    Test event emission methods.
    """

    def test_emit_realtime_event_calls_registry(
        self,
        feature_computation: FeatureComputationComponent,
    ) -> None:
        """
        Verify realtime event is emitted when registry is set.
        """
        registry = MagicMock()
        feature_computation.data_registry = registry

        bar = MockBar()

        with patch(
            "ml.stores.common.feature_computation.emit_dataset_event_and_watermark",
        ) as mock_emit:
            feature_computation._emit_realtime_event(bar, "SPY.DATABENTO")

            mock_emit.assert_called_once()
            call_kwargs = mock_emit.call_args[1]
            assert call_kwargs["dataset_id"] == "features"
            assert call_kwargs["instrument_id"] == "SPY.DATABENTO"

    def test_emit_realtime_event_noop_without_registry(
        self,
        feature_computation: FeatureComputationComponent,
    ) -> None:
        """
        Verify no event emitted when registry is None.
        """
        feature_computation.data_registry = None

        bar = MockBar()

        with patch(
            "ml.stores.common.feature_computation.emit_dataset_event_and_watermark",
        ) as mock_emit:
            feature_computation._emit_realtime_event(bar, "SPY.DATABENTO")

            mock_emit.assert_not_called()

    def test_emit_realtime_event_handles_exceptions_gracefully(
        self,
        feature_computation: FeatureComputationComponent,
    ) -> None:
        """
        Verify realtime event emission handles exceptions gracefully.
        """
        registry = MagicMock()
        feature_computation.data_registry = registry

        bar = MockBar()

        with patch(
            "ml.stores.common.feature_computation.emit_dataset_event_and_watermark",
            side_effect=Exception("Test error"),
        ):
            # Should not raise
            feature_computation._emit_realtime_event(bar, "SPY.DATABENTO")

    def test_emit_historical_event_calls_registry(
        self,
        feature_computation: FeatureComputationComponent,
    ) -> None:
        """
        Verify historical event is emitted when registry is set.
        """
        registry = MagicMock()
        feature_computation.data_registry = registry

        timestamps = np.array([1700000000000000000, 1700000001000000000], dtype=np.int64)

        with patch(
            "ml.stores.common.feature_computation.emit_dataset_event_and_watermark",
        ) as mock_emit:
            feature_computation._emit_historical_event(
                "SPY.DATABENTO",
                timestamps,
                row_count=2,
            )

            mock_emit.assert_called_once()
            call_kwargs = mock_emit.call_args[1]
            assert call_kwargs["count"] == 2

    def test_emit_historical_event_noop_without_registry(
        self,
        feature_computation: FeatureComputationComponent,
    ) -> None:
        """
        Verify no event emitted when registry is None.
        """
        feature_computation.data_registry = None

        timestamps = np.array([1700000000000000000], dtype=np.int64)

        with patch(
            "ml.stores.common.feature_computation.emit_dataset_event_and_watermark",
        ) as mock_emit:
            feature_computation._emit_historical_event(
                "SPY.DATABENTO",
                timestamps,
                row_count=1,
            )

            mock_emit.assert_not_called()


# =========================================================================
# Circuit Breaker Integration Tests
# =========================================================================


class TestCircuitBreakerIntegration:
    """
    Test circuit breaker integration.
    """

    def test_circuit_breaker_records_success_on_store(
        self,
        feature_computation: FeatureComputationComponent,
    ) -> None:
        """
        Verify circuit breaker records success after successful store.
        """
        bar = MockBar()
        indicator_manager = MockIndicatorManager(initialized=True)

        cb = MagicMock()
        cb.can_execute.return_value = True
        feature_computation.circuit_breaker = cb

        feature_computation._execute_realtime_upsert = MagicMock()

        feature_computation.compute_realtime(
            bar=bar,
            store=True,
            indicator_manager=indicator_manager,
        )

        cb.record_success.assert_called_once()

    def test_circuit_breaker_records_failure_on_store_error(
        self,
        feature_computation: FeatureComputationComponent,
    ) -> None:
        """
        Verify circuit breaker records failure after failed store.
        """
        bar = MockBar()
        indicator_manager = MockIndicatorManager(initialized=True)

        cb = MagicMock()
        cb.can_execute.return_value = True
        feature_computation.circuit_breaker = cb

        feature_computation._execute_realtime_upsert = MagicMock(
            side_effect=Exception("DB error"),
        )

        with pytest.raises(Exception, match="DB error"):
            feature_computation.compute_realtime(
                bar=bar,
                store=True,
                indicator_manager=indicator_manager,
            )

        cb.record_failure.assert_called_once()
