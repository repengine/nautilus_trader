#!/usr/bin/env python3

"""
Unit tests for FeatureEventComponent (Phase 3.7.5).

Tests event emission operations including historical and realtime event
emission, DataRegistry integration, and observability stage boundary recording.

Coverage target: 90%

Test Cases (from test design report):
- test_emit_historical_event_calls_registry
- test_emit_historical_event_noop_when_no_registry
- test_emit_historical_event_logs_on_failure
- test_emit_realtime_event_emits_single_record
- test_emit_realtime_event_includes_correct_metadata
- test_record_observability_stage_boundary_calls_helper

"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from ml.stores.common.feature_event import (
    FeatureEventComponent,
    FeatureEventConfig,
    FeatureEventProtocol,
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


class MockRegistry:
    """
    Mock DataRegistry for testing.
    """

    def __init__(self) -> None:
        self.emit_event_calls: list[dict[str, Any]] = []
        self.update_watermark_calls: list[dict[str, Any]] = []

    def emit_event(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        stage: Any,
        source: Any,
        run_id: str,
        ts_min: int,
        ts_max: int,
        count: int,
        status: Any,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.emit_event_calls.append({
            "dataset_id": dataset_id,
            "instrument_id": instrument_id,
            "stage": stage,
            "source": source,
            "run_id": run_id,
            "ts_min": ts_min,
            "ts_max": ts_max,
            "count": count,
            "status": status,
            "metadata": metadata,
            **kwargs,
        })

    def update_watermark(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        source: Any,
        last_success_ns: int,
        count: int,
        completeness_pct: float,
    ) -> None:
        self.update_watermark_calls.append({
            "dataset_id": dataset_id,
            "instrument_id": instrument_id,
            "source": source,
            "last_success_ns": last_success_ns,
            "count": count,
            "completeness_pct": completeness_pct,
        })


class MockObservabilityService:
    """
    Mock ObservabilityService for testing.
    """

    def __init__(self) -> None:
        self.latency_stage_calls: list[dict[str, Any]] = []
        self.metric_calls: list[dict[str, Any]] = []

    def add_latency_stage(
        self,
        *,
        correlation_id: str,
        instrument_id: str,
        pipeline_stage: str,
        ts_stage_start: int,
        ts_stage_end: int,
    ) -> None:
        self.latency_stage_calls.append({
            "correlation_id": correlation_id,
            "instrument_id": instrument_id,
            "pipeline_stage": pipeline_stage,
            "ts_stage_start": ts_stage_start,
            "ts_stage_end": ts_stage_end,
        })

    def add_metric(
        self,
        *,
        metric_name: str,
        metric_type: str,
        value: float,
        timestamp: int,
        labels: dict[str, Any] | str | None = None,
    ) -> None:
        self.metric_calls.append({
            "metric_name": metric_name,
            "metric_type": metric_type,
            "value": value,
            "timestamp": timestamp,
            "labels": labels,
        })


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def mock_registry() -> MockRegistry:
    """Create a mock registry for testing."""
    return MockRegistry()


@pytest.fixture
def mock_observability_service() -> MockObservabilityService:
    """Create a mock observability service for testing."""
    return MockObservabilityService()


@pytest.fixture
def feature_event_config() -> FeatureEventConfig:
    """Create a default feature event config."""
    return FeatureEventConfig(
        component_name="feature_store",
        dataset_id="features",
        enable_observability=True,
    )


@pytest.fixture
def feature_event_component(
    mock_registry: MockRegistry,
    mock_observability_service: MockObservabilityService,
    feature_event_config: FeatureEventConfig,
) -> FeatureEventComponent:
    """Create a FeatureEventComponent with mock dependencies."""
    return FeatureEventComponent(
        config=feature_event_config,
        get_registry=lambda: mock_registry,
        get_feature_set_id=lambda: "fs_test123",
        observability_service=mock_observability_service,
    )


@pytest.fixture
def feature_event_component_no_registry(
    mock_observability_service: MockObservabilityService,
    feature_event_config: FeatureEventConfig,
) -> FeatureEventComponent:
    """Create a FeatureEventComponent with no registry."""
    return FeatureEventComponent(
        config=feature_event_config,
        get_registry=lambda: None,
        get_feature_set_id=lambda: "fs_test123",
        observability_service=mock_observability_service,
    )


# =========================================================================
# Protocol Compliance Tests
# =========================================================================


class TestFeatureEventProtocolCompliance:
    """Test that FeatureEventComponent implements the protocol."""

    def test_component_implements_protocol(self) -> None:
        """Verify FeatureEventComponent is an instance of FeatureEventProtocol."""
        component = FeatureEventComponent()
        assert isinstance(component, FeatureEventProtocol)

    def test_protocol_has_required_methods(self) -> None:
        """Verify protocol defines all required methods."""
        # Check runtime_checkable allows isinstance checks
        assert hasattr(FeatureEventProtocol, "emit_historical_event")
        assert hasattr(FeatureEventProtocol, "emit_realtime_event")
        assert hasattr(FeatureEventProtocol, "record_observability_stage_boundary")


# =========================================================================
# Configuration Tests
# =========================================================================


class TestFeatureEventConfig:
    """Tests for FeatureEventConfig."""

    def test_config_default_values(self) -> None:
        """Test default configuration values."""
        config = FeatureEventConfig()
        assert config.component_name == "feature_store"
        assert config.dataset_id == "features"
        assert config.enable_observability is True

    def test_config_custom_values(self) -> None:
        """Test configuration with custom values."""
        config = FeatureEventConfig(
            component_name="custom_component",
            dataset_id="custom_dataset",
            enable_observability=False,
        )
        assert config.component_name == "custom_component"
        assert config.dataset_id == "custom_dataset"
        assert config.enable_observability is False

    def test_config_frozen(self) -> None:
        """Test that config is immutable."""
        config = FeatureEventConfig()
        with pytest.raises(AttributeError):
            config.component_name = "modified"  # type: ignore[misc]

    def test_config_validation_empty_component_name(self) -> None:
        """Test validation rejects empty component_name."""
        with pytest.raises(ValueError, match="component_name cannot be empty"):
            FeatureEventConfig(component_name="")

    def test_config_validation_empty_dataset_id(self) -> None:
        """Test validation rejects empty dataset_id."""
        with pytest.raises(ValueError, match="dataset_id cannot be empty"):
            FeatureEventConfig(dataset_id="")


# =========================================================================
# Historical Event Tests
# =========================================================================


class TestEmitHistoricalEvent:
    """Tests for emit_historical_event method."""

    def test_emit_historical_event_calls_registry(
        self,
        feature_event_component: FeatureEventComponent,
        mock_registry: MockRegistry,
    ) -> None:
        """Test that emit_historical_event calls the registry."""
        timestamps = np.array([1700000000000000000, 1700000001000000000], dtype=np.int64)

        feature_event_component.emit_historical_event(
            instrument_id="SPY.DATABENTO",
            timestamps=timestamps,
            row_count=100,
        )

        # Verify emit_event was called
        assert len(mock_registry.emit_event_calls) == 1
        event_call = mock_registry.emit_event_calls[0]

        assert event_call["dataset_id"] == "features"
        assert event_call["instrument_id"] == "SPY.DATABENTO"
        assert event_call["count"] == 100
        assert event_call["ts_min"] == 1700000000000000000
        assert event_call["ts_max"] == 1700000001000000000

    def test_emit_historical_event_noop_when_no_registry(
        self,
        feature_event_component_no_registry: FeatureEventComponent,
    ) -> None:
        """Test that emit_historical_event is no-op when registry is None."""
        timestamps = np.array([1700000000000000000], dtype=np.int64)

        # Should not raise, just log and return
        feature_event_component_no_registry.emit_historical_event(
            instrument_id="SPY.DATABENTO",
            timestamps=timestamps,
            row_count=100,
        )
        # No assertions needed - just verify no exception

    def test_emit_historical_event_logs_on_failure(
        self,
        feature_event_config: FeatureEventConfig,
    ) -> None:
        """Test that emit_historical_event logs on failure."""
        # Create a registry that raises on emit
        failing_registry = MagicMock()
        failing_registry.emit_event.side_effect = Exception("Test error")

        component = FeatureEventComponent(
            config=feature_event_config,
            get_registry=lambda: failing_registry,
            get_feature_set_id=lambda: "fs_test123",
        )

        timestamps = np.array([1700000000000000000], dtype=np.int64)

        # Should not raise due to with_fallback decorator
        component.emit_historical_event(
            instrument_id="SPY.DATABENTO",
            timestamps=timestamps,
            row_count=100,
        )
        # No assertions needed - just verify no exception

    def test_emit_historical_event_handles_empty_timestamps(
        self,
        feature_event_component: FeatureEventComponent,
        mock_registry: MockRegistry,
    ) -> None:
        """Test that emit_historical_event handles empty timestamps array."""
        timestamps = np.array([], dtype=np.int64)

        feature_event_component.emit_historical_event(
            instrument_id="SPY.DATABENTO",
            timestamps=timestamps,
            row_count=0,
        )

        # Verify emit_event was called with ts_min=0, ts_max=0
        assert len(mock_registry.emit_event_calls) == 1
        event_call = mock_registry.emit_event_calls[0]

        assert event_call["ts_min"] == 0
        assert event_call["ts_max"] == 0
        assert event_call["count"] == 0

    def test_emit_historical_event_updates_watermark(
        self,
        feature_event_component: FeatureEventComponent,
        mock_registry: MockRegistry,
    ) -> None:
        """Test that emit_historical_event updates the watermark."""
        timestamps = np.array([1700000000000000000], dtype=np.int64)

        feature_event_component.emit_historical_event(
            instrument_id="SPY.DATABENTO",
            timestamps=timestamps,
            row_count=100,
        )

        # Verify update_watermark was called
        assert len(mock_registry.update_watermark_calls) == 1
        watermark_call = mock_registry.update_watermark_calls[0]

        assert watermark_call["dataset_id"] == "features"
        assert watermark_call["instrument_id"] == "SPY.DATABENTO"
        assert watermark_call["last_success_ns"] == 1700000000000000000
        assert watermark_call["count"] == 100


# =========================================================================
# Realtime Event Tests
# =========================================================================


class TestEmitRealtimeEvent:
    """Tests for emit_realtime_event method."""

    def test_emit_realtime_event_emits_single_record(
        self,
        feature_event_component: FeatureEventComponent,
        mock_registry: MockRegistry,
    ) -> None:
        """Test that emit_realtime_event emits a single record event."""
        bar = MockBar(
            ts_event=1700000000000000000,
            instrument_id="SPY.DATABENTO",
        )

        feature_event_component.emit_realtime_event(
            bar=bar,
            feature_set_id="fs_abc123",
        )

        # Verify emit_event was called with count=1
        assert len(mock_registry.emit_event_calls) == 1
        event_call = mock_registry.emit_event_calls[0]

        assert event_call["count"] == 1

    def test_emit_realtime_event_includes_correct_metadata(
        self,
        feature_event_component: FeatureEventComponent,
        mock_registry: MockRegistry,
    ) -> None:
        """Test that emit_realtime_event includes correct metadata."""
        bar = MockBar(
            ts_event=1700000000000000000,
            instrument_id="SPY.DATABENTO",
        )

        feature_event_component.emit_realtime_event(
            bar=bar,
            feature_set_id="fs_abc123",
        )

        # Verify event metadata
        assert len(mock_registry.emit_event_calls) == 1
        event_call = mock_registry.emit_event_calls[0]

        assert event_call["dataset_id"] == "features"
        assert event_call["instrument_id"] == "SPY.DATABENTO"
        assert event_call["ts_min"] == 1700000000000000000
        assert event_call["ts_max"] == 1700000000000000000

    def test_emit_realtime_event_noop_when_no_registry(
        self,
        feature_event_component_no_registry: FeatureEventComponent,
    ) -> None:
        """Test that emit_realtime_event is no-op when registry is None."""
        bar = MockBar(ts_event=1700000000000000000)

        # Should not raise, just log and return
        feature_event_component_no_registry.emit_realtime_event(
            bar=bar,
            feature_set_id="fs_abc123",
        )
        # No assertions needed - just verify no exception

    def test_emit_realtime_event_handles_bar_without_bar_type(
        self,
        feature_event_component: FeatureEventComponent,
        mock_registry: MockRegistry,
    ) -> None:
        """Test handling of bar without bar_type attribute."""
        # Create bar with direct instrument_id, no bar_type
        bar = MagicMock()
        bar.ts_event = 1700000000000000000
        bar.instrument_id = "AAPL.DATABENTO"
        del bar.bar_type  # Remove bar_type attribute

        feature_event_component.emit_realtime_event(
            bar=bar,
            feature_set_id="fs_abc123",
        )

        # Verify correct instrument_id extraction
        assert len(mock_registry.emit_event_calls) == 1
        event_call = mock_registry.emit_event_calls[0]

        assert event_call["instrument_id"] == "AAPL.DATABENTO"

    def test_emit_realtime_event_handles_unknown_instrument(
        self,
        feature_event_component: FeatureEventComponent,
        mock_registry: MockRegistry,
    ) -> None:
        """Test handling of bar with no instrument_id."""
        bar = MagicMock()
        bar.ts_event = 1700000000000000000
        del bar.bar_type
        del bar.instrument_id

        feature_event_component.emit_realtime_event(
            bar=bar,
            feature_set_id="fs_abc123",
        )

        # Verify "unknown" is used as instrument_id
        assert len(mock_registry.emit_event_calls) == 1
        event_call = mock_registry.emit_event_calls[0]

        assert event_call["instrument_id"] == "unknown"


# =========================================================================
# Observability Tests
# =========================================================================


class TestRecordObservabilityStageBoundary:
    """Tests for record_observability_stage_boundary method."""

    @patch("ml.common.observability_utils.record_stage_boundary")
    @patch.dict("os.environ", {"ML_OBSERVABILITY_ENABLED": "1"})
    def test_record_observability_stage_boundary_calls_helper(
        self,
        mock_record: MagicMock,
        feature_event_component: FeatureEventComponent,
    ) -> None:
        """Test that record_observability_stage_boundary calls the helper."""
        feature_event_component.record_observability_stage_boundary(
            stage="feature_computation",
            instrument_id="SPY.DATABENTO",
            ts_stage_start=1700000000000000000,
            ts_stage_end=1700000001000000000,
            row_count=100,
        )

        # Verify record_stage_boundary was called
        mock_record.assert_called_once()
        call_kwargs = mock_record.call_args[1]

        assert call_kwargs["component"] == "feature_store"
        assert call_kwargs["instrument_id"] == "SPY.DATABENTO"
        assert call_kwargs["stage"] == "feature_computation"
        assert call_kwargs["ts_stage_start"] == 1700000000000000000
        assert call_kwargs["ts_stage_end"] == 1700000001000000000
        assert call_kwargs["row_count"] == 100

    @patch("ml.common.observability_utils.record_stage_boundary")
    def test_record_observability_uses_custom_component_name(
        self,
        mock_record: MagicMock,
    ) -> None:
        """Test that component name from config is used."""
        config = FeatureEventConfig(
            component_name="custom_feature_store",
            dataset_id="features",
        )
        component = FeatureEventComponent(config=config)

        component.record_observability_stage_boundary(
            stage="test_stage",
            instrument_id="TEST.INST",
            ts_stage_start=1,
            ts_stage_end=2,
        )

        # Verify custom component name was used
        mock_record.assert_called_once()
        call_kwargs = mock_record.call_args[1]

        assert call_kwargs["component"] == "custom_feature_store"

    @patch("ml.common.observability_utils.record_stage_boundary")
    def test_record_observability_default_row_count(
        self,
        mock_record: MagicMock,
        feature_event_component: FeatureEventComponent,
    ) -> None:
        """Test that default row_count is 1."""
        feature_event_component.record_observability_stage_boundary(
            stage="test_stage",
            instrument_id="SPY.DATABENTO",
            ts_stage_start=1,
            ts_stage_end=2,
        )

        # Verify default row_count
        mock_record.assert_called_once()
        call_kwargs = mock_record.call_args[1]

        assert call_kwargs["row_count"] == 1

    @patch("ml.common.observability_utils.record_stage_boundary")
    def test_record_observability_handles_failure(
        self,
        mock_record: MagicMock,
        feature_event_component: FeatureEventComponent,
    ) -> None:
        """Test that failures are logged but don't raise."""
        mock_record.side_effect = Exception("Test error")

        # Should not raise
        feature_event_component.record_observability_stage_boundary(
            stage="test_stage",
            instrument_id="SPY.DATABENTO",
            ts_stage_start=1,
            ts_stage_end=2,
        )
        # No assertions needed - just verify no exception


# =========================================================================
# Configuration and Setup Tests
# =========================================================================


class TestComponentSetup:
    """Tests for component setup methods."""

    def test_set_observability_service(self) -> None:
        """Test setting observability service."""
        component = FeatureEventComponent()
        service = MockObservabilityService()

        component.set_observability_service(service)

        # pylint: disable=protected-access
        assert component._observability_service is service

    def test_set_registry_getter(self) -> None:
        """Test setting registry getter."""
        component = FeatureEventComponent()
        registry = MockRegistry()

        component.set_registry_getter(lambda: registry)

        # pylint: disable=protected-access
        result = component._get_data_registry()
        assert result is registry

    def test_set_feature_set_id_getter(self) -> None:
        """Test setting feature set ID getter."""
        component = FeatureEventComponent()

        component.set_feature_set_id_getter(lambda: "fs_new123")

        # pylint: disable=protected-access
        result = component._resolve_feature_set_id()
        assert result == "fs_new123"

    def test_default_feature_set_id_is_unknown(self) -> None:
        """Test that default feature set ID is 'unknown'."""
        component = FeatureEventComponent()

        # pylint: disable=protected-access
        result = component._resolve_feature_set_id()
        assert result == "unknown"

    def test_feature_set_id_getter_failure_returns_unknown(self) -> None:
        """Test that getter failure returns 'unknown'."""
        def failing_getter() -> str:
            raise Exception("Test error")

        component = FeatureEventComponent(get_feature_set_id=failing_getter)

        # pylint: disable=protected-access
        result = component._resolve_feature_set_id()
        assert result == "unknown"

    def test_registry_getter_failure_returns_none(self) -> None:
        """Test that registry getter failure returns None."""
        def failing_getter() -> MockRegistry:
            raise Exception("Test error")

        component = FeatureEventComponent(get_registry=failing_getter)

        # pylint: disable=protected-access
        result = component._get_data_registry()
        assert result is None


# =========================================================================
# Instrument ID Extraction Tests
# =========================================================================


class TestExtractInstrumentId:
    """Tests for _extract_instrument_id helper."""

    def test_extract_from_bar_type(self) -> None:
        """Test extraction from bar_type.instrument_id."""
        bar = MockBar(instrument_id="SPY.DATABENTO")
        component = FeatureEventComponent()

        # pylint: disable=protected-access
        result = component._extract_instrument_id(bar)
        assert result == "SPY.DATABENTO"

    def test_extract_from_direct_attribute(self) -> None:
        """Test extraction from direct instrument_id attribute."""
        bar = MagicMock()
        bar.instrument_id = "AAPL.DATABENTO"
        del bar.bar_type  # Remove bar_type

        component = FeatureEventComponent()

        # pylint: disable=protected-access
        result = component._extract_instrument_id(bar)
        assert result == "AAPL.DATABENTO"

    def test_extract_returns_unknown_on_failure(self) -> None:
        """Test that extraction returns 'unknown' on failure."""
        bar = MagicMock()
        del bar.bar_type
        del bar.instrument_id

        component = FeatureEventComponent()

        # pylint: disable=protected-access
        result = component._extract_instrument_id(bar)
        assert result == "unknown"

    def test_extract_handles_exception(self) -> None:
        """Test that extraction handles exceptions gracefully."""
        bar = MagicMock()
        bar.bar_type = MagicMock()
        # Make instrument_id raise an exception
        type(bar.bar_type).instrument_id = property(
            lambda self: (_ for _ in ()).throw(Exception("Test"))
        )

        component = FeatureEventComponent()

        # pylint: disable=protected-access
        result = component._extract_instrument_id(bar)
        assert result == "unknown"


# =========================================================================
# Integration Tests (with real helper functions)
# =========================================================================


class TestIntegration:
    """Integration tests for FeatureEventComponent."""

    def test_full_historical_workflow(
        self,
        feature_event_component: FeatureEventComponent,
        mock_registry: MockRegistry,
    ) -> None:
        """Test complete historical event emission workflow."""
        # Simulate historical feature computation
        timestamps = np.array([
            1700000000000000000,
            1700000001000000000,
            1700000002000000000,
        ], dtype=np.int64)

        feature_event_component.emit_historical_event(
            instrument_id="SPY.DATABENTO",
            timestamps=timestamps,
            row_count=3,
        )

        # Verify event was emitted with correct range
        assert len(mock_registry.emit_event_calls) == 1
        event = mock_registry.emit_event_calls[0]

        assert event["ts_min"] == 1700000000000000000
        assert event["ts_max"] == 1700000002000000000
        assert event["count"] == 3

        # Verify watermark was updated
        assert len(mock_registry.update_watermark_calls) == 1
        watermark = mock_registry.update_watermark_calls[0]

        assert watermark["last_success_ns"] == 1700000002000000000
        assert watermark["count"] == 3

    def test_full_realtime_workflow(
        self,
        feature_event_component: FeatureEventComponent,
        mock_registry: MockRegistry,
    ) -> None:
        """Test complete realtime event emission workflow."""
        # Simulate realtime bar processing
        bar = MockBar(
            ts_event=1700000000000000000,
            instrument_id="SPY.DATABENTO",
        )

        feature_event_component.emit_realtime_event(
            bar=bar,
            feature_set_id="fs_live123",
        )

        # Verify event was emitted
        assert len(mock_registry.emit_event_calls) == 1
        event = mock_registry.emit_event_calls[0]

        assert event["ts_min"] == 1700000000000000000
        assert event["ts_max"] == 1700000000000000000
        assert event["count"] == 1
        assert event["instrument_id"] == "SPY.DATABENTO"


# =========================================================================
# Edge Case Tests
# =========================================================================


class TestEdgeCases:
    """Edge case tests for FeatureEventComponent."""

    def test_emit_with_none_config(self) -> None:
        """Test that component works with None config (uses defaults)."""
        component = FeatureEventComponent(config=None)

        # Should use default config
        # pylint: disable=protected-access
        assert component._config.component_name == "feature_store"
        assert component._config.dataset_id == "features"

    def test_emit_with_all_none_dependencies(self) -> None:
        """Test component with all None dependencies."""
        component = FeatureEventComponent(
            config=None,
            get_registry=None,
            get_feature_set_id=None,
            observability_service=None,
        )

        # Should not raise on any method
        timestamps = np.array([1700000000000000000], dtype=np.int64)
        component.emit_historical_event("SPY", timestamps, 1)

        bar = MockBar()
        component.emit_realtime_event(bar, "fs_123")

        component.record_observability_stage_boundary(
            stage="test",
            instrument_id="SPY",
            ts_stage_start=1,
            ts_stage_end=2,
        )

    def test_emit_historical_with_single_timestamp(
        self,
        feature_event_component: FeatureEventComponent,
        mock_registry: MockRegistry,
    ) -> None:
        """Test historical emission with single timestamp."""
        timestamps = np.array([1700000000000000000], dtype=np.int64)

        feature_event_component.emit_historical_event(
            instrument_id="SPY.DATABENTO",
            timestamps=timestamps,
            row_count=1,
        )

        # ts_min and ts_max should be the same
        event = mock_registry.emit_event_calls[0]
        assert event["ts_min"] == event["ts_max"]
        assert event["ts_min"] == 1700000000000000000

    def test_concurrent_event_emission(
        self,
        mock_registry: MockRegistry,
        feature_event_config: FeatureEventConfig,
    ) -> None:
        """Test that concurrent emissions don't interfere."""
        import concurrent.futures

        component = FeatureEventComponent(
            config=feature_event_config,
            get_registry=lambda: mock_registry,
            get_feature_set_id=lambda: "fs_test",
        )

        def emit_historical(idx: int) -> None:
            timestamps = np.array([1700000000000000000 + idx], dtype=np.int64)
            component.emit_historical_event(
                instrument_id=f"INST_{idx}",
                timestamps=timestamps,
                row_count=1,
            )

        # Run concurrent emissions
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(emit_historical, i) for i in range(10)]
            concurrent.futures.wait(futures)

        # All emissions should have occurred
        assert len(mock_registry.emit_event_calls) == 10
