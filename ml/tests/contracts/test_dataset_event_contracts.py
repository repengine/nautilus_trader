#!/usr/bin/env python3

"""
Contract tests for dataset event emission consolidation.

These tests validate that all dataset event emission follows consistent patterns:
1. Event shapes and required fields
2. Enum usage and value constraints
3. Metrics labeling consistency
4. Correlation ID attachment
5. Watermark behavior

"""

from __future__ import annotations

import time
from types import ModuleType
from unittest.mock import MagicMock
from typing import TYPE_CHECKING, Any, Dict, List, cast

import pytest

from ml.common.correlation import make_correlation_id
from ml.common.event_emitter import emit_dataset_event, emit_dataset_event_and_watermark
from ml.config.events import EventStatus, Source, Stage
from ml.registry.protocols import RegistryProtocol
if TYPE_CHECKING:
    from ml.stores.data_store import DataStore
else:
    DataStore = Any  # pragma: no cover - runtime fallback for imports


def _get_data_store_cls(module: ModuleType) -> type[DataStore]:
    """Return the DataStore class from the dynamically loaded module."""
    return cast("type[DataStore]", getattr(module, "DataStore"))


class TestDatasetEventContracts:
    """
    Contract tests for dataset event emission patterns.
    """

    @pytest.fixture
    def mock_registry(self) -> MagicMock:
        """
        Mock registry for testing event emission.
        """
        registry = MagicMock(spec=RegistryProtocol)
        return registry

    @pytest.fixture
    def data_store(
        self,
        mock_registry: MagicMock,
        datastore_module: ModuleType,
    ) -> DataStore:
        """
        Data store with mocked registry for testing.
        """
        data_store_cls = _get_data_store_cls(datastore_module)
        return data_store_cls(
            connection_string="sqlite:///:memory:",
            registry=mock_registry,
            publisher=None,  # Disable message bus for these tests
        )

    def test_emit_dataset_event_required_fields(self, mock_registry: MagicMock) -> None:
        """
        Test that emit_dataset_event requires all essential fields.
        """
        # Test successful call with all required fields
        emit_dataset_event(
            mock_registry,
            dataset_id="test_dataset",
            instrument_id="EUR/USD",
            stage=Stage.FEATURE_COMPUTED,
            source=Source.LIVE,
            run_id="test_run_123",
            ts_min=1000000000000000000,  # ns
            ts_max=2000000000000000000,  # ns
            count=100,
            status=EventStatus.SUCCESS,
        )

        # Verify registry.emit_event was called
        assert mock_registry.emit_event.called
        call_args = mock_registry.emit_event.call_args

        # Verify all required fields are present
        required_fields = {
            "dataset_id",
            "instrument_id",
            "stage",
            "source",
            "run_id",
            "ts_min",
            "ts_max",
            "count",
            "status",
        }
        for field in required_fields:
            assert field in call_args.kwargs, f"Missing required field: {field}"

    def test_emit_dataset_event_enum_types(self, mock_registry: MagicMock) -> None:
        """
        Test that emit_dataset_event properly handles enum types.
        """
        emit_dataset_event(
            mock_registry,
            dataset_id="test_dataset",
            instrument_id="EUR/USD",
            stage=Stage.PREDICTION_EMITTED,
            source=Source.HISTORICAL,
            run_id="test_run_456",
            ts_min=1000000000000000000,
            ts_max=2000000000000000000,
            count=50,
            status=EventStatus.PARTIAL,
        )

        call_args = mock_registry.emit_event.call_args

        # Verify enum instances are passed correctly
        assert isinstance(call_args.kwargs["stage"], Stage)
        assert isinstance(call_args.kwargs["source"], Source)
        assert isinstance(call_args.kwargs["status"], EventStatus)

        # Verify enum values
        assert call_args.kwargs["stage"] == Stage.PREDICTION_EMITTED
        assert call_args.kwargs["source"] == Source.HISTORICAL
        assert call_args.kwargs["status"] == EventStatus.PARTIAL

    def test_emit_dataset_event_correlation_id_attachment(self, mock_registry: MagicMock) -> None:
        """
        Test that correlation_id is deterministically attached.
        """
        emit_dataset_event(
            mock_registry,
            dataset_id="test_dataset",
            instrument_id="EUR/USD",
            stage=Stage.FEATURE_COMPUTED,
            source=Source.LIVE,
            run_id="test_run_789",
            ts_min=1000000000000000000,
            ts_max=2000000000000000000,
            count=75,
            status=EventStatus.SUCCESS,
        )

        call_args = mock_registry.emit_event.call_args
        metadata = call_args.kwargs.get("metadata", {})

        # Verify correlation_id is present
        assert "correlation_id" in metadata, "correlation_id must be automatically attached"

        # Verify correlation_id is deterministic
        expected_correlation_id = make_correlation_id(
            run_id="test_run_789",
            dataset_id="test_dataset",
            instrument_id="EUR/USD",
            ts_min=1000000000000000000,
            ts_max=2000000000000000000,
            count=75,
        )
        assert metadata["correlation_id"] == expected_correlation_id

    def test_emit_dataset_event_preserves_existing_correlation_id(
        self,
        mock_registry: MagicMock,
    ) -> None:
        """
        Test that existing correlation_id is preserved if provided.
        """
        custom_correlation_id = "custom_corr_id_12345"

        emit_dataset_event(
            mock_registry,
            dataset_id="test_dataset",
            instrument_id="EUR/USD",
            stage=Stage.FEATURE_COMPUTED,
            source=Source.LIVE,
            run_id="test_run_999",
            ts_min=1000000000000000000,
            ts_max=2000000000000000000,
            count=25,
            status=EventStatus.SUCCESS,
            metadata={"correlation_id": custom_correlation_id, "other_field": "value"},
        )

        call_args = mock_registry.emit_event.call_args
        metadata = call_args.kwargs.get("metadata", {})

        # Verify custom correlation_id is preserved
        assert metadata["correlation_id"] == custom_correlation_id
        assert metadata["other_field"] == "value"

    def test_emit_dataset_event_and_watermark_calls_both(self, mock_registry: MagicMock) -> None:
        """
        Test that emit_dataset_event_and_watermark calls both event and watermark
        updates.
        """
        emit_dataset_event_and_watermark(
            mock_registry,
            dataset_id="test_dataset",
            instrument_id="EUR/USD",
            stage=Stage.FEATURE_COMPUTED,
            source=Source.LIVE,
            run_id="test_run_watermark",
            ts_min=1000000000000000000,
            ts_max=2000000000000000000,
            count=200,
            status=EventStatus.SUCCESS,
        )

        # Verify both emit_event and update_watermark were called
        assert mock_registry.emit_event.called
        assert mock_registry.update_watermark.called

        # Verify watermark call has correct parameters
        watermark_call = mock_registry.update_watermark.call_args
        assert watermark_call.kwargs["dataset_id"] == "test_dataset"
        assert watermark_call.kwargs["instrument_id"] == "EUR/USD"
        assert watermark_call.kwargs["source"] == Source.LIVE
        assert watermark_call.kwargs["last_success_ns"] == 2000000000000000000
        assert watermark_call.kwargs["count"] == 200
        assert watermark_call.kwargs["completeness_pct"] == 100.0

    def test_emit_dataset_event_metrics_labeling(self, mock_registry: MagicMock) -> None:
        """
        Test consistent metrics labeling across event emissions.
        """
        test_cases = [
            {
                "dataset_type": "bars_data",
                "component": "DataStore",
                "stage": Stage.FEATURE_COMPUTED,
                "source": Source.LIVE,
                "status": EventStatus.SUCCESS,
            },
            {
                "dataset_type": "signals_data",
                "component": "FeatureStore",
                "stage": Stage.PREDICTION_EMITTED,
                "source": Source.HISTORICAL,
                "status": EventStatus.FAILED,
            },
        ]

        for case in test_cases:
            emit_dataset_event(
                mock_registry,
                dataset_id="test_dataset",
                instrument_id="EUR/USD",
                stage=case["stage"],
                source=case["source"],
                run_id="test_metrics",
                ts_min=1000000000000000000,
                ts_max=2000000000000000000,
                count=100,
                status=case["status"],
                dataset_type=case["dataset_type"],
                component=case["component"],
            )

        # Note: Metrics validation would require access to the metrics system
        # This test documents the expected metrics labeling contract
        assert mock_registry.emit_event.call_count == len(test_cases)

    def test_data_store_emit_event_uses_centralized_helper(self, data_store: DataStore) -> None:
        """
        Test that DataStore.emit_event uses centralized helper.
        """
        data_store.emit_event(
            dataset_id="test_dataset",
            instrument_id="EUR/USD",
            stage=Stage.FEATURE_COMPUTED,
            source=Source.LIVE,
            run_id="test_data_store",
            ts_min=1000000000000000000,
            ts_max=2000000000000000000,
            count=150,
            status="success",
        )

        assert data_store.registry.emit_event.called
        call_args = data_store.registry.emit_event.call_args
        assert call_args.kwargs["stage"] == Stage.FEATURE_COMPUTED
        assert call_args.kwargs["source"] == Source.LIVE
        metadata = call_args.kwargs.get("metadata", {})
        assert "correlation_id" in metadata

    def test_data_store_partial_event_uses_centralized_helper(self, data_store: DataStore) -> None:
        """
        Test that DataStore._emit_partial_event uses centralized helper.
        """
        data_store._emit_partial_event(
            dataset_id="test_dataset",
            instrument_id="EUR/USD",
            stage="feature_engineering",
            source="live",
            run_id="test_partial",
            ts_min=1000000000000000000,
            ts_max=2000000000000000000,
            count=75,
            reason="incomplete_data",
        )

        assert data_store.registry.emit_event.called

        call_args = data_store.registry.emit_event.call_args

        # Verify PARTIAL status
        assert call_args.kwargs["status"] == EventStatus.PARTIAL

        # Verify reason in metadata
        metadata = call_args.kwargs.get("metadata", {})
        assert metadata.get("reason") == "incomplete_data"

        # Verify correlation_id attachment
        assert "correlation_id" in metadata

    def test_data_store_failed_event_uses_centralized_helper(self, data_store: DataStore) -> None:
        """
        Test that DataStore._emit_failed_event uses centralized helper.
        """
        data_store._emit_failed_event(
            dataset_id="test_dataset",
            instrument_id="EUR/USD",
            stage="model_inference",
            source="historical",
            run_id="test_failed",
            ts_min=1000000000000000000,
            ts_max=2000000000000000000,
            count=0,
            error="model_loading_failed",
        )

        assert data_store.registry.emit_event.called

        call_args = data_store.registry.emit_event.call_args

        # Verify FAILED status
        assert call_args.kwargs["status"] == EventStatus.FAILED

        # Verify error message
        assert call_args.kwargs.get("error") == "model_loading_failed"

        # Verify correlation_id attachment by centralized helper
        metadata = call_args.kwargs.get("metadata", {})
        assert "correlation_id" in metadata

    def test_event_enum_value_constraints(self, mock_registry: MagicMock) -> None:
        """
        Test that event enums enforce valid values.
        """
        # Test valid enum values
        valid_stages = [Stage.FEATURE_COMPUTED, Stage.PREDICTION_EMITTED, Stage.CATALOG_WRITTEN]
        valid_sources = [Source.LIVE, Source.HISTORICAL, Source.BACKFILL]
        valid_statuses = [EventStatus.SUCCESS, EventStatus.FAILED, EventStatus.PARTIAL]

        for stage in valid_stages:
            for source in valid_sources:
                for status in valid_statuses:
                    emit_dataset_event(
                        mock_registry,
                        dataset_id="enum_test",
                        instrument_id="EUR/USD",
                        stage=stage,
                        source=source,
                        run_id="enum_validation",
                        ts_min=1000000000000000000,
                        ts_max=2000000000000000000,
                        count=10,
                        status=status,
                    )

        # Verify all combinations were accepted
        expected_calls = len(valid_stages) * len(valid_sources) * len(valid_statuses)
        assert mock_registry.emit_event.call_count == expected_calls

    def test_event_field_types_validation(self, mock_registry: MagicMock) -> None:
        """
        Test that event fields have correct types.
        """
        emit_dataset_event(
            mock_registry,
            dataset_id="type_test",
            instrument_id="EUR/USD",
            stage=Stage.FEATURE_COMPUTED,
            source=Source.LIVE,
            run_id="type_validation",
            ts_min=1000000000000000000,
            ts_max=2000000000000000000,
            count=42,
            status=EventStatus.SUCCESS,
        )

        call_args = mock_registry.emit_event.call_args

        # Verify field types
        assert isinstance(call_args.kwargs["dataset_id"], str)
        assert isinstance(call_args.kwargs["instrument_id"], str)
        assert isinstance(call_args.kwargs["stage"], Stage)
        assert isinstance(call_args.kwargs["source"], Source)
        assert isinstance(call_args.kwargs["run_id"], str)
        assert isinstance(call_args.kwargs["ts_min"], int)
        assert isinstance(call_args.kwargs["ts_max"], int)
        assert isinstance(call_args.kwargs["count"], int)
        assert isinstance(call_args.kwargs["status"], EventStatus)

        # Verify metadata is dict
        metadata = call_args.kwargs.get("metadata")
        assert isinstance(metadata, dict)

    def test_timestamp_ordering_validation(self, mock_registry: MagicMock) -> None:
        """
        Test that ts_min <= ts_max constraint is documented.
        """
        # This test documents the expected timestamp ordering
        # Implementation should validate ts_min <= ts_max

        valid_ts_min = 1000000000000000000
        valid_ts_max = 2000000000000000000

        emit_dataset_event(
            mock_registry,
            dataset_id="timestamp_test",
            instrument_id="EUR/USD",
            stage=Stage.FEATURE_COMPUTED,
            source=Source.LIVE,
            run_id="timestamp_validation",
            ts_min=valid_ts_min,
            ts_max=valid_ts_max,
            count=1,
            status=EventStatus.SUCCESS,
        )

        call_args = mock_registry.emit_event.call_args
        assert call_args.kwargs["ts_min"] <= call_args.kwargs["ts_max"]

    def test_correlation_id_determinism(self, mock_registry: MagicMock) -> None:
        """
        Test that correlation_id generation is deterministic.
        """
        params = {
            "dataset_id": "determinism_test",
            "instrument_id": "EUR/USD",
            "stage": Stage.FEATURE_COMPUTED,
            "source": Source.LIVE,
            "run_id": "determinism_run",
            "ts_min": 1000000000000000000,
            "ts_max": 2000000000000000000,
            "count": 100,
            "status": EventStatus.SUCCESS,
        }

        # Call twice with same parameters
        emit_dataset_event(mock_registry, **params)
        first_call = mock_registry.emit_event.call_args
        first_correlation_id = first_call.kwargs["metadata"]["correlation_id"]

        mock_registry.reset_mock()

        emit_dataset_event(mock_registry, **params)
        second_call = mock_registry.emit_event.call_args
        second_correlation_id = second_call.kwargs["metadata"]["correlation_id"]

        # Verify same correlation_id for same inputs
        assert first_correlation_id == second_correlation_id

    def test_backward_compatibility_with_string_enums(self, data_store: DataStore) -> None:
        """
        Test that DataStore methods accept string enum values for backward
        compatibility.
        """
        # Test emit_event with string values (should convert to enums)
        data_store.emit_event(
            dataset_id="compat_test",
            instrument_id="EUR/USD",
            stage="FEATURE_COMPUTED",  # string instead of Stage enum
            source="live",  # string instead of Source enum
            run_id="compat_run",
            ts_min=1000000000000000000,
            ts_max=2000000000000000000,
            count=50,
            status="success",  # string instead of EventStatus enum
        )

        assert data_store.registry.emit_event.called
        call_args = data_store.registry.emit_event.call_args
        assert isinstance(call_args.kwargs["stage"], Stage)
        assert isinstance(call_args.kwargs["source"], Source)
        assert isinstance(call_args.kwargs["status"], EventStatus)


class TestEventShapeContracts:
    """
    Test contracts for event data shapes and structures.
    """

    def test_metadata_structure_contract(self) -> None:
        """
        Test that metadata follows expected structure.
        """
        # Define expected metadata structure
        expected_metadata_fields = {
            "correlation_id": str,  # Always present, deterministic
            # Optional fields that may be present:
            # "reason": str (for partial events)
            # "component": str (for metrics labeling)
            # "dataset_type": str (for metrics labeling)
        }

        mock_registry = MagicMock(spec=RegistryProtocol)

        emit_dataset_event(
            mock_registry,
            dataset_id="metadata_test",
            instrument_id="EUR/USD",
            stage=Stage.FEATURE_COMPUTED,
            source=Source.LIVE,
            run_id="metadata_validation",
            ts_min=1000000000000000000,
            ts_max=2000000000000000000,
            count=100,
            status=EventStatus.SUCCESS,
            dataset_type="test_data",
            component="TestComponent",
        )

        call_args = mock_registry.emit_event.call_args
        metadata = call_args.kwargs.get("metadata", {})

        # Verify required fields
        assert "correlation_id" in metadata
        assert isinstance(metadata["correlation_id"], str)

        # Verify all metadata values are JSON-serializable
        import json

        try:
            json.dumps(metadata)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Metadata not JSON-serializable: {e}")

    def test_event_payload_completeness(self) -> None:
        """
        Test that event payloads contain all required fields for observability.
        """
        mock_registry = MagicMock(spec=RegistryProtocol)

        emit_dataset_event_and_watermark(
            mock_registry,
            dataset_id="completeness_test",
            instrument_id="EUR/USD",
            stage=Stage.PREDICTION_EMITTED,
            source=Source.HISTORICAL,
            run_id="completeness_run",
            ts_min=1000000000000000000,
            ts_max=2000000000000000000,
            count=250,
            status=EventStatus.SUCCESS,
            dataset_type="inference_results",
            component="ModelInferenceService",
        )

        # Verify emit_event call
        event_call = mock_registry.emit_event.call_args
        event_payload = event_call.kwargs

        # Check required fields for observability
        required_fields = [
            "dataset_id",
            "instrument_id",
            "stage",
            "source",
            "run_id",
            "ts_min",
            "ts_max",
            "count",
            "status",
            "metadata",
        ]

        for field in required_fields:
            assert field in event_payload, f"Missing required field in event: {field}"

        # Verify watermark call
        watermark_call = mock_registry.update_watermark.call_args
        watermark_payload = watermark_call.kwargs

        required_watermark_fields = [
            "dataset_id",
            "instrument_id",
            "source",
            "last_success_ns",
            "count",
            "completeness_pct",
        ]

        for field in required_watermark_fields:
            assert field in watermark_payload, f"Missing required field in watermark: {field}"


# Contract documentation for reference
CONTRACT_DOCUMENTATION = """
Dataset Event Emission Contracts

1. Event Shape Contract:
   - All events MUST include: dataset_id, instrument_id, stage, source, run_id, ts_min, ts_max, count, status
   - metadata MUST be a dict and include correlation_id
   - All timestamp fields MUST be nanoseconds since epoch (int)
   - ts_min MUST be <= ts_max

2. Enum Usage Contract:
   - stage MUST be Stage enum value
   - source MUST be Source enum value
   - status MUST be EventStatus enum value
   - String values are accepted but converted to enums internally

3. Correlation ID Contract:
   - correlation_id MUST be deterministically generated using make_correlation_id()
   - correlation_id MUST be preserved if provided in metadata
   - correlation_id enables end-to-end tracing

4. Metrics Labeling Contract:
   - dataset_type defaults to dataset_id for consistent labeling
   - component should identify the emitting component class
   - All metrics use the same label structure: (dataset_type, component, stage, source, status)

5. Watermark Contract:
   - emit_dataset_event_and_watermark() MUST call both emit_event() and update_watermark()
   - Watermark completeness_pct defaults to 100.0 for success events
   - last_success_ns MUST equal ts_max for successful events

6. Backward Compatibility Contract:
   - DataStore methods accept string enum values and convert internally
   - Mock registries receive appropriate types for testing
   - Registry API remains stable across refactoring
"""
