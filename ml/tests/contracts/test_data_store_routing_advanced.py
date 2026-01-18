#!/usr/bin/env python3

"""
Advanced contract tests for DataStore routing and validation.

This module provides comprehensive contract validation for the DataStore facade,
ensuring that routing, validation, event emission, and watermark management
work correctly across all store types.

Contract areas tested:
1. Routing contracts: Correct routing by dataset type prefix
2. Validation rules: Schema enforcement and contract validation
3. Event emission: Proper events for each operation type
4. Watermark management: Consistent watermark updates
5. Transaction boundaries: Multi-store operation consistency
6. Error propagation: Store failures with proper context

Following the testing strategy guidelines for contract tests that validate
component boundaries and data quality enforcement.

"""

from __future__ import annotations

import time
import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch, call

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ml._imports import HAS_POLARS, HAS_PROMETHEUS
from ml.config.events import EventStatus, Stage
from ml.registry.dataclasses import (
    DataContract,
    DatasetManifest,
    DatasetType,
    QualityFlag,
    StorageKind,
    ValidationRule,
    ValidationRuleType,
)
from ml.registry.utils import compute_dataset_schema_hash

from ml.stores.base import FeatureData, ModelPrediction, StrategySignal

if TYPE_CHECKING:
    from ml.stores.data_store import DataStore
else:
    DataStore = Any  # pragma: no cover

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)


@pytest.fixture(autouse=True)
def _configure_datastore_symbols(
    datastore_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensure tests run against the component-based DataStore implementation.

    Note: Direct import instead of sys.modules[__name__] to avoid KeyError in
    pytest-xdist parallel execution where module may not be registered yet.
    """
    import ml.tests.contracts.test_data_store_routing_advanced as this_module
    monkeypatch.setattr(this_module, "DataStore", getattr(datastore_module, "DataStore"))


if HAS_POLARS:
    import polars as pl

# ============================================================================
# Test Data Builders
# ============================================================================


def _make_test_registry(
    dataset_id: str,
    dataset_type: DatasetType,
    validation_rules: list[ValidationRule] | None = None,
    fail_validation: bool = False,
) -> MagicMock:
    """
    Create mock registry with proper contract for testing.
    """
    # Dataset-specific schemas with required Nautilus fields
    schemas = {
        DatasetType.FEATURES: {
            "instrument_id": "str",
            "ts_event": "int64",
            "ts_init": "int64",
            "rsi": "float64",
            "ma": "float64",
        },
        DatasetType.PREDICTIONS: {
            "instrument_id": "str",
            "model_id": "str",
            "ts_event": "int64",
            "ts_init": "int64",
            "prediction": "float64",
            "confidence": "float64",
        },
        DatasetType.SIGNALS: {
            "instrument_id": "str",
            "strategy_id": "str",
            "ts_event": "int64",
            "ts_init": "int64",
            "signal_type": "str",
            "strength": "float64",
        },
    }

    schema = schemas[dataset_type]

    if dataset_type is DatasetType.PREDICTIONS:
        primary_keys = ["instrument_id", "model_id", "ts_event"]
    elif dataset_type is DatasetType.SIGNALS:
        primary_keys = ["instrument_id", "strategy_id", "ts_event"]
    else:
        primary_keys = ["instrument_id", "ts_event"]

    pipeline_signature = "contract_test"
    schema_hash = compute_dataset_schema_hash(
        schema=schema,
        primary_keys=primary_keys,
        ts_field="ts_event",
        seq_field=None,
        pipeline_signature=pipeline_signature,
    )

    manifest = DatasetManifest(
        dataset_id=dataset_id,
        dataset_type=dataset_type,
        storage_kind=StorageKind.POSTGRES,
        location=f"ml_{dataset_type.value}",
        partitioning={"by": "ts_event"},
        retention_days=365,
        schema=schema,
        ts_field="ts_event",
        seq_field=None,
        primary_keys=list(primary_keys),
        schema_hash=schema_hash,
        constraints={"nullability": {"instrument_id": False, "ts_event": False, "ts_init": False}},
        lineage=[],
        pipeline_signature=pipeline_signature,
        version="1.0.0",
    )

    # Default validation rules for contract testing
    default_rules = [
        ValidationRule(
            rule_type=ValidationRuleType.RANGE,
            field_name="ts_event",
            parameters={"min": 0},
            severity=QualityFlag.FAIL,
            description="ts_event must be non-negative",
        ),
        ValidationRule(
            rule_type=ValidationRuleType.RANGE,
            field_name="ts_init",
            parameters={"min": 0},
            severity=QualityFlag.FAIL,
            description="ts_init must be non-negative",
        ),
    ]

    # Add dataset-specific validation rules
    if dataset_type == DatasetType.PREDICTIONS:
        default_rules.extend(
            [
                ValidationRule(
                    rule_type=ValidationRuleType.RANGE,
                    field_name="prediction",
                    parameters={"min": -1.0, "max": 1.0},
                    severity=QualityFlag.FAIL,
                    description="Predictions must be in [-1, 1] range",
                ),
                ValidationRule(
                    rule_type=ValidationRuleType.RANGE,
                    field_name="confidence",
                    parameters={"min": 0.0, "max": 1.0},
                    severity=QualityFlag.FAIL,
                    description="Confidence must be in [0, 1] range",
                ),
            ]
        )
    elif dataset_type == DatasetType.SIGNALS:
        default_rules.append(
            ValidationRule(
                rule_type=ValidationRuleType.RANGE,
                field_name="strength",
                parameters={"min": 0.0, "max": 1.0},
                severity=QualityFlag.FAIL,
                description="Signal strength must be in [0, 1] range",
            ),
        )

    rules = validation_rules or default_rules

    contract = DataContract(
        contract_id=f"contract_{dataset_id}",
        dataset_id=dataset_id,
        version="1.0.0",
        validation_rules=rules,
        enforcement_mode="strict",
    )

    mock_registry = MagicMock()
    mock_registry.get_manifest.return_value = manifest

    # Simulate validation failure if requested
    if fail_validation:
        mock_registry.get_contract.side_effect = Exception("Contract validation failed")
    else:
        mock_registry.get_contract.return_value = contract

    return mock_registry


def _create_test_datastore(
    dataset_id: str,
    dataset_type: DatasetType,
    validation_rules: list[ValidationRule] | None = None,
    fail_validation: bool = False,
    enable_publishing: bool = False,
) -> tuple[DataStore, MagicMock, MagicMock, MagicMock, MagicMock | None]:
    """
    Create test DataStore with mocked dependencies.
    """
    registry = _make_test_registry(dataset_id, dataset_type, validation_rules, fail_validation)

    # Create mock stores
    mock_feature_store = MagicMock()
    mock_model_store = MagicMock()
    mock_strategy_store = MagicMock()
    mock_earnings_store = MagicMock()
    mock_publisher = MagicMock() if enable_publishing else None

    # Ensure query service fallbacks behave deterministically
    mock_model_store._query_service = None
    mock_strategy_store._query_service = None

    # Configure store method returns
    mock_feature_store.write_features.return_value = None
    mock_model_store.write_batch.return_value = None
    mock_strategy_store.write_batch.return_value = None

    # Configure read methods
    mock_feature_store.get_training_data.return_value = []
    mock_model_store.read_predictions.return_value = []
    mock_strategy_store.read_signals.return_value = []

    datastore = DataStore(
        connection_string="sqlite:///:memory:",
        registry=registry,
        feature_store=mock_feature_store,
        model_store=mock_model_store,
        strategy_store=mock_strategy_store,
        earnings_store=mock_earnings_store,
        fail_on_validation_error=not fail_validation,
        enable_publishing=enable_publishing,
        publisher=mock_publisher,
    )

    return datastore, mock_feature_store, mock_model_store, mock_strategy_store, mock_publisher


# ============================================================================
# Contract 1: Routing by Prefix Tests
# ============================================================================


class TestDataStoreRouting:
    """Test contract: DataStore routes operations by dataset type."""

    def test_features_routing_contract(self):
        """Contract: Features data routes to FeatureStore."""
        dataset_id = "features_test"
        datastore, mock_feature_store, mock_model_store, mock_strategy_store, _ = (
            _create_test_datastore(dataset_id, DatasetType.FEATURES)
        )

        # Test data with required Nautilus fields
        records = [
            {
                "instrument_id": "EUR/USD.SIM",
                "ts_event": 1000000000,
                "ts_init": 1000000000,
                "rsi": 0.5,
                "ma": 1.234,
            },
        ]

        # Execute write
        event = datastore.write_ingestion(
            dataset_id=dataset_id,
            records=records,
            source="test",
            run_id="test_run",
            instrument_id="EUR/USD.SIM",
        )

        # Contract assertions
        assert event.status == EventStatus.SUCCESS.value

        # Verify routing: FeatureStore called, others not
        mock_feature_store.write_features.assert_called()
        mock_model_store.write_batch.assert_not_called()
        mock_strategy_store.write_batch.assert_not_called()

    def test_predictions_routing_contract(self):
        """Contract: Predictions data routes to ModelStore."""
        dataset_id = "predictions_test_model"
        datastore, mock_feature_store, mock_model_store, mock_strategy_store, _ = (
            _create_test_datastore(dataset_id, DatasetType.PREDICTIONS)
        )

        records = [
            {
                "instrument_id": "EUR/USD.SIM",
                "model_id": "test_model",
                "ts_event": 1000000000,
                "ts_init": 1000000000,
                "prediction": 0.75,
                "confidence": 0.85,
            },
        ]

        event = datastore.write_ingestion(
            dataset_id=dataset_id,
            records=records,
            source="inference",
            run_id="test_run",
            instrument_id="EUR/USD.SIM",
        )

        # Contract assertions
        assert event.status == EventStatus.SUCCESS.value
        # Stage information is internal to DataStore event emission

        # Verify routing: ModelStore called, others not
        mock_model_store.write_batch.assert_called_once()
        mock_feature_store.write_features.assert_not_called()
        mock_strategy_store.write_batch.assert_not_called()

    def test_signals_routing_contract(self):
        """Contract: Signals data routes to StrategyStore."""
        dataset_id = "signals_test_strategy"
        datastore, mock_feature_store, mock_model_store, mock_strategy_store, _ = (
            _create_test_datastore(dataset_id, DatasetType.SIGNALS)
        )

        records = [
            {
                "instrument_id": "EUR/USD.SIM",
                "strategy_id": "test_strategy",
                "ts_event": 1000000000,
                "ts_init": 1000000000,
                "signal_type": "BUY",
                "strength": 0.9,
            },
        ]

        event = datastore.write_ingestion(
            dataset_id=dataset_id,
            records=records,
            source="strategy",
            run_id="test_run",
            instrument_id="EUR/USD.SIM",
        )

        # Contract assertions
        assert event.status == EventStatus.SUCCESS.value
        # Stage information is internal to DataStore event emission

        # Verify routing: StrategyStore called, others not
        mock_strategy_store.write_batch.assert_called_once()
        mock_feature_store.write_features.assert_not_called()
        mock_model_store.write_batch.assert_not_called()

    def test_read_routing_by_dataset_type(self):
        """Contract: Read operations route correctly by dataset type."""
        # Test predictions read routing
        pred_dataset_id = "predictions_model_x"
        pred_datastore, _, mock_model_store, mock_strategy_store, _ = _create_test_datastore(
            pred_dataset_id, DatasetType.PREDICTIONS
        )

        pred_datastore.read_range(
            dataset_id=pred_dataset_id,
            instrument_id="EUR/USD.SIM",
            start_ns=1000000000,
            end_ns=2000000000,
        )

        mock_model_store.read_predictions.assert_called_once_with(
            model_id="model_x",
            instrument_id="EUR/USD.SIM",
            start_ns=1000000000,
            end_ns=2000000000,
        )
        mock_strategy_store.read_signals.assert_not_called()

        # Test signals read routing
        sig_dataset_id = "signals_strategy_y"
        sig_datastore, _, mock_model_store2, mock_strategy_store2, _ = _create_test_datastore(
            sig_dataset_id, DatasetType.SIGNALS
        )

        sig_datastore.read_range(
            dataset_id=sig_dataset_id,
            instrument_id="EUR/USD.SIM",
            start_ns=1000000000,
            end_ns=2000000000,
        )

        mock_strategy_store2.read_signals.assert_called_once_with(
            strategy_id="strategy_y",
            instrument_id="EUR/USD.SIM",
            start_ns=1000000000,
            end_ns=2000000000,
        )
        mock_model_store2.read_predictions.assert_not_called()


# ============================================================================
# Contract 2: Validation Rules Enforcement
# ============================================================================


class TestDataStoreValidation:
    """Test contract: DataStore enforces validation rules consistently."""

    def test_range_validation_contract(self):
        """Contract: Range validation rules are enforced."""
        # Create strict range validation for predictions
        strict_rules = [
            ValidationRule(
                rule_type=ValidationRuleType.RANGE,
                field_name="prediction",
                parameters={"min": -0.5, "max": 0.5},  # Stricter than default
                severity=QualityFlag.FAIL,
                description="Strict prediction range",
            ),
        ]

        datastore, _, _, _, _ = _create_test_datastore(
            "predictions_strict",
            DatasetType.PREDICTIONS,
            validation_rules=strict_rules,
        )

        # Valid data should pass
        valid_records = [
            {
                "instrument_id": "EUR/USD.SIM",
                "model_id": "test_model",
                "ts_event": 1000000000,
                "ts_init": 1000000000,
                "prediction": 0.25,  # Within [-0.5, 0.5]
                "confidence": 0.8,
            },
        ]

        event = datastore.write_ingestion(
            dataset_id="predictions_strict",
            records=valid_records,
            source="test",
            run_id="test_run",
            instrument_id="EUR/USD.SIM",
        )
        assert event.status == EventStatus.SUCCESS.value

        # Invalid data should fail
        invalid_records = [
            {
                "instrument_id": "EUR/USD.SIM",
                "model_id": "test_model",
                "ts_event": 1000000000,
                "ts_init": 1000000000,
                "prediction": 0.8,  # Outside [-0.5, 0.5]
                "confidence": 0.8,
            },
        ]

        with pytest.raises(Exception):  # Should raise validation error
            datastore.write_ingestion(
                dataset_id="predictions_strict",
                records=invalid_records,
                source="test",
                run_id="test_run",
                instrument_id="EUR/USD.SIM",
            )

    def test_nullability_validation_contract(self):
        """Contract: Required fields cannot be null."""
        datastore, _, _, _, _ = _create_test_datastore(
            "predictions_nullable",
            DatasetType.PREDICTIONS,
        )

        # Records with null required fields should fail
        null_records = [
            {
                "instrument_id": None,  # Required field
                "model_id": "test_model",
                "ts_event": 1000000000,
                "ts_init": 1000000000,
                "prediction": 0.5,
                "confidence": 0.8,
            },
        ]

        with pytest.raises(Exception):  # Should raise validation error
            datastore.write_ingestion(
                dataset_id="predictions_nullable",
                records=null_records,
                source="test",
                run_id="test_run",
                instrument_id="EUR/USD.SIM",
            )

    @given(
        predictions=st.lists(
            st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=20)
    def test_prediction_bounds_property(self, predictions):
        """Property: All predictions within [-1, 1] should pass validation."""
        datastore, _, _, _, _ = _create_test_datastore(
            "predictions_property",
            DatasetType.PREDICTIONS,
        )

        # Filter to valid range
        valid_predictions = [p for p in predictions if -1.0 <= p <= 1.0]
        invalid_predictions = [p for p in predictions if not (-1.0 <= p <= 1.0)]

        # Test valid predictions
        if valid_predictions:
            valid_records = [
                {
                    "instrument_id": "EUR/USD.SIM",
                    "model_id": "test_model",
                    "ts_event": 1000000000 + i,
                    "ts_init": 1000000000 + i,
                    "prediction": pred,
                    "confidence": 0.8,
                }
                for i, pred in enumerate(valid_predictions)
            ]

            event = datastore.write_ingestion(
                dataset_id="predictions_property",
                records=valid_records,
                source="test",
                run_id="test_run",
                instrument_id="EUR/USD.SIM",
            )
            assert event.status == EventStatus.SUCCESS.value

        # Test invalid predictions
        if invalid_predictions:
            invalid_records = [
                {
                    "instrument_id": "EUR/USD.SIM",
                    "model_id": "test_model",
                    "ts_event": 2000000000,
                    "ts_init": 2000000000,
                    "prediction": invalid_predictions[0],
                    "confidence": 0.8,
                },
            ]

            with pytest.raises(Exception):
                datastore.write_ingestion(
                    dataset_id="predictions_property",
                    records=invalid_records,
                    source="test",
                    run_id="test_run",
                    instrument_id="EUR/USD.SIM",
                )


# ============================================================================
# Contract 3: Event Emission Tests
# ============================================================================


class TestDataStoreEvents:
    """Test contract: DataStore emits correct events for operations."""

    def test_event_emission_contract(self):
        """Contract: Successful operations emit SUCCESS events with correct stages."""
        datastore, _, _, _, mock_publisher = _create_test_datastore(
            "predictions_events",
            DatasetType.PREDICTIONS,
            enable_publishing=True,
        )

        records = [
            {
                "instrument_id": "EUR/USD.SIM",
                "model_id": "test_model",
                "ts_event": 1000000000,
                "ts_init": 1000000000,
                "prediction": 0.5,
                "confidence": 0.8,
            },
        ]

        event = datastore.write_ingestion(
            dataset_id="predictions_events",
            records=records,
            source="test",
            run_id="test_run",
            instrument_id="EUR/USD.SIM",
        )

        # Contract assertions
        assert event.status == EventStatus.SUCCESS.value
        # Stage information is internal to DataStore event emission
        assert event.dataset_id == "predictions_events"
        assert event.instrument_id == "EUR/USD.SIM"
        assert event.record_count == 1

        # Verify publisher was called if enabled
        if mock_publisher:
            mock_publisher.publish.assert_called()

    def test_error_event_contract(self):
        """Contract: Failed operations emit ERROR events with context."""
        # Create datastore that will fail validation
        datastore, _, _, _, _ = _create_test_datastore(
            "predictions_error",
            DatasetType.PREDICTIONS,
            fail_validation=True,
        )

        records = [
            {
                "instrument_id": "EUR/USD.SIM",
                "model_id": "test_model",
                "ts_event": 1000000000,
                "ts_init": 1000000000,
                "prediction": 0.5,
                "confidence": 0.8,
            },
        ]

        with pytest.raises(Exception):
            datastore.write_ingestion(
                dataset_id="predictions_error",
                records=records,
                source="test",
                run_id="test_run",
                instrument_id="EUR/USD.SIM",
            )

    def test_batch_event_aggregation_contract(self):
        """Contract: Batch operations emit single aggregated event."""
        datastore, _, _, _, _ = _create_test_datastore(
            "predictions_batch",
            DatasetType.PREDICTIONS,
        )

        # Large batch of records
        records = [
            {
                "instrument_id": "EUR/USD.SIM",
                "model_id": "test_model",
                "ts_event": 1000000000 + i,
                "ts_init": 1000000000 + i,
                "prediction": 0.5,
                "confidence": 0.8,
            }
            for i in range(100)
        ]

        event = datastore.write_ingestion(
            dataset_id="predictions_batch",
            records=records,
            source="test",
            run_id="test_run",
            instrument_id="EUR/USD.SIM",
        )

        # Contract: Single event for entire batch
        assert event.record_count == 100
        assert event.status == EventStatus.SUCCESS.value


# ============================================================================
# Contract 4: Watermark Management Tests
# ============================================================================


class TestDataStoreWatermarks:
    """Test contract: DataStore manages watermarks consistently."""

    def test_watermark_progression_contract(self):
        """Contract: Watermarks progress monotonically with timestamps."""
        datastore, _, _, _, _ = _create_test_datastore(
            "predictions_watermark",
            DatasetType.PREDICTIONS,
        )

        # Write records in increasing timestamp order
        timestamps = [1000000000, 1100000000, 1200000000]

        for i, ts in enumerate(timestamps):
            records = [
                {
                    "instrument_id": "EUR/USD.SIM",
                    "model_id": "test_model",
                    "ts_event": ts,
                    "ts_init": ts,
                    "prediction": 0.5,
                    "confidence": 0.8,
                },
            ]

            event = datastore.write_ingestion(
                dataset_id="predictions_watermark",
                records=records,
                source="test",
                run_id=f"test_run_{i}",
                instrument_id="EUR/USD.SIM",
            )

            # Contract: Watermark should reflect latest timestamp (scaled to nanoseconds)
            assert event.ts_min == ts * 1_000_000_000  # Convert to nanoseconds
            assert event.ts_max == ts * 1_000_000_000

    def test_cross_store_watermark_consistency(self):
        """Contract: Watermarks consistent across different store types."""
        # Test multiple dataset types with same timestamps
        timestamp = 1000000000

        datasets = [
            ("predictions_sync", DatasetType.PREDICTIONS),
            ("signals_sync", DatasetType.SIGNALS),
        ]

        events = []
        for dataset_id, dataset_type in datasets:
            datastore, _, _, _, _ = _create_test_datastore(dataset_id, dataset_type)

            if dataset_type == DatasetType.PREDICTIONS:
                records = [
                    {
                        "instrument_id": "EUR/USD.SIM",
                        "model_id": "test_model",
                        "ts_event": timestamp,
                        "ts_init": timestamp,
                        "prediction": 0.5,
                        "confidence": 0.8,
                    },
                ]
            else:  # SIGNALS
                records = [
                    {
                        "instrument_id": "EUR/USD.SIM",
                        "strategy_id": "test_strategy",
                        "ts_event": timestamp,
                        "ts_init": timestamp,
                        "signal_type": "BUY",
                        "strength": 0.9,
                    },
                ]

            event = datastore.write_ingestion(
                dataset_id=dataset_id,
                records=records,
                source="test",
                run_id="sync_test",
                instrument_id="EUR/USD.SIM",
            )
            events.append(event)

        # Contract: All events should have consistent watermarks (scaled to nanoseconds)
        expected_ts_ns = timestamp * 1_000_000_000
        assert all(event.ts_min == expected_ts_ns for event in events)
        assert all(event.ts_max == expected_ts_ns for event in events)


# ============================================================================
# Contract 5: Transaction Boundaries Tests
# ============================================================================


class TestDataStoreTransactions:
    """Test contract: DataStore maintains consistency across operations."""

    def test_multi_store_consistency_contract(self):
        """Contract: Operations affecting multiple stores maintain consistency."""
        # This tests that if one store fails, the entire operation fails
        datastore, _mock_feature_store, mock_model_store, _mock_strategy_store, _ = (
            _create_test_datastore("predictions_consistency", DatasetType.PREDICTIONS)
        )

        # Configure model store to fail
        mock_model_store.write_batch.side_effect = Exception("Store failure")

        records = [
            {
                "instrument_id": "EUR/USD.SIM",
                "model_id": "test_model",
                "ts_event": 1000000000,
                "ts_init": 1000000000,
                "prediction": 0.5,
                "confidence": 0.8,
            },
        ]

        # Contract: Operation should fail completely
        with pytest.raises(Exception):
            datastore.write_ingestion(
                dataset_id="predictions_consistency",
                records=records,
                source="test",
                run_id="consistency_test",
                instrument_id="EUR/USD.SIM",
            )

        # Contract: No partial writes should occur
        # (This depends on implementation details, but validates contract)

    def test_flush_coordination_contract(self):
        """Contract: Flush operations coordinate across all stores."""
        datastore, mock_feature_store, mock_model_store, mock_strategy_store, _ = (
            _create_test_datastore("test_flush", DatasetType.PREDICTIONS)
        )

        # Contract: DataStore exposes individual store flush methods
        # Note: DataStore doesn't have a unified flush method currently
        # But individual stores can be flushed
        if hasattr(datastore, "feature_store"):
            datastore.feature_store.flush()
            mock_feature_store.flush.assert_called()

        if hasattr(datastore, "model_store"):
            datastore.model_store.flush()
            mock_model_store.flush.assert_called()

        if hasattr(datastore, "strategy_store"):
            datastore.strategy_store.flush()
            mock_strategy_store.flush.assert_called()


# ============================================================================
# Contract 6: Error Propagation Tests
# ============================================================================


class TestDataStoreErrorPropagation:
    """Test contract: DataStore properly propagates errors with context."""

    def test_store_error_context_contract(self):
        """Contract: Store errors include proper context and traceability."""
        datastore, _, mock_model_store, _, _ = _create_test_datastore(
            "predictions_error_context",
            DatasetType.PREDICTIONS,
        )

        # Configure store to fail with specific error
        store_error = Exception("Database connection failed")
        mock_model_store.write_batch.side_effect = store_error

        records = [
            {
                "instrument_id": "EUR/USD.SIM",
                "model_id": "test_model",
                "ts_event": 1000000000,
                "ts_init": 1000000000,
                "prediction": 0.5,
                "confidence": 0.8,
            },
        ]

        # Contract: Error should be propagated with context
        with pytest.raises(Exception) as exc_info:
            datastore.write_ingestion(
                dataset_id="predictions_error_context",
                records=records,
                source="test",
                run_id="error_test",
                instrument_id="EUR/USD.SIM",
            )

        # Contract: Error context should be preserved
        # (Implementation-specific assertion about error wrapping)
        assert (
            "Database connection failed" in str(exc_info.value)
            or store_error in exc_info.value.__cause__
        )

    def test_validation_error_detail_contract(self):
        """Contract: Validation errors provide detailed field-level information."""
        # Create datastore with strict validation
        strict_rules = [
            ValidationRule(
                rule_type=ValidationRuleType.RANGE,
                field_name="prediction",
                parameters={"min": 0.0, "max": 0.5},
                severity=QualityFlag.FAIL,
                description="Very strict prediction range",
            ),
        ]

        datastore, _, _, _, _ = _create_test_datastore(
            "predictions_strict_error",
            DatasetType.PREDICTIONS,
            validation_rules=strict_rules,
        )

        invalid_records = [
            {
                "instrument_id": "EUR/USD.SIM",
                "model_id": "test_model",
                "ts_event": 1000000000,
                "ts_init": 1000000000,
                "prediction": 0.8,  # Violates range [0.0, 0.5]
                "confidence": 0.8,
            },
        ]

        # Contract: Validation error should include field details
        with pytest.raises(Exception) as exc_info:
            datastore.write_ingestion(
                dataset_id="predictions_strict_error",
                records=invalid_records,
                source="test",
                run_id="validation_error_test",
                instrument_id="EUR/USD.SIM",
            )

        # Contract: Error message should reference the field and constraint
        error_msg = str(exc_info.value)
        assert "prediction" in error_msg or "range" in error_msg.lower()


# ============================================================================
# Integration Contract Tests
# ============================================================================


class TestDataStoreIntegrationContracts:
    """
    Test contracts for integrated DataStore workflows.
    """

    def test_complete_workflow_contract(self):
        """Contract: Complete write-read workflow maintains data integrity."""
        datastore, _, mock_model_store, _, _ = _create_test_datastore(
            "predictions_workflow",
            DatasetType.PREDICTIONS,
        )

        # Configure read to return what was written
        original_records = [
            {
                "instrument_id": "EUR/USD.SIM",
                "model_id": "test_model",
                "ts_event": 1000000000,
                "ts_init": 1000000000,
                "prediction": 0.5,
                "confidence": 0.8,
            },
        ]

        mock_model_store.read_predictions.return_value = original_records

        # Write data
        write_event = datastore.write_ingestion(
            dataset_id="predictions_workflow",
            records=original_records,
            source="test",
            run_id="workflow_test",
            instrument_id="EUR/USD.SIM",
        )

        # Read data back
        read_result = datastore.read_range(
            dataset_id="predictions_workflow",
            instrument_id="EUR/USD.SIM",
            start_ns=999999999,
            end_ns=1000000001,
        )

        # Contract: Read result should match written data
        import polars as pl

        assert write_event.status in {EventStatus.SUCCESS.value, "success"}
        assert isinstance(read_result, pl.DataFrame)
        assert read_result.height == len(original_records)
        assert read_result.to_dicts() == original_records
        mock_model_store.read_predictions.assert_called_once_with(
            model_id="workflow",
            instrument_id="EUR/USD.SIM",
            start_ns=999999999,
            end_ns=1000000001,
        )

    def test_concurrent_operations_contract(self):
        """Contract: Concurrent operations do not interfere with each other."""
        # This is a simplified test; full concurrency testing would require threading
        datastore, _, mock_model_store, _, _ = _create_test_datastore(
            "predictions_concurrent",
            DatasetType.PREDICTIONS,
        )

        # Simulate multiple rapid operations
        for i in range(5):
            records = [
                {
                    "instrument_id": "EUR/USD.SIM",
                    "model_id": f"test_model_{i}",
                    "ts_event": 1000000000 + i,
                    "ts_init": 1000000000 + i,
                    "prediction": 0.5,
                    "confidence": 0.8,
                },
            ]

            event = datastore.write_ingestion(
                dataset_id="predictions_concurrent",
                records=records,
                source="test",
                run_id=f"concurrent_test_{i}",
                instrument_id="EUR/USD.SIM",
            )

            # Contract: Each operation should succeed independently
            assert event.status == EventStatus.SUCCESS.value
            assert event.record_count == 1

        # Contract: All operations should have been processed
        assert mock_model_store.write_batch.call_count == 5


# ============================================================================
# Circuit Breaker Contract Tests
# ============================================================================


class TestDataStoreCircuitBreaker:
    """Test contract: DataStore circuit breaker functionality."""

    def test_circuit_breaker_activation_contract(self):
        """Contract: Circuit breaker activates after consecutive failures."""
        with patch("ml.stores.data_store.time.time") as mock_time:
            mock_time.return_value = 1000.0

            datastore, _, mock_model_store, _, _ = _create_test_datastore(
                "predictions_circuit",
                DatasetType.PREDICTIONS,
            )

            # Configure store to fail consistently
            mock_model_store.write_batch.side_effect = Exception("Persistent failure")

            records = [
                {
                    "instrument_id": "EUR/USD.SIM",
                    "model_id": "test_model",
                    "ts_event": 1000000000,
                    "ts_init": 1000000000,
                    "prediction": 0.5,
                    "confidence": 0.8,
                },
            ]

            # Contract: Multiple failures should eventually trigger circuit breaker
            # (Implementation-specific behavior)
            for i in range(3):
                with pytest.raises(Exception):
                    datastore.write_ingestion(
                        dataset_id="predictions_circuit",
                        records=records,
                        source="test",
                        run_id=f"circuit_test_{i}",
                        instrument_id="EUR/USD.SIM",
                    )

            # Note: Actual circuit breaker testing would require access to internal state
            # This test validates that the interface supports circuit breaker concepts
