#!/usr/bin/env python3

"""
Unit tests for DataWriterComponent.

Tests all 6 write methods with success cases, error handling, and edge cases.

Phase 2.4.2 - DataStore Decomposition

"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from ml.config.events import EventStatus
from ml.registry.dataclasses import DataContract, DatasetManifest, DatasetType, QualityFlag, ValidationRule
from ml.registry.dataclasses import ValidationRuleType
from ml.stores.common.data_writer import DataEvent, DataWriterComponent
from ml.stores.common.schema_validator import QualityReport, ValidationViolation


if TYPE_CHECKING:
    from ml.stores.common.schema_validator import SchemaValidatorComponent


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def mock_feature_store() -> Mock:
    """Mock FeatureStore."""
    store = Mock()
    store.write_features = Mock()
    return store


@pytest.fixture
def mock_model_store() -> Mock:
    """Mock ModelStore."""
    store = Mock()
    store.write_batch = Mock()
    return store


@pytest.fixture
def mock_strategy_store() -> Mock:
    """Mock StrategyStore."""
    store = Mock()
    store.write_batch = Mock()
    return store


@pytest.fixture
def mock_earnings_store() -> Mock:
    """Mock EarningsStore."""
    store = Mock()
    store.write_actuals = Mock()
    store.write_estimates = Mock()
    return store


@pytest.fixture
def mock_validator() -> Mock:
    """Mock SchemaValidatorComponent."""
    validator = Mock()

    # Default preflight success
    validator.preflight_check = Mock(
        return_value=(True, None, {"preflight_passed": True, "warnings": []})
    )

    # Default validation success
    validator.validate_batch = Mock(
        return_value=QualityReport(
            dataset_id="test_dataset",
            total_records=100,
            passed_records=100,
            failed_records=0,
            quality_score=1.0,
            violations=[],
            validation_time_ms=10.0,
            metadata={},
        )
    )

    return validator


@pytest.fixture
def mock_registry() -> Mock:
    """Mock DataRegistry."""
    registry = Mock()

    # Default manifest
    manifest = DatasetManifest(
        dataset_id="test_dataset",
        dataset_type=DatasetType.BARS,
        storage_kind="postgres",
        location="test_location",
        partitioning={},
        retention_days=365,
        schema={
            "instrument_id": "str",
            "ts_event": "int64",
            "ts_init": "int64",
            "close": "float64",
        },
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["instrument_id", "ts_event"],
        schema_hash="test_hash",
        constraints={},
        lineage=[],
        pipeline_signature="test",
        version="1.0.0",
    )

    # Default contract
    contract = DataContract(
        contract_id="test_contract",
        dataset_id="test_dataset",
        version="1.0.0",
        validation_rules=[
            ValidationRule(
                rule_type=ValidationRuleType.TYPE_CHECK,
                field_name="*",
                parameters={},
                severity=QualityFlag.WARN,
                description="Type checking",
            )
        ],
        quality_thresholds={},
        enforcement_mode="lenient",
    )

    registry.get_manifest = Mock(return_value=manifest)
    registry.get_contract = Mock(return_value=contract)

    return registry


@pytest.fixture
def data_writer(
    mock_feature_store: Mock,
    mock_model_store: Mock,
    mock_strategy_store: Mock,
    mock_earnings_store: Mock,
    mock_validator: Mock,
    mock_registry: Mock,
) -> DataWriterComponent:
    """DataWriterComponent with mocked dependencies."""
    return DataWriterComponent(
        feature_store=mock_feature_store,
        model_store=mock_model_store,
        strategy_store=mock_strategy_store,
        earnings_store=mock_earnings_store,
        validator=mock_validator,
        registry=mock_registry,
        fail_on_validation_error=True,
    )


# =========================================================================
# Test: write_ingestion
# =========================================================================


def test_write_ingestion_success(data_writer: DataWriterComponent, mock_validator: Mock) -> None:
    """Test successful ingestion write with validation."""
    # Arrange
    dataset_id = "bars_eurusd_1m"
    records = [
        {"instrument_id": "EURUSD.SIM", "ts_event": 1000000000, "ts_init": 1000000000, "close": 1.0850},
        {"instrument_id": "EURUSD.SIM", "ts_event": 2000000000, "ts_init": 2000000000, "close": 1.0851},
    ]
    source = "historical"
    run_id = "test_run_123"

    # Act
    event = data_writer.write_ingestion(
        dataset_id=dataset_id,
        records=records,
        source=source,
        run_id=run_id,
        instrument_id="EURUSD.SIM",
    )

    # Assert
    assert event.status == EventStatus.SUCCESS.value
    assert event.record_count == 2
    assert event.dataset_id == dataset_id
    assert event.instrument_id == "EURUSD.SIM"
    assert event.operation == "write_ingestion"
    assert event.source == source
    assert event.run_id == run_id

    # Verify preflight check was called
    mock_validator.preflight_check.assert_called_once()

    # Verify validation was called
    mock_validator.validate_batch.assert_called_once()


def test_write_ingestion_with_invalid_schema(
    data_writer: DataWriterComponent, mock_validator: Mock
) -> None:
    """Test write_ingestion fails when preflight check fails."""
    # Arrange
    dataset_id = "bars_eurusd_1m"
    records = [{"invalid": "data"}]
    source = "test"
    run_id = "test_run_123"

    # Configure mock to return preflight failure
    mock_validator.preflight_check = Mock(
        return_value=(False, "Missing required columns", {"missing_columns": ["ts_event"]})
    )

    # Act & Assert
    with pytest.raises(ValueError, match="Preflight check failed"):
        data_writer.write_ingestion(
            dataset_id=dataset_id,
            records=records,
            source=source,
            run_id=run_id,
        )


# =========================================================================
# Test: write_features
# =========================================================================


def test_write_features_success(data_writer: DataWriterComponent, mock_feature_store: Mock) -> None:
    """Test successful feature write."""
    # Arrange
    instrument_id = "EURUSD.SIM"

    # Create mock FeatureData objects
    features = []
    for i in range(3):
        feature = Mock()
        feature.instrument_id = instrument_id
        feature.feature_set_id = "test_features"
        feature.ts_event = 1000000000 + i * 1000000000
        feature.ts_init = 1000000000 + i * 1000000000
        feature.values = {"close": 1.0850 + i * 0.0001}
        features.append(feature)

    source = "computed"

    # Act
    event = data_writer.write_features(
        instrument_id=instrument_id,
        features=features,
        source=source,
    )

    # Assert
    assert event.status == EventStatus.SUCCESS.value
    assert event.record_count == 3
    assert event.instrument_id == instrument_id
    assert event.operation == "write_features"
    assert event.source == source

    # Verify feature store was called for each feature
    assert mock_feature_store.write_features.call_count == 3


def test_write_features_with_missing_columns(
    data_writer: DataWriterComponent, mock_feature_store: Mock
) -> None:
    """Test write_features fails when instrument_id mismatch."""
    # Arrange
    instrument_id = "EURUSD.SIM"

    # Create mock feature with wrong instrument_id
    feature = Mock()
    feature.instrument_id = "WRONG.SIM"
    feature.feature_set_id = "test_features"
    feature.ts_event = 1000000000
    feature.ts_init = 1000000000
    feature.values = {"close": 1.0850}

    # Act & Assert
    with pytest.raises(ValueError, match="Instrument mismatch"):
        data_writer.write_features(
            instrument_id=instrument_id,
            features=[feature],
            source="computed",
        )


# =========================================================================
# Test: write_predictions
# =========================================================================


def test_write_predictions_success(
    data_writer: DataWriterComponent, mock_model_store: Mock
) -> None:
    """Test successful prediction write."""
    # Arrange
    instrument_id = "EURUSD.SIM"
    model_id = "test_model"

    # Create mock ModelPrediction objects
    predictions = []
    for i in range(3):
        prediction = Mock()
        prediction.instrument_id = instrument_id
        prediction.model_id = model_id
        prediction.ts_event = 1000000000 + i * 1000000000
        prediction.ts_init = 1000000000 + i * 1000000000
        predictions.append(prediction)

    source = "inference"

    # Act
    event = data_writer.write_predictions(
        predictions=predictions,
        source=source,
    )

    # Assert
    assert event.status == EventStatus.SUCCESS.value
    assert event.record_count == 3
    assert event.instrument_id == instrument_id
    assert event.operation == "write_predictions"
    assert event.source == source
    assert event.metadata["model_id"] == model_id

    # Verify model store was called
    mock_model_store.write_batch.assert_called_once()


def test_write_predictions_with_invalid_types(data_writer: DataWriterComponent) -> None:
    """Test write_predictions fails when predictions list is empty."""
    # Act & Assert
    with pytest.raises(ValueError, match="No predictions to write"):
        data_writer.write_predictions(
            predictions=[],
            source="inference",
        )


# =========================================================================
# Test: write_signals
# =========================================================================


def test_write_signals_success(
    data_writer: DataWriterComponent, mock_strategy_store: Mock
) -> None:
    """Test successful signal write."""
    # Arrange
    instrument_id = "EURUSD.SIM"
    strategy_id = "test_strategy"

    # Create mock StrategySignal objects
    signals = []
    for i in range(3):
        signal = Mock()
        signal.instrument_id = instrument_id
        signal.strategy_id = strategy_id
        signal.ts_event = 1000000000 + i * 1000000000
        signal.ts_init = 1000000000 + i * 1000000000
        signals.append(signal)

    source = "strategy"

    # Act
    event = data_writer.write_signals(
        signals=signals,
        source=source,
    )

    # Assert
    assert event.status == EventStatus.SUCCESS.value
    assert event.record_count == 3
    assert event.instrument_id == instrument_id
    assert event.operation == "write_signals"
    assert event.source == source
    assert event.metadata["strategy_id"] == strategy_id

    # Verify strategy store was called
    mock_strategy_store.write_batch.assert_called_once()


def test_write_signals_with_duplicate_timestamps(data_writer: DataWriterComponent) -> None:
    """Test write_signals fails when signals list is empty."""
    # Act & Assert
    with pytest.raises(ValueError, match="No signals to write"):
        data_writer.write_signals(
            signals=[],
            source="strategy",
        )


# =========================================================================
# Test: write_earnings_actual
# =========================================================================


def test_write_earnings_actual_success(
    data_writer: DataWriterComponent, mock_earnings_store: Mock, mock_validator: Mock
) -> None:
    """Test successful earnings actual write."""
    # Arrange
    ticker = "AAPL"
    period_end = "2024-03-31"
    filing_date = "2024-04-15"
    eps_diluted = 1.52
    revenue = 90753000000.0
    ts_event = 1713196800000000000
    ts_init = 1713196800000000000

    # Act
    event = data_writer.write_earnings_actual(
        ticker=ticker,
        period_end=period_end,
        filing_date=filing_date,
        eps_diluted=eps_diluted,
        revenue=revenue,
        ts_event=ts_event,
        ts_init=ts_init,
    )

    # Assert
    assert event.status == EventStatus.SUCCESS.value
    assert event.record_count == 1
    assert event.instrument_id == ticker
    assert event.operation == "write_earnings_actual"

    # Verify earnings store was called
    mock_earnings_store.write_actuals.assert_called_once()

    # Verify validation was called
    mock_validator.validate_batch.assert_called_once()


def test_write_earnings_actual_with_empty_dataframe(
    data_writer: DataWriterComponent, mock_validator: Mock
) -> None:
    """Test write_earnings_actual with strict validation failure."""
    # Arrange
    ticker = "AAPL"
    period_end = "2024-03-31"
    filing_date = "2024-04-15"
    ts_event = 1713196800000000000
    ts_init = 1713196800000000000

    # Configure validator to fail with strict mode
    mock_validator.validate_batch = Mock(
        return_value=QualityReport(
            dataset_id="earnings_actuals",
            total_records=1,
            passed_records=0,
            failed_records=1,
            quality_score=0.0,
            violations=[
                ValidationViolation(
                    rule_type=ValidationRuleType.NULLABILITY,
                    field_name="eps_diluted",
                    severity=QualityFlag.FAIL,
                    violation_count=1,
                    sample_values=[],
                    description="Required field contains null",
                )
            ],
            validation_time_ms=5.0,
            metadata={},
        )
    )

    # Configure registry to return strict contract
    strict_contract = DataContract(
        contract_id="earnings_contract",
        dataset_id="earnings_actuals",
        version="1.0.0",
        validation_rules=[
            ValidationRule(
                rule_type=ValidationRuleType.NULLABILITY,
                field_name="eps_diluted",
                parameters={"nullable": False},
                severity=QualityFlag.FAIL,
                description="eps_diluted cannot be null",
            )
        ],
        quality_thresholds={},
        enforcement_mode="strict",
    )
    data_writer._registry.get_contract = Mock(return_value=strict_contract)

    # Act & Assert
    with pytest.raises(ValueError, match="Data validation failed"):
        data_writer.write_earnings_actual(
            ticker=ticker,
            period_end=period_end,
            filing_date=filing_date,
            eps_diluted=None,
            revenue=None,
            ts_event=ts_event,
            ts_init=ts_init,
        )


# =========================================================================
# Test: write_earnings_estimate
# =========================================================================


def test_write_earnings_estimate_success(
    data_writer: DataWriterComponent, mock_earnings_store: Mock, mock_validator: Mock
) -> None:
    """Test successful earnings estimate write."""
    # Arrange
    ticker = "AAPL"
    estimate_date = "2024-03-15"
    period_end = "2024-03-31"
    eps_consensus = 1.50
    ts_event = 1710547200000000000
    ts_init = 1710547200000000000

    # Act
    event = data_writer.write_earnings_estimate(
        ticker=ticker,
        estimate_date=estimate_date,
        period_end=period_end,
        eps_consensus=eps_consensus,
        ts_event=ts_event,
        ts_init=ts_init,
    )

    # Assert
    assert event.status == EventStatus.SUCCESS.value
    assert event.record_count == 1
    assert event.instrument_id == ticker
    assert event.operation == "write_earnings_estimate"

    # Verify earnings store was called
    mock_earnings_store.write_estimates.assert_called_once()

    # Verify validation was called
    mock_validator.validate_batch.assert_called_once()


def test_write_earnings_estimate_with_future_dates(
    data_writer: DataWriterComponent, mock_earnings_store: Mock
) -> None:
    """Test write_earnings_estimate with future dates (should succeed)."""
    # Arrange
    ticker = "AAPL"
    estimate_date = "2025-12-31"
    period_end = "2025-12-31"
    eps_consensus = 2.00
    ts_event = 1735689600000000000
    ts_init = 1735689600000000000

    # Act
    event = data_writer.write_earnings_estimate(
        ticker=ticker,
        estimate_date=estimate_date,
        period_end=period_end,
        eps_consensus=eps_consensus,
        ts_event=ts_event,
        ts_init=ts_init,
    )

    # Assert
    assert event.status == EventStatus.SUCCESS.value
    assert event.record_count == 1


# =========================================================================
# Test: Error Handling
# =========================================================================


def test_write_features_store_failure(
    data_writer: DataWriterComponent, mock_feature_store: Mock
) -> None:
    """Test write_features handles store failures gracefully."""
    # Arrange
    instrument_id = "EURUSD.SIM"

    feature = Mock()
    feature.instrument_id = instrument_id
    feature.feature_set_id = "test_features"
    feature.ts_event = 1000000000
    feature.ts_init = 1000000000
    feature.values = {"close": 1.0850}

    # Configure store to raise exception
    mock_feature_store.write_features = Mock(side_effect=RuntimeError("Database connection failed"))

    # Act & Assert
    with pytest.raises(RuntimeError, match="Feature write failed"):
        data_writer.write_features(
            instrument_id=instrument_id,
            features=[feature],
            source="computed",
        )


def test_write_earnings_actual_store_failure(
    data_writer: DataWriterComponent, mock_earnings_store: Mock
) -> None:
    """Test write_earnings_actual handles store failures."""
    # Arrange
    ticker = "AAPL"
    period_end = "2024-03-31"
    filing_date = "2024-04-15"
    ts_event = 1713196800000000000
    ts_init = 1713196800000000000

    # Configure store to raise exception
    mock_earnings_store.write_actuals = Mock(side_effect=RuntimeError("Database error"))

    # Act & Assert
    with pytest.raises(RuntimeError, match="Earnings actual write failed"):
        data_writer.write_earnings_actual(
            ticker=ticker,
            period_end=period_end,
            filing_date=filing_date,
            eps_diluted=1.52,
            revenue=90753000000.0,
            ts_event=ts_event,
            ts_init=ts_init,
        )


# =========================================================================
# Test: Metrics Emission
# =========================================================================


@patch("ml.stores.common.data_writer.HAS_PROMETHEUS", True)
@patch("ml.stores.common.data_writer.write_rejection_counter")
def test_write_ingestion_emits_rejection_metric(
    mock_counter: Mock, data_writer: DataWriterComponent, mock_validator: Mock
) -> None:
    """Test write_ingestion emits rejection metric on preflight failure."""
    # Arrange
    dataset_id = "test_dataset"
    records = [{"invalid": "data"}]

    # Configure mock to return preflight failure
    mock_validator.preflight_check = Mock(
        return_value=(False, "Missing columns", {"missing_columns": ["ts_event"]})
    )

    # Act
    try:
        data_writer.write_ingestion(
            dataset_id=dataset_id,
            records=records,
            source="test",
            run_id="test_run",
        )
    except ValueError:
        pass

    # Assert
    mock_counter.labels.assert_called_once_with(
        dataset_id=dataset_id,
        reason="preflight_failed",
    )


# =========================================================================
# Test: Helper Methods
# =========================================================================


def test_extract_instrument_id(data_writer: DataWriterComponent) -> None:
    """Test _extract_instrument_id helper."""
    # Arrange
    from ml._imports import pl

    if pl is None:
        pytest.skip("Polars not available")

    data_frame = pl.DataFrame([
        {"instrument_id": "EURUSD.SIM", "close": 1.0850},
        {"instrument_id": "EURUSD.SIM", "close": 1.0851},
    ])

    # Act
    instrument_id = data_writer._extract_instrument_id(data_frame)

    # Assert
    assert instrument_id == "EURUSD.SIM"


def test_extract_timestamp_range(
    data_writer: DataWriterComponent, mock_registry: Mock
) -> None:
    """Test _extract_timestamp_range helper."""
    # Arrange
    from ml._imports import pl

    if pl is None:
        pytest.skip("Polars not available")

    data_frame = pl.DataFrame([
        {"ts_event": 1000000000, "close": 1.0850},
        {"ts_event": 2000000000, "close": 1.0851},
        {"ts_event": 3000000000, "close": 1.0852},
    ])

    manifest = mock_registry.get_manifest("test_dataset")

    # Act
    ts_min, ts_max = data_writer._extract_timestamp_range(data_frame, manifest)

    # Assert
    assert ts_min == 1000000000
    assert ts_max == 3000000000
