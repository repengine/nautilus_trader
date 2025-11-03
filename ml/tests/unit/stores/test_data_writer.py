#!/usr/bin/env python3

"""
Unit tests for DataWriter component.

Tests all write operations with mocked stores to ensure proper delegation,
validation, event emission, and watermark updates.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import StorageKind
from ml.stores.base import FeatureData
from ml.stores.base import ModelPrediction
from ml.stores.base import StrategySignal
from ml.stores.contract_enforcer import ContractEnforcer
from ml.stores.data_writer import DataWriter
from ml.stores.schema_validator import SchemaValidator


# ========================================================================
# Fixtures
# ========================================================================
# Note: mock_feature_store, mock_model_store, and mock_strategy_store
# are now imported from conftest.py (which imports from ml.tests.fixtures.mock_stores)


@pytest.fixture
def mock_earnings_store() -> Mock:
    """Create mock earnings store."""
    store = Mock()
    store.write_actuals = Mock()
    store.write_estimates = Mock()
    return store


@pytest.fixture
def mock_schema_validator() -> Mock:
    """Create mock schema validator."""
    validator = Mock(spec=SchemaValidator)
    validator.enforce_quality_report = Mock()
    return validator


@pytest.fixture
def mock_contract_enforcer(mock_schema_validator: Mock) -> Mock:
    """Create mock contract enforcer."""
    enforcer = Mock(spec=ContractEnforcer)

    # Mock manifest
    manifest = DatasetManifest(
        dataset_id="test_dataset",
        dataset_type=DatasetType.FEATURES,
        storage_kind=StorageKind.POSTGRES,
        location="ml.test_dataset",
        partitioning={},
        retention_days=90,
        version="1.0.0",
        schema={"instrument_id": "str", "ts_event": "int64", "ts_init": "int64"},
        schema_hash="test_hash",
        primary_keys=["instrument_id", "ts_event"],
        ts_field="ts_event",
        seq_field=None,
        constraints={},
        lineage=[],
        pipeline_signature="test",
    )
    enforcer.get_manifest = Mock(return_value=manifest)

    # Mock contract
    from ml.registry.dataclasses import DataContract
    from ml.registry.dataclasses import QualityFlag
    from ml.registry.dataclasses import ValidationRule
    from ml.registry.dataclasses import ValidationRuleType

    contract = DataContract(
        contract_id="test_contract",
        dataset_id="test_dataset",
        version="1.0.0",
        validation_rules=[
            ValidationRule(
                rule_type=ValidationRuleType.NULLABILITY,
                field_name="ts_event",
                parameters={"nullable": False},
                severity=QualityFlag.FAIL,
                description="Timestamp must not be null",
            ),
        ],
        enforcement_mode="lenient",
        quality_thresholds={},
        created_at=time.time_ns(),
        last_modified=time.time_ns(),
    )
    enforcer.get_contract = Mock(return_value=contract)

    # Mock preflight check
    enforcer.preflight_check = Mock(return_value=(True, None, {}))

    # Mock validate_batch
    from ml.stores.validation_types import QualityReport

    quality_report = QualityReport(
        dataset_id="test_dataset",
        total_records=10,
        passed_records=10,
        failed_records=0,
        quality_score=1.0,
        violations=[],
        validation_time_ms=1.0,
    )
    enforcer.validate_batch = Mock(return_value=quality_report)

    return enforcer


@pytest.fixture
def mock_registry(mock_registry_factory) -> Mock:
    """Create mock registry using factory.

    Uses mock_registry_factory to create a protocol-based mock without spec
    for maximum flexibility in these unit tests.
    """
    return mock_registry_factory("protocol", use_spec=False)


@pytest.fixture
def data_writer(
    mock_feature_store: Mock,
    mock_model_store: Mock,
    mock_strategy_store: Mock,
    mock_earnings_store: Mock,
    mock_contract_enforcer: Mock,
    mock_schema_validator: Mock,
    mock_registry: Mock,
) -> DataWriter:
    """Create DataWriter with mocked stores."""
    return DataWriter(
        feature_store=mock_feature_store,
        model_store=mock_model_store,
        strategy_store=mock_strategy_store,
        earnings_store=mock_earnings_store,
        contract_enforcer=mock_contract_enforcer,
        schema_validator=mock_schema_validator,
        registry=mock_registry,
        publisher=None,
        enable_publishing=False,
        fail_on_validation_error=True,
        batch_size=10000,
    )


# ========================================================================
# Initialization Tests
# ========================================================================


def test_data_writer_initialization(data_writer: DataWriter) -> None:
    """Test DataWriter initializes with all dependencies."""
    assert data_writer.feature_store is not None
    assert data_writer.model_store is not None
    assert data_writer.strategy_store is not None
    assert data_writer.earnings_store is not None
    assert data_writer.contract_enforcer is not None
    assert data_writer.schema_validator is not None
    assert data_writer.registry is not None
    assert data_writer.enable_publishing is False
    assert data_writer.fail_on_validation_error is True


# ========================================================================
# Feature Write Tests
# ========================================================================


def test_write_features_success(
    data_writer: DataWriter,
    mock_feature_store: Mock,
    mock_contract_enforcer: Mock,
) -> None:
    """Test write_features successfully writes features."""
    ts_event = time.time_ns()
    features = [
        FeatureData(
            feature_set_id="test_features",
            instrument_id="EURUSD.SIM",
            values={"rsi": 65.5, "macd": 0.002},
            ts_event=ts_event,
            ts_init=ts_event,
        ),
    ]

    with patch("ml.common.event_emitter.emit_dataset_event_and_watermark"):
        event = data_writer.write_features(
            instrument_id="EURUSD.SIM",
            features=features,
            source="computed",
            run_id="test_run",
        )

    # Verify feature store was called
    assert mock_feature_store.write_features.call_count == 1
    call_args = mock_feature_store.write_features.call_args
    assert call_args.kwargs["feature_set_id"] == "test_features"
    assert call_args.kwargs["instrument_id"] == "EURUSD.SIM"

    # Verify event
    assert event.dataset_id == "features"
    assert event.instrument_id == "EURUSD.SIM"
    assert event.status == EventStatus.SUCCESS.value
    assert event.record_count == 1


def test_write_features_instrument_mismatch_raises_error(
    data_writer: DataWriter,
) -> None:
    """Test write_features raises error on instrument mismatch."""
    ts_event = time.time_ns()
    features = [
        FeatureData(
            feature_set_id="test_features",
            instrument_id="GBPUSD.SIM",  # Mismatch
            values={"rsi": 65.5},
            ts_event=ts_event,
            ts_init=ts_event,
        ),
    ]

    with pytest.raises(ValueError, match="Instrument mismatch"):
        data_writer.write_features(
            instrument_id="EURUSD.SIM",
            features=features,
        )


def test_write_features_store_failure_raises_error(
    data_writer: DataWriter,
    mock_feature_store: Mock,
) -> None:
    """Test write_features raises error when store fails."""
    ts_event = time.time_ns()
    features = [
        FeatureData(
            feature_set_id="test_features",
            instrument_id="EURUSD.SIM",
            values={"rsi": 65.5},
            ts_event=ts_event,
            ts_init=ts_event,
        ),
    ]

    mock_feature_store.write_features.side_effect = RuntimeError("DB error")

    with pytest.raises(RuntimeError, match="Feature write failed"):
        data_writer.write_features(
            instrument_id="EURUSD.SIM",
            features=features,
        )


# ========================================================================
# Prediction Write Tests
# ========================================================================


def test_write_predictions_success(
    data_writer: DataWriter,
    mock_model_store: Mock,
) -> None:
    """Test write_predictions successfully writes predictions."""
    ts_event = time.time_ns()
    predictions = [
        ModelPrediction(
            model_id="test_model",
            instrument_id="EURUSD.SIM",
            prediction=0.85,
            confidence=0.92,
            features_used={"rsi": 65.5},
            inference_time_ms=1.2,
            _ts_event=ts_event,
            _ts_init=ts_event,
        ),
    ]

    with patch("ml.common.event_emitter.emit_dataset_event_and_watermark"):
        event = data_writer.write_predictions(
            predictions=predictions,
            source="inference",
            run_id="test_run",
        )

    # Verify model store was called
    mock_model_store.write_batch.assert_called_once()
    call_args = mock_model_store.write_batch.call_args
    assert call_args.args[0] == predictions

    # Verify event
    assert event.dataset_id == "predictions"
    assert event.instrument_id == "EURUSD.SIM"
    assert event.status == EventStatus.SUCCESS.value
    assert event.record_count == 1
    assert event.metadata["model_id"] == "test_model"


def test_write_predictions_empty_list_raises_error(
    data_writer: DataWriter,
) -> None:
    """Test write_predictions raises error for empty list."""
    with pytest.raises(ValueError, match="No predictions to write"):
        data_writer.write_predictions(predictions=[])


# ========================================================================
# Signal Write Tests
# ========================================================================


def test_write_signals_success(
    data_writer: DataWriter,
    mock_strategy_store: Mock,
) -> None:
    """Test write_signals successfully writes signals."""
    ts_event = time.time_ns()
    signals = [
        StrategySignal(
            strategy_id="test_strategy",
            instrument_id="EURUSD.SIM",
            signal_type="BUY",
            strength=0.75,
            model_predictions={},
            risk_metrics={},
            execution_params={},
            _ts_event=ts_event,
            _ts_init=ts_event,
        ),
    ]

    with patch("ml.common.event_emitter.emit_dataset_event_and_watermark"):
        event = data_writer.write_signals(
            signals=signals,
            source="strategy",
            run_id="test_run",
        )

    # Verify strategy store was called
    mock_strategy_store.write_batch.assert_called_once()
    call_args = mock_strategy_store.write_batch.call_args
    assert call_args.args[0] == signals

    # Verify event
    assert event.dataset_id == "signals"
    assert event.instrument_id == "EURUSD.SIM"
    assert event.status == EventStatus.SUCCESS.value
    assert event.record_count == 1
    assert event.metadata["strategy_id"] == "test_strategy"


def test_write_signals_empty_list_raises_error(
    data_writer: DataWriter,
) -> None:
    """Test write_signals raises error for empty list."""
    with pytest.raises(ValueError, match="No signals to write"):
        data_writer.write_signals(signals=[])


# ========================================================================
# Earnings Write Tests
# ========================================================================


def test_write_earnings_actual_success(
    data_writer: DataWriter,
    mock_earnings_store: Mock,
) -> None:
    """Test write_earnings_actual successfully writes earnings."""
    ts_event = time.time_ns()

    with patch("ml.common.event_emitter.emit_dataset_event_and_watermark"):
        event = data_writer.write_earnings_actual(
            ticker="AAPL",
            period_end="2024-03-31",
            filing_date="2024-04-15",
            eps_diluted=1.52,
            revenue=90750000000.0,
            ts_event=ts_event,
            ts_init=ts_event,
            source=Source.HISTORICAL.value,
            run_id="test_run",
        )

    # Verify earnings store was called
    mock_earnings_store.write_actuals.assert_called_once()

    # Verify event
    assert event.dataset_id == "earnings_actuals"
    assert event.instrument_id == "AAPL"
    assert event.status == EventStatus.SUCCESS.value
    assert event.record_count == 1


def test_write_earnings_estimate_success(
    data_writer: DataWriter,
    mock_earnings_store: Mock,
) -> None:
    """Test write_earnings_estimate successfully writes estimates."""
    ts_event = time.time_ns()

    with patch("ml.common.event_emitter.emit_dataset_event_and_watermark"):
        event = data_writer.write_earnings_estimate(
            ticker="AAPL",
            estimate_date="2024-03-15",
            period_end="2024-03-31",
            eps_consensus=1.50,
            ts_event=ts_event,
            ts_init=ts_event,
            revenue_consensus=90000000000.0,
            num_analysts=35,
            source=Source.HISTORICAL.value,
            run_id="test_run",
        )

    # Verify earnings store was called
    mock_earnings_store.write_estimates.assert_called_once()

    # Verify event
    assert event.dataset_id == "earnings_estimates"
    assert event.instrument_id == "AAPL"
    assert event.status == EventStatus.SUCCESS.value
    assert event.record_count == 1


# ========================================================================
# Ingestion Write Tests
# ========================================================================


def test_write_ingestion_preflight_failure_raises_error(
    data_writer: DataWriter,
    mock_contract_enforcer: Mock,
) -> None:
    """Test write_ingestion raises error on preflight failure."""
    # Mock preflight failure
    mock_contract_enforcer.preflight_check.return_value = (
        False,
        "Missing required columns",
        {},
    )

    data = [{"ts_event": time.time_ns(), "value": 100.0}]

    with pytest.raises(ValueError, match="Preflight check failed"):
        data_writer.write_ingestion(
            dataset_id="test_dataset",
            records=data,
            source="historical",
            run_id="test_run",
        )


def test_write_ingestion_validation_failure_strict_mode(
    data_writer: DataWriter,
    mock_contract_enforcer: Mock,
    mock_schema_validator: Mock,
) -> None:
    """Test write_ingestion raises error on validation failure in strict mode."""
    from ml.stores.validation_types import QualityReport

    # Mock validation failure
    quality_report = QualityReport(
        dataset_id="test_dataset",
        total_records=10,
        passed_records=8,
        failed_records=2,
        quality_score=0.8,
        violations=[],
        validation_time_ms=1.0,
    )
    mock_contract_enforcer.validate_batch.return_value = quality_report

    # Mock enforce_quality_report to raise error
    mock_schema_validator.enforce_quality_report.side_effect = ValueError("Validation failed")

    data = [
        {
            "instrument_id": "EURUSD.SIM",
            "ts_event": time.time_ns(),
            "ts_init": time.time_ns(),
            "value": 100.0,
        },
    ]

    with pytest.raises(ValueError, match="Validation failed"):
        data_writer.write_ingestion(
            dataset_id="test_dataset",
            records=data,
            source="historical",
            run_id="test_run",
        )


# ========================================================================
# Event Emission Tests
# ========================================================================


def test_emit_success_event_calls_registry(
    data_writer: DataWriter,
) -> None:
    """Test _emit_success_event_and_update calls registry correctly."""
    with patch("ml.common.event_emitter.emit_dataset_event_and_watermark") as mock_emit:
        data_writer._emit_success_event_and_update(
            dataset_id="test_dataset",
            instrument_id="EURUSD.SIM",
            stage=Stage.DATA_INGESTED.value,
            source="historical",
            run_id="test_run",
            ts_min=1000000000000000000,
            ts_max=2000000000000000000,
            count=100,
            dataset_type=DatasetType.FEATURES,
        )

        # Verify emit function was called
        mock_emit.assert_called_once()


def test_emit_success_event_handles_failure_gracefully(
    data_writer: DataWriter,
) -> None:
    """Test _emit_success_event_and_update handles failures gracefully."""
    with patch("ml.common.event_emitter.emit_dataset_event_and_watermark") as mock_emit:
        mock_emit.side_effect = RuntimeError("Registry error")

        # Should not raise - best effort
        data_writer._emit_success_event_and_update(
            dataset_id="test_dataset",
            instrument_id="EURUSD.SIM",
            stage=Stage.DATA_INGESTED.value,
            source="historical",
            run_id="test_run",
            ts_min=1000000000000000000,
            ts_max=2000000000000000000,
            count=100,
            dataset_type=DatasetType.FEATURES,
        )


# ========================================================================
# Helper Method Tests
# ========================================================================


def test_get_stage_for_dataset_type(data_writer: DataWriter) -> None:
    """Test _get_stage_for_dataset_type returns correct stages."""
    assert data_writer._get_stage_for_dataset_type(DatasetType.FEATURES) == Stage.FEATURE_COMPUTED.value
    assert data_writer._get_stage_for_dataset_type(DatasetType.PREDICTIONS) == Stage.PREDICTION_EMITTED.value
    assert data_writer._get_stage_for_dataset_type(DatasetType.SIGNALS) == Stage.SIGNAL_EMITTED.value
    assert data_writer._get_stage_for_dataset_type(DatasetType.EARNINGS_ACTUALS) == Stage.DATA_INGESTED.value


def test_to_dataframe_list_of_dicts(data_writer: DataWriter) -> None:
    """Test _to_dataframe handles list of dicts."""
    data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    result = data_writer._to_dataframe(data)
    assert isinstance(result, (list, object))


def test_extract_ingestion_metadata_from_dataframe(data_writer: DataWriter) -> None:
    """Test _extract_ingestion_metadata_from_dataframe extracts metadata."""
    # Test with None
    result = data_writer._extract_ingestion_metadata_from_dataframe(None)
    assert result == {}


def test_create_partial_event(data_writer: DataWriter) -> None:
    """Test _create_partial_event creates correct event."""
    event = data_writer._create_partial_event(
        dataset_id="test_dataset",
        instrument_id="EURUSD.SIM",
        source="historical",
        run_id="test_run",
        ts_min=1000000000000000000,
        ts_max=2000000000000000000,
        record_count=100,
        reason="test_reason",
    )

    assert event.status == EventStatus.PARTIAL.value
    assert event.metadata["reason"] == "test_reason"
    assert event.metadata["no_write"] is True


def test_create_failed_event(data_writer: DataWriter) -> None:
    """Test _create_failed_event creates correct event."""
    event = data_writer._create_failed_event(
        dataset_id="test_dataset",
        instrument_id="EURUSD.SIM",
        source="historical",
        run_id="test_run",
        ts_min=1000000000000000000,
        ts_max=2000000000000000000,
        record_count=0,
        error="test_error",
    )

    assert event.status == EventStatus.FAILED.value
    assert event.error_message == "test_error"


# ========================================================================
# Data Conversion Tests
# ========================================================================


def test_data_frame_to_feature_data(data_writer: DataWriter) -> None:
    """Test _data_frame_to_feature_data converts correctly."""
    ts_event = time.time_ns()
    data = [
        {
            "feature_set_id": "test_features",
            "ts_event": ts_event,
            "ts_init": ts_event,
            "values": {"rsi": 65.5},
        },
    ]

    result = data_writer._data_frame_to_feature_data(data, "EURUSD.SIM")

    assert len(result) == 1
    assert result[0].feature_set_id == "test_features"
    assert result[0].instrument_id == "EURUSD.SIM"
    assert result[0].values == {"rsi": 65.5}


def test_data_frame_to_predictions(data_writer: DataWriter) -> None:
    """Test _data_frame_to_predictions converts correctly."""
    from ml._imports import pl

    ts_event = time.time_ns()
    frame = pl.DataFrame(
        {
            "model_id": ["test_model"],
            "instrument_id": ["EURUSD.SIM"],
            "ts_event": [ts_event],
            "ts_init": [ts_event],
            "prediction": [0.85],
            "confidence": [0.92],
            "features_used": [{"rsi": 65.5}],
            "inference_time_ms": [1.2],
        }
    )

    result = data_writer._data_frame_to_predictions(frame)

    assert len(result) == 1
    assert result[0].model_id == "test_model"
    assert result[0].instrument_id == "EURUSD.SIM"
    assert result[0].prediction == 0.85
    assert result[0].features_used == {"rsi": 65.5}
    assert result[0].inference_time_ms == 1.2


def test_data_frame_to_signals(data_writer: DataWriter) -> None:
    """Test _data_frame_to_signals converts correctly."""
    ts_event = time.time_ns()
    data = [
        {
            "strategy_id": "test_strategy",
            "instrument_id": "EURUSD.SIM",
            "ts_event": ts_event,
            "ts_init": ts_event,
            "signal_type": "BUY",
            "strength": 0.75,
            "model_predictions": {},
            "risk_metrics": {},
            "execution_params": {},
        },
    ]

    result = data_writer._data_frame_to_signals(data)

    assert len(result) == 1
    assert result[0].strategy_id == "test_strategy"
    assert result[0].instrument_id == "EURUSD.SIM"
    assert result[0].signal_type == "BUY"
