"""Tests for DataStore earnings write paths."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ml.registry.dataclasses import DataContract
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import QualityFlag
from ml.registry.dataclasses import ValidationRule
from ml.registry.dataclasses import ValidationRuleType
from ml.stores import DataStore
from ml.stores.validation_types import QualityReport


def _make_contract(dataset_id: str) -> DataContract:
    return DataContract(
        contract_id=f"{dataset_id}_contract",
        dataset_id=dataset_id,
        version="1.0.0",
        validation_rules=[
            ValidationRule(
                rule_type=ValidationRuleType.REGEX,
                field_name="ticker",
                parameters={"pattern": ".*"},
                severity=QualityFlag.WARN,
                description="allow any ticker",
            ),
        ],
    )


def _make_manifest(dataset_id: str) -> DatasetManifest:
    return DatasetManifest(
        dataset_id=dataset_id,
        description="test manifest",
        dataset_type=DatasetType.EARNINGS_ACTUALS if "actual" in dataset_id else DatasetType.EARNINGS_ESTIMATES,
        schema_hash="test",
        ts_field="ts_event",
    )


def _make_store(raw_writer: MagicMock) -> DataStore:
    registry = MagicMock()
    registry.get_manifest.side_effect = lambda dataset_id: _make_manifest(dataset_id)
    registry.get_contract.side_effect = lambda dataset_id: _make_contract(dataset_id)

    mock_earnings_store = MagicMock()
    mock_feature_store = MagicMock()
    mock_model_store = MagicMock()
    mock_strategy_store = MagicMock()

    # Create mock schema validator that returns passing quality reports
    mock_schema_validator = MagicMock()

    def _quality_report(dataset_id: str, *_args: object, **_kwargs: object) -> QualityReport:
        return QualityReport(
            dataset_id=dataset_id,
            total_records=1,
            passed_records=1,
            failed_records=0,
            quality_score=1.0,
            violations=[],
            validation_time_ms=0.0,
        )

    mock_schema_validator.preflight_check.return_value = (True, None, {})
    mock_schema_validator.validate_batch.side_effect = _quality_report

    # Create DataStore with all required dependencies for facade
    with patch("ml.stores.data_store.DataStoreFacade._create_schema_validator", return_value=mock_schema_validator):
        store = DataStore(
            connection_string="postgresql://unused",
            registry=registry,
            feature_store=mock_feature_store,
            model_store=mock_model_store,
            strategy_store=mock_strategy_store,
            earnings_store=mock_earnings_store,
            raw_writer=raw_writer,
            schema_validator=mock_schema_validator,
        )

    return store


def test_write_earnings_actual_invokes_raw_writer() -> None:
    raw_writer = MagicMock()
    store = _make_store(raw_writer)

    event = store.write_earnings_actual(
        ticker="MSFT",
        period_end="2024-06-30",
        filing_date="2024-08-01",
        eps_diluted=2.45,
        revenue=62000000000.0,
        ts_event=1722470400000000000,
        ts_init=1722470400000000000,
    )

    raw_writer.write.assert_called_once()
    kwargs = raw_writer.write.call_args.kwargs
    assert kwargs["dataset_type"] == DatasetType.EARNINGS_ACTUALS
    assert kwargs["data"][0]["ticker"] == "MSFT"
    assert event.metadata["raw_writer_status"] == "ok"


def test_write_earnings_estimate_reports_raw_writer_failures() -> None:
    raw_writer = MagicMock()
    raw_writer.write.side_effect = RuntimeError("disk full")
    store = _make_store(raw_writer)

    event = store.write_earnings_estimate(
        ticker="AAPL",
        estimate_date="2024-09-15",
        period_end="2024-09-30",
        eps_consensus=1.42,
        ts_event=1726358400000000000,
        ts_init=1726358400000000000,
    )

    raw_writer.write.assert_called_once()
    kwargs = raw_writer.write.call_args.kwargs
    assert kwargs["dataset_type"] == DatasetType.EARNINGS_ESTIMATES
    assert kwargs["data"][0]["ticker"] == "AAPL"
    assert event.metadata["raw_writer_status"] == "failed"
