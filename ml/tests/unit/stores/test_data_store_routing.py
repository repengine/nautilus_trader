"""
DataStore routing tests (DB-free) for predictions and signals datasets.

Verifies that write_ingestion routes to the correct underlying store method based on the
dataset manifest type.

"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from ml.registry.dataclasses import DataContract
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import QualityFlag
from ml.registry.dataclasses import StorageKind
from ml.registry.dataclasses import ValidationRule
from ml.registry.dataclasses import ValidationRuleType
from ml.registry.utils import compute_dataset_schema_hash
from ml.features.earnings.store import DummyEarningsStore
from ml.stores.data_store import DataStore


def _make_registry_for(
    dataset_id: str,
    dataset_type: DatasetType,
) -> MagicMock:
    # Minimal schema with Nautilus-required fields and a couple of dataset-specific fields
    if dataset_type is DatasetType.PREDICTIONS:
        schema = {
            "instrument_id": "str",
            "model_id": "str",
            "ts_event": "int64",
            "ts_init": "int64",
            "prediction": "float64",
            "confidence": "float64",
        }
    elif dataset_type is DatasetType.SIGNALS:
        schema = {
            "instrument_id": "str",
            "strategy_id": "str",
            "ts_event": "int64",
            "ts_init": "int64",
            "signal_type": "str",
            "strength": "float64",
        }
    else:
        raise AssertionError("Unsupported dataset_type for this test")

    if dataset_type is DatasetType.PREDICTIONS:
        primary_keys = ["instrument_id", "model_id", "ts_event"]
    elif dataset_type is DatasetType.SIGNALS:
        primary_keys = ["instrument_id", "strategy_id", "ts_event"]
    else:
        primary_keys = ["instrument_id", "ts_event"]

    pipeline_signature = "unit_test"
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

    contract = DataContract(
        contract_id=f"contract_{dataset_id}",
        dataset_id=dataset_id,
        version="1.0.0",
        validation_rules=[
            ValidationRule(
                rule_type=ValidationRuleType.RANGE,
                field_name="ts_event",
                parameters={"min": 0},
                severity=QualityFlag.FAIL,
                description="ts_event non-negative",
            ),
        ],
        enforcement_mode="strict",
    )

    mock_registry = MagicMock()
    mock_registry.get_manifest.return_value = manifest
    mock_registry.get_contract.return_value = contract
    return mock_registry


def test_routing_predictions_to_model_store(
    mock_model_store: MagicMock,
    mock_strategy_store: MagicMock,
    mock_feature_store: MagicMock,
) -> None:
    dataset_id = "predictions_modelA"
    registry = _make_registry_for(dataset_id, DatasetType.PREDICTIONS)

    ds = DataStore(
        connection_string="sqlite:///:memory:",
        registry=registry,
        feature_store=mock_feature_store,
        model_store=mock_model_store,
        strategy_store=mock_strategy_store,
        earnings_store=DummyEarningsStore(),
        fail_on_validation_error=False,
    )

    rows: list[dict[str, Any]] = [
        {
            "model_id": "modelA",
            "instrument_id": "EUR/USD",
            "prediction": 0.5,
            "confidence": 0.8,
            "ts_event": 123,
            "ts_init": 123,
        },
    ]

    ds.write_ingestion(
        dataset_id=dataset_id,
        records=rows,
        source="historical",
        run_id="run_1",
        instrument_id="EUR/USD",
    )

    # Model store should be called with a batch of ModelPrediction
    mock_model_store.write_batch.assert_called_once()
    args, _ = mock_model_store.write_batch.call_args
    batch = args[0]
    assert isinstance(batch, list) and len(batch) == 1
    # Strategy store should not be called
    mock_strategy_store.write_batch.assert_not_called()


def test_routing_signals_to_strategy_store(
    mock_model_store: MagicMock,
    mock_strategy_store: MagicMock,
    mock_feature_store: MagicMock,
) -> None:
    dataset_id = "signals_stratA"
    registry = _make_registry_for(dataset_id, DatasetType.SIGNALS)

    ds = DataStore(
        connection_string="sqlite:///:memory:",
        registry=registry,
        feature_store=mock_feature_store,
        model_store=mock_model_store,
        strategy_store=mock_strategy_store,
        earnings_store=DummyEarningsStore(),
        fail_on_validation_error=False,
    )

    rows: list[dict[str, Any]] = [
        {
            "strategy_id": "stratA",
            "instrument_id": "EUR/USD",
            "signal_type": "BUY",
            "strength": 0.9,
            "ts_event": 123,
            "ts_init": 123,
        },
    ]

    ds.write_ingestion(
        dataset_id=dataset_id,
        records=rows,
        source="historical",
        run_id="run_2",
        instrument_id="EUR/USD",
    )

    mock_strategy_store.write_batch.assert_called_once()
    args, _ = mock_strategy_store.write_batch.call_args
    batch = args[0]
    assert isinstance(batch, list) and len(batch) == 1
    mock_model_store.write_batch.assert_not_called()


def test_read_routing_predictions_to_model_store(
    mock_model_store: MagicMock,
    mock_strategy_store: MagicMock,
    mock_feature_store: MagicMock,
) -> None:
    dataset_id = "predictions_modelX"
    registry = _make_registry_for(dataset_id, DatasetType.PREDICTIONS)

    sentinel = [{"ok": True}]
    mock_model_store.read_predictions.return_value = sentinel

    ds = DataStore(
        connection_string="sqlite:///:memory:",
        registry=registry,
        feature_store=mock_feature_store,
        model_store=mock_model_store,
        strategy_store=mock_strategy_store,
        earnings_store=DummyEarningsStore(),
        fail_on_validation_error=False,
    )

    out = ds.read_range(
        dataset_id=dataset_id,
        instrument_id="EUR/USD",
        start_ns=111,
        end_ns=222,
    )
    assert hasattr(out, "columns")
    assert "ok" in out.columns
    assert out.to_dicts() == sentinel
    mock_model_store.read_predictions.assert_called_once_with(
        model_id="modelX",
        instrument_id="EUR/USD",
        start_ns=111,
        end_ns=222,
    )
    mock_strategy_store.read_signals.assert_not_called()


def test_read_routing_signals_to_strategy_store(
    mock_model_store: MagicMock,
    mock_strategy_store: MagicMock,
    mock_feature_store: MagicMock,
) -> None:
    dataset_id = "signals_stratB"
    registry = _make_registry_for(dataset_id, DatasetType.SIGNALS)

    sentinel = [{"ok": True}]
    mock_strategy_store.read_signals.return_value = sentinel

    ds = DataStore(
        connection_string="sqlite:///:memory:",
        registry=registry,
        feature_store=mock_feature_store,
        model_store=mock_model_store,
        strategy_store=mock_strategy_store,
        earnings_store=DummyEarningsStore(),
        fail_on_validation_error=False,
    )

    out = ds.read_range(
        dataset_id=dataset_id,
        instrument_id="EUR/USD",
        start_ns=333,
        end_ns=444,
    )
    assert hasattr(out, "columns")
    assert "ok" in out.columns
    assert out.to_dicts() == sentinel
    mock_strategy_store.read_signals.assert_called_once_with(
        strategy_id="stratB",
        instrument_id="EUR/USD",
        start_ns=333,
        end_ns=444,
    )
    mock_model_store.read_predictions.assert_not_called()
