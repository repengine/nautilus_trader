"""
Unit tests covering DataStore validation rules (range, uniqueness, monotonicity,
nullability).
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pandas as pd

from ml.registry.dataclasses import (
    DataContract,
    DatasetManifest,
    DatasetType,
    QualityFlag,
    StorageKind,
    ValidationRule,
    ValidationRuleType,
)
from ml.stores.data_store_facade import DataStore


def _make_store_with_contract(rules: list[ValidationRule]) -> DataStore:
    schema = {
        "instrument_id": "str",
        "ts_event": "int64",
        "ts_init": "int64",
        "value": "float64",
    }
    manifest = DatasetManifest(
        dataset_id="test_ds",
        dataset_type=DatasetType.BARS,
        storage_kind=StorageKind.POSTGRES,
        location="ml_bars",
        partitioning={"by": "ts_event"},
        retention_days=365,
        schema=schema,
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["instrument_id", "ts_event"],
        schema_hash="hash",
        constraints={"nullability": {"instrument_id": False, "ts_event": False, "ts_init": False}},
        lineage=[],
        pipeline_signature="unit",
        version="1.0.0",
    )
    contract = DataContract(
        contract_id="c1",
        dataset_id="test_ds",
        version="1.0.0",
        validation_rules=rules,
        enforcement_mode="strict",
    )
    reg = MagicMock()
    reg.get_manifest.return_value = manifest
    reg.get_contract.return_value = contract

    from ml.stores.base import DummyStore

    return cast(Any, DataStore)(
        connection_string="sqlite:///:memory:",
        registry=reg,
        feature_store=cast(Any, DummyStore()),
        model_store=cast(Any, DummyStore()),
        strategy_store=cast(Any, DummyStore()),
        fail_on_validation_error=False,
    )


def test_validate_range_rule_violation() -> None:
    rules = [
        ValidationRule(
            ValidationRuleType.RANGE,
            "value",
            {"min": 0.0, "max": 1.0},
            QualityFlag.FAIL,
            "range",
        ),
    ]
    store = _make_store_with_contract(rules)
    df = pd.DataFrame(
        [
            {"instrument_id": "X", "ts_event": 1, "ts_init": 1, "value": 1.2},
        ],
    )
    report = store.validate_batch("test_ds", df)
    assert report.violations and any(v.field_name == "value" for v in report.violations)


def test_validate_uniqueness_violation() -> None:
    rules = [
        ValidationRule(
            ValidationRuleType.UNIQUENESS,
            "instrument_id,ts_event",
            {},
            QualityFlag.FAIL,
            "uniq",
        ),
    ]
    store = _make_store_with_contract(rules)
    df = pd.DataFrame(
        [
            {"instrument_id": "X", "ts_event": 1, "ts_init": 1, "value": 0.1},
            {"instrument_id": "X", "ts_event": 1, "ts_init": 1, "value": 0.2},
        ],
    )
    report = store.validate_batch("test_ds", df)
    assert report.violations and report.violations[0].rule_type == ValidationRuleType.UNIQUENESS


def test_validate_monotonicity_violation() -> None:
    rules = [
        ValidationRule(
            ValidationRuleType.MONOTONICITY,
            "ts_event",
            {"direction": "increasing", "strict": True},
            QualityFlag.FAIL,
            "mono",
        ),
    ]
    store = _make_store_with_contract(rules)
    df = pd.DataFrame(
        [
            {"instrument_id": "X", "ts_event": 2, "ts_init": 2, "value": 0.1},
            {"instrument_id": "X", "ts_event": 1, "ts_init": 1, "value": 0.2},
        ],
    )
    report = store.validate_batch("test_ds", df)
    assert report.violations and report.violations[0].rule_type == ValidationRuleType.MONOTONICITY


def test_validate_required_field_nullability_violation() -> None:
    # Explicit nullability rule for instrument_id
    rules = [
        ValidationRule(
            ValidationRuleType.NULLABILITY,
            "instrument_id",
            {"nullable": False},
            QualityFlag.FAIL,
            "nonnull",
        ),
    ]
    store = _make_store_with_contract(rules)
    df = pd.DataFrame([{"instrument_id": None, "ts_event": 1, "ts_init": 1, "value": 0.1}])
    report = store.validate_batch("test_ds", df)
    assert report.violations and any("null" in v.description.lower() for v in report.violations)


def test_validate_type_check_violation() -> None:
    # Expect a type mismatch on 'value' when schema expects float64
    rules = [ValidationRule(ValidationRuleType.TYPE_CHECK, "*", {}, QualityFlag.FAIL, "types")]
    store = _make_store_with_contract(rules)
    df = pd.DataFrame(
        [
            {"instrument_id": "X", "ts_event": 1, "ts_init": 1, "value": "not_a_float"},
        ],
    )
    report = store.validate_batch("test_ds", df)
    assert report.violations and any(
        v.rule_type == ValidationRuleType.TYPE_CHECK for v in report.violations
    )


def test_validate_lateness_violation() -> None:
    # Very small lateness threshold to force violation
    rules = [
        ValidationRule(
            ValidationRuleType.LATENESS,
            "ts_event",
            {"max_lateness_ns": 1},
            QualityFlag.FAIL,
            "lateness",
        ),
    ]
    store = _make_store_with_contract(rules)
    df = pd.DataFrame(
        [
            {"instrument_id": "X", "ts_event": 0, "ts_init": 0, "value": 0.1},
        ],
    )
    report = store.validate_batch("test_ds", df)
    assert report.violations and any(
        v.rule_type == ValidationRuleType.LATENESS for v in report.violations
    )


def _make_store_with_thresholds(null_rate_threshold: float) -> DataStore:
    schema = {
        "instrument_id": "str",
        "ts_event": "int64",
        "ts_init": "int64",
        "value": "float64",
    }
    manifest = DatasetManifest(
        dataset_id="test_ds",
        dataset_type=DatasetType.BARS,
        storage_kind=StorageKind.POSTGRES,
        location="ml_bars",
        partitioning={"by": "ts_event"},
        retention_days=365,
        schema=schema,
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["instrument_id", "ts_event"],
        schema_hash="hash",
        constraints={"nullability": {"instrument_id": False, "ts_event": False, "ts_init": False}},
        lineage=[],
        pipeline_signature="unit",
        version="1.0.0",
    )
    # Include a benign rule to satisfy contract requirement
    contract = DataContract(
        contract_id="c1",
        dataset_id="test_ds",
        version="1.0.0",
        validation_rules=[
            ValidationRule(ValidationRuleType.RANGE, "value", {"min": 0.0}, QualityFlag.WARN, "ok"),
        ],
        quality_thresholds={"null_rate": null_rate_threshold},
        enforcement_mode="strict",
    )
    reg = MagicMock()
    reg.get_manifest.return_value = manifest
    reg.get_contract.return_value = contract

    from ml.stores.base import DummyStore
    from typing import Any, cast as _cast

    return DataStore(
        connection_string="sqlite:///:memory:",
        registry=reg,
        feature_store=_cast(Any, DummyStore()),
        model_store=_cast(Any, DummyStore()),
        strategy_store=_cast(Any, DummyStore()),
        fail_on_validation_error=False,
    )


def test_quality_thresholds_null_rate_warn_default_and_strict() -> None:
    store = _make_store_with_thresholds(null_rate_threshold=0.0)
    # Construct a DataFrame with a null
    df = pd.DataFrame(
        [
            {"instrument_id": "X", "ts_event": 1, "ts_init": 1, "value": None},
        ],
    )
    # Default mode
    report1 = store.validate_batch("test_ds", df, strict_mode=False)
    assert any(
        v.rule_type == ValidationRuleType.NULLABILITY and v.severity == QualityFlag.WARN
        for v in report1.violations
    )
    # Strict mode still records threshold breach as WARN (rule escalation doesn't apply here)
    report2 = store.validate_batch("test_ds", df, strict_mode=True)
    assert any(
        v.rule_type == ValidationRuleType.NULLABILITY and v.severity == QualityFlag.WARN
        for v in report2.violations
    )
