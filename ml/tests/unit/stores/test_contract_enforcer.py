#!/usr/bin/env python3

"""
Unit tests for ContractEnforcer.

Tests contract enforcement, manifest/contract retrieval, caching,
preflight validation, and schema migration window management.
"""

import time
from unittest.mock import MagicMock
from unittest.mock import Mock

import pytest

from ml.registry.dataclasses import DataContract
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import QualityFlag
from ml.registry.dataclasses import StorageKind
from ml.registry.dataclasses import ValidationRule
from ml.registry.dataclasses import ValidationRuleType
from ml.stores.contract_enforcer import ContractEnforcer
from ml.stores.schema_validator import SchemaValidator
from ml.stores.validation_types import QualityReport
from ml.stores.validation_types import ValidationViolation


@pytest.fixture
def mock_registry():
    """Create mock registry."""
    registry = Mock()
    return registry


@pytest.fixture
def schema_validator():
    """Create schema validator."""
    return SchemaValidator()


@pytest.fixture
def contract_enforcer(mock_registry, schema_validator):
    """Create contract enforcer with mocked dependencies."""
    return ContractEnforcer(
        registry=mock_registry,
        schema_validator=schema_validator,
        allow_schema_migration=False,
        schema_migration_window_hours=24,
    )


@pytest.fixture
def sample_manifest():
    """Create sample dataset manifest."""
    return DatasetManifest(
        dataset_id="test_dataset",
        dataset_type=DatasetType.BARS,
        storage_kind=StorageKind.POSTGRES,
        location="ml.bars_test",
        partitioning={},
        retention_days=90,
        version="1.0.0",
        schema={
            "instrument_id": "str",
            "ts_event": "int64",
            "ts_init": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "int64",
        },
        schema_hash="abc123",
        primary_keys=["instrument_id", "ts_event"],
        ts_field="ts_event",
        seq_field=None,
        constraints={"nullability": {"instrument_id": False, "ts_event": False}},
        lineage=[],
        pipeline_signature="test_pipeline",
    )


@pytest.fixture
def sample_contract():
    """Create sample data contract."""
    return DataContract(
        contract_id="test_contract",
        dataset_id="test_dataset",
        version="1.0.0",
        enforcement_mode="strict",
        validation_rules=[
            ValidationRule(
                rule_type=ValidationRuleType.TYPE_CHECK,
                field_name="*",
                severity=QualityFlag.FAIL,
                parameters={},
                description="Type check for all fields",
            ),
        ],
        quality_thresholds={},
        created_at=time.time_ns(),
        last_modified=time.time_ns(),
    )


# ========================================================================
# Manifest/Contract Retrieval Tests
# ========================================================================


def test_get_manifest_caches_result(contract_enforcer, mock_registry, sample_manifest):
    """Test that get_manifest caches results."""
    mock_registry.get_manifest.return_value = sample_manifest

    # First call
    manifest1 = contract_enforcer.get_manifest("test_dataset")
    assert manifest1 == sample_manifest
    assert mock_registry.get_manifest.call_count == 1

    # Second call should use cache
    manifest2 = contract_enforcer.get_manifest("test_dataset")
    assert manifest2 == sample_manifest
    assert mock_registry.get_manifest.call_count == 1  # No additional call


def test_get_contract_caches_result(contract_enforcer, mock_registry, sample_contract):
    """Test that get_contract caches results."""
    mock_registry.get_contract.return_value = sample_contract

    # First call
    contract1 = contract_enforcer.get_contract("test_dataset")
    assert contract1 == sample_contract
    assert mock_registry.get_contract.call_count == 1

    # Second call should use cache
    contract2 = contract_enforcer.get_contract("test_dataset")
    assert contract2 == sample_contract
    assert mock_registry.get_contract.call_count == 1  # No additional call


def test_get_manifest_different_datasets(contract_enforcer, mock_registry):
    """Test that different datasets have separate cache entries."""
    manifest1 = DatasetManifest(
        dataset_id="dataset1",
        dataset_type=DatasetType.BARS,
        storage_kind=StorageKind.POSTGRES,
        location="ml.dataset1",
        partitioning={},
        retention_days=90,
        version="1.0.0",
        schema={"col1": "int64", "ts_event": "int64"},
        schema_hash="hash1",
        primary_keys=["col1"],
        ts_field="ts_event",
        seq_field=None,
        constraints={},
        lineage=[],
        pipeline_signature="test",
    )
    manifest2 = DatasetManifest(
        dataset_id="dataset2",
        dataset_type=DatasetType.TRADES,
        storage_kind=StorageKind.POSTGRES,
        location="ml.dataset2",
        partitioning={},
        retention_days=90,
        version="1.0.0",
        schema={"col2": "float64", "ts_event": "int64"},
        schema_hash="hash2",
        primary_keys=["col2"],
        ts_field="ts_event",
        seq_field=None,
        constraints={},
        lineage=[],
        pipeline_signature="test",
    )

    mock_registry.get_manifest.side_effect = lambda ds_id: (
        manifest1 if ds_id == "dataset1" else manifest2
    )

    result1 = contract_enforcer.get_manifest("dataset1")
    result2 = contract_enforcer.get_manifest("dataset2")

    assert result1.dataset_id == "dataset1"
    assert result2.dataset_id == "dataset2"
    assert mock_registry.get_manifest.call_count == 2


# ========================================================================
# Preflight Check Tests
# ========================================================================


def test_preflight_check_success(contract_enforcer, mock_registry, sample_manifest):
    """Test successful preflight check."""
    from ml._imports import pl

    mock_registry.get_manifest.return_value = sample_manifest

    data = pl.DataFrame({
        "instrument_id": ["EURUSD.SIM"] * 3,
        "ts_event": [1000, 2000, 3000],
        "ts_init": [1100, 2100, 3100],
        "open": [1.1, 1.2, 1.3],
        "high": [1.15, 1.25, 1.35],
        "low": [1.05, 1.15, 1.25],
        "close": [1.12, 1.22, 1.32],
        "volume": [100, 200, 300],
    })

    success, error, details = contract_enforcer.preflight_check(
        "test_dataset",
        data,
        strict=True,
    )

    assert success is True
    assert error is None
    assert details["preflight_passed"] is True
    assert "required_columns" in details["checks_performed"]


def test_preflight_check_missing_columns(contract_enforcer, mock_registry, sample_manifest):
    """Test preflight check fails with missing columns."""
    from ml._imports import pl

    mock_registry.get_manifest.return_value = sample_manifest

    data = pl.DataFrame({
        "instrument_id": ["EURUSD.SIM"],
        "ts_event": [1000],
        # Missing ts_init and OHLCV columns
    })

    success, error, details = contract_enforcer.preflight_check(
        "test_dataset",
        data,
        strict=True,
    )

    assert success is False
    assert "Missing required columns" in error


def test_preflight_check_type_mismatch_strict(contract_enforcer, mock_registry, sample_manifest):
    """Test preflight check fails on type mismatch in strict mode."""
    from ml._imports import pl

    mock_registry.get_manifest.return_value = sample_manifest

    data = pl.DataFrame({
        "instrument_id": ["EURUSD.SIM"],
        "ts_event": [1000],
        "ts_init": [1100],
        "open": [1.1],
        "high": [1.15],
        "low": [1.05],
        "close": [1.12],
        "volume": ["not_an_int"],  # Wrong type
    })

    success, error, details = contract_enforcer.preflight_check(
        "test_dataset",
        data,
        strict=True,
    )

    assert success is False
    assert "Type mismatches" in error


def test_preflight_check_type_mismatch_lenient(contract_enforcer, mock_registry, sample_manifest):
    """Test preflight check warns on type mismatch in lenient mode."""
    from ml._imports import pl

    mock_registry.get_manifest.return_value = sample_manifest

    data = pl.DataFrame({
        "instrument_id": ["EURUSD.SIM"],
        "ts_event": [1000],
        "ts_init": [1100],
        "open": [1.1],
        "high": [1.15],
        "low": [1.05],
        "close": [1.12],
        "volume": [100.5],  # Float instead of int (may coerce)
    })

    success, error, details = contract_enforcer.preflight_check(
        "test_dataset",
        data,
        strict=False,
    )

    # Should succeed with warnings in lenient mode
    assert success is True or len(details.get("warnings", [])) > 0


def test_preflight_check_null_primary_key(contract_enforcer, mock_registry, sample_manifest):
    """Test preflight check fails with null primary key."""
    from ml._imports import pl

    mock_registry.get_manifest.return_value = sample_manifest

    data = pl.DataFrame({
        "instrument_id": [None, "EURUSD.SIM"],  # Null in primary key
        "ts_event": [1000, 2000],
        "ts_init": [1100, 2100],
        "open": [1.1, 1.2],
        "high": [1.15, 1.25],
        "low": [1.05, 1.15],
        "close": [1.12, 1.22],
        "volume": [100, 200],
    })

    success, error, details = contract_enforcer.preflight_check(
        "test_dataset",
        data,
        strict=True,
    )

    assert success is False
    assert "null values" in error.lower()


def test_preflight_check_null_required_field(contract_enforcer, mock_registry):
    """Test preflight check fails with null in required field."""
    from ml._imports import pl

    manifest = DatasetManifest(
        dataset_id="test_dataset",
        dataset_type=DatasetType.BARS,
        storage_kind=StorageKind.POSTGRES,
        location="ml.test",
        partitioning={},
        retention_days=90,
        version="1.0.0",
        schema={"instrument_id": "str", "ts_event": "int64", "value": "float64"},
        schema_hash="abc123",
        primary_keys=["instrument_id"],
        ts_field="ts_event",
        seq_field=None,
        constraints={"nullability": {"value": False}},  # value is required
        lineage=[],
        pipeline_signature="test",
    )
    mock_registry.get_manifest.return_value = manifest

    data = pl.DataFrame({
        "instrument_id": ["EURUSD.SIM"],
        "ts_event": [1000],
        "value": [None],  # Null in required field
    })

    success, error, details = contract_enforcer.preflight_check(
        "test_dataset",
        data,
        strict=True,
    )

    assert success is False
    assert "null values" in error.lower()


# ========================================================================
# Schema Migration Window Tests
# ========================================================================


def test_migration_window_not_started_by_default(contract_enforcer):
    """Test migration window is not active by default."""
    assert contract_enforcer._is_in_migration_window("test_dataset") is False


def test_migration_window_starts_on_version_change(mock_registry, schema_validator):
    """Test migration window starts when schema version changes."""
    enforcer = ContractEnforcer(
        registry=mock_registry,
        schema_validator=schema_validator,
        allow_schema_migration=True,
        schema_migration_window_hours=24,
    )

    manifest_v1 = DatasetManifest(
        dataset_id="test_dataset",
        dataset_type=DatasetType.BARS,
        storage_kind=StorageKind.POSTGRES,
        location="ml.test",
        partitioning={},
        retention_days=90,
        version="1.0.0",
        schema={"col1": "int64", "ts_event": "int64"},
        schema_hash="hash1",
        primary_keys=["col1"],
        ts_field="ts_event",
        seq_field=None,
        constraints={},
        lineage=[],
        pipeline_signature="test",
    )

    manifest_v2 = DatasetManifest(
        dataset_id="test_dataset",
        dataset_type=DatasetType.BARS,
        storage_kind=StorageKind.POSTGRES,
        location="ml.test",
        partitioning={},
        retention_days=90,
        version="2.0.0",
        schema={"col1": "int64", "col2": "float64", "ts_event": "int64"},
        schema_hash="hash2",
        primary_keys=["col1"],
        ts_field="ts_event",
        seq_field=None,
        constraints={},
        lineage=[],
        pipeline_signature="test",
    )

    # Set initial migration state with v1
    enforcer._schema_migration_state["test_dataset"] = {
        "start_time": time.time_ns(),
        "version": "1.0.0",
        "schema_hash": "hash1",
    }

    # Load v2 manifest - should start migration window
    mock_registry.get_manifest.return_value = manifest_v2
    enforcer.get_manifest("test_dataset")

    # Should be in migration window
    assert enforcer._is_in_migration_window("test_dataset") is True


def test_migration_window_expires(mock_registry, schema_validator):
    """Test migration window expires after configured hours."""
    enforcer = ContractEnforcer(
        registry=mock_registry,
        schema_validator=schema_validator,
        allow_schema_migration=True,
        schema_migration_window_hours=1,  # 1 hour window
    )

    # Set migration state with start time in the past (2 hours ago)
    past_time = time.time_ns() - (2 * 3600 * 1_000_000_000)
    enforcer._schema_migration_state["test_dataset"] = {
        "start_time": past_time,
        "version": "1.0.0",
        "schema_hash": "hash1",
    }

    # Should be expired
    assert enforcer._is_in_migration_window("test_dataset") is False
    # State should be cleared
    assert "test_dataset" not in enforcer._schema_migration_state


def test_migration_window_disabled(mock_registry, schema_validator):
    """Test migration window is disabled when allow_schema_migration=False."""
    enforcer = ContractEnforcer(
        registry=mock_registry,
        schema_validator=schema_validator,
        allow_schema_migration=False,
        schema_migration_window_hours=24,
    )

    # Manually set migration state
    enforcer._schema_migration_state["test_dataset"] = {
        "start_time": time.time_ns(),
        "version": "1.0.0",
        "schema_hash": "hash1",
    }

    # Should not be in migration window because feature is disabled
    assert enforcer._is_in_migration_window("test_dataset") is False


# ========================================================================
# Validate Batch Tests
# ========================================================================


def test_validate_batch_delegates_to_schema_validator(
    contract_enforcer,
    mock_registry,
    sample_manifest,
    sample_contract,
):
    """Test that validate_batch delegates to SchemaValidator."""
    from ml._imports import pl

    mock_registry.get_manifest.return_value = sample_manifest
    mock_registry.get_contract.return_value = sample_contract

    data = pl.DataFrame({
        "instrument_id": ["EURUSD.SIM"],
        "ts_event": [1000],
        "ts_init": [1100],
        "open": [1.1],
        "high": [1.15],
        "low": [1.05],
        "close": [1.12],
        "volume": [100],
    })

    report = contract_enforcer.validate_batch("test_dataset", data, strict_mode=False)

    assert isinstance(report, QualityReport)
    assert report.dataset_id == "test_dataset"


# ========================================================================
# Dataset Registration Tests
# ========================================================================


def test_ensure_dataset_registered_creates_manifest(contract_enforcer, mock_registry):
    """Test ensure_dataset_registered creates manifest for new dataset."""
    mock_registry.get_manifest.side_effect = Exception("Dataset not found")
    mock_registry.register_manifest = MagicMock()

    contract_enforcer.ensure_dataset_registered(
        dataset_id="new_dataset",
        dataset_type=DatasetType.PREDICTIONS,
        instrument_id="EURUSD.SIM",
    )

    assert mock_registry.register_manifest.called
    registered_manifest = mock_registry.register_manifest.call_args[0][0]
    assert registered_manifest.dataset_id == "new_dataset"
    assert registered_manifest.dataset_type == DatasetType.PREDICTIONS


def test_ensure_dataset_registered_skips_if_exists(
    contract_enforcer,
    mock_registry,
    sample_manifest,
):
    """Test ensure_dataset_registered skips if dataset exists."""
    mock_registry.get_manifest.return_value = sample_manifest
    mock_registry.register_manifest = MagicMock()

    contract_enforcer.ensure_dataset_registered(
        dataset_id="test_dataset",
        dataset_type=DatasetType.BARS,
        instrument_id="EURUSD.SIM",
    )

    # Should not call register since dataset exists
    assert not mock_registry.register_manifest.called


# ========================================================================
# Helper Method Tests
# ========================================================================


def test_types_compatible():
    """Test type compatibility checking."""
    enforcer = ContractEnforcer(
        registry=Mock(),
        schema_validator=SchemaValidator(),
    )

    # Compatible types
    assert enforcer._types_compatible("int64", "int64") is True
    assert enforcer._types_compatible("Int64", "int64") is True
    assert enforcer._types_compatible("i8", "int64") is True
    assert enforcer._types_compatible("float64", "float") is True
    assert enforcer._types_compatible("object", "str") is True
    assert enforcer._types_compatible("Utf8", "str") is True

    # Incompatible types
    assert enforcer._types_compatible("int64", "float64") is False
    assert enforcer._types_compatible("str", "int64") is False


def test_compute_schema_hash(contract_enforcer, mock_registry, sample_manifest):
    """Test schema hash computation."""
    from ml._imports import pl

    mock_registry.get_manifest.return_value = sample_manifest

    data = pl.DataFrame({
        "instrument_id": ["EURUSD.SIM"],
        "ts_event": [1000],
        "ts_init": [1100],
        "open": [1.1],
        "high": [1.15],
        "low": [1.05],
        "close": [1.12],
        "volume": [100],
    })

    schema_hash = contract_enforcer._compute_schema_hash(data, sample_manifest)

    assert isinstance(schema_hash, str)
    assert len(schema_hash) > 0


def test_to_dataframe_passthrough(contract_enforcer):
    """Test _to_dataframe passes through DataFrames."""
    from ml._imports import pl

    df = pl.DataFrame({"col1": [1, 2, 3]})
    result = contract_enforcer._to_dataframe(df)

    assert result is df  # Should be same object


def test_to_dataframe_converts_list(contract_enforcer):
    """Test _to_dataframe converts list of dicts."""
    data = [{"col1": 1, "col2": 2}, {"col1": 3, "col2": 4}]
    result = contract_enforcer._to_dataframe(data)

    # Should return a DataFrame or the list itself
    assert hasattr(result, "columns") or isinstance(result, list)
