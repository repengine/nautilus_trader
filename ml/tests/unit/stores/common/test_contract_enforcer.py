#!/usr/bin/env python3

"""
Unit tests for ContractEnforcerComponent.

Tests contract retrieval, schema migration, and quality enforcement for Phase 2.4.5.

"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock
from unittest.mock import patch
import time

import pytest

from ml.registry.dataclasses import DataContract
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import QualityFlag
from ml.registry.dataclasses import StorageKind
from ml.registry.dataclasses import ValidationRule
from ml.registry.dataclasses import ValidationRuleType
from ml.registry.utils import compute_dataset_schema_hash
from ml.stores.common.contract_enforcer import ContractEnforcerComponent
from ml.stores.common.schema_validator import QualityReport
from ml.stores.common.schema_validator import ValidationViolation


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def mock_registry() -> MagicMock:
    """
    Create mock DataRegistry with default manifest and contract.
    """
    registry = MagicMock()

    # Default OHLCV manifest
    schema = {
        "instrument_id": "str",
        "ts_event": "int64",
        "ts_init": "int64",
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64",
    }

    schema_hash = compute_dataset_schema_hash(
        schema=schema,
        primary_keys=["instrument_id", "ts_event"],
        ts_field="ts_event",
        seq_field=None,
        pipeline_signature="test",
    )

    manifest = DatasetManifest(
        dataset_id="bars_eurusd_1m",
        dataset_type=DatasetType.BARS,
        storage_kind=StorageKind.POSTGRES,
        location="bars_eurusd_1m",
        partitioning={"by": "ts_event", "interval": "daily"},
        retention_days=365,
        schema=schema,
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["instrument_id", "ts_event"],
        schema_hash=schema_hash,
        constraints={"nullability": {"instrument_id": False, "ts_event": False}},
        lineage=[],
        pipeline_signature="test",
        version="1.0.0",
    )

    contract = DataContract(
        contract_id="bars_contract",
        dataset_id="bars_eurusd_1m",
        version="1.0.0",
        validation_rules=[
            ValidationRule(
                rule_type=ValidationRuleType.TYPE_CHECK,
                field_name="*",
                parameters={},
                severity=QualityFlag.FAIL,
                description="Type checking",
            ),
            ValidationRule(
                rule_type=ValidationRuleType.RANGE,
                field_name="close",
                parameters={"min": 0.0},
                severity=QualityFlag.FAIL,
                description="Close price must be positive",
            ),
            ValidationRule(
                rule_type=ValidationRuleType.MONOTONICITY,
                field_name="ts_event",
                parameters={"direction": "increasing", "strict": True},
                severity=QualityFlag.FAIL,
                description="Timestamps must be strictly increasing",
            ),
        ],
        quality_thresholds={"null_rate": 0.01, "duplicate_rate": 0.0},
        enforcement_mode="strict",
    )

    registry.get_manifest.return_value = manifest
    registry.get_contract.return_value = contract

    return registry


@pytest.fixture
def contract_enforcer(mock_registry: MagicMock) -> ContractEnforcerComponent:
    """
    Provide isolated ContractEnforcerComponent for unit testing.
    """
    return ContractEnforcerComponent(
        registry=mock_registry,
        allow_schema_migration=True,
        schema_migration_window_hours=24,
        fail_on_validation_error=True,
    )


# =========================================================================
# Manifest Retrieval Tests
# =========================================================================


def test_get_manifest_from_registry(contract_enforcer: ContractEnforcerComponent, mock_registry: MagicMock) -> None:
    """
    Test that manifest is retrieved from registry on first access.
    """
    # Execute
    manifest = contract_enforcer.get_manifest("bars_eurusd_1m")

    # Verify
    assert manifest.dataset_id == "bars_eurusd_1m"
    assert manifest.version == "1.0.0"
    assert "ts_event" in manifest.schema
    assert "close" in manifest.schema
    mock_registry.get_manifest.assert_called_once_with("bars_eurusd_1m")


def test_get_manifest_caching(contract_enforcer: ContractEnforcerComponent, mock_registry: MagicMock) -> None:
    """
    Test that manifest is cached and registry is only called once.
    """
    # Execute - call twice
    manifest1 = contract_enforcer.get_manifest("bars_eurusd_1m")
    manifest2 = contract_enforcer.get_manifest("bars_eurusd_1m")

    # Verify - same object returned, registry called only once
    assert manifest1 is manifest2
    assert mock_registry.get_manifest.call_count == 1


def test_get_manifest_version_change_starts_migration(
    contract_enforcer: ContractEnforcerComponent,
    mock_registry: MagicMock,
) -> None:
    """
    Test that schema version change triggers migration window start.
    """
    # Setup: Set initial migration state with old version
    contract_enforcer._schema_migration_state["bars_eurusd_1m"] = {
        "start_time": time.time_ns(),
        "version": "0.9.0",
        "schema_hash": "old_hash",
    }

    # Setup: Create new manifest with updated version using dataclasses.replace
    base_manifest = mock_registry.get_manifest.return_value
    new_manifest = replace(base_manifest, version="2.0.0")
    mock_registry.get_manifest.return_value = new_manifest

    # Execute
    manifest = contract_enforcer.get_manifest("bars_eurusd_1m")

    # Verify - migration window should be started (logged)
    assert manifest.version == "2.0.0"
    assert "bars_eurusd_1m" in contract_enforcer._schema_migration_state


# =========================================================================
# Contract Retrieval Tests
# =========================================================================


def test_get_contract_from_registry(contract_enforcer: ContractEnforcerComponent, mock_registry: MagicMock) -> None:
    """
    Test that contract is retrieved from registry on first access.
    """
    # Execute
    contract = contract_enforcer.get_contract("bars_eurusd_1m")

    # Verify
    assert contract.dataset_id == "bars_eurusd_1m"
    assert contract.version == "1.0.0"
    assert contract.enforcement_mode == "strict"
    assert len(contract.validation_rules) == 3
    mock_registry.get_contract.assert_called_once_with("bars_eurusd_1m")


def test_get_contract_caching(contract_enforcer: ContractEnforcerComponent, mock_registry: MagicMock) -> None:
    """
    Test that contract is cached and registry is only called once.
    """
    # Execute - call twice
    contract1 = contract_enforcer.get_contract("bars_eurusd_1m")
    contract2 = contract_enforcer.get_contract("bars_eurusd_1m")

    # Verify - same object returned, registry called only once
    assert contract1 is contract2
    assert mock_registry.get_contract.call_count == 1


# =========================================================================
# Schema Hash Tests
# =========================================================================


def test_compute_schema_hash_deterministic(contract_enforcer: ContractEnforcerComponent, mock_registry: MagicMock) -> None:
    """
    Test that schema hash is deterministic for same data.
    """
    from ml._imports import HAS_POLARS, pl

    if not HAS_POLARS:
        pytest.skip("Polars not available")

    # Setup
    manifest = mock_registry.get_manifest.return_value
    df = pl.DataFrame({
        "instrument_id": ["EURUSD.SIM"],
        "ts_event": [1699999990000000000],
        "ts_init": [1699999990000000000],
        "open": [1.0800],
        "high": [1.0850],
        "low": [1.0750],
        "close": [1.0820],
        "volume": [1000.0],
    })

    # Execute - compute hash twice
    hash1 = contract_enforcer.compute_schema_hash(df, manifest)
    hash2 = contract_enforcer.compute_schema_hash(df, manifest)

    # Verify - hashes match (deterministic)
    assert hash1 == hash2
    assert isinstance(hash1, str)
    assert len(hash1) > 0


def test_compute_schema_hash_changes_on_schema_change(
    contract_enforcer: ContractEnforcerComponent,
    mock_registry: MagicMock,
) -> None:
    """
    Test that schema hash changes when manifest schema differs.
    """
    from ml._imports import HAS_POLARS, pl

    if not HAS_POLARS:
        pytest.skip("Polars not available")

    # Setup - create two different manifests with different schemas
    base_manifest = mock_registry.get_manifest.return_value

    # Manifest 1: minimal schema (3 required + 1 optional columns)
    manifest1 = replace(
        base_manifest,
        schema={
            "instrument_id": "str",
            "ts_event": "int64",
            "ts_init": "int64",  # Required for bars dataset type
            "close": "float64",
        },
    )

    # Manifest 2: extended schema (3 required + 2 optional columns)
    manifest2 = replace(
        base_manifest,
        schema={
            "instrument_id": "str",
            "ts_event": "int64",
            "ts_init": "int64",  # Required for bars dataset type
            "close": "float64",
            "volume": "float64",  # Additional column in manifest
        },
    )

    # Same DataFrame used for both
    df = pl.DataFrame({
        "instrument_id": ["EURUSD.SIM"],
        "ts_event": [1699999990000000000],
        "close": [1.0820],
    })

    # Execute
    hash1 = contract_enforcer.compute_schema_hash(df, manifest1)
    hash2 = contract_enforcer.compute_schema_hash(df, manifest2)

    # Verify - hashes differ because manifest schemas differ
    assert hash1 != hash2


# =========================================================================
# Migration Window Tests
# =========================================================================


def test_is_in_migration_window_active(contract_enforcer: ContractEnforcerComponent, mock_registry: MagicMock) -> None:
    """
    Test that active migration window is detected.
    """
    # Setup - start migration window
    manifest = mock_registry.get_manifest.return_value
    contract_enforcer.start_migration_window("bars_eurusd_1m", manifest)

    # Execute
    result = contract_enforcer.is_in_migration_window("bars_eurusd_1m")

    # Verify - window is active
    assert result is True


def test_is_in_migration_window_expired(contract_enforcer: ContractEnforcerComponent, mock_registry: MagicMock) -> None:
    """
    Test that expired migration window is cleaned up.
    """
    # Setup - start migration window with expired timestamp
    contract_enforcer._schema_migration_state["bars_eurusd_1m"] = {
        "start_time": time.time_ns() - (25 * 3600 * 1e9),  # 25 hours ago (expired)
        "version": "1.0.0",
        "schema_hash": "test_hash",
    }

    # Execute
    result = contract_enforcer.is_in_migration_window("bars_eurusd_1m")

    # Verify - window expired and state cleared
    assert result is False
    assert "bars_eurusd_1m" not in contract_enforcer._schema_migration_state


def test_start_migration_window(contract_enforcer: ContractEnforcerComponent, mock_registry: MagicMock) -> None:
    """
    Test that migration window is started with correct state.
    """
    # Setup
    manifest = mock_registry.get_manifest.return_value

    # Execute
    contract_enforcer.start_migration_window("bars_eurusd_1m", manifest)

    # Verify - migration state created
    assert "bars_eurusd_1m" in contract_enforcer._schema_migration_state
    state = contract_enforcer._schema_migration_state["bars_eurusd_1m"]
    assert "start_time" in state
    assert state["version"] == "1.0.0"
    assert "schema_hash" in state


def test_migration_window_respects_allow_flag(mock_registry: MagicMock) -> None:
    """
    Test that migration window is not started when allow_schema_migration=False.
    """
    # Setup - enforcer with migration disabled
    enforcer = ContractEnforcerComponent(
        registry=mock_registry,
        allow_schema_migration=False,
    )

    # Execute
    manifest = mock_registry.get_manifest.return_value
    enforcer.start_migration_window("bars_eurusd_1m", manifest)
    result = enforcer.is_in_migration_window("bars_eurusd_1m")

    # Verify - window not active (flag disabled)
    assert result is False


# =========================================================================
# Quality Enforcement Tests
# =========================================================================


def test_enforce_quality_strict_mode_success(
    contract_enforcer: ContractEnforcerComponent,
    mock_registry: MagicMock,
) -> None:
    """
    Test that quality enforcement passes with perfect score.
    """
    # Setup - perfect quality report
    report = QualityReport(
        dataset_id="bars_eurusd_1m",
        total_records=100,
        passed_records=100,
        failed_records=0,
        quality_score=1.0,
        violations=[],
        validation_time_ms=1.5,
        metadata={},
    )
    contract = mock_registry.get_contract.return_value

    # Execute - should not raise
    contract_enforcer.enforce_quality(report, contract, "bars_eurusd_1m")

    # Verify - no exception raised


def test_enforce_quality_strict_mode_failure(
    contract_enforcer: ContractEnforcerComponent,
    mock_registry: MagicMock,
) -> None:
    """
    Test that quality enforcement fails in strict mode with violations.
    """
    # Setup - quality report with critical violations
    violations = [
        ValidationViolation(
            rule_type=ValidationRuleType.RANGE,
            field_name="close",
            severity=QualityFlag.FAIL,
            violation_count=5,
            sample_values=[-1.0, -2.0],
            description="negative value",
        ),
    ]
    report = QualityReport(
        dataset_id="bars_eurusd_1m",
        total_records=100,
        passed_records=95,
        failed_records=5,
        quality_score=0.95,
        violations=violations,
        validation_time_ms=2.0,
        metadata={},
    )
    contract = mock_registry.get_contract.return_value

    # Execute - should raise ValueError
    with pytest.raises(ValueError, match=r"Data validation failed.*fail-closed"):
        contract_enforcer.enforce_quality(report, contract, "bars_eurusd_1m")


def test_enforce_quality_lenient_mode(
    mock_registry: MagicMock,
) -> None:
    """
    Test that quality enforcement logs warnings in lenient mode.
    """
    # Setup - enforcer with lenient contract
    enforcer = ContractEnforcerComponent(
        registry=mock_registry,
        fail_on_validation_error=False,
    )
    # Create lenient contract using dataclasses.replace
    base_contract = mock_registry.get_contract.return_value
    lenient_contract = replace(base_contract, enforcement_mode="lenient")
    mock_registry.get_contract.return_value = lenient_contract

    violations = [
        ValidationViolation(
            rule_type=ValidationRuleType.RANGE,
            field_name="close",
            severity=QualityFlag.WARN,
            violation_count=5,
            sample_values=[-0.5, -0.3],
            description="negative value",
        ),
    ]
    report = QualityReport(
        dataset_id="bars_eurusd_1m",
        total_records=100,
        passed_records=95,
        failed_records=5,
        quality_score=0.95,
        violations=violations,
        validation_time_ms=1.8,
        metadata={},
    )

    # Execute - should log warning but not raise
    enforcer.enforce_quality(report, lenient_contract, "bars_eurusd_1m")

    # Verify - no exception raised (lenient mode)


def test_enforce_quality_monitor_only(
    mock_registry: MagicMock,
) -> None:
    """
    Test that quality enforcement logs info in monitor-only mode.
    """
    # Setup - enforcer with monitor-only contract
    enforcer = ContractEnforcerComponent(
        registry=mock_registry,
        fail_on_validation_error=False,
    )
    # Create monitor-only contract using dataclasses.replace
    base_contract = mock_registry.get_contract.return_value
    monitor_contract = replace(base_contract, enforcement_mode="monitor_only")
    mock_registry.get_contract.return_value = monitor_contract

    violations = [
        ValidationViolation(
            rule_type=ValidationRuleType.RANGE,
            field_name="close",
            severity=QualityFlag.FAIL,  # Even FAIL violations don't raise in monitor_only
            violation_count=5,
            sample_values=[-1.5, -2.0],
            description="negative value",
        ),
    ]
    report = QualityReport(
        dataset_id="bars_eurusd_1m",
        total_records=100,
        passed_records=95,
        failed_records=5,
        quality_score=0.95,
        violations=violations,
        validation_time_ms=1.6,
        metadata={},
    )

    # Execute - should log info but not raise
    enforcer.enforce_quality(report, monitor_contract, "bars_eurusd_1m")

    # Verify - no exception raised (monitor_only mode)


# =========================================================================
# Violation Formatting Tests
# =========================================================================


def test_format_violations_readable(contract_enforcer: ContractEnforcerComponent) -> None:
    """
    Test that violations are formatted as human-readable strings.
    """
    # Setup
    violations = [
        ValidationViolation(
            rule_type=ValidationRuleType.RANGE,
            field_name="close",
            severity=QualityFlag.FAIL,
            violation_count=10,
            sample_values=[-1.0, -2.0],
            description="negative value",
        ),
        ValidationViolation(
            rule_type=ValidationRuleType.NULLABILITY,
            field_name="volume",
            severity=QualityFlag.WARN,
            violation_count=5,
            sample_values=[None, None, None],
            description="null value",
        ),
    ]

    # Execute
    formatted = contract_enforcer.format_violations(violations)

    # Verify - readable format with counts
    assert "close: negative value (10 records)" in formatted
    assert "volume: null value (5 records)" in formatted


def test_format_violations_empty(contract_enforcer: ContractEnforcerComponent) -> None:
    """
    Test that empty violations list returns 'None'.
    """
    # Execute
    formatted = contract_enforcer.format_violations([])

    # Verify
    assert formatted == "None"


def test_format_violations_truncates_long_list(contract_enforcer: ContractEnforcerComponent) -> None:
    """
    Test that violation list is truncated to first 3 violations.
    """
    # Setup - 5 violations
    violations = [
        ValidationViolation(
            rule_type=ValidationRuleType.RANGE,
            field_name=f"field_{i}",
            severity=QualityFlag.FAIL,
            violation_count=i,
            sample_values=[float(i), float(i + 1), float(i + 2)],
            description="violation",
        )
        for i in range(5)
    ]

    # Execute
    formatted = contract_enforcer.format_violations(violations)

    # Verify - only first 3 shown, plus "... and 2 more"
    assert "field_0" in formatted
    assert "field_1" in formatted
    assert "field_2" in formatted
    assert "... and 2 more" in formatted
    assert "field_3" not in formatted
    assert "field_4" not in formatted


# =========================================================================
# Schema Migration Tests
# =========================================================================


def test_migrate_schema_success(contract_enforcer: ContractEnforcerComponent) -> None:
    """
    Test that schema migration returns data (simplified implementation).
    """
    from ml._imports import HAS_POLARS, pl

    if not HAS_POLARS:
        pytest.skip("Polars not available")

    # Setup
    df = pl.DataFrame({"old_column": [1, 2, 3]})

    # Execute
    migrated = contract_enforcer.migrate_schema("1.0", "2.0", df)

    # Verify - data returned (simplified migration)
    assert len(migrated) == 3
    assert isinstance(migrated, pl.DataFrame)


# =========================================================================
# Additional API Tests
# =========================================================================


def test_register_contract_logs(contract_enforcer: ContractEnforcerComponent) -> None:
    """
    Test that register_contract logs registration.
    """
    # Execute - should log (convenience method)
    contract_enforcer.register_contract("test_contract", {}, "1.0.0")

    # Verify - no exception (convenience method logs but doesn't persist)


def test_update_contract_clears_cache(
    contract_enforcer: ContractEnforcerComponent,
    mock_registry: MagicMock,
) -> None:
    """
    Test that update_contract clears cache.
    """
    # Setup - populate cache
    contract_enforcer.get_contract("bars_eurusd_1m")
    assert "bars_eurusd_1m" in contract_enforcer._contract_cache

    # Execute
    contract_enforcer.update_contract("bars_eurusd_1m", {}, "1.1.0")

    # Verify - cache cleared
    assert "bars_eurusd_1m" not in contract_enforcer._contract_cache


def test_register_manifest_logs(contract_enforcer: ContractEnforcerComponent) -> None:
    """
    Test that register_manifest logs registration.
    """
    # Execute - should log (convenience method)
    contract_enforcer.register_manifest("test_manifest", {})

    # Verify - no exception (convenience method logs but doesn't persist)


def test_validate_contract_success(contract_enforcer: ContractEnforcerComponent) -> None:
    """
    Test that validate_contract returns True when contract exists.
    """
    from ml._imports import HAS_POLARS, pl

    if not HAS_POLARS:
        pytest.skip("Polars not available")

    # Setup
    df = pl.DataFrame({"close": [1.0, 2.0, 3.0]})

    # Execute
    result = contract_enforcer.validate_contract("bars_eurusd_1m", df)

    # Verify - validation succeeds (simplified implementation)
    assert result is True


def test_validate_contract_failure(mock_registry: MagicMock) -> None:
    """
    Test that validate_contract returns False when contract missing.
    """
    from ml._imports import HAS_POLARS, pl

    if not HAS_POLARS:
        pytest.skip("Polars not available")

    # Setup - registry raises exception
    mock_registry.get_contract.side_effect = KeyError("Contract not found")
    enforcer = ContractEnforcerComponent(registry=mock_registry)
    df = pl.DataFrame({"close": [1.0, 2.0, 3.0]})

    # Execute
    result = enforcer.validate_contract("missing_contract", df)

    # Verify - validation fails
    assert result is False
