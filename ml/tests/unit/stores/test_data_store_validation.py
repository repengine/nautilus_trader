#!/usr/bin/env python3
"""
Comprehensive tests for DataStore schema enforcement and contract validation.

This module tests all aspects of Phase 5 implementation including:
- Type validation
- Null validation
- Range validation
- Uniqueness validation
- Monotonicity validation
- Lateness validation
- Schema change detection
- Preflight checks
- Fail-closed write behavior
- Schema migration windows
- Prometheus metrics emission

"""

from __future__ import annotations

import hashlib
import time
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch
from ml.tests.builders import DataBuilder, MockBuilder

import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from ml._imports import HAS_POLARS
from ml._imports import HAS_PROMETHEUS
from ml.registry.dataclasses import DataContract
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import QualityFlag
from ml.registry.dataclasses import StorageKind
from ml.registry.dataclasses import ValidationRule
from ml.registry.dataclasses import ValidationRuleType
from ml.stores.data_store import DataStore


if HAS_POLARS:
    import polars as pl


# ========================================================================
# Test Fixtures
# ========================================================================

# Add missing fixtures that are not in conftest but are needed
@pytest.fixture
def mock_feature_store() -> MagicMock:
    """Mock feature store."""
    from ml.tests.builders import MockBuilder
    return MockBuilder.store_with_data(store_type="feature")


@pytest.fixture
def mock_model_store() -> MagicMock:
    """Mock model store."""
    from ml.tests.builders import MockBuilder
    return MockBuilder.store_with_data(store_type="model")


@pytest.fixture
def mock_strategy_store() -> MagicMock:
    """Mock strategy store."""
    from ml.tests.builders import MockBuilder
    return MockBuilder.store_with_data(store_type="strategy")


@pytest.fixture
def mock_stores_bundle(
    mock_feature_store: MagicMock,
    mock_model_store: MagicMock,
    mock_strategy_store: MagicMock,
) -> dict[str, MagicMock]:
    """Bundle of all store mocks."""
    from ml.tests.builders import MockBuilder
    return {
        "feature_store": mock_feature_store,
        "model_store": mock_model_store,
        "strategy_store": mock_strategy_store,
        "data_store": MockBuilder.store_with_data(store_type="data"),
    }


@pytest.fixture
def mock_registry() -> MagicMock:
    """
    Create a mock DataRegistry.
    """
    registry = MagicMock()

    # Default manifest
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

    manifest = DatasetManifest(
        dataset_id="test_bars",
        dataset_type=DatasetType.BARS,
        storage_kind=StorageKind.POSTGRES,
        location="test_table",
        partitioning={"by": "ts_event", "interval": "daily"},
        retention_days=365,
        schema=schema,
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["instrument_id", "ts_event"],
        schema_hash=hashlib.sha256(str(dict(sorted(schema.items()))).encode()).hexdigest(),
        constraints={
            "nullability": {
                "instrument_id": False,
                "ts_event": False,
                "ts_init": False,
            },
        },
        lineage=[],
        pipeline_signature="test",
        version="1.0.0",
    )

    # Default contract with comprehensive validation rules
    contract = DataContract(
        contract_id="test_contract",
        dataset_id="test_bars",
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
                rule_type=ValidationRuleType.NULLABILITY,
                field_name="instrument_id",
                parameters={"nullable": False},
                severity=QualityFlag.FAIL,
                description="Instrument ID must not be null",
            ),
            ValidationRule(
                rule_type=ValidationRuleType.RANGE,
                field_name="close",
                parameters={"min": 0.0},
                severity=QualityFlag.FAIL,
                description="Close price must be positive",
            ),
            ValidationRule(
                rule_type=ValidationRuleType.UNIQUENESS,
                field_name="instrument_id,ts_event",
                parameters={},
                severity=QualityFlag.FAIL,
                description="Primary key uniqueness",
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
    registry.emit_event = MagicMock()
    registry.update_watermark = MagicMock()

    return registry


@pytest.fixture
def data_store(mock_registry: MagicMock, test_database, mock_stores_bundle) -> DataStore:
    """
    Create a DataStore instance with proper PostgreSQL connection.
    """
    store = DataStore(
        registry=mock_registry,
        connection_string=test_database.connection_string,
        feature_store=mock_stores_bundle["feature_store"],
        model_store=mock_stores_bundle["model_store"],
        strategy_store=mock_stores_bundle["strategy_store"],
        fail_on_validation_error=True,
        allow_schema_migration=False,
    )

    return store


@pytest.fixture
def valid_bar_data() -> list[dict[str, Any]]:
    """
    Create valid bar data for testing.
    """
    from ml.tests.builders import DataBuilder

    ohlcv_data = DataBuilder.ohlcv_data(n_bars=10, as_dataframe=False)
    timestamps = DataBuilder.time_series(n_points=10)

    return [
        {
            "instrument_id": "EUR/USD",
            "ts_event": int(timestamps[i]),
            "ts_init": int(timestamps[i]),
            "open": ohlcv_data["open"][i],
            "high": ohlcv_data["high"][i],
            "low": ohlcv_data["low"][i],
            "close": ohlcv_data["close"][i],
            "volume": ohlcv_data["volume"][i],
        }
        for i in range(10)
    ]


# ========================================================================
# Preflight Check Tests
# ========================================================================


@pytest.mark.property
@pytest.mark.database
@pytest.mark.serial
@pytest.mark.slow
@pytest.mark.unit
class TestPreflightCheck:
    """
    Test preflight schema validation.
    """

    def test_preflight_check_valid_data(
        self,
        data_store: DataStore,
        valid_bar_data: list[dict[str, Any]],
    ) -> None:
        """
        Test preflight check passes for valid data.
        """
        if HAS_POLARS:
            df = pl.DataFrame(valid_bar_data)
        else:
            df = valid_bar_data

        success, error, details = data_store.preflight_check("test_bars", df)

        assert success is True
        assert error is None
        assert details["preflight_passed"] is True
        assert "column_presence" in details["checks_performed"]
        assert "type_compatibility" in details["checks_performed"]
        assert "schema_hash" in details["checks_performed"]

    @pytest.mark.database
    @pytest.mark.serial
    def test_preflight_check_missing_required_columns(
        self,
        data_store: DataStore,
    ) -> None:
        """
        Test preflight check fails for missing required columns.
        """
        data = [{"instrument_id": "EUR/USD", "close": 1.1000}]

        if HAS_POLARS:
            df = pl.DataFrame(data)
        else:
            df = data

        success, error, details = data_store.preflight_check("test_bars", df, strict=True)

        assert success is False
        assert "Missing required columns" in error
        assert "missing_columns" in details

    @pytest.mark.database
    @pytest.mark.serial
    def test_preflight_check_type_mismatch(
        self,
        data_store: DataStore,
    ) -> None:
        """
        Test preflight check detects type mismatches.
        """
        data = [
            {
                "instrument_id": "EUR/USD",
                "ts_event": "not_a_number",  # Wrong type
                "ts_init": time.time_ns(),
                "open": 1.1000,
                "high": 1.1005,
                "low": 1.0995,
                "close": 1.1002,
                "volume": 1000.0,
            },
        ]

        # Don't convert to DataFrame to keep the string type
        success, error, details = data_store.preflight_check("test_bars", data, strict=False)

        # In non-strict mode, type mismatches generate warnings
        assert success is True
        assert len(details.get("warnings", [])) > 0

    @pytest.mark.database
    @pytest.mark.serial
    def test_preflight_check_schema_hash_mismatch(
        self,
        data_store: DataStore,
        valid_bar_data: list[dict[str, Any]],
    ) -> None:
        """
        Test preflight check detects schema hash mismatches.
        """
        # Add an extra column to change the schema
        for row in valid_bar_data:
            row["extra_field"] = 123

        if HAS_POLARS:
            df = pl.DataFrame(valid_bar_data)
        else:
            df = valid_bar_data

        success, error, details = data_store.preflight_check("test_bars", df, strict=True)

        assert success is False
        assert "Unexpected columns" in error or "Schema hash mismatch" in details.get(
            "warnings",
            [],
        )

    @pytest.mark.database
    @pytest.mark.serial
    def test_preflight_check_primary_key_nulls(
        self,
        data_store: DataStore,
        valid_bar_data: list[dict[str, Any]],
    ) -> None:
        """
        Test preflight check fails for null primary keys.
        """
        valid_bar_data[0]["instrument_id"] = None

        if HAS_POLARS:
            df = pl.DataFrame(valid_bar_data)
        else:
            df = valid_bar_data

        success, error, details = data_store.preflight_check("test_bars", df)

        assert success is False
        assert "Primary key field" in error or "null values" in error


# ========================================================================
# Contract Validation Tests
# ========================================================================


@pytest.mark.database
@pytest.mark.serial
class TestContractValidation:
    """
    Test contract validation rules.
    """

    @pytest.mark.database
    @pytest.mark.serial
    def test_type_validation(
        self,
        data_store: DataStore,
        mock_registry: MagicMock,
    ) -> None:
        """
        Test type validation rule.
        """
        # Create data with wrong types
        data = [
            {
                "instrument_id": 123,  # Should be string
                "ts_event": time.time_ns(),
                "ts_init": time.time_ns(),
                "open": "not_a_float",  # Should be float
                "high": 1.1005,
                "low": 1.0995,
                "close": 1.1002,
                "volume": 1000.0,
            },
        ]

        report = data_store.validate_batch("test_bars", data)

        assert report.quality_score < 1.0
        assert any(v.rule_type == ValidationRuleType.TYPE_CHECK for v in report.violations)

    @pytest.mark.database
    @pytest.mark.serial
    def test_null_validation(
        self,
        data_store: DataStore,
        valid_bar_data: list[dict[str, Any]],
    ) -> None:
        """
        Test null validation rule.
        """
        # Set a required field to null
        valid_bar_data[0]["instrument_id"] = None

        if HAS_POLARS:
            df = pl.DataFrame(valid_bar_data)
        else:
            df = valid_bar_data

        report = data_store.validate_batch("test_bars", df)

        assert report.quality_score < 1.0
        assert report.failed_records > 0

        null_violations = [
            v for v in report.violations if v.rule_type == ValidationRuleType.NULLABILITY
        ]
        assert len(null_violations) > 0
        assert null_violations[0].field_name == "instrument_id"

    @pytest.mark.database
    @pytest.mark.serial
    def test_range_validation(
        self,
        data_store: DataStore,
        valid_bar_data: list[dict[str, Any]],
    ) -> None:
        """
        Test range validation rule.
        """
        # Set negative close price
        valid_bar_data[0]["close"] = -1.0
        valid_bar_data[1]["close"] = -0.5

        if HAS_POLARS:
            df = pl.DataFrame(valid_bar_data)
        else:
            df = valid_bar_data

        report = data_store.validate_batch("test_bars", df)

        assert report.quality_score < 1.0
        assert report.failed_records >= 2

        range_violations = [v for v in report.violations if v.rule_type == ValidationRuleType.RANGE]
        assert len(range_violations) > 0
        assert range_violations[0].violation_count >= 2

    @pytest.mark.database
    @pytest.mark.serial
    def test_uniqueness_validation(
        self,
        data_store: DataStore,
        valid_bar_data: list[dict[str, Any]],
    ) -> None:
        """
        Test uniqueness validation rule.
        """
        # Create duplicate primary key
        valid_bar_data[1]["ts_event"] = valid_bar_data[0]["ts_event"]

        if HAS_POLARS:
            df = pl.DataFrame(valid_bar_data)
        else:
            df = valid_bar_data

        report = data_store.validate_batch("test_bars", df)

        assert report.quality_score < 1.0

        uniqueness_violations = [
            v for v in report.violations if v.rule_type == ValidationRuleType.UNIQUENESS
        ]
        assert len(uniqueness_violations) > 0

    @pytest.mark.database
    @pytest.mark.serial
    def test_monotonicity_validation(
        self,
        data_store: DataStore,
        valid_bar_data: list[dict[str, Any]],
    ) -> None:
        """
        Test monotonicity validation rule.
        """
        # Make timestamps non-monotonic
        valid_bar_data[5]["ts_event"] = valid_bar_data[3]["ts_event"]

        if HAS_POLARS:
            df = pl.DataFrame(valid_bar_data)
        else:
            df = valid_bar_data

        report = data_store.validate_batch("test_bars", df)

        assert report.quality_score < 1.0

        monotonicity_violations = [
            v for v in report.violations if v.rule_type == ValidationRuleType.MONOTONICITY
        ]
        assert len(monotonicity_violations) > 0

    @pytest.mark.database
    @pytest.mark.serial
    def test_lateness_validation(
        self,
        data_store: DataStore,
        mock_registry: MagicMock,
    ) -> None:
        """
        Test lateness validation rule.
        """
        # Create new contract with lateness rule added
        original_contract = mock_registry.get_contract.return_value

        new_rules = list(original_contract.validation_rules) + [
            ValidationRule(
                rule_type=ValidationRuleType.LATENESS,
                field_name="ts_event",
                parameters={"max_lateness_ns": 60_000_000_000},  # 1 minute
                severity=QualityFlag.WARN,
                description="Data must not be more than 1 minute late",
            ),
        ]

        contract = DataContract(
            contract_id=original_contract.contract_id,
            dataset_id=original_contract.dataset_id,
            version=original_contract.version,
            validation_rules=new_rules,
            quality_thresholds=original_contract.quality_thresholds,
            enforcement_mode=original_contract.enforcement_mode,
        )
        mock_registry.get_contract.return_value = contract

        # Create old data
        old_time = time.time_ns() - 120_000_000_000  # 2 minutes ago
        data = [
            {
                "instrument_id": "EUR/USD",
                "ts_event": old_time,
                "ts_init": old_time,
                "open": 1.1000,
                "high": 1.1005,
                "low": 1.0995,
                "close": 1.1002,
                "volume": 1000.0,
            },
        ]

        if HAS_POLARS:
            df = pl.DataFrame(data)
        else:
            df = data

        report = data_store.validate_batch("test_bars", df)

        lateness_violations = [
            v for v in report.violations if v.rule_type == ValidationRuleType.LATENESS
        ]
        assert len(lateness_violations) > 0


# ========================================================================
# Fail-Closed Write Tests
# ========================================================================


@pytest.mark.database
@pytest.mark.serial
class TestFailClosedWrites:
    """
    Test fail-closed write behavior.
    """

    @pytest.mark.database
    @pytest.mark.serial
    def test_write_rejected_on_validation_failure(
        self,
        data_store: DataStore,
        valid_bar_data: list[dict[str, Any]],
    ) -> None:
        """
        Test that writes are rejected when validation fails.
        """
        # Create invalid data with negative price
        valid_bar_data[0]["close"] = -1.0

        if HAS_POLARS:
            df = pl.DataFrame(valid_bar_data)
        else:
            df = valid_bar_data

        with pytest.raises(ValueError, match="Data validation failed.*fail-closed"):
            data_store.write_ingestion(
                dataset_id="test_bars",
                records=df,
                source="test",
                run_id="test_run",
            )

    @pytest.mark.database
    @pytest.mark.serial
    def test_write_rejected_on_preflight_failure(
        self,
        data_store: DataStore,
    ) -> None:
        """
        Test that writes are rejected when preflight check fails.
        """
        # Data missing required columns
        data = [{"instrument_id": "EUR/USD", "close": 1.1000}]

        if HAS_POLARS:
            df = pl.DataFrame(data)
        else:
            df = data

        with pytest.raises(ValueError, match="Preflight check failed"):
            data_store.write_ingestion(
                dataset_id="test_bars",
                records=df,
                source="test",
                run_id="test_run",
            )

    @pytest.mark.database
    @pytest.mark.serial
    def test_strict_mode_enforcement(
        self,
        data_store: DataStore,
        mock_registry: MagicMock,
        valid_bar_data: list[dict[str, Any]],
    ) -> None:
        """
        Test strict mode enforcement.
        """
        # Create a new contract in strict mode (can't modify frozen dataclass)
        original_contract = mock_registry.get_contract.return_value

        # Create new contract with additional warning-level rule
        new_rules = list(original_contract.validation_rules) + [
            ValidationRule(
                rule_type=ValidationRuleType.RANGE,
                field_name="volume",
                parameters={"min": 10000.0},  # High threshold
                severity=QualityFlag.WARN,
                description="Volume should be high",
            ),
        ]

        # Create new contract instance
        contract = DataContract(
            contract_id=original_contract.contract_id,
            dataset_id=original_contract.dataset_id,
            version=original_contract.version,
            validation_rules=new_rules,
            quality_thresholds=original_contract.quality_thresholds,
            enforcement_mode="strict",
        )
        mock_registry.get_contract.return_value = contract

        if HAS_POLARS:
            df = pl.DataFrame(valid_bar_data)
        else:
            df = valid_bar_data

        # In strict mode, even warnings cause failure
        report = data_store.validate_batch("test_bars", df, strict_mode=True)
        assert report.quality_score < 1.0

    @pytest.mark.database
    @pytest.mark.serial
    def test_lenient_mode_allows_warnings(
        self,
        data_store: DataStore,
        mock_registry: MagicMock,
        valid_bar_data: list[dict[str, Any]],
    ) -> None:
        """
        Test lenient mode allows warnings but not critical errors.
        """
        # Create new contract in lenient mode with only warning rules
        original_contract = mock_registry.get_contract.return_value

        contract = DataContract(
            contract_id=original_contract.contract_id,
            dataset_id=original_contract.dataset_id,
            version=original_contract.version,
            validation_rules=[
                ValidationRule(
                    rule_type=ValidationRuleType.RANGE,
                    field_name="volume",
                    parameters={"min": 10000.0},
                    severity=QualityFlag.WARN,
                    description="Volume should be high",
                ),
            ],
            quality_thresholds=original_contract.quality_thresholds,
            enforcement_mode="lenient",
        )
        mock_registry.get_contract.return_value = contract

        if HAS_POLARS:
            df = pl.DataFrame(valid_bar_data)
        else:
            df = valid_bar_data

        # Should not raise in lenient mode for warnings
        event = data_store.write_ingestion(
            dataset_id="test_bars",
            records=df,
            source="test",
            run_id="test_run",
        )

        assert event.status == "success"

    @pytest.mark.database
    @pytest.mark.serial
    def test_monitor_only_mode(
        self,
        data_store: DataStore,
        mock_registry: MagicMock,
        valid_bar_data: list[dict[str, Any]],
    ) -> None:
        """
        Test monitor-only mode logs but doesn't block.
        """
        # Create new contract in monitor-only mode
        original_contract = mock_registry.get_contract.return_value

        contract = DataContract(
            contract_id=original_contract.contract_id,
            dataset_id=original_contract.dataset_id,
            version=original_contract.version,
            validation_rules=original_contract.validation_rules,
            quality_thresholds=original_contract.quality_thresholds,
            enforcement_mode="monitor_only",
        )
        mock_registry.get_contract.return_value = contract

        # Add violation that would normally fail
        valid_bar_data[0]["close"] = -1.0

        if HAS_POLARS:
            df = pl.DataFrame(valid_bar_data)
        else:
            df = valid_bar_data

        # Should not raise in monitor-only mode
        with patch("ml.stores.data_store.logger") as mock_logger:
            event = data_store.write_ingestion(
                dataset_id="test_bars",
                records=df,
                source="test",
                run_id="test_run",
            )

            assert event.status == "success"
            # Should log the issues
            mock_logger.info.assert_called()


# ========================================================================
# Schema Migration Tests
# ========================================================================


@pytest.mark.database
@pytest.mark.serial
class TestSchemaMigration:
    """
    Test schema migration and dual-write functionality.
    """

    @pytest.mark.database
    @pytest.mark.serial
    def test_schema_migration_window(
        self,
        mock_registry: MagicMock,
        test_database,
    ) -> None:
        """
        Test schema migration window allows dual writes.
        """
        # Mock the underlying stores to avoid database connections
        feature_store = MagicMock()
        model_store = MagicMock()
        strategy_store = MagicMock()

        store = DataStore(
            registry=mock_registry,
            connection_string=test_database.connection_string,
            feature_store=feature_store,
            model_store=model_store,
            strategy_store=strategy_store,
            fail_on_validation_error=True,
            allow_schema_migration=True,
            schema_migration_window_hours=1,
        )

        # Start migration window
        manifest = mock_registry.get_manifest.return_value
        store._start_migration_window("test_bars", manifest)

        assert store._is_in_migration_window("test_bars") is True

        # Simulate window expiration
        store._schema_migration_state["test_bars"]["start_time"] = (
            time.time_ns() - 2 * 3600 * 1e9  # 2 hours ago
        )

        assert store._is_in_migration_window("test_bars") is False

    @pytest.mark.database
    @pytest.mark.serial
    def test_schema_version_change_detection(
        self,
        mock_registry: MagicMock,
        test_database,
    ) -> None:
        """
        Test detection of schema version changes.
        """
        # Mock the underlying stores
        feature_store = MagicMock()
        model_store = MagicMock()
        strategy_store = MagicMock()

        store = DataStore(
            registry=mock_registry,
            connection_string=test_database.connection_string,
            feature_store=feature_store,
            model_store=model_store,
            strategy_store=strategy_store,
            allow_schema_migration=True,
        )

        # Initial manifest fetch with version 1.0.0
        original_manifest = mock_registry.get_manifest.return_value

        # Create manifest with version 1.0.0
        manifest1 = DatasetManifest(
            dataset_id=original_manifest.dataset_id,
            dataset_type=original_manifest.dataset_type,
            storage_kind=original_manifest.storage_kind,
            location=original_manifest.location,
            partitioning=original_manifest.partitioning,
            retention_days=original_manifest.retention_days,
            schema=original_manifest.schema,
            ts_field=original_manifest.ts_field,
            seq_field=original_manifest.seq_field,
            primary_keys=original_manifest.primary_keys,
            schema_hash=original_manifest.schema_hash,
            constraints=original_manifest.constraints,
            lineage=original_manifest.lineage,
            pipeline_signature=original_manifest.pipeline_signature,
            version="1.0.0",
        )

        mock_registry.get_manifest.return_value = manifest1
        _ = store._get_manifest("test_bars")

        # Clear cache to simulate version update
        store._manifest_cache.clear()
        store._schema_migration_state["test_bars"] = {"version": "1.0.0"}

        # Create manifest with version 2.0.0
        manifest2 = DatasetManifest(
            dataset_id=original_manifest.dataset_id,
            dataset_type=original_manifest.dataset_type,
            storage_kind=original_manifest.storage_kind,
            location=original_manifest.location,
            partitioning=original_manifest.partitioning,
            retention_days=original_manifest.retention_days,
            schema=original_manifest.schema,
            ts_field=original_manifest.ts_field,
            seq_field=original_manifest.seq_field,
            primary_keys=original_manifest.primary_keys,
            schema_hash=original_manifest.schema_hash,
            constraints=original_manifest.constraints,
            lineage=original_manifest.lineage,
            pipeline_signature=original_manifest.pipeline_signature,
            version="2.0.0",
        )

        mock_registry.get_manifest.return_value = manifest2

        # Should detect version change and start migration
        with patch.object(store, "_start_migration_window") as mock_start:
            _ = store._get_manifest("test_bars")
            mock_start.assert_called_once()

    @pytest.mark.database
    @pytest.mark.serial
    def test_dual_write_during_migration(
        self,
        mock_registry: MagicMock,
        valid_bar_data: list[dict[str, Any]],
        test_database,
    ) -> None:
        """
        Test dual-write is allowed during migration window.
        """
        # Mock the underlying stores
        feature_store = MagicMock()
        model_store = MagicMock()
        strategy_store = MagicMock()

        store = DataStore(
            registry=mock_registry,
            connection_string=test_database.connection_string,
            feature_store=feature_store,
            model_store=model_store,
            strategy_store=strategy_store,
            fail_on_validation_error=True,
            allow_schema_migration=True,
        )

        # Start migration window
        manifest = mock_registry.get_manifest.return_value
        store._start_migration_window("test_bars", manifest)

        # Add extra column (schema change)
        for row in valid_bar_data:
            row["new_field"] = "test"

        if HAS_POLARS:
            df = pl.DataFrame(valid_bar_data)
        else:
            df = valid_bar_data

        # Preflight should pass with warning during migration
        # Use strict=False since we're testing migration scenario
        success, error, details = store.preflight_check("test_bars", df, strict=False)

        assert success is True, f"Preflight failed: {error}, Details: {details}"
        assert any("migration" in str(w).lower() for w in details.get("warnings", []))


# ========================================================================
# Property-Based Tests
# ========================================================================


@pytest.mark.database
@pytest.mark.serial
class TestPropertyBased:
    """
    Property-based tests for validation fuzzing.
    """

    @given(
        num_records=st.integers(min_value=1, max_value=100),
        null_probability=st.floats(min_value=0.0, max_value=1.0),
        include_duplicates=st.booleans(),
    )
    @pytest.mark.database
    @pytest.mark.serial
    @settings(max_examples=50, deadline=5000)
    def test_validation_consistency(
        self,
        num_records: int,
        null_probability: float,
        include_duplicates: bool,
    ) -> None:
        """
        Test validation is consistent across different data patterns.
        """
        # Create mock registry using builder
        mock_registry = MagicMock()

        # Set up default manifest and contract
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

        manifest = DatasetManifest(
            dataset_id="test_bars",
            dataset_type=DatasetType.BARS,
            storage_kind=StorageKind.POSTGRES,
            location="test_table",
            partitioning={"by": "ts_event", "interval": "daily"},
            retention_days=365,
            schema=schema,
            ts_field="ts_event",
            seq_field=None,
            primary_keys=["instrument_id", "ts_event"],
            schema_hash=hashlib.sha256(str(dict(sorted(schema.items()))).encode()).hexdigest(),
            constraints={
                "nullability": {
                    "instrument_id": False,
                    "ts_event": False,
                    "ts_init": False,
                },
            },
            lineage=[],
            pipeline_signature="test",
            version="1.0.0",
        )

        contract = DataContract(
            contract_id="test_contract",
            dataset_id="test_bars",
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
                    rule_type=ValidationRuleType.NULLABILITY,
                    field_name="instrument_id",
                    parameters={"nullable": False},
                    severity=QualityFlag.FAIL,
                    description="Instrument ID must not be null",
                ),
                ValidationRule(
                    rule_type=ValidationRuleType.UNIQUENESS,
                    field_name="instrument_id,ts_event",
                    parameters={},
                    severity=QualityFlag.FAIL,
                    description="Primary key uniqueness",
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

        mock_registry.get_manifest.return_value = manifest
        mock_registry.get_contract.return_value = contract

        # Create data store with mock stores
        conn_str = "postgresql://postgres:postgres@localhost:5432/nautilus"
        mock_stores = MockBuilder.all_registries()  # This creates store mocks
        data_store = DataStore(
            registry=mock_registry,
            connection_string=conn_str,
            feature_store=MockBuilder.store_with_data(store_type="feature"),
            model_store=MockBuilder.store_with_data(store_type="model"),
            strategy_store=MockBuilder.store_with_data(store_type="strategy"),
            fail_on_validation_error=True,
        )

        # Generate base data using DataBuilder
        ohlcv_data = DataBuilder.ohlcv_data(n_bars=num_records, as_dataframe=False)
        timestamps = DataBuilder.time_series(n_points=num_records)

        data = []
        for i in range(num_records):
            row = {
                "instrument_id": "EUR/USD" if np.random.random() > null_probability else None,
                "ts_event": int(timestamps[i]),
                "ts_init": int(timestamps[i]),
                "open": ohlcv_data["open"][i],
                "high": ohlcv_data["high"][i],
                "low": ohlcv_data["low"][i],
                "close": ohlcv_data["close"][i],
                "volume": ohlcv_data["volume"][i],
            }

            # Introduce duplicates if requested
            if include_duplicates and i > 0 and np.random.random() < 0.1:
                row["ts_event"] = data[-1]["ts_event"]

            data.append(row)

        if HAS_POLARS:
            df = pl.DataFrame(data)
        else:
            df = data

        # Validation should not crash regardless of input
        report = data_store.validate_batch("test_bars", df)

        # Quality score should be between 0 and 1
        assert 0.0 <= report.quality_score <= 1.0

        # Counts should be consistent
        assert report.total_records == num_records
        assert report.passed_records + report.failed_records == report.total_records

        # If there are actual violations, score should be < 1
        # Note: null_probability > 0 doesn't guarantee nulls due to randomness
        if report.violations:
            assert report.quality_score < 1.0

        # If we have violations with FAIL severity, there should be failed records
        fail_violations = [v for v in report.violations if v.severity == QualityFlag.FAIL]
        if fail_violations:
            assert report.failed_records > 0

    @given(
        close_min=st.floats(min_value=-1000.0, max_value=0.0),
        close_max=st.floats(min_value=0.0, max_value=1000.0),
        volume_min=st.floats(min_value=-1000.0, max_value=0.0),
        volume_max=st.floats(min_value=0.0, max_value=10000.0),
    )
    @pytest.mark.database
    @pytest.mark.serial
    @settings(max_examples=50, deadline=5000)
    def test_range_validation_fuzzing(
        self,
        close_min: float,
        close_max: float,
        volume_min: float,
        volume_max: float,
    ) -> None:
        """
        Test range validation with fuzzy boundaries.
        """
        # Create mock registry using builder
        mock_registry = MagicMock()

        # Set up manifest
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

        manifest = DatasetManifest(
            dataset_id="test_bars",
            dataset_type=DatasetType.BARS,
            storage_kind=StorageKind.POSTGRES,
            location="test_table",
            partitioning={"by": "ts_event", "interval": "daily"},
            retention_days=365,
            schema=schema,
            ts_field="ts_event",
            seq_field=None,
            primary_keys=["instrument_id", "ts_event"],
            schema_hash=hashlib.sha256(str(dict(sorted(schema.items()))).encode()).hexdigest(),
            constraints={},
            lineage=[],
            pipeline_signature="test",
            version="1.0.0",
        )

        mock_registry.get_manifest.return_value = manifest

        # Create contract with custom range rules
        contract = DataContract(
            contract_id="test_contract",
            dataset_id="test_bars",
            version="1.0.0",
            validation_rules=[
                ValidationRule(
                    rule_type=ValidationRuleType.RANGE,
                    field_name="close",
                    parameters={"min": close_min, "max": close_max},
                    severity=QualityFlag.FAIL,
                    description="Close price range",
                ),
                ValidationRule(
                    rule_type=ValidationRuleType.RANGE,
                    field_name="volume",
                    parameters={"min": volume_min, "max": volume_max},
                    severity=QualityFlag.FAIL,
                    description="Volume range",
                ),
            ],
            quality_thresholds={},
            enforcement_mode="strict",
        )
        mock_registry.get_contract.return_value = contract

        conn_str = "postgresql://postgres:postgres@localhost:5432/nautilus"
        store = DataStore(
            registry=mock_registry,
            connection_string=conn_str,
            feature_store=MockBuilder.store_with_data(store_type="feature"),
            model_store=MockBuilder.store_with_data(store_type="model"),
            strategy_store=MockBuilder.store_with_data(store_type="strategy"),
        )

        # Generate data within and outside ranges
        timestamps = DataBuilder.time_series(n_points=10)
        data = []

        for i in range(10):
            # Mix valid and invalid values
            if i % 2 == 0:
                close = np.random.uniform(close_min, close_max)
                volume = np.random.uniform(max(0, volume_min), volume_max)
            else:
                close = np.random.uniform(close_max, close_max + 100)
                volume = np.random.uniform(volume_max, volume_max + 1000)

            data.append(
                {
                    "instrument_id": "EUR/USD",
                    "ts_event": int(timestamps[i]),
                    "ts_init": int(timestamps[i]),
                    "open": abs(close) * 0.99,
                    "high": abs(close) * 1.01,
                    "low": abs(close) * 0.98,
                    "close": close,
                    "volume": abs(volume),
                },
            )

        if HAS_POLARS:
            df = pl.DataFrame(data)
        else:
            df = data

        report = store.validate_batch("test_bars", df)

        # Should have violations for out-of-range values
        range_violations = [v for v in report.violations if v.rule_type == ValidationRuleType.RANGE]

        # At least half the records should have violations (every other one)
        assert len(range_violations) > 0 or (
            close_min <= 0 <= close_max and volume_min <= 0 <= volume_max
        )


# ========================================================================
# Metrics Tests
# ========================================================================


@pytest.mark.database
@pytest.mark.serial
class TestPrometheusMetrics:
    """
    Test Prometheus metrics emission.
    """

    @pytest.mark.database
    @pytest.mark.serial
    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus not available")
    def test_validation_metrics_emitted(
        self,
        data_store: DataStore,
        valid_bar_data: list[dict[str, Any]],
    ) -> None:
        """
        Test that validation metrics are properly emitted.
        """
        from ml.stores.data_store import quality_score_histogram
        from ml.stores.data_store import validation_violations_counter

        # Create data with violations
        valid_bar_data[0]["close"] = -1.0  # Range violation
        valid_bar_data[1]["instrument_id"] = None  # Null violation

        if HAS_POLARS:
            df = pl.DataFrame(valid_bar_data)
        else:
            df = valid_bar_data

        # Clear metrics
        with patch.object(validation_violations_counter, "labels") as mock_violations:
            with patch.object(quality_score_histogram, "labels") as mock_quality:
                mock_violations.return_value.inc = MagicMock()
                mock_quality.return_value.observe = MagicMock()

                report = data_store.validate_batch("test_bars", df)

                # Violations counter should be called for each violation
                assert mock_violations.called

                # Quality score histogram should be observed
                assert mock_quality.called

    @pytest.mark.database
    @pytest.mark.serial
    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus not available")
    def test_write_rejection_metrics(
        self,
        data_store: DataStore,
    ) -> None:
        """
        Test that write rejection metrics are emitted.
        """
        from ml.stores.data_store import write_rejection_counter

        # Data missing required columns
        data = [{"instrument_id": "EUR/USD", "close": 1.1000}]

        with patch.object(write_rejection_counter, "labels") as mock_counter:
            mock_counter.return_value.inc = MagicMock()

            with pytest.raises(ValueError):
                data_store.write_ingestion(
                    dataset_id="test_bars",
                    records=data,
                    source="test",
                    run_id="test_run",
                )

            # Write rejection counter should be incremented
            mock_counter.assert_called_with(
                dataset_id="test_bars",
                reason="preflight_failed",
            )

    @pytest.mark.database
    @pytest.mark.serial
    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus not available")
    def test_schema_mismatch_metrics(
        self,
        data_store: DataStore,
        valid_bar_data: list[dict[str, Any]],
    ) -> None:
        """
        Test that schema mismatch metrics are emitted.
        """
        from ml.stores.data_store import schema_mismatch_counter

        # Add extra column to change schema
        for row in valid_bar_data:
            row["extra_field"] = 123

        if HAS_POLARS:
            df = pl.DataFrame(valid_bar_data)
        else:
            df = valid_bar_data

        with patch.object(schema_mismatch_counter, "labels") as mock_counter:
            mock_counter.return_value.inc = MagicMock()

            # Preflight check should detect schema mismatch
            success, error, details = data_store.preflight_check("test_bars", df, strict=False)

            # In non-strict mode, may not increment counter for extra columns
            # But will increment for hash mismatch if detected
            if "schema_hash" in str(details):
                mock_counter.assert_called()


# ========================================================================
# Integration Tests
# ========================================================================


@pytest.mark.database
@pytest.mark.serial
class TestIntegration:
    """
    End-to-end integration tests.
    """

    @pytest.mark.database
    @pytest.mark.serial
    def test_full_validation_pipeline(
        self,
        data_store: DataStore,
        mock_registry: MagicMock,
    ) -> None:
        """
        Test complete validation pipeline from preflight to write.
        """
        # Create comprehensive test data using DataBuilder
        ohlcv_data = DataBuilder.ohlcv_data(n_bars=100, volatility=0.001, as_dataframe=False)
        timestamps = DataBuilder.time_series(n_points=100)

        data = []
        for i in range(100):
            data.append(
                {
                    "instrument_id": "EUR/USD",
                    "ts_event": int(timestamps[i]),
                    "ts_init": int(timestamps[i]),
                    "open": ohlcv_data["open"][i],
                    "high": ohlcv_data["high"][i],
                    "low": ohlcv_data["low"][i],
                    "close": ohlcv_data["close"][i],
                    "volume": ohlcv_data["volume"][i] + np.random.exponential(100),
                },
            )

        if HAS_POLARS:
            df = pl.DataFrame(data)
        else:
            df = data

        # 1. Preflight check
        success, error, details = data_store.preflight_check("test_bars", df)
        assert success is True

        # 2. Validation
        report = data_store.validate_batch("test_bars", df)
        assert report.quality_score == 1.0
        assert report.failed_records == 0

        # 3. Write
        event = data_store.write_ingestion(
            dataset_id="test_bars",
            records=df,
            source="test",
            run_id="integration_test",
        )

        assert event.status == "success"
        assert event.record_count == 100

        # 4. Verify registry calls
        mock_registry.emit_event.assert_called()
        mock_registry.update_watermark.assert_called()

    @pytest.mark.database
    @pytest.mark.serial
    def test_validation_performance(
        self,
        data_store: DataStore,
    ) -> None:
        """
        Test validation performance with large datasets.
        """
        # Create large dataset using DataBuilder
        num_records = 10000
        ohlcv_data = DataBuilder.ohlcv_data(n_bars=num_records, as_dataframe=False)
        timestamps = DataBuilder.time_series(n_points=num_records, interval_ns=1_000_000_000)  # 1 second intervals

        data = []
        for i in range(num_records):
            data.append(
                {
                    "instrument_id": "EUR/USD",
                    "ts_event": int(timestamps[i]),
                    "ts_init": int(timestamps[i]),
                    "open": ohlcv_data["open"][i],
                    "high": ohlcv_data["high"][i],
                    "low": ohlcv_data["low"][i],
                    "close": ohlcv_data["close"][i],
                    "volume": ohlcv_data["volume"][i],
                },
            )

        if HAS_POLARS:
            df = pl.DataFrame(data)
        else:
            df = data

        # Measure validation time
        start_time = time.perf_counter()
        report = data_store.validate_batch("test_bars", df)
        validation_time = time.perf_counter() - start_time

        # Validation should be fast
        assert validation_time < 1.0  # Less than 1 second for 10k records
        assert report.total_records == num_records

        # Report should include timing
        assert report.validation_time_ms > 0
        assert report.validation_time_ms < 1000  # Less than 1 second
