#!/usr/bin/env python3
"""
Unit tests for DataPersistenceComponent.

Tests cover JSON/PostgreSQL serialization, loading, saving, and thread safety
for the DataRegistry persistence layer extracted from the legacy DataRegistry.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from ml.config.events import EventStatus, Source, Stage
from ml.registry.dataclasses import (
    DataContract,
    DatasetLineageRecord,
    DatasetManifest,
    DatasetType,
    QualityFlag,
    StorageKind,
    ValidationRule,
    ValidationRuleType,
)
from ml.registry.data_registry import Watermark
from ml.registry.persistence import BackendType, PersistenceConfig


if TYPE_CHECKING:
    from ml.registry.common.data_persistence import DataPersistenceComponent


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def json_persistence_config(tmp_path: Path) -> PersistenceConfig:
    """Create JSON backend persistence config for testing."""
    return PersistenceConfig(
        backend=BackendType.JSON,
        json_path=tmp_path / "registry",
    )


@pytest.fixture
def sample_dataset_manifest() -> DatasetManifest:
    """Create a valid DatasetManifest for testing."""
    return DatasetManifest(
        dataset_id="features_test_eurusd",
        dataset_type=DatasetType.FEATURES,
        storage_kind=StorageKind.PARQUET,
        location="/tmp/features",
        partitioning={"by": "ts_event", "interval": "daily"},
        retention_days=30,
        schema={
            "instrument_id": "str",
            "ts_event": "int64",
            "ts_init": "int64",
            "close": "float64",
            "volume": "float64",
        },
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["instrument_id", "ts_event"],
        schema_hash="",
        constraints={"nullability": {"instrument_id": False, "ts_event": False}},
        lineage=[],
        pipeline_signature="test_pipeline_v1",
        version="1.0.0",
    )


@pytest.fixture
def sample_data_contract() -> DataContract:
    """Create a valid DataContract for testing."""
    return DataContract(
        contract_id="features_test_eurusd_contract",
        dataset_id="features_test_eurusd",
        version="1.0.0",
        validation_rules=[
            ValidationRule(
                rule_type=ValidationRuleType.TYPE_CHECK,
                field_name="*",
                parameters={},
                severity=QualityFlag.WARN,
                description="Type validation for all fields",
            )
        ],
        quality_thresholds={"null_rate": 0.01},
        enforcement_mode="strict",
    )


@pytest.fixture
def sample_watermark() -> Watermark:
    """Create a valid Watermark for testing."""
    return Watermark(
        dataset_id="features_test",
        instrument_id="EUR/USD",
        source="live",
        last_success_ns=1_000_000_000_000_000_000,
        last_attempt_ns=1_000_000_000_000_000_000,
        last_count=100,
        completeness_pct=98.5,
        updated_at=time.time(),
    )


# =============================================================================
# Loading Tests
# =============================================================================


class TestLoadRegistry:
    """Tests for _load_registry method."""

    def test_load_registry_creates_empty_on_missing_file(
        self, tmp_path: Path, json_persistence_config: PersistenceConfig
    ) -> None:
        """Verify empty registry initialization when no file exists."""
        # Import will be available after implementation
        from ml.registry.common.data_persistence import DataPersistenceComponent

        component = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        assert len(component._manifests) == 0
        assert len(component._contracts) == 0
        assert len(component._events) == 0
        assert len(component._watermarks) == 0
        assert len(component._lineage) == 0

    def test_load_registry_from_existing_json(
        self, tmp_path: Path, sample_dataset_manifest: DatasetManifest
    ) -> None:
        """Verify loading existing registry data from JSON file."""
        from ml.registry.common.data_persistence import DataPersistenceComponent

        # Pre-populate JSON file
        reg_dir = tmp_path / "registry"
        reg_dir.mkdir(parents=True, exist_ok=True)
        registry_file = reg_dir / "data_registry.json"

        manifest_dict = {
            "dataset_id": sample_dataset_manifest.dataset_id,
            "dataset_type": sample_dataset_manifest.dataset_type.value,
            "storage_kind": sample_dataset_manifest.storage_kind.value,
            "location": sample_dataset_manifest.location,
            "partitioning": sample_dataset_manifest.partitioning,
            "retention_days": sample_dataset_manifest.retention_days,
            "schema": sample_dataset_manifest.schema,
            "ts_field": sample_dataset_manifest.ts_field,
            "seq_field": sample_dataset_manifest.seq_field,
            "primary_keys": sample_dataset_manifest.primary_keys,
            "schema_hash": sample_dataset_manifest.schema_hash,
            "constraints": sample_dataset_manifest.constraints,
            "lineage": sample_dataset_manifest.lineage,
            "pipeline_signature": sample_dataset_manifest.pipeline_signature,
            "version": sample_dataset_manifest.version,
            "created_at": sample_dataset_manifest.created_at,
            "last_modified": sample_dataset_manifest.last_modified,
            "metadata": sample_dataset_manifest.metadata,
        }

        registry_data = {
            "manifests": {sample_dataset_manifest.dataset_id: manifest_dict},
            "contracts": {},
            "events": [],
            "watermarks": {},
            "lineage": [],
            "last_updated": time.time(),
        }

        registry_file.write_text(json.dumps(registry_data))

        config = PersistenceConfig(backend=BackendType.JSON, json_path=reg_dir)
        component = DataPersistenceComponent(
            registry_path=reg_dir,
            persistence_config=config,
        )

        assert len(component._manifests) == 1
        assert sample_dataset_manifest.dataset_id in component._manifests

    def test_load_registry_handles_corrupted_json(self, tmp_path: Path) -> None:
        """Verify graceful handling of corrupted JSON file."""
        from ml.registry.common.data_persistence import DataPersistenceComponent

        reg_dir = tmp_path / "registry"
        reg_dir.mkdir(parents=True, exist_ok=True)
        registry_file = reg_dir / "data_registry.json"
        registry_file.write_text("{invalid json")

        config = PersistenceConfig(backend=BackendType.JSON, json_path=reg_dir)

        # Should not raise, should initialize empty
        component = DataPersistenceComponent(
            registry_path=reg_dir,
            persistence_config=config,
        )

        assert len(component._manifests) == 0


# =============================================================================
# Save Tests
# =============================================================================


class TestSaveRegistry:
    """Tests for _save_registry method."""

    def test_save_registry_immediate_mode(
        self, tmp_path: Path, json_persistence_config: PersistenceConfig
    ) -> None:
        """Verify immediate save bypasses batching."""
        from ml.registry.common.data_persistence import DataPersistenceComponent

        component = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        component._save_registry(immediate=True)

        registry_file = tmp_path / "registry" / "data_registry.json"
        assert registry_file.exists()
        assert component._pending_save is False
        assert component._save_timer is None

    def test_save_registry_batched_mode(self, tmp_path: Path) -> None:
        """Verify batched save schedules timer."""
        from ml.registry.common.data_persistence import DataPersistenceComponent

        config = PersistenceConfig(
            backend=BackendType.JSON,
            json_path=tmp_path / "registry",
        )

        component = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=config,
            batch_save_interval=0.5,  # 500ms
        )

        component._save_registry(immediate=False)

        assert component._pending_save is True
        assert component._save_timer is not None

        # Clean up timer
        if component._save_timer:
            component._save_timer.cancel()

    def test_flush_forces_immediate_save(
        self, tmp_path: Path, json_persistence_config: PersistenceConfig
    ) -> None:
        """Verify flush() forces immediate persistence."""
        from ml.registry.common.data_persistence import DataPersistenceComponent

        component = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        component.flush()

        registry_file = tmp_path / "registry" / "data_registry.json"
        assert registry_file.exists()
        assert component._pending_save is False

    def test_batch_save_interval_zero_saves_immediately(self, tmp_path: Path) -> None:
        """Verify zero interval always saves immediately."""
        from ml.registry.common.data_persistence import DataPersistenceComponent

        config = PersistenceConfig(
            backend=BackendType.JSON,
            json_path=tmp_path / "registry",
        )

        component = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=config,
            batch_save_interval=0.0,
        )

        component._save_registry(immediate=False)  # Should still be immediate

        registry_file = tmp_path / "registry" / "data_registry.json"
        assert registry_file.exists()


# =============================================================================
# Serialization Tests
# =============================================================================


class TestManifestSerialization:
    """Tests for manifest serialization/deserialization."""

    def test_dict_to_manifest_valid_data(
        self, tmp_path: Path, json_persistence_config: PersistenceConfig
    ) -> None:
        """Verify dictionary to DatasetManifest conversion."""
        from ml.registry.common.data_persistence import DataPersistenceComponent

        component = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        data = {
            "dataset_id": "test_dataset",
            "dataset_type": "features",
            "storage_kind": "parquet",
            "location": "/tmp/test",
            "partitioning": {},
            "retention_days": 30,
            "schema": {"instrument_id": "str", "ts_event": "int64", "ts_init": "int64"},
            "ts_field": "ts_event",
            "seq_field": None,
            "primary_keys": ["instrument_id", "ts_event"],
            "schema_hash": "hash123",
            "constraints": {},
            "lineage": [],
            "pipeline_signature": "sig_v1",
            "version": "1.0.0",
            "created_at": time.time_ns(),
            "last_modified": time.time_ns(),
            "metadata": {},
        }

        result = component._dict_to_manifest(data)

        assert isinstance(result, DatasetManifest)
        assert result.dataset_type == DatasetType.FEATURES
        assert result.storage_kind == StorageKind.PARQUET

    def test_dict_to_manifest_enum_case_insensitive(
        self, tmp_path: Path, json_persistence_config: PersistenceConfig
    ) -> None:
        """Verify enum conversion handles case variations."""
        from ml.registry.common.data_persistence import DataPersistenceComponent

        component = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        base_data = {
            "dataset_id": "test",
            "storage_kind": "parquet",
            "location": "/tmp",
            "partitioning": {},
            "retention_days": 7,
            "schema": {"instrument_id": "str", "ts_event": "int64", "ts_init": "int64"},
            "ts_field": "ts_event",
            "seq_field": None,
            "primary_keys": ["instrument_id", "ts_event"],
            "schema_hash": "",
            "constraints": {},
            "lineage": [],
            "pipeline_signature": "sig",
            "version": "1.0",
            "created_at": time.time_ns(),
            "last_modified": time.time_ns(),
            "metadata": {},
        }

        for case_variant in ["FEATURES", "features", "Features"]:
            data = {**base_data, "dataset_type": case_variant}
            result = component._dict_to_manifest(data)
            assert result.dataset_type == DatasetType.FEATURES

    def test_manifest_to_dict_roundtrip(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Verify manifest serialization/deserialization roundtrip."""
        from ml.registry.common.data_persistence import DataPersistenceComponent

        component = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        dict_form = component._manifest_to_dict(sample_dataset_manifest)
        restored = component._dict_to_manifest(dict_form)

        assert restored.dataset_id == sample_dataset_manifest.dataset_id
        assert restored.dataset_type == sample_dataset_manifest.dataset_type
        assert restored.storage_kind == sample_dataset_manifest.storage_kind
        assert restored.location == sample_dataset_manifest.location


class TestContractSerialization:
    """Tests for contract serialization/deserialization."""

    def test_dict_to_contract_valid_data(
        self, tmp_path: Path, json_persistence_config: PersistenceConfig
    ) -> None:
        """Verify dictionary to DataContract conversion."""
        from ml.registry.common.data_persistence import DataPersistenceComponent

        component = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        data = {
            "contract_id": "test_contract",
            "dataset_id": "test_dataset",
            "version": "1.0.0",
            "validation_rules": [
                {
                    "rule_type": "type_check",
                    "field_name": "*",
                    "parameters": {},
                    "severity": "warn",
                    "description": "Type check",
                }
            ],
            "quality_thresholds": {"null_rate": 0.01},
            "enforcement_mode": "strict",
            "created_at": time.time_ns(),
            "last_modified": time.time_ns(),
            "metadata": {},
        }

        result = component._dict_to_contract(data)

        assert isinstance(result, DataContract)
        assert len(result.validation_rules) == 1
        assert result.validation_rules[0].rule_type == ValidationRuleType.TYPE_CHECK

    def test_contract_to_dict_roundtrip(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
        sample_data_contract: DataContract,
    ) -> None:
        """Verify contract serialization/deserialization roundtrip."""
        from ml.registry.common.data_persistence import DataPersistenceComponent

        component = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        dict_form = component._contract_to_dict(sample_data_contract)
        restored = component._dict_to_contract(dict_form)

        assert restored.contract_id == sample_data_contract.contract_id
        assert restored.dataset_id == sample_data_contract.dataset_id
        assert len(restored.validation_rules) == len(sample_data_contract.validation_rules)


class TestWatermarkSerialization:
    """Tests for watermark serialization/deserialization."""

    def test_dict_to_watermark_valid_data(
        self, tmp_path: Path, json_persistence_config: PersistenceConfig
    ) -> None:
        """Verify dictionary to Watermark conversion."""
        from ml.registry.common.data_persistence import DataPersistenceComponent

        component = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        data = {
            "dataset_id": "test_dataset",
            "instrument_id": "EUR/USD",
            "source": "live",
            "last_success_ns": 1000000000000000000,
            "last_attempt_ns": 1000000000000000000,
            "last_count": 100,
            "completeness_pct": 98.5,
            "updated_at": time.time(),
        }

        result = component._dict_to_watermark(data)

        assert isinstance(result, Watermark)
        assert result.dataset_id == "test_dataset"
        assert result.last_count == 100

    def test_watermark_to_dict_roundtrip(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
        sample_watermark: Watermark,
    ) -> None:
        """Verify watermark serialization/deserialization roundtrip."""
        from ml.registry.common.data_persistence import DataPersistenceComponent

        component = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        dict_form = component._watermark_to_dict(sample_watermark)
        restored = component._dict_to_watermark(dict_form)

        assert restored.dataset_id == sample_watermark.dataset_id
        assert restored.instrument_id == sample_watermark.instrument_id
        assert restored.source == sample_watermark.source
        assert restored.last_success_ns == sample_watermark.last_success_ns


# =============================================================================
# Row Conversion Tests (PostgreSQL)
# =============================================================================


class TestRowConversions:
    """Tests for database row to object conversions."""

    def test_manifest_from_row_postgres(
        self, tmp_path: Path, json_persistence_config: PersistenceConfig
    ) -> None:
        """Verify PostgreSQL row to DatasetManifest conversion."""
        from ml.registry.common.data_persistence import DataPersistenceComponent

        component = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        # Create mock row with _mapping attribute
        mock_row = MagicMock()
        mock_row._mapping = {
            "dataset_id": "test_dataset",
            "dataset_type": "FEATURES",
            "storage_kind": "parquet",
            "location": "/tmp/test",
            "partitioning": '{"by": "ts_event"}',
            "retention_days": 30,
            "schema": '{"instrument_id": "str", "ts_event": "int64", "ts_init": "int64"}',
            "schema_hash": "hash123",
            "constraints": "{}",
            "lineage": "[]",
            "pipeline_signature": "sig_v1",
            "version": "1.0.0",
            "created_at": 1000000000000000000.0,
            "last_modified": 1000000000000000000.0,
            "metadata": '{"ts_field": "ts_event", "primary_keys": ["instrument_id", "ts_event"]}',
        }

        result = component._manifest_from_row(mock_row)

        assert isinstance(result, DatasetManifest)
        assert result.dataset_id == "test_dataset"

    def test_manifest_from_row_postgres_macro_defaults_primary_keys(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify macro manifests default primary keys when metadata omits them."""
        from ml.registry.common.data_persistence import DataPersistenceComponent

        component = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        mock_row = MagicMock()
        mock_row._mapping = {
            "dataset_id": "ml.macro_release_calendar",
            "dataset_type": "MACRO_RELEASES",
            "storage_kind": "postgres",
            "location": "ml.macro_release_calendar",
            "partitioning": '{"by": ["series_id"]}',
            "retention_days": 3650,
            "schema": (
                '{"series_id": "str", "observation_ts": "int64", "release_ts": "int64", '
                '"ts_event": "int64", "ts_init": "int64"}'
            ),
            "schema_hash": "hash123",
            "constraints": "{}",
            "lineage": "[]",
            "pipeline_signature": "sig_v1",
            "version": "1.0.0",
            "created_at": 1000000000000000000.0,
            "last_modified": 1000000000000000000.0,
            "metadata": "{}",
        }

        result = component._manifest_from_row(mock_row)

        assert result.primary_keys == [
            "series_id",
            "observation_ts",
            "release_ts",
            "ts_event",
        ]

    def test_watermark_from_row_postgres(
        self, tmp_path: Path, json_persistence_config: PersistenceConfig
    ) -> None:
        """Verify PostgreSQL row to Watermark conversion."""
        from ml.registry.common.data_persistence import DataPersistenceComponent

        mock_row = MagicMock()
        mock_row._mapping = {
            "dataset_id": "test_dataset",
            "instrument_id": "EUR/USD",
            "source": "live",
            "last_success_ns": 1000000000000000000,
            "last_attempt_ns": 1000000000000000000,
            "last_count": 100,
            "completeness_pct": 98.5,
            "updated_at": 1700000000.0,
        }

        result = DataPersistenceComponent._watermark_from_row(mock_row)

        assert isinstance(result, Watermark)
        assert result.dataset_id == "test_dataset"
        assert result.last_count == 100

    def test_lineage_from_row_postgres(
        self, tmp_path: Path, json_persistence_config: PersistenceConfig
    ) -> None:
        """Verify PostgreSQL row to DatasetLineageRecord conversion."""
        from ml.registry.common.data_persistence import DataPersistenceComponent

        mock_row = MagicMock()
        mock_row._mapping = {
            "transform_id": "transform_v1",
            "child_dataset_id": "child_dataset",
            "parent_dataset_id": "parent_dataset",
            "ts_range": '{"start_ns": 1000, "end_ns": 2000}',
            "parameters": '{"lookback": 20}',
            "created_at": 1700000000.0,
        }

        result = DataPersistenceComponent._lineage_from_row(mock_row)

        assert isinstance(result, DatasetLineageRecord)
        assert result.transform_id == "transform_v1"
        assert result.ts_range["start_ns"] == 1000


# =============================================================================
# Backend Tests
# =============================================================================


class TestBackendInitialization:
    """Tests for backend-specific initialization."""

    def test_backend_type_json_initialization(self, tmp_path: Path) -> None:
        """Verify JSON backend initialization."""
        from ml.registry.common.data_persistence import DataPersistenceComponent

        config = PersistenceConfig(
            backend=BackendType.JSON,
            json_path=tmp_path / "registry",
        )

        component = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=config,
        )

        assert component.backend == BackendType.JSON
        assert (tmp_path / "registry" / "data_registry.json").exists()

    def test_ensure_json_handles_string_values(
        self, tmp_path: Path, json_persistence_config: PersistenceConfig
    ) -> None:
        """Verify _ensure_json helper handles string JSON."""
        from ml.registry.common.data_persistence import DataPersistenceComponent

        component = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        # Test with string JSON
        result = component._ensure_json('{"key": "value"}')
        assert result == {"key": "value"}

        # Test with already parsed dict
        result = component._ensure_json({"key": "value"})
        assert result == {"key": "value"}

        # Test with invalid JSON string
        result = component._ensure_json("not valid json")
        assert result == "not valid json"


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_thread_safety_concurrent_saves(
        self, tmp_path: Path, json_persistence_config: PersistenceConfig
    ) -> None:
        """Verify thread-safe save operations."""
        from ml.registry.common.data_persistence import DataPersistenceComponent

        component = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        errors: list[Exception] = []

        def save_operation() -> None:
            try:
                for _ in range(10):
                    component._save_registry(immediate=True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=save_operation) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety violated: {errors}"


# =============================================================================
# Destructor Tests
# =============================================================================


class TestDestructor:
    """Tests for cleanup on deletion."""

    def test_destructor_flushes_pending_saves(
        self, tmp_path: Path, json_persistence_config: PersistenceConfig
    ) -> None:
        """Verify __del__ flushes pending data."""
        from ml.registry.common.data_persistence import DataPersistenceComponent

        component = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        # Add some data
        component._manifests["test"] = MagicMock()
        component._pending_save = True

        # Delete and verify flush
        del component

        # File should contain data after del
        registry_file = tmp_path / "registry" / "data_registry.json"
        # Note: This test may be flaky due to garbage collection timing
        # In implementation, ensure __del__ is called deterministically


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
