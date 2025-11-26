#!/usr/bin/env python3
"""
Unit tests for ManifestManagerComponent.

Tests cover dataset manifest CRUD operations, contract creation,
and validation logic extracted from the legacy DataRegistry.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from ml.registry.dataclasses import (
    DataContract,
    DatasetManifest,
    DatasetType,
    QualityFlag,
    StorageKind,
    ValidationRule,
    ValidationRuleType,
)
from ml.registry.persistence import BackendType, PersistenceConfig


if TYPE_CHECKING:
    from ml.registry.common.manifest_manager import ManifestManagerComponent


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
        constraints={
            "nullability": {"instrument_id": False, "ts_event": False},
            "ranges": {"close": {"min": 0.0}},
        },
        lineage=[],
        pipeline_signature="test_pipeline_v1",
        version="1.0.0",
    )


@pytest.fixture
def earnings_manifest() -> DatasetManifest:
    """Create earnings dataset manifest for contract auto-creation tests."""
    return DatasetManifest(
        dataset_id="ml.earnings_actuals",
        dataset_type=DatasetType.EARNINGS_ACTUALS,
        storage_kind=StorageKind.POSTGRES,
        location="ml_earnings_actuals",
        partitioning={},
        retention_days=365,
        schema={
            "instrument_id": "str",
            "ts_event": "int64",
            "ts_init": "int64",
            "actual_eps": "float64",
        },
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["instrument_id", "ts_event"],
        schema_hash="",
        constraints={},
        lineage=[],
        pipeline_signature="earnings_pipeline",
        version="1.0.0",
    )


# =============================================================================
# Registration Tests
# =============================================================================


class TestRegisterDataset:
    """Tests for register_dataset method."""

    def test_register_dataset_success(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Verify successful dataset registration."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.manifest_manager import ManifestManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = ManifestManagerComponent(persistence=persistence)

        result = component.register_dataset(sample_dataset_manifest)

        assert result == sample_dataset_manifest.dataset_id
        manifest = component.get_manifest(sample_dataset_manifest.dataset_id)
        assert manifest.dataset_id == sample_dataset_manifest.dataset_id

    def test_register_dataset_duplicate_raises(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Verify duplicate registration raises ValueError."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.manifest_manager import ManifestManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = ManifestManagerComponent(persistence=persistence)

        component.register_dataset(sample_dataset_manifest)

        with pytest.raises(ValueError, match="already exists"):
            component.register_dataset(sample_dataset_manifest)

    def test_register_dataset_creates_contract_for_earnings(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
        earnings_manifest: DatasetManifest,
    ) -> None:
        """Verify contract auto-created for earnings datasets."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.manifest_manager import ManifestManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = ManifestManagerComponent(persistence=persistence)

        component.register_dataset(earnings_manifest)

        contract = component.get_contract(earnings_manifest.dataset_id)
        assert contract is not None
        assert contract.dataset_id == earnings_manifest.dataset_id


# =============================================================================
# Update Tests
# =============================================================================


class TestUpdateManifest:
    """Tests for update_manifest method."""

    def test_update_manifest_success(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Verify manifest update with valid changes."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.manifest_manager import ManifestManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = ManifestManagerComponent(persistence=persistence)

        component.register_dataset(sample_dataset_manifest)

        original_modified = sample_dataset_manifest.last_modified

        component.update_manifest(
            sample_dataset_manifest.dataset_id,
            {"retention_days": 60, "version": "1.1.0"},
        )

        updated = component.get_manifest(sample_dataset_manifest.dataset_id)
        assert updated.retention_days == 60
        assert updated.version == "1.1.0"
        assert updated.last_modified >= original_modified

    def test_update_manifest_not_found_raises(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify update on non-existent dataset raises."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.manifest_manager import ManifestManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = ManifestManagerComponent(persistence=persistence)

        with pytest.raises(ValueError, match="not found"):
            component.update_manifest("unknown_dataset", {"version": "2.0"})

    def test_update_manifest_no_valid_fields_raises(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Verify update with no valid fields raises."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.manifest_manager import ManifestManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = ManifestManagerComponent(persistence=persistence)

        component.register_dataset(sample_dataset_manifest)

        with pytest.raises(ValueError, match="No valid fields"):
            component.update_manifest(sample_dataset_manifest.dataset_id, {})


# =============================================================================
# Deprecation Tests
# =============================================================================


class TestDeprecate:
    """Tests for deprecate method."""

    def test_deprecate_dataset(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Verify deprecation marks metadata correctly."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.manifest_manager import ManifestManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = ManifestManagerComponent(persistence=persistence)

        component.register_dataset(sample_dataset_manifest)
        component.deprecate(sample_dataset_manifest.dataset_id)

        manifest = component.get_manifest(sample_dataset_manifest.dataset_id)
        assert manifest.metadata.get("deprecated") is True
        assert "deprecated_at" in manifest.metadata

    def test_deprecate_not_found_raises(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify deprecate on non-existent dataset raises."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.manifest_manager import ManifestManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = ManifestManagerComponent(persistence=persistence)

        with pytest.raises(ValueError, match="not found"):
            component.deprecate("unknown_dataset")


# =============================================================================
# List/Get Tests
# =============================================================================


class TestListAndGet:
    """Tests for list_manifests and get_manifest methods."""

    def test_list_manifests_empty(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify list_manifests returns empty list when no manifests."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.manifest_manager import ManifestManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = ManifestManagerComponent(persistence=persistence)

        result = component.list_manifests()

        assert len(result) == 0

    def test_list_manifests_sorted(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify list_manifests returns sorted list."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.manifest_manager import ManifestManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = ManifestManagerComponent(persistence=persistence)

        # Register in non-alphabetical order
        for name in ["zebra", "alpha", "middle"]:
            manifest = DatasetManifest(
                dataset_id=f"{name}_dataset",
                dataset_type=DatasetType.FEATURES,
                storage_kind=StorageKind.PARQUET,
                location=f"/tmp/{name}",
                partitioning={},
                retention_days=7,
                schema={"instrument_id": "str", "ts_event": "int64", "ts_init": "int64"},
                ts_field="ts_event",
                seq_field=None,
                primary_keys=["instrument_id", "ts_event"],
                schema_hash="",
                constraints={},
                lineage=[],
                pipeline_signature="sig",
                version="1.0",
            )
            component.register_dataset(manifest)

        result = component.list_manifests()
        ids = [m.dataset_id for m in result]

        assert ids == sorted(ids)

    def test_get_manifest_success(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Verify get_manifest retrieves correct manifest."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.manifest_manager import ManifestManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = ManifestManagerComponent(persistence=persistence)

        component.register_dataset(sample_dataset_manifest)

        result = component.get_manifest(sample_dataset_manifest.dataset_id)

        assert result.dataset_id == sample_dataset_manifest.dataset_id

    def test_get_manifest_not_found_raises(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify get_manifest raises for unknown dataset."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.manifest_manager import ManifestManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = ManifestManagerComponent(persistence=persistence)

        with pytest.raises(ValueError, match="not found"):
            component.get_manifest("unknown_dataset")


# =============================================================================
# Contract Creation Tests
# =============================================================================


class TestContractCreation:
    """Tests for _create_contract_from_manifest and get_contract methods."""

    def test_create_contract_from_manifest_with_ranges(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Verify contract creation extracts range constraints."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.manifest_manager import ManifestManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = ManifestManagerComponent(persistence=persistence)

        contract = component._create_contract_from_manifest(sample_dataset_manifest)

        range_rules = [
            r for r in contract.validation_rules
            if r.rule_type == ValidationRuleType.RANGE
        ]
        assert len(range_rules) >= 1

    def test_create_contract_from_manifest_with_nullability(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Verify contract creation extracts nullability constraints."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.manifest_manager import ManifestManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = ManifestManagerComponent(persistence=persistence)

        contract = component._create_contract_from_manifest(sample_dataset_manifest)

        null_rules = [
            r for r in contract.validation_rules
            if r.rule_type == ValidationRuleType.NULLABILITY
        ]
        assert len(null_rules) >= 1

    def test_create_contract_from_manifest_with_regex(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify contract creation extracts regex constraints."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.manifest_manager import ManifestManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = ManifestManagerComponent(persistence=persistence)

        manifest = DatasetManifest(
            dataset_id="regex_test",
            dataset_type=DatasetType.FEATURES,
            storage_kind=StorageKind.PARQUET,
            location="/tmp/regex",
            partitioning={},
            retention_days=7,
            schema={"instrument_id": "str", "ts_event": "int64", "ts_init": "int64"},
            ts_field="ts_event",
            seq_field=None,
            primary_keys=["instrument_id", "ts_event"],
            schema_hash="",
            constraints={"regex": {"instrument_id": r"^[A-Z]+\.[A-Z]+$"}},
            lineage=[],
            pipeline_signature="sig",
            version="1.0",
        )

        contract = component._create_contract_from_manifest(manifest)

        regex_rules = [
            r for r in contract.validation_rules
            if r.rule_type == ValidationRuleType.REGEX
        ]
        assert len(regex_rules) == 1
        assert regex_rules[0].parameters["pattern"] == r"^[A-Z]+\.[A-Z]+$"

    def test_create_contract_from_manifest_default_rule(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify default TYPE_CHECK rule when no constraints."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.manifest_manager import ManifestManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = ManifestManagerComponent(persistence=persistence)

        manifest = DatasetManifest(
            dataset_id="no_constraints",
            dataset_type=DatasetType.FEATURES,
            storage_kind=StorageKind.PARQUET,
            location="/tmp/no_constraints",
            partitioning={},
            retention_days=7,
            schema={"instrument_id": "str", "ts_event": "int64", "ts_init": "int64"},
            ts_field="ts_event",
            seq_field=None,
            primary_keys=["instrument_id", "ts_event"],
            schema_hash="",
            constraints={},
            lineage=[],
            pipeline_signature="sig",
            version="1.0",
        )

        contract = component._create_contract_from_manifest(manifest)

        assert len(contract.validation_rules) == 1
        assert contract.validation_rules[0].rule_type == ValidationRuleType.TYPE_CHECK
        assert contract.validation_rules[0].field_name == "*"

    def test_get_contract_creates_if_missing(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Verify get_contract creates contract on demand."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.manifest_manager import ManifestManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = ManifestManagerComponent(persistence=persistence)

        component.register_dataset(sample_dataset_manifest)

        contract = component.get_contract(sample_dataset_manifest.dataset_id)

        assert isinstance(contract, DataContract)
        assert contract.dataset_id == sample_dataset_manifest.dataset_id

    def test_get_contract_returns_cached(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Verify get_contract returns cached contract."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.manifest_manager import ManifestManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = ManifestManagerComponent(persistence=persistence)

        component.register_dataset(sample_dataset_manifest)

        contract1 = component.get_contract(sample_dataset_manifest.dataset_id)
        contract2 = component.get_contract(sample_dataset_manifest.dataset_id)

        # Should be same cached object
        assert contract1 is contract2

    def test_get_contract_dataset_not_found_raises(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify get_contract raises for unknown dataset."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.manifest_manager import ManifestManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = ManifestManagerComponent(persistence=persistence)

        with pytest.raises(ValueError, match="not found"):
            component.get_contract("unknown_dataset")


# =============================================================================
# Event Emission Tests
# =============================================================================


class TestEventEmission:
    """Tests for event emission during manifest operations."""

    def test_register_dataset_emits_catalog_written_event(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Verify registration emits CATALOG_WRITTEN event."""
        from ml.config.events import Stage
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.manifest_manager import ManifestManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = ManifestManagerComponent(persistence=persistence)

        component.register_dataset(sample_dataset_manifest)

        # Check events were emitted
        events = persistence._events
        catalog_events = [
            e for e in events
            if e.get("stage") == Stage.CATALOG_WRITTEN.value
        ]
        assert len(catalog_events) >= 1

    def test_deprecate_emits_event_with_deprecated_metadata(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Verify deprecation event includes deprecated flag."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.manifest_manager import ManifestManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = ManifestManagerComponent(persistence=persistence)

        component.register_dataset(sample_dataset_manifest)
        component.deprecate(sample_dataset_manifest.dataset_id)

        events = persistence._events
        deprecate_events = [
            e for e in events
            if e.get("metadata", {}).get("deprecated") is True
        ]
        assert len(deprecate_events) >= 1


# =============================================================================
# Audit Logging Tests
# =============================================================================


class TestAuditLogging:
    """Tests for audit log entries."""

    def test_register_dataset_logs_audit(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Verify audit log written on registration."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.manifest_manager import ManifestManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        with patch.object(persistence.persistence, "log_audit") as mock_audit:
            component = ManifestManagerComponent(persistence=persistence)
            component.register_dataset(sample_dataset_manifest)

            mock_audit.assert_called()
            call_kwargs = mock_audit.call_args[1]
            assert call_kwargs["entity_type"] == "dataset"
            assert call_kwargs["action"] == "register"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
