#!/usr/bin/env python3
"""
Parity tests for DataRegistryFacade vs legacy DataRegistry.

Tests verify that the facade produces identical results to the legacy
implementation for all public operations.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from ml.config.events import EventStatus, Source, Stage
from ml.registry.dataclasses import (
    DatasetManifest,
    DatasetType,
    StorageKind,
)
from ml.registry.persistence import BackendType, PersistenceConfig


if TYPE_CHECKING:
    from ml.registry.data_registry import DataRegistry
    from ml.registry.data_registry_facade import DataRegistryFacade


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
        dataset_id="features_parity_test",
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
def legacy_data_registry(tmp_path: Path) -> DataRegistry:
    """Create legacy DataRegistry for parity testing."""
    from ml.registry.data_registry import DataRegistry

    config = PersistenceConfig(
        backend=BackendType.JSON,
        json_path=tmp_path / "legacy_registry",
    )

    return DataRegistry(
        registry_path=tmp_path / "legacy_registry",
        persistence_config=config,
        batch_save_interval=0.0,
    )


@pytest.fixture
def data_registry_facade(tmp_path: Path) -> DataRegistryFacade:
    """Create DataRegistryFacade for parity testing."""
    from ml.registry.data_registry_facade import DataRegistryFacade

    config = PersistenceConfig(
        backend=BackendType.JSON,
        json_path=tmp_path / "facade_registry",
    )

    return DataRegistryFacade(
        registry_path=tmp_path / "facade_registry",
        persistence_config=config,
        batch_save_interval=0.0,
    )


# =============================================================================
# Manifest Parity Tests
# =============================================================================


class TestManifestParity:
    """Tests for manifest operation parity."""

    def test_register_dataset_parity(
        self,
        legacy_data_registry: DataRegistry,
        data_registry_facade: DataRegistryFacade,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Verify legacy and facade produce identical results."""
        # Register in both
        legacy_id = legacy_data_registry.register_dataset(sample_dataset_manifest)

        # Create new manifest for facade (same content, same ID)
        facade_manifest = DatasetManifest(
            dataset_id=sample_dataset_manifest.dataset_id,
            dataset_type=sample_dataset_manifest.dataset_type,
            storage_kind=sample_dataset_manifest.storage_kind,
            location=sample_dataset_manifest.location,
            partitioning=sample_dataset_manifest.partitioning,
            retention_days=sample_dataset_manifest.retention_days,
            schema=sample_dataset_manifest.schema,
            ts_field=sample_dataset_manifest.ts_field,
            seq_field=sample_dataset_manifest.seq_field,
            primary_keys=sample_dataset_manifest.primary_keys,
            schema_hash=sample_dataset_manifest.schema_hash,
            constraints=sample_dataset_manifest.constraints,
            lineage=sample_dataset_manifest.lineage,
            pipeline_signature=sample_dataset_manifest.pipeline_signature,
            version=sample_dataset_manifest.version,
        )
        facade_id = data_registry_facade.register_dataset(facade_manifest)

        # Results should match
        assert legacy_id == facade_id

        # Retrieved manifests should match
        legacy_manifest = legacy_data_registry.get_manifest(sample_dataset_manifest.dataset_id)
        facade_retrieved = data_registry_facade.get_manifest(sample_dataset_manifest.dataset_id)

        assert legacy_manifest.dataset_id == facade_retrieved.dataset_id
        assert legacy_manifest.dataset_type == facade_retrieved.dataset_type
        assert legacy_manifest.storage_kind == facade_retrieved.storage_kind

    def test_update_manifest_parity(
        self,
        legacy_data_registry: DataRegistry,
        data_registry_facade: DataRegistryFacade,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Verify update produces identical results."""
        # Register in both
        legacy_data_registry.register_dataset(sample_dataset_manifest)

        facade_manifest = DatasetManifest(
            dataset_id=sample_dataset_manifest.dataset_id,
            dataset_type=sample_dataset_manifest.dataset_type,
            storage_kind=sample_dataset_manifest.storage_kind,
            location=sample_dataset_manifest.location,
            partitioning=sample_dataset_manifest.partitioning,
            retention_days=sample_dataset_manifest.retention_days,
            schema=sample_dataset_manifest.schema,
            ts_field=sample_dataset_manifest.ts_field,
            seq_field=sample_dataset_manifest.seq_field,
            primary_keys=sample_dataset_manifest.primary_keys,
            schema_hash=sample_dataset_manifest.schema_hash,
            constraints=sample_dataset_manifest.constraints,
            lineage=sample_dataset_manifest.lineage,
            pipeline_signature=sample_dataset_manifest.pipeline_signature,
            version=sample_dataset_manifest.version,
        )
        data_registry_facade.register_dataset(facade_manifest)

        # Update both
        changes = {"version": "2.0.0", "retention_days": 60}
        legacy_data_registry.update_manifest(sample_dataset_manifest.dataset_id, changes)
        data_registry_facade.update_manifest(sample_dataset_manifest.dataset_id, changes)

        # Compare results
        legacy_manifest = legacy_data_registry.get_manifest(sample_dataset_manifest.dataset_id)
        facade_manifest = data_registry_facade.get_manifest(sample_dataset_manifest.dataset_id)

        assert legacy_manifest.version == facade_manifest.version
        assert legacy_manifest.retention_days == facade_manifest.retention_days

    def test_list_manifests_parity(
        self,
        legacy_data_registry: DataRegistry,
        data_registry_facade: DataRegistryFacade,
    ) -> None:
        """Verify manifest listing identical."""
        # Register multiple manifests in both
        for i in range(3):
            manifest = DatasetManifest(
                dataset_id=f"dataset_{i}",
                dataset_type=DatasetType.FEATURES,
                storage_kind=StorageKind.PARQUET,
                location=f"/tmp/{i}",
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
            legacy_data_registry.register_dataset(manifest)

            # Create fresh manifest for facade
            facade_manifest = DatasetManifest(
                dataset_id=f"dataset_{i}",
                dataset_type=DatasetType.FEATURES,
                storage_kind=StorageKind.PARQUET,
                location=f"/tmp/{i}",
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
            data_registry_facade.register_dataset(facade_manifest)

        legacy_list = legacy_data_registry.list_manifests()
        facade_list = data_registry_facade.list_manifests()

        assert len(legacy_list) == len(facade_list)
        assert [m.dataset_id for m in legacy_list] == [m.dataset_id for m in facade_list]

    def test_deprecate_parity(
        self,
        legacy_data_registry: DataRegistry,
        data_registry_facade: DataRegistryFacade,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Verify deprecation identical."""
        # Register in both
        legacy_data_registry.register_dataset(sample_dataset_manifest)

        facade_manifest = DatasetManifest(
            dataset_id=sample_dataset_manifest.dataset_id,
            dataset_type=sample_dataset_manifest.dataset_type,
            storage_kind=sample_dataset_manifest.storage_kind,
            location=sample_dataset_manifest.location,
            partitioning=sample_dataset_manifest.partitioning,
            retention_days=sample_dataset_manifest.retention_days,
            schema=sample_dataset_manifest.schema,
            ts_field=sample_dataset_manifest.ts_field,
            seq_field=sample_dataset_manifest.seq_field,
            primary_keys=sample_dataset_manifest.primary_keys,
            schema_hash=sample_dataset_manifest.schema_hash,
            constraints=sample_dataset_manifest.constraints,
            lineage=sample_dataset_manifest.lineage,
            pipeline_signature=sample_dataset_manifest.pipeline_signature,
            version=sample_dataset_manifest.version,
        )
        data_registry_facade.register_dataset(facade_manifest)

        # Deprecate both
        legacy_data_registry.deprecate(sample_dataset_manifest.dataset_id)
        data_registry_facade.deprecate(sample_dataset_manifest.dataset_id)

        # Compare results
        legacy_manifest = legacy_data_registry.get_manifest(sample_dataset_manifest.dataset_id)
        facade_manifest = data_registry_facade.get_manifest(sample_dataset_manifest.dataset_id)

        assert legacy_manifest.metadata.get("deprecated") == facade_manifest.metadata.get("deprecated")


# =============================================================================
# Contract Parity Tests
# =============================================================================


class TestContractParity:
    """Tests for contract operation parity."""

    def test_get_contract_parity(
        self,
        legacy_data_registry: DataRegistry,
        data_registry_facade: DataRegistryFacade,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Verify contract retrieval identical."""
        # Register in both
        legacy_data_registry.register_dataset(sample_dataset_manifest)

        facade_manifest = DatasetManifest(
            dataset_id=sample_dataset_manifest.dataset_id,
            dataset_type=sample_dataset_manifest.dataset_type,
            storage_kind=sample_dataset_manifest.storage_kind,
            location=sample_dataset_manifest.location,
            partitioning=sample_dataset_manifest.partitioning,
            retention_days=sample_dataset_manifest.retention_days,
            schema=sample_dataset_manifest.schema,
            ts_field=sample_dataset_manifest.ts_field,
            seq_field=sample_dataset_manifest.seq_field,
            primary_keys=sample_dataset_manifest.primary_keys,
            schema_hash=sample_dataset_manifest.schema_hash,
            constraints=sample_dataset_manifest.constraints,
            lineage=sample_dataset_manifest.lineage,
            pipeline_signature=sample_dataset_manifest.pipeline_signature,
            version=sample_dataset_manifest.version,
        )
        data_registry_facade.register_dataset(facade_manifest)

        # Get contracts
        legacy_contract = legacy_data_registry.get_contract(sample_dataset_manifest.dataset_id)
        facade_contract = data_registry_facade.get_contract(sample_dataset_manifest.dataset_id)

        assert legacy_contract.contract_id == facade_contract.contract_id
        assert legacy_contract.dataset_id == facade_contract.dataset_id
        assert len(legacy_contract.validation_rules) == len(facade_contract.validation_rules)


# =============================================================================
# Event Parity Tests
# =============================================================================


class TestEventParity:
    """Tests for event operation parity."""

    def test_emit_event_parity(
        self,
        legacy_data_registry: DataRegistry,
        data_registry_facade: DataRegistryFacade,
    ) -> None:
        """Verify event emission identical."""
        event_params = {
            "dataset_id": "test_dataset",
            "instrument_id": "EUR/USD",
            "stage": Stage.CATALOG_WRITTEN,
            "source": Source.HISTORICAL,
            "run_id": "run_1",
            "ts_min": 1000000000000000000,
            "ts_max": 2000000000000000000,
            "count": 100,
            "status": EventStatus.SUCCESS,
            "metadata": {"key": "value"},
        }

        legacy_data_registry.emit_event(**event_params)
        data_registry_facade.emit_event(**event_params)

        # Compare events
        legacy_events = legacy_data_registry._events
        facade_events = data_registry_facade._persistence._events

        assert len(legacy_events) == len(facade_events)

        legacy_event = legacy_events[-1]
        facade_event = facade_events[-1]

        assert legacy_event["dataset_id"] == facade_event["dataset_id"]
        assert legacy_event["stage"] == facade_event["stage"]
        assert legacy_event["count"] == facade_event["count"]


# =============================================================================
# Watermark Parity Tests
# =============================================================================


class TestWatermarkParity:
    """Tests for watermark operation parity."""

    def test_update_watermark_parity(
        self,
        legacy_data_registry: DataRegistry,
        data_registry_facade: DataRegistryFacade,
    ) -> None:
        """Verify watermark updates identical."""
        watermark_params = {
            "dataset_id": "test_dataset",
            "instrument_id": "EUR/USD",
            "source": Source.LIVE,
            "last_success_ns": 1000000000000000000,
            "count": 100,
            "completeness_pct": 99.5,
        }

        legacy_data_registry.update_watermark(**watermark_params)
        data_registry_facade.update_watermark(**watermark_params)

        # Get and compare
        legacy_wm = legacy_data_registry.get_watermark("test_dataset", "EUR/USD", Source.LIVE)
        facade_wm = data_registry_facade.get_watermark("test_dataset", "EUR/USD", Source.LIVE)

        assert legacy_wm is not None
        assert facade_wm is not None
        assert legacy_wm.last_success_ns == facade_wm.last_success_ns
        assert legacy_wm.last_count == facade_wm.last_count
        assert legacy_wm.completeness_pct == facade_wm.completeness_pct

    def test_iter_watermarks_parity(
        self,
        legacy_data_registry: DataRegistry,
        data_registry_facade: DataRegistryFacade,
    ) -> None:
        """Verify watermark iteration identical."""
        for i in range(3):
            watermark_params = {
                "dataset_id": f"dataset_{i}",
                "instrument_id": "EUR/USD",
                "source": Source.LIVE,
                "last_success_ns": 1000000000000000000 + i,
                "count": 100 + i,
                "completeness_pct": 99.5,
            }
            legacy_data_registry.update_watermark(**watermark_params)
            data_registry_facade.update_watermark(**watermark_params)

        legacy_wms = list(legacy_data_registry.iter_watermarks())
        facade_wms = list(data_registry_facade.iter_watermarks())

        assert len(legacy_wms) == len(facade_wms)


# =============================================================================
# Lineage Parity Tests
# =============================================================================


class TestLineageParity:
    """Tests for lineage operation parity."""

    def test_link_lineage_parity(
        self,
        legacy_data_registry: DataRegistry,
        data_registry_facade: DataRegistryFacade,
    ) -> None:
        """Verify lineage linking identical."""
        lineage_params = {
            "child_dataset_id": "features_child",
            "parent_ids": ["bars_parent", "quotes_parent"],
            "transform_id": "transform_v1",
            "ts_range": {"start_ns": 1000, "end_ns": 2000},
            "params": {"lookback": 20},
        }

        legacy_data_registry.link_lineage(**lineage_params)
        data_registry_facade.link_lineage(**lineage_params)

        legacy_lineage = list(legacy_data_registry.iter_lineage(child="features_child"))
        facade_lineage = list(data_registry_facade.iter_lineage(child="features_child"))

        assert len(legacy_lineage) == len(facade_lineage)

    def test_iter_lineage_parity(
        self,
        legacy_data_registry: DataRegistry,
        data_registry_facade: DataRegistryFacade,
    ) -> None:
        """Verify lineage iteration identical."""
        for i in range(3):
            legacy_data_registry.link_lineage(
                child_dataset_id=f"child_{i}",
                parent_ids=[f"parent_{i}"],
                transform_id=f"transform_{i}",
                ts_range={"start_ns": 1000, "end_ns": 2000},
                params={},
            )
            data_registry_facade.link_lineage(
                child_dataset_id=f"child_{i}",
                parent_ids=[f"parent_{i}"],
                transform_id=f"transform_{i}",
                ts_range={"start_ns": 1000, "end_ns": 2000},
                params={},
            )

        legacy_lineage = list(legacy_data_registry.iter_lineage())
        facade_lineage = list(data_registry_facade.iter_lineage())

        assert len(legacy_lineage) == len(facade_lineage)


# =============================================================================
# Pipeline Signature Parity Tests
# =============================================================================


class TestPipelineSignatureParity:
    """Tests for pipeline signature operation parity."""

    def test_get_pipeline_signature_parity(
        self,
        legacy_data_registry: DataRegistry,
        data_registry_facade: DataRegistryFacade,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Verify signature retrieval identical."""
        legacy_data_registry.register_dataset(sample_dataset_manifest)

        facade_manifest = DatasetManifest(
            dataset_id=sample_dataset_manifest.dataset_id,
            dataset_type=sample_dataset_manifest.dataset_type,
            storage_kind=sample_dataset_manifest.storage_kind,
            location=sample_dataset_manifest.location,
            partitioning=sample_dataset_manifest.partitioning,
            retention_days=sample_dataset_manifest.retention_days,
            schema=sample_dataset_manifest.schema,
            ts_field=sample_dataset_manifest.ts_field,
            seq_field=sample_dataset_manifest.seq_field,
            primary_keys=sample_dataset_manifest.primary_keys,
            schema_hash=sample_dataset_manifest.schema_hash,
            constraints=sample_dataset_manifest.constraints,
            lineage=sample_dataset_manifest.lineage,
            pipeline_signature=sample_dataset_manifest.pipeline_signature,
            version=sample_dataset_manifest.version,
        )
        data_registry_facade.register_dataset(facade_manifest)

        legacy_sig = legacy_data_registry.get_pipeline_signature(sample_dataset_manifest.dataset_id)
        facade_sig = data_registry_facade.get_pipeline_signature(sample_dataset_manifest.dataset_id)

        assert legacy_sig == facade_sig

    def test_set_pipeline_signature_parity(
        self,
        legacy_data_registry: DataRegistry,
        data_registry_facade: DataRegistryFacade,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Verify signature setting identical."""
        legacy_data_registry.register_dataset(sample_dataset_manifest)

        facade_manifest = DatasetManifest(
            dataset_id=sample_dataset_manifest.dataset_id,
            dataset_type=sample_dataset_manifest.dataset_type,
            storage_kind=sample_dataset_manifest.storage_kind,
            location=sample_dataset_manifest.location,
            partitioning=sample_dataset_manifest.partitioning,
            retention_days=sample_dataset_manifest.retention_days,
            schema=sample_dataset_manifest.schema,
            ts_field=sample_dataset_manifest.ts_field,
            seq_field=sample_dataset_manifest.seq_field,
            primary_keys=sample_dataset_manifest.primary_keys,
            schema_hash=sample_dataset_manifest.schema_hash,
            constraints=sample_dataset_manifest.constraints,
            lineage=sample_dataset_manifest.lineage,
            pipeline_signature=sample_dataset_manifest.pipeline_signature,
            version=sample_dataset_manifest.version,
        )
        data_registry_facade.register_dataset(facade_manifest)

        new_sig = "new_sig_v2_sha256"

        legacy_data_registry.set_pipeline_signature(sample_dataset_manifest.dataset_id, new_sig)
        data_registry_facade.set_pipeline_signature(sample_dataset_manifest.dataset_id, new_sig)

        legacy_result = legacy_data_registry.get_pipeline_signature(sample_dataset_manifest.dataset_id)
        facade_result = data_registry_facade.get_pipeline_signature(sample_dataset_manifest.dataset_id)

        assert legacy_result == facade_result


# =============================================================================
# Full Workflow Parity Test
# =============================================================================


class TestFullWorkflowParity:
    """Tests for complete workflow parity."""

    def test_roundtrip_parity(
        self,
        legacy_data_registry: DataRegistry,
        data_registry_facade: DataRegistryFacade,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Full workflow roundtrip parity."""
        # Register
        legacy_data_registry.register_dataset(sample_dataset_manifest)

        facade_manifest = DatasetManifest(
            dataset_id=sample_dataset_manifest.dataset_id,
            dataset_type=sample_dataset_manifest.dataset_type,
            storage_kind=sample_dataset_manifest.storage_kind,
            location=sample_dataset_manifest.location,
            partitioning=sample_dataset_manifest.partitioning,
            retention_days=sample_dataset_manifest.retention_days,
            schema=sample_dataset_manifest.schema,
            ts_field=sample_dataset_manifest.ts_field,
            seq_field=sample_dataset_manifest.seq_field,
            primary_keys=sample_dataset_manifest.primary_keys,
            schema_hash=sample_dataset_manifest.schema_hash,
            constraints=sample_dataset_manifest.constraints,
            lineage=sample_dataset_manifest.lineage,
            pipeline_signature=sample_dataset_manifest.pipeline_signature,
            version=sample_dataset_manifest.version,
        )
        data_registry_facade.register_dataset(facade_manifest)

        # Emit event
        event_params = {
            "dataset_id": sample_dataset_manifest.dataset_id,
            "instrument_id": "EUR/USD",
            "stage": Stage.CATALOG_WRITTEN,
            "source": Source.HISTORICAL,
            "run_id": "run_1",
            "ts_min": 1000000000000000000,
            "ts_max": 2000000000000000000,
            "count": 100,
            "status": EventStatus.SUCCESS,
        }
        legacy_data_registry.emit_event(**event_params)
        data_registry_facade.emit_event(**event_params)

        # Update watermark
        watermark_params = {
            "dataset_id": sample_dataset_manifest.dataset_id,
            "instrument_id": "EUR/USD",
            "source": Source.LIVE,
            "last_success_ns": 2000000000000000000,
            "count": 200,
            "completeness_pct": 100.0,
        }
        legacy_data_registry.update_watermark(**watermark_params)
        data_registry_facade.update_watermark(**watermark_params)

        # Link lineage
        lineage_params = {
            "child_dataset_id": sample_dataset_manifest.dataset_id,
            "parent_ids": ["parent_bars"],
            "transform_id": "transform_v1",
            "ts_range": {"start_ns": 1000000000000000000, "end_ns": 2000000000000000000},
            "params": {"lookback": 20},
        }
        legacy_data_registry.link_lineage(**lineage_params)
        data_registry_facade.link_lineage(**lineage_params)

        # Update manifest
        legacy_data_registry.update_manifest(sample_dataset_manifest.dataset_id, {"version": "1.1.0"})
        data_registry_facade.update_manifest(sample_dataset_manifest.dataset_id, {"version": "1.1.0"})

        # Deprecate
        legacy_data_registry.deprecate(sample_dataset_manifest.dataset_id)
        data_registry_facade.deprecate(sample_dataset_manifest.dataset_id)

        # Flush
        legacy_data_registry.flush()
        data_registry_facade.flush()

        # Final comparison
        legacy_manifest = legacy_data_registry.get_manifest(sample_dataset_manifest.dataset_id)
        facade_manifest = data_registry_facade.get_manifest(sample_dataset_manifest.dataset_id)

        assert legacy_manifest.version == facade_manifest.version
        assert legacy_manifest.metadata.get("deprecated") == facade_manifest.metadata.get("deprecated")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
