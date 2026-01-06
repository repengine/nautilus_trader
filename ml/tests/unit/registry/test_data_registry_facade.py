#!/usr/bin/env python3
"""
Unit tests for DataRegistryFacade.

Tests cover delegation to components, feature flags, backward compatibility,
and the unified interface for the decomposed DataRegistry.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from ml.common.protocols import MLComponentMixin
from ml.config.events import EventStatus, Source, Stage
from ml.registry.dataclasses import (
    DatasetManifest,
    DatasetType,
    StorageKind,
)
from ml.registry.persistence import BackendType, PersistenceConfig


if TYPE_CHECKING:
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


# =============================================================================
# Initialization Tests
# =============================================================================


class TestFacadeInitialization:
    """Tests for facade initialization."""

    def test_facade_init_creates_components(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify facade initializes all components."""
        from ml.registry.data_registry_facade import DataRegistryFacade

        facade = DataRegistryFacade(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        assert facade._persistence is not None
        assert facade._manifest_manager is not None
        assert facade._event_emission is not None
        assert facade._watermark_manager is not None
        assert facade._lineage_tracker is not None

    def test_facade_backward_compatible_init_params(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify init accepts legacy parameters."""
        from ml.registry.data_registry_facade import DataRegistryFacade

        # Should accept all legacy __init__ parameters
        facade = DataRegistryFacade(
            registry_path=tmp_path / "registry",
            batch_save_interval=0.5,
            persistence_config=PersistenceConfig(
                backend=BackendType.JSON,
                json_path=tmp_path / "registry",
            ),
        )

        assert facade is not None


# =============================================================================
# Delegation Tests
# =============================================================================


class TestDelegation:
    """Tests for method delegation to components."""

    def test_facade_delegates_register_dataset(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Verify register_dataset delegated to ManifestManagerComponent."""
        from ml.registry.data_registry_facade import DataRegistryFacade

        facade = DataRegistryFacade(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        result = facade.register_dataset(sample_dataset_manifest)

        assert result == sample_dataset_manifest.dataset_id
        manifest = facade.get_manifest(sample_dataset_manifest.dataset_id)
        assert manifest.dataset_id == sample_dataset_manifest.dataset_id

    def test_facade_delegates_update_manifest(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Verify update_manifest delegated."""
        from ml.registry.data_registry_facade import DataRegistryFacade

        facade = DataRegistryFacade(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        facade.register_dataset(sample_dataset_manifest)
        facade.update_manifest(sample_dataset_manifest.dataset_id, {"version": "2.0.0"})

        manifest = facade.get_manifest(sample_dataset_manifest.dataset_id)
        assert manifest.version == "2.0.0"

    def test_facade_delegates_deprecate(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Verify deprecate delegated."""
        from ml.registry.data_registry_facade import DataRegistryFacade

        facade = DataRegistryFacade(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        facade.register_dataset(sample_dataset_manifest)
        facade.deprecate(sample_dataset_manifest.dataset_id)

        manifest = facade.get_manifest(sample_dataset_manifest.dataset_id)
        assert manifest.metadata.get("deprecated") is True

    def test_facade_delegates_list_manifests(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Verify list_manifests delegated."""
        from ml.registry.data_registry_facade import DataRegistryFacade

        facade = DataRegistryFacade(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        facade.register_dataset(sample_dataset_manifest)

        result = facade.list_manifests()

        assert len(result) == 1
        assert result[0].dataset_id == sample_dataset_manifest.dataset_id

    def test_facade_delegates_get_manifest(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Verify get_manifest delegated."""
        from ml.registry.data_registry_facade import DataRegistryFacade

        facade = DataRegistryFacade(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        facade.register_dataset(sample_dataset_manifest)

        result = facade.get_manifest(sample_dataset_manifest.dataset_id)

        assert result.dataset_id == sample_dataset_manifest.dataset_id

    def test_facade_delegates_get_contract(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Verify get_contract delegated."""
        from ml.registry.data_registry_facade import DataRegistryFacade

        facade = DataRegistryFacade(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        facade.register_dataset(sample_dataset_manifest)

        contract = facade.get_contract(sample_dataset_manifest.dataset_id)

        assert contract.dataset_id == sample_dataset_manifest.dataset_id

    def test_facade_delegates_emit_event(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify emit_event delegated."""
        from ml.registry.data_registry_facade import DataRegistryFacade

        facade = DataRegistryFacade(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        facade.emit_event(
            dataset_id="test_dataset",
            instrument_id="EUR/USD",
            stage=Stage.CATALOG_WRITTEN,
            source=Source.HISTORICAL,
            run_id="run_1",
            ts_min=1000000000000000000,
            ts_max=2000000000000000000,
            count=100,
            status=EventStatus.SUCCESS,
        )

        # Events should be stored via persistence
        assert len(facade._persistence._events) >= 1

    def test_facade_delegates_update_watermark(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify update_watermark delegated."""
        from ml.registry.data_registry_facade import DataRegistryFacade

        facade = DataRegistryFacade(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        facade.update_watermark(
            dataset_id="test_dataset",
            instrument_id="EUR/USD",
            source=Source.LIVE,
            last_success_ns=1000000000000000000,
            count=100,
            completeness_pct=99.5,
        )

        watermark = facade.get_watermark("test_dataset", "EUR/USD", Source.LIVE)
        assert watermark is not None
        assert watermark.last_count == 100

    def test_facade_delegates_get_watermark(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify get_watermark delegated."""
        from ml.registry.data_registry_facade import DataRegistryFacade

        facade = DataRegistryFacade(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        facade.update_watermark(
            dataset_id="test_dataset",
            instrument_id="EUR/USD",
            source=Source.LIVE,
            last_success_ns=1000000000000000000,
            count=100,
            completeness_pct=99.5,
        )

        watermark = facade.get_watermark("test_dataset", "EUR/USD", Source.LIVE)

        assert watermark is not None

    def test_facade_delegates_iter_watermarks(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify iter_watermarks delegated."""
        from ml.registry.data_registry_facade import DataRegistryFacade

        facade = DataRegistryFacade(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        for i in range(3):
            facade.update_watermark(
                dataset_id=f"dataset_{i}",
                instrument_id="EUR/USD",
                source=Source.LIVE,
                last_success_ns=1000000000000000000 + i,
                count=100,
                completeness_pct=99.5,
            )

        result = list(facade.iter_watermarks())

        assert len(result) == 3

    def test_facade_delegates_link_lineage(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify link_lineage delegated."""
        from ml.registry.data_registry_facade import DataRegistryFacade

        facade = DataRegistryFacade(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        facade.link_lineage(
            child_dataset_id="features_child",
            parent_ids=["bars_parent"],
            transform_id="transform_v1",
            ts_range={"start_ns": 1000, "end_ns": 2000},
            params={"lookback": 20},
        )

        lineage = list(facade.iter_lineage(child="features_child"))
        assert len(lineage) == 1

    def test_facade_delegates_iter_lineage(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify iter_lineage delegated."""
        from ml.registry.data_registry_facade import DataRegistryFacade

        facade = DataRegistryFacade(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        facade.link_lineage(
            child_dataset_id="features_child",
            parent_ids=["bars_parent"],
            transform_id="transform_v1",
            ts_range={"start_ns": 1000, "end_ns": 2000},
            params={},
        )

        result = list(facade.iter_lineage())

        assert len(result) == 1

    def test_facade_delegates_get_pipeline_signature(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Verify get_pipeline_signature delegated."""
        from ml.registry.data_registry_facade import DataRegistryFacade

        facade = DataRegistryFacade(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        facade.register_dataset(sample_dataset_manifest)

        result = facade.get_pipeline_signature(sample_dataset_manifest.dataset_id)

        assert result == sample_dataset_manifest.pipeline_signature

    def test_facade_delegates_set_pipeline_signature(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Verify set_pipeline_signature delegated."""
        from ml.registry.data_registry_facade import DataRegistryFacade

        facade = DataRegistryFacade(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        facade.register_dataset(sample_dataset_manifest)

        new_sig = "new_pipeline_sig_v2"
        facade.set_pipeline_signature(sample_dataset_manifest.dataset_id, new_sig)

        result = facade.get_pipeline_signature(sample_dataset_manifest.dataset_id)
        assert result == new_sig

    def test_facade_delegates_flush(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify flush delegated to persistence."""
        from ml.registry.data_registry_facade import DataRegistryFacade

        facade = DataRegistryFacade(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        facade.flush()

        # File should exist after flush
        registry_file = tmp_path / "registry" / "data_registry.json"
        assert registry_file.exists()


# =============================================================================
# Public API Tests
# =============================================================================


class TestPublicAPI:
    """Tests for public API compatibility."""

    def test_facade_preserves_public_api(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify all public methods present."""
        from ml.registry.data_registry_facade import DataRegistryFacade

        facade = DataRegistryFacade(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        # All expected public methods
        public_methods = [
            "register_dataset",
            "update_manifest",
            "deprecate",
            "list_manifests",
            "get_manifest",
            "get_contract",
            "emit_event",
            "update_watermark",
            "get_watermark",
            "iter_watermarks",
            "link_lineage",
            "iter_lineage",
            "get_pipeline_signature",
            "set_pipeline_signature",
            "flush",
        ]

        for method_name in public_methods:
            assert hasattr(facade, method_name), f"Missing public method: {method_name}"
            assert callable(getattr(facade, method_name))

    def test_facade_inherits_ml_component_mixin(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify MLComponentMixin inheritance."""
        from ml.registry.data_registry_facade import DataRegistryFacade

        facade = DataRegistryFacade(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        assert isinstance(facade, MLComponentMixin)


# =============================================================================
# Feature Flag Tests
# =============================================================================


class TestRegistryFactory:
    """Tests for data registry factory behavior."""

    def test_factory_returns_facade(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify factory returns DataRegistryFacade."""
        from ml.registry.data_registry_facade import DataRegistryFacade

        from ml.registry import create_data_registry

        result = create_data_registry(
            registry_path=tmp_path / "registry",
            persistence_config=PersistenceConfig(
                backend=BackendType.JSON,
                json_path=tmp_path / "registry",
            ),
        )

        assert isinstance(result, DataRegistryFacade)


# =============================================================================
# Thread Safety and Destructor Tests
# =============================================================================


class TestThreadSafetyAndCleanup:
    """Tests for thread safety and cleanup."""

    def test_facade_destructor_flushes(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Verify __del__ flushes pending data."""
        from ml.registry.data_registry_facade import DataRegistryFacade

        facade = DataRegistryFacade(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        facade.register_dataset(sample_dataset_manifest)

        del facade

        # File should contain data after del
        registry_file = tmp_path / "registry" / "data_registry.json"
        assert registry_file.exists()

    def test_facade_thread_safe_operations(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify thread-safe concurrent operations."""
        from ml.registry.data_registry_facade import DataRegistryFacade

        facade = DataRegistryFacade(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )

        errors: list[Exception] = []

        def operations(thread_id: int) -> None:
            try:
                for i in range(5):
                    manifest = DatasetManifest(
                        dataset_id=f"dataset_t{thread_id}_{i}",
                        dataset_type=DatasetType.FEATURES,
                        storage_kind=StorageKind.PARQUET,
                        location=f"/tmp/t{thread_id}_{i}",
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
                    facade.register_dataset(manifest)

                    facade.emit_event(
                        dataset_id=f"dataset_t{thread_id}_{i}",
                        instrument_id="EUR/USD",
                        stage=Stage.CATALOG_WRITTEN,
                        source=Source.HISTORICAL,
                        run_id=f"run_t{thread_id}_{i}",
                        ts_min=i,
                        ts_max=i + 1,
                        count=1,
                        status=EventStatus.SUCCESS,
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=operations, args=(i,)) for i in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety violated: {errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
