#!/usr/bin/env python3
"""
Unit tests for LineageTrackerComponent.

Tests cover lineage linking, iteration, pipeline signature management,
and persistence for the DataRegistry lineage tracking extracted from legacy DataRegistry.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from ml.registry.dataclasses import (
    DatasetLineageRecord,
    DatasetManifest,
    DatasetType,
    StorageKind,
)
from ml.registry.persistence import BackendType, PersistenceConfig


if TYPE_CHECKING:
    from ml.registry.common.lineage_tracker import LineageTrackerComponent


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
        },
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["instrument_id", "ts_event"],
        schema_hash="",
        constraints={},
        lineage=[],
        pipeline_signature="test_pipeline_v1",
        version="1.0.0",
    )


@pytest.fixture
def sample_lineage_record() -> DatasetLineageRecord:
    """Create a valid DatasetLineageRecord for testing."""
    return DatasetLineageRecord(
        transform_id="feature_pipeline_v1",
        child_dataset_id="features_microstructure",
        parent_dataset_id="bars_eurusd_1m",
        ts_range={"start_ns": 1_000_000_000_000, "end_ns": 2_000_000_000_000},
        parameters={"lookback_bars": 20},
        created_at=time.time(),
    )


def _build_stub_persistence(
    *,
    backend: BackendType,
    session: Any | None = None,
) -> Any:
    from ml.registry.common.data_persistence import DataPersistenceComponent

    return SimpleNamespace(
        _lock=threading.RLock(),
        backend=backend,
        _lineage=[],
        _manifests={},
        _save_registry=MagicMock(),
        persistence=SimpleNamespace(get_session=lambda: session),
        _lineage_from_row=DataPersistenceComponent._lineage_from_row,
    )


# =============================================================================
# Link Lineage Tests
# =============================================================================


class TestLinkLineage:
    """Tests for link_lineage method."""

    def test_link_lineage_single_parent(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify linking with single parent."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.lineage_tracker import LineageTrackerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = LineageTrackerComponent(persistence=persistence)

        component.link_lineage(
            child_dataset_id="features_child",
            parent_ids=["bars_parent"],
            transform_id="transform_v1",
            ts_range={"start_ns": 1000, "end_ns": 2000},
            params={"lookback": 20},
        )

        assert len(persistence._lineage) == 1
        assert persistence._lineage[0]["parent_dataset_id"] == "bars_parent"

    def test_link_lineage_multiple_parents(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify linking with multiple parents."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.lineage_tracker import LineageTrackerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = LineageTrackerComponent(persistence=persistence)

        parent_ids = ["bars_eurusd", "quotes_eurusd", "trades_eurusd"]

        component.link_lineage(
            child_dataset_id="features_child",
            parent_ids=parent_ids,
            transform_id="transform_v1",
            ts_range={"start_ns": 1000, "end_ns": 2000},
            params={"lookback": 20},
        )

        assert len(persistence._lineage) == 3
        stored_parents = [l["parent_dataset_id"] for l in persistence._lineage]
        assert set(stored_parents) == set(parent_ids)

    def test_link_lineage_stores_ts_range(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify ts_range correctly stored."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.lineage_tracker import LineageTrackerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = LineageTrackerComponent(persistence=persistence)

        ts_range = {"start_ns": 1000000000000, "end_ns": 2000000000000}

        component.link_lineage(
            child_dataset_id="features_child",
            parent_ids=["bars_parent"],
            transform_id="transform_v1",
            ts_range=ts_range,
            params={},
        )

        assert persistence._lineage[0]["ts_range"] == ts_range

    def test_link_lineage_stores_parameters(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify params correctly stored."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.lineage_tracker import LineageTrackerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = LineageTrackerComponent(persistence=persistence)

        params = {"lookback": 20, "normalize": True}

        component.link_lineage(
            child_dataset_id="features_child",
            parent_ids=["bars_parent"],
            transform_id="transform_v1",
            ts_range={"start_ns": 1000, "end_ns": 2000},
            params=params,
        )

        assert persistence._lineage[0]["parameters"] == params

    def test_link_lineage_sets_created_at(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify created_at timestamp set."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.lineage_tracker import LineageTrackerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = LineageTrackerComponent(persistence=persistence)

        before = time.time()

        component.link_lineage(
            child_dataset_id="features_child",
            parent_ids=["bars_parent"],
            transform_id="transform_v1",
            ts_range={"start_ns": 1000, "end_ns": 2000},
            params={},
        )

        after = time.time()

        assert before <= persistence._lineage[0]["created_at"] <= after

    def test_link_lineage_trims_old_entries_json(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify JSON backend trims at 5000 entries."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.lineage_tracker import LineageTrackerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = LineageTrackerComponent(persistence=persistence)

        # Link enough lineage entries to exceed limit
        for i in range(5001):
            component.link_lineage(
                child_dataset_id=f"child_{i}",
                parent_ids=[f"parent_{i}"],
                transform_id=f"transform_{i}",
                ts_range={"start_ns": i, "end_ns": i + 1},
                params={},
            )

        assert len(persistence._lineage) == 5000


# =============================================================================
# Iter Lineage Tests
# =============================================================================


class TestIterLineage:
    """Tests for iter_lineage method."""

    def test_iter_lineage_no_filter(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify all lineage returned without filters."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.lineage_tracker import LineageTrackerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = LineageTrackerComponent(persistence=persistence)

        for i in range(3):
            component.link_lineage(
                child_dataset_id=f"child_{i}",
                parent_ids=[f"parent_{i}"],
                transform_id=f"transform_{i}",
                ts_range={"start_ns": 1000, "end_ns": 2000},
                params={},
            )

        result = list(component.iter_lineage())

        assert len(result) == 3

    def test_iter_lineage_filter_by_child(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify filtering by child_dataset_id."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.lineage_tracker import LineageTrackerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = LineageTrackerComponent(persistence=persistence)

        # Link with multiple children
        for child in ["child_a", "child_b", "child_a"]:
            component.link_lineage(
                child_dataset_id=child,
                parent_ids=[f"parent_for_{child}"],
                transform_id="transform_v1",
                ts_range={"start_ns": 1000, "end_ns": 2000},
                params={},
            )

        result = list(component.iter_lineage(child="child_a"))

        assert len(result) == 2
        assert all(r.child_dataset_id == "child_a" for r in result)

    def test_iter_lineage_filter_by_parent(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify filtering by parent_dataset_id."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.lineage_tracker import LineageTrackerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = LineageTrackerComponent(persistence=persistence)

        # Link with multiple parents
        component.link_lineage(
            child_dataset_id="child_1",
            parent_ids=["parent_x", "parent_y"],
            transform_id="transform_v1",
            ts_range={"start_ns": 1000, "end_ns": 2000},
            params={},
        )
        component.link_lineage(
            child_dataset_id="child_2",
            parent_ids=["parent_x"],
            transform_id="transform_v1",
            ts_range={"start_ns": 1000, "end_ns": 2000},
            params={},
        )

        result = list(component.iter_lineage(parent="parent_x"))

        assert len(result) == 2
        assert all(r.parent_dataset_id == "parent_x" for r in result)

    def test_iter_lineage_with_limit(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify limit parameter respected."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.lineage_tracker import LineageTrackerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = LineageTrackerComponent(persistence=persistence)

        for i in range(5):
            component.link_lineage(
                child_dataset_id=f"child_{i}",
                parent_ids=[f"parent_{i}"],
                transform_id=f"transform_{i}",
                ts_range={"start_ns": 1000, "end_ns": 2000},
                params={},
            )

        result = list(component.iter_lineage(limit=2))

        assert len(result) == 2

    def test_iter_lineage_sorted_by_created_at_desc(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify entries sorted by created_at descending."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.lineage_tracker import LineageTrackerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = LineageTrackerComponent(persistence=persistence)

        for i in range(3):
            component.link_lineage(
                child_dataset_id=f"child_{i}",
                parent_ids=[f"parent_{i}"],
                transform_id=f"transform_{i}",
                ts_range={"start_ns": 1000, "end_ns": 2000},
                params={},
            )
            time.sleep(0.01)

        result = list(component.iter_lineage())

        # Most recent first
        assert result[0].created_at >= result[1].created_at
        assert result[1].created_at >= result[2].created_at


# =============================================================================
# Pipeline Signature Tests
# =============================================================================


class TestPipelineSignature:
    """Tests for pipeline signature get/set methods."""

    def test_get_pipeline_signature_from_manifest(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Verify retrieval from manifest.pipeline_signature."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.lineage_tracker import LineageTrackerComponent
        from ml.registry.common.manifest_manager import ManifestManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        manifest_manager = ManifestManagerComponent(persistence=persistence)
        component = LineageTrackerComponent(persistence=persistence)

        manifest_manager.register_dataset(sample_dataset_manifest)

        result = component.get_pipeline_signature(sample_dataset_manifest.dataset_id)

        assert result == sample_dataset_manifest.pipeline_signature

    def test_get_pipeline_signature_from_metadata(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify retrieval from manifest.metadata['pipeline_signature']."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.lineage_tracker import LineageTrackerComponent
        from ml.registry.common.manifest_manager import ManifestManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        manifest_manager = ManifestManagerComponent(persistence=persistence)
        component = LineageTrackerComponent(persistence=persistence)

        manifest = DatasetManifest(
            dataset_id="test_with_metadata_sig",
            dataset_type=DatasetType.FEATURES,
            storage_kind=StorageKind.PARQUET,
            location="/tmp/test",
            partitioning={},
            retention_days=7,
            schema={"instrument_id": "str", "ts_event": "int64", "ts_init": "int64"},
            ts_field="ts_event",
            seq_field=None,
            primary_keys=["instrument_id", "ts_event"],
            schema_hash="",
            constraints={},
            lineage=[],
            pipeline_signature="",  # Empty direct field
            version="1.0",
            metadata={"pipeline_signature": "sig_from_metadata"},
        )

        manifest_manager.register_dataset(manifest)

        result = component.get_pipeline_signature(manifest.dataset_id)

        assert result == "sig_from_metadata"

    def test_get_pipeline_signature_not_found_returns_none(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify None for missing signature."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.lineage_tracker import LineageTrackerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = LineageTrackerComponent(persistence=persistence)

        result = component.get_pipeline_signature("unknown_dataset")

        assert result is None

    def test_set_pipeline_signature_success(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Verify setting pipeline signature."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.lineage_tracker import LineageTrackerComponent
        from ml.registry.common.manifest_manager import ManifestManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        manifest_manager = ManifestManagerComponent(persistence=persistence)
        component = LineageTrackerComponent(persistence=persistence)

        manifest_manager.register_dataset(sample_dataset_manifest)

        new_signature = "new_sig_v2_sha256abcdef"
        component.set_pipeline_signature(sample_dataset_manifest.dataset_id, new_signature)

        result = component.get_pipeline_signature(sample_dataset_manifest.dataset_id)

        assert result == new_signature

    def test_set_pipeline_signature_empty_raises(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
        sample_dataset_manifest: DatasetManifest,
    ) -> None:
        """Verify empty signature raises ValueError."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.lineage_tracker import LineageTrackerComponent
        from ml.registry.common.manifest_manager import ManifestManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        manifest_manager = ManifestManagerComponent(persistence=persistence)
        component = LineageTrackerComponent(persistence=persistence)

        manifest_manager.register_dataset(sample_dataset_manifest)

        with pytest.raises(ValueError, match="cannot be empty"):
            component.set_pipeline_signature(sample_dataset_manifest.dataset_id, "")

    def test_set_pipeline_signature_missing_dataset_raises(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.lineage_tracker import LineageTrackerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = LineageTrackerComponent(persistence=persistence)

        with pytest.raises(ValueError, match="Dataset 'missing' not found"):
            component.set_pipeline_signature("missing", "sig")


class TestPostgresBranches:
    def test_link_lineage_postgres_success_and_failure_paths(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from ml.registry.common.lineage_tracker import LineageTrackerComponent

        monkeypatch.setattr("ml.registry.common.lineage_tracker.time.time", lambda: 10.0)

        session_success = MagicMock()
        persistence_success = _build_stub_persistence(
            backend=BackendType.POSTGRES,
            session=session_success,
        )
        component_success = LineageTrackerComponent(cast(Any, persistence_success))
        component_success.link_lineage(
            child_dataset_id="child",
            parent_ids=["p1", "p2"],
            transform_id="transform",
            ts_range={"start_ns": 1, "end_ns": 2},
            params={"lookback": 20},
        )

        assert session_success.execute.call_count == 2
        session_success.commit.assert_called_once()
        session_success.close.assert_called_once()

        session_failure = MagicMock()
        session_failure.execute.side_effect = RuntimeError("insert-failed")
        persistence_failure = _build_stub_persistence(
            backend=BackendType.POSTGRES,
            session=session_failure,
        )
        component_failure = LineageTrackerComponent(cast(Any, persistence_failure))

        with pytest.raises(RuntimeError, match="insert-failed"):
            component_failure.link_lineage(
                child_dataset_id="child",
                parent_ids=["p1"],
                transform_id="transform",
                ts_range={"start_ns": 1, "end_ns": 2},
                params={},
            )

        session_failure.rollback.assert_called_once()

    def test_link_lineage_postgres_raises_when_session_missing(self) -> None:
        from ml.registry.common.lineage_tracker import LineageTrackerComponent

        persistence = _build_stub_persistence(backend=BackendType.POSTGRES, session=None)
        component = LineageTrackerComponent(cast(Any, persistence))

        with pytest.raises(RuntimeError, match="Failed to get database session"):
            component.link_lineage(
                child_dataset_id="child",
                parent_ids=["p1"],
                transform_id="transform",
                ts_range={"start_ns": 1, "end_ns": 2},
                params={},
            )

    def test_iter_lineage_json_invalid_payload_and_postgres_query_paths(self) -> None:
        from ml.registry.common.lineage_tracker import LineageTrackerComponent

        json_persistence = _build_stub_persistence(backend=BackendType.JSON)
        json_persistence._lineage = [
            {
                "transform_id": "transform",
                "child_dataset_id": "child",
                "parent_dataset_id": "parent",
                "ts_range": "{invalid",
                "parameters": "{invalid",
                "created_at": 2.0,
            },
        ]
        json_component = LineageTrackerComponent(cast(Any, json_persistence))
        records = list(json_component.iter_lineage(child="child", parent="parent", limit=1))
        assert len(records) == 1
        assert records[0].ts_range == {}
        assert records[0].parameters == {}

        pg_session = MagicMock()
        pg_session.execute.return_value.fetchall.return_value = [
            {
                "transform_id": "transform",
                "child_dataset_id": "child",
                "parent_dataset_id": "parent",
                "ts_range": '{"start_ns": 1, "end_ns": 2}',
                "parameters": '{"window": 5}',
                "created_at": 12.0,
            },
        ]
        pg_persistence = _build_stub_persistence(backend=BackendType.POSTGRES, session=pg_session)
        pg_component = LineageTrackerComponent(cast(Any, pg_persistence))
        pg_records = list(pg_component.iter_lineage(child="child", parent="parent", limit=1))
        assert len(pg_records) == 1
        assert pg_records[0].transform_id == "transform"
        assert "LIMIT :limit" in str(pg_session.execute.call_args.args[0])
        assert pg_session.execute.call_args.args[1] == {
            "child": "child",
            "parent": "parent",
            "limit": 1,
        }
        pg_session.close.assert_called_once()

    def test_iter_lineage_postgres_raises_when_session_missing(self) -> None:
        from ml.registry.common.lineage_tracker import LineageTrackerComponent

        persistence = _build_stub_persistence(backend=BackendType.POSTGRES, session=None)
        component = LineageTrackerComponent(cast(Any, persistence))

        with pytest.raises(RuntimeError, match="Failed to get database session"):
            list(component.iter_lineage())

    def test_get_pipeline_signature_fallback_and_exception_paths(self) -> None:
        from ml.registry.common.lineage_tracker import LineageTrackerComponent

        persistence = _build_stub_persistence(backend=BackendType.JSON)
        manifest = DatasetManifest(
            dataset_id="dataset_sig",
            dataset_type=DatasetType.FEATURES,
            storage_kind=StorageKind.PARQUET,
            location="/tmp/features",
            partitioning={},
            retention_days=7,
            schema={"instrument_id": "str", "ts_event": "int64", "ts_init": "int64"},
            ts_field="ts_event",
            seq_field=None,
            primary_keys=["instrument_id", "ts_event"],
            schema_hash="",
            constraints={},
            lineage=[],
            pipeline_signature="from_manifest",
            version="1.0.0",
            metadata={"pipeline_signature": 123},
        )
        persistence._manifests["dataset_sig"] = manifest
        component = LineageTrackerComponent(cast(Any, persistence))
        assert component.get_pipeline_signature("dataset_sig") == "from_manifest"

        class _KeyErrorMap(dict[str, DatasetManifest]):
            def __contains__(self, key: object) -> bool:
                del key
                return True

            def __getitem__(self, key: str) -> DatasetManifest:
                del key
                raise KeyError("missing")

        persistence._manifests = _KeyErrorMap()
        assert component.get_pipeline_signature("missing") is None

        persistence._manifests = {"dataset_bad": cast(Any, object())}
        assert component.get_pipeline_signature("dataset_bad") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
