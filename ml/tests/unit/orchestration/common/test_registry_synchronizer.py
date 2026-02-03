"""RegistrySynchronizer component tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from ml.data import DatasetMetadata
from ml.data.vintage import VintagePolicy
from ml.orchestration.config_types import DatasetBuildConfig
from ml.orchestration.registry_synchronizer import RegistrySynchronizer
from ml.tests.utils.targets import build_default_target_semantics_payload


@pytest.fixture
def data_registry() -> Mock:
    """Provide a mock DataRegistry."""
    registry = Mock()
    registry.get_manifest.return_value = Mock(metadata={})
    return registry


@pytest.fixture
def registry_synchronizer(data_registry: Mock) -> RegistrySynchronizer:
    """Create RegistrySynchronizer instance for testing."""
    return RegistrySynchronizer(
        data_registry=data_registry,
        feature_registry=None,
        model_registry=None,
        message_bus=None,
    )


@pytest.fixture
def dataset_cfg(tmp_path: Path) -> DatasetBuildConfig:
    """Construct a minimal DatasetBuildConfig for registry sync operations."""
    return DatasetBuildConfig(
        data_dir=str(tmp_path / "data"),
        out_dir=str(tmp_path / "out"),
        dataset_id="test.dataset",
        symbols="SPY",
        target_semantics=build_default_target_semantics_payload(),
    )


@pytest.fixture
def dataset_metadata() -> DatasetMetadata:
    """Construct DatasetMetadata with required fields."""
    return DatasetMetadata(
        dataset_id="test.dataset",
        vintage_policy=VintagePolicy.REAL_TIME,
        vintage_cutoff=None,
        build_ts="2024-01-01T00:00:00Z",
        ts_event_start=None,
        ts_event_end=None,
        overall_window=None,
        train_window=None,
        validation_window=None,
        test_window=None,
        macro_observation_counts={},
        market_bindings=(),
    )


def test_registry_synchronizer_initializes_with_registries(
    data_registry: Mock,
) -> None:
    """Ensure registries are retained on initialization."""
    synchronizer = RegistrySynchronizer(
        data_registry=data_registry,
        feature_registry=None,
        model_registry=None,
        message_bus=None,
    )

    assert synchronizer.data_registry is data_registry


def test_record_build_artifacts_sets_build_artifacts(
    registry_synchronizer: RegistrySynchronizer,
    dataset_cfg: DatasetBuildConfig,
) -> None:
    """_record_build_artifacts should store artifacts for later access."""
    registry_synchronizer._record_build_artifacts(
        cfg=dataset_cfg,
        feature_set_id="fs1",
        feature_names=("a", "b"),
        feature_registry_dir="/tmp/features",
        dataset_metadata=None,
    )

    artifacts = registry_synchronizer.build_artifacts
    assert artifacts is not None
    assert artifacts.feature_set_id == "fs1"
    assert artifacts.feature_names == ("a", "b")


def test_guard_dataset_metadata_accepts_valid_metadata(
    registry_synchronizer: RegistrySynchronizer,
    dataset_cfg: DatasetBuildConfig,
    dataset_metadata: DatasetMetadata,
) -> None:
    """_guard_dataset_metadata should not raise for valid metadata."""
    registry_synchronizer._guard_dataset_metadata(cfg=dataset_cfg, metadata=dataset_metadata)


def test_synchronize_dataset_manifest_updates_registry(
    registry_synchronizer: RegistrySynchronizer,
    data_registry: Mock,
    dataset_cfg: DatasetBuildConfig,
    dataset_metadata: DatasetMetadata,
) -> None:
    """synchronize_dataset_manifest should call registry.update_manifest."""
    registry_synchronizer.synchronize_dataset_manifest(cfg=dataset_cfg, metadata=dataset_metadata)

    assert data_registry.update_manifest.called
