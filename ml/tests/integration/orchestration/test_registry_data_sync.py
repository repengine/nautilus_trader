"""Integration-style tests for RegistrySynchronizer registry updates."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from ml.data import DatasetMetadata
from ml.data.vintage import VintagePolicy
from ml.orchestration.config_types import DatasetBuildConfig
from ml.orchestration.registry_synchronizer import RegistrySynchronizer
from ml.tests.utils.targets import build_default_target_semantics_payload


def _make_metadata() -> DatasetMetadata:
    return DatasetMetadata(
        dataset_id="spy_2024_ohlcv",
        vintage_policy=VintagePolicy.REAL_TIME,
        vintage_cutoff=None,
        build_ts="2024-10-21T00:00:00Z",
        ts_event_start="2024-01-01T00:00:00Z",
        ts_event_end="2024-12-31T00:00:00Z",
        overall_window=None,
        train_window=None,
        validation_window=None,
        test_window=None,
        macro_observation_counts={},
        market_bindings=(),
    )


@pytest.mark.integration
def test_synchronize_dataset_manifest_updates_registry(tmp_path: Path) -> None:
    """synchronize_dataset_manifest should update registry metadata."""
    data_registry = Mock()
    data_registry.get_manifest.return_value = Mock(metadata={})
    synchronizer = RegistrySynchronizer(data_registry=data_registry)
    cfg = DatasetBuildConfig(
        data_dir=str(tmp_path / "data"),
        out_dir=str(tmp_path / "out"),
        dataset_id="spy_2024_ohlcv",
        symbols="SPY",
        target_semantics=build_default_target_semantics_payload(),
    )
    metadata = _make_metadata()

    synchronizer.synchronize_dataset_manifest(cfg=cfg, metadata=metadata)

    data_registry.update_manifest.assert_called_once()


@pytest.mark.integration
def test_guard_dataset_metadata_enforces_macro_requirements(tmp_path: Path) -> None:
    """_guard_dataset_metadata should reject missing macro observations when required."""
    synchronizer = RegistrySynchronizer(data_registry=None)
    cfg = DatasetBuildConfig(
        data_dir=str(tmp_path / "data"),
        out_dir=str(tmp_path / "out"),
        dataset_id="spy_2024_ohlcv",
        symbols="SPY",
        include_macro=True,
        macro_series_ids=("gdp",),
        target_semantics=build_default_target_semantics_payload(),
    )
    metadata = _make_metadata()

    with pytest.raises(ValueError):
        synchronizer._guard_dataset_metadata(cfg=cfg, metadata=metadata)
