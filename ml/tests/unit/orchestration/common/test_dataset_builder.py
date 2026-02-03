from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ml.data import DatasetMetadataExpectations
from ml.data.validation import DatasetValidationConfig
from ml.data.vintage import VintagePolicy
from ml.orchestration.config_types import DatasetBuildConfig
from ml.orchestration.dataset_builder import DatasetBuilder
from ml.tests.utils.targets import build_default_target_semantics_payload


@pytest.fixture
def dataset_cfg(tmp_path: Path) -> DatasetBuildConfig:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    return DatasetBuildConfig(
        data_dir=str(data_dir),
        symbols="SPY",
        out_dir=str(out_dir),
        dataset_id="test_dataset",
        target_semantics=build_default_target_semantics_payload(),
    )


@pytest.fixture
def dataset_builder() -> DatasetBuilder:
    return DatasetBuilder(build_main=Mock(return_value=0))


def test_build_dataset_rejects_non_config(dataset_builder: DatasetBuilder) -> None:
    with pytest.raises(TypeError):
        dataset_builder.build_dataset(object())  # type: ignore[arg-type]


def test_build_dataset_falls_back_to_cli_when_api_fails(
    dataset_builder: DatasetBuilder,
    dataset_cfg: DatasetBuildConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_builder.build_main.return_value = 3  # type: ignore[assignment]
    monkeypatch.setattr("ml.data.build_tft_dataset", Mock(side_effect=RuntimeError("api failure")))

    rc = dataset_builder.build_dataset(dataset_cfg)

    assert rc == 3
    dataset_builder.build_main.assert_called_once()


def test_build_dataset_records_artifacts_from_api_success(
    dataset_builder: DatasetBuilder,
    dataset_cfg: DatasetBuildConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api_result = SimpleNamespace(
        dataset_parquet=tmp_path / "dataset.parquet",
        dataset_csv=tmp_path / "dataset.csv",
        features_npz=tmp_path / "features.npz",
        feature_names=["f1", "f2"],
        feature_set_id="fs-123",
        metadata=None,
    )
    monkeypatch.setattr("ml.data.build_tft_dataset", Mock(return_value=api_result))

    rc = dataset_builder.build_dataset(dataset_cfg)

    assert rc == 0
    dataset_builder.build_main.assert_not_called()
    artifacts = dataset_builder.build_artifacts
    assert artifacts is not None
    assert artifacts.out_dir == Path(dataset_cfg.out_dir)
    assert artifacts.feature_set_id == "fs-123"
    assert artifacts.feature_names == ("f1", "f2")


def test_validate_dataset_returns_false_without_metadata(tmp_path: Path) -> None:
    builder = DatasetBuilder()
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.touch()

    passed, metadata = builder.validate_dataset(
        dataset_path,
        DatasetMetadataExpectations(),
        DatasetValidationConfig(),
    )

    assert passed is False
    assert metadata is None


def test_validate_dataset_passes_with_matching_metadata(tmp_path: Path) -> None:
    builder = DatasetBuilder()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / "dataset.parquet"
    dataset_path.touch()

    ts_start = "2024-01-01T00:00:00+00:00"
    ts_end = "2024-01-02T00:00:00+00:00"
    metadata_payload = {
        "dataset_id": "test_dataset",
        "vintage_policy": VintagePolicy.REAL_TIME.value,
        "vintage_cutoff": None,
        "build_ts": "2024-01-03T00:00:00+00:00",
        "ts_event_start": ts_start,
        "ts_event_end": ts_end,
        "overall_window": [ts_start, ts_end],
        "train_window": [ts_start, ts_end],
        "validation_window": None,
        "test_window": None,
        "macro_observation_counts": {"CPI": 2},
        "market_bindings": [],
    }
    (dataset_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata_payload, indent=2),
        encoding="utf-8",
    )

    expectations = DatasetMetadataExpectations(
        dataset_id="test_dataset",
        vintage_policy=VintagePolicy.REAL_TIME,
        vintage_cutoff=None,
        ts_event_start=ts_start,
        ts_event_end=ts_end,
    )
    validation_cfg = DatasetValidationConfig(
        min_rows=1,
        expected_vintage_policy=VintagePolicy.REAL_TIME,
        require_macro_series=("CPI",),
        macro_min_vintage_observations=1,
    )

    passed, metadata = builder.validate_dataset(
        dataset_path,
        expectations,
        validation_cfg,
    )

    assert passed is True
    assert metadata is not None
    assert metadata.dataset_id == "test_dataset"
    assert metadata.vintage_policy is VintagePolicy.REAL_TIME
