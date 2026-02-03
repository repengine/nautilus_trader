from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.cli import build_tft_dataset as cli
from ml.stores.feature_raw_writer import FeatureDatasetParquetRawWriter
from ml.tests.utils.targets import build_default_target_semantics_payload


def test_build_tft_dataset_cli_initializes_data_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    dataset_paths = {
        "parquet": tmp_path / "dataset.parquet",
        "csv": tmp_path / "dataset.csv",
        "npz": tmp_path / "features.npz",
    }

    class _StubDataStore:
        def __init__(self, *, connection_string: str, raw_writer: object, **_: object) -> None:
            captured["connection_string"] = connection_string
            captured["raw_writer"] = raw_writer

    def _fake_build(cfg: object, *, data_store: object) -> object:
        captured["config"] = cfg
        captured["data_store"] = data_store

        class _Result:
            dataset_parquet = dataset_paths["parquet"]
            dataset_csv = dataset_paths["csv"]
            features_npz = dataset_paths["npz"]
            feature_names: list[str] = []
            feature_set_id: str | None = None
            metadata = None

        return _Result()

    monkeypatch.setenv("DB_CONNECTION", "postgresql://stub")
    monkeypatch.setattr(cli, "DataStore", _StubDataStore)
    monkeypatch.setattr(cli, "build_tft_dataset", _fake_build)

    target_semantics = build_default_target_semantics_payload()
    args = [
        "--symbols",
        "AAPL",
        "--out_dir",
        str(tmp_path / "out"),
        "--target-semantics",
        json.dumps(target_semantics),
        "--events_dir",
        str(tmp_path / "events.parquet"),
        "--micro-base-dir",
        str(tmp_path / "micro"),
        "--l2-base-dir",
        str(tmp_path / "l2"),
    ]

    exit_code = cli.main(args)

    assert exit_code == 0
    assert captured["connection_string"] == "postgresql://stub"
    assert isinstance(captured["raw_writer"], FeatureDatasetParquetRawWriter)
    assert isinstance(captured["data_store"], _StubDataStore)
    assert hasattr(captured["config"], "symbols")
