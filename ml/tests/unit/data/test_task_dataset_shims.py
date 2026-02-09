from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ml._imports import pl
import ml.cli.build_tft_dataset as tft_cli
from ml.data.validation import DatasetReportConfig
from ml.data.validation import generate_dataset_report
from ml.tests.utils.targets import build_default_target_semantics_payload


def test_tft_cli_main_when_injected_dependencies_builds_task_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _StubDataStore:
        def __init__(self, *, connection_string: str, raw_writer: object, **_: object) -> None:
            captured["connection_string"] = connection_string
            captured["raw_writer"] = raw_writer

    def _fake_build(cfg: object, *, data_store: object) -> object:
        captured["config"] = cfg
        captured["data_store"] = data_store

        class _Result:
            dataset_parquet = tmp_path / "dataset.parquet"
            dataset_csv = tmp_path / "dataset.csv"
            features_npz = tmp_path / "features.npz"
            feature_names: list[str] = []
            feature_set_id: str | None = None
            metadata = None

        return _Result()

    monkeypatch.setenv("DB_CONNECTION", "postgresql://stub")
    args = [
        "--symbols",
        "AAPL",
        "--out_dir",
        str(tmp_path / "out"),
        "--target-semantics",
        json.dumps(build_default_target_semantics_payload()),
        "--skip_csv",
    ]

    rc = tft_cli._main_with_dependencies(
        args,
        build_fn=_fake_build,
        data_store_cls=_StubDataStore,
    )

    assert rc == 0
    assert captured["connection_string"] == "postgresql://stub"
    assert isinstance(captured["data_store"], _StubDataStore)
    assert getattr(captured["config"], "write_csv") is False


def test_generate_dataset_report_reads_parquet_shape(tmp_path: Path) -> None:
    if pl is None:
        pytest.skip("polars not installed")

    dataset_path = tmp_path / "dataset.parquet"
    frame = pl.DataFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-02"],
            "instrument_id": ["AAPL", "AAPL"],
            "y": [1, 0],
            "FEDFUNDS": [5.0, None],
            "feature_x": [0.1, 0.2],
        },
    )
    frame.write_parquet(dataset_path)

    report = generate_dataset_report(
        DatasetReportConfig(dataset_path=dataset_path),
    )

    assert report.data["shape"] == [2, 5]
    assert "FEDFUNDS" in report.data["macro_null_rates"]
    assert report.data["target"]["overall"]["total"] == 2
