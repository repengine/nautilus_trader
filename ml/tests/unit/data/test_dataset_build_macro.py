"""Dataset build macro refresh tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import polars as pl
import pytest

from ml.data import DatasetBuildConfig
from ml.data import DatasetValidationConfig
from ml.data import build_tft_dataset
from ml.data.validation import DatasetValidationError
from ml.data.ingest.macro_refresh import MacroRefreshResult
from ml.data.vintage import VintagePolicy


def test_build_tft_dataset_invokes_macro_refresh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded: dict[str, object] = {}

    def _fake_macro(**kwargs: Any) -> MacroRefreshResult:
        recorded.update(kwargs)
        fred_path = cast(Path, kwargs["fred_path"])
        vintage_dir = cast(Path, kwargs["vintage_dir"])
        return MacroRefreshResult(
            fred_refreshed=False,
            alfred_refreshed=False,
            fred_path=fred_path,
            alfred_base_dir=vintage_dir,
        )

    class _CatalogStub:
        def __init__(self, path: str) -> None:
            self.path = path

    class _BuilderStub:
        def __init__(self, **kwargs: Any) -> None:
            recorded["builder_params"] = kwargs

        def build_training_dataset(self, **kwargs: Any) -> pl.DataFrame:
            return pl.DataFrame(
                {
                    "time_index": [0, 1],
                    "ts_event": [1_000_000_000, 2_000_000_000],
                    "feature_a": [1.0, 2.0],
                    "y": [0.0, 1.0],
                },
            )

    monkeypatch.setattr(
        "ml.data.ingest.macro_refresh.ensure_macro_ready",
        _fake_macro,
    )
    monkeypatch.setattr(
        "nautilus_trader.persistence.catalog.parquet.ParquetDataCatalog",
        _CatalogStub,
    )
    monkeypatch.setattr("ml.data.TFTDatasetBuilder", _BuilderStub)

    cfg = DatasetBuildConfig(
        data_dir=tmp_path,
        out_dir=tmp_path / "out",
        symbols=["SPY"],
        include_macro=True,
        macro_series_ids=("DGS10",),
        macro_fred_path=tmp_path / "macro" / "fred.parquet",
        fred_vintage_dir=tmp_path / "macro" / "vintages",
        validation=DatasetValidationConfig(
            min_rows=1,
            min_positive_rate=None,
            max_positive_rate=None,
            min_feature_coverage=0.0,
            require_macro_series=(),
            macro_min_vintage_observations=None,
        ),
    )

    result = build_tft_dataset(cfg)

    assert recorded["fred_path"] == cfg.macro_fred_path
    builder_params = recorded["builder_params"]
    assert isinstance(builder_params, dict)
    assert builder_params["fred_path"] == str(cfg.macro_fred_path)
    assert builder_params["macro_series_ids"] == cfg.macro_series_ids
    assert result.dataset_parquet.exists()
    assert result.dataset_csv.exists()
    assert result.features_npz.exists()
    assert result.metadata is not None
    assert result.metadata.vintage_policy is VintagePolicy.REAL_TIME
    metadata_path = cfg.out_dir / "dataset_metadata.json"
    assert metadata_path.exists()
    metadata_raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata_raw["vintage_policy"] == VintagePolicy.REAL_TIME.value
    assert metadata_raw["dataset_id"] == cfg.dataset_id
    assert metadata_raw["train_window"] is not None
    assert metadata_raw["ts_event_start"] is not None
    assert metadata_raw["ts_event_end"] is not None
    assert builder_params["vintage_policy"] is VintagePolicy.REAL_TIME
    assert builder_params["vintage_as_of"] is None


def test_build_tft_dataset_rejects_missing_macro_observations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fake_macro(**kwargs: Any) -> MacroRefreshResult:
        fred_path = cast(Path, kwargs["fred_path"])
        vintage_dir = cast(Path, kwargs["vintage_dir"])
        return MacroRefreshResult(
            fred_refreshed=False,
            alfred_refreshed=False,
            fred_path=fred_path,
            alfred_base_dir=vintage_dir,
        )

    class _CatalogStub:
        def __init__(self, path: str) -> None:
            self.path = path

    class _BuilderStub:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        def build_training_dataset(self, **kwargs: Any) -> pl.DataFrame:
            del kwargs
            return pl.DataFrame(
                {
                    "time_index": [0, 1],
                    "ts_event": [1_000_000_000, 2_000_000_000],
                    "DGS10": [0.0, 0.0],
                    "DGS10__value_vintage_ts": [None, None],
                    "y": [0.0, 1.0],
                },
            )

    monkeypatch.setattr("ml.data.ingest.macro_refresh.ensure_macro_ready", _fake_macro)
    monkeypatch.setattr(
        "nautilus_trader.persistence.catalog.parquet.ParquetDataCatalog",
        _CatalogStub,
    )
    monkeypatch.setattr("ml.data.TFTDatasetBuilder", _BuilderStub)

    cfg = DatasetBuildConfig(
        data_dir=tmp_path,
        out_dir=tmp_path / "out",
        symbols=["SPY"],
        include_macro=True,
        macro_series_ids=("DGS10",),
        macro_fred_path=tmp_path / "macro" / "fred.parquet",
        fred_vintage_dir=tmp_path / "macro" / "vintages",
    )

    with pytest.raises(DatasetValidationError):
        build_tft_dataset(cfg)
