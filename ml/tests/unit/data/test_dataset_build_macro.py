"""Dataset build macro refresh tests."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from ml.data import DatasetBuildConfig
from ml.data import DatasetValidationConfig
from ml.data import build_tft_dataset
from ml.data.ingest.macro_refresh import MacroRefreshResult


def test_build_tft_dataset_invokes_macro_refresh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded: dict[str, object] = {}

    def _fake_macro(**kwargs: object) -> MacroRefreshResult:
        recorded.update(kwargs)
        return MacroRefreshResult(
            fred_refreshed=False,
            alfred_refreshed=False,
            fred_path=kwargs["fred_path"],
            alfred_base_dir=kwargs["vintage_dir"],
        )

    class _CatalogStub:
        def __init__(self, path: str) -> None:
            self.path = path

    class _BuilderStub:
        def __init__(self, **kwargs: object) -> None:
            recorded["builder_params"] = kwargs

        def build_training_dataset(self, **kwargs: object) -> pl.DataFrame:
            return pl.DataFrame(
                {
                    "time_index": [0, 1],
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
