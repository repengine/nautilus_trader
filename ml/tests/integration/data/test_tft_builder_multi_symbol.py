"""Integration coverage for multi-symbol TFT dataset builds."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from ml.data.tft_dataset_builder import TFTDatasetBuilder
from ml.tests.utils.targets import build_default_target_semantics
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


@pytest.mark.integration
def test_build_training_dataset_when_multiple_symbols_includes_all_instruments(
    patch_bars_to_dataframe,
    sample_bar_series_config_factory,
    tmp_path: Path,
) -> None:
    base_start = datetime(2025, 1, 1, 9, 30, tzinfo=UTC)
    rows = 4

    config = sample_bar_series_config_factory(
        instrument_id="SPY",
        rows=rows,
        start=base_start,
    )
    patch_bars_to_dataframe("ml.data.tft_dataset_builder", config)

    builder = TFTDatasetBuilder(
        ParquetDataCatalog(path=str(tmp_path)),
        symbols=["SPY", "QQQ"],
        include_macro=False,
        include_micro=False,
    )
    target_semantics = build_default_target_semantics(
        horizon_minutes=1,
        threshold=0.001,
        legacy_aliases=True,
    )
    df = builder.build_training_dataset(
        target_semantics=target_semantics,
        use_polars=True,
        lookback_periods=1,
    )

    assert isinstance(df, pl.DataFrame)
    instruments = set(df.get_column("instrument_id").unique().to_list())
    assert instruments == {"SPY", "QQQ"}
    assert df.height > 0
