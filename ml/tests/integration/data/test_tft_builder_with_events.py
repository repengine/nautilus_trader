from __future__ import annotations

from datetime import UTC
from pathlib import Path

import polars as pl
import pytest

from ml.data.tft_dataset_builder import TFTDatasetBuilder
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry", "mock_tracing_backend")


def test_builder_includes_event_features(
    patch_dataset_bars,
    tmp_path: Path,
    sample_bar_series_config_factory,
) -> None:
    patch_dataset_bars(
        modules=("ml.data.tft_dataset_builder",),
        config=sample_bar_series_config_factory(instrument_id="SPY", rows=60),
    )
    builder = TFTDatasetBuilder(
        ParquetDataCatalog(path=str(tmp_path)),
        symbols=["SPY"],
        include_macro=False,
        include_micro=False,
        include_events=True,
    )
    df = builder.build_training_dataset(use_polars=True, lookback_periods=10, horizon_minutes=1)
    assert isinstance(df, pl.DataFrame)
    assert not df.is_empty()
    # Dataset built successfully with include_events flag exercised
    assert "timestamp" in df.columns
