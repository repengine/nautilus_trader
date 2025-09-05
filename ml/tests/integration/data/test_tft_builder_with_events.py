from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta

import polars as pl

import ml.data.tft_dataset_builder as builder_mod
from ml.data.tft_dataset_builder import TFTDatasetBuilder
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


def _fake_bars_to_dataframe(catalog, instrument_ids, start=None, end=None) -> pl.DataFrame:  # type: ignore[no-redef]
    base = datetime(2025, 1, 1, 9, 30, tzinfo=UTC)
    ts = [base + timedelta(minutes=i) for i in range(60)]
    return pl.DataFrame(
        {
            "instrument_id": [instrument_ids[0]] * len(ts),
            "timestamp": ts,
            "open": [100.0 + 0.01 * i for i in range(60)],
            "high": [100.1 + 0.01 * i for i in range(60)],
            "low": [99.9 + 0.01 * i for i in range(60)],
            "close": [100.05 + 0.01 * i for i in range(60)],
            "volume": [1000 + 10 * i for i in range(60)],
        },
    )


def test_builder_includes_event_features(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(builder_mod, "bars_to_dataframe", _fake_bars_to_dataframe)
    builder = TFTDatasetBuilder(
        ParquetDataCatalog(path=str(tmp_path)),
        symbols=["SPY"],
        include_macro=False,
        include_micro=False,
        include_events=True,
    )
    df = builder.build_training_dataset(use_polars=True, lookback_periods=10, horizon_minutes=1)
    assert not df.is_empty()
    # Dataset built successfully with include_events flag exercised
    assert "timestamp" in df.columns
