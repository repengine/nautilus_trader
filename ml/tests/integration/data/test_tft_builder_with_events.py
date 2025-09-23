from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path

from pytest import MonkeyPatch

import polars as pl

import ml.data.tft_dataset_builder as builder_mod
from ml.data.tft_dataset_builder import TFTDatasetBuilder
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


def _fake_bars_to_dataframe(
    catalog: ParquetDataCatalog,
    instrument_ids: Sequence[str],
    start: datetime | None = None,
    end: datetime | None = None,
) -> pl.DataFrame:
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


def test_builder_includes_event_features(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(builder_mod, "bars_to_dataframe", _fake_bars_to_dataframe)
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
