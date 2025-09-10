from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import polars as pl

import ml.data.tft_dataset_builder as builder_mod
from ml.data.tft_dataset_builder import TFTDatasetBuilder


def _fake_bars_to_dataframe(catalog, instrument_ids, start=None, end=None) -> pl.DataFrame:  # type: ignore[no-redef]
    # Produce 5 minutes of bars
    base = datetime(2025, 1, 1, 9, 30, tzinfo=UTC)
    ts = [base + timedelta(minutes=i) for i in range(5)]
    return pl.DataFrame(
        {
            "instrument_id": [instrument_ids[0]] * len(ts),
            "timestamp": ts,
            "open": [100.0, 100.1, 100.2, 100.3, 100.4],
            "high": [100.2, 100.2, 100.3, 100.4, 100.5],
            "low": [99.9, 100.0, 100.1, 100.2, 100.3],
            "close": [100.1, 100.15, 100.25, 100.35, 100.45],
            "volume": [1000, 1100, 1200, 1300, 1400],
        },
    )


def test_tft_builder_macro_and_micro(monkeypatch, tmp_path) -> None:
    # Monkeypatch bars loader
    monkeypatch.setattr(builder_mod, "bars_to_dataframe", _fake_bars_to_dataframe)

    # Monkeypatch micro aggregator
    class _FakeAgg:
        def __init__(self, base_dir: Path) -> None:
            pass

        def compute_for_symbol(self, symbol: str) -> pl.DataFrame:
            base2 = datetime(2025, 1, 1, 9, 30, tzinfo=UTC)
            ts2 = [base2 + timedelta(minutes=i) for i in range(5)]
            return pl.DataFrame(
                {"timestamp": ts2, "midprice": [100.05, 100.1, 100.2, 100.3, 100.4]},
            )

    import ml.features.micro_aggregate as micro_mod

    monkeypatch.setattr(micro_mod, "MicrostructureAggregator", _FakeAgg)

    # Macro path is exercised elsewhere; keep this test focused on micro + core

    from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

    builder = TFTDatasetBuilder(
        ParquetDataCatalog(path=str(tmp_path)),
        symbols=["SPY"],
        include_macro=False,
        include_micro=True,
        micro_base_dir=str(tmp_path),
    )
    df = builder.build_training_dataset(use_polars=True, lookback_periods=2, horizon_minutes=1)
    assert not df.is_empty()
    # Has target and known features
    assert "y" in df.columns
    assert "midprice" in df.columns  # micro feature joined
    # Dataset built successfully
