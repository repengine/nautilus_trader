from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import polars as pl
from pytest import MonkeyPatch

from ml.data.tft_dataset_builder import TFTDatasetBuilder
from ml.tests.builders import DataBuilder
from ml.tests.utils.earnings_facade import build_test_data_store


def test_tft_builder_macro_and_micro(
    monkeypatch: MonkeyPatch,
    patch_bars_to_dataframe,
    tmp_path: Path,
) -> None:
    patch_bars_to_dataframe("ml.data.tft_dataset_builder")

    # Monkeypatch micro aggregator
    class _FakeAgg:
        def __init__(self, base_dir: Path) -> None:
            del base_dir

        def compute_for_symbol(self, symbol: str) -> pl.DataFrame:
            base2 = datetime(2025, 1, 1, 9, 30, tzinfo=UTC)
            base2_ns = int(base2.timestamp() * 1e9)
            ts2_ns = DataBuilder.time_series(
                n_points=5,
                start_time=base2_ns,
                interval_ns=60_000_000_000,
            )
            ts2 = [datetime.fromtimestamp(t / 1e9, tz=UTC) for t in ts2_ns]
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
    assert isinstance(df, pl.DataFrame)
    assert not df.is_empty()
    # Has target and known features
    assert "y" in df.columns
    assert "midprice" in df.columns  # micro feature joined
    # Dataset built successfully


def test_tft_builder_earnings_join(
    monkeypatch: MonkeyPatch,
    patch_bars_to_dataframe,
    tmp_path: Path,
    sample_bar_series_config_factory,
) -> None:
    patch_bars_to_dataframe(
        "ml.data.tft_dataset_builder",
        sample_bar_series_config_factory(instrument_id="AAPL", rows=8, freq_minutes=5),
    )

    store = build_test_data_store()

    def _to_ns(value: datetime) -> int:
        value_utc = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return int(value_utc.timestamp() * 1_000_000_000)

    quarters = [
        ("2023-06-30", "2023-07-15", 1.10, 1.05),
        ("2023-09-30", "2023-10-15", 1.25, 1.20),
        ("2023-12-31", "2024-01-20", 1.40, 1.30),
        ("2024-03-31", "2024-04-20", 1.55, 1.45),
    ]

    for period_end, filing_date, eps_actual, eps_consensus in quarters:
        filing_dt = datetime.fromisoformat(filing_date).replace(tzinfo=UTC)
        event_ns = _to_ns(filing_dt)
        store.write_earnings_actual(
            ticker="AAPL",
            period_end=period_end,
            filing_date=filing_date,
            eps_diluted=eps_actual,
            revenue=95_000_000_000,
            ts_event=event_ns,
            ts_init=event_ns,
        )
        estimate_dt = filing_dt - timedelta(days=30)
        store.write_earnings_estimate(
            ticker="AAPL",
            estimate_date=estimate_dt.date().isoformat(),
            period_end=period_end,
            eps_consensus=eps_consensus,
            ts_event=_to_ns(estimate_dt),
            ts_init=_to_ns(estimate_dt),
        )

    from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

    builder = TFTDatasetBuilder(
        ParquetDataCatalog(path=str(tmp_path)),
        symbols=["AAPL"],
        data_store=store,
        include_macro=False,
        include_micro=False,
        include_l2=False,
        include_earnings=True,
        earnings_lag_days=0,
    )

    dataset = builder.build_training_dataset(
        use_polars=True,
        lookback_periods=1,
        horizon_minutes=1,
    )

    assert isinstance(dataset, pl.DataFrame)
    assert not dataset.is_empty()
    earnings_cols = {
        "eps_surprise_q0_AAPL",
        "eps_surprise_pct_q0_AAPL",
        "eps_growth_yoy_AAPL",
        "earnings_beat_streak_AAPL",
        "is_earnings_available",
    }
    for column in earnings_cols:
        assert column in dataset.columns
    assert dataset.select(pl.col("is_earnings_available").sum())[0, 0] > 0
