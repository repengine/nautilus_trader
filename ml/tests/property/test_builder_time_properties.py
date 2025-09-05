from __future__ import annotations

import polars as pl
import pytest


try:
    from hypothesis import HealthCheck
    from hypothesis import given
    from hypothesis import settings
    from hypothesis import strategies as st
except Exception:  # pragma: no cover
    pytest.skip("hypothesis not available", allow_module_level=True)

import ml.data.tft_dataset_builder as builder_mod
from ml.data.tft_dataset_builder import TFTDatasetBuilder
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


def _bars_df(ts: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "instrument_id": ["SPY.NYSE"] * len(ts),
            "timestamp": pl.Series(ts).cast(pl.Datetime("ns", "UTC")),
            "open": [100.0 + 0.01 * i for i in range(len(ts))],
            "high": [100.1 + 0.01 * i for i in range(len(ts))],
            "low": [99.9 + 0.01 * i for i in range(len(ts))],
            "close": [100.05 + 0.01 * i for i in range(len(ts))],
            "volume": [1000 + 10 * i for i in range(len(ts))],
        }
    )


@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    n=st.integers(min_value=5, max_value=30),
    step=st.integers(min_value=60_000_000_000, max_value=120_000_000_000),  # 1-2 minutes in ns
)
def test_builder_time_index_monotonic(monkeypatch, tmp_path, n: int, step: int) -> None:
    base = 1_600_000_000_000_000_000
    ts = [base + i * step for i in range(n)]

    def _fake_bars_to_dataframe(catalog, instrument_ids, start=None, end=None):  # type: ignore[no-redef]
        return _bars_df(ts)

    monkeypatch.setattr(builder_mod, "bars_to_dataframe", _fake_bars_to_dataframe)
    builder = TFTDatasetBuilder(ParquetDataCatalog(path=str(tmp_path)), symbols=["SPY"])
    df = builder.build_training_dataset(use_polars=True, lookback_periods=0, horizon_minutes=1)
    assert not df.is_empty()
    ti = df.get_column("time_index")
    # time_index must be monotonic increasing 0..n-1
    assert list(ti.to_list()) == list(range(len(ti)))
