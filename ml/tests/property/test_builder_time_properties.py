from __future__ import annotations

import polars as pl
import pytest
from pathlib import Path
from typing import Sequence, cast

from pytest import MonkeyPatch


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
    data: dict[str, list[object]] = {
        "instrument_id": ["SPY.NYSE"] * len(ts),
        "timestamp": list(ts),
        "open": [100.0 + 0.01 * i for i in range(len(ts))],
        "high": [100.1 + 0.01 * i for i in range(len(ts))],
        "low": [99.9 + 0.01 * i for i in range(len(ts))],
        "close": [100.05 + 0.01 * i for i in range(len(ts))],
        "volume": [1000 + 10 * i for i in range(len(ts))],
    }
    return pl.DataFrame(
        data,
        schema={
            "instrument_id": pl.Utf8,
            "timestamp": pl.Datetime("ns", "UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
        },
    )


@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    n=st.integers(min_value=5, max_value=30),
    step=st.integers(min_value=60_000_000_000, max_value=120_000_000_000),  # 1-2 minutes in ns
)
def test_builder_time_index_monotonic(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    n: int,
    step: int,
) -> None:
    base = 1_600_000_000_000_000_000
    ts = [base + i * step for i in range(n)]

    def _fake_bars_to_dataframe(
        catalog: ParquetDataCatalog,
        instrument_ids: Sequence[str],
        start: int | None = None,
        end: int | None = None,
    ) -> pl.DataFrame:
        del catalog, instrument_ids, start, end
        return _bars_df(ts)

    monkeypatch.setattr(builder_mod, "bars_to_dataframe", _fake_bars_to_dataframe)
    builder = TFTDatasetBuilder(ParquetDataCatalog(path=str(tmp_path)), symbols=["SPY"])
    df_raw = builder.build_training_dataset(
        use_polars=True,
        lookback_periods=0,
        horizon_minutes=1,
    )
    df = cast(pl.DataFrame, df_raw)
    assert not df.is_empty()
    ti: pl.Series = df.get_column("time_index")
    # time_index must be monotonic increasing 0..n-1
    assert list(ti.to_list()) == list(range(len(ti)))
