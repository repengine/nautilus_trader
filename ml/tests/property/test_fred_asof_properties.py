from __future__ import annotations

import polars as pl
import pytest
from polars import Series
from typing import cast


try:
    from hypothesis import given
    from hypothesis import settings
    from hypothesis import strategies as st
except Exception:  # pragma: no cover
    pytest.skip("hypothesis not available", allow_module_level=True)

from ml.data.fred_join import join_fred_asof


def _timeseries(start: int, n: int, step: int) -> list[int]:
    return [start + i * step for i in range(n)]


@settings(max_examples=25)
@given(
    n_releases=st.integers(min_value=1, max_value=5),
    n_rows=st.integers(min_value=3, max_value=20),
    step=st.integers(min_value=60_000_000_000, max_value=300_000_000_000),  # 1-5 minutes in ns
)
def test_fred_lag_increases_nulls(n_releases: int, n_rows: int, step: int) -> None:
    # Build synthetic FRED (single series) and left timestamps
    base_ns = 1_600_000_000_000_000_000
    fred_ts = _timeseries(base_ns, n_releases, step * 5)
    fred_data: dict[str, list[object]] = {
        "timestamp": list(fred_ts),
        "series_id": ["S"] * n_releases,
        "value": list(range(n_releases)),
    }
    fred = pl.DataFrame(
        fred_data,
        schema={
            "timestamp": pl.Datetime("ns"),
            "series_id": pl.Utf8,
            "value": pl.Int64,
        },
    )
    left_ts = _timeseries(base_ns, n_rows, step)
    left_data: dict[str, list[object]] = {
        "timestamp": list(left_ts),
        "x": list(range(n_rows)),
    }
    left = pl.DataFrame(
        left_data,
        schema={
            "timestamp": pl.Datetime("ns"),
            "x": pl.Int64,
        },
    )

    out0 = cast(pl.DataFrame, join_fred_asof(left, timestamp_col="timestamp", lag_days=0))
    out1 = cast(pl.DataFrame, join_fred_asof(left, timestamp_col="timestamp", lag_days=1))
    if "S" in out0.columns:
        series0: Series = out0.get_column("S")
        s0_values = series0.to_list()
    else:
        s0_values = [None] * n_rows
    if "S" in out1.columns:
        series1: Series = out1.get_column("S")
        s1_values = series1.to_list()
    else:
        s1_values = [None] * n_rows
    # Nulls should not decrease with added lag
    nulls0 = sum(1 for x in s0_values if x is None)
    nulls1 = sum(1 for x in s1_values if x is None)
    assert nulls1 >= nulls0
    # Where both present, values should match
    for i in range(n_rows):
        v0 = s0_values[i]
        v1 = s1_values[i]
        if v0 is not None and v1 is not None:
            assert v0 == v1
