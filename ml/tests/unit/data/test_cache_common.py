from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from ml.data.cache_common import day_partition_path, filter_df_by_ns_range, iter_days


def test_iter_days_half_open_inclusive_start_exclusive_end() -> None:
    start = datetime(2025, 1, 31, 23, 0, tzinfo=UTC)
    end = datetime(2025, 2, 2, 0, 0, tzinfo=UTC)
    days = list(iter_days(start, end))
    assert days == [date(2025, 1, 31), date(2025, 2, 1)]


def test_iter_days_includes_end_day_when_time_not_midnight() -> None:
    start = datetime(2025, 2, 1, 0, 0, tzinfo=UTC)
    end = datetime(2025, 2, 2, 12, 0, tzinfo=UTC)
    days = list(iter_days(start, end))
    assert days == [date(2025, 2, 1), date(2025, 2, 2)]


def test_day_partition_path_layout(tmp_path: Path) -> None:
    base = tmp_path
    p = day_partition_path(base, "SPY", date(2025, 8, 11))
    assert str(p).endswith("SPY/year=2025/month=08/day=11.parquet")


@pytest.mark.parametrize("use_int_ts", [True, False])
def test_filter_df_by_ns_range_half_open(use_int_ts: bool) -> None:
    # Build a simple frame with three timestamps
    ts = [
        datetime(2025, 8, 11, 10, 0, tzinfo=UTC),
        datetime(2025, 8, 11, 10, 1, tzinfo=UTC),
        datetime(2025, 8, 11, 10, 2, tzinfo=UTC),
    ]
    values = [1, 2, 3]

    if use_int_ts:
        df = pl.DataFrame(
            {
                "timestamp": [int(t.timestamp() * 1_000_000_000) for t in ts],
                "v": values,
            },
        )
    else:
        df = pl.DataFrame(
            {
                "timestamp": ts,
                "v": values,
            },
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("ns", "UTC")))

    start = ts[0]
    end = ts[2]  # exclude the last element in half-open

    out = filter_df_by_ns_range(df, start=start, end=end)
    assert out.shape == (2, 2)
    # Confirm monotone and boundaries
    out_ts = out["timestamp"].to_list()
    assert out_ts[0] == start
    assert out_ts[-1] == ts[1]
