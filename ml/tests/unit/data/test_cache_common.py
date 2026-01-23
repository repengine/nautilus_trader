from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from ml.data.cache_common import day_partition_path
from ml.data.cache_common import filter_df_by_ns_range
from ml.data.cache_common import iter_days
from ml.data.cache_common import resolve_cache_partition_path
from ml.data.cache_common import resolve_cache_read_symbol_dirs


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


def test_resolve_cache_read_symbol_dirs_when_full_and_base_exist_returns_full_then_base(
    tmp_path: Path,
) -> None:
    (tmp_path / "SPY").mkdir()
    (tmp_path / "SPY.XNAS").mkdir()

    resolved = resolve_cache_read_symbol_dirs(tmp_path, "spy.xnas")

    assert resolved[:2] == ("SPY.XNAS", "SPY")


def test_resolve_cache_partition_path_when_full_exists_prefers_full(tmp_path: Path) -> None:
    day = date(2024, 1, 2)
    base = day_partition_path(tmp_path, "SPY", day)
    full = day_partition_path(tmp_path, "SPY.XNAS", day)
    base.parent.mkdir(parents=True, exist_ok=True)
    full.parent.mkdir(parents=True, exist_ok=True)
    base.touch()
    full.touch()

    resolved, is_write = resolve_cache_partition_path(tmp_path, "SPY.XNAS", day)

    assert resolved == full
    assert is_write is True


def test_resolve_cache_partition_path_when_only_base_exists_returns_base(tmp_path: Path) -> None:
    day = date(2024, 1, 3)
    base = day_partition_path(tmp_path, "SPY", day)
    base.parent.mkdir(parents=True, exist_ok=True)
    base.touch()

    resolved, is_write = resolve_cache_partition_path(tmp_path, "SPY.XNAS", day)

    assert resolved == base
    assert is_write is False
