from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import polars as pl

from ml.data.fred_join import join_fred_asof
from ml.data.vintage import VintagePolicy


def _write_alfred_release(base_dir: Path) -> None:
    series_dir = base_dir / "TEST"
    series_dir.mkdir(parents=True, exist_ok=True)
    release_df = pl.DataFrame(
        {
            "series_id": ["TEST", "TEST"],
            "observation_ts": [
                datetime(2025, 1, 1, tzinfo=UTC),
                datetime(2025, 1, 2, tzinfo=UTC),
            ],
            "value": [1.1, 2.2],
            "release_ts": [
                datetime(2025, 1, 1, 12, tzinfo=UTC),
                datetime(2025, 1, 4, tzinfo=UTC),
            ],
            "release_end_ts": [
                datetime(2025, 1, 4, tzinfo=UTC),
                datetime(2025, 1, 5, tzinfo=UTC),
            ],
        },
    )
    release_df.write_parquet(series_dir / "release_calendar.parquet")


def _write_fred_ml(path: Path) -> None:
    fred_df = pd.DataFrame(
        {
            "timestamp": [
                datetime(2025, 1, 1, tzinfo=UTC),
                datetime(2025, 1, 2, tzinfo=UTC),
            ],
            "series_id": ["TEST", "TEST"],
            "value": [1.0, 2.0],
        },
    )
    fred_df.to_parquet(path)


def test_join_fred_asof_respects_vintage_cutoff(tmp_path: Path) -> None:
    fred_path = tmp_path / "fred.parquet"
    vintage_dir = tmp_path / "vintages"
    _write_fred_ml(fred_path)
    _write_alfred_release(vintage_dir)

    market = pl.DataFrame(
        {
            "timestamp": [
                datetime(2025, 1, 1, 13, tzinfo=UTC),
                datetime(2025, 1, 2, 13, tzinfo=UTC),
            ],
        },
    )

    cutoff = datetime(2025, 1, 3, tzinfo=UTC)
    joined = join_fred_asof(
        market,
        timestamp_col="timestamp",
        lag_days=0,
        fred_path=fred_path,
        vintage_base_dir=vintage_dir,
        series_filter={"TEST"},
        vintage_policy=VintagePolicy.REAL_TIME,
        vintage_cutoff=cutoff,
    )
    assert isinstance(joined, pl.DataFrame)

    assert joined["TEST"].to_list() == [1.1, 2.0]
    assert joined["TEST__value_real_time"].to_list() == [1.1, 2.0]
    assert joined["TEST__value_final"].to_list() == [1.0, 2.0]
    vintage_strings = (
        joined.select(
            pl.col("TEST__value_vintage_ts").dt.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        .to_series()
        .to_list()
    )
    assert vintage_strings[0] == "2025-01-01T12:00:00"
    assert vintage_strings[1] is None


def test_join_fred_asof_final_policy_uses_latest(tmp_path: Path) -> None:
    fred_path = tmp_path / "fred.parquet"
    vintage_dir = tmp_path / "vintages"
    _write_fred_ml(fred_path)
    _write_alfred_release(vintage_dir)

    market = pl.DataFrame(
        {
            "timestamp": [
                datetime(2025, 1, 1, 13, tzinfo=UTC),
                datetime(2025, 1, 2, 13, tzinfo=UTC),
            ],
        },
    )

    joined = join_fred_asof(
        market,
        timestamp_col="timestamp",
        lag_days=0,
        fred_path=fred_path,
        vintage_base_dir=vintage_dir,
        series_filter={"TEST"},
        vintage_policy=VintagePolicy.FINAL,
    )
    assert isinstance(joined, pl.DataFrame)

    assert joined["TEST"].to_list() == [1.0, 2.0]
    assert joined["TEST__value_real_time"].to_list() == [1.0, 2.0]
    assert joined["TEST__value_final"].to_list() == [1.0, 2.0]
    assert joined.select(pl.col("TEST__value_vintage_ts")).to_series().to_list() == [None, None]
