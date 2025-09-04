"""
FRED as-of join utilities for macro feature integration.

Provides helpers to join long- or wide-format FRED data to a time-indexed
market DataFrame using as-of semantics with a configurable publication lag.
"""

# ruff: noqa: I001

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import ml._imports as _ml_imports

pd = _ml_imports.pd
pl = _ml_imports.pl
check_ml_dependencies = _ml_imports.check_ml_dependencies

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    import polars as pl


def _load_fred_ml_pl(fred_path: str | Path | None = None) -> pl.DataFrame:
    """
    Load FRED ML-format parquet (timestamp, series_id, value) as a Polars DataFrame.

    Falls back to wide updated format if ML-format is unavailable, converting to long.
    """
    if pl is None:
        check_ml_dependencies(["polars"])  # pragma: no cover
    path_ml = Path(fred_path) if fred_path else Path("data/fred/fred_indicators_ml_format.parquet")
    path_wide = Path("data/fred/fred_indicators_updated.parquet")

    if path_ml.exists():
        return pl.read_parquet(str(path_ml))

    if path_wide.exists():
        wide = pl.read_parquet(str(path_wide))
        # Convert to long format: (timestamp, series_id, value)
        value_cols = [c for c in wide.columns if c not in {"date", "timestamp_ns"}]
        long = wide.melt(id_vars=["date"], value_vars=value_cols, variable_name="series_id", value_name="value")
        return long.rename({"date": "timestamp"}).select(["timestamp", "series_id", "value"]).with_columns(
            [pl.col("timestamp").cast(pl.Datetime("ns"))],
        )

    # Return empty frame with expected schema if nothing present
    return pl.DataFrame({"timestamp": [], "series_id": [], "value": []})


def join_fred_asof(  # noqa: C901
    df: pl.DataFrame | pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    lag_days: int = 1,
    fred_path: str | Path | None = None,
) -> pl.DataFrame | pd.DataFrame:
    """
    Join FRED macro features to a time-indexed market DataFrame using as-of semantics.

    - Computes an effective publication time = FRED timestamp + lag_days.
    - Performs a backward as-of join so each market row sees the latest macro value
      available at that time (respecting publication lag).

    Parameters
    ----------
    df : DataFrame (Polars or pandas)
        Market data with a timestamp column.
    timestamp_col : str, default "timestamp"
        Name of the timestamp column in `df`.
    lag_days : int, default 1
        Publication lag, added to FRED timestamps before the as-of join.
    fred_path : str | Path, optional
        Explicit path to ML-format FRED parquet. Defaults to data/fred/...
    """
    # Polars path
    if pl is not None and isinstance(df, pl.DataFrame):
        fred = _load_fred_ml_pl(fred_path)
        if fred.is_empty():
            return df

        # Wide pivot for efficient as-of join (cast due to stub signature variance)
        fred_wide = cast(Any, fred).pivot(values="value", index="timestamp", columns="series_id").sort("timestamp")  # noqa: PD010
        if fred_wide.is_empty():
            return df

        fred_wide = fred_wide.with_columns(
            [pl.col("timestamp").dt.offset_by(f"{int(lag_days)}d").alias("ts_effective")],
        )

        # Ensure left is sorted by timestamp for join_asof
        if timestamp_col not in df.columns:
            return df
        left_pl = df.sort(timestamp_col)
        right_pl = fred_wide.sort("ts_effective")

        joined = left_pl.join_asof(right_pl, left_on=timestamp_col, right_on="ts_effective", strategy="backward")
        # Drop join key column
        if "ts_effective" in joined.columns:
            joined = joined.drop("ts_effective")
        return joined

    # Pandas path
    if pd is not None and isinstance(df, pd.DataFrame):
        fred_path_ml = Path(fred_path) if fred_path else Path("data/fred/fred_indicators_ml_format.parquet")
        fred_path_wide = Path("data/fred/fred_indicators_updated.parquet")

        if fred_path_ml.exists():
            fred_ml = pd.read_parquet(str(fred_path_ml))
            wide = fred_ml.pivot_table(index="timestamp", columns="series_id", values="value")
        elif fred_path_wide.exists():
            wide_src = pd.read_parquet(str(fred_path_wide))
            value_cols = [c for c in wide_src.columns if c not in {"date", "timestamp_ns"}]
            wide = wide_src.rename(columns={"date": "timestamp"})["timestamp"].to_frame()
            for c in value_cols:
                wide[c] = wide_src[c].to_numpy()
            wide = wide.set_index("timestamp")
        else:
            return df

        if wide.empty or timestamp_col not in df.columns:
            return df

        # Effective time and asof merge
        wide = wide.sort_index()
        wide["ts_effective"] = wide.index + pd.to_timedelta(lag_days, unit="D")
        left_pd = df.sort_values(timestamp_col)
        merged = pd.merge_asof(
            left_pd,
            wide.reset_index()[["ts_effective", *[c for c in wide.columns if c != "ts_effective"]]],
            left_on=timestamp_col,
            right_on="ts_effective",
            direction="backward",
        )
        return merged.drop(columns=["ts_effective"], errors="ignore")

    # If neither pandas nor polars matches, return as-is
    return df
