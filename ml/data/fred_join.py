"""
FRED as-of join utilities for macro feature integration.

Provides helpers to join long- or wide-format FRED data to a time-indexed market
DataFrame using as-of semantics with a configurable publication lag.

"""

# ruff: noqa: I001

from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable
from typing import Any, cast

import ml._imports as _ml_imports
from ml.ml_types import DataFrameLike, PolarsDF

pd = _ml_imports.pd
pl = _ml_imports.pl
check_ml_dependencies = _ml_imports.check_ml_dependencies


def _load_fred_ml_pl(fred_path: str | Path | None = None) -> PolarsDF:
    """
    Load FRED ML-format parquet (timestamp, series_id, value) as a Polars DataFrame.

    Falls back to wide updated format if ML-format is unavailable, converting to long.

    """
    if pl is None:
        check_ml_dependencies(["polars"])  # pragma: no cover
    _pl = pl
    assert _pl is not None
    path_ml = Path(fred_path) if fred_path else Path("data/fred/fred_indicators_ml_format.parquet")
    path_wide = Path("data/fred/fred_indicators_updated.parquet")

    if path_ml.exists():
        return cast(PolarsDF, _pl.read_parquet(str(path_ml)))

    if path_wide.exists():
        wide = _pl.read_parquet(str(path_wide))
        # Convert to long format: (timestamp, series_id, value)
        value_cols = [c for c in wide.columns if c not in {"date", "timestamp_ns"}]
        long = wide.melt(
            id_vars=["date"],
            value_vars=value_cols,
            variable_name="series_id",
            value_name="value",
        )
        return cast(
            PolarsDF,
            long.rename({"date": "timestamp"})
            .select(["timestamp", "series_id", "value"])
            .with_columns([_pl.col("timestamp").cast(_pl.Datetime("ns"))]),
        )

    # Return empty frame with expected schema if nothing present
    return cast(PolarsDF, _pl.DataFrame({"timestamp": [], "series_id": [], "value": []}))


def _iter_vintage_series_dirs(
    base_dir: Path,
    series_filter: set[str] | None,
) -> Iterable[tuple[str, Path]]:
    """
    Yield series id and directory pairs filtered by series ids when provided.
    """
    if not base_dir.exists():
        return []
    dirs: list[tuple[str, Path]] = []
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        series_id = child.name
        if series_filter is not None and series_id not in series_filter:
            continue
        dirs.append((series_id, child))
    return dirs


def _load_vintage_release_pl(
    base_dir: Path,
    series_filter: set[str] | None,
) -> PolarsDF:
    """
    Load vintage release metadata as a Polars DataFrame.
    """
    if pl is None:
        check_ml_dependencies(["polars"])  # pragma: no cover
    _pl = pl
    assert _pl is not None
    frames: list[PolarsDF] = []
    for series_id, series_dir in _iter_vintage_series_dirs(base_dir, series_filter):
        cal_path = series_dir / "release_calendar.parquet"
        if not cal_path.exists():
            continue
        df = _pl.read_parquet(str(cal_path))
        if df.is_empty():
            continue
        if "series_id" not in df.columns:
            df = df.with_columns([_pl.lit(series_id).alias("series_id")])
        frames.append(df)
    if not frames:
        return cast(PolarsDF, _pl.DataFrame())
    return cast(
        PolarsDF,
        _pl.concat(frames, how="vertical").sort(["release_ts", "observation_ts"]),
    )


def _load_vintage_release_pd(
    base_dir: Path,
    series_filter: set[str] | None,
) -> DataFrameLike:
    """
    Load vintage release metadata as a pandas DataFrame.
    """
    if pd is None:
        check_ml_dependencies(["pandas"])  # pragma: no cover
    _pd = pd
    assert _pd is not None
    frames: list[DataFrameLike] = []
    for series_id, series_dir in _iter_vintage_series_dirs(base_dir, series_filter):
        cal_path = series_dir / "release_calendar.parquet"
        if not cal_path.exists():
            continue
        df = _pd.read_parquet(cal_path)
        if df.empty:
            continue
        if "series_id" not in df.columns:
            df["series_id"] = series_id
        frames.append(df)
    if not frames:
        empty = _pd.DataFrame(
            columns=[
                "series_id",
                "observation_ts",
                "value",
                "release_ts",
                "release_end_ts",
            ],
        )
        return cast(DataFrameLike, empty)
    combined = _pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["release_ts", "observation_ts"])
    return cast(DataFrameLike, combined)


def join_fred_asof(
    df: DataFrameLike,
    *,
    timestamp_col: str = "timestamp",
    lag_days: int = 1,
    fred_path: str | Path | None = None,
    vintage_base_dir: str | Path | None = None,
) -> DataFrameLike:
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
    vintage_base_dir : str | Path | None, optional
        Base directory containing ALFRED vintage release calendars. When provided,
        actual release timestamps are used for point-in-time joins, falling back to
        ``lag_days`` for any series without vintage data.

    """
    vintage_dir = Path(vintage_base_dir).expanduser() if vintage_base_dir else None

    # Polars path
    if pl is not None and isinstance(df, pl.DataFrame):
        _pl = pl
        assert _pl is not None

        fred = _load_fred_ml_pl(fred_path)
        release_df = cast(PolarsDF, _pl.DataFrame())
        if vintage_dir is not None:
            series_filter: set[str] | None = None
            if not fred.is_empty() and "series_id" in fred.columns:
                series_filter = set(fred.get_column("series_id").unique().to_list())
            release_df = _load_vintage_release_pl(vintage_dir, series_filter)
            if not release_df.is_empty():
                release_series = set(release_df.get_column("series_id").unique().to_list())
                if not fred.is_empty() and release_series:
                    fred = fred.filter(~_pl.col("series_id").is_in(list(release_series)))

        frames: list[PolarsDF] = []
        if not fred.is_empty():
            if {"timestamp", "series_id", "value"} - set(fred.columns):
                return cast(DataFrameLike, df)
            frames.append(
                fred.select(["timestamp", "series_id", "value"]).with_columns(
                    _pl.lit(None).cast(_pl.Datetime("ns")).alias("release_ts"),
                ),
            )

        if release_df is not None and not release_df.is_empty():
            frames.append(
                release_df.rename({"observation_ts": "timestamp"}).select(
                    ["timestamp", "series_id", "value", "release_ts"],
                ),
            )

        if not frames:
            return cast(DataFrameLike, df)

        combined = _pl.concat(frames, how="vertical")
        combined = combined.with_columns(
            [
                _pl.col("timestamp").cast(_pl.Datetime("ns")),
                _pl.col("release_ts").cast(_pl.Datetime("ns")),
            ],
        )

        fred_wide = cast(Any, combined).pivot(
            values="value",
            index=["timestamp", "release_ts"],
            columns="series_id",
            aggregate_function="last",
        )
        if fred_wide.is_empty():
            return cast(DataFrameLike, df)

        fred_wide = fred_wide.with_columns(
            [
                _pl.when(_pl.col("release_ts").is_not_null())
                .then(_pl.col("release_ts"))
                .otherwise(_pl.col("timestamp").dt.offset_by(f"{int(lag_days)}d"))
                .alias("ts_effective"),
            ],
        )

        if timestamp_col not in df.columns:
            return cast(DataFrameLike, df)

        left_pl = df.sort(timestamp_col)
        target_dtype = left_pl.schema.get(timestamp_col)
        if target_dtype is not None and target_dtype != fred_wide.schema.get("ts_effective"):
            fred_wide = fred_wide.with_columns(
                _pl.col("ts_effective").cast(target_dtype),
            )
        right_pl = fred_wide.sort(["ts_effective", "timestamp"])

        joined = left_pl.join_asof(
            right_pl,
            left_on=timestamp_col,
            right_on="ts_effective",
            strategy="backward",
        )

        for col in ("ts_effective", "release_ts"):
            if col in joined.columns:
                joined = joined.drop(col)

        from typing import cast as _cast

        return _cast(DataFrameLike, joined)

    # Pandas path
    if pd is not None and isinstance(df, pd.DataFrame):
        fred_path_ml = (
            Path(fred_path) if fred_path else Path("data/fred/fred_indicators_ml_format.parquet")
        )
        fred_path_wide = Path("data/fred/fred_indicators_updated.parquet")

        if fred_path_ml.exists():
            fred_ml = pd.read_parquet(str(fred_path_ml))
        elif fred_path_wide.exists():
            wide_src = pd.read_parquet(str(fred_path_wide))
            value_cols = [c for c in wide_src.columns if c not in {"date", "timestamp_ns"}]
            fred_ml = wide_src.rename(columns={"date": "timestamp"}).melt(
                id_vars=["timestamp"],
                value_vars=value_cols,
                var_name="series_id",
                value_name="value",
            )
        else:
            fred_ml = pd.DataFrame(columns=["timestamp", "series_id", "value"])

        if not fred_ml.empty:
            fred_ml["timestamp"] = (
                pd.to_datetime(fred_ml["timestamp"], utc=True)
                .dt.tz_convert("UTC")
                .dt.tz_localize(None)
            )

        release_df_pd: pd.DataFrame = pd.DataFrame()
        if vintage_dir is not None:
            series_filter = (
                {str(x) for x in list(fred_ml["series_id"].unique())}
                if not fred_ml.empty
                else None
            )
            release_df_pd = _load_vintage_release_pd(vintage_dir, series_filter)
            if not release_df_pd.empty:
                release_series = {
                    str(x) for x in list(release_df_pd["series_id"].unique())
                }
                if not fred_ml.empty and release_series:
                    fred_ml = fred_ml[~fred_ml["series_id"].isin(release_series)]

        frames_pd: list[DataFrameLike] = []
        if not fred_ml.empty:
            base = fred_ml[["timestamp", "series_id", "value"]].copy()
            base["release_ts"] = pd.NaT
            frames_pd.append(base)

        if not release_df_pd.empty:
            rel = release_df_pd.rename(columns={"observation_ts": "timestamp"})[
                ["timestamp", "series_id", "value", "release_ts"]
            ].copy()
            rel["timestamp"] = (
                pd.to_datetime(rel["timestamp"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
            )
            rel["release_ts"] = (
                pd.to_datetime(rel["release_ts"], utc=True)
                .dt.tz_convert("UTC")
                .dt.tz_localize(None)
            )
            frames_pd.append(rel)

        if not frames_pd or timestamp_col not in df.columns:
            return cast(DataFrameLike, df)

        combined_pd = pd.concat(frames_pd, ignore_index=True)
        combined_pd = combined_pd.sort_values(["release_ts", "timestamp"])

        lag_offset = pd.to_timedelta(lag_days, unit="D")
        combined_pd["ts_effective"] = combined_pd["release_ts"].fillna(
            combined_pd["timestamp"] + lag_offset,
        )

        wide = combined_pd.pivot_table(
            index=["ts_effective", "release_ts", "timestamp"],
            columns="series_id",
            values="value",
            aggfunc="last",
        ).reset_index()
        wide.columns.name = None
        wide = wide.sort_values(["ts_effective", "timestamp"])

        left_pd = df.sort_values(timestamp_col)
        merged = pd.merge_asof(
            left_pd,
            wide,
            left_on=timestamp_col,
            right_on="ts_effective",
            direction="backward",
        )
        merged = merged.drop(columns=["ts_effective", "release_ts"], errors="ignore")
        return cast(DataFrameLike, merged)

    # If neither pandas nor polars matches, return as-is
    return df
