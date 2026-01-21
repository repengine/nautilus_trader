"""
FRED as-of join utilities for macro feature integration.

Provides helpers to join long- or wide-format FRED data to a time-indexed market
DataFrame using as-of semantics with a configurable publication lag.

"""

# ruff: noqa: I001

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import ml._imports as _ml_imports
from ml.ml_types import DataFrameLike, PolarsDF
from ml.data.vintage import VintagePolicy

pd = _ml_imports.pd
pl = _ml_imports.pl
check_ml_dependencies = _ml_imports.check_ml_dependencies

if TYPE_CHECKING:
    from pandas import DataFrame as PandasDataFrame
else:
    PandasDataFrame = Any


def _load_fred_ml_pl(fred_path: str | Path | None = None) -> PolarsDF:
    """
    Load FRED ML-format parquet (timestamp, series_id, value) as a Polars DataFrame.

    Falls back to wide updated format if ML-format is unavailable, converting to long.

    """
    if pl is None:
        check_ml_dependencies(["polars"])  # pragma: no cover
    _pl = pl
    assert _pl is not None
    path_ml = Path(fred_path) if fred_path else Path("data/features/macro/fred_indicators_ml_format.parquet")
    path_wide = Path("data/features/macro/fred_indicators_updated.parquet")

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
        raise FileNotFoundError(f"Vintage directory not found: {base_dir}")
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
        required_base = {"release_ts", "observation_ts", "value"}
        if not required_base.issubset(set(df.columns)):
            continue
        if "series_id" not in df.columns:
            df = df.with_columns([_pl.lit(series_id).alias("series_id")])
        if "release_end_ts" not in df.columns:
            df = df.with_columns(
                [
                    _pl.lit(None)
                    .cast(_pl.Datetime("ns", "UTC"))
                    .alias("release_end_ts"),
                ],
            )
        df = df.with_columns(
            [
                _pl.col("observation_ts").cast(_pl.Datetime("ns", "UTC")),
                _pl.col("release_ts").cast(_pl.Datetime("ns", "UTC")),
                _pl.col("release_end_ts").cast(_pl.Datetime("ns", "UTC")),
            ],
        ).select(
            [
                "series_id",
                "observation_ts",
                "value",
                "release_ts",
                "release_end_ts",
            ],
        )
        frames.append(cast(PolarsDF, df))
    if not frames:
        return cast(PolarsDF, _pl.DataFrame())
    return cast(
        PolarsDF,
        _pl.concat(frames, how="vertical").sort(["release_ts", "observation_ts"]),
    )


def _load_vintage_release_pd(
    base_dir: Path,
    series_filter: set[str] | None,
) -> PandasDataFrame:
    """
    Load vintage release metadata as a pandas DataFrame.
    """
    if pd is None:
        check_ml_dependencies(["pandas"])  # pragma: no cover
    _pd = pd
    assert _pd is not None
    frames: list[PandasDataFrame] = []
    for series_id, series_dir in _iter_vintage_series_dirs(base_dir, series_filter):
        cal_path = series_dir / "release_calendar.parquet"
        if not cal_path.exists():
            continue
        df = _pd.read_parquet(cal_path)
        if df.empty:
            continue
        if "series_id" not in df.columns:
            df["series_id"] = series_id
        frames.append(cast(PandasDataFrame, df))
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
        return cast(PandasDataFrame, empty)
    combined = _pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["release_ts", "observation_ts"])
    return cast(PandasDataFrame, combined)


def join_fred_asof(
    df: DataFrameLike,
    *,
    timestamp_col: str = "timestamp",
    lag_days: int = 1,
    fred_path: str | Path | None = None,
    vintage_base_dir: str | Path | None = None,
    series_filter: set[str] | None = None,
    vintage_policy: VintagePolicy = VintagePolicy.REAL_TIME,
    vintage_cutoff: datetime | None = None,
    include_revisions: bool = False,
    revision_mode: str = "core",
    revision_windows: list[int] | None = None,
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
        Explicit path to ML-format FRED parquet. Defaults to data/features/macro/...
    vintage_base_dir : str | Path | None, optional
        Base directory containing ALFRED vintage release calendars. When provided,
        actual release timestamps are used for point-in-time joins, falling back to
        ``lag_days`` for any series without vintage data.
    include_revisions : bool, default False
        If True, compute revision-aware features (prior values, revision deltas, net signals).
        Requires vintage_base_dir to be set and vintage data to be available.
    revision_mode : {"minimal", "core", "full"}, default "core"
        Feature mode when include_revisions=True:
        - minimal: current, prior_1m, revision_1m (3 features/series)
        - core: + mom_1m, pct_1m, net_signal_1m (6 features/series)
        - full: + prior_3m/12m, revision_3m, mom_3m/12m, pct_12m (12 features/series)
    revision_windows : list[int] | None, optional
        Months to use for prior/revision features. Defaults to [1, 3, 12].

    """
    cutoff_utc: datetime | None
    if vintage_cutoff is None:
        cutoff_utc = None
    elif vintage_cutoff.tzinfo is None:
        cutoff_utc = vintage_cutoff.replace(tzinfo=UTC)
    else:
        cutoff_utc = vintage_cutoff.astimezone(UTC)
    cutoff_naive = cutoff_utc.replace(tzinfo=None) if cutoff_utc is not None else None

    use_vintage = vintage_policy is VintagePolicy.REAL_TIME and vintage_base_dir is not None
    vintage_dir = None
    if use_vintage and vintage_base_dir is not None:
        vintage_dir = Path(vintage_base_dir).expanduser()

    # Polars path
    if pl is not None and isinstance(df, pl.DataFrame):
        _pl = pl
        assert _pl is not None

        fred = _load_fred_ml_pl(fred_path)
        series_names: set[str] = set()
        if "series_id" in fred.columns and not fred.is_empty():
            series_names.update(str(item) for item in fred.get_column("series_id").unique().to_list())
        if series_filter is not None and not fred.is_empty():
            fred = fred.filter(_pl.col("series_id").is_in(list(series_filter)))
        if cutoff_utc is not None and not fred.is_empty():
            fred = fred.filter(_pl.col("timestamp") <= cutoff_utc)
        release_df = cast(PolarsDF, _pl.DataFrame())
        if vintage_dir is not None:
            release_filter = series_filter
            if release_filter is None and not fred.is_empty() and "series_id" in fred.columns:
                release_filter = set(fred.get_column("series_id").unique().to_list())
            release_df = _load_vintage_release_pl(vintage_dir, release_filter)
            if not release_df.is_empty():
                # Normalize timestamp dtypes to avoid Null casts when filtering
                release_ts_dtype = release_df.schema.get("release_ts")
                observation_ts_dtype = release_df.schema.get("observation_ts")
                if release_ts_dtype is None or release_ts_dtype == _pl.Null:
                    release_df = release_df.with_columns(
                        _pl.col("release_ts").cast(_pl.Datetime("ns")),
                    )
                    release_ts_dtype = _pl.Datetime("ns")
                if observation_ts_dtype is None or observation_ts_dtype == _pl.Null:
                    release_df = release_df.with_columns(
                        _pl.col("observation_ts").cast(_pl.Datetime("ns")),
                    )
                    observation_ts_dtype = _pl.Datetime("ns")
                series_names.update(str(item) for item in release_df.get_column("series_id").unique().to_list())
                if cutoff_utc is not None:
                    release_dtype = release_df.schema.get("release_ts")
                    observation_dtype = release_df.schema.get("observation_ts")
                    release_lit = (
                        _pl.lit(cutoff_utc).cast(release_dtype)
                        if release_dtype is not None
                        else None
                    )
                    observation_lit = (
                        _pl.lit(cutoff_utc).cast(observation_dtype)
                        if observation_dtype is not None
                        else None
                    )
                    if release_lit is not None and observation_lit is not None:
                        release_df = release_df.filter(
                            (_pl.col("release_ts") <= release_lit)
                            & (_pl.col("observation_ts") <= observation_lit)
                        )
                    elif release_lit is not None:
                        release_df = release_df.filter(_pl.col("release_ts") <= release_lit)
        timestamp_dtype = _pl.Datetime("ns")
        if "timestamp" in fred.columns:
            inferred_dtype = fred.schema["timestamp"]
            if inferred_dtype == _pl.Null:
                timestamp_dtype = _pl.Datetime("ns")
            else:
                timestamp_dtype = inferred_dtype

        if cutoff_utc is not None and not fred.is_empty():
            cutoff_lit = _pl.lit(cutoff_utc).cast(timestamp_dtype)
            fred = fred.filter(_pl.col("timestamp") <= cutoff_lit)

        frames: list[PolarsDF] = []
        if not fred.is_empty():
            if {"timestamp", "series_id", "value"} - set(fred.columns):
                return cast(DataFrameLike, df)
            frames.append(
                fred.select(["timestamp", "series_id", "value"]).with_columns(
                    _pl.lit(None).cast(timestamp_dtype).alias("release_ts"),
                ),
            )

        if release_df is not None and not release_df.is_empty():
            release_select = (
                release_df.rename({"observation_ts": "timestamp"})
                .with_columns(
                    [
                        _pl.col("timestamp").cast(timestamp_dtype),
                        _pl.col("release_ts").cast(timestamp_dtype),
                    ],
                )
                .select(["timestamp", "series_id", "value", "release_ts"])
            )
            frames.append(release_select)

            # Compute revision features if requested
            if include_revisions and vintage_dir is not None:
                from ml.data.macro_revisions import compute_revision_features_pl

                revision_features = compute_revision_features_pl(
                    release_df,
                    series_filter=series_filter,
                    mode=revision_mode,  # type: ignore[arg-type]
                    monthly_windows=revision_windows,
                )

                if not revision_features.is_empty():
                    # revision_features has: timestamp, series_id, value, release_ts, feature_type
                    # We need to rename feature_type to series_id for pivot compatibility
                    revision_features = revision_features.select(
                        [
                            _pl.col("timestamp").cast(timestamp_dtype),
                            _pl.col("feature_type").alias("series_id"),  # Use feature name as series
                            _pl.col("value"),
                            _pl.col("release_ts").cast(timestamp_dtype),
                        ],
                    )
                    frames.append(revision_features)

        if not frames:
            return cast(DataFrameLike, df)

        combined = _pl.concat(frames, how="vertical")
        combined = combined.with_columns(
            [
                _pl.col("timestamp").cast(timestamp_dtype),
                _pl.col("release_ts").cast(timestamp_dtype),
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

        series_list = sorted(series_names)

        joined = left_pl.join_asof(
            right_pl,
            left_on=timestamp_col,
            right_on="ts_effective",
            strategy="backward",
        )

        if series_list:
            realtime_exprs = [
                _pl.col(name).alias(f"{name}__value_real_time")
                for name in series_list
                if name in joined.columns
            ]
            if realtime_exprs:
                joined = joined.with_columns(realtime_exprs)

        release_dtype_joined = joined.schema.get("release_ts")
        if release_dtype_joined is not None and series_list:
            vintage_exprs = []
            for name in series_list:
                if name in joined.columns:
                    vintage_exprs.append(
                        _pl.when(_pl.col(name).is_not_null())
                        .then(_pl.col("release_ts"))
                        .otherwise(None)
                        .cast(release_dtype_joined)
                        .alias(f"{name}__value_vintage_ts")
                    )
            if vintage_exprs:
                joined = joined.with_columns(vintage_exprs)

        if not fred.is_empty() and series_list:
            final_values = fred.select(
                [
                    _pl.col("timestamp").alias("observation_ts"),
                    _pl.col("series_id"),
                    _pl.col("value"),
                ],
            ).with_columns(
                [
                    _pl.col("observation_ts").cast(timestamp_dtype),
                    _pl.col("observation_ts")
                    .cast(timestamp_dtype)
                    .dt.offset_by(f"{int(lag_days)}d")
                    .alias("ts_effective_final"),
                ],
            )

            final_pivot = final_values.pivot(
                "series_id",
                index="ts_effective_final",
                values="value",
                aggregate_function="last",
            )

            if target_dtype is not None and final_pivot.schema.get("ts_effective_final") != target_dtype:
                final_pivot = final_pivot.with_columns(
                    _pl.col("ts_effective_final").cast(target_dtype),
                )

            rename_map = {
                name: f"{name}__value_final"
                for name in series_list
                if name in final_pivot.columns
            }
            if rename_map:
                final_pivot = final_pivot.rename(rename_map)

            final_columns = [col for col in final_pivot.columns if col != "ts_effective_final"]
            if final_columns:
                joined = joined.join_asof(
                    final_pivot.sort("ts_effective_final"),
                    left_on=timestamp_col,
                    right_on="ts_effective_final",
                    strategy="backward",
                )
                if "ts_effective_final" in joined.columns:
                    joined = joined.drop("ts_effective_final")

        for col in ("ts_effective", "release_ts"):
            if col in joined.columns:
                joined = joined.drop(col)

        from typing import cast as _cast

        return _cast(DataFrameLike, joined)

    # Pandas path
    if pd is not None and isinstance(df, pd.DataFrame):
        df_pd = cast(PandasDataFrame, df)
        fred_path_ml = (
            Path(fred_path) if fred_path else Path("data/features/macro/fred_indicators_ml_format.parquet")
        )
        fred_path_wide = Path("data/features/macro/fred_indicators_updated.parquet")

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

        series_list_pd: set[str] = set()
        if series_filter is not None and not fred_ml.empty:
            fred_ml = fred_ml[fred_ml["series_id"].isin(series_filter)]

        if not fred_ml.empty:
            fred_ml["timestamp"] = (
                pd.to_datetime(fred_ml["timestamp"], utc=True)
                .dt.tz_convert("UTC")
                .dt.tz_localize(None)
            )
            if cutoff_naive is not None:
                fred_ml = fred_ml[fred_ml["timestamp"] <= cutoff_naive]
            series_list_pd.update(str(x) for x in fred_ml["series_id"].unique())

        release_df_pd: PandasDataFrame = pd.DataFrame()
        if vintage_dir is not None:
            release_filter = series_filter
            if release_filter is None and not fred_ml.empty:
                release_filter = {str(x) for x in list(fred_ml["series_id"].unique())}
            release_df_pd = _load_vintage_release_pd(vintage_dir, release_filter)
            if not release_df_pd.empty:
                if cutoff_naive is not None:
                    release_df_pd = release_df_pd[
                        (release_df_pd["release_ts"] <= cutoff_naive)
                        & (release_df_pd["observation_ts"] <= cutoff_naive)
                    ]
                series_list_pd.update(str(x) for x in release_df_pd["series_id"].unique())

        frames_pd: list[PandasDataFrame] = []
        if not fred_ml.empty:
            base = fred_ml[["timestamp", "series_id", "value"]].copy()
            base["release_ts"] = pd.NaT
            frames_pd.append(cast(PandasDataFrame, base))

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

        if not frames_pd or timestamp_col not in df_pd.columns:
            return df_pd

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

        if series_list_pd:
            for name in series_list_pd:
                if name in merged.columns:
                    merged[f"{name}__value_real_time"] = merged[name]
                    if "release_ts" in merged.columns:
                        merged[f"{name}__value_vintage_ts"] = merged["release_ts"].where(
                            merged[name].notna(),
                            pd.NaT,
                        )

        final_wide = pd.DataFrame()
        if not fred_ml.empty and series_list_pd:
            final_ml = fred_ml[["timestamp", "series_id", "value"]].copy()
            final_ml["ts_effective_final"] = final_ml["timestamp"] + lag_offset
            final_wide = (
                final_ml.pivot_table(
                    index="ts_effective_final",
                    columns="series_id",
                    values="value",
                    aggfunc="last",
                )
                .reset_index()
                .sort_values("ts_effective_final")
            )
            rename_map_pd = {
                name: f"{name}__value_final"
                for name in series_list_pd
                if name in final_wide.columns
            }
            if rename_map_pd:
                final_wide = final_wide.rename(columns=rename_map_pd)
                final_merge = pd.merge_asof(
                    left_pd[[timestamp_col]],
                    final_wide,
                    left_on=timestamp_col,
                    right_on="ts_effective_final",
                    direction="backward",
                )
                final_merge = final_merge.drop(columns=["ts_effective_final"], errors="ignore")
                final_merge.index = left_pd.index
                merged = merged.join(
                    final_merge.drop(columns=[timestamp_col], errors="ignore"),
                )

        merged = merged.drop(columns=["ts_effective", "release_ts"], errors="ignore")
        return cast(DataFrameLike, merged)

    # If neither pandas nor polars matches, return as-is
    return df
