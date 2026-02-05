"""
Batch execution backend for the canonical feature pipeline spec.

This module executes `PipelineSpec` transforms over Polars/Pandas DataFrames,
reusing existing batch-capable components (FeatureCalculator, calendar/events providers,
macro joins) to ensure training/inference parity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ml._imports import pd as pd_runtime
from ml._imports import pl as pl_runtime
from ml.data.fred_join import join_fred_asof
from ml.data.providers.factory import ProviderFactory
from ml.data.vintage import VintagePolicy
from ml.features.common.feature_calculator import FeatureCalculator
from ml.features.config import FeatureConfig
from ml.features.config import derive_ohlcv_feature_config
from ml.features.pipeline import PipelineRunner
from ml.features.pipeline import PipelineSpec
from ml.features.pipeline import TransformSpec
from ml.features.pipeline import transform_feature_names
from ml.registry.base import DataRequirements


if TYPE_CHECKING:
    import pandas as _pd
    import polars as _pl

    from ml.data.providers.calendar import MarketCalendarProvider
    from ml.data.providers.events import EventScheduleProvider
else:  # pragma: no cover - typing fallback
    _pd = Any
    _pl = Any
    MarketCalendarProvider = Any
    EventScheduleProvider = Any


logger = logging.getLogger(__name__)

PL: Any = cast(Any, pl_runtime)
PD: Any = cast(Any, pd_runtime)

_OHLCV_TRANSFORMS: set[str] = {
    "returns",
    "momentum",
    "volatility",
    "volume_ratio",
    "core_indicators",
    "microstructure",
    "trade_flow",
}


@dataclass(slots=True)
class PipelineBatchContext:
    """
    Context for executing batch pipeline transforms.

    Attributes:
        feature_config: Base FeatureConfig for indicator parameters.
        macro_lag_days: Lag days for macro joins.
        fred_path: Path to macro parquet data.
        vintage_base_dir: Directory containing ALFRED vintages.
        macro_series_ids: Series IDs for macro joins/deltas.
        vintage_policy: Vintage policy for macro joins.
        vintage_as_of: Optional vintage cutoff timestamp.
        include_macro_revisions: Whether to include revision features.
        macro_revision_mode: Revision mode ("minimal", "core", "full").
        macro_revision_windows: Optional revision windows.
        calendar_provider: Optional calendar provider.
        event_provider: Optional event schedule provider.
        calendar_exchange: Exchange identifier for calendar features.
        event_instruments: Optional instruments for event features.
    """

    feature_config: FeatureConfig
    macro_lag_days: int = 1
    fred_path: str | None = None
    vintage_base_dir: Path | None = None
    macro_series_ids: tuple[str, ...] | None = None
    vintage_policy: VintagePolicy = VintagePolicy.REAL_TIME
    vintage_as_of: datetime | None = None
    include_macro_revisions: bool = False
    macro_revision_mode: str = "core"
    macro_revision_windows: tuple[int, ...] | None = None
    calendar_provider: MarketCalendarProvider | None = None
    event_provider: EventScheduleProvider | None = None
    calendar_exchange: str = "NYSE"
    event_instruments: list[str] | None = None


class PipelineBatchExecutor:
    """
    Execute canonical feature transforms over batch DataFrames.

    This executor is used by dataset builders to produce the same canonical
    feature names as the pipeline spec while delegating to batch-capable
    implementations for each transform family.
    """

    def __init__(
        self,
        spec: PipelineSpec,
        *,
        allowable: DataRequirements,
        context: PipelineBatchContext,
    ) -> None:
        self._spec = spec
        self._allowable = allowable
        self._context = context
        # Validate transform gating
        self._runner = PipelineRunner(spec, allowable)

    def execute_polars(self, df: _pl.DataFrame) -> _pl.DataFrame:
        """
        Execute transforms over a Polars DataFrame.

        Args:
            df: Polars DataFrame with OHLCV + timestamp columns.

        Returns:
            Polars DataFrame with canonical feature columns appended.
        """
        if PL is None:
            raise RuntimeError("Polars is required for execute_polars")
        if df.is_empty():
            return df

        out = df
        transforms = self._spec.transforms

        if self._needs_ohlcv(transforms):
            out = self._append_ohlcv_features_polars(out, transforms)

        if self._has_transform(transforms, "macro"):
            out = self._append_macro_features_polars(out)
            macro_names = self._feature_names_for(transforms, {"macro"})
            out = self._ensure_polars_columns(out, macro_names)

        if self._has_transform(transforms, "macro_deltas"):
            series_ids = self._macro_series_ids_for_transform("macro_deltas")
            out = self._append_macro_deltas_polars(out, series_ids)
            delta_names = self._feature_names_for(transforms, {"macro_deltas"})
            out = self._ensure_polars_columns(out, delta_names)

        if self._has_transform(transforms, "macro_composites"):
            out = self._append_macro_composites_polars(out)
            composite_names = self._feature_names_for(transforms, {"macro_composites"})
            out = self._ensure_polars_columns(out, composite_names)

        if self._has_transform(transforms, "calendar"):
            out = self._append_calendar_features_polars(out, transforms)

        if self._has_transform(transforms, "event_schedule"):
            out = self._append_event_features_polars(out, transforms)

        return out

    def execute_pandas(self, df: _pd.DataFrame) -> _pd.DataFrame:
        """
        Execute transforms over a Pandas DataFrame.

        Args:
            df: Pandas DataFrame with OHLCV + timestamp columns.

        Returns:
            Pandas DataFrame with canonical feature columns appended.
        """
        if PD is None:
            raise RuntimeError("Pandas is required for execute_pandas")
        if df.empty:
            return df

        out = df
        transforms = self._spec.transforms

        if self._needs_ohlcv(transforms):
            out = self._append_ohlcv_features_pandas(out, transforms)

        if self._has_transform(transforms, "macro"):
            out = self._append_macro_features_pandas(out)
            macro_names = self._feature_names_for(transforms, {"macro"})
            out = self._ensure_pandas_columns(out, macro_names)

        if self._has_transform(transforms, "macro_deltas"):
            series_ids = self._macro_series_ids_for_transform("macro_deltas")
            out = self._append_macro_deltas_pandas(out, series_ids)
            delta_names = self._feature_names_for(transforms, {"macro_deltas"})
            out = self._ensure_pandas_columns(out, delta_names)

        if self._has_transform(transforms, "macro_composites"):
            out = self._append_macro_composites_pandas(out)
            composite_names = self._feature_names_for(transforms, {"macro_composites"})
            out = self._ensure_pandas_columns(out, composite_names)

        if self._has_transform(transforms, "calendar"):
            out = self._append_calendar_features_pandas(out, transforms)

        if self._has_transform(transforms, "event_schedule"):
            out = self._append_event_features_pandas(out, transforms)

        return out

    def _needs_ohlcv(self, transforms: list[TransformSpec]) -> bool:
        return any(ts.name in _OHLCV_TRANSFORMS for ts in transforms)

    @staticmethod
    def _has_transform(transforms: list[TransformSpec], name: str) -> bool:
        return any(ts.name == name for ts in transforms)

    def _macro_series_ids_for_transform(self, name: str) -> tuple[str, ...]:
        for ts in self._spec.transforms:
            if ts.name == name:
                series_ids = ts.params.get("series_ids", [])
                return tuple(series_ids)
        return tuple(self._context.macro_series_ids or ())

    def _append_ohlcv_features_polars(self, df: _pl.DataFrame, transforms: list[TransformSpec]) -> _pl.DataFrame:
        cfg = derive_ohlcv_feature_config(
            self._context.feature_config,
            transforms,
            allowable=self._allowable,
        )
        calculator = FeatureCalculator(cfg)
        features_df, _ = calculator.calculate_features(df, mode="batch")
        feature_names = self._feature_names_for(transforms, _OHLCV_TRANSFORMS)
        features_df = self._ensure_polars_columns(features_df, feature_names)
        return df.hstack(features_df.select(feature_names))

    def _append_ohlcv_features_pandas(self, df: _pd.DataFrame, transforms: list[TransformSpec]) -> _pd.DataFrame:
        cfg = derive_ohlcv_feature_config(
            self._context.feature_config,
            transforms,
            allowable=self._allowable,
        )
        calculator = FeatureCalculator(cfg)
        features_df, _ = calculator.calculate_features(df, mode="batch")
        feature_names = self._feature_names_for(transforms, _OHLCV_TRANSFORMS)
        features_df = self._ensure_pandas_columns(features_df, feature_names)
        combined = PD.concat([df.reset_index(drop=True), features_df[feature_names]], axis=1)
        return cast("_pd.DataFrame", combined)

    def _append_macro_features_polars(self, df: _pl.DataFrame) -> _pl.DataFrame:
        ts_col = self._timestamp_column(df)
        before_cols = set(df.columns)
        revision_windows = (
            list(self._context.macro_revision_windows)
            if self._context.macro_revision_windows is not None
            else None
        )
        joined = join_fred_asof(
            df,
            timestamp_col=ts_col,
            lag_days=self._context.macro_lag_days,
            fred_path=self._context.fred_path,
            vintage_base_dir=self._context.vintage_base_dir,
            series_filter=set(self._context.macro_series_ids) if self._context.macro_series_ids else None,
            vintage_policy=self._context.vintage_policy,
            vintage_cutoff=self._context.vintage_as_of,
            include_revisions=self._context.include_macro_revisions,
            revision_mode=self._context.macro_revision_mode,
            revision_windows=revision_windows,
        )
        df_joined = cast("_pl.DataFrame", joined if hasattr(joined, "schema") else PL.from_pandas(joined))
        if "timestamp_right" in df_joined.columns:
            df_joined = df_joined.drop("timestamp_right")
        macro_cols = [c for c in df_joined.columns if c not in before_cols]
        if macro_cols:
            exprs = [PL.col(c).is_not_null() for c in macro_cols]
            if exprs:
                any_macro = exprs[0]
                for ex in exprs[1:]:
                    any_macro = any_macro | ex
                df_joined = df_joined.with_columns(
                    [any_macro.cast(PL.Int32).alias("is_macro_available")],
                )
            fills = [
                PL.col(c).fill_null(0)
                for c in macro_cols
                if df_joined.schema.get(c) is not None and df_joined.schema[c].is_numeric()
            ]
            if fills:
                df_joined = df_joined.with_columns(fills)
        else:
            df_joined = df_joined.with_columns(
                [PL.lit(0).cast(PL.Int32).alias("is_macro_available")],
            )
        return df_joined

    def _append_macro_features_pandas(self, df: _pd.DataFrame) -> _pd.DataFrame:
        ts_col = self._timestamp_column(df)
        before_cols = set(df.columns)
        revision_windows = (
            list(self._context.macro_revision_windows)
            if self._context.macro_revision_windows is not None
            else None
        )
        joined = join_fred_asof(
            df,
            timestamp_col=ts_col,
            lag_days=self._context.macro_lag_days,
            fred_path=self._context.fred_path,
            vintage_base_dir=self._context.vintage_base_dir,
            series_filter=set(self._context.macro_series_ids) if self._context.macro_series_ids else None,
            vintage_policy=self._context.vintage_policy,
            vintage_cutoff=self._context.vintage_as_of,
            include_revisions=self._context.include_macro_revisions,
            revision_mode=self._context.macro_revision_mode,
            revision_windows=revision_windows,
        )
        df_joined = cast("_pd.DataFrame", joined)
        if "timestamp_right" in df_joined.columns:
            df_joined = df_joined.drop(columns=["timestamp_right"])
        macro_cols = [c for c in df_joined.columns if c not in before_cols]
        if macro_cols:
            df_joined[macro_cols] = df_joined[macro_cols].fillna(0)
            if "is_macro_available" not in df_joined.columns:
                df_joined["is_macro_available"] = (
                    df_joined[macro_cols].notna().any(axis=1).astype(int)
                    if macro_cols
                    else 0
                )
        else:
            df_joined["is_macro_available"] = 0
        return df_joined

    def _append_macro_deltas_polars(self, df: _pl.DataFrame, series_ids: tuple[str, ...]) -> _pl.DataFrame:
        if not series_ids:
            return df
        if df.is_empty():
            return df
        time_col = self._timestamp_column(df)
        present = [series_id for series_id in series_ids if series_id in df.columns]
        if not present:
            return df
        if "instrument_id" in df.columns:
            df_sorted = df.sort(["instrument_id", time_col])
            exprs = [
                PL.col(series_id)
                .diff()
                .over("instrument_id")
                .fill_null(0.0)
                .alias(f"{series_id}_delta_1d")
                for series_id in present
            ]
            return df_sorted.with_columns(exprs)
        df_sorted = df.sort(time_col)
        exprs = [
            PL.col(series_id).diff().fill_null(0.0).alias(f"{series_id}_delta_1d")
            for series_id in present
        ]
        return df_sorted.with_columns(exprs)

    def _append_macro_deltas_pandas(self, df: _pd.DataFrame, series_ids: tuple[str, ...]) -> _pd.DataFrame:
        if not series_ids:
            return df
        if df.empty:
            return df
        time_col = self._timestamp_column(df)
        present = [series_id for series_id in series_ids if series_id in df.columns]
        if not present:
            return df
        df_sorted = df.sort_values(
            ["instrument_id", time_col] if "instrument_id" in df.columns else [time_col],
        ).copy()
        if "instrument_id" in df_sorted.columns:
            for series_id in present:
                df_sorted[f"{series_id}_delta_1d"] = (
                    df_sorted.groupby("instrument_id")[series_id].diff().fillna(0.0)
                )
            return df_sorted
        for series_id in present:
            df_sorted[f"{series_id}_delta_1d"] = df_sorted[series_id].diff().fillna(0.0)
        return df_sorted

    def _append_macro_composites_polars(self, df: _pl.DataFrame) -> _pl.DataFrame:
        from ml.features.macro_composites import compute_macro_composites_pl

        return compute_macro_composites_pl(df)

    def _append_macro_composites_pandas(self, df: _pd.DataFrame) -> _pd.DataFrame:
        if PL is None:
            raise RuntimeError("Polars required for macro composites")
        pl_df = PL.from_pandas(df)
        pl_df = self._append_macro_composites_polars(pl_df)
        return pl_df.to_pandas()

    def _append_calendar_features_polars(
        self,
        df: _pl.DataFrame,
        transforms: list[TransformSpec],
    ) -> _pl.DataFrame:
        provider = self._context.calendar_provider
        if provider is None:
            provider = ProviderFactory().get_calendar_provider()
        ts_col = self._timestamp_column(df)
        ts_series_ns, target_dtype = self._timestamp_series_ns(df, ts_col)
        calendar_df = provider.compute_features(ts_series_ns, exchange=self._context.calendar_exchange)
        calendar_df = self._align_timestamp_column(calendar_df, ts_col, target_dtype)
        feature_names = self._feature_names_for(transforms, {"calendar"})
        calendar_df = self._ensure_polars_columns(calendar_df, feature_names)
        return self._join_polars(df, calendar_df, ts_col, feature_names)

    def _append_calendar_features_pandas(
        self,
        df: _pd.DataFrame,
        transforms: list[TransformSpec],
    ) -> _pd.DataFrame:
        provider = self._context.calendar_provider
        if provider is None:
            provider = ProviderFactory().get_calendar_provider()
        ts_col = self._timestamp_column(df)
        ts_series_ns = self._timestamp_series_ns_pandas(df, ts_col)
        calendar_df = provider.compute_features(ts_series_ns, exchange=self._context.calendar_exchange)
        calendar_pd = calendar_df.to_pandas()
        calendar_pd = self._align_timestamp_column_pandas(calendar_pd, ts_col, df[ts_col].dtype)
        feature_names = self._feature_names_for(transforms, {"calendar"})
        calendar_pd = self._ensure_pandas_columns(calendar_pd, feature_names)
        return self._join_pandas(df, calendar_pd, ts_col, feature_names)

    def _append_event_features_polars(
        self,
        df: _pl.DataFrame,
        transforms: list[TransformSpec],
    ) -> _pl.DataFrame:
        provider = self._context.event_provider
        if provider is None:
            provider = ProviderFactory().get_event_provider()
        ts_col = self._timestamp_column(df)
        ts_series_ns, target_dtype = self._timestamp_series_ns(df, ts_col)
        instruments = self._context.event_instruments
        if instruments is None and "instrument_id" in df.columns:
            instruments = list(dict.fromkeys(df.get_column("instrument_id").to_list()))
        event_df = provider.compute_features(ts_series_ns, instruments=instruments or [])
        event_df = self._align_timestamp_column(event_df, ts_col, target_dtype)
        feature_names = self._feature_names_for(transforms, {"event_schedule"})
        event_df = self._ensure_polars_columns(event_df, feature_names)
        return self._join_polars(df, event_df, ts_col, feature_names)

    def _append_event_features_pandas(
        self,
        df: _pd.DataFrame,
        transforms: list[TransformSpec],
    ) -> _pd.DataFrame:
        provider = self._context.event_provider
        if provider is None:
            provider = ProviderFactory().get_event_provider()
        ts_col = self._timestamp_column(df)
        ts_series_ns = self._timestamp_series_ns_pandas(df, ts_col)
        instruments = self._context.event_instruments
        if instruments is None and "instrument_id" in df.columns:
            instruments = list(dict.fromkeys(df["instrument_id"].tolist()))
        event_df = provider.compute_features(ts_series_ns, instruments=instruments or [])
        event_pd = event_df.to_pandas()
        event_pd = self._align_timestamp_column_pandas(event_pd, ts_col, df[ts_col].dtype)
        feature_names = self._feature_names_for(transforms, {"event_schedule"})
        event_pd = self._ensure_pandas_columns(event_pd, feature_names)
        return self._join_pandas(df, event_pd, ts_col, feature_names)

    @staticmethod
    def _feature_names_for(transforms: list[TransformSpec], allowed: set[str]) -> list[str]:
        names: list[str] = []
        for ts in transforms:
            if ts.name in allowed:
                names.extend(transform_feature_names(ts))
        return names

    @staticmethod
    def _timestamp_column(df: Any) -> str:
        if "timestamp" in df.columns:
            return "timestamp"
        if "ts_event" in df.columns:
            return "ts_event"
        raise KeyError("Missing timestamp column (timestamp or ts_event)")

    @staticmethod
    def _timestamp_series_ns(df: _pl.DataFrame, ts_col: str) -> tuple[_pl.Series, Any]:
        series = df.get_column(ts_col)
        dtype = series.dtype
        if dtype == PL.Datetime:
            return series.cast(PL.Int64), dtype
        if dtype == PL.Datetime("ns", "UTC"):
            return series.cast(PL.Int64), dtype
        if dtype == PL.Int64:
            return series, dtype
        return series.cast(PL.Int64), dtype

    @staticmethod
    def _timestamp_series_ns_pandas(df: _pd.DataFrame, ts_col: str) -> _pl.Series:
        series = df[ts_col]
        if series.dtype.kind == "M":
            if getattr(series.dt, "tz", None) is not None:
                series = series.dt.tz_convert("UTC").dt.tz_localize(None)
            ts_ns = (series.astype("datetime64[ns]").astype("int64")).tolist()
        else:
            ts_ns = series.astype("int64").tolist()
        return cast("_pl.Series", PL.Series(ts_ns))

    @staticmethod
    def _align_timestamp_column(features_df: _pl.DataFrame, ts_col: str, target_dtype: Any) -> _pl.DataFrame:
        if ts_col != "timestamp":
            features_df = features_df.rename({"timestamp": ts_col})
        if target_dtype == PL.Datetime or str(target_dtype).startswith("Datetime"):
            features_df = features_df.with_columns(
                PL.from_epoch(PL.col(ts_col), time_unit="ns").cast(target_dtype).alias(ts_col),
            )
        else:
            features_df = features_df.with_columns(PL.col(ts_col).cast(target_dtype).alias(ts_col))
        return features_df

    @staticmethod
    def _align_timestamp_column_pandas(features_df: _pd.DataFrame, ts_col: str, target_dtype: Any) -> _pd.DataFrame:
        if ts_col != "timestamp":
            features_df = features_df.rename(columns={"timestamp": ts_col})
        if str(target_dtype).startswith("datetime"):
            ts_series = PD.to_datetime(features_df[ts_col], unit="ns", utc=True)
            if "UTC" not in str(target_dtype):
                ts_series = ts_series.dt.tz_convert(None)
            features_df[ts_col] = ts_series
        else:
            features_df[ts_col] = features_df[ts_col].astype(target_dtype)
        return features_df

    @staticmethod
    def _ensure_polars_columns(df: _pl.DataFrame, columns: list[str]) -> _pl.DataFrame:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            df = df.with_columns([PL.lit(0.0).alias(col) for col in missing])
        return df

    @staticmethod
    def _ensure_pandas_columns(df: _pd.DataFrame, columns: list[str]) -> _pd.DataFrame:
        for col in columns:
            if col not in df.columns:
                df[col] = 0.0
        return df

    @staticmethod
    def _join_polars(
        base: _pl.DataFrame,
        features: _pl.DataFrame,
        ts_col: str,
        feature_cols: list[str],
    ) -> _pl.DataFrame:
        selected = [ts_col] + [c for c in feature_cols if c not in base.columns]
        if len(selected) == 1:
            return base
        return base.join(features.select(selected), on=ts_col, how="left")

    @staticmethod
    def _join_pandas(
        base: _pd.DataFrame,
        features: _pd.DataFrame,
        ts_col: str,
        feature_cols: list[str],
    ) -> _pd.DataFrame:
        selected = [ts_col] + [c for c in feature_cols if c not in base.columns]
        if len(selected) == 1:
            return base
        features_subset = features[selected]
        return base.merge(features_subset, on=ts_col, how="left")
