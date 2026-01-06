"""
Rolling soft label generation for Chronos distillation.

This module generates teacher soft labels using rolling forecasts
and aligns them to forecasted timestamps for student training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import numpy as np

from ml._imports import HAS_AUTOGLUON
from ml._imports import HAS_PANDAS
from ml._imports import HAS_POLARS
from ml._imports import TimeSeriesDataFrame
from ml._imports import check_ml_dependencies
from ml._imports import pd
from ml._imports import pl
from ml.config.autogluon import AutoGluonDataConfig
from ml.config.autogluon import ChronosDistillationConfig
from ml.config.autogluon import ChronosTrainingConfig
from ml.data.autogluon_adapter import convert_to_timeseries_pandas
from ml.data.common.known_future_features import KnownFutureFeatureComponent


if TYPE_CHECKING:
    import pandas as _pd
    import polars as _pl
    from autogluon.timeseries import TimeSeriesPredictor

    PandasDataFrame: TypeAlias = _pd.DataFrame
    PolarsDataFrame: TypeAlias = _pl.DataFrame
else:
    PandasDataFrame: TypeAlias = Any
    PolarsDataFrame: TypeAlias = Any


__all__ = [
    "DistillationDataset",
    "SoftLabelStats",
    "build_distillation_dataset",
    "generate_rolling_soft_labels",
]


logger = logging.getLogger(__name__)


_TIME_BASED_KNOWN_FEATURES = {
    "hour",
    "minute",
    "tod_sin",
    "tod_cos",
    "dow",
    "dow_sin",
    "dow_cos",
    "is_market_open",
    "is_premarket",
    "is_aftermarket",
}


@dataclass(frozen=True)
class SoftLabelStats:
    """Summary statistics for rolling soft label generation."""

    total_candidates: int
    eligible_candidates: int
    generated: int
    total_series: int
    used_series: int

    @property
    def coverage(self) -> float:
        """Return the fraction of eligible rows that received a soft label."""
        if self.eligible_candidates == 0:
            return 0.0
        return self.generated / self.eligible_candidates


@dataclass(frozen=True)
class DistillationDataset:
    """Container for distilled training data and soft label metadata."""

    data: _pl.DataFrame | _pd.DataFrame
    labels: _pd.DataFrame
    stats: SoftLabelStats


def _predictions_to_frame(predictions: Any) -> _pd.DataFrame:
    if not HAS_PANDAS or pd is None:
        check_ml_dependencies(["pandas"])
        raise ImportError("Pandas not available")

    if hasattr(predictions, "reset_index"):
        df = predictions.reset_index()
        if isinstance(df, pd.DataFrame):
            return cast(PandasDataFrame, df)
    return cast(PandasDataFrame, pd.DataFrame(predictions))


def _future_index_to_frame(future_index: Any) -> _pd.DataFrame:
    if not HAS_PANDAS or pd is None:
        check_ml_dependencies(["pandas"])
        raise ImportError("Pandas not available")

    if hasattr(future_index, "reset_index"):
        df = future_index.reset_index()
        if isinstance(df, pd.DataFrame):
            return cast(PandasDataFrame, df)
    return cast(PandasDataFrame, pd.DataFrame(future_index))


def _missing_covariates(
    future_covariates: _pd.DataFrame,
    *,
    known_covariates: list[str],
) -> list[str]:
    missing: list[str] = []
    for covariate in known_covariates:
        if covariate not in future_covariates.columns:
            missing.append(covariate)
            continue
        if future_covariates[covariate].isna().any():
            missing.append(covariate)
    return missing


def _build_future_covariates(
    future_index: _pd.DataFrame,
    *,
    series_df: _pd.DataFrame,
    history_df: _pd.DataFrame,
    known_covariates: list[str],
) -> _pd.DataFrame:
    if not HAS_PANDAS or pd is None:
        check_ml_dependencies(["pandas"])
        raise ImportError("Pandas not available")

    future_covariates = future_index.copy()
    if "item_id" in future_covariates.columns:
        future_covariates["item_id"] = future_covariates["item_id"].astype(str)

    available_covariates = [cov for cov in known_covariates if cov in series_df.columns]
    if available_covariates:
        covariate_frame = series_df[
            ["item_id", "timestamp", *available_covariates]
        ].copy()
        future_covariates = future_covariates.merge(
            covariate_frame,
            on=["item_id", "timestamp"],
            how="left",
        )

    missing_covariates = _missing_covariates(
        future_covariates,
        known_covariates=known_covariates,
    )
    if missing_covariates:
        if "time_index" not in history_df.columns:
            raise ValueError(
                "Missing time_index column required to compute future known covariates "
                f"({missing_covariates})"
            )
        last_time_index = int(history_df["time_index"].iloc[-1])
        future_covariates = future_covariates.copy()
        future_covariates["time_index"] = np.arange(
            last_time_index + 1,
            last_time_index + 1 + len(future_covariates),
        )
        include_calendar = any(
            covariate not in _TIME_BASED_KNOWN_FEATURES for covariate in missing_covariates
        )
        component = KnownFutureFeatureComponent(include_calendar=include_calendar)
        component_frame = future_covariates.copy()
        if "instrument_id" not in component_frame.columns:
            component_frame["instrument_id"] = component_frame["item_id"]
        enriched = component.add_known_future_features_pandas(component_frame)
        for covariate in missing_covariates:
            if covariate in enriched.columns:
                future_covariates[covariate] = enriched[covariate]

    remaining_missing = _missing_covariates(
        future_covariates,
        known_covariates=known_covariates,
    )
    if remaining_missing:
        raise ValueError(
            "Future known_covariates missing values for columns: "
            f"{sorted(remaining_missing)}"
        )

    return future_covariates[["item_id", "timestamp", *known_covariates]]


def _select_prediction_column(columns: list[str]) -> str:
    if "mean" in columns:
        return "mean"
    if "0.5" in columns:
        return "0.5"
    numeric_candidates = [col for col in columns if col not in {"item_id", "timestamp"}]
    if not numeric_candidates:
        raise ValueError("Prediction output contains no numeric columns")
    return numeric_candidates[0]


def _extract_prediction(
    predictions: Any,
    *,
    item_id: str,
    timestamp: Any,
) -> float:
    frame = _predictions_to_frame(predictions)
    if "item_id" not in frame.columns or "timestamp" not in frame.columns:
        raise ValueError("Prediction output missing item_id/timestamp columns")

    row = frame[(frame["item_id"] == item_id) & (frame["timestamp"] == timestamp)]
    if row.empty:
        raise ValueError("Prediction output missing expected item/timestamp row")

    pred_col = _select_prediction_column(list(frame.columns))
    return float(row.iloc[0][pred_col])


def _sample_item_ids(
    item_ids: list[str],
    *,
    max_series: int | None,
    rng: np.random.Generator,
) -> list[str]:
    if max_series is None or max_series >= len(item_ids):
        return item_ids
    selected = rng.choice(item_ids, size=max_series, replace=False)
    return list(selected)


def _sample_indices(
    indices: list[int],
    *,
    max_windows: int | None,
    sample_fraction: float | None,
    strategy: str,
    rng: np.random.Generator,
) -> list[int]:
    selected = indices
    if sample_fraction is not None and sample_fraction < 1.0:
        target_count = max(1, int(len(selected) * sample_fraction))
        if strategy == "contiguous":
            start = int(rng.integers(0, max(1, len(selected) - target_count + 1)))
            selected = selected[start : start + target_count]
        else:
            selected = list(rng.choice(selected, size=target_count, replace=False))
    if max_windows is not None and len(selected) > max_windows:
        if strategy == "contiguous":
            start = int(rng.integers(0, max(1, len(selected) - max_windows + 1)))
            selected = selected[start : start + max_windows]
        else:
            step = max(1, len(selected) // max_windows)
            selected = selected[::step][:max_windows]
    return sorted(set(selected))


def generate_rolling_soft_labels(
    df: _pl.DataFrame | _pd.DataFrame,
    predictor: TimeSeriesPredictor,
    *,
    teacher_config: ChronosTrainingConfig,
    distillation_config: ChronosDistillationConfig,
) -> tuple[_pd.DataFrame, SoftLabelStats]:
    """
    Generate rolling soft labels aligned to forecasted timestamps.

    Parameters
    ----------
    df : pl.DataFrame | pd.DataFrame
        Input dataset containing instrument_id, ts_event, target, and covariates.
    predictor : TimeSeriesPredictor
        Trained AutoGluon predictor used to generate forecasts.
    teacher_config : ChronosTrainingConfig
        Teacher training configuration (for prediction length and data config).
    distillation_config : ChronosDistillationConfig
        Distillation configuration controlling rolling windows and alignment.

    Returns
    -------
    tuple[pd.DataFrame, SoftLabelStats]
        DataFrame of soft labels with columns (item_id, timestamp, soft_target)
        and summary statistics.

    """
    if not HAS_AUTOGLUON:
        check_ml_dependencies(["autogluon"])

    if not HAS_PANDAS or pd is None:
        check_ml_dependencies(["pandas"])
        raise ImportError("Pandas not available")

    data_config = teacher_config.get_data_config()
    if data_config.timestamp_column != "ts_event":
        raise ValueError("Chronos distillation requires ts_event as the timestamp column")

    prediction_length = int(teacher_config.prediction_length)
    if distillation_config.forecast_step > prediction_length:
        raise ValueError(
            "forecast_step cannot exceed prediction_length "
            f"({distillation_config.forecast_step} > {prediction_length})"
        )

    base_df = convert_to_timeseries_pandas(df, data_config)
    base_df = base_df.sort_values(["item_id", "timestamp"]).reset_index(drop=True)

    known_covariates = list(data_config.known_covariates)
    if known_covariates:
        missing = [col for col in known_covariates if col not in base_df.columns]
        if missing:
            raise ValueError(f"Missing known covariate columns: {missing}")

    rng = np.random.default_rng(int(teacher_config.random_seed))
    temperature = float(distillation_config.soft_label_temperature)
    item_ids = _sample_item_ids(
        base_df["item_id"].unique().tolist(),
        max_series=distillation_config.max_series,
        rng=rng,
    )

    soft_rows: list[dict[str, Any]] = []
    total_candidates = 0
    eligible_candidates = 0
    used_series = 0

    for item_id in item_ids:
        series_df = base_df[base_df["item_id"] == item_id].reset_index(drop=True)
        if series_df.empty:
            continue

        max_index = len(series_df) - prediction_length
        if max_index <= distillation_config.min_history:
            logger.info(
                "Skipping %s for distillation (len=%d, min_history=%d, pred_len=%d)",
                item_id,
                len(series_df),
                distillation_config.min_history,
                prediction_length,
            )
            continue

        candidate_indices = list(
            range(
                distillation_config.min_history,
                max_index + 1,
                distillation_config.stride,
            )
        )
        selected_indices = _sample_indices(
            candidate_indices,
            max_windows=distillation_config.max_windows_per_series,
            sample_fraction=distillation_config.sample_fraction,
            strategy=distillation_config.window_sampling_strategy,
            rng=rng,
        )
        if not selected_indices:
            continue
        total_candidates += len(selected_indices)

        used_series += 1
        observed_timestamps = set(series_df["timestamp"].tolist())

        for idx in selected_indices:
            history_df = series_df.iloc[:idx].copy()
            history_tsdf = TimeSeriesDataFrame.from_data_frame(
                history_df,
                id_column="item_id",
                timestamp_column="timestamp",
            )

            if not hasattr(predictor, "make_future_data_frame"):
                raise ValueError("Predictor does not support make_future_data_frame")
            future_index = predictor.make_future_data_frame(history_tsdf)
            future_index_df = _future_index_to_frame(future_index)
            if "item_id" not in future_index_df.columns or "timestamp" not in future_index_df.columns:
                raise ValueError("Future index missing item_id/timestamp columns")

            target_timestamp = future_index_df.iloc[distillation_config.forecast_step - 1][
                "timestamp"
            ]
            if target_timestamp not in observed_timestamps:
                logger.debug(
                    "Skipping distillation window for %s (missing forecast timestamp)",
                    item_id,
                )
                continue
            eligible_candidates += 1

            if known_covariates:
                future_covariates = _build_future_covariates(
                future_index_df,
                series_df=series_df,
                history_df=history_df,
                known_covariates=known_covariates,
            )
                predictions = predictor.predict(
                    history_tsdf,
                    known_covariates=future_covariates,
                )
            else:
                predictions = predictor.predict(history_tsdf)

            soft_value = _extract_prediction(
                predictions,
                item_id=item_id,
                timestamp=target_timestamp,
            )
            if temperature != 1.0:
                soft_value = soft_value / temperature
            soft_rows.append(
                {
                    "item_id": item_id,
                    "timestamp": target_timestamp,
                    distillation_config.soft_target_column: soft_value,
                }
            )

    stats = SoftLabelStats(
        total_candidates=total_candidates,
        eligible_candidates=eligible_candidates,
        generated=len(soft_rows),
        total_series=len(item_ids),
        used_series=used_series,
    )

    if not soft_rows:
        raise ValueError("No soft labels generated; check rolling window configuration")

    labels_df = pd.DataFrame(soft_rows)
    return labels_df, stats


def _merge_soft_labels(
    df: _pl.DataFrame | _pd.DataFrame,
    labels: _pd.DataFrame,
    *,
    data_config: AutoGluonDataConfig,
    soft_target_column: str,
) -> _pl.DataFrame | _pd.DataFrame:
    if not HAS_PANDAS or pd is None:
        check_ml_dependencies(["pandas"])
        raise ImportError("Pandas not available")

    if HAS_POLARS and pl is not None and isinstance(df, pl.DataFrame):
        df_polars = cast(PolarsDataFrame, df)
        ts_dtype_polars = df_polars[data_config.timestamp_column].dtype
        labels_pl = cast(
            PolarsDataFrame,
            pl.from_pandas(labels).rename({"item_id": data_config.item_id_column}),
        )
        if hasattr(ts_dtype_polars, "is_integer") and ts_dtype_polars.is_integer():
            labels_pl = labels_pl.with_columns(
                pl.col("timestamp").dt.epoch("ns").alias(data_config.timestamp_column)
            )
        else:
            if hasattr(ts_dtype_polars, "time_zone") and ts_dtype_polars.time_zone:
                labels_pl = labels_pl.with_columns(
                    pl.col("timestamp")
                    .dt.replace_time_zone(ts_dtype_polars.time_zone)
                    .alias(data_config.timestamp_column)
                )
            else:
                labels_pl = labels_pl.rename({"timestamp": data_config.timestamp_column})

        if data_config.timestamp_column not in labels_pl.columns:
            labels_pl = labels_pl.rename({"timestamp": data_config.timestamp_column})

        joined = df_polars.join(
            labels_pl.select(
                [data_config.item_id_column, data_config.timestamp_column, soft_target_column]
            ),
            on=[data_config.item_id_column, data_config.timestamp_column],
            how="left",
        )
        return joined

    df_pandas = cast(PandasDataFrame, df).copy()
    labels_pandas = labels.rename(columns={"item_id": data_config.item_id_column})
    labels_pandas = labels_pandas.rename(columns={"timestamp": data_config.timestamp_column})
    ts_dtype_pandas = df_pandas[data_config.timestamp_column].dtype
    if str(ts_dtype_pandas).startswith("int"):
        labels_pandas[data_config.timestamp_column] = labels_pandas[
            data_config.timestamp_column
        ].astype("int64")
    elif pd.api.types.is_datetime64tz_dtype(ts_dtype_pandas):
        labels_pandas[data_config.timestamp_column] = pd.to_datetime(
            labels_pandas[data_config.timestamp_column],
            utc=True,
        )
    return df_pandas.merge(
        labels_pandas[[data_config.item_id_column, data_config.timestamp_column, soft_target_column]],
        on=[data_config.item_id_column, data_config.timestamp_column],
        how="left",
    )


def build_distillation_dataset(
    df: _pl.DataFrame | _pd.DataFrame,
    predictor: TimeSeriesPredictor,
    *,
    teacher_config: ChronosTrainingConfig,
    distillation_config: ChronosDistillationConfig,
) -> DistillationDataset:
    """
    Build a distilled training dataset with aligned soft labels.

    Parameters
    ----------
    df : pl.DataFrame | pd.DataFrame
        Input dataset containing instrument_id, ts_event, target, and covariates.
    predictor : TimeSeriesPredictor
        Trained teacher predictor used for generating soft labels.
    teacher_config : ChronosTrainingConfig
        Teacher training configuration (for prediction length and data config).
    distillation_config : ChronosDistillationConfig
        Distillation configuration controlling alignment and blending.

    Returns
    -------
    DistillationDataset
        Distilled dataset with soft targets and summary stats.

    """
    data_config = teacher_config.get_data_config()
    labels_df, stats = generate_rolling_soft_labels(
        df,
        predictor,
        teacher_config=teacher_config,
        distillation_config=distillation_config,
    )

    merged = _merge_soft_labels(
        df,
        labels_df,
        data_config=data_config,
        soft_target_column=distillation_config.soft_target_column,
    )

    if HAS_POLARS and hasattr(merged, "with_columns") and pl is not None:
        merged_polars = cast(PolarsDataFrame, merged)
        soft_col = distillation_config.soft_target_column
        target_col = data_config.target_column
        distilled_col = distillation_config.distilled_target_column
        if distillation_config.label_strategy == "teacher_only":
            merged_polars = merged_polars.filter(pl.col(soft_col).is_not_null())
            merged_polars = merged_polars.with_columns(pl.col(soft_col).alias(distilled_col))
        else:
            if target_col not in merged_polars.columns:
                raise ValueError(f"Missing hard target column: {target_col}")
            merged_polars = merged_polars.filter(
                pl.col(soft_col).is_not_null() & pl.col(target_col).is_not_null()
            )
            merged_polars = merged_polars.with_columns(
                (
                    pl.col(soft_col) * float(distillation_config.distillation_alpha)
                    + pl.col(target_col) * float(1.0 - distillation_config.distillation_alpha)
                ).alias(distilled_col)
            )
        return DistillationDataset(data=merged_polars, labels=labels_df, stats=stats)

    merged_pandas = cast(PandasDataFrame, merged)
    soft_col = distillation_config.soft_target_column
    target_col = data_config.target_column
    distilled_col = distillation_config.distilled_target_column
    if distillation_config.label_strategy == "teacher_only":
        merged_pandas = merged_pandas[merged_pandas[soft_col].notna()].copy()
        merged_pandas[distilled_col] = merged_pandas[soft_col]
    else:
        if target_col not in merged_pandas.columns:
            raise ValueError(f"Missing hard target column: {target_col}")
        merged_pandas = merged_pandas[
            merged_pandas[soft_col].notna() & merged_pandas[target_col].notna()
        ].copy()
        merged_pandas[distilled_col] = (
            merged_pandas[soft_col] * float(distillation_config.distillation_alpha)
            + merged_pandas[target_col] * float(1.0 - distillation_config.distillation_alpha)
        )

    return DistillationDataset(data=merged_pandas, labels=labels_df, stats=stats)
