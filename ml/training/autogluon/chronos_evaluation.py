"""
Time-split evaluation utilities for Chronos training runs.

This module provides cold-path helpers to split datasets by timestamp, compute naive
baselines, and (optionally) train/evaluate Chronos models.

"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ml._imports import HAS_AUTOGLUON
from ml._imports import HAS_POLARS
from ml._imports import check_ml_dependencies
from ml._imports import pl
from ml.common.validation_strategies import require_holdout_strategy
from ml.config.autogluon import ChronosBaselineStrategy
from ml.config.autogluon import ChronosEvaluationConfig
from ml.config.autogluon import ChronosTrainingConfig
from ml.data.autogluon_adapter import convert_to_timeseries_dataframe
from ml.data.feature_columns import split_feature_columns
from ml.ml_types import PolarsDF
from ml.training.autogluon.chronos_trainer import ChronosTrainer
from ml.training.common.evaluation import calculate_regression_metrics


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChronosSplitBoundaries:
    """
    Boundary timestamps for train/val/test splits.

    Attributes
    ----------
    train_end_ts : int
        Last timestamp (inclusive) for the training split.
    val_end_ts : int
        Last timestamp (inclusive) for the validation split.

    """

    train_end_ts: int
    val_end_ts: int


@dataclass(frozen=True)
class ChronosSplitResult:
    """
    Output of a time-based split for evaluation.

    Attributes
    ----------
    train : pl.DataFrame
        Training split.
    val : pl.DataFrame
        Validation split.
    test : pl.DataFrame
        Test split.
    boundaries : ChronosSplitBoundaries
        Boundary timestamps for the split.
    row_counts : dict[str, int]
        Row counts for each split.

    """

    train: PolarsDF
    val: PolarsDF
    test: PolarsDF
    boundaries: ChronosSplitBoundaries
    row_counts: dict[str, int]


@dataclass(frozen=True)
class ChronosSeriesCoverageResult:
    """
    Result of filtering series to ensure split coverage.

    Attributes
    ----------
    split : ChronosSplitResult
        Filtered split result.
    enabled : bool
        Whether per-series filtering was applied.
    min_rows_per_split : int
        Minimum rows required per series in each split.
    total_series_before : int
        Number of series before filtering.
    total_series_after : int
        Number of series after filtering.
    dropped_series : tuple[str, ...]
        Series identifiers removed due to insufficient split coverage.

    """

    split: ChronosSplitResult
    enabled: bool
    min_rows_per_split: int
    total_series_before: int
    total_series_after: int
    dropped_series: tuple[str, ...]


@dataclass(frozen=True)
class ChronosSanitizationResult:
    """
    Result of sanitizing Chronos input frames.

    Attributes
    ----------
    frame : pl.DataFrame
        Sanitized dataset frame.
    dropped_non_numeric : tuple[str, ...]
        Non-numeric columns removed from the dataset.
    dropped_excluded : tuple[str, ...]
        Columns removed because they matched exclusion rules.
    dropped_constant : tuple[str, ...]
        Numeric columns removed because they are constant.

    """

    frame: PolarsDF
    dropped_non_numeric: tuple[str, ...]
    dropped_excluded: tuple[str, ...]
    dropped_constant: tuple[str, ...]


@dataclass(frozen=True)
class ChronosMarketHoursFilterResult:
    """
    Result of filtering to regular trading minutes.

    Attributes
    ----------
    frame : pl.DataFrame
        Filtered dataset frame.
    rows_before : int
        Row count before filtering.
    rows_after : int
        Row count after filtering.
    filtered_rows : int
        Number of rows removed by the filter.
    enabled : bool
        Whether filtering was applied.

    """

    frame: PolarsDF
    rows_before: int
    rows_after: int
    filtered_rows: int
    enabled: bool


def sanitize_chronos_frame(
    df: PolarsDF,
    config: ChronosEvaluationConfig,
) -> ChronosSanitizationResult:
    """
    Sanitize a dataset by dropping non-numeric or non-informative features.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataset frame.
    config : ChronosEvaluationConfig
        Evaluation configuration controlling sanitization behavior.

    Returns
    -------
    ChronosSanitizationResult
        Sanitized dataset and metadata about dropped columns.

    """
    _require_polars()
    if pl is None:
        raise RuntimeError("Polars dependency unavailable")

    numeric_cols, non_numeric_cols = split_feature_columns(
        df,
        exclude=config.feature_exclude_columns,
        exclude_suffixes=config.feature_exclude_suffixes,
    )

    required_columns = {
        config.timestamp_column,
        config.item_id_column,
        config.target_column,
    }
    keep_columns = set(required_columns) | set(numeric_cols)
    excluded_columns = [
        name
        for name in df.columns
        if name not in required_columns
        and name not in numeric_cols
        and name not in non_numeric_cols
    ]
    dropped_non_numeric: list[str] = []
    if config.drop_non_numeric_features:
        dropped_non_numeric = [name for name in non_numeric_cols if name in df.columns]
    else:
        keep_columns |= set(non_numeric_cols)

    sanitized = df.select([pl.col(name) for name in df.columns if name in keep_columns])

    dropped_constant: list[str] = []
    if config.drop_constant_features and numeric_cols:
        candidates = [name for name in numeric_cols if name in sanitized.columns]
        if candidates:
            unique_counts = sanitized.select(
                [pl.col(name).n_unique().alias(name) for name in candidates],
            )
            dropped_constant = [name for name in candidates if int(unique_counts[name][0]) <= 1]
            if dropped_constant:
                sanitized = sanitized.drop(dropped_constant)

    return ChronosSanitizationResult(
        frame=sanitized,
        dropped_non_numeric=tuple(sorted(dropped_non_numeric)),
        dropped_excluded=tuple(sorted(excluded_columns)),
        dropped_constant=tuple(sorted(dropped_constant)),
    )


def _market_hours_value_literal(
    dtype: Any,
    value: bool,
    column: str,
) -> bool | int | float:
    if pl is None:
        raise RuntimeError("Polars dependency unavailable")

    bool_value = bool(value)
    int_dtypes = (
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    )
    float_dtypes = (pl.Float32, pl.Float64)

    if dtype in int_dtypes:
        return int(bool_value)
    if dtype in float_dtypes:
        return float(int(bool_value))
    if dtype == pl.Boolean:
        return bool_value

    raise ValueError(
        f"market_hours_column '{column}' must be boolean or numeric, got {dtype}",
    )


def _filter_market_hours_frame(
    df: PolarsDF,
    config: ChronosEvaluationConfig,
) -> ChronosMarketHoursFilterResult:
    """
    Filter datasets to regular trading minutes when configured.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataset frame.
    config : ChronosEvaluationConfig
        Evaluation configuration controlling filtering.

    Returns
    -------
    ChronosMarketHoursFilterResult
        Filtered dataset and row count metadata.

    """
    _require_polars()
    if pl is None:
        raise RuntimeError("Polars dependency unavailable")

    rows_before = int(df.height)
    if not config.filter_market_hours:
        return ChronosMarketHoursFilterResult(
            frame=df,
            rows_before=rows_before,
            rows_after=rows_before,
            filtered_rows=0,
            enabled=False,
        )

    column = config.market_hours_column
    if column not in df.columns:
        raise ValueError(f"Missing market hours column: {column}")

    dtype = df.schema.get(column)
    if dtype is None:
        raise ValueError(f"Missing market hours column: {column}")

    value_literal = _market_hours_value_literal(
        dtype,
        config.market_hours_value,
        column,
    )

    filtered = df.filter(pl.col(column) == pl.lit(value_literal))
    rows_after = int(filtered.height)
    if rows_after == 0:
        raise ValueError(
            f"Market hours filter removed all rows using column '{column}'",
        )

    return ChronosMarketHoursFilterResult(
        frame=filtered,
        rows_before=rows_before,
        rows_after=rows_after,
        filtered_rows=rows_before - rows_after,
        enabled=True,
    )


def _split_time_window_frame(
    df: PolarsDF,
    config: ChronosEvaluationConfig,
) -> ChronosSplitResult:
    if pl is None:
        raise RuntimeError("Polars dependency unavailable")
    timestamp_col = config.timestamp_column
    item_id_col = config.item_id_column

    ts_series = df.select(pl.col(timestamp_col).cast(pl.Int64).unique().sort()).to_series()
    if ts_series.len() < 3:
        raise ValueError("Need at least 3 unique timestamps to split train/val/test")

    n_timestamps = int(ts_series.len())
    train_count = max(1, int(n_timestamps * float(config.train_fraction)))
    val_count = max(1, int(n_timestamps * float(config.val_fraction)))

    if train_count + val_count >= n_timestamps:
        val_count = max(1, n_timestamps - train_count - 1)

    if train_count + val_count >= n_timestamps:
        raise ValueError("Not enough timestamps to create train/val/test splits")

    train_end_idx = train_count - 1
    val_end_idx = train_count + val_count - 1

    train_end_ts = int(ts_series[train_end_idx])
    val_end_ts = int(ts_series[val_end_idx])

    ts_expr = pl.col(timestamp_col).cast(pl.Int64)
    train_df = df.filter(ts_expr <= train_end_ts)
    val_df = df.filter((ts_expr > train_end_ts) & (ts_expr <= val_end_ts))
    test_df = df.filter(ts_expr > val_end_ts)

    sort_cols = [item_id_col, timestamp_col]
    train_df = train_df.sort(sort_cols)
    val_df = val_df.sort(sort_cols)
    test_df = test_df.sort(sort_cols)

    row_counts = {
        "train": int(train_df.height),
        "val": int(val_df.height),
        "test": int(test_df.height),
    }
    for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        if split_df.height < int(config.min_rows_per_split):
            raise ValueError(
                f"{split_name} split too small: {split_df.height} rows < "
                f"{int(config.min_rows_per_split)}",
            )

    boundaries = ChronosSplitBoundaries(
        train_end_ts=train_end_ts,
        val_end_ts=val_end_ts,
    )

    return ChronosSplitResult(
        train=train_df,
        val=val_df,
        test=test_df,
        boundaries=boundaries,
        row_counts=row_counts,
    )


def _split_purged_frame(
    df: PolarsDF,
    config: ChronosEvaluationConfig,
) -> ChronosSplitResult:
    from ml.preprocessing.stationarity import PurgedCrossValidator

    if pl is None:
        raise RuntimeError("Polars dependency unavailable")
    timestamp_col = config.timestamp_column
    item_id_col = config.item_id_column

    ts_series = df.select(pl.col(timestamp_col).cast(pl.Int64).unique().sort()).to_series()
    if ts_series.len() < 3:
        raise ValueError("Need at least 3 unique timestamps to split train/val/test")

    timestamps = ts_series.to_numpy().astype(np.int64)
    n_timestamps = int(timestamps.shape[0])
    test_count = max(1, int(n_timestamps * float(config.test_fraction)))
    if test_count >= n_timestamps:
        raise ValueError("test_fraction leaves no timestamps for training")

    train_ts_len = n_timestamps - test_count
    if train_ts_len < 2:
        raise ValueError("Not enough timestamps for purged validation")
    if int(config.cv_splits) < 2:
        raise ValueError("cv_splits must be >= 2 for purged validation")
    if train_ts_len // int(config.cv_splits) < 1:
        raise ValueError("Not enough timestamps for requested purged splits")

    cv = PurgedCrossValidator(
        n_splits=int(config.cv_splits),
        purge_gap=int(config.purge_gap),
        embargo_pct=float(config.embargo_pct),
    )
    splits = cv.split(np.arange(train_ts_len).reshape(-1, 1))
    if not splits:
        raise ValueError("Purged CV produced no splits for Chronos evaluation")

    train_idx, val_idx = splits[-1]
    train_ts = timestamps[train_idx]
    val_ts = timestamps[val_idx]
    test_ts = timestamps[train_ts_len:]
    if train_ts.size == 0 or val_ts.size == 0 or test_ts.size == 0:
        raise ValueError("Purged split produced empty train/val/test timestamps")

    ts_expr = pl.col(timestamp_col).cast(pl.Int64)
    train_df = df.filter(ts_expr.is_in(train_ts))
    val_df = df.filter(ts_expr.is_in(val_ts))
    test_df = df.filter(ts_expr.is_in(test_ts))

    sort_cols = [item_id_col, timestamp_col]
    train_df = train_df.sort(sort_cols)
    val_df = val_df.sort(sort_cols)
    test_df = test_df.sort(sort_cols)

    row_counts = {
        "train": int(train_df.height),
        "val": int(val_df.height),
        "test": int(test_df.height),
    }
    for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        if split_df.height < int(config.min_rows_per_split):
            raise ValueError(
                f"{split_name} split too small: {split_df.height} rows < "
                f"{int(config.min_rows_per_split)}",
            )

    train_end_ts = int(train_ts.max())
    val_end_ts = int(val_ts.max())
    boundaries = ChronosSplitBoundaries(
        train_end_ts=train_end_ts,
        val_end_ts=val_end_ts,
    )

    return ChronosSplitResult(
        train=train_df,
        val=val_df,
        test=test_df,
        boundaries=boundaries,
        row_counts=row_counts,
    )


def split_time_series_frame(
    df: PolarsDF,
    config: ChronosEvaluationConfig,
) -> ChronosSplitResult:
    """
    Split a dataset into train/val/test partitions by timestamp.

    Uses ``ChronosEvaluationConfig.validation_strategy`` to select either
    time-window splits or purged validation splits.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataset containing timestamp, item id, and target columns.
    config : ChronosEvaluationConfig
        Evaluation configuration.

    Returns
    -------
    ChronosSplitResult
        Train/val/test splits with boundary metadata.

    Raises
    ------
    ValueError
        If required columns are missing or the split is infeasible.

    """
    _require_polars()
    if pl is None:
        raise RuntimeError("Polars dependency unavailable")

    timestamp_col = config.timestamp_column
    item_id_col = config.item_id_column

    if timestamp_col not in df.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")
    if item_id_col not in df.columns:
        raise ValueError(f"Missing item id column: {item_id_col}")

    strategy = require_holdout_strategy(str(config.validation_strategy))
    if strategy == "time_window":
        return _split_time_window_frame(df, config)
    if strategy == "purged":
        return _split_purged_frame(df, config)

    raise ValueError(f"Unsupported validation_strategy '{strategy}'")


def _filter_split_by_series_coverage(
    split: ChronosSplitResult,
    config: ChronosEvaluationConfig,
) -> ChronosSeriesCoverageResult:
    _require_polars()
    if pl is None:
        raise RuntimeError("Polars dependency unavailable")

    min_rows = int(config.min_rows_per_series_split)
    item_id_col = config.item_id_column

    series_union = pl.concat(
        [
            split.train.select(pl.col(item_id_col)),
            split.val.select(pl.col(item_id_col)),
            split.test.select(pl.col(item_id_col)),
        ],
        how="vertical",
    ).unique()
    total_series_before = int(series_union.height)

    if min_rows <= 0 or total_series_before == 0:
        return ChronosSeriesCoverageResult(
            split=split,
            enabled=False,
            min_rows_per_split=min_rows,
            total_series_before=total_series_before,
            total_series_after=total_series_before,
            dropped_series=(),
        )

    train_counts = split.train.group_by(item_id_col).len().rename({"len": "train_rows"})
    val_counts = split.val.group_by(item_id_col).len().rename({"len": "val_rows"})
    test_counts = split.test.group_by(item_id_col).len().rename({"len": "test_rows"})

    counts = (
        series_union.join(train_counts, on=item_id_col, how="left")
        .join(val_counts, on=item_id_col, how="left")
        .join(test_counts, on=item_id_col, how="left")
        .fill_null(0)
    )

    meets_requirement = (
        (pl.col("train_rows") >= min_rows)
        & (pl.col("val_rows") >= min_rows)
        & (pl.col("test_rows") >= min_rows)
    )
    kept_series = counts.filter(meets_requirement).get_column(item_id_col).to_list()
    dropped_series = counts.filter(~meets_requirement).get_column(item_id_col).to_list()

    if not kept_series:
        raise ValueError(
            "Series coverage filter removed all series; adjust min_rows_per_series_split",
        )

    filtered_train = split.train.filter(pl.col(item_id_col).is_in(kept_series))
    filtered_val = split.val.filter(pl.col(item_id_col).is_in(kept_series))
    filtered_test = split.test.filter(pl.col(item_id_col).is_in(kept_series))

    row_counts = {
        "train": int(filtered_train.height),
        "val": int(filtered_val.height),
        "test": int(filtered_test.height),
    }
    for split_name, split_df in (
        ("train", filtered_train),
        ("val", filtered_val),
        ("test", filtered_test),
    ):
        if split_df.height < int(config.min_rows_per_split):
            raise ValueError(
                f"{split_name} split too small: {split_df.height} rows < "
                f"{int(config.min_rows_per_split)}",
            )

    total_series_after = len(kept_series)
    normalized_dropped = tuple(sorted(str(value) for value in dropped_series))

    filtered_split = ChronosSplitResult(
        train=filtered_train,
        val=filtered_val,
        test=filtered_test,
        boundaries=split.boundaries,
        row_counts=row_counts,
    )

    return ChronosSeriesCoverageResult(
        split=filtered_split,
        enabled=True,
        min_rows_per_split=min_rows,
        total_series_before=total_series_before,
        total_series_after=total_series_after,
        dropped_series=normalized_dropped,
    )


def evaluate_baseline(
    train_df: PolarsDF,
    eval_df: PolarsDF,
    config: ChronosEvaluationConfig,
) -> dict[str, float]:
    """
    Evaluate a naive baseline on an evaluation split.

    Parameters
    ----------
    train_df : pl.DataFrame
        Training split used to derive baseline statistics.
    eval_df : pl.DataFrame
        Evaluation split to score.
    config : ChronosEvaluationConfig
        Evaluation configuration.

    Returns
    -------
    dict[str, float]
        Baseline regression metrics (mse, rmse, mae).

    """
    _require_polars()
    if pl is None:
        raise RuntimeError("Polars dependency unavailable")

    target_col = config.target_column
    item_id_col = config.item_id_column

    if target_col not in train_df.columns:
        raise ValueError(f"Missing target column in train split: {target_col}")
    if target_col not in eval_df.columns:
        raise ValueError(f"Missing target column in eval split: {target_col}")
    if item_id_col not in train_df.columns:
        raise ValueError(f"Missing item id column in train split: {item_id_col}")
    if item_id_col not in eval_df.columns:
        raise ValueError(f"Missing item id column in eval split: {item_id_col}")

    cleaned_train = train_df.filter(pl.col(target_col).is_not_null())
    cleaned_eval = eval_df.filter(pl.col(target_col).is_not_null())
    if cleaned_eval.is_empty():
        raise ValueError("Evaluation split contains no target values")

    preds = _baseline_predictions(
        cleaned_train,
        cleaned_eval,
        strategy=config.baseline_strategy,
        item_id_col=item_id_col,
        target_col=target_col,
    )

    y_true = cleaned_eval.select(pl.col(target_col)).to_numpy().reshape(-1).astype(np.float64)
    y_pred = np.asarray(preds, dtype=np.float64).reshape(-1)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("Baseline predictions misaligned with evaluation targets")

    return calculate_regression_metrics(y_true, y_pred)


def run_chronos_time_split_evaluation(
    df: PolarsDF,
    *,
    eval_config: ChronosEvaluationConfig,
    training_config: ChronosTrainingConfig | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Run time-split evaluation with baselines and optional Chronos training.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataset with target and timestamp columns.
    eval_config : ChronosEvaluationConfig
        Evaluation configuration.
    training_config : ChronosTrainingConfig | None, optional
        Chronos training configuration. If None, Chronos training is skipped.
    output_dir : Path | None, optional
        Directory to write the evaluation report. If None, no report is written.

    Returns
    -------
    dict[str, Any]
        Evaluation report payload.

    """
    filtering = _filter_market_hours_frame(df, eval_config)
    sanitization = sanitize_chronos_frame(filtering.frame, eval_config)
    split = split_time_series_frame(sanitization.frame, eval_config)
    series_coverage = _filter_split_by_series_coverage(split, eval_config)
    split = series_coverage.split

    baseline_val = evaluate_baseline(split.train, split.val, eval_config)
    baseline_test = evaluate_baseline(split.train, split.test, eval_config)
    # TODO(chronos-eval): add baseline-improvement gating once thresholds are config-driven.

    chronos_payload: dict[str, Any] | None = None
    if training_config is not None:
        chronos_payload = _evaluate_chronos(split, training_config)

    report: dict[str, Any] = {
        "split": {
            "boundaries": {
                "train_end_ts": split.boundaries.train_end_ts,
                "val_end_ts": split.boundaries.val_end_ts,
            },
            "row_counts": split.row_counts,
        },
        "series_filtering": {
            "enabled": series_coverage.enabled,
            "min_rows_per_split": series_coverage.min_rows_per_split,
            "total_series_before": series_coverage.total_series_before,
            "total_series_after": series_coverage.total_series_after,
            "dropped_series": series_coverage.dropped_series,
        },
        "filtering": {
            "enabled": eval_config.filter_market_hours,
            "column": eval_config.market_hours_column,
            "value": eval_config.market_hours_value,
            "rows_before": filtering.rows_before,
            "rows_after": filtering.rows_after,
            "filtered_rows": filtering.filtered_rows,
        },
        "sanitization": {
            "dropped_non_numeric": sanitization.dropped_non_numeric,
            "dropped_excluded": sanitization.dropped_excluded,
            "dropped_constant": sanitization.dropped_constant,
            "kept_columns": len(sanitization.frame.columns),
        },
        "baseline": {
            "val": _coerce_metrics(baseline_val),
            "test": _coerce_metrics(baseline_test),
        },
        "chronos": chronos_payload,
        "config": {
            "evaluation": eval_config.json_primitives(),
            "training": training_config.json_primitives() if training_config else None,
        },
    }

    if output_dir is not None:
        report_path = _write_report(report, output_dir, eval_config)
        report["report_path"] = str(report_path)

    if filtering.enabled:
        logger.info(
            "Filtered market hours from %s to %s rows using %s=%s",
            filtering.rows_before,
            filtering.rows_after,
            eval_config.market_hours_column,
            eval_config.market_hours_value,
        )
    if series_coverage.enabled:
        if series_coverage.dropped_series:
            logger.info(
                "Dropped %s series without split coverage (min_rows_per_series_split=%s): %s",
                len(series_coverage.dropped_series),
                series_coverage.min_rows_per_split,
                series_coverage.dropped_series,
            )
        else:
            logger.info(
                "All %s series meet split coverage (min_rows_per_series_split=%s)",
                series_coverage.total_series_after,
                series_coverage.min_rows_per_split,
            )
    logger.info(
        "Baseline metrics (val/test): %s / %s",
        report["baseline"]["val"],
        report["baseline"]["test"],
    )
    if chronos_payload is not None:
        logger.info(
            "Chronos metrics (val/test): %s / %s",
            chronos_payload.get("val"),
            chronos_payload.get("test"),
        )

    return report


def _baseline_predictions(
    train_df: PolarsDF,
    eval_df: PolarsDF,
    *,
    strategy: ChronosBaselineStrategy,
    item_id_col: str,
    target_col: str,
) -> npt.NDArray[np.float64]:
    if pl is None:
        raise RuntimeError("Polars dependency unavailable")

    if strategy == "global_mean":
        global_mean = float(train_df.select(pl.col(target_col).mean()).item())
        return np.full(eval_df.height, global_mean, dtype=np.float64)

    if strategy == "per_item_mean":
        per_item = train_df.group_by(item_id_col).agg(pl.col(target_col).mean().alias("_baseline"))
        global_mean = float(train_df.select(pl.col(target_col).mean()).item())
        if hasattr(eval_df, "with_row_index"):
            eval_indexed = eval_df.with_row_index("_row_id")
        else:
            eval_indexed = eval_df.with_row_count("_row_id")
        joined = eval_indexed.join(per_item, on=item_id_col, how="left").sort("_row_id")
        preds = joined.get_column("_baseline").fill_null(global_mean)
        return preds.to_numpy().astype(np.float64)

    raise ValueError(f"Unsupported baseline strategy: {strategy}")


def _evaluate_chronos(
    split: ChronosSplitResult,
    training_config: ChronosTrainingConfig,
) -> dict[str, Any]:
    if not HAS_AUTOGLUON:
        check_ml_dependencies(["autogluon"])
        raise ImportError("AutoGluon is required for Chronos evaluation")

    data_config = training_config.get_data_config()
    _validate_column_alignment(data_config, split)

    trainer = ChronosTrainer(training_config)
    train_result = trainer.train(split.train, validation_data=split.val)
    trainer.persist()

    val_metrics = _evaluate_predictor(trainer, split.val, training_config)
    test_metrics = _evaluate_predictor(trainer, split.test, training_config)

    return {
        "training": _normalize_metric_payload(
            _coerce_payload(train_result.get("metrics", {})),
            eval_metric=training_config.eval_metric,
        ),
        "val": _coerce_metrics(val_metrics),
        "test": _coerce_metrics(test_metrics),
    }


def _evaluate_predictor(
    trainer: ChronosTrainer,
    eval_df: PolarsDF,
    training_config: ChronosTrainingConfig,
) -> dict[str, float]:
    if trainer.predictor is None:
        raise ValueError("Chronos predictor unavailable for evaluation")

    tsdf = convert_to_timeseries_dataframe(eval_df, training_config)
    try:
        metrics = trainer.predictor.evaluate(tsdf)
    except Exception as exc:
        logger.warning("Chronos evaluation failed: %s", exc, exc_info=True)
        return {}

    if not isinstance(metrics, Mapping):
        return {}

    normalized = {str(key): float(value) for key, value in metrics.items()}
    return _normalize_metric_values(normalized, eval_metric=training_config.eval_metric)


def _normalize_metric_values(
    metrics: dict[str, float],
    *,
    eval_metric: str,
) -> dict[str, float]:
    metric_key = _match_metric_key(metrics, eval_metric)
    if metric_key is None:
        return dict(metrics)
    value = metrics.get(metric_key)
    if isinstance(value, (int, float)) and value < 0:
        updated = dict(metrics)
        updated[metric_key] = abs(float(value))
        return updated
    return dict(metrics)


def _normalize_metric_payload(
    metrics: Mapping[str, Any],
    *,
    eval_metric: str,
) -> dict[str, Any]:
    metric_key = _match_metric_key(metrics, eval_metric)
    if metric_key is None:
        return dict(metrics)
    value = metrics.get(metric_key)
    if isinstance(value, (int, float)) and value < 0:
        updated = dict(metrics)
        updated[metric_key] = abs(float(value))
        return updated
    return dict(metrics)


def _match_metric_key(metrics: Mapping[str, Any], metric_name: str) -> str | None:
    target = metric_name.strip().upper()
    for key in metrics:
        if str(key).upper() == target:
            return str(key)
    return None


def _validate_column_alignment(
    data_config: Any,
    split: ChronosSplitResult,
) -> None:
    timestamp_col = data_config.timestamp_column
    item_id_col = data_config.item_id_column
    target_col = data_config.target_column
    for split_name, split_df in (("train", split.train), ("val", split.val), ("test", split.test)):
        missing = [
            col for col in (timestamp_col, item_id_col, target_col) if col not in split_df.columns
        ]
        if missing:
            raise ValueError(
                f"{split_name} split missing required columns: {missing}",
            )


def _coerce_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    return {str(key): float(value) for key, value in metrics.items()}


def _coerce_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (np.integer, np.floating)):
            cleaned[str(key)] = float(value)
        elif isinstance(value, (int, float, str, bool)) or value is None:
            cleaned[str(key)] = value
        else:
            cleaned[str(key)] = str(value)
    return cleaned


def _write_report(
    report: dict[str, Any],
    output_dir: Path,
    config: ChronosEvaluationConfig,
) -> Path:
    report_dir = output_dir / config.report_dir_name
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / config.report_filename
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("Evaluation report written to %s", report_path)
    return report_path


def _require_polars() -> None:
    if not HAS_POLARS or pl is None:
        check_ml_dependencies(["polars"])
        raise ImportError("Polars is required for Chronos evaluation utilities")


__all__ = [
    "ChronosSanitizationResult",
    "ChronosSplitBoundaries",
    "ChronosSplitResult",
    "evaluate_baseline",
    "run_chronos_time_split_evaluation",
    "sanitize_chronos_frame",
    "split_time_series_frame",
]
