"""
Missing data auditing utilities for Phase 4 robustness checks.

These helpers analyse sector datasets for null coverage, evaluate alternative
imputation approaches, and quantify the potential impact on downstream metrics.
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import polars as pl
import structlog

from ml.common.metrics_bootstrap import get_gauge
from ml.common.metrics_bootstrap import get_histogram


LOGGER = structlog.get_logger(__name__)

MISSING_DATA_RATIO_GAUGE = get_gauge(
    "phase4_missing_data_ratio",
    "Overall missing data ratio detected during Phase 4 audits.",
    labelnames=("dataset",),
)
IMPUTATION_IMPACT_HIST = get_histogram(
    "phase4_imputation_impact_ratio",
    "Relative return drift introduced by imputation methods during Phase 4 audits.",
    labelnames=("method",),
    buckets=(0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
)


@dataclass(frozen=True)
class ImputationSummary:
    """
    Summary statistics for a specific imputation method.
    """

    method: str
    filled_ratio: float
    mean_shift: float | None
    impact_ratio: float | None
    remaining_missing_ratio: float
    note: str | None = None


@dataclass(frozen=True)
class MissingDataAuditResult:
    """
    Aggregate result of a missing data audit.
    """

    dataset_path: Path
    total_rows: int
    missing_ratio: float
    missing_by_column: Mapping[str, float]
    imputation_summaries: tuple[ImputationSummary, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialise audit result for JSON emission."""
        return {
            "dataset_path": str(self.dataset_path),
            "total_rows": self.total_rows,
            "missing_ratio": self.missing_ratio,
            "missing_by_column": dict(self.missing_by_column),
            "imputation_summaries": [
                {
                    "method": summary.method,
                    "filled_ratio": summary.filled_ratio,
                    "mean_shift": summary.mean_shift,
                    "impact_ratio": summary.impact_ratio,
                    "remaining_missing_ratio": summary.remaining_missing_ratio,
                    "note": summary.note,
                }
                for summary in self.imputation_summaries
            ],
        }


def audit_missing_data(
    dataset_path: Path,
    *,
    methods: Sequence[str] | None = None,
) -> MissingDataAuditResult:
    """
    Audit missing data coverage and evaluate imputation approaches.

    Parameters
    ----------
    dataset_path : Path
        Path to the Parquet/CSV dataset containing sector returns.
    methods : Sequence[str] | None, optional
        Ordered sequence of imputation methods to evaluate. Supported values:
        ``"forward_fill"``, ``"linear"`, ``"kalman"``. When ``None`` the default
        sequence (forward fill, linear, kalman) is used.

    Returns
    -------
    MissingDataAuditResult
        Summary of missing data coverage and imputation effects.
    """
    data = _load_dataset(dataset_path)
    numeric_columns = _numeric_columns(data)
    total_rows = data.height
    if total_rows == 0:
        msg = f"Dataset {dataset_path} is empty"
        raise ValueError(msg)

    missing_counts = {
        column: int(data.get_column(column).is_null().sum())
        for column in data.columns
    }
    total_cells = max(total_rows * len(data.columns), 1)
    missing_ratio = sum(missing_counts.values()) / total_cells
    missing_by_column = {
        column: count / total_rows
        for column, count in missing_counts.items()
    }
    MISSING_DATA_RATIO_GAUGE.labels(dataset=str(dataset_path)).set(missing_ratio)

    value_column = "return" if "return" in data.columns else numeric_columns[0] if numeric_columns else data.columns[0]
    baseline_series = data.get_column(value_column).drop_nulls()
    baseline_mean_value = baseline_series.mean() if not baseline_series.is_empty() else None
    baseline_mean = float(cast(float, baseline_mean_value)) if baseline_mean_value is not None else 0.0
    baseline_std_value = baseline_series.std() if baseline_series.len() > 1 else None
    baseline_std = float(cast(float, baseline_std_value)) if baseline_std_value is not None else 0.0

    method_sequence = methods if methods is not None else ("forward_fill", "linear", "kalman")
    summaries: list[ImputationSummary] = []
    for method in method_sequence:
        method_lower = method.strip().lower()
        if method_lower not in {"forward_fill", "linear", "kalman"}:
            LOGGER.warning("Unsupported imputation method requested", method=method)
            continue
        summary = _evaluate_imputation_method(
            method=method_lower,
            frame=data,
            numeric_columns=numeric_columns,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            total_cells=total_cells,
            initial_missing=sum(missing_counts.values()),
        )
        summaries.append(summary)

    return MissingDataAuditResult(
        dataset_path=dataset_path,
        total_rows=total_rows,
        missing_ratio=missing_ratio,
        missing_by_column=missing_by_column,
        imputation_summaries=tuple(summaries),
    )


def _load_dataset(dataset_path: Path) -> pl.DataFrame:
    """Load dataset from supported formats and ensure timestamp ordering."""
    if not dataset_path.exists():
        msg = f"Dataset path does not exist: {dataset_path}"
        raise FileNotFoundError(msg)
    if dataset_path.suffix == ".parquet":
        frame = pl.read_parquet(dataset_path)
    elif dataset_path.suffix == ".csv":
        frame = pl.read_csv(dataset_path)
    else:
        msg = f"Unsupported dataset extension: {dataset_path.suffix}"
        raise ValueError(msg)
    if "timestamp" in frame.columns:
        if frame.get_column("timestamp").dtype != pl.Datetime:
            try:
                frame = frame.with_columns(pl.col("timestamp").str.to_datetime(time_zone="UTC"))
            except Exception:
                LOGGER.warning("Failed to parse timestamps via to_datetime; attempting strptime", exc_info=True)
                frame = frame.with_columns(
                    pl.col("timestamp").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False).dt.replace_time_zone("UTC"),
                )
        frame = frame.sort("timestamp")
    return frame


def _numeric_columns(frame: pl.DataFrame) -> list[str]:
    """Return numeric columns eligible for imputation."""
    numeric_columns: list[str] = []
    for column in frame.columns:
        dtype = frame.get_column(column).dtype
        if dtype.is_numeric():
            numeric_columns.append(column)
    if not numeric_columns:
        msg = "Dataset contains no numeric columns to audit for missing data"
        raise ValueError(msg)
    return numeric_columns


def _evaluate_imputation_method(
    *,
    method: str,
    frame: pl.DataFrame,
    numeric_columns: Iterable[str],
    baseline_mean: float,
    baseline_std: float,
    total_cells: int,
    initial_missing: int,
) -> ImputationSummary:
    """Apply an imputation method and compute summary statistics."""
    if method == "kalman":
        LOGGER.info("Skipping Kalman imputation - backend not available")
        return ImputationSummary(
            method="kalman",
            filled_ratio=0.0,
            mean_shift=None,
            impact_ratio=None,
            remaining_missing_ratio=initial_missing / total_cells if total_cells else 0.0,
            note="Kalman smoothing not available in lightweight environment.",
        )

    if method == "forward_fill":
        filled = _apply_forward_fill(frame, numeric_columns)
    elif method == "linear":
        filled = _apply_linear_interpolation(frame, numeric_columns)
    else:
        msg = f"Unsupported imputation method: {method}"
        raise ValueError(msg)

    post_missing = sum(int(filled.get_column(column).is_null().sum()) for column in filled.columns)
    filled_ratio = (
        (initial_missing - post_missing) / initial_missing
        if initial_missing > 0
        else 0.0
    )

    series = filled.get_column("return" if "return" in filled.columns else next(iter(numeric_columns)))
    series_filled = series.drop_nulls()
    filled_mean_value = series_filled.mean() if not series_filled.is_empty() else None
    filled_mean = float(cast(float, filled_mean_value)) if filled_mean_value is not None else baseline_mean
    mean_shift = abs(filled_mean - baseline_mean)
    impact_ratio = None
    if baseline_std > 0.0:
        impact_ratio = mean_shift / baseline_std
        IMPUTATION_IMPACT_HIST.labels(method=method).observe(impact_ratio)

    remaining_missing_ratio = post_missing / total_cells if total_cells else 0.0

    return ImputationSummary(
        method=method,
        filled_ratio=filled_ratio,
        mean_shift=mean_shift,
        impact_ratio=impact_ratio,
        remaining_missing_ratio=remaining_missing_ratio,
    )


def _apply_forward_fill(frame: pl.DataFrame, numeric_columns: Iterable[str]) -> pl.DataFrame:
    """Apply forward-fill imputation to numeric columns."""
    columns = list(numeric_columns)
    working = frame.sort("timestamp") if "timestamp" in frame.columns else frame.clone()
    if not columns:
        return working
    return working.with_columns([pl.col(column).fill_null(strategy="forward") for column in columns])


def _apply_linear_interpolation(frame: pl.DataFrame, numeric_columns: Iterable[str]) -> pl.DataFrame:
    """Apply linear interpolation to numeric columns."""
    columns = list(numeric_columns)
    working = frame.sort("timestamp") if "timestamp" in frame.columns else frame.clone()
    if not columns:
        return working
    return working.with_columns([pl.col(column).interpolate() for column in columns])
