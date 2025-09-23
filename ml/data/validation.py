"""Dataset validation utilities for TFT builds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram
from ml.data.vintage import VintagePolicy


pl: Any
try:
    import polars as pl
except ImportError:  # pragma: no cover - optional dependency
    pl = None

pd: Any
try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None


logger = structlog.get_logger(__name__)

_VALIDATION_COUNTER = get_counter(
    "ml_dataset_validation_total",
    "Dataset validation attempts",
    ["status"],
)
_VALIDATION_SECONDS = get_histogram(
    "ml_dataset_validation_seconds",
    "Dataset validation durations",
    ["status"],
)


class DatasetValidationError(RuntimeError):
    """Raised when a dataset fails validation checks."""


@dataclass(frozen=True)
class DatasetValidationConfig:
    """Configuration for dataset validation rules."""

    min_rows: int = 1
    min_positive_rate: float | None = 0.001
    max_positive_rate: float | None = 0.999
    min_feature_coverage: float | None = 0.9
    require_macro_series: tuple[str, ...] | None = None
    expected_vintage_policy: VintagePolicy | None = None
    macro_min_vintage_observations: int | None = None


@dataclass(frozen=True)
class DatasetValidationResult:
    """Summary statistics collected during validation."""

    row_count: int
    positive_rate: float | None
    feature_coverage: dict[str, float]
    macro_columns_present: tuple[str, ...]
    macro_observation_counts: dict[str, int]


def _as_polars(df: Any) -> tuple[Any, bool]:
    if pl is not None and isinstance(df, pl.DataFrame):
        return df, True
    if pd is not None and isinstance(df, pd.DataFrame):
        if pl is not None:
            return pl.from_pandas(df), True
        return df, False
    return df, False


def _infer_feature_columns(df: Any) -> list[str]:
    exclude = {"y", "time_index", "timestamp", "instrument_id", "ts_event"}
    if pl is not None and isinstance(df, pl.DataFrame):
        return [name for name in df.columns if name not in exclude]
    if pd is not None and isinstance(df, pd.DataFrame):
        return [name for name in df.columns if name not in exclude]
    return []


def _macro_columns(df: Any) -> list[str]:
    if pl is not None and isinstance(df, pl.DataFrame):
        return [c for c in df.columns if c.isupper() and len(c) >= 3]
    if pd is not None and isinstance(df, pd.DataFrame):
        return [c for c in df.columns if c.isupper() and len(c) >= 3]
    return []


def validate_dataset(
    df_any: Any,
    *,
    config: DatasetValidationConfig,
) -> DatasetValidationResult:
    """Validate a dataset against the supplied configuration."""
    import time

    start = time.perf_counter()
    df, is_polars = _as_polars(df_any)
    status = "error"
    try:
        row_count = int(df.height if is_polars else len(df))
        if row_count < config.min_rows:
            msg = f"Dataset has {row_count} rows; minimum required is {config.min_rows}"
            raise DatasetValidationError(msg)

        positives: float | None = None
        positive_rate: float | None = None
        if "y" in (df.columns if is_polars else df_any.columns):
            if is_polars:
                positives = float(df.select(pl.col("y").sum().alias("p")).item())
            else:
                positives = float(np.nansum(np.asarray(df_any["y"], dtype=float)))
            positive_rate = positives / float(row_count) if row_count else None
            if positive_rate is not None:
                if config.min_positive_rate is not None and positive_rate < config.min_positive_rate:
                    msg = (
                        f"Target positive rate {positive_rate:.4f} below minimum "
                        f"{config.min_positive_rate:.4f}"
                    )
                    raise DatasetValidationError(msg)
                if config.max_positive_rate is not None and positive_rate > config.max_positive_rate:
                    msg = (
                        f"Target positive rate {positive_rate:.4f} above maximum "
                        f"{config.max_positive_rate:.4f}"
                    )
                    raise DatasetValidationError(msg)

        feature_cols = _infer_feature_columns(df)
        coverage: dict[str, float] = {}
        if feature_cols and config.min_feature_coverage is not None:
            if is_polars:
                for name in feature_cols:
                    if name not in df.columns:
                        continue
                    valid = float(df.select(pl.col(name).is_null().not_().sum().alias("n")).item())
                    coverage[name] = valid / float(row_count) if row_count else 0.0
            else:
                for name in feature_cols:
                    if name not in df_any.columns:
                        continue
                    valid = float(df_any[name].notna().sum())
                    coverage[name] = valid / float(row_count) if row_count else 0.0
            low_coverage = [
                (name, ratio)
                for name, ratio in coverage.items()
                if ratio < config.min_feature_coverage
            ]
            if low_coverage:
                worst = min(low_coverage, key=lambda item: item[1])
                msg = (
                    "Feature coverage below acceptance threshold; "
                    f"example: {worst[0]}={worst[1]:.3f} < {config.min_feature_coverage:.3f}"
                )
                raise DatasetValidationError(msg)

        macro_cols_present: tuple[str, ...] = ()
        macro_counts: dict[str, int] = {}
        if config.require_macro_series:
            macros = set(config.require_macro_series)
            actual = {col for col in _macro_columns(df) if col in macros}
            missing = macros - actual
            if missing:
                msg = f"Missing macro series: {sorted(missing)}"
                raise DatasetValidationError(msg)
            macro_cols_present = tuple(sorted(actual))
            for macro in macro_cols_present:
                vintage_col = f"{macro}__value_vintage_ts"
                count = 0
                if is_polars and vintage_col in df.columns:
                    count = int(df.select(pl.col(vintage_col).is_not_null().sum().alias("n")).item())
                elif not is_polars and vintage_col in df_any.columns:
                    count = int(df_any[vintage_col].notna().sum())
                macro_counts[macro] = count

            min_obs = config.macro_min_vintage_observations
            policy = config.expected_vintage_policy or VintagePolicy.REAL_TIME
            if min_obs is not None and policy is VintagePolicy.REAL_TIME:
                failing = [macro for macro, cnt in macro_counts.items() if cnt < min_obs]
                if failing:
                    worst_macro = min(failing, key=lambda name: macro_counts.get(name, 0))
                    worst_macro_count = macro_counts.get(worst_macro, 0)
                    msg = (
                        "Macro vintage coverage below threshold; "
                        f"series {worst_macro} has {worst_macro_count} observations < {min_obs}"
                    )
                    raise DatasetValidationError(msg)

        status = "success"
        duration = time.perf_counter() - start
        _VALIDATION_COUNTER.labels(status=status).inc()
        _VALIDATION_SECONDS.labels(status=status).observe(duration)
        return DatasetValidationResult(
            row_count=row_count,
            positive_rate=positive_rate,
            feature_coverage=coverage,
            macro_columns_present=macro_cols_present,
            macro_observation_counts=macro_counts,
        )
    except Exception:
        duration = time.perf_counter() - start
        _VALIDATION_COUNTER.labels(status=status).inc()
        _VALIDATION_SECONDS.labels(status=status).observe(duration)
        raise
