"""Dataset validation utilities for TFT builds."""

from __future__ import annotations

import json
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from ml._imports import check_ml_dependencies
from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram
from ml.data.feature_columns import DEFAULT_FEATURE_EXCLUDE_COLUMNS
from ml.data.feature_columns import DEFAULT_FEATURE_EXCLUDE_PREFIXES
from ml.data.feature_columns import DEFAULT_FEATURE_EXCLUDE_SUFFIXES
from ml.data.feature_columns import split_feature_columns
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


class MacroCoverageError(ValueError):
    """Raised when macro coverage checks fail."""


@dataclass(slots=True, frozen=True)
class MacroCoverageValidator:
    """Validate macro series coverage prior to dataset promotion."""

    min_coverage: float = 0.9

    def validate_macro_coverage(
        self,
        df_any: Any,
        required_series: Sequence[str],
    ) -> dict[str, float]:
        """Ensure required macro series exist and meet coverage thresholds."""
        if not required_series:
            return {}

        df, is_polars = _as_polars(df_any)

        columns = tuple(required_series)
        if is_polars:
            available = {str(col) for col in df.columns}
        else:
            if hasattr(df_any, "columns"):
                available = {str(col) for col in df_any.columns}
            else:
                available = set()

        missing = [name for name in columns if name not in available]
        if missing:
            raise MacroCoverageError(f"Missing macro series: {missing}")

        coverage: dict[str, float] = {}
        for name in columns:
            if is_polars:
                valid = df.select(pl.col(name).is_not_null().mean()).item()
                coverage[name] = float(valid)
            elif pd is not None and isinstance(df_any, pd.DataFrame):
                series = df_any[name]
                coverage[name] = float(series.notna().mean())
            else:
                data = getattr(df_any, name, None)
                if data is None:
                    raise MacroCoverageError(f"Cannot access macro series: {name}")
                if hasattr(data, "__len__") and len(data):
                    non_null = sum(1 for value in data if value is not None)
                    coverage[name] = float(non_null) / float(len(data))
                else:
                    coverage[name] = 0.0

            if coverage[name] < self.min_coverage:
                ratio = coverage[name]
                msg = (
                    "Macro coverage sparse for series "
                    f"{name}: {ratio:.3f} < {self.min_coverage:.3f}"
                )
                raise MacroCoverageError(msg)

        return coverage


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
    require_monotonic_timestamps: bool = True
    timestamp_columns: tuple[str, ...] = ("ts_event", "timestamp")
    instrument_id_column: str = "instrument_id"
    forward_return_column: str = "forward_return"
    forward_return_horizon: int | None = None
    forward_return_price_column: str = "close"
    forward_return_tolerance: float = 1e-6
    require_numeric_features: bool = True
    feature_exclude_columns: tuple[str, ...] = DEFAULT_FEATURE_EXCLUDE_COLUMNS
    feature_exclude_suffixes: tuple[str, ...] = DEFAULT_FEATURE_EXCLUDE_SUFFIXES
    feature_exclude_prefixes: tuple[str, ...] = DEFAULT_FEATURE_EXCLUDE_PREFIXES


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


def _resolve_timestamp_column(
    columns: Sequence[str],
    candidates: Sequence[str],
) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _infer_feature_columns(
    df_any: Any,
    *,
    exclude: Sequence[str],
    exclude_suffixes: Sequence[str],
    exclude_prefixes: Sequence[str],
    require_numeric: bool,
) -> list[str]:
    numeric, non_numeric = split_feature_columns(
        df_any,
        exclude=exclude,
        exclude_suffixes=exclude_suffixes,
        exclude_prefixes=exclude_prefixes,
    )
    if require_numeric and non_numeric:
        msg = f"Non-numeric feature columns detected: {sorted(non_numeric)}"
        raise DatasetValidationError(msg)
    return numeric


def _validate_monotonic_timestamps(
    df_any: Any,
    *,
    timestamp_col: str,
    instrument_col: str,
) -> None:
    if pl is not None and isinstance(df_any, pl.DataFrame):
        if timestamp_col not in df_any.columns:
            raise DatasetValidationError(f"Missing timestamp column: {timestamp_col}")
        if instrument_col not in df_any.columns:
            raise DatasetValidationError(f"Missing instrument column: {instrument_col}")
        diffs = (
            pl.col(timestamp_col)
            .cast(pl.Int64)
            .diff()
            .over(instrument_col)
        )
        violations = (
            df_any.with_columns(diffs.alias("_diff"))
            .filter(pl.col("_diff") < 0)
            .select(pl.col(instrument_col))
            .unique()
        )
        if violations.height > 0:
            offenders = [str(item) for item in violations.to_series().to_list()]
            raise DatasetValidationError(
                f"Timestamp reversals detected for instruments: {offenders[:5]}",
            )
        return
    if pd is not None and isinstance(df_any, pd.DataFrame):
        if timestamp_col not in df_any.columns:
            raise DatasetValidationError(f"Missing timestamp column: {timestamp_col}")
        if instrument_col not in df_any.columns:
            raise DatasetValidationError(f"Missing instrument column: {instrument_col}")
        diffs = df_any.groupby(instrument_col)[timestamp_col].diff()
        if pd.api.types.is_timedelta64_dtype(diffs):
            reversals = diffs < pd.Timedelta(0)
        else:
            reversals = diffs < 0
        if reversals.any():
            offenders = df_any.loc[reversals, instrument_col].astype(str).unique().tolist()
            raise DatasetValidationError(
                f"Timestamp reversals detected for instruments: {offenders[:5]}",
            )
        return


def _validate_forward_return_alignment(
    df_any: Any,
    *,
    forward_return_column: str,
    price_column: str,
    horizon: int,
    instrument_col: str,
    tolerance: float,
) -> None:
    if horizon <= 0:
        raise DatasetValidationError(f"forward_return_horizon must be positive, got {horizon}")
    if pl is not None and isinstance(df_any, pl.DataFrame):
        if forward_return_column not in df_any.columns:
            raise DatasetValidationError(f"Missing forward return column: {forward_return_column}")
        if price_column not in df_any.columns:
            raise DatasetValidationError(f"Missing price column: {price_column}")
        if instrument_col not in df_any.columns:
            raise DatasetValidationError(f"Missing instrument column: {instrument_col}")
        expected = (
            pl.col(price_column).shift(-horizon).over(instrument_col)
            - pl.col(price_column)
        ) / pl.col(price_column)
        expected = expected.fill_null(0.0)
        expected = (
            pl.when(expected.is_infinite() | expected.is_nan())
            .then(0.0)
            .otherwise(expected)
        )
        diff = (pl.col(forward_return_column).cast(pl.Float64) - expected.cast(pl.Float64)).abs()
        max_diff = df_any.select(diff.max()).item()
        max_diff_value = float(max_diff) if max_diff is not None else 0.0
        if max_diff_value > tolerance:
            msg = (
                "forward_return misaligned with future prices; "
                f"max_diff={max_diff_value:.6f} > tolerance={tolerance:.6f}"
            )
            raise DatasetValidationError(msg)
        return
    if pd is not None and isinstance(df_any, pd.DataFrame):
        if forward_return_column not in df_any.columns:
            raise DatasetValidationError(f"Missing forward return column: {forward_return_column}")
        if price_column not in df_any.columns:
            raise DatasetValidationError(f"Missing price column: {price_column}")
        if instrument_col not in df_any.columns:
            raise DatasetValidationError(f"Missing instrument column: {instrument_col}")
        grouped = df_any.groupby(instrument_col)[price_column]
        future_prices = grouped.shift(-horizon)
        expected = (future_prices - df_any[price_column]) / df_any[price_column]
        expected = expected.fillna(0.0).replace([np.inf, -np.inf], 0.0)
        diff = (df_any[forward_return_column].astype(float) - expected.astype(float)).abs()
        max_diff_value = float(diff.max()) if len(diff) else 0.0
        if max_diff_value > tolerance:
            msg = (
                "forward_return misaligned with future prices; "
                f"max_diff={max_diff_value:.6f} > tolerance={tolerance:.6f}"
            )
            raise DatasetValidationError(msg)
        return


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
        if row_count == 0 and config.min_rows == 0:
            status = "ok"
            return DatasetValidationResult(
                row_count=0,
                positive_rate=None,
                feature_coverage={},
                macro_columns_present=(),
                macro_observation_counts={},
            )

        df_for_checks = df if is_polars else df_any
        if is_polars:
            columns = [str(name) for name in df.columns]
        else:
            columns = [str(name) for name in df_any.columns] if hasattr(df_any, "columns") else []

        if config.require_monotonic_timestamps:
            timestamp_col = _resolve_timestamp_column(columns, config.timestamp_columns)
            if timestamp_col is None:
                raise DatasetValidationError(
                    f"None of the timestamp columns found: {config.timestamp_columns}",
                )
            _validate_monotonic_timestamps(
                df_for_checks,
                timestamp_col=timestamp_col,
                instrument_col=config.instrument_id_column,
            )

        if config.forward_return_horizon is not None:
            _validate_forward_return_alignment(
                df_for_checks,
                forward_return_column=config.forward_return_column,
                price_column=config.forward_return_price_column,
                horizon=config.forward_return_horizon,
                instrument_col=config.instrument_id_column,
                tolerance=config.forward_return_tolerance,
            )

        positives: float | None = None
        positive_rate: float | None = None
        target_col: str | None = None
        if "y" in columns:
            target_col = "y"
        else:
            binary_cols = [col for col in columns if col.startswith("target_bin_")]
            if binary_cols:
                target_col = sorted(binary_cols)[0]

        if target_col:
            if is_polars:
                positives = float(df.select(pl.col(target_col).sum().alias("p")).item())
            else:
                positives = float(np.nansum(np.asarray(df_any[target_col], dtype=float)))
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

        feature_cols = _infer_feature_columns(
            df_for_checks,
            exclude=config.feature_exclude_columns,
            exclude_suffixes=config.feature_exclude_suffixes,
            exclude_prefixes=config.feature_exclude_prefixes,
            require_numeric=config.require_numeric_features,
        )
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


@dataclass(slots=True, frozen=True)
class DatasetReportConfig:
    """Configuration for dataset quality report generation."""

    dataset_path: Path
    output_json: Path | None = None
    output_markdown: Path | None = None


@dataclass(slots=True)
class DatasetReport:
    """Structured dataset report payload with optional markdown rendering."""

    data: dict[str, Any]
    markdown: str | None = None

    def to_json(self) -> str:
        """Serialize report data to stable JSON."""
        return json.dumps(self.data, indent=2, sort_keys=True)


def _infer_report_macro_columns(columns: Iterable[str]) -> list[str]:
    known = {
        "DGS1",
        "DGS2",
        "DGS10",
        "DGS30",
        "FEDFUNDS",
        "SOFR",
        "VIXCLS",
        "GDP",
        "GDPC1",
        "CPIAUCSL",
        "CPILFESL",
        "PCEPI",
        "UNRATE",
        "PAYEMS",
        "CIVPART",
        "UMCSENT",
        "RSXFS",
        "HOUST",
    }
    cols = list(columns)
    present = [column for column in cols if column in known]
    if present:
        return sorted(present)

    import re

    pattern = re.compile(r"^[A-Z0-9_]{3,}$")
    return sorted(column for column in cols if pattern.match(column))


def _infer_report_feature_columns(columns: Iterable[str]) -> list[str]:
    exclude = {"y", "time_index", "timestamp", "instrument_id", "ts_event"}
    return [column for column in columns if column not in exclude]


def _compute_report_macro_null_rates(df_any: Any, macro_cols: list[str]) -> dict[str, float]:
    if not macro_cols:
        return {}
    if pl is not None and isinstance(df_any, pl.DataFrame):
        total = df_any.height or 1
        return {
            col: float(df_any.select(pl.col(col).is_null().sum().alias("n")).item()) / float(total)
            for col in macro_cols
            if col in df_any.columns
        }
    if pd is not None and isinstance(df_any, pd.DataFrame):
        total = len(df_any) or 1
        return {
            col: float(df_any[col].isna().sum()) / float(total)
            for col in macro_cols
            if col in df_any.columns
        }
    check_ml_dependencies(["polars", "pandas"])
    raise RuntimeError("No dataframe engine available")


def _compute_report_feature_coverage(df_any: Any, feature_cols: list[str]) -> dict[str, Any]:
    if not feature_cols:
        return {"overall": {}, "by_symbol": {}}
    by_symbol: dict[str, dict[str, float]] = {}
    if pl is not None and isinstance(df_any, pl.DataFrame):
        total = df_any.height or 1
        overall = {
            col: float(df_any.select(pl.col(col).is_null().not_().sum().alias("n")).item())
            / float(total)
            for col in feature_cols
            if col in df_any.columns
        }
        if "instrument_id" in df_any.columns:
            grouped = (
                df_any.lazy()
                .group_by("instrument_id")
                .agg(
                    [
                        pl.col(col).is_null().not_().sum().alias(col)
                        for col in feature_cols
                        if col in df_any.columns
                    ],
                )
                .collect()
            )
            counts = df_any.lazy().group_by("instrument_id").agg(pl.len().alias("count")).collect()
            count_map = {str(row[0]): int(row[1]) or 1 for row in counts.iter_rows()}
            for row in grouped.iter_rows(named=True):
                instrument = str(row["instrument_id"])
                denom = count_map.get(instrument, 1)
                by_symbol[instrument] = {
                    key: float(value) / float(denom)
                    for key, value in row.items()
                    if key != "instrument_id"
                }
        return {"overall": overall, "by_symbol": by_symbol}
    if pd is not None and isinstance(df_any, pd.DataFrame):
        total = len(df_any) or 1
        overall = {
            col: float(df_any[col].notna().sum()) / float(total)
            for col in feature_cols
            if col in df_any.columns
        }
        by_symbol = {}
        if "instrument_id" in df_any.columns:
            for instrument, subset in df_any.groupby("instrument_id"):
                denom = len(subset) or 1
                by_symbol[str(instrument)] = {
                    col: float(subset[col].notna().sum()) / float(denom)
                    for col in feature_cols
                    if col in subset.columns
                }
        return {"overall": overall, "by_symbol": by_symbol}
    check_ml_dependencies(["polars", "pandas"])
    raise RuntimeError("No dataframe engine available")


def _compute_report_target_stats(df_any: Any) -> dict[str, Any]:
    if pl is not None and isinstance(df_any, pl.DataFrame):
        if "y" not in df_any.columns:
            return {}
        total = df_any.height
        positives = int(df_any.select(pl.col("y").sum().alias("p")).item()) if total > 0 else 0
        overall = {
            "total": int(total),
            "positives": positives,
            "positive_rate": float(positives) / float(total) if total else 0.0,
        }
        by_symbol = {}
        if "instrument_id" in df_any.columns and total > 0:
            grouped = (
                df_any.lazy()
                .group_by("instrument_id")
                .agg([pl.col("y").sum().alias("positives"), pl.len().alias("total")])
                .collect()
            )
            for row in grouped.iter_rows(named=True):
                instrument_total = int(row["total"]) or 1
                positives_sym = int(row["positives"])
                instrument = str(row["instrument_id"])
                by_symbol[instrument] = {
                    "total": int(row["total"]),
                    "positives": positives_sym,
                    "positive_rate": float(positives_sym) / float(instrument_total),
                }
        return {"overall": overall, "by_symbol": by_symbol}
    if pd is not None and isinstance(df_any, pd.DataFrame):
        if "y" not in df_any.columns:
            return {}
        total = len(df_any)
        positives = int(df_any["y"].sum()) if total > 0 else 0
        overall = {
            "total": total,
            "positives": positives,
            "positive_rate": float(positives) / float(total) if total else 0.0,
        }
        by_symbol = {}
        if "instrument_id" in df_any.columns and total > 0:
            grouped = df_any.groupby("instrument_id")["y"].agg(["sum", "count"]).reset_index()
            for _, row in grouped.iterrows():
                total_sym = int(row["count"]) or 1
                positives_sym = int(row["sum"])
                instrument = str(row["instrument_id"])
                by_symbol[instrument] = {
                    "total": int(row["count"]),
                    "positives": positives_sym,
                    "positive_rate": float(positives_sym) / float(total_sym),
                }
        return {"overall": overall, "by_symbol": by_symbol}
    check_ml_dependencies(["polars", "pandas"])
    raise RuntimeError("No dataframe engine available")


def _render_dataset_report_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = ["# Dataset Report"]
    macro = report.get("macro_null_rates", {})
    if macro:
        lines.append("\n## Macro Null Rates")
        for key, value in sorted(macro.items()):
            lines.append(f"- {key}: {value:.4f}")
    target = report.get("target", {})
    if target:
        overall = target.get("overall", {})
        lines.append("\n## Target Distribution")
        lines.append(
            "- total: {total}; positives: {positives}; rate: {rate:.4f}".format(
                total=overall.get("total", 0),
                positives=overall.get("positives", 0),
                rate=overall.get("positive_rate", 0.0),
            ),
        )
    return "\n".join(lines) + "\n"


def generate_dataset_report(config: DatasetReportConfig) -> DatasetReport:
    """Generate a quality report for a persisted dataset artifact."""
    if not config.dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {config.dataset_path}")

    df_any: Any
    suffix = config.dataset_path.suffix.lower()
    if pl is not None and suffix == ".parquet":
        df_any = pl.read_parquet(str(config.dataset_path))
    elif pl is not None and suffix == ".csv":
        df_any = pl.read_csv(str(config.dataset_path))
    elif pd is not None:
        if suffix == ".parquet":
            df_any = pd.read_parquet(str(config.dataset_path))
        else:
            df_any = pd.read_csv(str(config.dataset_path))
    else:
        check_ml_dependencies(["polars", "pandas"])
        raise RuntimeError("Unable to load dataset: no dataframe engine available")

    if pl is not None and isinstance(df_any, pl.DataFrame):
        columns = list(df_any.columns)
        row_count = int(df_any.height)
    else:
        if pd is None:
            raise RuntimeError("pandas is required to compute dataset columns")
        columns = list(df_any.columns)
        row_count = len(df_any)

    macro_cols = _infer_report_macro_columns(columns)
    feature_cols = _infer_report_feature_columns(columns)

    report_data = {
        "shape": [row_count, len(columns)],
        "macro_null_rates": _compute_report_macro_null_rates(df_any, macro_cols),
        "feature_coverage": _compute_report_feature_coverage(df_any, feature_cols),
        "target": _compute_report_target_stats(df_any),
    }

    markdown = _render_dataset_report_markdown(report_data)
    report = DatasetReport(data=report_data, markdown=markdown)

    if config.output_json is not None:
        config.output_json.write_text(report.to_json() + "\n", encoding="utf-8")
    if config.output_markdown is not None:
        config.output_markdown.write_text(markdown, encoding="utf-8")

    return report
