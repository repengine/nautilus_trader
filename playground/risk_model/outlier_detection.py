"""
Factor return outlier detection utilities for Phase 4 robustness checks.

This module inspects factor return datasets for >3σ outliers, evaluates
alternative treatments (winsorisation versus exclusion), and quantifies the
impact on downstream regression betas. Results are emitted with structured
summaries so CLI tools and dashboards can persist reproducible artefacts.
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import numpy as np
import polars as pl
import structlog

from ml.common.metrics_bootstrap import get_gauge
from ml.common.metrics_bootstrap import get_histogram


LOGGER = structlog.get_logger(__name__)

OUTLIER_RATIO_GAUGE = get_gauge(
    "phase4_factor_outlier_ratio",
    "Share of rows flagged as factor outliers during Phase 4 robustness checks.",
    labelnames=("dataset",),
)
BETA_DELTA_HIST = get_histogram(
    "phase4_outlier_beta_delta",
    "L2 norm of regression beta deltas after outlier treatment.",
    labelnames=("treatment",),
    buckets=(0.0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0),
)


@dataclass(frozen=True)
class FactorOutlierSummary:
    """
    Summary statistics for a single factor column.
    """

    factor: str
    outlier_count: int
    outlier_ratio: float
    threshold: float
    standard_deviation: float

    def to_dict(self) -> dict[str, object]:
        """Serialise the summary into built-in Python types."""
        return {
            "factor": self.factor,
            "outlier_count": self.outlier_count,
            "outlier_ratio": self.outlier_ratio,
            "threshold": self.threshold,
            "standard_deviation": self.standard_deviation,
        }


@dataclass(frozen=True)
class TreatmentImpact:
    """
    Impact metrics for a given outlier treatment strategy.
    """

    treatment: str
    retained_rows: int
    beta_deltas: Mapping[str, float]
    delta_norm: float | None
    intercept_delta: float | None
    note: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialise treatment metrics."""
        return {
            "treatment": self.treatment,
            "retained_rows": self.retained_rows,
            "beta_deltas": dict(self.beta_deltas),
            "delta_norm": self.delta_norm,
            "intercept_delta": self.intercept_delta,
            "note": self.note,
        }


@dataclass(frozen=True)
class OutlierDetectionReport:
    """
    Aggregate results from evaluating factor return outliers.
    """

    dataset_path: Path | None
    total_rows: int
    threshold: float
    baseline_betas: Mapping[str, float]
    baseline_intercept: float
    outlier_rows: int
    outlier_ratio: float
    factor_summaries: tuple[FactorOutlierSummary, ...]
    treatment_impacts: tuple[TreatmentImpact, ...]
    recommended_treatment: str

    def to_dict(self) -> dict[str, object]:
        """Serialise the report for JSON emission."""
        return {
            "dataset_path": str(self.dataset_path) if self.dataset_path is not None else None,
            "total_rows": self.total_rows,
            "threshold": self.threshold,
            "baseline_betas": dict(self.baseline_betas),
            "baseline_intercept": self.baseline_intercept,
            "outlier_rows": self.outlier_rows,
            "outlier_ratio": self.outlier_ratio,
            "factor_summaries": [summary.to_dict() for summary in self.factor_summaries],
            "treatment_impacts": [impact.to_dict() for impact in self.treatment_impacts],
            "recommended_treatment": self.recommended_treatment,
        }


@dataclass(frozen=True)
class _FactorStats:
    """Internal container for factor mean and standard deviation."""

    mean: float
    std: float


def evaluate_factor_outliers(
    dataset_path: Path,
    *,
    factor_columns: Sequence[str] | None = None,
    return_column: str = "return",
    threshold: float = 3.0,
    treatments: Sequence[str] | None = None,
) -> OutlierDetectionReport:
    """
    Evaluate factor return outliers and quantify treatment impact.

    Parameters
    ----------
    dataset_path : Path
        Path to a Parquet/CSV dataset containing factor returns alongside the
        dependent return column.
    factor_columns : Sequence[str] | None, optional
        Explicit list of factor columns to analyse. When ``None`` the function
        infers factor columns by selecting those starting with ``"factor_"``
        while excluding the return column.
    return_column : str, default "return"
        Column name representing the dependent variable used for regression.
    threshold : float, default 3.0
        Z-score threshold above which samples are considered outliers.
    treatments : Sequence[str] | None, optional
        Ordered list of treatments to evaluate. Supported values are
        ``"winsorize"`` and ``"exclude"``. Defaults to both in the listed order.

    Returns
    -------
    OutlierDetectionReport
        Structured report capturing outlier ratios, per-factor summaries,
        regression beta impacts for each treatment, and the recommended
        approach based on minimal beta drift.
    """
    frame = _load_dataset(dataset_path)
    resolved_factors = _resolve_factor_columns(
        frame=frame,
        factor_columns=factor_columns,
        return_column=return_column,
    )
    clean_frame = frame.drop_nulls(subset=list(resolved_factors) + [return_column])
    total_rows = clean_frame.height
    if total_rows == 0:
        msg = "Dataset is empty after dropping null rows for factor analysis"
        raise ValueError(msg)

    stats = _compute_factor_stats(clean_frame, resolved_factors)
    factor_summaries, combined_mask = _summarise_outliers(
        clean_frame,
        stats=stats,
        factor_columns=resolved_factors,
        threshold=threshold,
    )
    outlier_rows = int(combined_mask.sum())
    outlier_ratio = outlier_rows / total_rows
    OUTLIER_RATIO_GAUGE.labels(dataset=str(dataset_path)).set(outlier_ratio)

    baseline_intercept, baseline_betas = _compute_regression_betas(
        clean_frame,
        factor_columns=resolved_factors,
        return_column=return_column,
    )

    if outlier_rows == 0:
        treatment_impacts: tuple[TreatmentImpact, ...] = ()
        recommended = "none"
    else:
        treatment_impacts = _evaluate_treatments(
            clean_frame,
            stats=stats,
            factor_columns=resolved_factors,
            combined_mask=combined_mask,
            return_column=return_column,
            baseline_intercept=baseline_intercept,
            baseline_betas=baseline_betas,
            threshold=threshold,
            treatments=treatments,
        )
        recommended = _select_recommended_treatment(treatment_impacts)

    LOGGER.info(
        "phase4_factor_outlier_evaluated",
        dataset=str(dataset_path.resolve()),
        total_rows=total_rows,
        outlier_rows=outlier_rows,
        outlier_ratio=outlier_ratio,
        recommended_treatment=recommended,
    )

    return OutlierDetectionReport(
        dataset_path=dataset_path,
        total_rows=total_rows,
        threshold=threshold,
        baseline_betas=baseline_betas,
        baseline_intercept=baseline_intercept,
        outlier_rows=outlier_rows,
        outlier_ratio=outlier_ratio,
        factor_summaries=factor_summaries,
        treatment_impacts=treatment_impacts,
        recommended_treatment=recommended,
    )


def _load_dataset(dataset_path: Path) -> pl.DataFrame:
    """Load supported dataset formats into a Polars DataFrame."""
    if not dataset_path.exists():
        msg = f"Dataset path does not exist: {dataset_path}"
        raise FileNotFoundError(msg)
    if dataset_path.suffix == ".parquet":
        return pl.read_parquet(dataset_path)
    if dataset_path.suffix == ".csv":
        return pl.read_csv(dataset_path)
    msg = f"Unsupported dataset extension: {dataset_path.suffix}"
    raise ValueError(msg)


def _resolve_factor_columns(
    *,
    frame: pl.DataFrame,
    factor_columns: Sequence[str] | None,
    return_column: str,
) -> tuple[str, ...]:
    """Derive the set of factor columns to inspect."""
    if factor_columns:
        resolved = tuple(dict.fromkeys(factor_columns))  # preserve order, drop duplicates
    else:
        resolved = tuple(
            column
            for column in frame.columns
            if column != return_column and column.startswith("factor_")
        )
    if not resolved:
        msg = "No factor columns provided or detected (looked for prefix 'factor_')"
        raise ValueError(msg)
    missing = [column for column in resolved if column not in frame.columns]
    if missing:
        msg = f"Factor columns missing in dataset: {', '.join(missing)}"
        raise ValueError(msg)
    if return_column not in frame.columns:
        msg = f"Return column '{return_column}' not found in dataset"
        raise ValueError(msg)
    return resolved


def _compute_factor_stats(
    frame: pl.DataFrame,
    factor_columns: Iterable[str],
) -> dict[str, _FactorStats]:
    """Compute mean and standard deviation per factor column."""
    stats: dict[str, _FactorStats] = {}
    for column in factor_columns:
        series = frame.get_column(column)
        values = series.to_numpy()
        mean = float(np.mean(values))
        std = float(np.std(values))
        stats[column] = _FactorStats(mean=mean, std=std)
    return stats


def _summarise_outliers(
    frame: pl.DataFrame,
    *,
    stats: Mapping[str, _FactorStats],
    factor_columns: Sequence[str],
    threshold: float,
) -> tuple[tuple[FactorOutlierSummary, ...], np.ndarray]:
    """Compute per-factor outlier metrics and combined mask."""
    total_rows = frame.height
    combined_mask = np.zeros(total_rows, dtype=bool)
    summaries: list[FactorOutlierSummary] = []

    for column in factor_columns:
        stat = stats[column]
        values = frame.get_column(column).to_numpy()
        mask = _compute_outlier_mask(values, stat.mean, stat.std, threshold=threshold)
        count = int(mask.sum())
        ratio = count / total_rows if total_rows else 0.0
        summaries.append(
            FactorOutlierSummary(
                factor=column,
                outlier_count=count,
                outlier_ratio=ratio,
                threshold=threshold,
                standard_deviation=stat.std,
            ),
        )
        combined_mask |= mask

    return tuple(summaries), combined_mask


def _compute_outlier_mask(
    values: np.ndarray,
    mean: float,
    std: float,
    *,
    threshold: float,
) -> np.ndarray:
    """Return a boolean mask for samples exceeding the z-score threshold."""
    if std <= 0.0:
        return np.zeros_like(values, dtype=bool)
    z_scores = np.abs((values - mean) / std)
    return z_scores > threshold


def _compute_regression_betas(
    frame: pl.DataFrame,
    *,
    factor_columns: Sequence[str],
    return_column: str,
) -> tuple[float, dict[str, float]]:
    """Estimate regression intercept and betas via least squares."""
    if frame.height <= len(factor_columns):
        msg = "Insufficient rows to estimate regression betas"
        raise ValueError(msg)
    response = frame.get_column(return_column).to_numpy()
    predictors = frame.select(list(factor_columns)).to_numpy()
    # Add intercept column
    design_matrix = np.column_stack([np.ones(predictors.shape[0]), predictors])
    betas, _, rank, _ = np.linalg.lstsq(design_matrix, response, rcond=None)
    if rank < len(factor_columns) + 1:
        LOGGER.warning("Regression design matrix is rank deficient", rank=rank)
    intercept = float(betas[0])
    factor_betas = {
        column: float(beta)
        for column, beta in zip(factor_columns, betas[1:], strict=False)
    }
    return intercept, factor_betas


def _evaluate_treatments(
    frame: pl.DataFrame,
    *,
    stats: Mapping[str, _FactorStats],
    factor_columns: Sequence[str],
    combined_mask: np.ndarray,
    return_column: str,
    baseline_intercept: float,
    baseline_betas: Mapping[str, float],
    threshold: float,
    treatments: Sequence[str] | None,
) -> tuple[TreatmentImpact, ...]:
    """Apply and score each requested outlier treatment."""
    resolved_treatments = _normalize_treatments(treatments)
    impacts: list[TreatmentImpact] = []
    if not resolved_treatments:
        return tuple(impacts)

    mask_series = pl.Series(combined_mask)
    for treatment in resolved_treatments:
        if treatment == "winsorize":
            treated = _apply_winsorisation(
                frame,
                stats=stats,
                factor_columns=factor_columns,
                threshold=threshold,
            )
        elif treatment == "exclude":
            treated = frame.filter(~mask_series)
        else:
            LOGGER.warning("Unsupported treatment requested", treatment=treatment)
            continue

        retained_rows = treated.height
        if retained_rows <= len(factor_columns):
            note = "Insufficient rows after treatment to recompute regression betas."
            impacts.append(
                TreatmentImpact(
                    treatment=treatment,
                    retained_rows=retained_rows,
                    beta_deltas={factor: float("nan") for factor in factor_columns},
                    delta_norm=None,
                    intercept_delta=None,
                    note=note,
                ),
            )
            continue

        try:
            treated_intercept, treated_betas = _compute_regression_betas(
                treated,
                factor_columns=factor_columns,
                return_column=return_column,
            )
        except ValueError as exc:
            LOGGER.warning("Failed to compute regression after treatment", treatment=treatment, exc_info=True)
            impacts.append(
                TreatmentImpact(
                    treatment=treatment,
                    retained_rows=retained_rows,
                    beta_deltas={factor: float("nan") for factor in factor_columns},
                    delta_norm=None,
                    intercept_delta=None,
                    note=str(exc),
                ),
            )
            continue

        beta_deltas: dict[str, float] = {}
        for factor in factor_columns:
            baseline_beta = float(baseline_betas.get(factor, 0.0))
            beta_deltas[factor] = abs(float(treated_betas.get(factor, 0.0)) - baseline_beta)

        delta_norm = sqrt(sum(delta * delta for delta in beta_deltas.values()))
        intercept_delta = abs(treated_intercept - baseline_intercept)
        BETA_DELTA_HIST.labels(treatment=treatment).observe(delta_norm)

        impacts.append(
            TreatmentImpact(
                treatment=treatment,
                retained_rows=retained_rows,
                beta_deltas=beta_deltas,
                delta_norm=delta_norm,
                intercept_delta=intercept_delta,
            ),
        )

    return tuple(impacts)


def _normalize_treatments(treatments: Sequence[str] | None) -> tuple[str, ...]:
    """Normalise and de-duplicate requested treatments."""
    if treatments is None:
        return ("winsorize", "exclude")
    normalised: list[str] = []
    seen: set[str] = set()
    for treatment in treatments:
        lowered = treatment.strip().lower()
        if not lowered or lowered in seen:
            continue
        if lowered not in {"winsorize", "exclude"}:
            LOGGER.warning("Ignoring unsupported outlier treatment", treatment=treatment)
            continue
        normalised.append(lowered)
        seen.add(lowered)
    return tuple(normalised)


def _apply_winsorisation(
    frame: pl.DataFrame,
    *,
    stats: Mapping[str, _FactorStats],
    factor_columns: Sequence[str],
    threshold: float,
) -> pl.DataFrame:
    """Return a frame with factor columns clipped to the z-score threshold."""
    updated = frame
    for column in factor_columns:
        stat = stats[column]
        if stat.std <= 0.0:
            # No variation - nothing to clip.
            continue
        lower = stat.mean - threshold * stat.std
        upper = stat.mean + threshold * stat.std
        updated = updated.with_columns(pl.col(column).clip(lower_bound=lower, upper_bound=upper))
    return updated


def _select_recommended_treatment(treatment_impacts: Sequence[TreatmentImpact]) -> str:
    """Choose the treatment with the smallest beta drift."""
    if not treatment_impacts:
        return "none"
    best_treatment = "none"
    best_delta: float | None = None
    for impact in treatment_impacts:
        delta_norm = impact.delta_norm
        if delta_norm is None or np.isnan(delta_norm):
            continue
        if best_delta is None or delta_norm < best_delta - 1e-12:
            best_treatment = impact.treatment
            best_delta = delta_norm
    return best_treatment
