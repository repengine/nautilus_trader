"""Analysis and summarization utilities for the 3D sector risk model."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime

import numpy as np
import polars as pl
import structlog

from ml.common.metrics_manager import MetricsManager
from playground.exposure.factor_exposure import FactorExposureConfig
from playground.exposure.factor_exposure import compute_factor_exposures
from playground.exposure.factor_exposure import prepare_factor_returns
from playground.exposure.optimizer import RiskPoint
from playground.exposure.optimizer import compute_optimal_weights
from playground.exposure.optimizer import default_target_point


LOGGER = structlog.get_logger(__name__)


@dataclass(slots=True)
class SectorExposureSummary:
    """Aggregate statistics describing a sector's factor exposures."""

    sector_id: str
    factor_means: dict[str, float]
    covariance: dict[str, dict[str, float]]
    observation_count: int


@dataclass(slots=True)
class AnnualRiskProfile:
    """Ideal portfolio and risk coordinates for a specific year."""

    year: int
    weights: dict[str, float]
    sharpe_scores: dict[str, float]
    risk_point: RiskPoint
    status: str = "success"
    diagnostics: dict[str, object] | None = None


@dataclass(slots=True)
class SectorDistanceReport:
    """Distance between actual sector coordinates and the ideal point."""

    sector_id: str
    distance: float
    coordinates: dict[str, float]
    deltas: dict[str, float]
    recommended_weight: float
    mahalanobis_distance: float | None = None


def summarize_sector_exposures(
    exposures: pl.DataFrame,
    *,
    factor_names: Sequence[str],
) -> dict[str, SectorExposureSummary]:
    """Return mean/covariance summaries for each sector across factors."""
    required = {"asset_id", "benchmark_id", "ewma_beta"}
    if not required.issubset(exposures.columns):
        missing = required - set(exposures.columns)
        msg = f"Exposure frame missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    wide = (
        exposures
        .pivot(index=["asset_id", "ts_event"], on="benchmark_id", values="ewma_beta")
        .drop_nulls()
    )
    summaries: dict[str, SectorExposureSummary] = {}
    for frame in wide.partition_by("asset_id", maintain_order=False):
        if frame.is_empty():
            continue
        sector_id = str(frame["asset_id"][0])
        numeric = frame.select([name for name in factor_names if name in frame.columns])
        if numeric.is_empty():
            continue
        means = {
            name: float(numeric.select(pl.col(name).mean()).item())
            for name in numeric.columns
        }
        matrix = numeric.to_numpy()
        if matrix.shape[0] < 2:
            covariance = {name: dict.fromkeys(numeric.columns, 0.0) for name in numeric.columns}
        else:
            cov = np.cov(matrix.T, bias=False)
            covariance = {
                numeric.columns[i]: {
                    numeric.columns[j]: float(cov[i, j])
                    for j in range(cov.shape[1])
                }
                for i in range(cov.shape[0])
            }
        summaries[sector_id] = SectorExposureSummary(
            sector_id=sector_id,
            factor_means=means,
            covariance=covariance,
            observation_count=matrix.shape[0],
        )
    return summaries


def compute_annual_risk_profiles(
    sector_returns: pl.DataFrame,
    factor_levels: pl.DataFrame,
    *,
    factor_columns: Sequence[str],
    exposure_config: FactorExposureConfig,
    min_weight: float = 0.0,
    max_weight: float | None = None,
    weight_caps: Mapping[str, float] | None = None,
    target_point: RiskPoint | None = None,
    metrics: MetricsManager | None = None,
) -> list[AnnualRiskProfile]:
    """
    Compute annual risk-adjusted portfolios and ideal coordinates.

    Parameters
    ----------
    sector_returns : pl.DataFrame
        Tidy returns with columns ``timestamp``, ``symbol``, and ``return``.
    factor_levels : pl.DataFrame
        Factor level data that will be converted into returns.
    factor_columns : Sequence[str]
        Factor level column names to use when computing returns.
    exposure_config : FactorExposureConfig
        Configuration for the EWMA beta calculation.
    min_weight : float, optional
        Lower bound applied to weights after optimization.
    max_weight : float | None, optional
        Optional global cap applied to each weight.
    weight_caps : Mapping[str, float] | None, optional
        Optional per-sector caps overriding ``max_weight``.
    target_point : RiskPoint | None, optional
        Target point for distance-based optimization. Defaults to ``default_target_point``.
    metrics : MetricsManager | None, optional
        Optional metrics manager for observability.
    """
    mm = metrics or MetricsManager.default()
    target = target_point or default_target_point()

    LOGGER.info(
        "Computing annual risk profiles",
        factors=list(factor_columns),
        feature_set=exposure_config.feature_set_id,
    )

    factor_returns = prepare_factor_returns(factor_levels, columns=factor_columns)
    exposures = compute_factor_exposures(sector_returns, factor_returns, exposure_config)

    returns_by_year = _returns_by_year(sector_returns)
    exposure_by_year = _exposure_by_year(exposures)
    cap_mapping = dict(weight_caps) if weight_caps is not None else None

    profiles: list[AnnualRiskProfile] = []
    for year in sorted(set(returns_by_year.keys()) & set(exposure_by_year.keys())):
        sharpe_scores = returns_by_year[year]
        exposure_frame = exposure_by_year[year]
        if exposure_frame.is_empty():
            LOGGER.warning("Skipping year due to empty exposure frame", year=year)
            continue

        status = "success"
        diagnostics: dict[str, object] = {}

        try:
            weights = _compute_risk_adjusted_weights(sharpe_scores, min_weight)
        except ValueError as exc:
            mm.inc(
                "playground_annual_risk_profile_total",
                "Count of annual risk profile computations",
                labels={"status": "error"},
            )
            LOGGER.exception("Unable to compute Sharpe-based weights", year=year)
            diagnostics["error"] = str(exc)
            continue

        optimizer_frame = _build_optimizer_frame(exposure_frame, year)
        eigenvalues = _compute_factor_eigenvalues(optimizer_frame, factor_columns)
        if eigenvalues:
            diagnostics["cov_eigenvalues"] = eigenvalues

        final_weights = weights
        try:
            optimized_weights = compute_optimal_weights(
                optimizer_frame,
                target,
                min_weight=min_weight,
                max_weight=max_weight,
                weight_caps=cap_mapping,
            )
            final_weights = _blend_weights(weights, optimized_weights)
            risk_point = _compute_risk_point(exposure_frame, final_weights, factor_columns)
            diagnostics["optimizer"] = {
                "status": "optimized",
                "max_weight": max_weight,
                "weight_caps": cap_mapping,
            }
        except Exception as exc:
            status = "fallback"
            diagnostics["reason"] = str(exc)
            diagnostics["optimizer"] = {
                "status": "fallback",
                "max_weight": max_weight,
                "weight_caps": cap_mapping,
            }
            LOGGER.warning(
                "Falling back to Sharpe-based weights",
                year=year,
                reason=str(exc),
            )
            try:
                risk_point = _compute_risk_point(exposure_frame, final_weights, factor_columns)
            except Exception:
                mm.inc(
                    "playground_annual_risk_profile_total",
                    "Count of annual risk profile computations",
                    labels={"status": "error"},
                )
                LOGGER.exception("Failed to compute fallback risk point", year=year)
                continue
        mm.inc(
            "playground_annual_risk_profile_total",
            "Count of annual risk profile computations",
            labels={"status": status},
        )
        profiles.append(
            AnnualRiskProfile(
                year=year,
                weights=final_weights,
                sharpe_scores=sharpe_scores,
                risk_point=risk_point,
                status=status,
                diagnostics=diagnostics or None,
            ),
        )

    return profiles


def compute_sector_distance_reports(
    exposures: pl.DataFrame,
    profiles: Sequence[AnnualRiskProfile],
    *,
    factor_columns: Sequence[str],
    metrics: MetricsManager | None = None,
) -> dict[int, list[SectorDistanceReport]]:
    """Compare actual sector positions against the ideal points for each year."""
    mm = metrics or MetricsManager.default()
    exposure_by_year = _sector_coordinates_by_year(exposures, factor_columns)

    inverse_covariances = _inverse_covariance_by_year(exposures, factor_columns)

    reports: dict[int, list[SectorDistanceReport]] = {}
    for profile in profiles:
        sector_coordinates = exposure_by_year.get(profile.year, {})
        inverse_cov = inverse_covariances.get(profile.year)
        entries: list[SectorDistanceReport] = []
        for sector_id, coords in sector_coordinates.items():
            distance = _euclidean_distance(coords, profile.risk_point.coordinates)
            deltas = {
                factor: float(coords.get(factor, 0.0) - profile.risk_point.coordinates.get(factor, 0.0))
                for factor in factor_columns
            }
            mahalanobis_distance = _compute_mahalanobis_distance(deltas, inverse_cov, factor_columns)
            if mahalanobis_distance is not None:
                mm.observe(
                    "playground_sector_mahalanobis_distance",
                    "Mahalanobis distance of sector from ideal point",
                    value=mahalanobis_distance,
                    labels={"sector": sector_id, "year": profile.year},
                )
            entries.append(
                SectorDistanceReport(
                    sector_id=sector_id,
                    distance=distance,
                    coordinates={factor: float(coords.get(factor, 0.0)) for factor in factor_columns},
                    deltas=deltas,
                    recommended_weight=float(profile.weights.get(sector_id, 0.0)),
                    mahalanobis_distance=mahalanobis_distance,
                ),
            )
        reports[profile.year] = entries
        mm.observe(
            "playground_sector_distance_mean",
            "Mean distance between sector positions and ideal point",
            value=float(np.mean([entry.distance for entry in entries]) if entries else 0.0),
            labels={"year": profile.year},
        )
    return reports


def summarize_eigenvalue_trends(
    profiles: Sequence[AnnualRiskProfile],
    *,
    bucket_size: int = 10,
) -> dict[str, dict[str, float]]:
    """Aggregate eigenvalue diagnostics by decade (or configurable bucket)."""
    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")

    aggregates: dict[str, dict[str, list[float]]] = {}
    for profile in profiles:
        diagnostics = profile.diagnostics or {}
        eigenvalues = diagnostics.get("cov_eigenvalues")
        if not eigenvalues:
            continue
        bucket_start = (profile.year // bucket_size) * bucket_size
        bucket_key = f"{bucket_start}s"
        bucket = aggregates.setdefault(bucket_key, {})
        for index, value in enumerate(eigenvalues, start=1):
            label = f"eig_{index}"
            bucket.setdefault(label, []).append(float(value))

    summary: dict[str, dict[str, float]] = {}
    for bucket_key, values in aggregates.items():
        summary[bucket_key] = {
            label: float(np.mean(series))
            for label, series in values.items()
            if series
        }
    return summary


def _returns_by_year(frame: pl.DataFrame) -> dict[int, dict[str, float]]:
    data = (
        frame
        .with_columns(pl.col("timestamp").dt.year().alias("year"))
        .group_by(["year", "symbol"])
        .agg(
            pl.mean("return").alias("mean_return"),
            pl.std("return").alias("volatility"),
        )
        .with_columns(
            pl.when(pl.col("volatility") > 0)
            .then(pl.col("mean_return") / pl.col("volatility"))
            .otherwise(0.0)
            .alias("sharpe"),
        )
        .select(["year", "symbol", "sharpe"])
    )

    result: dict[int, dict[str, float]] = defaultdict(dict)
    for row in data.iter_rows(named=True):
        result[int(row["year"])][str(row["symbol"])] = float(row["sharpe"])
    return result


def _exposure_by_year(frame: pl.DataFrame) -> dict[int, pl.DataFrame]:
    data = frame.with_columns(
        pl.from_epoch(pl.col("ts_event"), time_unit="ns").dt.replace_time_zone("UTC").alias("ts_dt"),
    ).with_columns(pl.col("ts_dt").dt.year().alias("year"))

    grouped: dict[int, pl.DataFrame] = {}
    for subset in data.partition_by("year", maintain_order=False):
        if subset.is_empty():
            continue
        year = int(subset["year"][0])
        grouped[year] = (
            subset
            .filter(pl.col("ewma_beta").is_finite())
            .group_by(["asset_id", "benchmark_id"])
            .agg(pl.mean("ewma_beta").alias("ewma_beta"))
        )
    return grouped


def compute_annual_sector_positions(
    exposures: pl.DataFrame,
    *,
    factor_columns: Sequence[str],
    stable_positions: dict[str, dict[str, float]] | None = None,
) -> dict[int, dict[str, dict[str, float]]]:
    """
    Compute sector positions for each year in the stable coordinate system.

    Parameters
    ----------
    exposures
        Long-form EWMA beta DataFrame from compute_factor_exposures().
    factor_columns
        Factor names (e.g., "factor_duration", "factor_credit").
    stable_positions
        Optional stable positions from compute_stable_sector_positions().
        If None, uses mean of current year's exposures (backward compatible).

    Returns
    -------
    dict[int, dict[str, dict[str, float]]]
        Nested dict: year -> sector_id -> {factor: value}

        Example:
        {
            2010: {
                "XLK": {"factor_duration": -0.11, "factor_credit": -0.18},
                "XLU": {"factor_duration": 0.24, "factor_credit": 0.06},
            },
            2020: {
                "XLK": {"factor_duration": -0.14, "factor_credit": -0.25},
                ...
            }
        }

    Notes
    -----
    When stable_positions is provided, sector positions are computed in that
    stable coordinate system. This ensures sectors remain in their "clouds"
    across years rather than the coordinate system shifting.

    The function partitions exposures by year (extracted from ts_event),
    then computes mean beta for each sector-factor pair within that year.
    """
    # Validate inputs
    if exposures.is_empty():
        return {}

    required_columns = {"asset_id", "benchmark_id", "ts_event", "ewma_beta"}
    missing_columns = required_columns - set(exposures.columns)
    if missing_columns:
        msg = f"exposures missing required columns: {sorted(missing_columns)}"
        raise ValueError(msg)

    factor_list = list(factor_columns)
    if not factor_list:
        msg = "factor_columns cannot be empty"
        raise ValueError(msg)

    # Convert ts_event (nanoseconds) to year
    exposures_with_year = exposures.with_columns(
        pl.from_epoch(pl.col("ts_event"), time_unit="ns").dt.replace_time_zone("UTC").alias("ts_dt"),
    ).with_columns(pl.col("ts_dt").dt.year().alias("year"))

    # Partition by year and compute positions
    annual_positions: dict[int, dict[str, dict[str, float]]] = {}

    for subset in exposures_with_year.partition_by("year", maintain_order=False):
        if subset.is_empty():
            continue

        year = int(subset["year"][0])

        # Filter to valid exposures and aggregate by sector-factor
        aggregated = (
            subset
            .filter(pl.col("ewma_beta").is_finite())
            .filter(pl.col("benchmark_id").is_in(factor_list))
            .group_by(["asset_id", "benchmark_id"])
            .agg(pl.mean("ewma_beta").alias("ewma_beta"))
        )

        if aggregated.is_empty():
            continue

        # Pivot to wide format: asset_id x factor columns
        pivot = aggregated.pivot(
            index="asset_id",
            on="benchmark_id",
            values="ewma_beta",
        ).drop_nulls()

        if pivot.is_empty():
            continue

        # Extract positions for each sector
        year_positions = _extract_sector_positions_from_pivot(
            pivot,
            factor_list,
            stable_positions,
        )
        annual_positions[year] = year_positions

    return annual_positions


def _extract_sector_positions_from_pivot(
    pivot: pl.DataFrame,
    factor_list: Sequence[str],
    stable_positions: dict[str, dict[str, float]] | None,
) -> dict[str, dict[str, float]]:
    """Extract sector positions from pivoted exposure data."""
    year_positions: dict[str, dict[str, float]] = {}

    for row in pivot.iter_rows(named=True):
        sector_id = str(row["asset_id"])
        sector_position: dict[str, float] = {}

        for factor in factor_list:
            if factor in row:
                sector_position[factor] = float(row[factor])
            elif stable_positions and sector_id in stable_positions:
                # Use stable position as fallback for missing factors
                sector_position[factor] = float(stable_positions[sector_id].get(factor, 0.0))
            else:
                # Default to zero if no data and no stable position
                sector_position[factor] = 0.0

        year_positions[sector_id] = sector_position

    return year_positions


def compute_portfolio_trajectory(
    weights_by_year: dict[int, dict[str, float]],
    stable_positions: dict[str, dict[str, float]],
    *,
    factor_columns: Sequence[str],
) -> dict[int, RiskPoint]:
    """
    Map ideal portfolio to stable coordinate system for each year.

    Parameters
    ----------
    weights_by_year
        Portfolio weights for each year: {year: {sector: weight}}
        Example: {2010: {"XLK": 0.30, "XLU": 0.50, ...}, ...}
    stable_positions
        Stable sector positions from compute_stable_sector_positions().
    factor_columns
        Factor names to compute coordinates for.

    Returns
    -------
    dict[int, RiskPoint]
        Trajectory of portfolio through stable risk space: {year: RiskPoint}

    Raises
    ------
    ValueError
        If weights_by_year is empty, if stable_positions is empty,
        if factor_columns is empty, or if a sector in weights has no
        stable position.

    Notes
    -----
    The portfolio's position in factor space is computed as a weighted
    sum of sector stable positions:

        portfolio_coord[factor] = Σ(weight[sector] * stable_pos[sector][factor])

    This ensures the portfolio's trajectory is mapped to the SAME stable
    coordinate system as the sectors, so sectors stay in "clouds" while
    the portfolio moves through space based on allocation changes.

    Examples
    --------
    >>> weights = {2010: {"XLK": 0.3, "XLU": 0.7}}
    >>> stable = {
    ...     "XLK": {"factor_duration": -0.12, "factor_credit": -0.20},
    ...     "XLU": {"factor_duration": 0.25, "factor_credit": 0.05},
    ... }
    >>> trajectory = compute_portfolio_trajectory(
    ...     weights, stable, factor_columns=["factor_duration", "factor_credit"]
    ... )
    >>> trajectory[2010].coordinates
    {"factor_duration": 0.139, "factor_credit": -0.025}  # 0.3*(-0.12) + 0.7*0.25 = 0.139
    """
    # Validate inputs
    if not weights_by_year:
        msg = "weights_by_year cannot be empty"
        raise ValueError(msg)

    if not stable_positions:
        msg = "stable_positions cannot be empty"
        raise ValueError(msg)

    factor_list = list(factor_columns)
    if not factor_list:
        msg = "factor_columns cannot be empty"
        raise ValueError(msg)

    # Compute portfolio coordinates for each year
    trajectory: dict[int, RiskPoint] = {}

    for year, weights in weights_by_year.items():
        # Initialize coordinates to zero
        portfolio_coordinates: dict[str, float] = {factor: 0.0 for factor in factor_list}

        # Compute weighted sum of sector stable positions
        for sector_id, weight in weights.items():
            # Skip if weight is zero or negligible
            if abs(weight) < 1e-10:
                continue

            # Check that sector exists in stable positions
            if sector_id not in stable_positions:
                msg = (
                    f"Sector {sector_id!r} in year {year} weights "
                    f"has no stable position"
                )
                raise ValueError(msg)

            sector_position = stable_positions[sector_id]

            # Add weighted contribution for each factor
            for factor in factor_list:
                # Use stable position if available, default to 0.0 if missing
                factor_value = float(sector_position.get(factor, 0.0))
                portfolio_coordinates[factor] += weight * factor_value

        # Create RiskPoint for this year
        trajectory[year] = RiskPoint(portfolio_coordinates)

    return trajectory


def _sector_coordinates_by_year(
    frame: pl.DataFrame,
    factor_columns: Sequence[str],
) -> dict[int, dict[str, dict[str, float]]]:
    exposures = frame.with_columns(
        pl.from_epoch(pl.col("ts_event"), time_unit="ns").dt.replace_time_zone("UTC").alias("ts_dt"),
    ).with_columns(pl.col("ts_dt").dt.year().alias("year"))

    coordinates: dict[int, dict[str, dict[str, float]]] = {}
    for subset in exposures.partition_by("year", maintain_order=False):
        if subset.is_empty():
            continue
        year = int(subset["year"][0])
        aggregated = (
            subset
            .filter(pl.col("ewma_beta").is_finite())
            .group_by(["asset_id", "benchmark_id"])
            .agg(pl.mean("ewma_beta").alias("ewma_beta"))
        )
        pivot = aggregated.pivot(index="asset_id", on="benchmark_id", values="ewma_beta").drop_nulls()
        year_coords: dict[str, dict[str, float]] = {}
        for row in pivot.iter_rows(named=True):
            sector_id = str(row["asset_id"])
            year_coords[sector_id] = {
                factor: float(row.get(factor, 0.0))
                for factor in factor_columns
            }
        coordinates[year] = year_coords
    return coordinates


def _compute_risk_adjusted_weights(sharpe_scores: Mapping[str, float], min_weight: float) -> dict[str, float]:
    adjusted = {sector: max(score, 0.0) for sector, score in sharpe_scores.items()}
    total = sum(adjusted.values())
    if total == 0:
        count = len(adjusted)
        if count == 0:
            raise ValueError("No sectors available for weight computation")
        uniform = 1.0 / count
        return {sector: max(uniform, min_weight) for sector in adjusted}
    weights = {sector: value / total for sector, value in adjusted.items()}
    weights = {sector: max(weight, min_weight) for sector, weight in weights.items()}
    norm = sum(weights.values())
    return {sector: weight / norm for sector, weight in weights.items()}


def _build_optimizer_frame(frame: pl.DataFrame, year: int) -> pl.DataFrame:
    timestamp = datetime(year, 12, 31, tzinfo=UTC)
    ts_ns = int(timestamp.timestamp() * 1_000_000_000)
    return frame.with_columns(
        pl.lit(ts_ns).alias("ts_event"),
    )


def _blend_weights(primary: Mapping[str, float], secondary: Mapping[str, float]) -> dict[str, float]:
    keys = set(primary) | set(secondary)
    blended = {key: (primary.get(key, 0.0) + secondary.get(key, 0.0)) / 2.0 for key in keys}
    total = sum(blended.values())
    if total <= 0:
        raise ValueError("Blended weights sum to zero")
    return {key: value / total for key, value in blended.items()}


def _compute_factor_eigenvalues(
    frame: pl.DataFrame,
    factor_columns: Sequence[str],
) -> list[float]:
    if frame.is_empty():
        return []
    pivot = frame.pivot(index="asset_id", on="benchmark_id", values="ewma_beta").drop_nulls()
    if pivot.height < 2:
        return []
    matrix = pivot.select([col for col in pivot.columns if col != "asset_id"]).to_numpy()
    if matrix.size == 0:
        return []
    cov = np.cov(matrix, rowvar=False)
    try:
        eigvals = np.linalg.eigvalsh(cov)
    except np.linalg.LinAlgError:
        return []
    return sorted((float(value) for value in eigvals if np.isfinite(value)), reverse=True)


def _compute_risk_point(
    frame: pl.DataFrame,
    weights: Mapping[str, float],
    factor_columns: Sequence[str],
) -> RiskPoint:
    coordinates: dict[str, float] = dict.fromkeys(factor_columns, 0.0)
    pivot = frame.pivot(index="asset_id", on="benchmark_id", values="ewma_beta").drop_nulls()
    for row in pivot.iter_rows(named=True):
        asset_id = str(row["asset_id"])
        weight = float(weights.get(asset_id, 0.0))
        for factor in factor_columns:
            coordinates[factor] += weight * float(row.get(factor, 0.0))
    return RiskPoint(coordinates)


def _euclidean_distance(coords: Mapping[str, float], target: Mapping[str, float]) -> float:
    squared = [
        (float(coords.get(key, 0.0)) - float(target.get(key, 0.0))) ** 2
        for key in set(coords) | set(target)
    ]
    return math.sqrt(sum(squared))


def _inverse_covariance_by_year(
    exposures: pl.DataFrame,
    factor_columns: Sequence[str],
) -> dict[int, np.ndarray]:
    yearly = _exposure_by_year(exposures)
    inverse: dict[int, np.ndarray] = {}
    for year, frame in yearly.items():
        matrix = _compute_inverse_covariance(frame, factor_columns)
        if matrix is not None:
            inverse[year] = matrix
    return inverse


def _compute_inverse_covariance(
    frame: pl.DataFrame,
    factor_columns: Sequence[str],
) -> np.ndarray | None:
    if frame.is_empty():
        return None
    pivot = frame.pivot(index="asset_id", on="benchmark_id", values="ewma_beta").drop_nulls()
    columns = [column for column in factor_columns if column in pivot.columns]
    if len(columns) < 1 or pivot.height < 2:
        return None
    matrix = pivot.select(columns).to_numpy()
    if matrix.size == 0:
        return None
    covariance = np.cov(matrix, rowvar=False)
    if covariance.shape[0] != covariance.shape[1]:
        return None
    try:
        return np.linalg.inv(covariance)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(covariance)


def _compute_mahalanobis_distance(
    deltas: Mapping[str, float],
    inverse_covariance: np.ndarray | None,
    factor_columns: Sequence[str],
) -> float | None:
    if inverse_covariance is None:
        return None
    vector = np.array([float(deltas.get(name, 0.0)) for name in factor_columns], dtype=float)
    if vector.size == 0:
        return None
    distance_squared = float(vector @ inverse_covariance @ vector.T)
    if distance_squared < 0.0:
        distance_squared = 0.0
    return math.sqrt(distance_squared)


__all__ = [
    "AnnualRiskProfile",
    "SectorDistanceReport",
    "SectorExposureSummary",
    "compute_annual_risk_profiles",
    "compute_annual_sector_positions",
    "compute_portfolio_trajectory",
    "compute_sector_distance_reports",
    "summarize_eigenvalue_trends",
    "summarize_sector_exposures",
]
