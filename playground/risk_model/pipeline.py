"""Pipeline orchestration for building the 3D risk dataset."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from pathlib import Path

import polars as pl
import structlog

from ml.common.metrics_manager import MetricsManager
from ml.data.validation import MacroCoverageError
from ml.data.validation import MacroCoverageValidator
from playground.exposure.factor_exposure import FactorExposureConfig
from playground.exposure.factor_exposure import compute_factor_exposures
from playground.exposure.factor_exposure import compute_stable_sector_positions
from playground.exposure.factor_exposure import prepare_factor_returns
from playground.exposure.optimizer import default_target_point
from playground.exposure.persistence import CrossAssetBetaPersistenceConfig
from playground.exposure.persistence import persist_cross_asset_betas
from playground.risk_model.analysis import AnnualRiskProfile
from playground.risk_model.analysis import SectorDistanceReport
from playground.risk_model.analysis import compute_annual_risk_profiles
from playground.risk_model.analysis import compute_annual_sector_positions
from playground.risk_model.analysis import compute_portfolio_trajectory
from playground.risk_model.analysis import compute_sector_distance_reports
from playground.risk_model.analysis import summarize_eigenvalue_trends
from playground.risk_model.dataset import CoverageSummary
from playground.risk_model.dataset import FactorDataRequest
from playground.risk_model.dataset import FactorReturnFetcher
from playground.risk_model.dataset import SectorDataRequest
from playground.risk_model.dataset import SectorDatasetAssembler
from playground.risk_model.dataset import SectorReturnFetcher
from playground.risk_model.fetchers import FactorFetcherConfig
from playground.risk_model.fetchers import SectorFetcherConfig
from playground.risk_model.fetchers import build_factor_fetcher
from playground.risk_model.fetchers import build_sector_fetcher
from playground.risk_model.visualization import VisualizationPayload
from playground.risk_model.visualization import build_visualization_payload


LOGGER = structlog.get_logger(__name__)


@dataclass(slots=True)
class RiskPipelineConfig:
    """Parameters controlling the risk dataset build."""

    sectors: tuple[str, ...]
    factor_columns: tuple[str, ...]
    start: datetime
    end: datetime
    feature_set_id: str
    min_weight: float = 0.0
    max_weight: float | None = None
    weight_caps: Mapping[str, float] | None = None
    persist_dir: Path | None = Path("playground/data/sector_dataset")
    cache_dir: Path | None = Path("playground/data/cache")
    visualization_dir: Path | None = Path("playground/data/visualizations")
    notes: str | None = None
    sector_fetcher_config: SectorFetcherConfig | None = None
    factor_fetcher_config: FactorFetcherConfig | None = None
    calendar: str = "XNYS"
    min_sector_coverage: float = 0.8
    min_factor_coverage: float = 0.8
    beta_persistence: CrossAssetBetaPersistenceConfig = field(
        default_factory=CrossAssetBetaPersistenceConfig,
    )

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("End timestamp must be after start")
        if self.start.tzinfo is None:
            self.start = self.start.replace(tzinfo=UTC)
        if self.end.tzinfo is None:
            self.end = self.end.replace(tzinfo=UTC)
        if not self.calendar.strip():
            raise ValueError("Trading calendar identifier must be a non-empty string")
        if not 0.0 <= self.min_sector_coverage <= 1.0:
            raise ValueError("min_sector_coverage must be between 0 and 1 inclusive")
        if not 0.0 <= self.min_factor_coverage <= 1.0:
            raise ValueError("min_factor_coverage must be between 0 and 1 inclusive")
        if self.max_weight is not None and self.max_weight <= 0:
            raise ValueError("max_weight must be positive when provided")
        if self.max_weight is not None and self.max_weight < self.min_weight:
            raise ValueError("max_weight must be greater than or equal to min_weight")
        if self.weight_caps is not None:
            for sector, cap in self.weight_caps.items():
                if cap <= 0:
                    raise ValueError(f"Weight cap for {sector} must be positive")
                if cap < self.min_weight:
                    raise ValueError(
                        f"Weight cap for {sector} must be greater than min_weight",
                    )
            if not isinstance(self.weight_caps, dict):
                object.__setattr__(self, "weight_caps", dict(self.weight_caps))


@dataclass(slots=True)
class RiskPipelineResult:
    """Artifacts produced by a risk dataset build."""

    sector_returns: pl.DataFrame
    factor_levels: pl.DataFrame
    factor_returns: pl.DataFrame
    exposures: pl.DataFrame
    profiles: list[AnnualRiskProfile]
    distance_reports: Mapping[int, list[SectorDistanceReport]]
    visualization_payloads: Mapping[int, VisualizationPayload]
    coverage_summary: CoverageSummary
    eigenvalue_trends: Mapping[str, dict[str, float]]
    coverage_alerts: dict[str, dict[str, float]]
    optimizer_recommendations: Mapping[int, dict[str, float]]
    beta_persisted_rows: int


def run_risk_pipeline(
    config: RiskPipelineConfig,
    *,
    metrics: MetricsManager | None = None,
    fred_config: FactorFetcherConfig | None = None,
    sector_fetcher_override: SectorReturnFetcher | None = None,
    factor_fetcher_override: FactorReturnFetcher | None = None,
) -> RiskPipelineResult:
    """Execute the full data-to-visualization pipeline."""
    mm = metrics or MetricsManager.default()

    cache_dir = config.cache_dir or Path("playground/data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    sector_fetcher = sector_fetcher_override or build_sector_fetcher(config.sector_fetcher_config)
    factor_cache = cache_dir / "fred_factors.parquet"
    factor_fetcher_config = config.factor_fetcher_config or fred_config
    factor_fetcher = factor_fetcher_override or build_factor_fetcher(
        factor_fetcher_config,
        cache_path=factor_cache,
    )
    assembler = SectorDatasetAssembler(sector_fetcher, factor_fetcher, metrics=mm)

    sector_request = SectorDataRequest(
        sectors=config.sectors,
        start=config.start,
        end=config.end,
        frequency="1d",
        calendar=config.calendar,
    )
    factor_request = FactorDataRequest(
        factor_columns=config.factor_columns,
        start=config.start,
        end=config.end,
        calendar=config.calendar,
    )

    dataset = assembler.build(
        sector_request,
        factor_request,
        persist_dir=config.persist_dir,
    )
    coverage_summary = dataset.coverage

    _enforce_coverage_requirements(
        coverage_summary,
        dataset.factor_returns,
        factor_columns=config.factor_columns,
        min_sector_coverage=config.min_sector_coverage,
        min_factor_coverage=config.min_factor_coverage,
        metrics=mm,
    )

    coverage_alerts = _build_coverage_alerts(
        coverage_summary,
        min_sector_threshold=config.min_sector_coverage,
        min_factor_threshold=config.min_factor_coverage,
    )
    _record_coverage_metrics(mm, coverage_alerts)

    factor_returns = prepare_factor_returns(dataset.factor_returns, columns=config.factor_columns)

    exposure_config = FactorExposureConfig(feature_set_id=config.feature_set_id)
    exposures = compute_factor_exposures(dataset.sector_returns, factor_returns, exposure_config)

    # Compute stable sector positions (long-term centers in factor space)
    stable_positions = compute_stable_sector_positions(
        exposures,
        factor_columns=config.factor_columns,
        aggregation="median",
    )

    # Compute annual sector positions (year-by-year tracking in stable coordinates)
    annual_positions = compute_annual_sector_positions(
        exposures,
        factor_columns=config.factor_columns,
        stable_positions=stable_positions,
    )

    beta_persisted_rows = persist_cross_asset_betas(
        exposures,
        config.beta_persistence,
        metrics=mm,
    )

    target_point = default_target_point()
    profiles = compute_annual_risk_profiles(
        dataset.sector_returns,
        dataset.factor_returns,
        factor_columns=config.factor_columns,
        exposure_config=exposure_config,
        min_weight=config.min_weight,
        max_weight=config.max_weight,
        weight_caps=config.weight_caps,
        target_point=target_point,
        metrics=mm,
    )
    eigenvalue_trends = summarize_eigenvalue_trends(profiles)

    # Compute portfolio trajectory through stable coordinate system
    weights_by_year = {profile.year: profile.weights for profile in profiles}
    if weights_by_year:
        try:
            portfolio_trajectory = compute_portfolio_trajectory(
                weights_by_year,
                stable_positions,
                factor_columns=config.factor_columns,
            )
        except ValueError as exc:
            LOGGER.warning(
                "Failed to compute portfolio trajectory",
                reason=str(exc),
            )
            mm.inc(
                "playground_portfolio_trajectory_total",
                "Portfolio trajectory computation outcomes",
                labels={"status": "error"},
            )
    else:
        LOGGER.warning("No profiles available, skipping portfolio trajectory computation")

    reports = compute_sector_distance_reports(
        exposures,
        profiles,
        factor_columns=config.factor_columns,
        metrics=mm,
    )

    visualization_payloads: dict[int, VisualizationPayload] = {}
    vis_dir = config.visualization_dir
    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)
    for profile in profiles:
        payload = build_visualization_payload(
            profile,
            reports.get(profile.year, []),
            notes=config.notes,
            coverage=coverage_summary,
            eigenvalue_trends=dict(eigenvalue_trends),
            coverage_alerts=coverage_alerts,
            stable_positions=stable_positions,
            annual_positions=annual_positions.get(profile.year),
        )
        if vis_dir is not None:
            payload_path = vis_dir / f"risk_{profile.year}.json"
            payload.to_json(payload_path)
        visualization_payloads[profile.year] = payload

    if config.persist_dir is not None:
        coverage_alerts_path = config.persist_dir / "coverage_alerts.json"
        coverage_alerts_path.parent.mkdir(parents=True, exist_ok=True)
        coverage_alerts_payload = {
            "coverage_alerts": coverage_alerts,
            "min_sector_threshold": config.min_sector_coverage,
            "min_factor_threshold": config.min_factor_coverage,
        }
        coverage_alerts_path.write_text(
            json.dumps(coverage_alerts_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    optimizer_recommendations = {
        profile.year: dict(profile.weights) for profile in profiles
    }

    return RiskPipelineResult(
        sector_returns=dataset.sector_returns,
        factor_levels=dataset.factor_returns,
        factor_returns=factor_returns,
        exposures=exposures,
        profiles=profiles,
        distance_reports=reports,
        visualization_payloads=visualization_payloads,
        coverage_summary=coverage_summary,
        eigenvalue_trends=eigenvalue_trends,
        coverage_alerts=coverage_alerts,
        optimizer_recommendations=optimizer_recommendations,
        beta_persisted_rows=beta_persisted_rows,
    )
def _enforce_coverage_requirements(
    coverage: CoverageSummary,
    factor_levels: pl.DataFrame,
    *,
    factor_columns: tuple[str, ...],
    min_sector_coverage: float,
    min_factor_coverage: float,
    metrics: MetricsManager,
) -> None:
    low_sectors = {
        sector: ratio
        for sector, ratio in coverage.sector_coverage.items()
        if ratio < min_sector_coverage
    }
    if low_sectors:
        metrics.inc(
            "playground_sector_coverage_validation_total",
            "Sector coverage validation outcomes",
            labels={"status": "error"},
        )
        raise MacroCoverageError(
            "Sector coverage below threshold "
            f"{min_sector_coverage:.3f}: "
            + ", ".join(
                f"{sector}={ratio:.3f}" for sector, ratio in sorted(low_sectors.items())
            ),
        )

    metrics.inc(
        "playground_sector_coverage_validation_total",
        "Sector coverage validation outcomes",
        labels={"status": "success"},
    )

    validator = MacroCoverageValidator(min_coverage=min_factor_coverage)
    try:
        validator.validate_macro_coverage(factor_levels, factor_columns)
    except MacroCoverageError:
        metrics.inc(
            "playground_factor_coverage_validation_total",
            "Factor coverage validation outcomes",
            labels={"status": "error"},
        )
        raise

    metrics.inc(
        "playground_factor_coverage_validation_total",
        "Factor coverage validation outcomes",
        labels={"status": "success"},
    )


def _build_coverage_alerts(
    coverage: CoverageSummary,
    *,
    min_sector_threshold: float,
    min_factor_threshold: float,
) -> dict[str, dict[str, float]]:
    """Compile alerts for any series falling below configured coverage thresholds."""
    sector_alerts = {
        sector: ratio
        for sector, ratio in coverage.sector_coverage.items()
        if ratio < min_sector_threshold
    }

    factor_alerts = {
        name: ratio
        for name, ratio in coverage.factor_coverage.items()
        if ratio < min_factor_threshold
    }

    composite_alerts = {
        name: ratio
        for name, ratio in coverage.composite_coverage.items()
        if ratio < min_factor_threshold
    }

    return {
        "sector": sector_alerts,
        "factor": factor_alerts,
        "composite": composite_alerts,
    }


def _record_coverage_metrics(
    metrics: MetricsManager,
    alerts: Mapping[str, Mapping[str, float]],
) -> None:
    """Publish coverage alert metrics for observability dashboards."""
    buckets = {
        "sector": alerts.get("sector", {}),
        "factor": alerts.get("factor", {}),
        "composite": alerts.get("composite", {}),
    }
    total_metric = "playground_coverage_alert_total"
    ratio_metric = "playground_coverage_alert_ratio"
    total_desc = "Total number of coverage alert entries by dimension"
    ratio_desc = "Coverage ratio for entries falling below configured thresholds"

    for dimension, bucket in buckets.items():
        metrics.set_gauge(
            total_metric,
            total_desc,
            float(len(bucket)),
            labels={"dimension": dimension},
        )
        for series, ratio in bucket.items():
            metrics.set_gauge(
                ratio_metric,
                ratio_desc,
                float(ratio),
                labels={"dimension": dimension, "series": series},
            )


__all__ = ["RiskPipelineConfig", "RiskPipelineResult", "run_risk_pipeline"]
