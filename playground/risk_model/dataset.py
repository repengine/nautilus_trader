"""Dataset assembly helpers for the 3D sector risk model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import polars as pl
import structlog

from ml._imports import mcal
from ml.common.metrics_manager import MetricsManager
from ml.features.macro_composites import get_composite_feature_names


LOGGER = structlog.get_logger(__name__)


@dataclass(slots=True)
class SectorDataRequest:
    """Describe the sector universe and time range to assemble."""

    sectors: tuple[str, ...]
    start: datetime
    end: datetime
    frequency: str = "1d"
    price_column: str = "adj_close"
    calendar: str = "XNYS"

    def __post_init__(self) -> None:
        if not self.sectors:
            msg = "At least one sector symbol must be provided"
            raise ValueError(msg)
        if self.end <= self.start:
            msg = "End timestamp must be after the start timestamp"
            raise ValueError(msg)
        if self.start.tzinfo is None:
            self.start = self.start.replace(tzinfo=UTC)
        if self.end.tzinfo is None:
            self.end = self.end.replace(tzinfo=UTC)
        if not self.calendar.strip():
            msg = "Trading calendar identifier must be a non-empty string"
            raise ValueError(msg)
        self.calendar = self.calendar.strip()


@dataclass(slots=True)
class FactorDataRequest:
    """Describe factor columns to align with sector returns."""

    factor_columns: tuple[str, ...]
    start: datetime
    end: datetime
    calendar: str = "XNYS"

    def __post_init__(self) -> None:
        if not self.factor_columns:
            msg = "Factor columns must be provided"
            raise ValueError(msg)
        if self.end <= self.start:
            msg = "End timestamp must be after the start timestamp"
            raise ValueError(msg)
        if self.start.tzinfo is None:
            self.start = self.start.replace(tzinfo=UTC)
        if self.end.tzinfo is None:
            self.end = self.end.replace(tzinfo=UTC)
        if not self.calendar.strip():
            msg = "Trading calendar identifier must be a non-empty string"
            raise ValueError(msg)
        self.calendar = self.calendar.strip()


@dataclass(slots=True)
class SectorDataset:
    """Container holding sector return and factor series aligned to a timeline."""

    sector_returns: pl.DataFrame
    factor_returns: pl.DataFrame
    coverage: CoverageSummary


@dataclass(slots=True)
class CoverageSummary:
    """Coverage metrics for sector and factor series."""

    calendar_name: str
    sector_expected_days: int
    factor_expected_days: int
    sector_coverage: dict[str, float]
    factor_coverage: dict[str, float]
    composite_coverage: dict[str, float] = field(default_factory=dict)

    @property
    def expected_days(self) -> int:
        """Alias for backwards compatibility with earlier API."""
        return self.sector_expected_days

    def to_dict(self) -> dict[str, Any]:
        """Serialize the summary into standard Python types."""
        return {
            "calendar_name": self.calendar_name,
            "sector_expected_days": self.sector_expected_days,
            "factor_expected_days": self.factor_expected_days,
            "sector_coverage": self.sector_coverage,
            "factor_coverage": self.factor_coverage,
            "composite_coverage": self.composite_coverage,
        }


class SectorReturnFetcher(Protocol):
    """Protocol describing callables that provide sector returns."""

    def __call__(self, request: SectorDataRequest) -> pl.DataFrame: ...


class FactorReturnFetcher(Protocol):
    """Protocol describing callables that provide factor return series."""

    def __call__(self, request: FactorDataRequest) -> pl.DataFrame: ...


class SectorDatasetAssembler:
    """Orchestrate loading and aligning sector/factor datasets with metrics."""

    def __init__(
        self,
        sector_fetcher: SectorReturnFetcher,
        factor_fetcher: FactorReturnFetcher,
        *,
        metrics: MetricsManager | None = None,
    ) -> None:
        self._sector_fetcher = sector_fetcher
        self._factor_fetcher = factor_fetcher
        self._metrics = metrics or MetricsManager.default()

    def _enforce_time_bounds(
        self,
        frame: pl.DataFrame,
        *,
        start: datetime,
        end: datetime,
        kind: str,
    ) -> pl.DataFrame:
        if frame.is_empty():
            return frame
        total = frame.height
        future_rows = frame.filter(pl.col("timestamp") > end).height
        past_rows = frame.filter(pl.col("timestamp") < start).height
        self._record_trim_metric(
            kind=kind,
            reason="future",
            trimmed=future_rows,
            total=total,
            start=start,
            end=end,
        )
        self._record_trim_metric(
            kind=kind,
            reason="past",
            trimmed=past_rows,
            total=total,
            start=start,
            end=end,
        )
        return _clip_time_range(frame, start=start, end=end)

    def _record_trim_metric(
        self,
        *,
        kind: str,
        reason: str,
        trimmed: int,
        total: int,
        start: datetime,
        end: datetime,
    ) -> None:
        if trimmed <= 0 or total <= 0:
            return
        ratio = float(trimmed) / float(total)
        self._metrics.observe(
            "playground_dataset_trim_ratio",
            "Fraction of dataset rows removed when enforcing time bounds.",
            ratio,
            labels={"kind": kind, "reason": reason},
        )
        LOGGER.warning(
            "Dropped observations outside requested range",
            kind=kind,
            reason=reason,
            trimmed=trimmed,
            total=total,
            start=start.isoformat(),
            end=end.isoformat(),
        )

    def build(
        self,
        sector_request: SectorDataRequest,
        factor_request: FactorDataRequest,
        *,
        persist_dir: Path | None = None,
    ) -> SectorDataset:
        """Fetch sector and factor data and align on the overlapping date range."""
        LOGGER.info(
            "Building sector dataset",
            sectors=sector_request.sectors,
            start=sector_request.start,
            end=sector_request.end,
        )

        try:
            sector_returns = self._sector_fetcher(sector_request)
            factor_returns = self._factor_fetcher(factor_request)

            aligned_sector = self._enforce_time_bounds(
                sector_returns,
                start=sector_request.start,
                end=sector_request.end,
                kind="sector",
            )
            aligned_factors = self._enforce_time_bounds(
                factor_returns,
                start=factor_request.start,
                end=factor_request.end,
                kind="factor",
            )

            dataset = _align_on_common_timestamps(
                aligned_sector,
                aligned_factors,
                sector_request=sector_request,
                factor_request=factor_request,
                metrics=self._metrics,
            )

            if persist_dir is not None:
                persist_dir.mkdir(parents=True, exist_ok=True)
                dataset.sector_returns.write_parquet(persist_dir / "sector_returns.parquet")
                dataset.factor_returns.write_parquet(persist_dir / "factor_returns.parquet")
                coverage_path = persist_dir / "coverage_summary.json"
                coverage_path.write_text(
                    f"{_serialize_json(dataset.coverage.to_dict())}\n",
                    encoding="utf-8",
                )
        except Exception:
            self._metrics.inc(
                "playground_sector_dataset_build_total",
                "Count of sector dataset builds",
                labels={"status": "error"},
            )
            LOGGER.exception("Failed to assemble sector dataset")
            raise

        self._metrics.inc(
            "playground_sector_dataset_build_total",
            "Count of sector dataset builds",
            labels={"status": "success"},
        )

        return dataset


def _clip_time_range(
    frame: pl.DataFrame,
    *,
    start: datetime,
    end: datetime,
) -> pl.DataFrame:
    return (
        frame
        .filter(pl.col("timestamp") >= start)
        .filter(pl.col("timestamp") <= end)
        .sort("timestamp")
    )


def _align_on_common_timestamps(
    sector_returns: pl.DataFrame,
    factor_returns: pl.DataFrame,
    *,
    sector_request: SectorDataRequest,
    factor_request: FactorDataRequest,
    metrics: MetricsManager,
) -> SectorDataset:
    sector_columns = {"timestamp", "symbol", "return"}
    if not sector_columns.issubset(sector_returns.columns):
        missing = sector_columns - set(sector_returns.columns)
        msg = f"Sector returns missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    factor_columns = [col for col in factor_returns.columns if col != "timestamp"]
    if not factor_columns:
        raise ValueError("Factor returns must contain at least one factor column")

    merged = sector_returns.join(factor_returns, on="timestamp", how="inner")
    if merged.is_empty():
        raise ValueError("No overlapping timestamps between sector and factor data")

    merged = merged.sort(["timestamp", "symbol"]).with_columns(
        pl.col("timestamp").alias("timestamp"),
    )

    sector_aligned = merged.select(["timestamp", "symbol", "return"])
    factor_aligned = (
        merged
        .select(["timestamp", *factor_columns])
        .unique(subset=["timestamp"], keep="first")
        .sort("timestamp")
    )

    coverage = _compute_coverage(
        sector_aligned,
        factor_aligned,
        sector_request=sector_request,
        factor_request=factor_request,
        metrics=metrics,
    )

    return SectorDataset(
        sector_returns=sector_aligned,
        factor_returns=factor_aligned,
        coverage=coverage,
    )


def _compute_coverage(
    sector_returns: pl.DataFrame,
    factor_returns: pl.DataFrame,
    *,
    sector_request: SectorDataRequest,
    factor_request: FactorDataRequest,
    metrics: MetricsManager,
) -> CoverageSummary:
    sector_expected_days, calendar_name = _expected_sessions(
        sector_request.start,
        sector_request.end,
        calendar_name=sector_request.calendar,
    )
    effective_expected = max(sector_expected_days - 1, 1)

    sector_ratios: dict[str, float] = {}
    for sector in sector_request.sectors:
        observed = (
            sector_returns
            .filter(pl.col("symbol") == sector)
            .height
        )
        ratio = float(observed) / float(effective_expected)
        ratio = max(0.0, min(1.0, ratio))
        sector_ratios[sector] = ratio
        metrics.observe(
            "playground_sector_coverage_ratio",
            "Ratio of observed sector returns versus expected sessions",
            ratio,
            labels={"sector": sector},
        )
        if ratio < 0.8:
            LOGGER.warning(
                "Low sector coverage",
                sector=sector,
                ratio=ratio,
            )

    factor_ratios: dict[str, float] = {}
    factor_expected_days, _ = _expected_sessions(
        factor_request.start,
        factor_request.end,
        calendar_name=factor_request.calendar,
    )
    factor_effective_expected = max(factor_expected_days - 1, 1)

    for column in factor_returns.columns:
        if column == "timestamp":
            continue
        observed = (
            factor_returns.select(
                pl.col(column).is_not_null().sum().alias("observed"),
            ).get_column("observed")[0]
        )
        ratio = float(observed) / float(factor_effective_expected)
        ratio = max(0.0, min(1.0, ratio))
        factor_ratios[column] = ratio
        metrics.observe(
            "playground_factor_coverage_ratio",
            "Ratio of observed factor values versus expected sessions",
            ratio,
            labels={"factor": column},
        )
        if ratio < 0.8:
            LOGGER.warning(
                "Low factor coverage",
                factor=column,
                ratio=ratio,
            )

    composite_names = set(get_composite_feature_names())
    composite_ratios = {
        name: factor_ratios[name]
        for name in composite_names
        if name in factor_ratios
    }

    return CoverageSummary(
        calendar_name=calendar_name,
        sector_expected_days=sector_expected_days,
        factor_expected_days=factor_expected_days,
        sector_coverage=sector_ratios,
        factor_coverage=factor_ratios,
        composite_coverage=composite_ratios,
    )


def _expected_sessions(start: datetime, end: datetime, *, calendar_name: str) -> tuple[int, str]:
    if mcal is None:
        LOGGER.warning(
            "pandas-market-calendars unavailable; using weekday-only business day count",
            calendar=calendar_name,
        )
        start_date = start.date()
        end_date = end.date()
        expected = int(pd.date_range(start=start_date, end=end_date, freq="B").size)
        return expected, "WEEKDAY"

    try:
        calendar = mcal.get_calendar(calendar_name)
    except Exception:  # pragma: no cover - invalid calendar configuration
        LOGGER.exception("Failed to load trading calendar", calendar=calendar_name)
        start_date = start.date()
        end_date = end.date()
        expected = int(pd.date_range(start=start_date, end=end_date, freq="B").size)
        return expected, "WEEKDAY"

    start_ts = pd.Timestamp(start.date())
    end_ts = pd.Timestamp(end.date())
    schedule = calendar.schedule(start_date=start_ts, end_date=end_ts)
    session_count = len(schedule.index)
    return session_count, calendar_name


def _serialize_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)


__all__ = [
    "CoverageSummary",
    "FactorDataRequest",
    "FactorReturnFetcher",
    "SectorDataRequest",
    "SectorDataset",
    "SectorDatasetAssembler",
    "SectorReturnFetcher",
]
