"""Tests for the risk model pipeline orchestration."""

from __future__ import annotations

import json
from collections.abc import Iterable
from collections.abc import Mapping
from datetime import UTC
from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from ml.common.metrics_manager import MetricsManager
from ml.data.validation import MacroCoverageError
from playground.risk_model.dataset import CoverageSummary
from playground.risk_model.pipeline import RiskPipelineConfig
from playground.risk_model.pipeline import _build_coverage_alerts
from playground.risk_model.pipeline import _record_coverage_metrics
from playground.risk_model.pipeline import run_risk_pipeline


class _StubSectorFetcher:
    def __init__(self, frame: pl.DataFrame) -> None:
        self._frame = frame

    def __call__(self, request) -> pl.DataFrame:  # type: ignore[no-untyped-def]
        return self._frame


class _StubFactorFetcher:
    def __init__(self, frame: pl.DataFrame) -> None:
        self._frame = frame

    def __call__(self, request) -> pl.DataFrame:  # type: ignore[no-untyped-def]
        return self._frame


@pytest.fixture()
def _sample_dataset() -> tuple[pl.DataFrame, pl.DataFrame]:
    date_index = pd.date_range("2020-01-01", "2020-12-31", freq="B", tz=UTC)
    timestamps = [ts.to_pydatetime() for ts in date_index]
    sector_returns = pl.DataFrame(
        {
            "timestamp": timestamps + timestamps,
            "symbol": ["XLF"] * len(timestamps) + ["XLK"] * len(timestamps),
            "return": [0.01] * len(timestamps) + [0.02] * len(timestamps),
        },
    )
    factor_levels = pl.DataFrame(
        {
            "timestamp": timestamps,
            "factor_duration": [1.0 + index * 0.0001 for index in range(len(timestamps))],
            "factor_credit": [0.5 + index * 0.0001 for index in range(len(timestamps))],
            "factor_liquidity": [0.2 + index * 0.0001 for index in range(len(timestamps))],
        },
    )
    return sector_returns, factor_levels


def test_run_risk_pipeline_produces_profiles(tmp_path: Path, _sample_dataset: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    sector_returns, factor_levels = _sample_dataset
    sector_fetcher = _StubSectorFetcher(sector_returns)
    factor_fetcher = _StubFactorFetcher(factor_levels)

    config = RiskPipelineConfig(
        sectors=("XLF", "XLK"),
        factor_columns=("factor_duration", "factor_credit", "factor_liquidity"),
        start=datetime(2020, 1, 1, tzinfo=UTC),
        end=datetime(2020, 12, 31, tzinfo=UTC),
        feature_set_id="unit-test",
        min_weight=0.01,
        persist_dir=tmp_path / "dataset",
        cache_dir=tmp_path / "cache",
        visualization_dir=tmp_path / "vis",
        notes="unit-test",
    )

    result = run_risk_pipeline(
        config,
        sector_fetcher_override=sector_fetcher,
        factor_fetcher_override=factor_fetcher,
    )

    assert not result.sector_returns.is_empty()
    assert not result.factor_levels.is_empty()
    assert not result.exposures.is_empty()
    assert len(result.profiles) == 1
    assert set(result.distance_reports) == {2020}
    assert result.coverage_summary.expected_days > 0
    assert "XLF" in result.coverage_summary.sector_coverage
    assert result.coverage_summary.sector_coverage["XLF"] >= 0.0
    assert result.coverage_summary.composite_coverage == {}
    assert result.coverage_alerts == {
        "sector": {},
        "factor": {},
        "composite": {},
    }
    assert result.eigenvalue_trends
    assert {profile.status for profile in result.profiles} <= {"success", "fallback"}
    assert result.beta_persisted_rows == 0
    assert result.optimizer_recommendations
    weights_2020 = result.optimizer_recommendations[2020]
    assert pytest.approx(sum(weights_2020.values())) == 1.0

    vis_file = tmp_path / "vis" / "risk_2020.json"
    assert vis_file.exists()
    assert vis_file.read_text(encoding="utf-8") != ""

    for payload in result.visualization_payloads.values():
        assert payload.metadata["notes"] == "unit-test"
        assert payload.metadata["status"] in {"success", "fallback"}
        assert "coverage" in payload.metadata
        coverage_meta = payload.metadata["coverage"]
        assert coverage_meta["sector_expected_days"] == result.coverage_summary.sector_expected_days
        alerts_meta = payload.metadata.get("coverage_alerts", {})
        assert isinstance(alerts_meta, dict)
        assert alerts_meta.get("sector", {}) == {}
        assert alerts_meta.get("factor", {}) == {}
        assert alerts_meta.get("composite", {}) == {}
        assert "eigenvalue_trends" in payload.metadata
        assert set(payload.sectors[0]) >= {"sector", "mahalanobis_distance"}
        mahal = payload.sectors[0]["mahalanobis_distance"]
        assert mahal is None or isinstance(mahal, float)

    alerts_path = (tmp_path / "dataset") / "coverage_alerts.json"
    assert alerts_path.exists()
    persisted_alerts = json.loads(alerts_path.read_text(encoding="utf-8"))
    assert persisted_alerts["coverage_alerts"] == result.coverage_alerts
    assert persisted_alerts["min_sector_threshold"] == pytest.approx(config.min_sector_coverage)
    assert persisted_alerts["min_factor_threshold"] == pytest.approx(config.min_factor_coverage)


def test_run_risk_pipeline_raises_when_factor_coverage_below_min(tmp_path: Path) -> None:
    timestamps = [
        datetime(2020, 1, 1, tzinfo=UTC),
        datetime(2020, 1, 2, tzinfo=UTC),
        datetime(2020, 1, 3, tzinfo=UTC),
    ]
    sector_returns = pl.DataFrame(
        {
            "timestamp": timestamps + timestamps,
            "symbol": ["XLF"] * len(timestamps) + ["XLK"] * len(timestamps),
            "return": [0.01, 0.015, 0.02, 0.02, 0.018, 0.022],
        },
    )
    factor_levels = pl.DataFrame(
        {
            "timestamp": timestamps,
            "factor_duration": [1.0, None, 1.2],
            "factor_credit": [0.5, 0.55, 0.6],
            "factor_liquidity": [0.2, 0.25, 0.3],
        },
    )

    config = RiskPipelineConfig(
        sectors=("XLF", "XLK"),
        factor_columns=("factor_duration", "factor_credit", "factor_liquidity"),
        start=timestamps[0],
        end=timestamps[-1],
        feature_set_id="unit-test",
        min_weight=0.01,
        persist_dir=tmp_path / "dataset",
        cache_dir=tmp_path / "cache",
        visualization_dir=tmp_path / "vis",
        min_factor_coverage=0.9,
        min_sector_coverage=0.5,
    )

    sector_fetcher = _StubSectorFetcher(sector_returns)
    factor_fetcher = _StubFactorFetcher(factor_levels)

    with pytest.raises(MacroCoverageError):
        run_risk_pipeline(
            config,
            sector_fetcher_override=sector_fetcher,
            factor_fetcher_override=factor_fetcher,
        )


def test_run_risk_pipeline_raises_when_sector_coverage_below_min(tmp_path: Path) -> None:
    timestamps = [
        datetime(2020, 1, 1, tzinfo=UTC),
        datetime(2020, 1, 2, tzinfo=UTC),
        datetime(2020, 1, 3, tzinfo=UTC),
    ]
    sector_returns = pl.DataFrame(
        {
            "timestamp": [timestamps[0]],
            "symbol": ["XLF"],
            "return": [0.01],
        },
    )
    factor_levels = pl.DataFrame(
        {
            "timestamp": timestamps,
            "factor_duration": [1.0, 1.1, 1.2],
            "factor_credit": [0.5, 0.55, 0.6],
            "factor_liquidity": [0.2, 0.25, 0.3],
        },
    )

    config = RiskPipelineConfig(
        sectors=("XLF",),
        factor_columns=("factor_duration", "factor_credit", "factor_liquidity"),
        start=timestamps[0],
        end=datetime(2020, 1, 10, tzinfo=UTC),
        feature_set_id="unit-test",
        min_weight=0.01,
        persist_dir=tmp_path / "dataset",
        cache_dir=tmp_path / "cache",
        visualization_dir=tmp_path / "vis",
        min_sector_coverage=0.9,
        min_factor_coverage=0.0,
    )

    sector_fetcher = _StubSectorFetcher(sector_returns)
    factor_fetcher = _StubFactorFetcher(factor_levels)

    with pytest.raises(MacroCoverageError):
        run_risk_pipeline(
            config,
            sector_fetcher_override=sector_fetcher,
            factor_fetcher_override=factor_fetcher,
        )


def test_record_coverage_metrics_emits_gauges() -> None:
    metrics = _RecordingMetricsManager()
    alerts = {
        "sector": {},
        "factor": {"factor_duration": 0.75},
        "composite": {"macro_liquidity": 0.58, "macro_credit": 0.81},
    }

    _record_coverage_metrics(metrics, alerts)

    def _get_value(name: str, dimension: str, series: str | None = None) -> float:
        for metric_name, labels, value in metrics.gauge_records:
            if metric_name != name:
                continue
            if labels.get("dimension") != dimension:
                continue
            if series is not None and labels.get("series") != series:
                continue
            if series is None and "series" in labels:
                continue
            return value
        msg = (
            f"Metric '{name}' with dimension '{dimension}'"
            + (f" and series '{series}'" if series else "")
            + " not recorded"
        )
        raise AssertionError(msg)

    # Totals should be reported for every dimension, including zero when no alerts exist.
    assert _get_value("playground_coverage_alert_total", "sector") == pytest.approx(0.0)
    assert _get_value("playground_coverage_alert_total", "factor") == pytest.approx(1.0)
    assert _get_value("playground_coverage_alert_total", "composite") == pytest.approx(2.0)

    # Ratios must be recorded for each alerted series with original values.
    assert _get_value(
        "playground_coverage_alert_ratio",
        "factor",
        "factor_duration",
    ) == pytest.approx(0.75)
    assert _get_value(
        "playground_coverage_alert_ratio",
        "composite",
        "macro_liquidity",
    ) == pytest.approx(0.58)
    assert _get_value(
        "playground_coverage_alert_ratio",
        "composite",
        "macro_credit",
    ) == pytest.approx(0.81)


def test_build_coverage_alerts_highlights_deficits() -> None:
    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=10,
        factor_expected_days=10,
        sector_coverage={"XLF": 0.5, "XLK": 0.95},
        factor_coverage={"factor_credit": 0.7, "factor_duration": 0.95},
        composite_coverage={"credit_spread_hy": 0.6, "term_spread": 0.92},
    )

    alerts = _build_coverage_alerts(
        coverage,
        min_sector_threshold=0.9,
        min_factor_threshold=0.8,
    )

    assert alerts["sector"] == {"XLF": 0.5}
    assert alerts["factor"] == {"factor_credit": 0.7}
    assert alerts["composite"] == {"credit_spread_hy": 0.6}


class _RecordingMetricsManager(MetricsManager):
    def __init__(self) -> None:
        super().__init__()
        self.gauge_records: list[tuple[str, dict[str, object], float]] = []

    def set_gauge(
        self,
        name: str,
        description: str,
        value: float,
        *,
        labels: Mapping[str, object] | None = None,
        labelnames: Iterable[str] | None = None,
    ) -> None:
        record = (name, dict(labels or {}), float(value))
        self.gauge_records.append(record)
        super().set_gauge(name, description, value, labels=labels, labelnames=labelnames)
