"""Tests covering risk model analysis functions."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime

import numpy as np
import polars as pl
import pytest

from playground.exposure.factor_exposure import FactorExposureConfig
from playground.exposure.factor_exposure import compute_factor_exposures
from playground.exposure.factor_exposure import prepare_factor_returns
from playground.exposure.optimizer import RiskPoint
from playground.exposure.optimizer import compute_optimal_weights
from playground.risk_model.analysis import compute_annual_risk_profiles
from playground.risk_model.analysis import compute_sector_distance_reports
from playground.risk_model.analysis import summarize_eigenvalue_trends
from playground.risk_model.analysis import summarize_sector_exposures


@pytest.fixture()
def _sample_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    timestamps = [
        datetime(2020, 1, 1, tzinfo=UTC),
        datetime(2020, 1, 2, tzinfo=UTC),
        datetime(2020, 1, 3, tzinfo=UTC),
    ]
    sector_returns = pl.DataFrame(
        {
            "timestamp": timestamps * 2,
            "symbol": ["XLF"] * 3 + ["XLK"] * 3,
            "return": [0.01, 0.02, 0.015, 0.005, 0.007, 0.009],
        },
    )
    factor_levels = pl.DataFrame(
        {
            "timestamp": timestamps,
            "factor_duration": [1.00, 1.02, 1.01],
            "factor_credit": [0.99, 1.01, 1.00],
            "factor_liquidity": [1.0, 1.001, 1.002],
        },
    )
    return sector_returns, factor_levels


def test_compute_annual_risk_profiles_and_distances(_sample_data: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    sector_returns, factor_levels = _sample_data

    config = FactorExposureConfig(feature_set_id="unit_test", alpha=0.5)
    factor_columns = ("factor_duration", "factor_credit", "factor_liquidity")

    profiles = compute_annual_risk_profiles(
        sector_returns,
        factor_levels,
        factor_columns=factor_columns,
        exposure_config=config,
        min_weight=0.05,
    )

    assert len(profiles) == 1
    profile = profiles[0]
    assert profile.year == 2020
    assert pytest.approx(sum(profile.weights.values())) == 1.0
    assert set(profile.risk_point.coordinates) == set(factor_columns)
    assert profile.status in {"success", "fallback"}
    assert profile.diagnostics is not None
    assert "cov_eigenvalues" in profile.diagnostics
    assert profile.diagnostics.get("optimizer") is not None

    factor_returns = prepare_factor_returns(factor_levels, columns=factor_columns)
    exposures = compute_factor_exposures(sector_returns, factor_returns, config)
    summaries = summarize_sector_exposures(exposures, factor_names=factor_columns)
    assert set(summaries) == {"XLF", "XLK"}
    assert summaries["XLF"].observation_count > 0

    distance_reports = compute_sector_distance_reports(
        exposures,
        profiles,
        factor_columns=factor_columns,
    )
    assert set(distance_reports.keys()) == {2020}
    report_entries = distance_reports[2020]
    assert len(report_entries) == 2
    for report in report_entries:
        assert report.distance >= 0
        assert set(report.coordinates) == set(factor_columns)
        assert 0.0 <= report.recommended_weight <= 1.0
        assert report.mahalanobis_distance is None or report.mahalanobis_distance >= 0

    eigen_trends = summarize_eigenvalue_trends(profiles)
    assert eigen_trends
    decade_key = next(iter(eigen_trends))
    assert decade_key.endswith("s")


def test_compute_annual_risk_profiles_handles_optimizer_failure(
    monkeypatch: pytest.MonkeyPatch,
    _sample_data: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    sector_returns, factor_levels = _sample_data
    config = FactorExposureConfig(feature_set_id="unit_test", alpha=0.5)
    factor_columns = ("factor_duration", "factor_credit", "factor_liquidity")

    def _fail(*args: object, **kwargs: object) -> dict[str, float]:
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(
        "playground.risk_model.analysis.compute_optimal_weights",
        _fail,
    )

    profiles = compute_annual_risk_profiles(
        sector_returns,
        factor_levels,
        factor_columns=factor_columns,
        exposure_config=config,
        min_weight=0.05,
    )

    assert len(profiles) == 1
    profile = profiles[0]
    assert profile.status == "fallback"
    assert profile.diagnostics is not None
    assert profile.diagnostics.get("reason") == "singular"
    assert "cov_eigenvalues" in profile.diagnostics
    optimizer_diag = profile.diagnostics.get("optimizer")
    assert optimizer_diag is not None
    assert optimizer_diag.get("status") == "fallback"


def test_compute_optimal_weights_respects_caps() -> None:
    ts = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1_000_000_000)
    exposures = pl.DataFrame(
        {
            "asset_id": ["SEC_A", "SEC_A", "SEC_B", "SEC_B"],
            "benchmark_id": [
                "factor_duration",
                "factor_credit",
                "factor_duration",
                "factor_credit",
            ],
            "ts_event": [ts, ts, ts, ts],
            "ewma_beta": [0.5, 0.3, 0.2, 0.6],
        },
    )
    target = RiskPoint({"factor_duration": 0.4, "factor_credit": 0.6})

    weights = compute_optimal_weights(
        exposures,
        target,
        min_weight=0.0,
        max_weight=0.75,
        weight_caps={"SEC_A": 0.55},
    )

    assert pytest.approx(sum(weights.values())) == 1.0
    assert 0.0 <= weights["SEC_A"] <= 0.55
    assert weights["SEC_B"] <= 0.75
