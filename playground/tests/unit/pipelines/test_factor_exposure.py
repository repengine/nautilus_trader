"""Tests for the factor exposure pipeline."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime

import polars as pl
import pytest

from playground.exposure.factor_exposure import FactorExposureConfig
from playground.exposure.factor_exposure import compute_factor_exposures
from playground.exposure.factor_exposure import prepare_factor_returns


@pytest.fixture()
def timestamps() -> list[datetime]:
    return [
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 2, tzinfo=UTC),
        datetime(2024, 1, 3, tzinfo=UTC),
    ]


def test_prepare_factor_returns_produces_pct_change(timestamps: list[datetime]) -> None:
    factor_levels = pl.DataFrame(
        {
            "timestamp": timestamps,
            "credit": [1.0, 1.5, 1.2],
            "liquidity": [0.5, 0.6, 0.9],
        },
    )

    factor_returns = prepare_factor_returns(factor_levels, columns=["credit", "liquidity"])

    assert factor_returns.height == 2
    credit_returns = factor_returns.select("credit").to_series().to_list()
    assert credit_returns == [0.5, pytest.approx(-0.2)]


def test_compute_factor_exposures_matches_expected_values(timestamps: list[datetime]) -> None:
    asset_returns = pl.DataFrame(
        {
            "timestamp": timestamps * 2,
            "symbol": ["AAA"] * 3 + ["BBB"] * 3,
            "return": [0.01, 0.02, -0.01, 0.005, 0.007, 0.009],
        },
    )

    factor_returns = pl.DataFrame(
        {
            "timestamp": timestamps,
            "factor_credit": [0.005, 0.015, -0.005],
            "factor_liquidity": [0.004, 0.006, 0.002],
        },
    )

    config = FactorExposureConfig(feature_set_id="test_set", alpha=0.5, source="backfill", ts_init=datetime(2024, 1, 10, tzinfo=UTC))
    exposures = compute_factor_exposures(asset_returns, factor_returns, config)

    assert exposures.height == 3 * 2 * 2  # timesteps * assets * factors
    assert set(exposures.columns) == {
        "feature_set_id",
        "asset_id",
        "benchmark_id",
        "ts_event",
        "ts_init",
        "ewma_beta",
        "ewma_cov",
        "ewma_var_market",
        "n_observations",
        "alpha",
        "source",
    }

    credit_exposure = (
        exposures
        .filter((pl.col("asset_id") == "AAA") & (pl.col("benchmark_id") == "factor_credit"))
        .sort("ts_event")
        .select("ewma_beta")
        .to_series()
        .to_list()
    )
    assert credit_exposure == [pytest.approx(2.0), pytest.approx(1.4), pytest.approx(1.5)]

    observations = exposures.filter(pl.col("asset_id") == "AAA").select("n_observations").to_series().to_list()
    assert observations[-1] == 3
    assert exposures.select("alpha").unique()[0, 0] == config.alpha
    assert exposures.select("source").unique()[0, 0] == config.source
