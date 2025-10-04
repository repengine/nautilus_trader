"""Tests for the factor exposure optimizer."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime

import numpy as np
import polars as pl
import pytest

from playground.exposure.optimizer import RiskPoint
from playground.exposure.optimizer import compute_optimal_weights
from playground.exposure.optimizer import default_target_point


@pytest.fixture()
def exposures_df() -> pl.DataFrame:
    timestamp_ns = int(datetime(2024, 1, 2, tzinfo=UTC).timestamp() * 1_000_000_000)
    base = pl.DataFrame(
        {
            "asset_id": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"],
            "benchmark_id": [
                "factor_duration",
                "factor_credit",
                "factor_duration",
                "factor_credit",
                "factor_duration",
                "factor_credit",
            ],
            "ewma_beta": [0.6, 0.2, 0.2, 0.5, 0.1, 0.2],
            "ts_event": [timestamp_ns] * 6,
        },
    )
    liquidity = pl.DataFrame(
        {
            "asset_id": ["AAA", "BBB", "CCC"],
            "benchmark_id": ["factor_liquidity"] * 3,
            "ewma_beta": [0.1, 0.3, 0.7],
            "ts_event": [timestamp_ns] * 3,
        },
    )
    return base.vstack(liquidity)


def test_compute_optimal_weights_matches_target(exposures_df: pl.DataFrame) -> None:
    target = RiskPoint(
        {
            "factor_duration": 0.3,
            "factor_credit": 0.4,
            "factor_liquidity": 0.3,
        },
    )

    weights = compute_optimal_weights(exposures_df, target)

    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    assert all(weight >= 0 for weight in weights.values())

    asset_ids = ["AAA", "BBB", "CCC"]
    exposure_matrix = np.array([
        [0.6, 0.2, 0.1],
        [0.2, 0.5, 0.3],
        [0.1, 0.2, 0.7],
    ])
    weight_vector = np.array([weights[asset] for asset in asset_ids])
    portfolio_exposure = exposure_matrix.T @ weight_vector
    assert portfolio_exposure == pytest.approx(np.array([0.3, 0.4, 0.3]), abs=0.015)


def test_default_target_point_coordinates() -> None:
    target = default_target_point()
    vector = target.to_vector(["factor_duration", "factor_credit", "factor_liquidity"])
    assert vector.sum() == pytest.approx(1.0)
