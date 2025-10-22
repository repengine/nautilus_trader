"""Microbenchmarks for turnover smoothing allocation path."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import TypeVar, cast

import polars as pl
import pytest

from playground.backtest.strategies import FactorTiltStrategy
from playground.risk_model.dataset import CoverageSummary
from playground.risk_model.dataset import SectorDataset


F = TypeVar("F", bound=Callable[..., object])


def performance_test(func: F) -> F:
    """Typed wrapper around pytest performance marker."""
    return cast(F, pytest.mark.performance(func))


def _create_mock_sector_dataset(num_days: int = 252) -> SectorDataset:
    """Create a lightweight dataset for benchmarking turnover smoothing."""
    start_date = datetime(2022, 1, 1, tzinfo=UTC)
    sectors = ["SPY", "AGG", "XLK"]

    sector_records: list[dict[str, object]] = []
    for offset in range(num_days):
        current = start_date + timedelta(days=offset)
        for index, symbol in enumerate(sectors):
            sector_records.append({
                "timestamp": current,
                "symbol": symbol,
                "return": 0.0005 - 0.0001 * index,
            })
    sector_returns = pl.DataFrame(sector_records)

    timestamps = [start_date + timedelta(days=offset) for offset in range(num_days)]
    factor_returns = pl.DataFrame({
        "timestamp": timestamps,
        "factor_duration": [0.0001 * ((-1) ** offset) for offset in range(num_days)],
        "factor_credit": [0.00005 * ((-1) ** (offset + 1)) for offset in range(num_days)],
        "factor_liquidity": [0.00008 * ((-1) ** (offset // 2)) for offset in range(num_days)],
    })

    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=num_days,
        factor_expected_days=num_days,
        sector_coverage=dict.fromkeys(sectors, 1.0),
        factor_coverage={
            "factor_duration": 1.0,
            "factor_credit": 1.0,
            "factor_liquidity": 1.0,
        },
    )

    return SectorDataset(
        sector_returns=sector_returns,
        factor_returns=factor_returns,
        coverage=coverage,
    )


@performance_test
def test_turnover_smoothing_compute_weights_microbench() -> None:
    """Ensure turnover smoothing weight computation stays within the microbench target."""
    dataset = _create_mock_sector_dataset()
    strategy = FactorTiltStrategy(
        use_rolling_betas=False,
        turnover_smoothing=0.55,
        max_weight=0.40,
        factor_forecasts={
            "factor_duration": 0.01,
            "factor_credit": 0.005,
            "factor_liquidity": 0.002,
        },
    )

    evaluation_dates = [
        datetime(2022, 6, 1, tzinfo=UTC),
        datetime(2022, 9, 1, tzinfo=UTC),
        datetime(2022, 12, 1, tzinfo=UTC),
    ]

    iterations = 250
    start = time.perf_counter()
    for index in range(iterations):
        date = evaluation_dates[index % len(evaluation_dates)]
        weights = strategy.compute_weights(date, dataset)
        assert weights, "Strategy returned empty weights during microbench"
    elapsed = (time.perf_counter() - start) / iterations

    assert elapsed < 0.005, "Turnover smoothing compute_weights exceeded 5ms budget"
