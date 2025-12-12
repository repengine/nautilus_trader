"""
Integration test demonstrating benchmark strategies with backtesting engine.

This test shows how the three benchmark strategies integrate with the
FactorBacktester engine from Phase 3.1.1.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime

import numpy as np
import polars as pl
import pytest

from playground.backtest.benchmarks import MinimumVarianceStrategy
from playground.backtest.benchmarks import RiskParityStrategy
from playground.backtest.benchmarks import SixtyFortyStrategy
from playground.backtest.engine import BacktestConfig
from playground.risk_model.dataset import CoverageSummary
from playground.risk_model.dataset import SectorDataset


@pytest.fixture
def sample_dataset() -> SectorDataset:
    """Create a realistic sector dataset for integration testing."""
    # Generate 2 years of daily data
    start_date = datetime(2022, 1, 1, tzinfo=UTC)
    num_days = 504  # ~2 years of trading days
    dates = [start_date + np.timedelta64(i, "D") for i in range(num_days)]

    # 9 sector ETFs
    sectors = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLU", "XLV", "XLY"]

    np.random.seed(42)

    records = []
    for date in dates:
        # Generate sector returns with different risk profiles
        for i, sector in enumerate(sectors):
            # Different volatility per sector
            vol = 0.01 + (i * 0.002)  # 1% to 2.6% daily vol
            mean_return = 0.0004  # ~10% annualized
            ret = np.random.normal(mean_return, vol)
            records.append({"timestamp": date, "symbol": sector, "return": ret})

    sector_returns = pl.DataFrame(records)

    # Simple factor returns
    factor_returns = pl.DataFrame({
        "timestamp": dates,
        "factor_duration": np.random.normal(0, 0.01, len(dates)),
        "factor_credit": np.random.normal(0, 0.01, len(dates)),
        "factor_liquidity": np.random.normal(0, 0.01, len(dates)),
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


def test_benchmark_strategies_integration(sample_dataset: SectorDataset) -> None:
    """
    Test that all benchmark strategies work with the backtesting engine.

    This demonstrates the full integration path:
    1. Create backtest config
    2. Instantiate strategies
    3. Compute weights at multiple rebalance dates
    4. Verify all constraints are satisfied
    """
    # Create backtest config (monthly rebalancing, 1 year backtest)
    config = BacktestConfig(
        start_date=datetime(2022, 1, 1, tzinfo=UTC),
        end_date=datetime(2022, 12, 31, tzinfo=UTC),
        initial_capital=1_000_000.0,
        rebalance_frequency="monthly",
        transaction_cost_bps=10.0,
    )

    # Test date (mid-backtest period)
    test_date = datetime(2022, 6, 15, tzinfo=UTC)

    # 60/40 Strategy
    sixty_forty = SixtyFortyStrategy(use_sector_proxies=True)
    weights_60_40 = sixty_forty.compute_weights(test_date, sample_dataset)

    assert len(weights_60_40) > 0, "60/40 should produce weights"
    assert abs(sum(weights_60_40.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"

    # Risk Parity Strategy
    risk_parity = RiskParityStrategy(lookback_days=126)
    weights_rp = risk_parity.compute_weights(test_date, sample_dataset)

    assert len(weights_rp) > 0, "Risk Parity should produce weights"
    assert abs(sum(weights_rp.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"

    # Min Variance Strategy
    min_var = MinimumVarianceStrategy(lookback_days=252)
    weights_mv = min_var.compute_weights(test_date, sample_dataset)

    assert len(weights_mv) > 0, "Min Variance should produce weights"
    assert abs(sum(weights_mv.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"

    # Verify weights are different (each strategy has unique allocation)
    # 60/40 should be tilted toward growth sectors
    # Risk Parity should be more balanced
    # Min Variance should favor low-volatility sectors
    assert weights_60_40 != weights_rp != weights_mv, "Strategies should differ"


def test_benchmark_strategy_performance_characteristics(
    sample_dataset: SectorDataset,
) -> None:
    """
    Test that benchmark strategies exhibit expected performance characteristics.

    Expected properties:
    - Risk Parity: Produces diversified weights
    - Min Variance: Respects constraints
    - 60/40: Static allocation independent of market conditions
    """
    test_date = datetime(2022, 12, 15, tzinfo=UTC)

    # Risk Parity: Should produce valid diversified weights
    risk_parity = RiskParityStrategy(lookback_days=126)
    weights_rp = risk_parity.compute_weights(test_date, sample_dataset)

    # Should be diversified (no single sector > 50%)
    for weight in weights_rp.values():
        assert weight <= 0.50, "Risk Parity should be diversified"

    # Min Variance: Should respect max weight constraint
    min_var = MinimumVarianceStrategy(lookback_days=252, max_weight=0.30)
    weights_mv = min_var.compute_weights(test_date, sample_dataset)

    # Should respect max weight constraint
    for weight in weights_mv.values():
        assert weight <= 0.30 + 1e-6, "Max weight constraint violated"

    # 60/40: Should use sector proxies consistently
    sixty_forty = SixtyFortyStrategy(use_sector_proxies=True)
    weights_1 = sixty_forty.compute_weights(
        datetime(2022, 6, 1, tzinfo=UTC), sample_dataset
    )
    weights_2 = sixty_forty.compute_weights(
        datetime(2022, 12, 1, tzinfo=UTC), sample_dataset
    )

    # Static allocation should be similar across time (sector proxies)
    # Growth sectors (XLK, XLY, XLC) should get equity allocation
    # Defensive sectors (XLU, XLV) should get bond allocation
    growth_sectors = ["XLK", "XLY", "XLC"]
    defensive_sectors = ["XLU", "XLV"]

    growth_weight_1 = sum(weights_1.get(s, 0) for s in growth_sectors)
    defensive_weight_1 = sum(weights_1.get(s, 0) for s in defensive_sectors)

    # Should have some allocation to both growth and defensive
    assert growth_weight_1 > 0, "Should allocate to growth sectors"
    assert defensive_weight_1 > 0, "Should allocate to defensive sectors"

    # Weights should be static (same allocation at different times)
    assert weights_1 == weights_2, "60/40 allocation should be time-invariant"
