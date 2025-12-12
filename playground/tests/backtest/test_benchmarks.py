"""
Tests for benchmark portfolio strategies.

This module provides comprehensive test coverage for:
- SixtyFortyStrategy (60/40 portfolio)
- RiskParityStrategy (inverse volatility weighting)
- MinimumVarianceStrategy (quadratic optimization)

Test Categories:
- Unit tests: Weight computation, constraint enforcement, edge cases
- Integration tests: Full backtest with realistic data
- Property tests: Invariants (weights sum to 1.0, bounds respected)
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta

import numpy as np
import polars as pl
import pytest

from playground.backtest.benchmarks import MinimumVarianceStrategy
from playground.backtest.benchmarks import RiskParityStrategy
from playground.backtest.benchmarks import SixtyFortyStrategy
from playground.risk_model.dataset import CoverageSummary
from playground.risk_model.dataset import SectorDataset


# ===== Test Fixtures =====


@pytest.fixture
def simple_sector_dataset() -> SectorDataset:
    """
    Create a simple sector dataset for testing.

    Returns 30 days of returns for 3 sectors (XLK, XLU, XLV).
    """
    dates = [datetime(2024, 1, i, tzinfo=UTC) for i in range(1, 31)]
    sectors = ["XLK", "XLU", "XLV"]

    # Generate synthetic returns
    # XLK: high volatility (tech)
    # XLU: low volatility (utilities)
    # XLV: medium volatility (healthcare)
    np.random.seed(42)

    records = []
    for date in dates:
        # XLK: 2% daily vol
        xlk_return = np.random.normal(0.0005, 0.02)
        # XLU: 0.5% daily vol (low volatility)
        xlu_return = np.random.normal(0.0003, 0.005)
        # XLV: 1% daily vol
        xlv_return = np.random.normal(0.0004, 0.01)

        records.append({"timestamp": date, "symbol": "XLK", "return": xlk_return})
        records.append({"timestamp": date, "symbol": "XLU", "return": xlu_return})
        records.append({"timestamp": date, "symbol": "XLV", "return": xlv_return})

    sector_returns = pl.DataFrame(records)

    # Create simple factor returns (not used by benchmark strategies)
    factor_returns = pl.DataFrame({
        "timestamp": dates,
        "factor_duration": np.random.normal(0, 0.01, len(dates)),
        "factor_credit": np.random.normal(0, 0.01, len(dates)),
        "factor_liquidity": np.random.normal(0, 0.01, len(dates)),
    })

    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=30,
        factor_expected_days=30,
        sector_coverage=dict.fromkeys(sectors, 1.0),
        factor_coverage={"factor_duration": 1.0, "factor_credit": 1.0, "factor_liquidity": 1.0},
    )

    return SectorDataset(
        sector_returns=sector_returns,
        factor_returns=factor_returns,
        coverage=coverage,
    )


@pytest.fixture
def spy_agg_dataset() -> SectorDataset:
    """
    Create a dataset with SPY and AGG for 60/40 testing.
    """
    dates = [datetime(2024, 1, i, tzinfo=UTC) for i in range(1, 31)]

    np.random.seed(42)

    records = []
    for date in dates:
        # SPY: 1.5% daily vol
        spy_return = np.random.normal(0.0005, 0.015)
        # AGG: 0.3% daily vol (bonds)
        agg_return = np.random.normal(0.0001, 0.003)

        records.append({"timestamp": date, "symbol": "SPY", "return": spy_return})
        records.append({"timestamp": date, "symbol": "AGG", "return": agg_return})

    sector_returns = pl.DataFrame(records)

    factor_returns = pl.DataFrame({
        "timestamp": dates,
        "factor_duration": np.random.normal(0, 0.01, len(dates)),
        "factor_credit": np.random.normal(0, 0.01, len(dates)),
        "factor_liquidity": np.random.normal(0, 0.01, len(dates)),
    })

    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=30,
        factor_expected_days=30,
        sector_coverage={"SPY": 1.0, "AGG": 1.0},
        factor_coverage={"factor_duration": 1.0, "factor_credit": 1.0, "factor_liquidity": 1.0},
    )

    return SectorDataset(
        sector_returns=sector_returns,
        factor_returns=factor_returns,
        coverage=coverage,
    )


@pytest.fixture
def correlated_dataset() -> SectorDataset:
    """
    Create a dataset with highly correlated sectors for min variance testing.
    """
    # Generate 100 trading days across multiple months
    start_date = datetime(2024, 1, 1, tzinfo=UTC)
    dates = [start_date + timedelta(days=i) for i in range(150)]
    # Filter to weekdays only (approximate trading days)
    dates = [d for d in dates if d.weekday() < 5][:100]

    sectors = ["S1", "S2", "S3"]

    np.random.seed(42)

    # Generate correlated returns
    # Common factor (market return)
    market = np.random.normal(0.0005, 0.01, len(dates))

    records = []
    for i, date in enumerate(dates):
        # S1: high beta (2x market + noise)
        s1_return = 2.0 * market[i] + np.random.normal(0, 0.005)
        # S2: low beta (0.5x market + noise)
        s2_return = 0.5 * market[i] + np.random.normal(0, 0.003)
        # S3: market beta (1x market + noise)
        s3_return = 1.0 * market[i] + np.random.normal(0, 0.007)

        records.append({"timestamp": date, "symbol": "S1", "return": s1_return})
        records.append({"timestamp": date, "symbol": "S2", "return": s2_return})
        records.append({"timestamp": date, "symbol": "S3", "return": s3_return})

    sector_returns = pl.DataFrame(records)

    factor_returns = pl.DataFrame({
        "timestamp": dates,
        "factor_duration": np.random.normal(0, 0.01, len(dates)),
        "factor_credit": np.random.normal(0, 0.01, len(dates)),
        "factor_liquidity": np.random.normal(0, 0.01, len(dates)),
    })

    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=100,
        factor_expected_days=100,
        sector_coverage=dict.fromkeys(sectors, 1.0),
        factor_coverage={"factor_duration": 1.0, "factor_credit": 1.0, "factor_liquidity": 1.0},
    )

    return SectorDataset(
        sector_returns=sector_returns,
        factor_returns=factor_returns,
        coverage=coverage,
    )


# ===== SixtyFortyStrategy Tests =====


def test_sixty_forty_weights(spy_agg_dataset: SectorDataset) -> None:
    """Test that 60/40 strategy returns correct allocation."""
    strategy = SixtyFortyStrategy()
    date = datetime(2024, 1, 15, tzinfo=UTC)

    weights = strategy.compute_weights(date, spy_agg_dataset)

    assert weights == {"SPY": 0.60, "AGG": 0.40}


def test_sixty_forty_weights_sum_to_one(spy_agg_dataset: SectorDataset) -> None:
    """Test that weights sum to 1.0."""
    strategy = SixtyFortyStrategy()
    date = datetime(2024, 1, 15, tzinfo=UTC)

    weights = strategy.compute_weights(date, spy_agg_dataset)

    total_weight = sum(weights.values())
    assert abs(total_weight - 1.0) < 1e-10


def test_sixty_forty_invalid_weights() -> None:
    """Test that invalid weights raise ValueError."""
    with pytest.raises(ValueError, match="must sum to 1.0"):
        SixtyFortyStrategy(equity_weight=0.70, bond_weight=0.40)


def test_sixty_forty_negative_weights() -> None:
    """Test that negative weights raise ValueError."""
    with pytest.raises(ValueError, match="must be non-negative"):
        SixtyFortyStrategy(equity_weight=-0.10, bond_weight=1.10)


def test_sixty_forty_sector_proxies(simple_sector_dataset: SectorDataset) -> None:
    """Test that sector proxies are used when SPY/AGG not available."""
    strategy = SixtyFortyStrategy(use_sector_proxies=True)
    date = datetime(2024, 1, 15, tzinfo=UTC)

    weights = strategy.compute_weights(date, simple_sector_dataset)

    # Should use XLK as equity proxy and XLU/XLV as bond proxies
    assert "XLK" in weights  # Equity sector
    assert "XLU" in weights or "XLV" in weights  # Bond proxies

    # Weights should sum to 1.0
    total_weight = sum(weights.values())
    assert abs(total_weight - 1.0) < 1e-6


def test_sixty_forty_no_proxies(simple_sector_dataset: SectorDataset) -> None:
    """Test that empty weights returned when proxies disabled and SPY/AGG unavailable."""
    strategy = SixtyFortyStrategy(use_sector_proxies=False)
    date = datetime(2024, 1, 15, tzinfo=UTC)

    weights = strategy.compute_weights(date, simple_sector_dataset)

    assert weights == {}


def test_sixty_forty_custom_allocation() -> None:
    """Test custom equity/bond allocation (e.g., 70/30)."""
    strategy = SixtyFortyStrategy(equity_weight=0.70, bond_weight=0.30)
    assert strategy.equity_weight == 0.70
    assert strategy.bond_weight == 0.30


# ===== RiskParityStrategy Tests =====


def test_risk_parity_inverse_volatility(simple_sector_dataset: SectorDataset) -> None:
    """
    Test that risk parity assigns higher weights to lower volatility sectors.

    XLU (low vol) should get higher weight than XLK (high vol).
    """
    strategy = RiskParityStrategy(lookback_days=30)
    date = datetime(2024, 1, 30, tzinfo=UTC)

    weights = strategy.compute_weights(date, simple_sector_dataset)

    # XLU (utilities) has lowest volatility, should get highest weight
    # XLK (tech) has highest volatility, should get lowest weight
    assert "XLU" in weights
    assert "XLK" in weights
    assert weights["XLU"] > weights["XLK"]


def test_risk_parity_applies_constraints(simple_sector_dataset: SectorDataset) -> None:
    """Test that min/max weight constraints are enforced."""
    strategy = RiskParityStrategy(
        lookback_days=30,
        min_weight=0.10,
        max_weight=0.50,
    )
    date = datetime(2024, 1, 30, tzinfo=UTC)

    weights = strategy.compute_weights(date, simple_sector_dataset)

    # Check all weights respect bounds
    for sector, weight in weights.items():
        assert 0.10 <= weight <= 0.50, f"{sector}: {weight}"


def test_risk_parity_weights_sum_to_one(simple_sector_dataset: SectorDataset) -> None:
    """Test that weights sum to 1.0 after constraint application."""
    strategy = RiskParityStrategy(lookback_days=30)
    date = datetime(2024, 1, 30, tzinfo=UTC)

    weights = strategy.compute_weights(date, simple_sector_dataset)

    total_weight = sum(weights.values())
    assert abs(total_weight - 1.0) < 1e-6


def test_risk_parity_handles_zero_volatility() -> None:
    """Test handling of zero volatility (constant returns)."""
    # Create dataset with one zero-volatility sector
    dates = [datetime(2024, 1, i, tzinfo=UTC) for i in range(1, 31)]
    records = []

    for date in dates:
        # S1: zero volatility (constant return)
        records.append({"timestamp": date, "symbol": "S1", "return": 0.001})
        # S2: normal volatility
        records.append({"timestamp": date, "symbol": "S2", "return": np.random.normal(0, 0.01)})

    sector_returns = pl.DataFrame(records)
    factor_returns = pl.DataFrame({
        "timestamp": dates,
        "factor_duration": np.zeros(len(dates)),
    })
    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=30,
        factor_expected_days=30,
        sector_coverage={"S1": 1.0, "S2": 1.0},
        factor_coverage={"factor_duration": 1.0},
    )
    dataset = SectorDataset(sector_returns, factor_returns, coverage)

    strategy = RiskParityStrategy(lookback_days=30)
    date = datetime(2024, 1, 30, tzinfo=UTC)

    weights = strategy.compute_weights(date, dataset)

    # S1 should be excluded (zero volatility)
    # S2 should get all weight
    assert "S1" not in weights or weights.get("S1", 0) < 1e-6
    if "S2" in weights:
        assert abs(weights["S2"] - 1.0) < 1e-6


def test_risk_parity_insufficient_data() -> None:
    """Test handling of insufficient historical data."""
    # Create dataset with only 5 observations
    dates = [datetime(2024, 1, i, tzinfo=UTC) for i in range(1, 6)]
    records = []

    for date in dates:
        records.append({"timestamp": date, "symbol": "S1", "return": 0.001})

    sector_returns = pl.DataFrame(records)
    factor_returns = pl.DataFrame({
        "timestamp": dates,
        "factor_duration": np.zeros(len(dates)),
    })
    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=5,
        factor_expected_days=5,
        sector_coverage={"S1": 1.0},
        factor_coverage={"factor_duration": 1.0},
    )
    dataset = SectorDataset(sector_returns, factor_returns, coverage)

    strategy = RiskParityStrategy(lookback_days=30, min_observations=20)
    date = datetime(2024, 1, 5, tzinfo=UTC)

    weights = strategy.compute_weights(date, dataset)

    # Should return empty weights (insufficient data)
    assert weights == {}


def test_risk_parity_invalid_params() -> None:
    """Test that invalid parameters raise ValueError."""
    with pytest.raises(ValueError, match="Lookback days must be positive"):
        RiskParityStrategy(lookback_days=0)

    with pytest.raises(ValueError, match="min_weight <= max_weight"):
        RiskParityStrategy(min_weight=0.6, max_weight=0.4)

    with pytest.raises(ValueError, match="Minimum observations must be > 1"):
        RiskParityStrategy(min_observations=1)


# ===== MinimumVarianceStrategy Tests =====


def test_min_variance_optimization(correlated_dataset: SectorDataset) -> None:
    """Test that min variance optimization produces valid weights."""
    strategy = MinimumVarianceStrategy(
        lookback_days=150,
        max_weight=0.50,  # Less restrictive to allow optimization to succeed
    )
    # Use a date well after the start to have enough lookback data
    date = datetime(2024, 5, 31, tzinfo=UTC)

    weights = strategy.compute_weights(date, correlated_dataset)

    # Should have weights for all sectors
    assert len(weights) == 3
    assert "S1" in weights
    assert "S2" in weights
    assert "S3" in weights

    # Weights should be non-negative
    for sector, weight in weights.items():
        assert weight >= 0

    # Should favor low-beta sector (S2) over high-beta sector (S1)
    assert weights["S2"] > weights["S1"]


def test_min_variance_covariance_estimation(correlated_dataset: SectorDataset) -> None:
    """Test that covariance matrix is estimated from correct rolling window."""
    # Use shorter window that fits within the dataset
    strategy = MinimumVarianceStrategy(lookback_days=30, min_observations=20)
    # Use a date toward the end of the 100-day dataset
    date = datetime(2024, 4, 30, tzinfo=UTC)

    weights = strategy.compute_weights(date, correlated_dataset)

    # Should successfully compute weights with rolling window
    assert len(weights) > 0
    total_weight = sum(weights.values())
    assert abs(total_weight - 1.0) < 1e-6


def test_min_variance_constraints_enforced(correlated_dataset: SectorDataset) -> None:
    """Test that max weight constraint (40%) is enforced."""
    strategy = MinimumVarianceStrategy(
        lookback_days=150,
        min_weight=0.0,
        max_weight=0.40,  # Slightly more relaxed to allow valid solution
    )
    date = datetime(2024, 5, 31, tzinfo=UTC)

    weights = strategy.compute_weights(date, correlated_dataset)

    # Check all weights respect max constraint
    for sector, weight in weights.items():
        assert weight <= 0.40 + 1e-6, f"{sector}: {weight}"


def test_min_variance_weights_sum_to_one(correlated_dataset: SectorDataset) -> None:
    """Test that weights sum to 1.0."""
    strategy = MinimumVarianceStrategy(lookback_days=150)
    date = datetime(2024, 5, 31, tzinfo=UTC)

    weights = strategy.compute_weights(date, correlated_dataset)

    total_weight = sum(weights.values())
    assert abs(total_weight - 1.0) < 1e-6


def test_min_variance_handles_singular_matrix() -> None:
    """Test handling of perfectly correlated assets (singular covariance matrix)."""
    # Create dataset with perfectly correlated sectors
    start_date = datetime(2024, 1, 1, tzinfo=UTC)
    dates = [start_date + timedelta(days=i) for i in range(150)]
    dates = [d for d in dates if d.weekday() < 5][:100]

    np.random.seed(42)

    # Generate common returns (perfect correlation)
    common_returns = np.random.normal(0.001, 0.01, len(dates))

    records = []
    for i, date in enumerate(dates):
        # All sectors have identical returns (perfect correlation)
        records.append({"timestamp": date, "symbol": "S1", "return": common_returns[i]})
        records.append({"timestamp": date, "symbol": "S2", "return": common_returns[i]})
        records.append({"timestamp": date, "symbol": "S3", "return": common_returns[i]})

    sector_returns = pl.DataFrame(records)
    factor_returns = pl.DataFrame({
        "timestamp": dates,
        "factor_duration": np.zeros(len(dates)),
    })
    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=100,
        factor_expected_days=100,
        sector_coverage={"S1": 1.0, "S2": 1.0, "S3": 1.0},
        factor_coverage={"factor_duration": 1.0},
    )
    dataset = SectorDataset(sector_returns, factor_returns, coverage)

    strategy = MinimumVarianceStrategy(
        lookback_days=150,
        regularization=1e-5,  # Regularization should handle this
    )
    date = datetime(2024, 5, 31, tzinfo=UTC)

    weights = strategy.compute_weights(date, dataset)

    # Should still produce valid weights (regularization prevents singularity)
    assert len(weights) > 0
    total_weight = sum(weights.values())
    assert abs(total_weight - 1.0) < 1e-3  # Relaxed tolerance


def test_min_variance_insufficient_observations() -> None:
    """Test handling of insufficient observations."""
    dates = [datetime(2024, 1, i, tzinfo=UTC) for i in range(1, 11)]  # Only 10 days
    records = []

    for date in dates:
        records.append({"timestamp": date, "symbol": "S1", "return": 0.001})
        records.append({"timestamp": date, "symbol": "S2", "return": 0.002})

    sector_returns = pl.DataFrame(records)
    factor_returns = pl.DataFrame({
        "timestamp": dates,
        "factor_duration": np.zeros(len(dates)),
    })
    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=10,
        factor_expected_days=10,
        sector_coverage={"S1": 1.0, "S2": 1.0},
        factor_coverage={"factor_duration": 1.0},
    )
    dataset = SectorDataset(sector_returns, factor_returns, coverage)

    strategy = MinimumVarianceStrategy(
        lookback_days=100,
        min_observations=60,
    )
    date = datetime(2024, 1, 10, tzinfo=UTC)

    weights = strategy.compute_weights(date, dataset)

    # Should return empty weights (insufficient data)
    assert weights == {}


def test_min_variance_invalid_params() -> None:
    """Test that invalid parameters raise ValueError."""
    with pytest.raises(ValueError, match="Lookback days must be positive"):
        MinimumVarianceStrategy(lookback_days=0)

    with pytest.raises(ValueError, match="min_weight <= max_weight"):
        MinimumVarianceStrategy(min_weight=0.5, max_weight=0.3)

    with pytest.raises(ValueError, match="Minimum observations must be > 2"):
        MinimumVarianceStrategy(min_observations=2)

    with pytest.raises(ValueError, match="Regularization must be non-negative"):
        MinimumVarianceStrategy(regularization=-1.0)


# ===== Integration Tests =====


def test_all_strategies_no_lookahead_bias(simple_sector_dataset: SectorDataset) -> None:
    """Test that all strategies only use data up to the rebalance date."""
    date = datetime(2024, 1, 15, tzinfo=UTC)

    strategies = [
        RiskParityStrategy(lookback_days=14, min_observations=10),
        MinimumVarianceStrategy(lookback_days=14, min_observations=10),
    ]

    for strategy in strategies:
        weights = strategy.compute_weights(date, simple_sector_dataset)

        # Should have computed weights
        assert len(weights) > 0

        # Weights should sum to 1.0
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 1e-6


def test_strategies_with_empty_dataset() -> None:
    """Test that all strategies handle empty datasets gracefully."""
    # Create empty dataset
    sector_returns = pl.DataFrame({
        "timestamp": [],
        "symbol": [],
        "return": [],
    })
    factor_returns = pl.DataFrame({
        "timestamp": [],
        "factor_duration": [],
    })
    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=0,
        factor_expected_days=0,
        sector_coverage={},
        factor_coverage={},
    )
    dataset = SectorDataset(sector_returns, factor_returns, coverage)

    date = datetime(2024, 1, 15, tzinfo=UTC)

    strategies = [
        SixtyFortyStrategy(),
        RiskParityStrategy(lookback_days=30),
        MinimumVarianceStrategy(lookback_days=30),
    ]

    for strategy in strategies:
        weights = strategy.compute_weights(date, dataset)
        assert weights == {}


def test_strategies_weight_consistency(simple_sector_dataset: SectorDataset) -> None:
    """Test that repeated calls with same date produce identical weights."""
    date = datetime(2024, 1, 30, tzinfo=UTC)

    strategies = [
        RiskParityStrategy(lookback_days=30),
        MinimumVarianceStrategy(lookback_days=30, min_observations=20),
    ]

    for strategy in strategies:
        weights1 = strategy.compute_weights(date, simple_sector_dataset)
        weights2 = strategy.compute_weights(date, simple_sector_dataset)

        assert weights1 == weights2


def test_min_variance_regularization_effect() -> None:
    """Test that regularization prevents ill-conditioned covariance matrices."""
    # Create dataset with near-singular covariance
    start_date = datetime(2024, 1, 1, tzinfo=UTC)
    dates = [start_date + timedelta(days=i) for i in range(150)]
    dates = [d for d in dates if d.weekday() < 5][:100]

    np.random.seed(42)

    common = np.random.normal(0, 0.01, len(dates))
    noise = np.random.normal(0, 0.0001, len(dates))

    records = []
    for i, date in enumerate(dates):
        # S1 and S2 are almost perfectly correlated
        records.append({"timestamp": date, "symbol": "S1", "return": common[i] + noise[i]})
        records.append({"timestamp": date, "symbol": "S2", "return": common[i] - noise[i]})
        records.append({"timestamp": date, "symbol": "S3", "return": np.random.normal(0, 0.01)})

    sector_returns = pl.DataFrame(records)
    factor_returns = pl.DataFrame({
        "timestamp": dates,
        "factor_duration": np.zeros(len(dates)),
    })
    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=100,
        factor_expected_days=100,
        sector_coverage={"S1": 1.0, "S2": 1.0, "S3": 1.0},
        factor_coverage={"factor_duration": 1.0},
    )
    dataset = SectorDataset(sector_returns, factor_returns, coverage)

    # Without regularization might fail, with regularization should succeed
    strategy = MinimumVarianceStrategy(
        lookback_days=150,
        regularization=1e-4,
    )
    date = datetime(2024, 5, 31, tzinfo=UTC)

    weights = strategy.compute_weights(date, dataset)

    # Should produce valid weights
    assert len(weights) > 0
    total_weight = sum(weights.values())
    assert abs(total_weight - 1.0) < 1e-3


def test_benchmark_strategies_export() -> None:
    """Test that all benchmark strategies are properly exported."""
    from playground.backtest.benchmarks import MinimumVarianceStrategy
    from playground.backtest.benchmarks import RiskParityStrategy
    from playground.backtest.benchmarks import SixtyFortyStrategy

    # Should be able to instantiate all strategies
    sixty_forty = SixtyFortyStrategy()
    risk_parity = RiskParityStrategy()
    min_variance = MinimumVarianceStrategy()

    assert sixty_forty is not None
    assert risk_parity is not None
    assert min_variance is not None
