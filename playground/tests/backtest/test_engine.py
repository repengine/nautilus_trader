"""
Comprehensive test suite for backtesting engine.

Tests cover:
- Configuration validation
- Rebalance date generation
- Transaction cost calculation
- Equal-weight strategy execution
- Performance metrics computation
- Look-ahead bias prevention
- Reproducibility with fixed seed
"""

from __future__ import annotations

# Direct import from playground module files to avoid circular dependencies in ml.stores
import importlib.util
import sys
from datetime import UTC
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest


playground_dir = Path(__file__).parent.parent.parent

# Load modules directly without going through __init__ files
def load_module_directly(module_name: str, file_path: Path):  # type: ignore[no-untyped-def]
    """Load a module directly from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    raise ImportError(f"Could not load {module_name} from {file_path}")

# Load dataset module
dataset_mod = load_module_directly(
    "playground.risk_model.dataset",
    playground_dir / "risk_model" / "dataset.py"
)

# Load engine module
engine_mod = load_module_directly(
    "playground.backtest.engine",
    playground_dir / "backtest" / "engine.py"
)

# Load strategies module
strategies_mod = load_module_directly(
    "playground.backtest.strategies",
    playground_dir / "backtest" / "strategies.py"
)

BacktestConfig = engine_mod.BacktestConfig
FactorBacktester = engine_mod.FactorBacktester
EqualWeightStrategy = strategies_mod.EqualWeightStrategy
CoverageSummary = dataset_mod.CoverageSummary
SectorDataset = dataset_mod.SectorDataset


# ===== Fixtures =====


@pytest.fixture
def simple_config() -> BacktestConfig:
    """Create a simple backtest configuration."""
    return BacktestConfig(
        start_date=datetime(2020, 1, 1, tzinfo=UTC),
        end_date=datetime(2020, 12, 31, tzinfo=UTC),
        initial_capital=1_000_000.0,
        rebalance_frequency="monthly",
        transaction_cost_bps=10.0,
        random_seed=42,
    )


@pytest.fixture
def sample_sector_returns() -> pl.DataFrame:
    """Create sample sector return data."""
    dates = [datetime(2020, 1, i, tzinfo=UTC) for i in range(1, 32)]
    sectors = ["XLK", "XLF", "XLU"]

    data = []
    for date in dates:
        for sector in sectors:
            # Generate deterministic returns
            np.random.seed(hash(str(date) + sector) % 2**32)
            ret = np.random.normal(0.0005, 0.015)
            data.append({
                "timestamp": date,
                "symbol": sector,
                "return": ret,
            })

    return pl.DataFrame(data)


@pytest.fixture
def sample_factor_returns() -> pl.DataFrame:
    """Create sample factor return data."""
    dates = [datetime(2020, 1, i, tzinfo=UTC) for i in range(1, 32)]

    data = []
    for date in dates:
        np.random.seed(hash(str(date)) % 2**32)
        data.append({
            "timestamp": date,
            "factor_duration": np.random.normal(0.0, 0.01),
            "factor_credit": np.random.normal(0.0, 0.02),
            "factor_liquidity": np.random.normal(0.0, 0.015),
        })

    return pl.DataFrame(data)


@pytest.fixture
def sample_dataset(
    sample_sector_returns: pl.DataFrame,
    sample_factor_returns: pl.DataFrame,
) -> SectorDataset:
    """Create sample dataset for testing."""
    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=21,
        factor_expected_days=21,
        sector_coverage={"XLK": 1.0, "XLF": 1.0, "XLU": 1.0},
        factor_coverage={
            "factor_duration": 1.0,
            "factor_credit": 1.0,
            "factor_liquidity": 1.0,
        },
    )

    return SectorDataset(
        sector_returns=sample_sector_returns,
        factor_returns=sample_factor_returns,
        coverage=coverage,
    )


# ===== Test: Configuration Validation =====


def test_backtest_engine_initialization(simple_config: BacktestConfig) -> None:
    """Test that backtester initializes correctly with valid config."""
    backtester = FactorBacktester(simple_config)

    assert backtester.config == simple_config
    assert backtester.logger is not None


def test_config_rejects_invalid_dates() -> None:
    """Test that config rejects end_date before start_date."""
    with pytest.raises(ValueError, match="End date must be after start date"):
        BacktestConfig(
            start_date=datetime(2020, 12, 31, tzinfo=UTC),
            end_date=datetime(2020, 1, 1, tzinfo=UTC),
        )


def test_config_rejects_negative_capital() -> None:
    """Test that config rejects negative initial capital."""
    with pytest.raises(ValueError, match="Initial capital must be positive"):
        BacktestConfig(
            start_date=datetime(2020, 1, 1, tzinfo=UTC),
            end_date=datetime(2020, 12, 31, tzinfo=UTC),
            initial_capital=-1000.0,
        )


def test_config_rejects_negative_transaction_costs() -> None:
    """Test that config rejects negative transaction costs."""
    with pytest.raises(ValueError, match="Transaction cost must be non-negative"):
        BacktestConfig(
            start_date=datetime(2020, 1, 1, tzinfo=UTC),
            end_date=datetime(2020, 12, 31, tzinfo=UTC),
            transaction_cost_bps=-10.0,
        )


def test_config_rejects_invalid_frequency() -> None:
    """Test that config rejects invalid rebalance frequency."""
    with pytest.raises(ValueError, match="Invalid rebalance frequency"):
        BacktestConfig(
            start_date=datetime(2020, 1, 1, tzinfo=UTC),
            end_date=datetime(2020, 12, 31, tzinfo=UTC),
            rebalance_frequency="yearly",
        )


# ===== Test: Rebalance Date Generation =====


def test_rebalance_dates_monthly(simple_config: BacktestConfig) -> None:
    """Test monthly rebalance schedule generates correct dates."""
    backtester = FactorBacktester(simple_config)

    # Create trading dates for Jan-Mar 2020
    trading_dates = []
    for month in range(1, 4):
        for day in range(1, 29):  # Approximate month length
            try:
                date = datetime(2020, month, day, tzinfo=UTC)
                trading_dates.append(date)
            except ValueError:
                continue

    rebalance_dates = backtester._get_rebalance_dates(
        start=datetime(2020, 1, 1, tzinfo=UTC),
        end=datetime(2020, 3, 31, tzinfo=UTC),
        frequency="monthly",
        trading_dates=trading_dates,
    )

    # Should get last trading day of each month
    assert len(rebalance_dates) == 3  # Jan, Feb, Mar

    # Check that rebalance dates are end-of-month
    assert rebalance_dates[0].month == 1
    assert rebalance_dates[1].month == 2
    assert rebalance_dates[2].month == 3


def test_rebalance_dates_quarterly(simple_config: BacktestConfig) -> None:
    """Quarterly cadence should rebalance on the last trading day of each quarter."""
    backtester = FactorBacktester(simple_config)
    trading_dates = [datetime(2020, month, 28, tzinfo=UTC) for month in range(1, 13)]
    rebalance_dates = backtester._get_rebalance_dates(
        start=datetime(2020, 1, 1, tzinfo=UTC),
        end=datetime(2020, 12, 31, tzinfo=UTC),
        frequency="quarterly",
        trading_dates=trading_dates,
    )
    assert len(rebalance_dates) == 4
    assert [date.month for date in rebalance_dates] == [3, 6, 9, 12]


def test_rebalance_dates_semi_annual(simple_config: BacktestConfig) -> None:
    """Semi-annual cadence should rebalance twice per year."""
    backtester = FactorBacktester(simple_config)
    trading_dates = [datetime(2020, month, 28, tzinfo=UTC) for month in range(1, 13)]
    rebalance_dates = backtester._get_rebalance_dates(
        start=datetime(2020, 1, 1, tzinfo=UTC),
        end=datetime(2020, 12, 31, tzinfo=UTC),
        frequency="semi_annual",
        trading_dates=trading_dates,
    )
    assert len(rebalance_dates) == 2
    assert [date.month for date in rebalance_dates] == [6, 12]


def test_rebalance_dates_daily(simple_config: BacktestConfig) -> None:
    """Test daily rebalance schedule returns all trading dates."""
    backtester = FactorBacktester(simple_config)

    trading_dates = [
        datetime(2020, 1, i, tzinfo=UTC) for i in range(1, 6)
    ]

    rebalance_dates = backtester._get_rebalance_dates(
        start=datetime(2020, 1, 1, tzinfo=UTC),
        end=datetime(2020, 1, 5, tzinfo=UTC),
        frequency="daily",
        trading_dates=trading_dates,
    )

    assert len(rebalance_dates) == len(trading_dates)
    assert rebalance_dates == trading_dates


# ===== Test: Transaction Cost Calculation =====


def test_transaction_cost_calculation() -> None:
    """Test transaction costs match hand-calculated example."""
    config = BacktestConfig(
        start_date=datetime(2020, 1, 1, tzinfo=UTC),
        end_date=datetime(2020, 12, 31, tzinfo=UTC),
        transaction_cost_bps=10.0,
        slippage_bps=0.0,
    )
    backtester = FactorBacktester(config)

    # Example from specification
    old_weights = {"XLK": 0.20, "XLU": 0.10, "XLF": 0.0}
    new_weights = {"XLK": 0.25, "XLU": 0.08, "XLF": 0.05}
    portfolio_value = 1_000_000.0

    cost = backtester._apply_transaction_costs(
        old_weights,
        new_weights,
        portfolio_value,
    )

    # Turnover: |0.25-0.20| + |0.08-0.10| + |0.05-0.0| = 0.05 + 0.02 + 0.05 = 0.12
    # Cost at 10 bps: 0.12 × 1,000,000 × (10/10000) = $120
    expected_cost = 120.0
    assert abs(cost - expected_cost) < 0.01


def test_transaction_cost_zero_when_no_change() -> None:
    """Test transaction cost is zero when weights don't change."""
    config = BacktestConfig(
        start_date=datetime(2020, 1, 1, tzinfo=UTC),
        end_date=datetime(2020, 12, 31, tzinfo=UTC),
        transaction_cost_bps=10.0,
    )
    backtester = FactorBacktester(config)

    weights = {"XLK": 0.5, "XLF": 0.5}
    cost = backtester._apply_transaction_costs(
        weights,
        weights,
        1_000_000.0,
    )

    assert cost == 0.0


def test_transaction_cost_with_slippage() -> None:
    """Test transaction costs include slippage."""
    config = BacktestConfig(
        start_date=datetime(2020, 1, 1, tzinfo=UTC),
        end_date=datetime(2020, 12, 31, tzinfo=UTC),
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
    )
    backtester = FactorBacktester(config)

    old_weights = {"XLK": 0.5}
    new_weights = {"XLK": 0.6}
    portfolio_value = 1_000_000.0

    cost = backtester._apply_transaction_costs(
        old_weights,
        new_weights,
        portfolio_value,
    )

    # Turnover: |0.6-0.5| = 0.1
    # Cost at 15 bps (10 + 5): 0.1 × 1,000,000 × 0.0015 = $150
    expected_cost = 150.0
    assert abs(cost - expected_cost) < 0.01


# ===== Test: Equal-Weight Backtest =====


def test_equal_weight_backtest_deterministic(
    simple_config: BacktestConfig,
    sample_dataset: SectorDataset,
) -> None:
    """Test equal-weight backtest produces deterministic results with fixed seed."""
    backtester1 = FactorBacktester(simple_config)
    result1 = backtester1.run_backtest(
        dataset=sample_dataset,
        strategy="equal_weight",
    )

    # Run again with same seed
    backtester2 = FactorBacktester(simple_config)
    result2 = backtester2.run_backtest(
        dataset=sample_dataset,
        strategy="equal_weight",
    )

    # Results should be identical
    assert result1.total_return == result2.total_return
    assert result1.sharpe_ratio == result2.sharpe_ratio
    assert result1.max_drawdown == result2.max_drawdown
    assert len(result1.returns) == len(result2.returns)


def test_equal_weight_strategy_assigns_equal_weights() -> None:
    """Test equal-weight strategy assigns 1/N to each sector."""
    strategy = EqualWeightStrategy()

    # Create simple dataset
    sector_returns = pl.DataFrame({
        "timestamp": [datetime(2020, 1, 1, tzinfo=UTC)] * 3,
        "symbol": ["XLK", "XLF", "XLU"],
        "return": [0.01, 0.02, -0.01],
    })

    dataset = SectorDataset(
        sector_returns=sector_returns,
        factor_returns=pl.DataFrame({
            "timestamp": [datetime(2020, 1, 1, tzinfo=UTC)],
            "factor_duration": [0.0],
        }),
        coverage=CoverageSummary(
            calendar_name="XNYS",
            sector_expected_days=1,
            factor_expected_days=1,
            sector_coverage={"XLK": 1.0, "XLF": 1.0, "XLU": 1.0},
            factor_coverage={"factor_duration": 1.0},
        ),
    )

    weights = strategy.compute_weights(
        date=datetime(2020, 1, 1, tzinfo=UTC),
        dataset=dataset,
    )

    # Should have 1/3 for each sector
    assert len(weights) == 3
    expected_weight = 1.0 / 3.0
    for sector in ["XLK", "XLF", "XLU"]:
        assert abs(weights[sector] - expected_weight) < 1e-10


# ===== Test: Look-Ahead Bias Prevention =====


def test_backtest_no_lookahead_bias() -> None:
    """Test that future data is never used in past decisions."""
    # Create dataset where returns change dramatically after a date
    dates_before = [datetime(2020, 1, i, tzinfo=UTC) for i in range(1, 16)]
    dates_after = [datetime(2020, 1, i, tzinfo=UTC) for i in range(16, 32)]

    data = []
    # Before: positive returns
    for date in dates_before:
        for sector in ["XLK", "XLF"]:
            data.append({
                "timestamp": date,
                "symbol": sector,
                "return": 0.01,  # Positive
            })

    # After: negative returns
    for date in dates_after:
        for sector in ["XLK", "XLF"]:
            data.append({
                "timestamp": date,
                "symbol": sector,
                "return": -0.01,  # Negative
            })

    sector_returns = pl.DataFrame(data)

    dataset = SectorDataset(
        sector_returns=sector_returns,
        factor_returns=pl.DataFrame({
            "timestamp": dates_before + dates_after,
            "factor_duration": [0.0] * len(dates_before + dates_after),
        }),
        coverage=CoverageSummary(
            calendar_name="XNYS",
            sector_expected_days=31,
            factor_expected_days=31,
            sector_coverage={"XLK": 1.0, "XLF": 1.0},
            factor_coverage={"factor_duration": 1.0},
        ),
    )

    # Run backtest only on first half
    config = BacktestConfig(
        start_date=datetime(2020, 1, 1, tzinfo=UTC),
        end_date=datetime(2020, 1, 15, tzinfo=UTC),
        rebalance_frequency="monthly",
    )

    backtester = FactorBacktester(config)
    result = backtester.run_backtest(dataset, strategy="equal_weight")

    # Should have positive returns (only saw positive period)
    assert result.total_return > 0


# ===== Test: Performance Metrics =====


def test_performance_metrics_calculation() -> None:
    """Test performance metrics match hand-calculated values."""
    config = BacktestConfig(
        start_date=datetime(2020, 1, 1, tzinfo=UTC),
        end_date=datetime(2020, 1, 10, tzinfo=UTC),
    )
    backtester = FactorBacktester(config)

    # Simple returns: 1% daily for 10 days
    returns = [0.01] * 10
    transaction_costs = [0.0] * 10
    rebalances = 1

    metrics = backtester._compute_performance_metrics(
        returns,
        transaction_costs,
        rebalances,
    )

    # Total return: (1.01)^10 - 1 ≈ 0.1046
    assert abs(metrics["total_return"] - 0.1046) < 0.001

    # Annualized volatility should be near 0 (constant returns)
    # Allow for floating point precision
    assert abs(metrics["annualized_volatility"]) < 1e-10

    # Max drawdown should be 0 (no negative returns)
    assert metrics["max_drawdown"] == 0.0


def test_performance_metrics_handles_empty_returns() -> None:
    """Test performance metrics handle empty returns gracefully."""
    config = BacktestConfig(
        start_date=datetime(2020, 1, 1, tzinfo=UTC),
        end_date=datetime(2020, 1, 10, tzinfo=UTC),
    )
    backtester = FactorBacktester(config)

    metrics = backtester._compute_performance_metrics(
        returns=[],
        transaction_costs=[],
        rebalances=0,
    )

    # All metrics should be zero
    assert metrics["total_return"] == 0.0
    assert metrics["annualized_return"] == 0.0
    assert metrics["sharpe_ratio"] == 0.0


def test_max_drawdown_calculation() -> None:
    """Test maximum drawdown calculation is correct."""
    config = BacktestConfig(
        start_date=datetime(2020, 1, 1, tzinfo=UTC),
        end_date=datetime(2020, 1, 10, tzinfo=UTC),
    )
    backtester = FactorBacktester(config)

    # Returns that go: +10%, -5%, -5% (peak then drawdown)
    returns = [0.10, -0.05, -0.05]
    transaction_costs = [0.0] * 3
    rebalances = 1

    metrics = backtester._compute_performance_metrics(
        returns,
        transaction_costs,
        rebalances,
    )

    # Cumulative: 1.10, 1.045, 0.99275
    # Drawdown from peak: (0.99275 - 1.10) / 1.10 ≈ -0.0975
    assert metrics["max_drawdown"] < -0.09
    assert metrics["max_drawdown"] > -0.10


# ===== Test: Portfolio Value Evolution =====


def test_portfolio_value_evolution() -> None:
    """Test portfolio value compounds returns correctly."""
    # Create simple dataset with known returns
    dates = [datetime(2020, 1, i, tzinfo=UTC) for i in range(1, 6)]
    data = []
    for date in dates:
        data.append({
            "timestamp": date,
            "symbol": "XLK",
            "return": 0.01,  # 1% daily
        })

    sector_returns = pl.DataFrame(data)

    dataset = SectorDataset(
        sector_returns=sector_returns,
        factor_returns=pl.DataFrame({
            "timestamp": dates,
            "factor_duration": [0.0] * len(dates),
        }),
        coverage=CoverageSummary(
            calendar_name="XNYS",
            sector_expected_days=5,
            factor_expected_days=5,
            sector_coverage={"XLK": 1.0},
            factor_coverage={"factor_duration": 1.0},
        ),
    )

    config = BacktestConfig(
        start_date=datetime(2020, 1, 1, tzinfo=UTC),
        end_date=datetime(2020, 1, 5, tzinfo=UTC),
        initial_capital=1_000_000.0,
        transaction_cost_bps=0.0,  # No costs for clean test
        rebalance_frequency="daily",
    )

    backtester = FactorBacktester(config)
    result = backtester.run_backtest(dataset, strategy="equal_weight")

    # Final value should be 1M × (1.01)^4 ≈ 1,040,604
    # (4 days of returns, not 5, since first day is initialization)
    expected_final_value = 1_000_000.0 * (1.01 ** 4)
    actual_final_value = result.portfolio_values[-1]

    assert abs(actual_final_value - expected_final_value) < 1000.0


# ===== Test: Position Constraints =====


def test_position_constraints_enforced() -> None:
    """Test position limits are enforced correctly."""
    config = BacktestConfig(
        start_date=datetime(2020, 1, 1, tzinfo=UTC),
        end_date=datetime(2020, 1, 31, tzinfo=UTC),
        position_limits={
            "XLK": (0.0, 0.2),  # Max 20% in tech
            "XLF": (0.1, 0.5),  # Min 10%, max 50% in financials
        },
    )

    backtester = FactorBacktester(config)

    # Original weights: equal
    weights = {"XLK": 0.33, "XLF": 0.33, "XLU": 0.34}

    # Apply constraints
    constrained = backtester._apply_position_limits(weights)

    # XLK should be capped at 0.2 (with small tolerance for renormalization)
    assert constrained["XLK"] <= 0.201

    # XLF should be at least 0.1
    assert constrained["XLF"] >= 0.099

    # Weights should sum to 1.0
    assert abs(sum(constrained.values()) - 1.0) < 1e-10


# ===== Test: Rebalance Threshold Trigger =====


def test_rebalance_threshold_trigger() -> None:
    """Test rebalance triggers when weight deviates beyond threshold."""
    config = BacktestConfig(
        start_date=datetime(2020, 1, 1, tzinfo=UTC),
        end_date=datetime(2020, 12, 31, tzinfo=UTC),
        rebalance_threshold=0.05,
    )
    backtester = FactorBacktester(config)

    current_weights = {"XLK": 0.50, "XLF": 0.50}
    target_weights = {"XLK": 0.50, "XLF": 0.50}

    # No deviation: should not trigger
    assert not backtester._check_rebalance_trigger(current_weights, target_weights)

    # Small deviation (4%): should not trigger
    current_weights = {"XLK": 0.54, "XLF": 0.46}
    assert not backtester._check_rebalance_trigger(current_weights, target_weights)

    # Large deviation (6%): should trigger
    current_weights = {"XLK": 0.56, "XLF": 0.44}
    assert backtester._check_rebalance_trigger(current_weights, target_weights)


# ===== Test: Result Structure =====


def test_backtest_result_structure(
    simple_config: BacktestConfig,
    sample_dataset: SectorDataset,
) -> None:
    """Test backtest result contains all required fields."""
    backtester = FactorBacktester(simple_config)
    result = backtester.run_backtest(
        dataset=sample_dataset,
        strategy="equal_weight",
    )

    # Check all required fields exist
    assert result.strategy_name == "equal_weight"
    assert result.start_date == simple_config.start_date
    assert result.end_date == simple_config.end_date

    # Time series data
    assert len(result.dates) > 0
    assert len(result.portfolio_values) == len(result.dates)
    assert len(result.returns) == len(result.dates) - 1  # One less than dates

    # Positions DataFrame
    assert not result.positions.is_empty()
    assert "timestamp" in result.positions.columns

    # Performance metrics
    assert isinstance(result.total_return, float)
    assert isinstance(result.sharpe_ratio, float)
    assert isinstance(result.max_drawdown, float)

    # Transaction metrics
    assert result.num_rebalances > 0
    assert result.total_transaction_costs >= 0.0


def test_backtest_handles_empty_dataset() -> None:
    """Test backtest raises error with empty dataset."""
    config = BacktestConfig(
        start_date=datetime(2020, 1, 1, tzinfo=UTC),
        end_date=datetime(2020, 12, 31, tzinfo=UTC),
    )

    empty_dataset = SectorDataset(
        sector_returns=pl.DataFrame({
            "timestamp": [],
            "symbol": [],
            "return": [],
        }),
        factor_returns=pl.DataFrame({
            "timestamp": [],
            "factor_duration": [],
        }),
        coverage=CoverageSummary(
            calendar_name="XNYS",
            sector_expected_days=0,
            factor_expected_days=0,
            sector_coverage={},
            factor_coverage={},
        ),
    )

    backtester = FactorBacktester(config)

    with pytest.raises(ValueError, match="Sector returns dataset is empty"):
        backtester.run_backtest(empty_dataset, strategy="equal_weight")
