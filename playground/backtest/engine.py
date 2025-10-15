"""
Backtesting engine for factor-based portfolio strategies.

This module provides a robust backtesting framework for evaluating portfolio
allocation strategies using historical sector ETF returns and factor data.

Key Features:
- Monthly rebalancing with realistic transaction costs
- Multiple strategy support (factor-based, equal-weight, benchmarks)
- Comprehensive performance metrics (Sharpe, Calmar, drawdown, etc.)
- No look-ahead bias enforcement
- Deterministic execution with fixed random seed

Performance Targets (Cold Path):
- Full backtest (2010-2024): < 30 seconds
- Monthly rebalance computation: < 100ms
- Performance metric calculation: < 1 second

Hot/Cold Path Separation:
- This is a cold-path module (backtesting is offline analysis)
- No real-time constraints, optimized for correctness over speed
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
import structlog

from playground.backtest.strategies import FactorTiltStrategy


if TYPE_CHECKING:
    from playground.risk_model.dataset import SectorDataset


LOGGER = structlog.get_logger(__name__)


# ===== Configuration Classes =====


@dataclass(slots=True)
class BacktestConfig:
    """Configuration for backtest run."""

    start_date: datetime
    end_date: datetime
    initial_capital: float = 1_000_000.0  # $1M default
    rebalance_frequency: str = "monthly"  # "daily", "weekly", "monthly"
    transaction_cost_bps: float = 10.0  # 10 basis points
    slippage_bps: float = 0.0  # Optional market impact
    position_limits: dict[str, tuple[float, float]] | None = None  # (min, max) weights per sector
    rebalance_threshold: float = 0.05  # Trigger if weight deviates >5%
    random_seed: int = 42  # For reproducibility

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.end_date <= self.start_date:
            msg = "End date must be after start date"
            raise ValueError(msg)

        if self.initial_capital <= 0:
            msg = "Initial capital must be positive"
            raise ValueError(msg)

        if self.transaction_cost_bps < 0:
            msg = "Transaction cost must be non-negative"
            raise ValueError(msg)

        if self.slippage_bps < 0:
            msg = "Slippage must be non-negative"
            raise ValueError(msg)

        if self.rebalance_frequency not in {"daily", "weekly", "monthly"}:
            msg = f"Invalid rebalance frequency: {self.rebalance_frequency}"
            raise ValueError(msg)

        if self.rebalance_threshold < 0 or self.rebalance_threshold > 1:
            msg = "Rebalance threshold must be between 0 and 1"
            raise ValueError(msg)


@dataclass(slots=True)
class BacktestResult:
    """Results from a backtest run."""

    strategy_name: str
    start_date: datetime
    end_date: datetime

    # Time series
    dates: list[datetime]
    portfolio_values: list[float]
    returns: list[float]  # Daily returns
    positions: pl.DataFrame  # timestamp × sector weights

    # Performance metrics
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float

    # Transaction metrics
    total_transaction_costs: float
    turnover_rate: float  # Average monthly turnover
    num_rebalances: int

    # Factor exposures (if applicable)
    avg_duration_exposure: float | None = None
    avg_credit_exposure: float | None = None
    avg_liquidity_exposure: float | None = None


# ===== Core Backtesting Engine =====


class FactorBacktester:
    """
    Backtesting engine for factor-based portfolio strategies.

    This class simulates portfolio performance over historical data with realistic
    constraints including transaction costs, rebalancing schedules, and position limits.

    Performance Targets (Cold Path):
    - Full backtest (2010-2024): < 30 seconds
    - Monthly rebalance computation: < 100ms
    - Performance metric calculation: < 1 second

    Examples
    --------
    >>> config = BacktestConfig(
    ...     start_date=datetime(2010, 1, 1),
    ...     end_date=datetime(2024, 12, 31),
    ...     rebalance_frequency="monthly",
    ...     transaction_cost_bps=10.0,
    ... )
    >>> backtester = FactorBacktester(config)
    >>> result = backtester.run_backtest(dataset, strategy="equal_weight")
    >>> print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    """

    def __init__(self, config: BacktestConfig) -> None:
        """
        Initialize backtester with configuration.

        Parameters
        ----------
        config : BacktestConfig
            Backtest configuration including dates, costs, and constraints.
        """
        self.config = config
        self.logger = LOGGER.bind(
            start_date=config.start_date.isoformat(),
            end_date=config.end_date.isoformat(),
        )
        np.random.seed(config.random_seed)
        self._strategy_cache: dict[str, FactorTiltStrategy] = {}

    def run_backtest(
        self,
        dataset: SectorDataset,
        strategy: str,
        strategy_params: dict[str, object] | None = None,
    ) -> BacktestResult:
        """
        Run backtest for specified strategy.

        Parameters
        ----------
        dataset : SectorDataset
            Historical sector returns and factor data.
        strategy : str
            Strategy name: "equal_weight", "3d_factor_stable", "3d_factor_rolling".
        strategy_params : dict, optional
            Strategy-specific parameters.

        Returns
        -------
        BacktestResult
            Complete backtest results with performance metrics.

        Raises
        ------
        ValueError
            If strategy is not recognized or dataset is invalid.
        """
        if strategy_params is None:
            strategy_params = {}

        self.logger.info("Starting backtest", strategy=strategy)

        # Validate dataset
        if dataset.sector_returns.is_empty():
            msg = "Sector returns dataset is empty"
            raise ValueError(msg)

        # Filter dataset to backtest period
        filtered_returns = self._filter_to_backtest_period(dataset.sector_returns)
        if filtered_returns.is_empty():
            msg = "No data available in backtest period"
            raise ValueError(msg)

        # Get unique trading dates
        trading_dates = sorted(filtered_returns["timestamp"].unique().to_list())
        if not trading_dates:
            msg = "No trading dates found"
            raise ValueError(msg)

        # Get rebalance dates
        rebalance_dates = self._get_rebalance_dates(
            self.config.start_date,
            self.config.end_date,
            self.config.rebalance_frequency,
            trading_dates,
        )

        # Get list of sectors
        sectors = sorted(filtered_returns["symbol"].unique().to_list())

        # Initialize tracking variables
        portfolio_value = self.config.initial_capital
        current_weights: dict[str, float] = {}
        portfolio_values: list[float] = [portfolio_value]
        dates: list[datetime] = [trading_dates[0]]
        returns_list: list[float] = []
        transaction_costs_list: list[float] = []
        position_records: list[dict[str, object]] = []
        num_rebalances = 0

        # Main backtest loop
        for i, date in enumerate(trading_dates):
            # Check if rebalance is needed
            should_rebalance = date in rebalance_dates

            if should_rebalance or not current_weights:
                # Compute target weights
                target_weights = self._compute_target_weights(
                    strategy=strategy,
                    date=date,
                    dataset=dataset,
                    sectors=sectors,
                    params=strategy_params,
                )

                # Apply position limits if specified
                if self.config.position_limits is not None:
                    target_weights = self._apply_position_limits(target_weights)

                # Calculate transaction costs
                transaction_cost = self._apply_transaction_costs(
                    current_weights,
                    target_weights,
                    portfolio_value,
                )
                transaction_costs_list.append(transaction_cost)

                # Deduct transaction costs
                portfolio_value -= transaction_cost

                # Update weights
                current_weights = target_weights.copy()
                num_rebalances += 1

                # Record positions
                position_records.append({
                    "timestamp": date,
                    **{sector: current_weights.get(sector, 0.0) for sector in sectors},
                })

                self.logger.debug(
                    "Rebalanced portfolio",
                    date=date.isoformat(),
                    transaction_cost=transaction_cost,
                    num_positions=len([w for w in current_weights.values() if w > 0]),
                )
            else:
                transaction_costs_list.append(0.0)

            # Calculate daily returns if not first day
            if i > 0:
                # Get returns for this date
                daily_returns = filtered_returns.filter(pl.col("timestamp") == date)

                # Calculate portfolio return as weighted sum
                portfolio_return = 0.0
                for sector in sectors:
                    weight = current_weights.get(sector, 0.0)
                    if weight > 0:
                        sector_return_data = daily_returns.filter(pl.col("symbol") == sector)
                        if not sector_return_data.is_empty():
                            sector_return = float(sector_return_data["return"][0])
                            portfolio_return += weight * sector_return

                # Update portfolio value
                portfolio_value *= (1 + portfolio_return)

                # Track return
                returns_list.append(portfolio_return)

                # Update drift in weights (prices change but we don't rebalance)
                if current_weights:
                    total_weight_after_drift = sum(
                        weight * (1 + self._get_sector_return(daily_returns, sector))
                        for sector, weight in current_weights.items()
                    )
                    if total_weight_after_drift > 0:
                        current_weights = {
                            sector: weight * (1 + self._get_sector_return(daily_returns, sector)) / total_weight_after_drift
                            for sector, weight in current_weights.items()
                        }

                # Record tracking
                portfolio_values.append(portfolio_value)
                dates.append(date)

        # Create positions DataFrame
        positions_df = pl.DataFrame(position_records)

        # Compute performance metrics
        metrics = self._compute_performance_metrics(
            returns_list,
            transaction_costs_list,
            num_rebalances,
        )

        # Calculate factor exposures if 3D factor strategy
        avg_duration_exposure = None
        avg_credit_exposure = None
        avg_liquidity_exposure = None

        if "3d_factor" in strategy:
            # Would compute from positions and factor betas
            # For now, set to None (can be enhanced later)
            pass

        result = BacktestResult(
            strategy_name=strategy,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            dates=dates,
            portfolio_values=portfolio_values,
            returns=returns_list,
            positions=positions_df,
            total_return=metrics["total_return"],
            annualized_return=metrics["annualized_return"],
            annualized_volatility=metrics["annualized_volatility"],
            sharpe_ratio=metrics["sharpe_ratio"],
            max_drawdown=metrics["max_drawdown"],
            calmar_ratio=metrics["calmar_ratio"],
            total_transaction_costs=metrics["total_transaction_costs"],
            turnover_rate=metrics["turnover_rate"],
            num_rebalances=num_rebalances,
            avg_duration_exposure=avg_duration_exposure,
            avg_credit_exposure=avg_credit_exposure,
            avg_liquidity_exposure=avg_liquidity_exposure,
        )

        self.logger.info(
            "Backtest completed",
            strategy=strategy,
            total_return=f"{result.total_return:.2%}",
            sharpe_ratio=f"{result.sharpe_ratio:.2f}",
            num_rebalances=num_rebalances,
        )

        return result

    def _filter_to_backtest_period(self, returns: pl.DataFrame) -> pl.DataFrame:
        """Filter returns to backtest period."""
        return returns.filter(
            (pl.col("timestamp") >= self.config.start_date)
            & (pl.col("timestamp") <= self.config.end_date)
        ).sort("timestamp")

    def _get_rebalance_dates(
        self,
        start: datetime,
        end: datetime,
        frequency: str,
        trading_dates: list[datetime],
    ) -> list[datetime]:
        """
        Generate rebalance dates based on frequency.

        Parameters
        ----------
        start : datetime
            Start date.
        end : datetime
            End date.
        frequency : str
            Rebalance frequency: "daily", "weekly", "monthly".
        trading_dates : list[datetime]
            Available trading dates.

        Returns
        -------
        list[datetime]
            List of rebalance dates.
        """
        if frequency == "daily":
            return trading_dates

        if frequency == "weekly":
            # Rebalance every Friday (or last trading day of week)
            rebalance_dates = []
            current_week = None
            for date in trading_dates:
                week_num = date.isocalendar()[1]
                if current_week is None or week_num != current_week:
                    rebalance_dates.append(date)
                    current_week = week_num
            return rebalance_dates

        if frequency == "monthly":
            # Rebalance on last trading day of each month
            rebalance_dates = []
            current_month = None
            last_date = None
            for date in trading_dates:
                month_key = (date.year, date.month)
                if current_month is None:
                    current_month = month_key
                    last_date = date
                elif month_key != current_month:
                    if last_date is not None:
                        rebalance_dates.append(last_date)
                    current_month = month_key
                    last_date = date
                else:
                    last_date = date

            # Add final month
            if last_date is not None:
                rebalance_dates.append(last_date)

            return rebalance_dates

        msg = f"Invalid rebalance frequency: {frequency}"
        raise ValueError(msg)

    def _compute_target_weights(
        self,
        strategy: str,
        date: datetime,
        dataset: SectorDataset,
        sectors: list[str],
        params: dict[str, object],
    ) -> dict[str, float]:
        """
        Compute target portfolio weights for given date.

        Parameters
        ----------
        strategy : str
            Strategy name.
        date : datetime
            Current date.
        dataset : SectorDataset
            Historical data.
        sectors : list[str]
            List of sector symbols.
        params : dict
            Strategy parameters.

        Returns
        -------
        dict[str, float]
            Target weights for each sector.
        """
        if strategy == "equal_weight":
            # Equal weight across all sectors
            n_sectors = len(sectors)
            if n_sectors == 0:
                return {}
            weight = 1.0 / n_sectors
            return dict.fromkeys(sectors, weight)

        if strategy in {"3d_factor_stable", "3d_factor_rolling"}:
            factor_strategy = self._get_factor_strategy(strategy, params)
            weights = factor_strategy.compute_weights(date, dataset)

            if not weights:
                self.logger.warning(
                    "Factor strategy returned no weights, reverting to equal weight",
                    strategy=strategy,
                )
                n_sectors = len(sectors)
                if n_sectors == 0:
                    return {}
                weight = 1.0 / n_sectors
                return dict.fromkeys(sectors, weight)

            return weights

        msg = f"Unknown strategy: {strategy}"
        raise ValueError(msg)

    def _get_factor_strategy(
        self,
        strategy: str,
        params: dict[str, object],
    ) -> FactorTiltStrategy:
        """Return cached factor strategy instance."""
        if strategy not in self._strategy_cache:
            rolling = strategy == "3d_factor_rolling"

            def _coerce_int(value: object | None, default: int) -> int:
                if isinstance(value, bool):
                    return int(value)
                if isinstance(value, int):
                    return value
                if isinstance(value, float):
                    return int(value)
                if isinstance(value, str):
                    try:
                        return int(value.strip())
                    except ValueError:
                        return default
                return default

            def _coerce_float(value: object | None, default: float) -> float:
                if isinstance(value, bool):
                    return float(value)
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, str):
                    try:
                        return float(value.strip())
                    except ValueError:
                        return default
                return default

            def _coerce_optional_float(value: object | None) -> float | None:
                if value is None:
                    return None
                return _coerce_float(value, 0.0)

            rolling_window = _coerce_int(params.get("rolling_window"), 252)
            factor_forecasts = params.get("factor_forecasts")
            max_weight = _coerce_optional_float(params.get("max_weight"))
            min_weight = _coerce_float(params.get("min_weight"), 0.0)
            volatility_floor = _coerce_float(params.get("volatility_floor"), 1e-3)
            min_observations = _coerce_int(params.get("min_observations"), 60)
            blend_to_equal = _coerce_float(params.get("blend_to_equal"), 0.25)
            turnover_smoothing = _coerce_float(params.get("turnover_smoothing"), 0.0)
            dynamic_factor_scaling = bool(params.get("dynamic_factor_scaling", False))
            scaling_lookback = _coerce_int(params.get("scaling_lookback"), 126)
            scaling_threshold = _coerce_float(params.get("scaling_threshold"), 0.01)
            scaling_floor = _coerce_float(params.get("scaling_floor"), 0.4)
            regime_scaling = bool(params.get("regime_scaling", False))
            regime_scaling_floor = _coerce_float(params.get("regime_scaling_floor"), 0.3)
            regime_scaling_map_param = params.get("regime_scaling_map")
            regime_scaling_map: Mapping[str, float] | None = None
            if isinstance(regime_scaling_map_param, Mapping):
                regime_scaling_map = {}
                for key, value in regime_scaling_map_param.items():
                    regime_scaling_map[str(key)] = _coerce_float(value, 1.0)
            regime_factor_param = params.get("regime_factor_multipliers")
            regime_factor_multipliers: Mapping[str, Mapping[str, float]] | None = None
            if isinstance(regime_factor_param, Mapping):
                nested: dict[str, Mapping[str, float]] = {}
                for regime_name, factor_map in regime_factor_param.items():
                    if isinstance(factor_map, Mapping):
                        normalized: dict[str, float] = {}
                        for factor, multiplier in factor_map.items():
                            normalized[str(factor)] = _coerce_float(multiplier, 1.0)
                        nested[str(regime_name)] = normalized
                regime_factor_multipliers = nested if nested else None
            regime_resolver_param = params.get("regime_resolver")
            regime_resolver: Callable[[datetime], str | None] | None = None
            if callable(regime_resolver_param):
                regime_resolver = cast(Callable[[datetime], str | None], regime_resolver_param)

            strategy_instance = FactorTiltStrategy(
                use_rolling_betas=rolling,
                rolling_window=rolling_window,
                factor_forecasts=factor_forecasts,  # type: ignore[arg-type]
                min_weight=min_weight,
                max_weight=max_weight,
                volatility_floor=volatility_floor,
                min_observations=min_observations,
                blend_to_equal=blend_to_equal,
                turnover_smoothing=turnover_smoothing,
                dynamic_factor_scaling=dynamic_factor_scaling,
                scaling_lookback=scaling_lookback,
                scaling_threshold=scaling_threshold,
                scaling_floor=scaling_floor,
                regime_scaling=regime_scaling,
                regime_scaling_map=regime_scaling_map,
                regime_scaling_floor=regime_scaling_floor,
                regime_resolver=regime_resolver,
                regime_factor_multipliers=regime_factor_multipliers,
            )
            self._strategy_cache[strategy] = strategy_instance
        else:
            strategy_instance = self._strategy_cache[strategy]
            if "factor_forecasts" in params and params["factor_forecasts"] is not None:
                strategy_instance.factor_forecasts = params["factor_forecasts"]  # type: ignore[assignment]

        return self._strategy_cache[strategy]

    def _apply_position_limits(
        self,
        weights: dict[str, float],
    ) -> dict[str, float]:
        """
        Apply position limits to weights.

        Parameters
        ----------
        weights : dict[str, float]
            Original weights.

        Returns
        -------
        dict[str, float]
            Constrained weights.

        Notes
        -----
        This method applies min/max constraints iteratively until convergence,
        ensuring all limits are respected while maintaining sum-to-one constraint.
        """
        if self.config.position_limits is None:
            return weights

        constrained_weights = weights.copy()

        # Iterative algorithm to enforce limits while maintaining sum=1.0
        max_iterations = 100
        for _ in range(max_iterations):
            violations = False

            # Apply constraints
            for sector in constrained_weights:
                if sector in self.config.position_limits:
                    min_weight, max_weight = self.config.position_limits[sector]
                    current = constrained_weights[sector]

                    if current < min_weight:
                        constrained_weights[sector] = min_weight
                        violations = True
                    elif current > max_weight:
                        constrained_weights[sector] = max_weight
                        violations = True

            if not violations:
                break

            # Renormalize to sum to 1.0
            total_weight = sum(constrained_weights.values())
            if total_weight > 0:
                constrained_weights = {
                    sector: weight / total_weight
                    for sector, weight in constrained_weights.items()
                }

        return constrained_weights

    def _apply_transaction_costs(
        self,
        old_weights: dict[str, float],
        new_weights: dict[str, float],
        portfolio_value: float,
    ) -> float:
        """
        Calculate transaction costs from rebalancing.

        Cost = Σ |new_weight - old_weight| × portfolio_value × (tc_bps + slippage_bps) / 10000

        Parameters
        ----------
        old_weights : dict[str, float]
            Previous weights.
        new_weights : dict[str, float]
            Target weights.
        portfolio_value : float
            Current portfolio value.

        Returns
        -------
        float
            Transaction cost in dollars.
        """
        all_sectors = set(old_weights.keys()) | set(new_weights.keys())

        turnover = sum(
            abs(new_weights.get(sector, 0.0) - old_weights.get(sector, 0.0))
            for sector in all_sectors
        )

        cost_bps = self.config.transaction_cost_bps + self.config.slippage_bps
        transaction_cost = turnover * portfolio_value * (cost_bps / 10_000)

        return transaction_cost

    def _check_rebalance_trigger(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> bool:
        """
        Determine if rebalance is needed based on threshold.

        Parameters
        ----------
        current_weights : dict[str, float]
            Current weights after drift.
        target_weights : dict[str, float]
            Target weights.

        Returns
        -------
        bool
            True if rebalance needed.
        """
        all_sectors = set(current_weights.keys()) | set(target_weights.keys())

        for sector in all_sectors:
            current = current_weights.get(sector, 0.0)
            target = target_weights.get(sector, 0.0)
            if abs(current - target) > self.config.rebalance_threshold:
                return True

        return False

    def _get_sector_return(
        self,
        daily_returns: pl.DataFrame,
        sector: str,
    ) -> float:
        """Get return for a specific sector on a given date."""
        sector_data = daily_returns.filter(pl.col("symbol") == sector)
        if sector_data.is_empty():
            return 0.0
        return float(sector_data["return"][0])

    def _compute_performance_metrics(
        self,
        returns: list[float],
        transaction_costs: list[float],
        rebalances: int,
    ) -> dict[str, float]:
        """
        Calculate all performance metrics.

        Parameters
        ----------
        returns : list[float]
            Daily returns.
        transaction_costs : list[float]
            Transaction costs per day.
        rebalances : int
            Number of rebalances.

        Returns
        -------
        dict[str, float]
            Performance metrics.
        """
        if not returns:
            return {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "annualized_volatility": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "calmar_ratio": 0.0,
                "total_transaction_costs": 0.0,
                "turnover_rate": 0.0,
            }

        returns_arr = np.array(returns)

        # Total return
        cumulative_return = np.prod(1 + returns_arr) - 1

        # Annualized return (assuming 252 trading days per year)
        n_days = len(returns)
        n_years = n_days / 252.0
        if n_years > 0:
            annualized_return = (1 + cumulative_return) ** (1 / n_years) - 1
        else:
            annualized_return = 0.0

        # Annualized volatility
        if len(returns_arr) > 1:
            daily_vol = float(np.std(returns_arr, ddof=1))
            annualized_vol = daily_vol * np.sqrt(252)
        else:
            annualized_vol = 0.0

        # Sharpe ratio (assuming risk-free rate = 0)
        if annualized_vol > 0:
            sharpe_ratio = annualized_return / annualized_vol
        else:
            sharpe_ratio = 0.0

        # Maximum drawdown
        cumulative = np.cumprod(1 + returns_arr)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = float(np.min(drawdown))

        # Calmar ratio
        if max_drawdown < 0:
            calmar_ratio = annualized_return / abs(max_drawdown)
        else:
            calmar_ratio = 0.0

        # Transaction costs
        total_transaction_costs = sum(transaction_costs)

        # Turnover rate (average per rebalance)
        if rebalances > 0:
            # Turnover is already computed in transaction costs
            # Approximate average monthly turnover
            months = n_years * 12
            turnover_rate = rebalances / max(months, 1)
        else:
            turnover_rate = 0.0

        return {
            "total_return": float(cumulative_return),
            "annualized_return": float(annualized_return),
            "annualized_volatility": float(annualized_vol),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "calmar_ratio": float(calmar_ratio),
            "total_transaction_costs": float(total_transaction_costs),
            "turnover_rate": float(turnover_rate),
        }


__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "FactorBacktester",
]
