"""
Benchmark portfolio strategies for performance comparison.

This module implements classic portfolio allocation strategies that serve as
benchmarks for evaluating the 3D Factor Model:

1. **60/40 Portfolio**: Classic balanced allocation (60% equities, 40% bonds)
2. **Risk Parity**: Equal risk contribution across assets
3. **Minimum Variance**: Optimal low-risk portfolio

These benchmarks are used in Phase 3.2 (out-of-sample testing) and Phase 3.3
(regime analysis) to compare against the factor-based strategies.

Performance Targets (Cold Path):
- Weight computation: < 100ms per rebalance
- Portfolio optimization: < 500ms per rebalance
- Full backtest (2010-2024): < 60 seconds

Hot/Cold Path Separation:
- This is a cold-path module (backtesting is offline analysis)
- No real-time constraints, optimized for correctness over speed

Integration Notes:
- All strategies implement the same compute_weights(date, dataset) interface
- No look-ahead bias: only use data up to `date`
- All weights sum to 1.0 and respect min/max constraints
- Compatible with FactorBacktester from engine.py
"""

from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import structlog
from scipy.optimize import minimize


if TYPE_CHECKING:
    from playground.risk_model.dataset import SectorDataset


LOGGER = structlog.get_logger(__name__)


# ===== Constants =====

TRADING_DAYS_PER_YEAR = 252
EPSILON = 1e-10  # Small value to prevent division by zero


# ===== 60/40 Benchmark Strategy =====


class SixtyFortyStrategy:
    """
    Classic 60/40 portfolio (60% equities, 40% bonds).

    Allocates 60% to SPY (S&P 500) and 40% to AGG (Aggregate Bonds)
    with monthly rebalancing.

    This is a widely-used benchmark for balanced portfolios, providing
    diversification between stocks and bonds.

    If SPY/AGG are not available in the dataset, this strategy uses sector
    ETFs as proxies:
    - Equity proxy: 60% weighted across growth sectors (XLK, XLY, XLC)
    - Bond proxy: 40% to defensive sectors (XLU, XLV)

    Examples
    --------
    >>> strategy = SixtyFortyStrategy()
    >>> weights = strategy.compute_weights(date, dataset)
    >>> print(weights)
    {'SPY': 0.60, 'AGG': 0.40}

    Performance Expectations (2010-2024):
    - Annualized Return: ~8-10%
    - Annualized Volatility: ~10%
    - Sharpe Ratio: ~0.7
    """

    def __init__(
        self,
        equity_weight: float = 0.60,
        bond_weight: float = 0.40,
        use_sector_proxies: bool = True,
    ) -> None:
        """
        Initialize 60/40 strategy.

        Parameters
        ----------
        equity_weight : float, default 0.60
            Allocation to equities (default 60%)
        bond_weight : float, default 0.40
            Allocation to bonds (default 40%)
        use_sector_proxies : bool, default True
            If True and SPY/AGG not available, use sector ETFs as proxies

        Raises
        ------
        ValueError
            If weights don't sum to 1.0
        """
        if abs(equity_weight + bond_weight - 1.0) > EPSILON:
            msg = f"Weights must sum to 1.0, got {equity_weight + bond_weight:.6f}"
            raise ValueError(msg)

        if equity_weight < 0 or bond_weight < 0:
            msg = "Weights must be non-negative"
            raise ValueError(msg)

        self.equity_weight = equity_weight
        self.bond_weight = bond_weight
        self.use_sector_proxies = use_sector_proxies
        self.logger = LOGGER.bind(strategy="60_40")

    def compute_weights(
        self,
        date: datetime,
        dataset: SectorDataset,
    ) -> dict[str, float]:
        """
        Compute 60/40 allocation.

        Parameters
        ----------
        date : datetime
            Current rebalance date (not used, allocation is static)
        dataset : SectorDataset
            Historical sector returns and factor data

        Returns
        -------
        dict[str, float]
            {"SPY": 0.60, "AGG": 0.40} or sector proxy weights

        Notes
        -----
        If SPY/AGG are not available and use_sector_proxies is True:
        - Equity allocation (60%) split across: XLK (tech), XLY (consumer), XLC (comm)
        - Bond allocation (40%) split across: XLU (utilities), XLV (healthcare)
        """
        # Get available sectors at this date (no look-ahead bias)
        available_sectors = self._get_available_sectors(date, dataset)

        if not available_sectors:
            self.logger.warning("No sectors available", date=date.isoformat())
            return {}

        # Check if SPY and AGG are available
        has_spy = "SPY" in available_sectors
        has_agg = "AGG" in available_sectors

        if has_spy and has_agg:
            # Standard 60/40 allocation
            result_weights = {
                "SPY": self.equity_weight,
                "AGG": self.bond_weight,
            }
            self.logger.debug(
                "Using standard 60/40 allocation",
                date=date.isoformat(),
            )
            return result_weights

        if not self.use_sector_proxies:
            self.logger.warning(
                "SPY/AGG not available and sector proxies disabled",
                date=date.isoformat(),
            )
            return {}

        # Use sector ETF proxies
        # Equity sectors (growth): XLK (tech), XLY (consumer discretionary), XLC (communication)
        equity_sectors = [s for s in ["XLK", "XLY", "XLC"] if s in available_sectors]
        # Bond proxies (defensive): XLU (utilities), XLV (healthcare)
        bond_sectors = [s for s in ["XLU", "XLV"] if s in available_sectors]

        if not equity_sectors and not bond_sectors:
            self.logger.warning(
                "No equity or bond proxy sectors available",
                date=date.isoformat(),
            )
            return {}

        weights: dict[str, float] = {}

        # Allocate equity weight across equity sectors
        if equity_sectors:
            equity_weight_per_sector = self.equity_weight / len(equity_sectors)
            for sector in equity_sectors:
                weights[sector] = equity_weight_per_sector

        # Allocate bond weight across bond sectors
        if bond_sectors:
            bond_weight_per_sector = self.bond_weight / len(bond_sectors)
            for sector in bond_sectors:
                weights[sector] = bond_weight_per_sector

        # Renormalize to sum to 1.0 (in case only equity or bond sectors available)
        total_weight = sum(weights.values())
        if total_weight > EPSILON:
            weights = {sector: w / total_weight for sector, w in weights.items()}

        self.logger.debug(
            "Using sector proxy allocation",
            date=date.isoformat(),
            num_equity_sectors=len(equity_sectors),
            num_bond_sectors=len(bond_sectors),
        )

        return weights

    def _get_available_sectors(
        self,
        date: datetime,
        dataset: SectorDataset,
    ) -> list[str]:
        """Get list of sectors with data available up to date."""
        # Filter to data before or on date (no look-ahead bias)
        historical_data = dataset.sector_returns.filter(
            pl.col("timestamp") <= date
        )

        if historical_data.is_empty():
            return []

        return sorted(historical_data["symbol"].unique().to_list())


# ===== Risk Parity Strategy =====


class RiskParityStrategy:
    """
    Risk parity portfolio strategy.

    Allocates weights such that each asset contributes equally to portfolio risk.
    Uses inverse volatility weighting as a simple approximation.

    Formula: w_i = (1/σ_i) / Σ(1/σ_j)

    Where σ_i is the rolling volatility of asset i.

    This strategy aims for balanced risk exposure across assets, typically
    resulting in lower overall portfolio volatility than equal-weight or
    market-cap weighted portfolios.

    Examples
    --------
    >>> strategy = RiskParityStrategy(lookback_days=126)
    >>> weights = strategy.compute_weights(date, dataset)
    >>> print(weights)
    {'XLU': 0.15, 'XLV': 0.14, ...}  # Lower vol sectors get higher weights

    Performance Expectations (2010-2024):
    - Annualized Return: ~6-8%
    - Annualized Volatility: ~8% (lower than 60/40)
    - Sharpe Ratio: ~0.8
    """

    def __init__(
        self,
        lookback_days: int = 126,  # 6 months
        min_weight: float = 0.01,  # Minimum 1% per asset
        max_weight: float = 0.50,  # Maximum 50% per asset
        min_observations: int = 20,  # Minimum observations for volatility estimation
    ) -> None:
        """
        Initialize risk parity strategy.

        Parameters
        ----------
        lookback_days : int, default 126
            Rolling window for volatility estimation (default 126 = 6 months)
        min_weight : float, default 0.01
            Minimum allocation per asset (default 1%)
        max_weight : float, default 0.50
            Maximum allocation per asset (default 50%)
        min_observations : int, default 20
            Minimum observations required for volatility estimation

        Raises
        ------
        ValueError
            If parameters are invalid
        """
        if lookback_days <= 0:
            msg = "Lookback days must be positive"
            raise ValueError(msg)

        if not (0 <= min_weight <= max_weight <= 1):
            msg = "Must have 0 <= min_weight <= max_weight <= 1"
            raise ValueError(msg)

        if min_observations <= 1:
            msg = "Minimum observations must be > 1"
            raise ValueError(msg)

        self.lookback_days = lookback_days
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_observations = min_observations
        self.logger = LOGGER.bind(strategy="risk_parity")

    def compute_weights(
        self,
        date: datetime,
        dataset: SectorDataset,
    ) -> dict[str, float]:
        """
        Compute risk parity weights based on inverse volatility.

        Algorithm:
        1. Estimate rolling volatility for each sector (lookback_days window)
        2. Compute inverse volatility: inv_vol_i = 1 / σ_i
        3. Normalize: w_i = inv_vol_i / Σ(inv_vol_j)
        4. Apply min/max constraints
        5. Renormalize to sum to 1.0

        Parameters
        ----------
        date : datetime
            Current rebalance date
        dataset : SectorDataset
            Historical sector returns and factor data

        Returns
        -------
        dict[str, float]
            Sector weights proportional to inverse volatility

        Notes
        -----
        - Sectors with zero volatility (constant returns) are assigned zero weight
        - If insufficient data, sector is excluded
        - Constraints are enforced iteratively with renormalization
        """
        # Get rolling window of returns
        start_date = date - timedelta(days=self.lookback_days)
        historical_returns = dataset.sector_returns.filter(
            (pl.col("timestamp") > start_date) & (pl.col("timestamp") <= date)
        )

        if historical_returns.is_empty():
            self.logger.warning(
                "No historical data in lookback window",
                date=date.isoformat(),
                lookback_days=self.lookback_days,
            )
            return {}

        # Get available sectors
        sectors = sorted(historical_returns["symbol"].unique().to_list())

        if not sectors:
            self.logger.warning("No sectors found", date=date.isoformat())
            return {}

        # Compute volatility per sector
        volatilities: dict[str, float] = {}
        for sector in sectors:
            sector_returns = historical_returns.filter(
                pl.col("symbol") == sector
            )["return"]

            if len(sector_returns) < self.min_observations:
                self.logger.debug(
                    "Insufficient observations for sector",
                    sector=sector,
                    observations=len(sector_returns),
                    min_required=self.min_observations,
                )
                continue

            # Compute volatility (standard deviation)
            # Convert to numpy for reliable std calculation
            returns_array = sector_returns.to_numpy()
            vol = float(np.std(returns_array, ddof=1))

            # Handle zero or near-zero volatility
            if vol < EPSILON:
                self.logger.debug(
                    "Zero volatility for sector",
                    sector=sector,
                    vol=vol,
                )
                continue

            volatilities[sector] = vol

        if not volatilities:
            self.logger.warning(
                "No valid volatilities computed",
                date=date.isoformat(),
            )
            return {}

        # Inverse volatility weights
        inv_vol = {sector: 1.0 / vol for sector, vol in volatilities.items()}
        total_inv_vol = sum(inv_vol.values())

        if total_inv_vol < EPSILON:
            self.logger.warning(
                "Total inverse volatility too small",
                date=date.isoformat(),
            )
            return {}

        # Normalize to sum to 1.0
        weights = {sector: iv / total_inv_vol for sector, iv in inv_vol.items()}

        # Apply constraints and renormalize
        weights = self._apply_constraints(weights)

        self.logger.debug(
            "Computed risk parity weights",
            date=date.isoformat(),
            num_sectors=len(weights),
        )

        return weights

    def _apply_constraints(
        self,
        weights: dict[str, float],
    ) -> dict[str, float]:
        """
        Apply min/max weight constraints iteratively.

        Algorithm:
        1. Clip weights to [min_weight, max_weight]
        2. Renormalize to sum to 1.0
        3. Repeat until no violations (max 100 iterations)

        Parameters
        ----------
        weights : dict[str, float]
            Unconstrained weights

        Returns
        -------
        dict[str, float]
            Constrained weights summing to 1.0
        """
        constrained = weights.copy()
        max_iterations = 100

        for iteration in range(max_iterations):
            violations = False

            # Apply constraints
            for sector in constrained:
                w = constrained[sector]
                if w < self.min_weight:
                    constrained[sector] = self.min_weight
                    violations = True
                elif w > self.max_weight:
                    constrained[sector] = self.max_weight
                    violations = True

            if not violations:
                break

            # Renormalize
            total = sum(constrained.values())
            if total > EPSILON:
                constrained = {s: w / total for s, w in constrained.items()}
            else:
                # Should not happen, but handle gracefully
                self.logger.warning(
                    "Zero total weight after constraint application",
                    iteration=iteration,
                )
                # Equal weight fallback
                equal_weight = 1.0 / len(constrained)
                constrained = dict.fromkeys(constrained, equal_weight)
                break

        return constrained


# ===== Minimum Variance Strategy =====


class MinimumVarianceStrategy:
    """
    Minimum variance portfolio optimization.

    Finds the portfolio with the lowest possible variance:

    minimize: w' Σ w
    subject to: Σw = 1, w ≥ 0

    Where Σ is the covariance matrix estimated from historical returns.

    This strategy typically tilts toward low-volatility, defensive sectors
    and can provide downside protection during market stress.

    Examples
    --------
    >>> strategy = MinimumVarianceStrategy(lookback_days=252)
    >>> weights = strategy.compute_weights(date, dataset)
    >>> print(weights)
    {'XLU': 0.25, 'XLV': 0.22, 'XLP': 0.18, ...}  # Defensive tilt

    Performance Expectations (2010-2024):
    - Annualized Return: ~5-7%
    - Annualized Volatility: ~7% (lowest of all strategies)
    - Sharpe Ratio: ~0.7
    """

    def __init__(
        self,
        lookback_days: int = 252,  # 1 year
        min_weight: float = 0.0,   # Allow zero weights (sector exclusion)
        max_weight: float = 0.30,  # Max 30% per sector (prevent concentration)
        min_observations: int = 60,  # Minimum observations for covariance estimation
        regularization: float = 1e-5,  # Regularization for covariance matrix
    ) -> None:
        """
        Initialize minimum variance strategy.

        Parameters
        ----------
        lookback_days : int, default 252
            Rolling window for covariance estimation (default 252 = 1 year)
        min_weight : float, default 0.0
            Minimum allocation per sector (default 0%)
        max_weight : float, default 0.30
            Maximum allocation per sector (default 30%)
        min_observations : int, default 60
            Minimum observations required for covariance estimation
        regularization : float, default 1e-5
            Regularization parameter for covariance matrix (ε × I)

        Raises
        ------
        ValueError
            If parameters are invalid
        """
        if lookback_days <= 0:
            msg = "Lookback days must be positive"
            raise ValueError(msg)

        if not (0 <= min_weight <= max_weight <= 1):
            msg = "Must have 0 <= min_weight <= max_weight <= 1"
            raise ValueError(msg)

        if min_observations <= 2:
            msg = "Minimum observations must be > 2"
            raise ValueError(msg)

        if regularization < 0:
            msg = "Regularization must be non-negative"
            raise ValueError(msg)

        self.lookback_days = lookback_days
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_observations = min_observations
        self.regularization = regularization
        self.logger = LOGGER.bind(strategy="min_variance")

    def compute_weights(
        self,
        date: datetime,
        dataset: SectorDataset,
    ) -> dict[str, float]:
        """
        Compute minimum variance portfolio weights via quadratic optimization.

        Algorithm:
        1. Extract rolling window of returns (lookback_days)
        2. Compute covariance matrix Σ = cov(returns)
        3. Add regularization: Σ_reg = Σ + ε × I
        4. Solve: min w'Σw s.t. Σw=1, min_weight ≤ w ≤ max_weight
        5. Use scipy.optimize.minimize with SLSQP solver

        Parameters
        ----------
        date : datetime
            Current rebalance date
        dataset : SectorDataset
            Historical sector returns and factor data

        Returns
        -------
        dict[str, float]
            Optimal minimum variance weights

        Notes
        -----
        - If optimization fails, falls back to equal weights
        - Covariance matrix is regularized to ensure positive definiteness
        - Uses Sequential Least Squares Programming (SLSQP) solver
        """
        # Get rolling window of returns
        start_date = date - timedelta(days=self.lookback_days)
        historical_returns = dataset.sector_returns.filter(
            (pl.col("timestamp") > start_date) & (pl.col("timestamp") <= date)
        )

        if historical_returns.is_empty():
            self.logger.warning(
                "No historical data in lookback window",
                date=date.isoformat(),
                lookback_days=self.lookback_days,
            )
            return {}

        # Get available sectors
        sectors = sorted(historical_returns["symbol"].unique().to_list())

        if not sectors:
            self.logger.warning("No sectors found", date=date.isoformat())
            return {}

        # Build return matrix (time × sectors)
        # First, get unique dates
        dates_in_window = sorted(historical_returns["timestamp"].unique().to_list())

        if len(dates_in_window) < self.min_observations:
            self.logger.warning(
                "Insufficient observations for covariance estimation",
                date=date.isoformat(),
                observations=len(dates_in_window),
                min_required=self.min_observations,
            )
            return {}

        # Build return matrix
        return_matrix = np.zeros((len(dates_in_window), len(sectors)))

        for i, sector in enumerate(sectors):
            sector_data = historical_returns.filter(
                pl.col("symbol") == sector
            ).sort("timestamp")

            # Map returns to date indices
            sector_returns_dict = dict(zip(
                sector_data["timestamp"].to_list(),
                sector_data["return"].to_list(),
            ))

            for j, dt in enumerate(dates_in_window):
                return_matrix[j, i] = sector_returns_dict.get(dt, 0.0)

        # Compute covariance matrix
        cov_matrix = np.cov(return_matrix, rowvar=False)

        # Add regularization (ensure positive definiteness)
        cov_matrix += self.regularization * np.eye(len(sectors))

        # Optimize weights
        weights = self._optimize_weights(cov_matrix, sectors)

        self.logger.debug(
            "Computed minimum variance weights",
            date=date.isoformat(),
            num_sectors=len(weights),
        )

        return weights

    def _optimize_weights(
        self,
        cov_matrix: np.ndarray,
        sectors: list[str],
    ) -> dict[str, float]:
        """
        Solve quadratic program for minimum variance weights.

        Uses scipy.optimize.minimize with Sequential Least Squares Programming (SLSQP).

        Objective: f(w) = w' Σ w
        Constraint: Σw = 1
        Bounds: min_weight ≤ w_i ≤ max_weight

        Parameters
        ----------
        cov_matrix : np.ndarray
            Covariance matrix (n_sectors × n_sectors)
        sectors : list[str]
            Sector names (length n_sectors)

        Returns
        -------
        dict[str, float]
            Optimal weights {sector: weight}

        Notes
        -----
        If optimization fails, returns equal weights with a warning.
        """
        n = len(sectors)

        # Objective function: portfolio variance
        def portfolio_variance(w: np.ndarray) -> float:
            return float(w @ cov_matrix @ w)

        # Gradient of portfolio variance: 2 * Σ * w
        def variance_gradient(w: np.ndarray) -> np.ndarray:
            result: np.ndarray = 2.0 * (cov_matrix @ w)
            return result

        # Constraints: sum(w) = 1
        constraints = [
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1.0,
                "jac": lambda w: np.ones(n),
            }
        ]

        # Bounds: min_weight ≤ w_i ≤ max_weight
        bounds = [(self.min_weight, self.max_weight) for _ in range(n)]

        # Initial guess: equal weight
        w0 = np.ones(n) / n

        # Optimize
        result = minimize(
            portfolio_variance,
            w0,
            method="SLSQP",
            jac=variance_gradient,
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        if not result.success:
            self.logger.warning(
                "Optimization failed, falling back to equal weights",
                message=result.message,
            )
            # Equal weight fallback
            equal_weight = 1.0 / n
            return dict.fromkeys(sectors, equal_weight)

        # Extract optimal weights
        optimal_weights = {s: float(w) for s, w in zip(sectors, result.x)}

        # Validate sum to 1.0 (should be guaranteed by optimization)
        total_weight = sum(optimal_weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            self.logger.warning(
                "Optimization weights don't sum to 1.0",
                total=total_weight,
            )
            # Renormalize
            optimal_weights = {s: w / total_weight for s, w in optimal_weights.items()}

        return optimal_weights


__all__ = [
    "MinimumVarianceStrategy",
    "RiskParityStrategy",
    "SixtyFortyStrategy",
]
