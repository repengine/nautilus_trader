"""
Dynamic position sizing for ML trading strategies.

This module provides position sizing implementations including Kelly criterion,
volatility targeting, and composite sizing methods. All implementations follow
the hot/cold path separation pattern with <5ms hot path latency.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import numpy.typing as npt

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram


if TYPE_CHECKING:
    from nautilus_trader.model.objects import Quantity
    from nautilus_trader.model.position import Position

    from ml.actors.base import MLSignal
    from ml.strategies.protocols import AccountLike


logger = logging.getLogger(__name__)

# ===== Metrics =====
sizing_calculations_total = get_counter(
    "ml_sizing_calculations_total",
    "Total position sizing calculations",
    labels=["method"],
)

sizing_latency_seconds = get_histogram(
    "ml_sizing_latency_seconds",
    "Position sizing calculation latency",
    buckets=[0.0001, 0.0005, 0.001, 0.002, 0.005],
    labels=["method"],
)

# ===== Configuration =====
@dataclass(frozen=True)
class SizingConfig:
    """Configuration for position sizing."""

    kelly_fraction: float = 0.25  # Conservative Kelly (1/4)
    target_volatility: float = 0.15  # 15% annual target vol
    max_position_pct: float = 0.15  # Max 15% per position
    min_position_pct: float = 0.01  # Min 1% per position
    confidence_scaling: bool = True  # Scale by signal confidence
    performance_scaling: bool = True  # Scale by recent performance
    lookback_periods: int = 20  # Periods for performance calc


# ===== Kelly Criterion Sizer =====
class KellySizer:
    """
    Kelly criterion position sizing with safety fraction.

    The Kelly criterion optimally sizes positions based on edge and odds.
    We use a fraction (default 25%) for safety and to reduce volatility.
    """

    def __init__(self, config: SizingConfig) -> None:
        """
        Initialize Kelly sizer.

        Parameters
        ----------
        config : SizingConfig
            Sizing configuration.

        """
        self.kelly_fraction: Final[float] = config.kelly_fraction
        self.max_position_pct: Final[float] = config.max_position_pct
        self.min_position_pct: Final[float] = config.min_position_pct

        # Track historical win/loss for Kelly calculation
        self._wins: list[float] = []
        self._losses: list[float] = []
        self._lookback: Final[int] = config.lookback_periods

    def calculate_kelly_pct(self) -> float:
        """
        Calculate Kelly percentage based on historical wins/losses.

        Returns
        -------
        float
            Kelly percentage of capital to risk.

        """
        if len(self._wins) < 5 or len(self._losses) < 5:
            # Not enough data, use minimum
            return self.min_position_pct

        # Calculate win rate and average win/loss
        recent_wins = self._wins[-self._lookback:] if len(self._wins) > self._lookback else self._wins
        recent_losses = self._losses[-self._lookback:] if len(self._losses) > self._lookback else self._losses

        total_trades = len(recent_wins) + len(recent_losses)
        win_rate = len(recent_wins) / total_trades

        avg_win = float(np.mean(recent_wins)) if recent_wins else 0.0
        avg_loss = float(abs(np.mean(recent_losses))) if recent_losses else 1.0

        # Kelly formula: f = (p * b - q) / b
        # where p = win_rate, q = 1 - win_rate, b = avg_win / avg_loss
        if avg_loss == 0:
            return self.min_position_pct

        edge = win_rate * avg_win - (1 - win_rate) * avg_loss

        if edge <= 0:
            # No edge, use minimum
            return self.min_position_pct

        kelly_pct = edge / avg_win if avg_win > 0 else 0.0

        # Apply safety fraction and caps
        safe_kelly = kelly_pct * self.kelly_fraction
        return min(max(safe_kelly, self.min_position_pct), self.max_position_pct)

    def update_performance(self, pnl: float) -> None:
        """
        Update performance history.

        Parameters
        ----------
        pnl : float
            P&L from closed position (positive for wins, negative for losses).

        """
        if pnl > 0:
            self._wins.append(pnl)
        elif pnl < 0:
            self._losses.append(pnl)


# ===== Volatility Sizer =====
class VolatilitySizer:
    """
    Position sizing based on volatility targeting.

    Scales position size inversely with volatility to maintain
    consistent risk exposure: size = target_vol / current_vol * capital
    """

    def __init__(self, config: SizingConfig) -> None:
        """
        Initialize volatility sizer.

        Parameters
        ----------
        config : SizingConfig
            Sizing configuration.

        """
        self.target_vol: Final[float] = config.target_volatility
        self.max_position_pct: Final[float] = config.max_position_pct
        self.min_position_pct: Final[float] = config.min_position_pct
        self.lookback: Final[int] = config.lookback_periods

        # Pre-allocated array for returns (hot path optimization)
        self._returns_buffer: npt.NDArray[np.float32] = np.zeros(self.lookback, dtype=np.float32)
        self._buffer_idx: int = 0
        self._buffer_filled: bool = False

    def update_returns(self, return_pct: float) -> None:
        """
        Update returns buffer for volatility calculation.

        Parameters
        ----------
        return_pct : float
            Latest return percentage.

        """
        self._returns_buffer[self._buffer_idx % self.lookback] = return_pct
        self._buffer_idx += 1
        if self._buffer_idx >= self.lookback:
            self._buffer_filled = True

    def calculate_vol_adjusted_pct(self) -> float:
        """
        Calculate volatility-adjusted position size percentage.

        Returns
        -------
        float
            Position size as percentage of capital.

        """
        if not self._buffer_filled:
            # Not enough data, use conservative size
            return self.min_position_pct * 2

        # Calculate realized volatility (annualized)
        returns_std: float = float(np.std(self._returns_buffer))
        annual_vol: float = returns_std * math.sqrt(252.0)  # Assuming daily returns

        if annual_vol < 0.01:  # Very low vol
            return self.max_position_pct

        # Scale position inversely with vol
        vol_scalar: float = self.target_vol / annual_vol
        base_size = 0.1  # 10% base size

        adjusted_pct: float = base_size * vol_scalar

        return float(min(max(adjusted_pct, self.min_position_pct), self.max_position_pct))


# ===== Composite Sizer (Main Implementation) =====
class CompositeSizer:
    """
    Composite position sizer combining multiple sizing methods.

    This is the main position sizing implementation that combines:
    - Kelly criterion for optimal growth
    - Volatility adjustment for risk consistency
    - Confidence scaling for signal quality
    - Performance scaling for drawdown protection
    """

    def __init__(self, config: SizingConfig | None = None) -> None:
        """
        Initialize composite sizer.

        Parameters
        ----------
        config : SizingConfig, optional
            Sizing configuration.

        """
        self.config = config or SizingConfig()
        self.kelly_sizer = KellySizer(self.config)
        self.vol_sizer = VolatilitySizer(self.config)

        # Performance tracking for scaling
        self._recent_pnl: list[float] = []
        self._max_equity: float = 0.0
        self._current_equity: float = 0.0

    def calculate(
        self,
        signal: MLSignal,
        account: AccountLike,
        current_positions: list[Position],
    ) -> Quantity | None:
        """
        Calculate position size using composite method.

        Parameters
        ----------
        signal : MLSignal
            The ML signal triggering the position.
        account : Account
            Current account state.
        current_positions : list[Position]
            Currently open positions.

        Returns
        -------
        Quantity | None
            Calculated position size, or None if should not trade.

        """
        import time

        start_time = time.perf_counter()
        open_positions = len(current_positions)
        if open_positions > 0:
            logger.debug(
                "Composite sizer processing with existing positions",
                extra={"open_positions": open_positions},
            )

        try:
            # Get account balance
            balance = float(account.balance_total().as_double())
            if balance <= 0:
                logger.warning("Account balance <= 0, cannot size position")
                return None

            # Start with Kelly sizing
            kelly_pct = self.kelly_sizer.calculate_kelly_pct()

            # Adjust for volatility
            vol_pct = self.vol_sizer.calculate_vol_adjusted_pct()

            # Average the two methods
            base_pct = (kelly_pct + vol_pct) / 2

            # Scale by confidence if enabled
            if self.config.confidence_scaling:
                # Higher confidence -> larger size (but capped at 80% confidence)
                confidence_scalar = min(signal.confidence, 0.8)
                base_pct *= confidence_scalar

            # Scale by performance if enabled
            if self.config.performance_scaling:
                perf_scalar = self._get_performance_scalar()
                base_pct *= perf_scalar

            # Apply final limits
            final_pct = min(max(base_pct, self.config.min_position_pct),
                          self.config.max_position_pct)

            # Convert to position value
            position_value = balance * final_pct

            # Get current price for quantity calculation
            # This would come from market data in real implementation
            # For now, return a Quantity based on value
            # The strategy will handle the actual quantity calculation
            from nautilus_trader.model.objects import Quantity

            # Return position value as quantity (strategy will convert based on price)
            return Quantity.from_str(str(position_value))

        finally:
            # Record metrics
            sizing_calculations_total.labels(method="composite").inc()
            sizing_latency_seconds.labels(method="composite").observe(
                time.perf_counter() - start_time
            )

    def _get_performance_scalar(self) -> float:
        """
        Calculate performance scalar for position sizing.

        Reduces size during drawdowns, increases during winning streaks.

        Returns
        -------
        float
            Scalar between 0.5 and 1.0.

        """
        if not self._recent_pnl:
            return 1.0

        # Calculate recent performance
        recent_pnl = self._recent_pnl[-10:] if len(self._recent_pnl) > 10 else self._recent_pnl
        win_rate = sum(1 for pnl in recent_pnl if pnl > 0) / len(recent_pnl)

        # Calculate drawdown
        if self._current_equity > self._max_equity:
            self._max_equity = self._current_equity

        drawdown_pct = 0.0
        if self._max_equity > 0:
            drawdown_pct = (self._max_equity - self._current_equity) / self._max_equity

        # Scalar based on win rate and drawdown
        win_scalar = 0.5 + (win_rate * 0.5)  # 0.5 to 1.0 based on win rate
        dd_scalar = 1.0 - (drawdown_pct * 0.5)  # Reduce by up to 50% in drawdown

        return min(win_scalar, dd_scalar)

    def update_performance(self, pnl: float) -> None:
        """
        Update performance tracking.

        Parameters
        ----------
        pnl : float
            P&L from closed position.

        """
        self._recent_pnl.append(pnl)
        self._current_equity += pnl

        # Update component sizers
        self.kelly_sizer.update_performance(pnl)

    def update_market_data(self, return_pct: float) -> None:
        """
        Update market data for volatility calculation.

        Parameters
        ----------
        return_pct : float
            Latest return percentage.

        """
        self.vol_sizer.update_returns(return_pct)


# ===== Public API =====
__all__ = [
    "CompositeSizer",
    "KellySizer",
    "SizingConfig",
    "VolatilitySizer",
]
