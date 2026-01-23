"""
Type protocols for ML strategy components.

This module defines the protocol interfaces for position sizing, risk management,
and other strategy components following the Protocol-First Interface Design pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


if TYPE_CHECKING:
    from collections.abc import Mapping

    from nautilus_trader.model.identifiers import ClientOrderId
    from nautilus_trader.model.identifiers import InstrumentId
    from nautilus_trader.model.identifiers import StrategyId
    from nautilus_trader.model.identifiers import TraderId
    from nautilus_trader.model.objects import Quantity
    from nautilus_trader.model.position import Position

    from ml.actors.base import MLSignal
    from nautilus_trader.core.uuid import UUID4
    from nautilus_trader.model.enums import OrderSide
    from nautilus_trader.model.instruments import Instrument
    from nautilus_trader.model.orders import Order
    from nautilus_trader.portfolio import Portfolio


class _HasAsDouble(Protocol):
    def as_double(self) -> float: ...


@runtime_checkable
class AccountLike(Protocol):
    """Minimal account surface required by sizing logic."""

    def balance_total(self) -> _HasAsDouble: ...


@runtime_checkable
class PositionSizerProtocol(Protocol):
    """Protocol for position sizing implementations."""

    def calculate(
        self,
        signal: MLSignal,
        account: AccountLike,
        current_positions: list[Position],
    ) -> Quantity | None:
        """
        Calculate position size based on signal and account state.

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
        ...


@runtime_checkable
class RiskManagerProtocol(Protocol):
    """Protocol for risk management implementations."""

    def check_position(
        self,
        proposed_size: Quantity | None,
        instrument: InstrumentId,
        portfolio: Portfolio,
    ) -> Quantity | None:
        """
        Check and potentially adjust proposed position size based on risk limits.

        Parameters
        ----------
        proposed_size : Quantity | None
            The proposed position size.
        instrument : InstrumentId
            The instrument to trade.
        portfolio : Portfolio
            Current portfolio state.

        Returns
        -------
        Quantity | None
            Approved position size (may be reduced), or None if rejected.

        """
        ...

    def check_daily_limits(self, ts_event: int | None = None) -> bool:
        """
        Check if daily risk limits have been exceeded.

        Parameters
        ----------
        ts_event : int | None, optional
            Event timestamp (nanoseconds) used for daily reset alignment.

        Returns
        -------
        bool
            True if trading can continue, False if limits exceeded.

        """
        ...

    def update_daily_pnl(self, pnl: float, ts_event: int | None = None) -> None:
        """
        Update daily P&L for risk tracking.

        Parameters
        ----------
        pnl : float
            P&L to add to daily total.
        ts_event : int | None, optional
            Event timestamp (nanoseconds) used for daily reset alignment.

        """
        ...


@runtime_checkable
class OrderExecutorProtocol(Protocol):
    """Protocol for order execution implementations."""

    def create_order(
        self,
        side: OrderSide,
        quantity: Quantity,
        signal: MLSignal,
        market_state: dict[str, float],
        instrument: Instrument,
        *,
        trader_id: TraderId | None = None,
        strategy_id: StrategyId | None = None,
        client_order_id: ClientOrderId | None = None,
        init_id: UUID4 | None = None,
        ts_init: int | None = None,
    ) -> Order | None:
        """
        Create an order based on signal confidence and market conditions.

        Parameters
        ----------
        side : OrderSide
            Order side (BUY or SELL).
        quantity : Quantity
            Position size.
        signal : MLSignal
            The ML signal with confidence.
        market_state : dict[str, float]
            Current market state (bid, ask, spread, etc).

        Returns
        -------
        Order | None
            The order to submit, or None if conditions not met.

        """
        ...


@runtime_checkable
class PortfolioManagerProtocol(Protocol):
    """Protocol for portfolio management implementations."""

    def allocate_signals(
        self,
        signals: list[MLSignal],
        available_capital: float,
    ) -> dict[InstrumentId, float]:
        """
        Allocate capital across multiple signals.

        Parameters
        ----------
        signals : list[MLSignal]
            List of ML signals to consider.
        available_capital : float
            Total capital available for allocation.

        Returns
        -------
        dict[InstrumentId, float]
            Capital allocation per instrument.

        """
        ...

    def get_correlation_matrix(
        self,
        instruments: list[InstrumentId],
    ) -> npt.NDArray[np.float64]:
        """
        Get correlation matrix for specified instruments.

        Parameters
        ----------
        instruments : list[InstrumentId]
            List of instruments.

        Returns
        -------
        np.ndarray
            Correlation matrix.

        """
        ...


@runtime_checkable
class PerformanceTrackerProtocol(Protocol):
    """Protocol for performance tracking implementations."""

    def record_signal(self, signal: MLSignal) -> None:
        """
        Record an ML signal for analysis.

        Parameters
        ----------
        signal : MLSignal
            The signal to record.

        """
        ...

    def record_order(self, order: Order, signal: MLSignal) -> None:
        """
        Record an order placement.

        Parameters
        ----------
        order : Order
            The order placed.
        signal : MLSignal
            The signal that triggered the order.

        """
        ...

    def get_win_rate_by_confidence(
        self,
    ) -> Mapping[str, float]:
        """
        Get win rates grouped by confidence bands.

        Returns
        -------
        Mapping[str, float]
            Win rates by confidence band.

        """
        ...

    def get_sharpe_ratio(self, lookback_days: int = 30) -> float:
        """
        Calculate Sharpe ratio over lookback period.

        Parameters
        ----------
        lookback_days : int
            Number of days to look back.

        Returns
        -------
        float
            Sharpe ratio.

        """
        ...
