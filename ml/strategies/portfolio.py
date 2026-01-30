"""
Portfolio management for ML trading strategies.

This module provides portfolio-level coordination including capital allocation,
correlation tracking, and multi-instrument position management. Designed for small
accounts with focus on diversification and risk-adjusted allocation.

"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import numpy.typing as npt

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_gauge
from ml.common.metrics_bootstrap import get_histogram
from ml.strategies.common.correlation import CorrelationSnapshot


if TYPE_CHECKING:
    from nautilus_trader.model.identifiers import InstrumentId

    from ml.actors.base import MLSignal


logger = logging.getLogger(__name__)

# ===== Metrics =====
allocation_calculations_total = get_counter(
    "ml_allocation_calculations_total",
    "Total portfolio allocation calculations",
    labels=["method"],
)

allocation_latency_seconds = get_histogram(
    "ml_allocation_latency_seconds",
    "Portfolio allocation calculation latency",
    buckets=[0.0001, 0.0005, 0.001, 0.002, 0.005],
    labels=["method"],
)

portfolio_concentration_gauge = get_gauge(
    "ml_portfolio_concentration",
    "Portfolio concentration (HHI)",
    labels=[],
)

active_positions_gauge = get_gauge(
    "ml_active_positions_count",
    "Number of active positions",
    labels=[],
)


# ===== Configuration =====
@dataclass(frozen=True)
class PortfolioConfig:
    """
    Configuration for portfolio management.
    """

    # Allocation limits
    max_positions: int = 10  # Max concurrent positions
    min_position_weight: float = 0.05  # Min 5% per position
    max_position_weight: float = 0.15  # Max 15% per position
    max_correlated_weight: float = 0.40  # Max 40% in correlated assets

    # Allocation method
    allocation_method: str = "risk_parity"  # Options: equal, risk_parity, kelly
    use_correlation_adjustment: bool = True  # Adjust for correlations
    rebalance_threshold: float = 0.10  # 10% deviation triggers rebalance

    # Correlation parameters
    correlation_lookback: int = 60  # Days for correlation calc
    correlation_threshold: float = 0.6  # Above this = correlated
    correlation_decay: float = 0.94  # Exponential decay for correlation

    # Performance tracking
    track_attribution: bool = True  # Track per-instrument P&L
    attribution_window: int = 30  # Days to track attribution
    annualization_factor: float | None = None  # Bars per year for volatility scaling


# ===== Portfolio Manager =====
class PortfolioManager:
    """
    Portfolio-level management for multi-instrument trading.

    Handles capital allocation, correlation tracking, rebalancing, and performance
    attribution across instruments.

    """

    def __init__(self, config: PortfolioConfig | None = None) -> None:
        """
        Initialize portfolio manager.

        Parameters
        ----------
        config : PortfolioConfig, optional
            Portfolio management configuration.

        """
        self.config = config or PortfolioConfig()

        # Allocation tracking
        self._current_weights: dict[InstrumentId, float] = {}
        self._target_weights: dict[InstrumentId, float] = {}
        self._last_rebalance_time: float = 0.0

        # Correlation matrix (pre-allocated for efficiency)
        self._max_instruments: Final[int] = 50
        self._correlation_matrix: npt.NDArray[np.float32] = np.eye(
            self._max_instruments,
            dtype=np.float32,
        )
        self._instrument_indices: dict[InstrumentId, int] = {}
        self._next_index: int = 0

        # Returns buffer for correlation calculation
        self._returns_buffer: dict[InstrumentId, npt.NDArray[np.float32]] = {}
        self._returns_index: dict[InstrumentId, int] = {}
        self._returns_count: dict[InstrumentId, int] = {}
        self._buffer_size: Final[int] = self.config.correlation_lookback
        self._correlation_last_update_ns: int | None = None
        self._annualization_factor: float | None = self.config.annualization_factor

        # Performance attribution
        self._instrument_pnl: dict[InstrumentId, float] = {}
        self._instrument_sharpe: dict[InstrumentId, float] = {}

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
        start_time = time.perf_counter()

        try:
            if not signals or available_capital <= 0:
                return {}

            # Filter signals by confidence and limit count
            viable_signals = self._filter_signals(signals)
            if not viable_signals:
                return {}

            # Get allocation method
            if self.config.allocation_method == "equal":
                allocations = self._allocate_equal(viable_signals, available_capital)
            elif self.config.allocation_method == "risk_parity":
                allocations = self._allocate_risk_parity(viable_signals, available_capital)
            elif self.config.allocation_method == "kelly":
                allocations = self._allocate_kelly(viable_signals, available_capital)
            else:
                allocations = self._allocate_equal(viable_signals, available_capital)

            # Apply correlation adjustments if enabled
            if self.config.use_correlation_adjustment:
                allocations = self._adjust_for_correlation(allocations)

            # Apply position limits
            allocations = self._apply_limits(allocations, available_capital)

            # Update tracking
            self._target_weights = {
                inst: alloc / available_capital for inst, alloc in allocations.items()
            }

            # Update metrics
            try:
                active_positions_gauge.set(len(allocations))
            except Exception:
                logger.debug(
                    "portfolio.active_positions_metric_failed",
                    exc_info=True,
                )
            self._update_concentration_metric()

            return allocations

        finally:
            allocation_calculations_total.labels(method=self.config.allocation_method).inc()
            allocation_latency_seconds.labels(method=self.config.allocation_method).observe(
                time.perf_counter() - start_time,
            )

    def _filter_signals(self, signals: list[MLSignal]) -> list[MLSignal]:
        """
        Filter signals by quality and count limits.

        Parameters
        ----------
        signals : list[MLSignal]
            Raw signals to filter.

        Returns
        -------
        list[MLSignal]
            Filtered signals ready for allocation.

        """
        # Sort by confidence
        sorted_signals = sorted(signals, key=lambda s: s.confidence, reverse=True)

        # Take top N based on max positions
        viable = sorted_signals[: self.config.max_positions]

        # Filter by minimum confidence (0.5)
        viable = [s for s in viable if s.confidence >= 0.5]

        return viable

    def _allocate_equal(
        self,
        signals: list[MLSignal],
        capital: float,
    ) -> dict[InstrumentId, float]:
        """
        Equal weight allocation across signals.

        Parameters
        ----------
        signals : list[MLSignal]
            Signals to allocate to.
        capital : float
            Available capital.

        Returns
        -------
        dict[InstrumentId, float]
            Equal allocations.

        """
        if not signals:
            return {}

        allocation_per_signal = capital / len(signals)
        return {signal.instrument_id: allocation_per_signal for signal in signals}

    def _allocate_risk_parity(
        self,
        signals: list[MLSignal],
        capital: float,
    ) -> dict[InstrumentId, float]:
        """
        Risk parity allocation (inverse volatility weighting).

        Parameters
        ----------
        signals : list[MLSignal]
            Signals to allocate to.
        capital : float
            Available capital.

        Returns
        -------
        dict[InstrumentId, float]
            Risk parity allocations.

        """
        allocations: dict[InstrumentId, float] = {}

        # Get volatilities for each instrument
        volatilities: dict[InstrumentId, float] = {}
        for signal in signals:
            vol = self._get_instrument_volatility(signal.instrument_id)
            if vol > 0:
                volatilities[signal.instrument_id] = vol

        if not volatilities:
            # Fall back to equal weight
            return self._allocate_equal(signals, capital)

        # Calculate inverse volatility weights
        inv_vols = {inst: 1.0 / vol for inst, vol in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())

        # Scale by confidence
        for signal in signals:
            if signal.instrument_id in inv_vols:
                weight = inv_vols[signal.instrument_id] / total_inv_vol
                # Scale by confidence (50-80% confidence maps to 50-100% of weight)
                confidence_scalar = min(max((signal.confidence - 0.5) * 2, 0.5), 1.0)
                allocations[signal.instrument_id] = capital * weight * confidence_scalar

        # Renormalize to use full capital
        total_alloc = sum(allocations.values())
        if total_alloc > 0:
            scale = capital / total_alloc
            allocations = {inst: alloc * scale for inst, alloc in allocations.items()}

        return allocations

    def _allocate_kelly(
        self,
        signals: list[MLSignal],
        capital: float,
    ) -> dict[InstrumentId, float]:
        """
        Kelly criterion-based allocation.

        Parameters
        ----------
        signals : list[MLSignal]
            Signals to allocate to.
        capital : float
            Available capital.

        Returns
        -------
        dict[InstrumentId, float]
            Kelly allocations.

        """
        allocations: dict[InstrumentId, float] = {}

        for signal in signals:
            # Get historical performance for Kelly calculation
            sharpe = self._instrument_sharpe.get(signal.instrument_id, 0.0)

            if sharpe <= 0:
                # No edge, skip
                continue

            # Simplified Kelly: f = sharpe / 2 (assuming Sharpe approximates edge/odds)
            # Apply safety fraction of 0.25
            kelly_fraction = min(sharpe / 2 * 0.25, 0.15)  # Cap at 15%

            # Scale by confidence
            kelly_fraction *= signal.confidence

            allocations[signal.instrument_id] = capital * kelly_fraction

        if not allocations:
            # Fall back to equal weight
            return self._allocate_equal(signals, capital)

        return allocations

    def _adjust_for_correlation(
        self,
        allocations: dict[InstrumentId, float],
    ) -> dict[InstrumentId, float]:
        """
        Adjust allocations for correlations between instruments.

        Parameters
        ----------
        allocations : dict[InstrumentId, float]
            Initial allocations.

        Returns
        -------
        dict[InstrumentId, float]
            Correlation-adjusted allocations.

        """
        if len(allocations) < 2:
            return allocations

        # Group correlated instruments
        correlation_groups = self._get_correlation_groups(list(allocations.keys()))

        # Reduce allocation to correlated groups
        adjusted: dict[InstrumentId, float] = {}
        for group in correlation_groups:
            group_alloc = sum(allocations.get(inst, 0) for inst in group)

            if len(group) > 1:
                # Reduce total allocation to correlated group
                max_group_alloc = sum(allocations.values()) * self.config.max_correlated_weight
                if group_alloc > max_group_alloc:
                    scale = max_group_alloc / group_alloc
                    for inst in group:
                        adjusted[inst] = allocations[inst] * scale
                else:
                    for inst in group:
                        adjusted[inst] = allocations[inst]
            else:
                # Single instrument, no adjustment
                adjusted[group[0]] = allocations[group[0]]

        return adjusted

    def _get_correlation_groups(
        self,
        instruments: list[InstrumentId],
    ) -> list[list[InstrumentId]]:
        """
        Group instruments by correlation.

        Parameters
        ----------
        instruments : list[InstrumentId]
            Instruments to group.

        Returns
        -------
        list[list[InstrumentId]]
            Groups of correlated instruments.

        """
        if len(instruments) < 2:
            return [[inst] for inst in instruments]

        # Build adjacency list from correlation matrix
        groups: list[list[InstrumentId]] = []
        visited: set[InstrumentId] = set()

        for inst in instruments:
            if inst in visited:
                continue

            group = [inst]
            visited.add(inst)

            # Find all correlated instruments
            for other in instruments:
                if other in visited:
                    continue

                corr = self.get_correlation(inst, other)
                if corr > self.config.correlation_threshold:
                    group.append(other)
                    visited.add(other)

            groups.append(group)

        return groups

    def _apply_limits(
        self,
        allocations: dict[InstrumentId, float],
        capital: float,
    ) -> dict[InstrumentId, float]:
        """
        Apply position size limits.

        Parameters
        ----------
        allocations : dict[InstrumentId, float]
            Raw allocations.
        capital : float
            Available capital.

        Returns
        -------
        dict[InstrumentId, float]
            Limited allocations.

        """
        limited: dict[InstrumentId, float] = {}

        min_size = capital * self.config.min_position_weight
        max_size = capital * self.config.max_position_weight

        for inst, alloc in allocations.items():
            if alloc < min_size:
                # Too small, skip
                continue

            limited[inst] = min(alloc, max_size)

        return limited

    def get_correlation(
        self,
        inst1: InstrumentId,
        inst2: InstrumentId,
    ) -> float:
        """
        Get correlation between two instruments.

        Parameters
        ----------
        inst1 : InstrumentId
            First instrument.
        inst2 : InstrumentId
            Second instrument.

        Returns
        -------
        float
            Correlation coefficient [-1, 1].

        """
        if inst1 == inst2:
            return 1.0

        # Get indices
        idx1 = self._get_instrument_index(inst1)
        idx2 = self._get_instrument_index(inst2)

        return float(self._correlation_matrix[idx1, idx2])

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
        n = len(instruments)
        matrix = np.zeros((n, n), dtype=np.float64)

        for i, inst1 in enumerate(instruments):
            for j, inst2 in enumerate(instruments):
                matrix[i, j] = self.get_correlation(inst1, inst2)

        return matrix

    def update_correlation(
        self,
        inst1: InstrumentId,
        inst2: InstrumentId,
        returns1: npt.NDArray[np.float32],
        returns2: npt.NDArray[np.float32],
        *,
        ts_event: int | None = None,
    ) -> None:
        """
        Update correlation between two instruments.

        Parameters
        ----------
        inst1 : InstrumentId
            First instrument.
        inst2 : InstrumentId
            Second instrument.
        returns1 : np.ndarray
            Returns for first instrument.
        returns2 : np.ndarray
            Returns for second instrument.
        ts_event : int | None, optional
            Event timestamp (nanoseconds) for freshness tracking.

        """
        if len(returns1) != len(returns2) or len(returns1) < 2:
            return

        # Calculate correlation
        corr = np.corrcoef(returns1, returns2)[0, 1]

        if np.isnan(corr):
            return

        # Get indices
        idx1 = self._get_instrument_index(inst1)
        idx2 = self._get_instrument_index(inst2)

        # Update with exponential decay
        old_corr = float(self._correlation_matrix[idx1, idx2])
        if abs(old_corr) < 1e-6:
            new_corr = corr
        else:
            new_corr = (
                self.config.correlation_decay * old_corr
                + (1 - self.config.correlation_decay) * corr
            )

        # Update symmetric matrix
        self._correlation_matrix[idx1, idx2] = new_corr
        self._correlation_matrix[idx2, idx1] = new_corr
        self._correlation_last_update_ns = ts_event or time.time_ns()

    def get_correlation_snapshot(
        self,
        inst1: InstrumentId,
        inst2: InstrumentId,
    ) -> CorrelationSnapshot:
        """
        Return correlation snapshot with last update timestamp.

        Parameters
        ----------
        inst1 : InstrumentId
            First instrument.
        inst2 : InstrumentId
            Second instrument.

        Returns
        -------
        CorrelationSnapshot
            Snapshot including correlation value and timestamp.

        """
        return CorrelationSnapshot(
            value=self.get_correlation(inst1, inst2),
            ts_event=self._correlation_last_update_ns,
            source="portfolio_manager",
        )

    def _get_instrument_index(self, instrument: InstrumentId) -> int:
        """
        Get or assign index for instrument in correlation matrix.

        Parameters
        ----------
        instrument : InstrumentId
            Instrument to index.

        Returns
        -------
        int
            Index in correlation matrix.

        """
        if instrument not in self._instrument_indices:
            if self._next_index >= self._max_instruments:
                # Recycle oldest index (simple FIFO)
                self._next_index = 0

            self._instrument_indices[instrument] = self._next_index
            self._next_index += 1

        return self._instrument_indices[instrument]

    def _get_instrument_volatility(self, instrument: InstrumentId) -> float:
        """
        Get estimated volatility for instrument.

        Parameters
        ----------
        instrument : InstrumentId
            Instrument to check.

        Returns
        -------
        float
            Annualized volatility (0 if unknown).

        """
        if instrument not in self._returns_buffer:
            return 0.15  # Default 15% volatility

        returns = self._returns_buffer[instrument]
        count = self._returns_count.get(instrument, 0)
        if count < 2:
            return 0.15

        # Calculate standard deviation
        if count < self._buffer_size:
            returns = returns[:count]
        std = np.std(returns)
        annualization_factor = self._annualization_factor or 1.0
        if annualization_factor <= 0:
            annualization_factor = 1.0
        annual_vol = std * np.sqrt(annualization_factor)

        return max(float(annual_vol), 0.01)  # Min 1% vol

    def update_returns(
        self,
        instrument: InstrumentId,
        return_pct: float,
    ) -> None:
        """
        Update returns history for an instrument.

        Parameters
        ----------
        instrument : InstrumentId
            Instrument to update.
        return_pct : float
            Latest return percentage.

        """
        if instrument not in self._returns_buffer:
            self._returns_buffer[instrument] = np.zeros(self._buffer_size, dtype=np.float32)
            self._returns_index[instrument] = 0
            self._returns_count[instrument] = 0

        idx = self._returns_index[instrument] % self._buffer_size
        self._returns_buffer[instrument][idx] = return_pct
        self._returns_index[instrument] += 1
        self._returns_count[instrument] = min(
            self._returns_count[instrument] + 1,
            self._buffer_size,
        )

    def update_performance(
        self,
        instrument: InstrumentId,
        pnl: float,
    ) -> None:
        """
        Update performance tracking for an instrument.

        Parameters
        ----------
        instrument : InstrumentId
            Instrument to update.
        pnl : float
            P&L from closed position.

        """
        # Update cumulative P&L
        if instrument not in self._instrument_pnl:
            self._instrument_pnl[instrument] = 0.0
        self._instrument_pnl[instrument] += pnl

        # Update Sharpe ratio estimate
        if instrument in self._returns_buffer:
            returns = self._returns_buffer[instrument]
            count = self._returns_count.get(instrument, 0)
            if count > 10:
                # Simple Sharpe calculation
                if count < self._buffer_size:
                    returns = returns[:count]
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                if std_return > 0:
                    annualization_factor = self._annualization_factor or 1.0
                    if annualization_factor <= 0:
                        annualization_factor = 1.0
                    sharpe = mean_return / std_return * np.sqrt(annualization_factor)
                    self._instrument_sharpe[instrument] = float(sharpe)

    def set_annualization_factor(self, factor: float) -> None:
        """
        Set annualization factor for volatility calculations.

        Parameters
        ----------
        factor : float
            Bars-per-year annualization factor.

        """
        if factor <= 0:
            return
        self._annualization_factor = float(factor)

    def should_rebalance(self) -> bool:
        """
        Check if portfolio should be rebalanced.

        Returns
        -------
        bool
            True if rebalancing is needed.

        """
        # Time-based check (minimum 1 hour between rebalances)
        current_time = time.time()
        if current_time - self._last_rebalance_time < 3600:
            return False

        # Deviation check
        for inst, target_weight in self._target_weights.items():
            current_weight = self._current_weights.get(inst, 0.0)
            deviation = abs(current_weight - target_weight)

            if deviation > self.config.rebalance_threshold:
                logger.info(
                    f"Rebalance triggered: {inst} deviation {deviation:.1%} > {self.config.rebalance_threshold:.1%}",
                )
                return True

        return False

    def _update_concentration_metric(self) -> None:
        """
        Update portfolio concentration metric (HHI).
        """
        if not self._target_weights:
            return

        # Calculate Herfindahl-Hirschman Index
        hhi = sum(w**2 for w in self._target_weights.values())
        try:
            portfolio_concentration_gauge.set(hhi)
        except Exception:
            logger.debug(
                "portfolio.concentration_metric_failed",
                exc_info=True,
            )

    def get_portfolio_metrics(self) -> dict[str, float]:
        """
        Get current portfolio metrics.

        Returns
        -------
        dict[str, float]
            Portfolio metrics.

        """
        total_pnl = sum(self._instrument_pnl.values())
        n_positions = len(self._current_weights)
        avg_sharpe = (
            float(np.mean(list(self._instrument_sharpe.values())))
            if self._instrument_sharpe
            else 0.0
        )

        # Calculate concentration (HHI)
        hhi = sum(w**2 for w in self._current_weights.values()) if self._current_weights else 0.0

        return {
            "total_pnl": total_pnl,
            "n_positions": float(n_positions),
            "avg_sharpe": float(avg_sharpe),
            "concentration_hhi": float(hhi),
            "max_correlation": float(self._get_max_correlation()),
        }

    def _get_max_correlation(self) -> float:
        """
        Get maximum correlation among active positions.
        """
        if len(self._current_weights) < 2:
            return 0.0

        instruments = list(self._current_weights.keys())
        max_corr = 0.0

        for i, inst1 in enumerate(instruments):
            for inst2 in instruments[i + 1 :]:
                corr = abs(self.get_correlation(inst1, inst2))
                max_corr = max(max_corr, corr)

        return max_corr


# ===== Public API =====
__all__ = [
    "PortfolioConfig",
    "PortfolioManager",
]
