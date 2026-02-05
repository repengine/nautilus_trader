"""
Performance analytics for ML trading strategies.

This module provides comprehensive tracking and analysis of signal quality,
execution performance, and strategy metrics. Designed to identify edge decay
and optimize signal-to-trade conversion.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ml.common import decision_from_probability
from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_gauge
from ml.common.metrics_bootstrap import get_histogram


if TYPE_CHECKING:
    from collections.abc import Mapping

    from nautilus_trader.model.identifiers import InstrumentId
    from nautilus_trader.model.position import Position

    from ml.actors.base import MLSignal
    from nautilus_trader.model.orders import Order


logger = logging.getLogger(__name__)

# ===== Metrics =====
signals_recorded_total = get_counter(
    "ml_signals_recorded_total",
    "Total signals recorded for analysis",
    labels=["instrument", "direction"],
)

signal_accuracy_gauge = get_gauge(
    "ml_signal_accuracy",
    "Signal accuracy by confidence band",
    labels=["confidence_band"],
)

sharpe_ratio_gauge = get_gauge(
    "ml_sharpe_ratio",
    "Rolling Sharpe ratio",
    labels=["period"],
)

signal_to_trade_latency_seconds = get_histogram(
    "ml_signal_to_trade_latency_seconds",
    "Latency from signal to trade execution",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
    labels=["instrument"],
)

# ===== Configuration =====
@dataclass(frozen=True)
class AnalyticsConfig:
    """Configuration for performance analytics."""

    # Tracking windows
    signal_lookback: int = 1000  # Signals to keep for analysis
    performance_window_days: int = 30  # Days for rolling metrics
    confidence_bands: list[float] = field(default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.9])  # Confidence bands

    # Analysis parameters
    min_signals_for_stats: int = 30  # Min signals for statistical validity
    decay_detection_threshold: float = 0.15  # 15% performance drop = decay
    outlier_z_score: float = 3.0  # Z-score for outlier detection

    # Reporting
    report_frequency_minutes: int = 60  # How often to generate reports
    track_execution_quality: bool = True  # Track slippage and fees
    track_signal_decay: bool = True  # Monitor for edge decay

    def __post_init__(self) -> None:
        # Nothing required; defaults provided via default_factory
        return None


# ===== Signal Record =====
@dataclass
class SignalRecord:
    """Record of an ML signal for analysis."""

    signal: MLSignal
    timestamp: datetime
    executed: bool = False
    order: Order | None = None
    execution_latency_ms: float = 0.0
    pnl: float | None = None
    closed_timestamp: datetime | None = None


# ===== Performance Tracker =====
class PerformanceTracker:
    """
    Comprehensive performance tracking and analytics.

    Tracks signal quality, execution performance, and identifies
    patterns in strategy behavior for optimization.
    """

    def __init__(self, config: AnalyticsConfig | None = None) -> None:
        """
        Initialize performance tracker.

        Parameters
        ----------
        config : AnalyticsConfig, optional
            Analytics configuration.

        """
        self.config = config or AnalyticsConfig()

        # Signal tracking
        self._signals: dict[InstrumentId, list[SignalRecord]] = defaultdict(list)
        self._signal_count: int = 0

        # Performance tracking
        self._daily_returns: list[float] = []
        self._cumulative_pnl: float = 0.0
        self._peak_pnl: float = 0.0

        # Execution quality
        self._total_slippage: float = 0.0
        self._total_fees: float = 0.0
        self._execution_latencies: list[float] = []

        # Win rate by confidence
        self._wins_by_confidence: dict[str, int] = defaultdict(int)
        self._total_by_confidence: dict[str, int] = defaultdict(int)

        # Last report time
        self._last_report_time: datetime = datetime.now()

    def record_signal(self, signal: MLSignal) -> None:
        """
        Record an ML signal for analysis.

        Parameters
        ----------
        signal : MLSignal
            The signal to record.

        """
        record = SignalRecord(
            signal=signal,
            timestamp=datetime.now(),
        )

        # Store by instrument
        signals = self._signals[signal.instrument_id]
        signals.append(record)

        # Trim to lookback limit
        if len(signals) > self.config.signal_lookback:
            signals.pop(0)

        self._signal_count += 1

        # Update metrics
        neutral_band = 0.0
        if hasattr(signal, "metadata") and isinstance(signal.metadata, dict):
            nb_value = signal.metadata.get("neutral_band")
            if isinstance(nb_value, (int, float)):
                neutral_band = float(nb_value)
        decision = decision_from_probability(
            float(signal.prediction),
            neutral_band=neutral_band,
        )
        if decision == "BUY":
            direction = "long"
        elif decision == "SELL":
            direction = "short"
        else:
            direction = "neutral"
        signals_recorded_total.labels(
            instrument=str(signal.instrument_id),
            direction=direction,
        ).inc()

        # Check if report needed
        self._check_report_schedule()

    def record_order(
        self,
        order: Order,
        signal: MLSignal,
    ) -> None:
        """
        Record an order placement.

        Parameters
        ----------
        order : Order
            The order placed.
        signal : MLSignal
            The signal that triggered the order.

        """
        # Find matching signal record
        signals = self._signals[signal.instrument_id]
        for record in reversed(signals):
            if record.signal == signal and not record.executed:
                record.executed = True
                record.order = order

                # Calculate execution latency
                latency = (datetime.now() - record.timestamp).total_seconds()
                record.execution_latency_ms = latency * 1000

                # Update latency metrics
                self._execution_latencies.append(record.execution_latency_ms)
                signal_to_trade_latency_seconds.labels(
                    instrument=str(signal.instrument_id)
                ).observe(latency)

                break

    def record_position_closed(
        self,
        position: Position,
        signal: MLSignal,
        pnl: float,
        fees: float = 0.0,
        slippage: float = 0.0,
    ) -> None:
        """
        Record a closed position.

        Parameters
        ----------
        position : Position
            The closed position.
        signal : MLSignal
            The signal that opened the position.
        pnl : float
            Realized P&L.
        fees : float, optional
            Trading fees paid.
        slippage : float, optional
            Execution slippage.

        """
        # Update signal record
        signals = self._signals[signal.instrument_id]
        for record in reversed(signals):
            if record.signal == signal:
                record.pnl = pnl
                record.closed_timestamp = datetime.now()
                break

        # Update performance tracking
        self._cumulative_pnl += pnl
        self._daily_returns.append(pnl)  # Simplified - should aggregate by day
        self._total_fees += fees
        self._total_slippage += slippage

        # Update win rate by confidence
        confidence_band = self._get_confidence_band(signal.confidence)
        self._total_by_confidence[confidence_band] += 1
        if pnl > 0:
            self._wins_by_confidence[confidence_band] += 1

        # Update peak for drawdown calculation
        if self._cumulative_pnl > self._peak_pnl:
            self._peak_pnl = self._cumulative_pnl

        # Update accuracy metric
        self._update_accuracy_metrics()

    def get_win_rate_by_confidence(self) -> Mapping[str, float]:
        """
        Get win rates grouped by confidence bands.

        Returns
        -------
        Mapping[str, float]
            Win rates by confidence band.

        """
        win_rates: dict[str, float] = {}

        for band in self._total_by_confidence:
            total = self._total_by_confidence[band]
            if total >= self.config.min_signals_for_stats:
                wins = self._wins_by_confidence[band]
                win_rates[band] = wins / total

        return win_rates

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
        if len(self._daily_returns) < 2:
            return 0.0

        # Get returns for period (simplified - should be by date)
        returns = self._daily_returns[-lookback_days:]
        if len(returns) < 2:
            return 0.0

        # Calculate Sharpe
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        # Annualize
        sharpe = mean_return / std_return * float(np.sqrt(252))

        # Update metric
        sharpe_ratio_gauge.labels(period=f"{lookback_days}d").set(float(sharpe))

        return float(sharpe)

    def get_signal_quality_metrics(
        self,
        instrument: InstrumentId | None = None,
    ) -> dict[str, float]:
        """
        Get signal quality metrics.

        Parameters
        ----------
        instrument : InstrumentId, optional
            Specific instrument, or None for all.

        Returns
        -------
        dict[str, float]
            Signal quality metrics.

        """
        # Collect relevant signals
        if instrument:
            signals = self._signals[instrument]
        else:
            signals = []
            for inst_signals in self._signals.values():
                signals.extend(inst_signals)

        if len(signals) < self.config.min_signals_for_stats:
            return {}

        # Calculate metrics
        executed = [s for s in signals if s.executed]
        profitable = [s for s in signals if s.pnl is not None and s.pnl > 0]

        execution_rate = len(executed) / len(signals) if signals else 0.0
        win_rate = len(profitable) / len(executed) if executed else 0.0

        # Average confidence of winners vs losers
        win_confidence = float(np.mean([s.signal.confidence for s in profitable])) if profitable else 0.0
        lose_confidence = (
            float(np.mean([s.signal.confidence for s in executed if s.pnl is not None and s.pnl <= 0]))
            if executed
            else 0.0
        )

        # Signal decay detection
        recent_win_rate = self._calculate_recent_win_rate(signals)
        historical_win_rate = win_rate
        decay_detected = (historical_win_rate - recent_win_rate) > self.config.decay_detection_threshold

        return {
            "total_signals": float(len(signals)),
            "execution_rate": float(execution_rate),
            "win_rate": float(win_rate),
            "avg_winner_confidence": float(win_confidence),
            "avg_loser_confidence": float(lose_confidence),
            "confidence_edge": float(win_confidence - lose_confidence),
            "recent_win_rate": float(recent_win_rate),
            "signal_decay": float(decay_detected),
        }

    def get_execution_quality_metrics(self) -> dict[str, float]:
        """
        Get execution quality metrics.

        Returns
        -------
        dict[str, float]
            Execution quality metrics.

        """
        avg_latency = float(np.mean(self._execution_latencies)) if self._execution_latencies else 0.0
        p99_latency = (
            float(np.percentile(self._execution_latencies, 99)) if self._execution_latencies else 0.0
        )

        total_costs = self._total_fees + self._total_slippage
        cost_ratio = total_costs / abs(self._cumulative_pnl) if self._cumulative_pnl != 0 else 0.0

        return {
            "avg_latency_ms": float(avg_latency),
            "p99_latency_ms": float(p99_latency),
            "total_fees": self._total_fees,
            "total_slippage": self._total_slippage,
            "cost_ratio": float(cost_ratio),
        }

    def get_performance_summary(self) -> dict[str, float]:
        """
        Get comprehensive performance summary.

        Returns
        -------
        dict[str, float]
            Performance summary metrics.

        """
        # Calculate drawdown
        drawdown = 0.0
        if self._peak_pnl > 0:
            drawdown = (self._peak_pnl - self._cumulative_pnl) / self._peak_pnl

        # Get component metrics
        signal_metrics = self.get_signal_quality_metrics()
        execution_metrics = self.get_execution_quality_metrics()

        # Combine
        summary = {
            "cumulative_pnl": self._cumulative_pnl,
            "current_drawdown": drawdown,
            "sharpe_30d": self.get_sharpe_ratio(30),
            "total_signals": float(self._signal_count),
        }

        summary.update(signal_metrics)
        summary.update(execution_metrics)

        return summary

    def _get_confidence_band(self, confidence: float) -> str:
        """
        Get confidence band label for a confidence value.

        Parameters
        ----------
        confidence : float
            Confidence value [0, 1].

        Returns
        -------
        str
            Confidence band label.

        """
        for i, threshold in enumerate(self.config.confidence_bands):
            if confidence < threshold:
                if i == 0:
                    return f"<{threshold:.0%}"
                else:
                    prev = self.config.confidence_bands[i - 1]
                    return f"{prev:.0%}-{threshold:.0%}"

        # Above highest band
        highest = self.config.confidence_bands[-1]
        return f">{highest:.0%}"

    def _calculate_recent_win_rate(
        self,
        signals: list[SignalRecord],
    ) -> float:
        """
        Calculate win rate for recent signals.

        Parameters
        ----------
        signals : list[SignalRecord]
            All signals to analyze.

        Returns
        -------
        float
            Recent win rate.

        """
        # Take most recent 20% of signals
        n_recent = max(int(len(signals) * 0.2), 10)
        recent = signals[-n_recent:]

        executed = [s for s in recent if s.executed and s.pnl is not None]
        if not executed:
            return 0.0

        wins = sum(1 for s in executed if (s.pnl is not None and s.pnl > 0))
        return wins / len(executed)

    def _update_accuracy_metrics(self) -> None:
        """Update signal accuracy metrics by confidence band."""
        win_rates = self.get_win_rate_by_confidence()
        for band, rate in win_rates.items():
            signal_accuracy_gauge.labels(confidence_band=band).set(rate)

    def _check_report_schedule(self) -> None:
        """Check if periodic report should be generated."""
        now = datetime.now()
        time_since_report = (now - self._last_report_time).total_seconds() / 60

        if time_since_report >= self.config.report_frequency_minutes:
            self._generate_report()
            self._last_report_time = now

    def _generate_report(self) -> None:
        """Generate and log performance report."""
        summary = self.get_performance_summary()

        logger.info(
            f"Performance Report - "
            f"P&L: ${summary.get('cumulative_pnl', 0):.2f}, "
            f"Sharpe: {summary.get('sharpe_30d', 0):.2f}, "
            f"Win Rate: {summary.get('win_rate', 0):.1%}, "
            f"Signals: {summary.get('total_signals', 0):.0f}"
        )

        # Warn on signal decay
        if summary.get("signal_decay", 0) > 0:
            logger.warning(
                f"Signal decay detected - recent win rate {summary.get('recent_win_rate', 0):.1%} "
                f"vs historical {summary.get('win_rate', 0):.1%}"
            )

    def detect_outliers(
        self,
        returns: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.bool_]:
        """
        Detect outlier returns using z-score method.

        Parameters
        ----------
        returns : np.ndarray
            Array of returns.

        Returns
        -------
        np.ndarray
            Boolean mask of outliers.

        """
        if len(returns) < 3:
            return np.zeros(len(returns), dtype=bool)

        mean = np.mean(returns)
        std = np.std(returns)

        if std == 0:
            return np.zeros(len(returns), dtype=bool)

        z_scores: npt.NDArray[np.float64] = np.abs((returns - mean) / std)
        mask: npt.NDArray[np.bool_] = z_scores > self.config.outlier_z_score
        return mask

    def get_feature_importance(
        self,
        feature_names: list[str],
    ) -> dict[str, float]:
        """
        Estimate feature importance from signal patterns.

        Parameters
        ----------
        feature_names : list[str]
            Names of features used in signals.

        Returns
        -------
        dict[str, float]
            Estimated feature importance scores.

        """
        # Placeholder - would correlate feature values with outcomes
        # For now return uniform importance
        n_features = len(feature_names)
        if n_features == 0:
            return {}

        uniform_importance = 1.0 / n_features
        return dict.fromkeys(feature_names, uniform_importance)


# ===== Public API =====
__all__ = [
    "AnalyticsConfig",
    "PerformanceTracker",
    "SignalRecord",
]
