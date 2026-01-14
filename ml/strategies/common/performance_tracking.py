"""
Performance tracking component for MLTradingStrategy decomposition.

This component extracts model performance tracking and metrics recording logic
from BaseMLStrategy following the Protocol-First Interface Design pattern.

Responsibility:
- Track model performance metrics (wins, losses, accuracy, profit)
- Record strategy usage metrics to Prometheus
- Provide performance data access with copy semantics for safety
- Support model performance comparison and best model selection

"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    pass


@runtime_checkable
class LoggerProtocol(Protocol):
    """
    Protocol for logging interface.
    """

    def debug(self, *args: object, **kwargs: object) -> None:
        """
        Log debug message.
        """
        ...

    def info(self, *args: object, **kwargs: object) -> None:
        """
        Log info message.
        """
        ...

    def warning(self, *args: object, **kwargs: object) -> None:
        """
        Log warning message.
        """
        ...

    def error(self, *args: object, **kwargs: object) -> None:
        """
        Log error message.
        """
        ...


class _NoOpLogger:
    """
    No-op logger for when no logger is provided.
    """

    def debug(self, *args: object, **kwargs: object) -> None:
        """
        No-op debug.
        """
        del args, kwargs

    def info(self, *args: object, **kwargs: object) -> None:
        """
        No-op info.
        """
        del args, kwargs

    def warning(self, *args: object, **kwargs: object) -> None:
        """
        No-op warning.
        """
        del args, kwargs

    def error(self, *args: object, **kwargs: object) -> None:
        """
        No-op error.
        """
        del args, kwargs


class PerformanceTrackingComponent:
    """
    Tracks model performance and strategy metrics.

    This component is extracted from BaseMLStrategy to provide focused,
    testable performance tracking functionality following the facade pattern.

    Responsibilities:
    - Track per-model performance (wins, losses, accuracy, total_profit)
    - Record strategy usage metrics to Prometheus
    - Provide safe access to performance data (returns copies)
    - Support model performance comparison

    Parameters
    ----------
    strategy_id : str
        The strategy identifier for labeling metrics.
    track_performance : bool, default False
        Whether to track model performance. When False, update calls are no-ops.
    log : Any | None, optional
        Logger instance for debug output. If None, uses no-op logger.

    Examples
    --------
    >>> component = PerformanceTrackingComponent(
    ...     strategy_id="strategy_1",
    ...     track_performance=True,
    ... )
    >>> component.update_model_performance("model_a", profit=100.0)
    >>> perf = component.get_model_performance("model_a")
    >>> assert perf["total_trades"] == 1
    >>> assert perf["wins"] == 1

    """

    def __init__(
        self,
        strategy_id: str,
        track_performance: bool = False,
        log: Any = None,
    ) -> None:
        """
        Initialize the performance tracking component.
        """
        self._strategy_id = strategy_id
        self._track_performance = track_performance
        self._log = log if log is not None else _NoOpLogger()

        # Model performance tracking
        self._model_performance: dict[str, dict[str, Any]] = {}

        # Metrics (lazily initialized)
        self._signals_counter: Any = None
        self._trades_counter: Any = None
        self._positions_gauge: Any = None
        self._init_metrics()

    def _init_metrics(self) -> None:
        """
        Initialize Prometheus metrics via centralized bootstrap.
        """
        try:
            from ml.common.metrics_bootstrap import get_counter
            from ml.common.metrics_bootstrap import get_gauge

            self._signals_counter = get_counter(
                "ml_strategy_signals_received_total",
                "Total signals received by strategy",
                labelnames=("strategy_id",),
            )
            self._trades_counter = get_counter(
                "ml_strategy_trades_executed_total",
                "Total trades executed by strategy",
                labelnames=("strategy_id",),
            )
            self._positions_gauge = get_gauge(
                "ml_strategy_active_positions",
                "Active positions for strategy",
                labelnames=("strategy_id",),
            )
        except Exception:
            # Metrics unavailable - degrade gracefully
            self._log_metric_failure("metrics_bootstrap")

    def _log_metric_failure(self, metric: str) -> None:
        """
        Log metric update failures with context.

        Args:
            metric: Metric identifier or operation name.

        """
        self._log.debug(
            "Metrics update failed",
            exc_info=True,
            extra={"strategy_id": self._strategy_id, "metric": metric},
        )

    # -------------------------------------------------------------------------
    # Public Properties
    # -------------------------------------------------------------------------

    @property
    def strategy_id(self) -> str:
        """
        Get the strategy identifier.
        """
        return self._strategy_id

    @property
    def track_performance(self) -> bool:
        """
        Get whether performance tracking is enabled.
        """
        return self._track_performance

    @track_performance.setter
    def track_performance(self, value: bool) -> None:
        """
        Set whether performance tracking is enabled.
        """
        self._track_performance = value

    # -------------------------------------------------------------------------
    # Model Performance Tracking
    # -------------------------------------------------------------------------

    def update_model_performance(self, model_id: str, profit: float) -> None:
        """
        Update performance tracking for a model after trade.

        This method tracks per-model performance metrics including:
        - Total trades count
        - Win/loss counts and accuracy
        - Total profit accumulation

        Parameters
        ----------
        model_id : str
            The model identifier.
        profit : float
            The profit from the trade (positive for win, negative for loss).

        Examples
        --------
        >>> component.update_model_performance("model_a", profit=100.0)
        >>> component.update_model_performance("model_a", profit=-50.0)
        >>> perf = component.get_model_performance("model_a")
        >>> assert perf["total_trades"] == 2
        >>> assert perf["wins"] == 1
        >>> assert perf["losses"] == 1
        >>> assert perf["total_profit"] == 50.0

        """
        if not self._track_performance:
            return

        # Initialize model entry if not present
        if model_id not in self._model_performance:
            self._model_performance[model_id] = {
                "total_trades": 0,
                "total_profit": 0.0,
                "wins": 0,
                "losses": 0,
                "accuracy": 0.0,
            }

        # Update counts
        self._model_performance[model_id]["total_trades"] += 1
        self._model_performance[model_id]["total_profit"] += profit

        # Track win/loss
        if profit > 0:
            self._model_performance[model_id]["wins"] += 1
        else:
            self._model_performance[model_id]["losses"] += 1

        # Update accuracy
        total = self._model_performance[model_id]["total_trades"]
        wins = self._model_performance[model_id]["wins"]
        self._model_performance[model_id]["accuracy"] = wins / total if total > 0 else 0.0

    def get_model_performance(
        self,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Get performance data for specific model or all models.

        Returns a deep copy of the performance data to prevent external
        modification of internal state.

        Parameters
        ----------
        model_id : str | None, optional
            The model identifier. If None, returns all model performance data.

        Returns
        -------
        dict[str, Any]
            Performance data for the specified model or all models.
            For a specific model: {"total_trades": int, "total_profit": float,
                                   "wins": int, "losses": int, "accuracy": float}
            For all models: {model_id: {...}, ...}
            Returns empty dict if model_id not found or no data.

        Examples
        --------
        >>> # Get specific model performance
        >>> perf = component.get_model_performance("model_a")
        >>> assert "total_trades" in perf
        >>>
        >>> # Get all model performance
        >>> all_perf = component.get_model_performance()
        >>> assert isinstance(all_perf, dict)

        """
        if model_id is not None:
            # Return copy of specific model performance
            if model_id in self._model_performance:
                return copy.deepcopy(self._model_performance[model_id])
            return {}

        # Return copy of all model performance
        return copy.deepcopy(self._model_performance)

    def reset_model_performance(self) -> None:
        """
        Clear all model performance data.

        This method resets the performance tracking state, clearing all
        accumulated statistics for all models.

        Examples
        --------
        >>> component.update_model_performance("model_a", profit=100.0)
        >>> component.reset_model_performance()
        >>> assert component.get_model_performance() == {}

        """
        self._model_performance.clear()

    def get_best_model(self, metric: str = "total_profit") -> str | None:
        """
        Get the model ID with best performance by specified metric.

        Parameters
        ----------
        metric : str, default "total_profit"
            The metric to compare models by. Supported metrics:
            - "total_profit": Total accumulated profit
            - "accuracy": Win rate (wins / total_trades)
            - "total_trades": Number of trades executed
            - "wins": Number of winning trades

        Returns
        -------
        str | None
            The model ID with the best performance, or None if no models tracked.

        Examples
        --------
        >>> component.update_model_performance("model_a", profit=100.0)
        >>> component.update_model_performance("model_b", profit=200.0)
        >>> best = component.get_best_model("total_profit")
        >>> assert best == "model_b"

        """
        if not self._model_performance:
            return None

        try:
            best_model = max(
                self._model_performance.keys(),
                key=lambda m: self._model_performance[m].get(metric, 0.0),
            )
            return best_model
        except (ValueError, KeyError):
            return None

    # -------------------------------------------------------------------------
    # Metrics Recording
    # -------------------------------------------------------------------------

    def record_metrics_usage(
        self,
        signals_received: int,
        trades_executed: int,
        active_positions: int,
    ) -> None:
        """
        Record usage metrics to Prometheus.

        This method updates Prometheus metrics for strategy monitoring.
        It is designed to be called periodically or after significant events.

        Parameters
        ----------
        signals_received : int
            Number of signals received (increment value).
        trades_executed : int
            Number of trades executed (increment value).
        active_positions : int
            Current number of active positions (gauge value).

        Examples
        --------
        >>> component.record_metrics_usage(
        ...     signals_received=5,
        ...     trades_executed=2,
        ...     active_positions=3,
        ... )

        """
        # Increment signals counter
        if self._signals_counter is not None and signals_received > 0:
            try:
                self._signals_counter.labels(
                    strategy_id=self._strategy_id,
                ).inc(signals_received)
            except Exception:
                self._log_metric_failure("signals_received")

        # Increment trades counter
        if self._trades_counter is not None and trades_executed > 0:
            try:
                self._trades_counter.labels(
                    strategy_id=self._strategy_id,
                ).inc(trades_executed)
            except Exception:
                self._log_metric_failure("trades_executed")

        # Set positions gauge
        if self._positions_gauge is not None:
            try:
                self._positions_gauge.labels(
                    strategy_id=self._strategy_id,
                ).set(active_positions)
            except Exception:
                self._log_metric_failure("active_positions")

    def increment_signals_received(self, count: int = 1) -> None:
        """
        Increment the signals received counter.

        Parameters
        ----------
        count : int, default 1
            Number to increment by.

        """
        if self._signals_counter is not None and count > 0:
            try:
                self._signals_counter.labels(
                    strategy_id=self._strategy_id,
                ).inc(count)
            except Exception:
                self._log_metric_failure("signals_received")

    def increment_trades_executed(self, count: int = 1) -> None:
        """
        Increment the trades executed counter.

        Parameters
        ----------
        count : int, default 1
            Number to increment by.

        """
        if self._trades_counter is not None and count > 0:
            try:
                self._trades_counter.labels(
                    strategy_id=self._strategy_id,
                ).inc(count)
            except Exception:
                self._log_metric_failure("trades_executed")

    def set_active_positions(self, count: int) -> None:
        """
        Set the active positions gauge.

        Parameters
        ----------
        count : int
            Current number of active positions.

        """
        if self._positions_gauge is not None:
            try:
                self._positions_gauge.labels(
                    strategy_id=self._strategy_id,
                ).set(count)
            except Exception:
                self._log_metric_failure("active_positions")

    # -------------------------------------------------------------------------
    # Statistics and Reporting
    # -------------------------------------------------------------------------

    def get_summary_statistics(self) -> dict[str, Any]:
        """
        Get summary statistics across all models.

        Returns
        -------
        dict[str, Any]
            Summary statistics including:
            - total_models: Number of tracked models
            - total_trades: Total trades across all models
            - total_profit: Total profit across all models
            - total_wins: Total wins across all models
            - total_losses: Total losses across all models
            - overall_accuracy: Overall win rate

        Examples
        --------
        >>> stats = component.get_summary_statistics()
        >>> print(f"Total profit: {stats['total_profit']}")

        """
        total_trades = 0
        total_profit = 0.0
        total_wins = 0
        total_losses = 0

        for perf in self._model_performance.values():
            total_trades += perf.get("total_trades", 0)
            total_profit += perf.get("total_profit", 0.0)
            total_wins += perf.get("wins", 0)
            total_losses += perf.get("losses", 0)

        overall_accuracy = total_wins / total_trades if total_trades > 0 else 0.0

        return {
            "total_models": len(self._model_performance),
            "total_trades": total_trades,
            "total_profit": total_profit,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "overall_accuracy": overall_accuracy,
        }


__all__ = [
    "LoggerProtocol",
    "PerformanceTrackingComponent",
]
