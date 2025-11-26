"""
Signal routing component for MLTradingStrategy decomposition.

This component extracts signal filtering, aggregation, and routing logic
from BaseMLStrategy following the Protocol-First Interface Design pattern.

Responsibility:
- Filter incoming ML signals by model ID, confidence threshold, and instrument
- Aggregate signals from multiple models
- Manage time windows for signal aggregation
- Maintain signal history

"""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np


if TYPE_CHECKING:
    from ml.actors.base import MLSignal


@runtime_checkable
class LoggerProtocol(Protocol):
    """Protocol for logging interface."""

    def debug(self, *args: object, **kwargs: object) -> None:
        """Log debug message."""
        ...

    def info(self, *args: object, **kwargs: object) -> None:
        """Log info message."""
        ...

    def warning(self, *args: object, **kwargs: object) -> None:
        """Log warning message."""
        ...


class _NoOpLogger:
    """No-op logger for when no logger is provided."""

    def debug(self, *args: object, **kwargs: object) -> None:
        """No-op debug."""
        del args, kwargs

    def info(self, *args: object, **kwargs: object) -> None:
        """No-op info."""
        del args, kwargs

    def warning(self, *args: object, **kwargs: object) -> None:
        """No-op warning."""
        del args, kwargs


class SignalRoutingComponent:
    """
    Routes and aggregates ML signals for strategy processing.

    This component is extracted from BaseMLStrategy to provide focused,
    testable signal routing functionality following the facade pattern.

    Responsibilities:
    - Filter signals by model ID, confidence, instrument
    - Aggregate signals from multiple models using weighted average or voting
    - Manage signal history and time windows
    - Track per-model signal buffers

    Parameters
    ----------
    target_model_ids : list[str] | None, optional
        List of model IDs to accept signals from. If None, accepts all models.
    aggregation_mode : str | None, optional
        Aggregation mode: "weighted_average" or "voting". If None, no aggregation.
    required_models : int, default 1
        Minimum number of model signals required before aggregation.
    time_window_ms : int, default 1000
        Time window in milliseconds for signal aggregation.
    conflict_resolution : str | None, optional
        Conflict resolution mode: "weighted_average" or "voting".
    model_weights : dict[str, float] | None, optional
        Weights for each model in weighted average aggregation.
    min_confidence : float, default 0.0
        Minimum confidence threshold for accepting signals.
    history_size : int, default 100
        Maximum number of signals to retain in history.
    instrument_id : InstrumentId | str | None, optional
        Target instrument ID for filtering. If None, accepts all instruments.
    log : LoggerProtocol | None, optional
        Logger instance for debug output.

    Examples
    --------
    >>> component = SignalRoutingComponent(
    ...     target_model_ids=["model_a", "model_b"],
    ...     aggregation_mode="weighted_average",
    ...     required_models=2,
    ...     min_confidence=0.6,
    ... )
    >>> result = component.route_signal(signal)

    """

    def __init__(
        self,
        target_model_ids: list[str] | None = None,
        aggregation_mode: str | None = None,
        required_models: int = 1,
        time_window_ms: int = 1000,
        conflict_resolution: str | None = None,
        model_weights: dict[str, float] | None = None,
        min_confidence: float = 0.0,
        history_size: int = 100,
        instrument_id: Any = None,
        log: Any = None,
    ) -> None:
        """Initialize the signal routing component."""
        self._target_model_ids = target_model_ids
        self._aggregation_mode = aggregation_mode
        self._required_models = required_models
        self._time_window_ms = time_window_ms
        self._conflict_resolution = conflict_resolution or aggregation_mode
        self._model_weights = model_weights or {}
        self._min_confidence = min_confidence
        self._instrument_id = instrument_id
        self._log = log if log is not None else _NoOpLogger()

        # Signal storage
        self._signal_history: deque[MLSignal] = deque(maxlen=history_size)
        self._signal_buffer: dict[str, MLSignal] = {}

    # -------------------------------------------------------------------------
    # Public Properties
    # -------------------------------------------------------------------------

    @property
    def signal_history(self) -> deque[MLSignal]:
        """
        Access signal history deque.

        Returns
        -------
        deque[MLSignal]
            The bounded deque of historical signals.

        """
        return self._signal_history

    @property
    def signal_buffer(self) -> dict[str, MLSignal]:
        """
        Access per-model signal buffer.

        Returns
        -------
        dict[str, MLSignal]
            Dictionary mapping model_id to latest signal.

        """
        return self._signal_buffer

    @property
    def target_model_ids(self) -> list[str] | None:
        """Get the list of target model IDs."""
        return self._target_model_ids

    @property
    def aggregation_mode(self) -> str | None:
        """Get the aggregation mode."""
        return self._aggregation_mode

    @property
    def required_models(self) -> int:
        """Get the required number of models for aggregation."""
        return self._required_models

    @property
    def time_window_ms(self) -> int:
        """Get the time window in milliseconds."""
        return self._time_window_ms

    @property
    def min_confidence(self) -> float:
        """Get the minimum confidence threshold."""
        return self._min_confidence

    # -------------------------------------------------------------------------
    # Filtering Methods
    # -------------------------------------------------------------------------

    def filter_by_model_id(self, signal: MLSignal) -> bool:
        """
        Check if signal should be accepted based on model ID.

        Parameters
        ----------
        signal : MLSignal
            The signal to check.

        Returns
        -------
        bool
            True if signal should be accepted, False otherwise.

        Examples
        --------
        >>> component = SignalRoutingComponent(target_model_ids=["model_a"])
        >>> component.filter_by_model_id(signal_from_model_a)
        True
        >>> component.filter_by_model_id(signal_from_model_b)
        False

        """
        if self._target_model_ids is None:
            return True

        model_id = self._extract_model_id(signal)
        return model_id in self._target_model_ids

    def filter_by_confidence(self, signal: MLSignal) -> bool:
        """
        Check if signal meets confidence threshold.

        Parameters
        ----------
        signal : MLSignal
            The signal to check.

        Returns
        -------
        bool
            True if signal confidence >= min_confidence, False otherwise.

        """
        return float(signal.confidence) >= self._min_confidence

    def filter_by_instrument(self, signal: MLSignal) -> bool:
        """
        Check if signal matches target instrument.

        Parameters
        ----------
        signal : MLSignal
            The signal to check.

        Returns
        -------
        bool
            True if signal matches target instrument or no filter set.

        """
        if self._instrument_id is None:
            return True

        # Handle both InstrumentId objects and strings
        signal_inst = signal.instrument_id
        target_inst = self._instrument_id

        # Compare using string representation for flexibility
        return str(signal_inst) == str(target_inst)

    # -------------------------------------------------------------------------
    # Aggregation Methods
    # -------------------------------------------------------------------------

    def should_aggregate(self) -> bool:
        """
        Check if aggregation should occur.

        Returns
        -------
        bool
            True if aggregation is enabled and sufficient models present.

        """
        if self._aggregation_mode is None:
            return False

        return len(self._signal_buffer) >= self._required_models

    def aggregate_signals(self) -> MLSignal | None:
        """
        Aggregate buffered signals into a single signal.

        Uses either weighted average or voting based on configuration.
        Returns None if aggregation cannot be performed.

        Returns
        -------
        MLSignal | None
            The aggregated signal, or None if aggregation failed.

        """
        if not self._signal_buffer:
            return None

        # Check time window - all signals must be within window
        signals = list(self._signal_buffer.values())
        if not self._signals_within_time_window(signals):
            self._purge_stale_signals_internal()
            return None

        # Check minimum required models
        if len(signals) < self._required_models:
            return None

        # Get latest timestamp for aggregated signal
        latest_ts = max(s.ts_event for s in signals)

        # Determine aggregation method
        if self._conflict_resolution == "weighted_average":
            return self._aggregate_weighted_average(signals, latest_ts)
        else:
            return self._aggregate_voting(signals, latest_ts)

    def _aggregate_weighted_average(
        self,
        signals: list[MLSignal],
        latest_ts: int,
    ) -> MLSignal:
        """
        Aggregate signals using weighted average.

        Parameters
        ----------
        signals : list[MLSignal]
            The signals to aggregate.
        latest_ts : int
            The latest timestamp among signals.

        Returns
        -------
        MLSignal
            The aggregated signal.

        """
        from ml.actors.base import MLSignal as MLSignalClass

        total_weight = 0.0
        weighted_sum = 0.0

        for sig in signals:
            model_id = self._extract_model_id(sig)
            weight = self._model_weights.get(model_id, 1.0)
            weighted_sum += weight * float(sig.prediction)
            total_weight += weight

        weighted_pred = weighted_sum / total_weight if total_weight > 0 else 0.5
        avg_confidence = float(np.mean([s.confidence for s in signals]))

        # Get instrument_id from first signal
        instrument_id = signals[0].instrument_id

        return MLSignalClass(
            instrument_id=instrument_id,
            model_id="aggregated",
            prediction=weighted_pred,
            confidence=avg_confidence,
            metadata={"aggregated_from": list(self._signal_buffer.keys())},
            ts_event=latest_ts,
            ts_init=time.time_ns(),
        )

    def _aggregate_voting(
        self,
        signals: list[MLSignal],
        latest_ts: int,
    ) -> MLSignal:
        """
        Aggregate signals using majority voting.

        Parameters
        ----------
        signals : list[MLSignal]
            The signals to aggregate.
        latest_ts : int
            The latest timestamp among signals.

        Returns
        -------
        MLSignal
            The aggregated signal.

        """
        from ml.actors.base import MLSignal as MLSignalClass

        bullish = sum(1 for s in signals if float(s.prediction) > 0.5)
        bearish = len(signals) - bullish

        # Determine action based on majority
        action = "BUY" if bullish > bearish else "SELL"
        prediction = 0.8 if action == "BUY" else 0.2
        confidence = max(float(s.confidence) for s in signals)

        instrument_id = signals[0].instrument_id

        return MLSignalClass(
            instrument_id=instrument_id,
            model_id="aggregated",
            prediction=prediction,
            confidence=confidence,
            metadata={
                "action": action,
                "aggregated_from": list(self._signal_buffer.keys()),
            },
            ts_event=latest_ts,
            ts_init=time.time_ns(),
        )

    def _signals_within_time_window(self, signals: list[MLSignal]) -> bool:
        """Check if all signals are within the time window."""
        if not signals:
            return False

        latest_time = max(s.ts_event for s in signals)
        earliest_time = min(s.ts_event for s in signals)
        time_diff_ms = (latest_time - earliest_time) / 1_000_000  # ns to ms

        return time_diff_ms <= self._time_window_ms

    def _purge_stale_signals_internal(self) -> None:
        """Remove signals older than time window from buffer."""
        if not self._signal_buffer:
            return

        latest_time = max(s.ts_event for s in self._signal_buffer.values())
        stale_models = []

        for model_id, sig in self._signal_buffer.items():
            time_diff_ms = (latest_time - sig.ts_event) / 1_000_000
            if time_diff_ms > self._time_window_ms:
                stale_models.append(model_id)

        for model_id in stale_models:
            del self._signal_buffer[model_id]

    # -------------------------------------------------------------------------
    # Buffer Management Methods
    # -------------------------------------------------------------------------

    def add_to_buffer(self, signal: MLSignal) -> None:
        """
        Add signal to per-model buffer.

        Parameters
        ----------
        signal : MLSignal
            The signal to add to the buffer.

        """
        model_id = self._extract_model_id(signal)
        self._signal_buffer[model_id] = signal

    def add_to_history(self, signal: MLSignal) -> None:
        """
        Add signal to history deque.

        Parameters
        ----------
        signal : MLSignal
            The signal to add to history.

        """
        self._signal_history.append(signal)

    def purge_stale_signals(self) -> None:
        """
        Remove signals older than time window from buffer.

        This is the public API for purging stale signals. It delegates
        to the internal implementation.

        """
        self._purge_stale_signals_internal()

    def clear_buffer(self) -> None:
        """Clear the signal buffer after aggregation."""
        self._signal_buffer.clear()

    def get_model_signal(self, model_id: str) -> MLSignal | None:
        """
        Get the latest signal from a specific model.

        Parameters
        ----------
        model_id : str
            The model identifier.

        Returns
        -------
        MLSignal | None
            The latest signal from the model, or None if not found.

        """
        return self._signal_buffer.get(model_id)

    # -------------------------------------------------------------------------
    # Main Routing Method
    # -------------------------------------------------------------------------

    def route_signal(self, signal: MLSignal) -> MLSignal | None:
        """
        Main entry point: filter, buffer, and optionally aggregate signals.

        This method applies all configured filters to the incoming signal,
        adds it to history, and handles aggregation if configured.

        Parameters
        ----------
        signal : MLSignal
            The incoming ML signal to route.

        Returns
        -------
        MLSignal | None
            The signal to process (original or aggregated), or None if filtered out.

        Examples
        --------
        >>> component = SignalRoutingComponent(min_confidence=0.6)
        >>> result = component.route_signal(low_confidence_signal)
        >>> result is None
        True

        """
        # Always add to history regardless of filtering
        self.add_to_history(signal)

        # Apply filters
        if not self.filter_by_model_id(signal):
            self._log.debug(
                f"Signal filtered: model_id not in target list "
                f"({self._extract_model_id(signal)})"
            )
            return None

        if not self.filter_by_confidence(signal):
            self._log.debug(
                f"Signal filtered: confidence below threshold "
                f"({signal.confidence} < {self._min_confidence})"
            )
            return None

        if not self.filter_by_instrument(signal):
            self._log.debug(
                f"Signal filtered: instrument mismatch "
                f"({signal.instrument_id} != {self._instrument_id})"
            )
            return None

        # Handle aggregation if configured
        if self._aggregation_mode is not None:
            self.add_to_buffer(signal)
            self.purge_stale_signals()

            if self.should_aggregate():
                aggregated = self.aggregate_signals()
                if aggregated is not None:
                    self.clear_buffer()
                    return aggregated
            # Not enough signals yet for aggregation
            return None

        # No aggregation - return original signal
        return signal

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def _extract_model_id(self, signal: MLSignal) -> str:
        """
        Extract model ID from signal.

        Parameters
        ----------
        signal : MLSignal
            The signal to extract model ID from.

        Returns
        -------
        str
            The model ID, or "unknown" if not found.

        """
        model_id = getattr(signal, "model_id", None)
        if model_id is not None:
            return str(model_id)

        # Fall back to metadata
        if hasattr(signal, "metadata") and signal.metadata:
            return str(signal.metadata.get("model_id", "unknown"))

        return "unknown"
