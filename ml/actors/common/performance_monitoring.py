"""
Performance Monitoring Component.

This module implements performance monitoring and metrics emission for MLSignalActor
decomposition.

The component provides:
- Ring buffer-based timing capture (hot path, zero allocations)
- Signal and error counting (hot path, zero allocations)
- Performance statistics calculation (cold path)
- Latency percentile calculation (P50, P90, P95, P99)
- Prometheus metrics initialization and management

Hot Path Performance:
- record_timing(): <50μs, zero allocations
- record_signal(): <1μs, zero allocations
- record_error(): <1μs, zero allocations

Cold Path:
- get_current_stats(): Allocations allowed (dict creation)
- get_latency_percentiles(): Allocations allowed (nested dict)
- initialize_metrics(): Allocations allowed (metric registration)

Architecture Patterns (CLAUDE.md):
- Pattern 3: Hot/Cold Path Separation (zero allocations in hot path)
- Pattern 2: Protocol-First Interface Design (property accessors)

Metrics Managed (9 total):
- ml_prediction_distribution
- ml_confidence_distribution
- ml_signal_generation_seconds
- ml_feature_time_by_set_seconds
- ml_signals_generated_total
- ml_adaptive_threshold
- ml_market_regime_total
- ml_feature_parity_checks_total
- ml_feature_parity_drift

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt


if TYPE_CHECKING:
    from logging import Logger


class PerformanceMonitoringComponent:
    """
    Component for performance monitoring and metrics emission.

    Manages timing measurements with zero-allocation hot path guarantees using
    fixed-size ring buffers. Provides performance statistics, latency percentiles,
    and Prometheus metrics integration.

    Hot Path Requirements:
    - record_timing() MUST use zero allocations
    - record_signal() MUST use zero allocations
    - record_error() MUST use zero allocations
    - P99 <50μs per record_timing()

    Cold Path:
    - get_current_stats() allocations allowed (dict creation)
    - get_latency_percentiles() allocations allowed (nested dict)
    - initialize_metrics() allocations allowed (metric registration)

    Example:
        >>> monitor = PerformanceMonitoringComponent(reservoir_size=1000)
        >>> monitor.record_timing(feature_time_ns=500_000, inference_time_ns=2_000_000, total_time_ns=2_500_000)
        >>> monitor.record_signal()
        >>> stats = monitor.get_current_stats()
        >>> assert stats["signal_count"] == 1
        >>> percentiles = monitor.get_latency_percentiles()

    """

    def __init__(
        self,
        reservoir_size: int = 1000,
        actor_id: str | None = None,
        log: Logger | None = None,
    ) -> None:
        """
        Initialize performance monitoring component.

        Parameters
        ----------
        reservoir_size : int, default=1000
            Fixed ring capacity for timing samples. Must be > 0.
        actor_id : str | None, default=None
            Actor identifier for metrics labeling (optional).
        log : Logger | None, default=None
            Logger instance (optional).

        Raises
        ------
        ValueError
            If reservoir_size <= 0.

        """
        if reservoir_size <= 0:
            raise ValueError(f"reservoir_size ({reservoir_size}) must be > 0")

        cap = max(1, int(reservoir_size))
        self._cap = cap
        self._idx = 0
        self._count = 0
        self._actor_id = actor_id
        self._log = log

        # Milliseconds stored as float32 to reduce footprint
        self._feature_times_ms: npt.NDArray[np.float32] = np.zeros(cap, dtype=np.float32)
        self._inference_times_ms: npt.NDArray[np.float32] = np.zeros(cap, dtype=np.float32)
        self._total_times_ms: npt.NDArray[np.float32] = np.zeros(cap, dtype=np.float32)

        # Counters
        self._prediction_count = 0
        self._signal_count = 0
        self._error_count = 0

        # Metrics management
        self._metrics_initialized = False
        self._metrics: dict[str, Any] = {}

    def record_timing(
        self,
        feature_time_ns: int,
        inference_time_ns: int,
        total_time_ns: int,
    ) -> None:
        """
        Record timing measurements (hot path).

        Stores timing values in ring buffers with zero allocations.
        Times are converted from nanoseconds to milliseconds.

        Parameters
        ----------
        feature_time_ns : int
            Feature computation time in nanoseconds.
        inference_time_ns : int
            Model inference time in nanoseconds.
        total_time_ns : int
            Total time (feature + inference) in nanoseconds.

        Notes
        -----
        Hot path safe:
        - Zero allocations (pre-allocated buffers)
        - Circular indexing with fixed capacity
        - P99 <50μs per call

        """
        i = self._idx
        # Convert to milliseconds and store (no temporary allocations)
        self._feature_times_ms[i] = np.float32(feature_time_ns / 1_000_000.0)
        self._inference_times_ms[i] = np.float32(inference_time_ns / 1_000_000.0)
        self._total_times_ms[i] = np.float32(total_time_ns / 1_000_000.0)

        # Circular indexing - no allocations
        i += 1
        if i >= self._cap:
            i = 0
        self._idx = i

        # Increment count until capacity reached
        if self._count < self._cap:
            self._count += 1

        # Increment prediction counter
        self._prediction_count += 1

    def record_signal(self) -> None:
        """
        Record successful signal generation event (hot path).

        Increments signal_count counter with zero allocations.

        Notes
        -----
        Hot path safe:
        - Zero allocations (in-memory counter increment)
        - Used to derive signal rate with prediction_count
        - P99 <1μs per call

        """
        self._signal_count += 1

    def record_error(self) -> None:
        """
        Record error during signal attempt (hot path).

        Increments error_count counter with zero allocations.

        Notes
        -----
        Hot path safe:
        - Zero allocations (in-memory counter increment)
        - Used to derive error rate with prediction_count
        - P99 <1μs per call

        """
        self._error_count += 1

    def get_current_stats(self) -> dict[str, Any]:
        """
        Get current performance statistics (cold path).

        Returns
        -------
        dict[str, Any]
            Statistics dictionary containing:
            - prediction_count: Total predictions made
            - signal_count: Total signals generated
            - error_count: Total errors encountered
            - signal_rate: signal_count / prediction_count
            - error_rate: error_count / prediction_count
            - avg_feature_time_ms: Average feature computation time
            - avg_inference_time_ms: Average inference time
            - avg_total_time_ms: Average total time
            - p99_total_time_ms: 99th percentile total time
            - last_feature_time_ms: Most recent feature time (if n > 0)
            - last_inference_time_ms: Most recent inference time (if n > 0)
            - last_total_time_ms: Most recent total time (if n > 0)

        Notes
        -----
        Cold path:
        - Allocations allowed (dict creation, numpy slicing)
        - Uses _count (actual samples) not _cap (capacity)
        - Returns 0.0 for averages when no data available

        """
        n = int(self._count)
        ft = self._feature_times_ms[:n]
        it = self._inference_times_ms[:n]
        tt = self._total_times_ms[:n]

        stats = {
            "prediction_count": self._prediction_count,
            "signal_count": self._signal_count,
            "error_count": self._error_count,
            "signal_rate": self._signal_count / max(self._prediction_count, 1),
            "error_rate": self._error_count / max(self._prediction_count, 1),
            "avg_feature_time_ms": float(np.mean(ft)) if n else 0.0,
            "avg_inference_time_ms": float(np.mean(it)) if n else 0.0,
            "avg_total_time_ms": float(np.mean(tt)) if n else 0.0,
            "p99_total_time_ms": float(np.percentile(tt, 99)) if n else 0.0,
        }

        # Add last timing values if available
        if n:
            last = (self._idx - 1) % self._cap
            stats["last_feature_time_ms"] = float(self._feature_times_ms[last])
            stats["last_inference_time_ms"] = float(self._inference_times_ms[last])
            stats["last_total_time_ms"] = float(self._total_times_ms[last])

        return stats

    def get_latency_percentiles(self) -> dict[str, dict[float, float]]:
        """
        Get latency percentiles for each measurement type (cold path).

        Calculates P50, P90, P95, P99 for feature, inference, and total times.

        Returns
        -------
        dict[str, dict[float, float]]
            Nested dictionary:
            - "feature_computation": {50.0: val, 90.0: val, 95.0: val, 99.0: val}
            - "inference": {50.0: val, 90.0: val, 95.0: val, 99.0: val}
            - "total": {50.0: val, 90.0: val, 95.0: val, 99.0: val}
            Returns empty dict if no data recorded.

        Notes
        -----
        Cold path:
        - Allocations allowed (dict creation, numpy percentile)
        - Uses _count (actual samples) not _cap (capacity)
        - Returns empty dict when n == 0

        """
        percentiles = [50.0, 90.0, 95.0, 99.0]
        result: dict[str, dict[float, float]] = {}
        n = int(self._count)

        if not n:
            return result

        ft = self._feature_times_ms[:n]
        it = self._inference_times_ms[:n]
        tt = self._total_times_ms[:n]

        result["feature_computation"] = {p: float(np.percentile(ft, p)) for p in percentiles}
        result["inference"] = {p: float(np.percentile(it, p)) for p in percentiles}
        result["total"] = {p: float(np.percentile(tt, p)) for p in percentiles}

        return result

    def initialize_metrics(self) -> None:
        """
        Initialize Prometheus metrics for performance tracking.

        Initializes 9 metrics:
        - ml_prediction_distribution
        - ml_confidence_distribution
        - ml_signal_generation_seconds
        - ml_feature_time_by_set_seconds
        - ml_signals_generated_total
        - ml_adaptive_threshold
        - ml_market_regime_total
        - ml_feature_parity_checks_total
        - ml_feature_parity_drift

        Idempotent: safe to call multiple times.

        Notes
        -----
        Cold path:
        - Allocations allowed (metric registration)
        - Uses MetricsManager singleton
        - Stores metric instances in self._metrics
        - Safe for repeated calls (idempotent)

        """
        if self._metrics_initialized:
            return

        from ml.common.metrics_manager import MetricsManager
        from ml.config.names import FEATURE_TIME_BUCKETS
        from ml.config.names import LABEL_ACTOR_ID
        from ml.config.names import LABEL_FEATURE_SET_ID
        from ml.config.names import METRIC_ADAPTIVE_THRESHOLD
        from ml.config.names import METRIC_CONFIDENCE_DISTRIBUTION
        from ml.config.names import METRIC_FEATURE_TIME_BY_SET_SECONDS
        from ml.config.names import METRIC_MARKET_REGIME_TOTAL
        from ml.config.names import METRIC_PREDICTION_DISTRIBUTION
        from ml.config.names import METRIC_SIGNAL_GENERATION_SECONDS
        from ml.config.names import METRIC_SIGNALS_GENERATED_TOTAL
        from ml.config.names import SIGNAL_LATENCY_BUCKETS

        mm = MetricsManager.default()

        # Initialize all 9 metrics
        self._metrics["prediction_distribution"] = mm.histogram(
            METRIC_PREDICTION_DISTRIBUTION,
            "Distribution of model predictions",
            [LABEL_ACTOR_ID],
        )
        self._metrics["confidence_distribution"] = mm.histogram(
            METRIC_CONFIDENCE_DISTRIBUTION,
            "Distribution of prediction confidence scores",
            [LABEL_ACTOR_ID],
        )
        self._metrics["signal_generation_time"] = mm.histogram(
            METRIC_SIGNAL_GENERATION_SECONDS,
            "Signal generation latency in seconds",
            [LABEL_ACTOR_ID, "strategy"],
            buckets=SIGNAL_LATENCY_BUCKETS,
        )
        self._metrics["feature_time_by_feature_set"] = mm.histogram(
            METRIC_FEATURE_TIME_BY_SET_SECONDS,
            "Feature computation latency by feature_set_id",
            [LABEL_ACTOR_ID, LABEL_FEATURE_SET_ID],
            buckets=FEATURE_TIME_BUCKETS,
        )
        self._metrics["signals_generated"] = mm.counter(
            METRIC_SIGNALS_GENERATED_TOTAL,
            "Total number of signals generated",
            [LABEL_ACTOR_ID, "strategy", "signal_type"],
        )
        self._metrics["adaptive_threshold"] = mm.histogram(
            METRIC_ADAPTIVE_THRESHOLD,
            "Adaptive threshold values",
            [LABEL_ACTOR_ID],
        )
        self._metrics["market_regime"] = mm.counter(
            METRIC_MARKET_REGIME_TOTAL,
            "Market regime detection counts",
            [LABEL_ACTOR_ID, "regime"],
        )

        # Parity smoke-check metrics
        self._metrics["feature_parity_checks_total"] = mm.counter(
            "ml_feature_parity_checks_total",
            "Total parity smoke-checks executed",
            [LABEL_ACTOR_ID],
        )
        self._metrics["feature_parity_drift"] = mm.gauge(
            "ml_feature_parity_drift",
            "Max absolute feature difference in parity smoke-check",
            [LABEL_ACTOR_ID],
        )

        self._metrics_initialized = True

        if self._log:
            self._log.debug(
                f"Initialized 9 performance metrics for actor_id={self._actor_id}",
            )

    @property
    def prediction_count(self) -> int:
        """
        Get total prediction count.

        Returns
        -------
        int
            Total number of predictions made (record_timing calls).

        """
        return self._prediction_count

    @property
    def signal_count(self) -> int:
        """
        Get total signal count.

        Returns
        -------
        int
            Total number of signals generated (record_signal calls).

        """
        return self._signal_count

    @property
    def error_count(self) -> int:
        """
        Get total error count.

        Returns
        -------
        int
            Total number of errors encountered (record_error calls).

        """
        return self._error_count

    @property
    def reservoir_size(self) -> int:
        """
        Get ring buffer capacity.

        Returns
        -------
        int
            Fixed capacity of timing ring buffers.

        """
        return self._cap

    @property
    def metrics(self) -> dict[str, Any]:
        """
        Get initialized metrics dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary of initialized Prometheus metrics.
            Empty dict if initialize_metrics() not called yet.

        """
        return self._metrics
