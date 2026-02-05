"""
Features Component.

This module implements hot path feature computation, buffering, and validation logic
extracted from BaseMLInferenceActor. This is the HOTTEST PATH component with strict
performance requirements: P99 <500μs per bar.

All ML actors MUST use this component for:
- Feature computation from Bar events
- Lookback buffer management (deque with maxlen)
- Feature schema validation (FeatureRegistry integration)
- Warm-up status tracking
- Async persistence coordination (MLPersistenceWorker)
- Health monitoring integration

Performance Requirements (CRITICAL):
- P99 latency: <500μs per bar (stricter than general <5ms)
- Zero allocations in hot path (pre-allocate all buffers)
- No I/O in hot path (all persistence async via queue)
- No locks in hot path
- Schema validation <50μs overhead

"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram
from ml.config.base import MLFeatureConfig


if TYPE_CHECKING:
    from nautilus_trader.model.data import Bar

    from ml.config.base import MLActorConfig
    from ml.observability.health_monitor import HealthMonitor
    from ml.observability.ml_async_persistence import MLPersistenceWorker
    from ml.registry.feature import FeatureRegistry
    from ml.stores.protocols import FeatureStoreStrictProtocol


class FeaturesProtocol(Protocol):
    """
    Protocol for features operations component.

    Defines the interface for feature computation, buffering, validation, warm-up
    tracking, and persistence coordination.

    """

    def compute_features(self, bar: Bar) -> npt.NDArray[np.float32] | None:
        """
        Compute features from Bar event (hot path <500μs).

        Args:
            bar: Bar event with OHLCV data

        Returns:
            Feature array or None if indicators not ready

        """
        ...

    def buffer_bar(self, bar: Bar) -> None:
        """
        Buffer Bar in lookback window.

        Args:
            bar: Bar to buffer

        """
        ...

    def get_buffered_bars(self) -> list[Bar]:
        """
        Retrieve buffered bars from lookback window.

        Returns:
            List of buffered bars (most recent last)

        """
        ...

    def is_warmed_up(self) -> bool:
        """
        Check if feature computation is warmed up.

        Returns:
            True if warmed up, False otherwise

        """
        ...

    def validate_features(self, features: npt.NDArray[np.float32]) -> bool:
        """
        Validate features against FeatureRegistry schema.

        Args:
            features: Feature array to validate

        Returns:
            True if valid, False otherwise

        """
        ...

    def persist_features_async(
        self,
        feature_set_id: str,
        instrument_id: str,
        features: dict[str, float],
        ts_event: int,
        ts_init: int,
    ) -> bool:
        """
        Persist features asynchronously via MLPersistenceWorker.

        Args:
            feature_set_id: Feature set identifier
            instrument_id: Instrument identifier
            features: Feature dict (name -> value)
            ts_event: Event timestamp (nanoseconds)
            ts_init: Initialization timestamp (nanoseconds)

        Returns:
            True if enqueued, False if queue full

        """
        ...

    def update_dependencies(
        self,
        *,
        health_monitor: HealthMonitor | None = None,
        persistence_worker: MLPersistenceWorker | None = None,
    ) -> None:
        """
        Update optional dependencies after initialization.

        Args:
            health_monitor: Optional health monitor to attach.
            persistence_worker: Optional async persistence worker to attach.

        """


def build_feature_dict(
    features: npt.NDArray[np.float32],
    *,
    feature_names: Sequence[str] | None = None,
) -> dict[str, float]:
    """
    Build a feature dictionary from a feature array.

    Args:
        features: Feature array of shape (n_features,).
        feature_names: Optional list of names aligned with the feature array.

    Returns:
        Mapping of feature name to float value.

    Example:
        >>> feats = np.array([0.1, 0.2], dtype=np.float32)
        >>> build_feature_dict(feats, feature_names=["rsi", "sma"])
        {'rsi': 0.1, 'sma': 0.2}
    """
    if feature_names is not None and len(feature_names) == int(features.shape[0]):
        return {feature_names[i]: float(features[i]) for i in range(len(feature_names))}
    return {f"feature_{i}": float(v) for i, v in enumerate(features)}

    def cleanup(self) -> None:
        """
        Release feature computation resources.
        """


class FeaturesComponent:
    """
    Manages feature computation, buffering, validation, and persistence.

    This is the HOTTEST PATH component in the ML system with strict performance
    requirements: P99 <500μs per bar, zero allocations in hot path.

    The component handles:
    - Feature computation via user-provided compute function
    - Lookback buffer management (deque with maxlen, FIFO eviction)
    - Feature schema validation (FeatureRegistry integration)
    - Warm-up status tracking (based on indicator initialization)
    - Async persistence coordination (MLPersistenceWorker queue)
    - Health monitoring integration (latency violation tracking)
    - Centralized metrics (computation count, latency, validation overhead)

    Hot Path Optimizations:
    - Pre-allocated feature buffer (zero allocation)
    - Deque with maxlen (automatic FIFO eviction)
    - Lock-free buffer operations
    - Async persistence (no I/O blocking)
    - Fast schema validation (<50μs)

    Example:
        >>> config = MLActorConfig(
        ...     model_path="/models/test.onnx",
        ...     model_id="test_model",
        ...     bar_type=BarType.from_str("EUR/USD.SIM-1-MINUTE-LAST-EXTERNAL"),
        ...     instrument_id=InstrumentId.from_str("EUR/USD.SIM"),
        ...     feature_config=MLFeatureConfig(lookback_window=50),
        ... )
        >>> feature_registry = FeatureRegistry(...)
        >>> feature_store = FeatureStore(...)
        >>> health_monitor = HealthMonitor(...)
        >>> persistence_worker = MLPersistenceWorker(...)
        >>>
        >>> # Define compute function
        >>> def compute_fn(bar: Bar) -> np.ndarray | None:
        ...     # Compute features from bar
        ...     return np.zeros(20, dtype=np.float32)
        >>>
        >>> component = FeaturesComponent(
        ...     config=config,
        ...     compute_function=compute_fn,
        ...     feature_registry=feature_registry,
        ...     feature_store=feature_store,
        ...     health_monitor=health_monitor,
        ...     persistence_worker=persistence_worker,
        ...     logger=logging.getLogger(__name__),
        ... )
        >>>
        >>> # Hot path usage
        >>> features = component.compute_features(bar)
        >>> if features is not None:
        ...     valid = component.validate_features(features)
        ...     if valid:
        ...         component.persist_features_async(...)

    """

    def __init__(
        self,
        config: MLActorConfig,
        compute_function: Callable[[Bar], npt.NDArray[np.float32] | None],
        feature_registry: FeatureRegistry,
        feature_store: FeatureStoreStrictProtocol,
        *,
        health_monitor: HealthMonitor | None = None,
        persistence_worker: MLPersistenceWorker | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize features operations component.

        Args:
            config: ML actor configuration with feature settings
            compute_function: Callable that computes features from Bar
            feature_registry: FeatureRegistry for schema validation
            feature_store: FeatureStore for synchronous persistence fallback
            health_monitor: Optional HealthMonitor for latency tracking
            persistence_worker: Optional MLPersistenceWorker for async persistence
            logger: Optional logger (defaults to module logger)

        Raises:
            ValueError: If config missing required feature settings

        """
        self._config = config
        self._compute_function = compute_function
        self._feature_registry = feature_registry
        self._feature_store = feature_store
        self._health_monitor = health_monitor
        self._persistence_worker = persistence_worker
        self._logger = logger or logging.getLogger(__name__)

        # Feature configuration
        # Use config.feature_config if provided, otherwise use default
        # Important: Handle None case explicitly (don't use getattr which returns None if attribute exists)
        self._feature_config: MLFeatureConfig = (
            config.feature_config if config.feature_config is not None else MLFeatureConfig()
        )

        # Lookback buffer (deque with maxlen for automatic FIFO eviction)
        self._bar_buffer: deque[Bar] = deque(
            maxlen=self._feature_config.lookback_window,
        )

        # Feature window (rolling window of computed features)
        self._feature_window: deque[npt.NDArray[np.float32]] = deque(
            maxlen=self._feature_config.lookback_window,
        )

        # Warm-up tracking
        self._bars_processed = 0
        self._is_warmed_up = False
        self._warmup_bars_required = getattr(
            self._feature_config,
            "warmup_bars",
            20,  # Default 20 bars for indicator initialization
        )
        self._sync_fallback_disabled_logged = False

        # Performance tracking
        self._total_feature_time = 0.0
        self._feature_count = 0

        # Centralized metrics with labelnames
        self._feature_computation_counter = get_counter(
            "ml_features_computed_total",
            "Total features computed",
            labelnames=("component", "instrument", "status"),
        )
        self._feature_latency_histogram = get_histogram(
            "ml_feature_computation_seconds",
            "Feature computation latency",
            labelnames=("component", "instrument"),
        )
        self._validation_counter = get_counter(
            "ml_feature_validation_total",
            "Total feature validations",
            labelnames=("component", "result"),
        )
        self._persistence_counter = get_counter(
            "ml_feature_persistence_total",
            "Total feature persistence operations",
            labelnames=("component", "mode", "result"),
        )

        self._logger.info(
            f"FeaturesComponent initialized: lookback_window={self._feature_config.lookback_window}, "
            f"warmup_bars={self._warmup_bars_required}",
        )

    def update_dependencies(
        self,
        *,
        health_monitor: HealthMonitor | None = None,
        persistence_worker: MLPersistenceWorker | None = None,
    ) -> None:
        """
        Update optional dependencies after initialization.

        Args:
            health_monitor: Optional health monitor to attach.
            persistence_worker: Optional async persistence worker to attach.

        Example:
            >>> component.update_dependencies(health_monitor=monitor, persistence_worker=worker)
        """
        self._health_monitor = health_monitor
        self._persistence_worker = persistence_worker

    def compute_features(self, bar: Bar) -> npt.NDArray[np.float32] | None:
        """
        Compute features from Bar event (hot path <500μs).

        Calls user-provided compute function and tracks performance.
        Updates warm-up status based on bars processed.

        Args:
            bar: Bar event with OHLCV data

        Returns:
            Feature array or None if indicators not ready

        Example:
            >>> component = FeaturesComponent(...)
            >>> features = component.compute_features(bar)
            >>> if features is not None:
            ...     assert features.dtype == np.float32
            ...     assert features.ndim == 1

        """
        # Buffer the bar FIRST (before any computation that might fail)
        # This ensures bars are always buffered, even if feature computation fails
        self._bar_buffer.append(bar)

        start_time = time.perf_counter()

        try:
            # Call user-provided compute function
            features = self._compute_function(bar)

            # Track performance
            feature_latency = (time.perf_counter() - start_time) * 1000  # milliseconds
            self._total_feature_time += feature_latency
            self._bars_processed += 1

            # Update warm-up status
            if not self._is_warmed_up and self._bars_processed >= self._warmup_bars_required:
                self._is_warmed_up = True
                self._logger.info(
                    f"Feature computation warmed up after {self._bars_processed} bars",
                )

            # Emit metrics
            instrument_id = str(bar.bar_type.instrument_id)
            if features is not None:
                self._feature_computation_counter.labels(
                    component="features",
                    instrument=instrument_id,
                    status="success",
                ).inc()
                self._feature_latency_histogram.labels(
                    component="features",
                    instrument=instrument_id,
                ).observe(
                    feature_latency / 1000
                )  # seconds
                self._feature_count += 1
            else:
                self._feature_computation_counter.labels(
                    component="features",
                    instrument=instrument_id,
                    status="not_ready",
                ).inc()

            # Check latency violation
            max_latency_ms = getattr(self._config, "max_feature_latency_ms", 5.0)
            if feature_latency > max_latency_ms:
                self._logger.warning(
                    f"Feature computation exceeded {max_latency_ms}ms: {feature_latency:.3f}ms",
                )
                if self._health_monitor:
                    self._health_monitor.update_latency_violation()

            # Add to feature window if not None
            if features is not None:
                self._feature_window.append(features)

            return features

        except Exception as e:
            self._logger.error(
                f"Feature computation failed: {e}",
                exc_info=True,
            )
            self._feature_computation_counter.labels(
                component="features",
                instrument=str(bar.bar_type.instrument_id),
                status="error",
            ).inc()
            return None

    def buffer_bar(self, bar: Bar) -> None:
        """
        Buffer Bar in lookback window.

        Uses deque with maxlen for automatic FIFO eviction.
        No explicit size check needed - deque handles it.

        Args:
            bar: Bar to buffer

        Example:
            >>> component = FeaturesComponent(...)
            >>> for bar in bars:
            ...     component.buffer_bar(bar)
            >>> assert len(component.get_buffered_bars()) <= lookback_window

        """
        self._bar_buffer.append(bar)

    def get_buffered_bars(self) -> list[Bar]:
        """
        Retrieve buffered bars from lookback window.

        Returns bars in order (oldest first, most recent last).

        Returns:
            List of buffered bars

        Example:
            >>> component = FeaturesComponent(...)
            >>> component.buffer_bar(bar1)
            >>> component.buffer_bar(bar2)
            >>> bars = component.get_buffered_bars()
            >>> assert bars[0] == bar1
            >>> assert bars[-1] == bar2

        """
        return list(self._bar_buffer)

    def is_warmed_up(self) -> bool:
        """
        Check if feature computation is warmed up.

        Warm-up requires processing minimum number of bars
        for indicator initialization.

        Returns:
            True if warmed up, False otherwise

        Example:
            >>> component = FeaturesComponent(...)
            >>> assert not component.is_warmed_up()
            >>> for bar in bars[:20]:
            ...     component.compute_features(bar)
            >>> assert component.is_warmed_up()

        """
        return self._is_warmed_up

    def validate_features(self, features: npt.NDArray[np.float32]) -> bool:
        """
        Validate features against FeatureRegistry schema.

        Fast validation (<50μs overhead):
        - Check feature count matches manifest
        - Check all values are finite
        - Optional: Check dtype matches

        Args:
            features: Feature array to validate

        Returns:
            True if valid, False otherwise

        Example:
            >>> component = FeaturesComponent(...)
            >>> features = np.zeros(20, dtype=np.float32)
            >>> valid = component.validate_features(features)
            >>> assert valid is True

        """
        start_time = time.perf_counter()

        try:
            # Fast checks first (most likely to fail)
            # Check finite values
            if not np.all(np.isfinite(features)):
                self._validation_counter.labels(
                    component="features",
                    result="non_finite",
                ).inc()
                return False

            # Check dtype
            if features.dtype != np.float32:
                self._validation_counter.labels(
                    component="features",
                    result="wrong_dtype",
                ).inc()
                return False

            # Check manifest length if available
            try:
                feature_set_id = getattr(self._config, "feature_set_id", None)
                if feature_set_id:
                    manifest = self._feature_registry.get_feature_manifest(feature_set_id)
                    if manifest is not None:
                        if len(manifest.feature_names) != len(features):
                            self._validation_counter.labels(
                                component="features",
                                result="length_mismatch",
                            ).inc()
                            self._logger.warning(
                                f"Feature length mismatch: expected {len(manifest.feature_names)}, "
                                f"got {len(features)}",
                            )
                            return False
            except Exception as e:
                self._logger.debug(
                    f"Manifest validation skipped: {e}",
                    exc_info=True,
                )

            # Validation passed
            self._validation_counter.labels(
                component="features",
                result="valid",
            ).inc()

            # Check validation overhead (<50μs requirement)
            validation_time_us = (time.perf_counter() - start_time) * 1_000_000
            if validation_time_us > 50:
                self._logger.debug(
                    f"Validation overhead {validation_time_us:.2f}μs exceeds 50μs target",
                )

            return True

        except Exception as e:
            self._logger.error(
                f"Feature validation failed: {e}",
                exc_info=True,
            )
            self._validation_counter.labels(
                component="features",
                result="error",
            ).inc()
            return False

    def persist_features_async(
        self,
        feature_set_id: str,
        instrument_id: str,
        features: dict[str, float],
        ts_event: int,
        ts_init: int,
    ) -> bool:
        """
        Persist features asynchronously via MLPersistenceWorker.

        Falls back to synchronous persistence if worker not available.

        Args:
            feature_set_id: Feature set identifier
            instrument_id: Instrument identifier
            features: Feature dict (name -> value)
            ts_event: Event timestamp (nanoseconds)
            ts_init: Initialization timestamp (nanoseconds)

        Returns:
            True if enqueued/written, False if queue full

        Example:
            >>> component = FeaturesComponent(...)
            >>> feature_dict = {"rsi_14": 0.5, "price_sma_20": 1.1050}
            >>> success = component.persist_features_async(
            ...     feature_set_id="default",
            ...     instrument_id="EUR/USD.SIM",
            ...     features=feature_dict,
            ...     ts_event=bar.ts_event,
            ...     ts_init=bar.ts_init,
            ... )
            >>> assert success is True

        """
        try:
            allow_sync_fallback = bool(
                getattr(self._config, "allow_sync_persistence_fallback", True),
            )
            # Async path: enqueue to persistence worker
            if self._persistence_worker is not None:
                enqueued = self._persistence_worker.enqueue_features(
                    feature_set_id=feature_set_id,
                    instrument_id=instrument_id,
                    features=features,
                    ts_event=ts_event,
                    ts_init=ts_init,
                )

                if enqueued:
                    self._persistence_counter.labels(
                        component="features",
                        mode="async",
                        result="enqueued",
                    ).inc()
                else:
                    self._persistence_counter.labels(
                        component="features",
                        mode="async",
                        result="queue_full",
                    ).inc()
                    self._logger.warning(
                        f"Persistence queue full - feature write dropped (instrument: {instrument_id})",
                    )

                return enqueued

            # Sync fallback: write directly to store
            else:
                if not allow_sync_fallback:
                    self._persistence_counter.labels(
                        component="features",
                        mode="sync",
                        result="disabled",
                    ).inc()
                    if not self._sync_fallback_disabled_logged:
                        self._logger.warning(
                            "Sync feature persistence disabled; dropping feature writes",
                        )
                        self._sync_fallback_disabled_logged = True
                    return False
                self._feature_store.write_features(
                    feature_set_id=feature_set_id,
                    instrument_id=instrument_id,
                    features=features,
                    ts_event=ts_event,
                    ts_init=ts_init,
                )
                self._persistence_counter.labels(
                    component="features",
                    mode="sync",
                    result="written",
                ).inc()
                return True

        except Exception as e:
            self._logger.error(
                f"Feature persistence failed: {e}",
                exc_info=True,
            )
            self._persistence_counter.labels(
                component="features",
                mode="sync" if self._persistence_worker is None else "async",
                result="error",
            ).inc()
            return False

    def cleanup(self) -> None:
        """
        Release feature computation resources.

        Clears buffers and resets state.

        Example:
            >>> component = FeaturesComponent(...)
            >>> component.cleanup()
            >>> assert len(component.get_buffered_bars()) == 0
            >>> assert not component.is_warmed_up()

        """
        self._bar_buffer.clear()
        self._feature_window.clear()
        self._is_warmed_up = False
        self._bars_processed = 0
        self._total_feature_time = 0.0
        self._feature_count = 0

        self._logger.debug("FeaturesComponent resources released")

    def get_statistics(self) -> dict[str, Any]:
        """
        Get feature computation statistics.

        Returns:
            Dictionary with performance stats

        Example:
            >>> component = FeaturesComponent(...)
            >>> stats = component.get_statistics()
            >>> assert "bars_processed" in stats
            >>> assert "average_latency_ms" in stats

        """
        avg_latency = self._total_feature_time / max(self._feature_count, 1)

        return {
            "bars_processed": self._bars_processed,
            "features_computed": self._feature_count,
            "is_warmed_up": self._is_warmed_up,
            "buffer_size": len(self._bar_buffer),
            "feature_window_size": len(self._feature_window),
            "average_latency_ms": avg_latency,
            "total_feature_time_ms": self._total_feature_time,
        }
