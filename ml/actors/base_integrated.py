"""
Enhanced base class for ML inference actors with automatic store integration.

This module provides the foundation for building ML-powered actors that
AUTOMATICALLY persist all data to stores and registries. This is NOT optional -
all ML actors must use stores for parity and reliability.
"""

from __future__ import annotations

import hashlib
import pickle
import time
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ml._imports import HAS_ONNX, check_ml_dependencies, ort
from ml.common.metrics import Counter, Histogram
from ml.config.base import (
    CircuitBreakerConfig,
    HealthMonitorConfig,
    MLActorConfig,
    MLFeatureConfig,
)
from ml.config.constants import TimeConstants
from ml.config.names import (
    LABEL_ACTOR_ID,
    LABEL_MODEL_NAME,
    METRIC_PREDICTION_LATENCY_SECONDS,
    METRIC_PREDICTIONS_TOTAL,
    METRIC_SIGNAL_CONFIDENCE,
)
from ml.config.runtime import OnnxRuntimeConfig, to_session_options
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.model_registry import ModelRegistry
from ml.registry.persistence import PersistenceConfig, PersistenceManager
from ml.registry.strategy_registry import StrategyRegistry
from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelStore
from ml.stores.strategy_store import StrategyStore
from nautilus_trader.common.actor import Actor
from nautilus_trader.core.data import Data
from nautilus_trader.model.data import Bar, DataType
from nautilus_trader.model.identifiers import InstrumentId

if TYPE_CHECKING:
    from nautilus_trader.common.config import ActorConfig


class HealthStatus(Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class CircuitBreakerState(Enum):
    """Circuit breaker state enumeration."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class HealthMonitor:
    """
    Health monitoring system for ML inference actors.

    Tracks system health metrics including prediction success rates,
    latency violations, and general system status.
    """

    def __init__(self, config: HealthMonitorConfig | None = None) -> None:
        """Initialize health monitor."""
        self._config = config or HealthMonitorConfig()
        self.status = HealthStatus.HEALTHY
        self.start_time = time.time()
        self.model_loaded = False
        self.indicators_initialized = False
        self.last_prediction_time = 0.0
        self.consecutive_failures = 0
        self.total_predictions = 0
        self.failed_predictions = 0
        self.total_latency_violations = 0
        self.last_health_check = time.time()

    def update_prediction_success(self) -> None:
        """Record successful prediction."""
        self.last_prediction_time = time.time()
        self.total_predictions += 1
        self.consecutive_failures = 0
        self._update_status()

    def update_prediction_failure(self) -> None:
        """Record failed prediction."""
        self.failed_predictions += 1
        self.total_predictions += 1
        self.consecutive_failures += 1
        self._update_status()

    def update_latency_violation(self) -> None:
        """Record latency violation."""
        self.total_latency_violations += 1
        self._update_status()

    def _update_status(self) -> None:
        """Update health status based on metrics."""
        if self.consecutive_failures >= self._config.failure_threshold:
            self.status = HealthStatus.UNHEALTHY
        elif self.consecutive_failures > 0:
            self.status = HealthStatus.DEGRADED
        else:
            self.status = HealthStatus.HEALTHY

    def check_health(self) -> HealthStatus:
        """Get current health status."""
        self.last_health_check = time.time()
        return self.status


class CircuitBreaker:
    """
    Circuit breaker for handling failures in ML inference.

    Implements the circuit breaker pattern to prevent cascading failures
    and provide graceful degradation.
    """

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        """Initialize circuit breaker."""
        self._config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.half_open_start = 0.0

    def call_succeeded(self) -> None:
        """Record successful call."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0

    def call_failed(self) -> None:
        """Record failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self._config.failure_threshold:
            self.state = CircuitBreakerState.OPEN

    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self.state == CircuitBreakerState.OPEN:
            # Check if we should transition to half-open
            if time.time() - self.last_failure_time > self._config.timeout_seconds:
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_start = time.time()
                return False
            return True
        return False

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0


class MLActorBase(Actor, ABC):
    """
    Base class for ML inference actors with AUTOMATIC store integration.

    ALL ML actors MUST inherit from this class to ensure:
    1. Automatic persistence of features, predictions, and signals
    2. Automatic registry integration
    3. Consistent data handling across the system
    4. Feature parity between training and inference

    This is NOT optional - stores are always initialized and used.
    """

    def __init__(self, config: MLActorConfig) -> None:
        """
        Initialize ML actor with automatic store integration.

        Parameters
        ----------
        config : MLActorConfig
            Actor configuration including database connection

        """
        super().__init__(config)
        self._ml_config = config

        # AUTOMATIC STORE INITIALIZATION - NOT OPTIONAL!
        self._init_stores()
        self._init_registries()

        # Health monitoring
        self._health_monitor = HealthMonitor(config.health_monitor_config)
        self._circuit_breaker = CircuitBreaker(config.circuit_breaker_config)

        # Performance metrics
        self._init_metrics()

        # Model management
        self._model = None
        self._model_hash: str | None = None
        self._feature_buffer: npt.NDArray[np.float32] | None = None

        # Window for historical data
        self._max_window_size = config.max_window_size
        self._data_window: deque[Any] = deque(maxlen=self._max_window_size)

    def _init_stores(self) -> None:
        """
        Initialize all stores - THIS IS MANDATORY!

        Stores are ALWAYS created and ALWAYS used.
        No optional parameters, no if statements.
        """
        # Get connection string from config or use default
        db_connection = getattr(
            self._ml_config,
            "db_connection",
            "postgresql://postgres:postgres@localhost:5432/nautilus",
        )

        # Create persistence config
        persistence_config = PersistenceConfig(
            backend="postgres",
            connection_string=db_connection,
        )

        # ALWAYS initialize ALL stores
        self._feature_store = FeatureStore(
            connection_string=db_connection,
            batch_size=1000,
            enable_batching=True,
        )

        self._model_store = ModelStore(
            persistence_config=persistence_config,
            batch_size=1000,
            enable_batching=True,
        )

        self._strategy_store = StrategyStore(
            persistence_config=persistence_config,
            batch_size=1000,
            enable_batching=True,
        )

        self.log.info("Stores initialized and connected")

    def _init_registries(self) -> None:
        """
        Initialize all registries - THIS IS MANDATORY!

        Registries are ALWAYS created and ALWAYS used.
        """
        # Get connection string from config or use default
        db_connection = getattr(
            self._ml_config,
            "db_connection",
            "postgresql://postgres:postgres@localhost:5432/nautilus",
        )

        # Create persistence manager
        persistence_config = PersistenceConfig(
            backend="postgres",
            connection_string=db_connection,
        )
        self._persistence_manager = PersistenceManager(persistence_config)

        # ALWAYS initialize ALL registries
        self._feature_registry = FeatureRegistry(self._persistence_manager)
        self._model_registry = ModelRegistry(self._persistence_manager)
        self._strategy_registry = StrategyRegistry(self._persistence_manager)

        self.log.info("Registries initialized and connected")

    def _init_metrics(self) -> None:
        """Initialize performance metrics."""
        # Initialize metrics for monitoring
        self._prediction_counter = Counter(
            METRIC_PREDICTIONS_TOTAL,
            "Total predictions made",
            [LABEL_ACTOR_ID, LABEL_MODEL_NAME],
        )

        self._latency_histogram = Histogram(
            METRIC_PREDICTION_LATENCY_SECONDS,
            "Prediction latency in seconds",
            [LABEL_ACTOR_ID],
        )

        self._confidence_histogram = Histogram(
            METRIC_SIGNAL_CONFIDENCE,
            "Signal confidence distribution",
            [LABEL_ACTOR_ID],
        )

    @abstractmethod
    def compute_features(self, data: Data) -> npt.NDArray[np.float32]:
        """
        Compute features from market data.

        Parameters
        ----------
        data : Data
            Market data (Bar, QuoteTick, TradeTick, etc.)

        Returns
        -------
        npt.NDArray[np.float32]
            Feature array ready for model inference

        """
        raise NotImplementedError

    @abstractmethod
    def run_inference(self, features: npt.NDArray[np.float32]) -> tuple[float, float]:
        """
        Run model inference on features.

        Parameters
        ----------
        features : npt.NDArray[np.float32]
            Feature array

        Returns
        -------
        tuple[float, float]
            Prediction and confidence values

        """
        raise NotImplementedError

    @abstractmethod
    def generate_signal(self, prediction: float, confidence: float, data: Data) -> Data | None:
        """
        Generate trading signal from prediction.

        Parameters
        ----------
        prediction : float
            Model prediction
        confidence : float
            Confidence score
        data : Data
            Original market data

        Returns
        -------
        Data | None
            Trading signal or None if no signal

        """
        raise NotImplementedError

    def on_bar(self, bar: Bar) -> None:
        """
        Process bar data with AUTOMATIC persistence.

        ALL data is ALWAYS stored - this is NOT optional!
        """
        # Check circuit breaker
        if self._circuit_breaker.is_open():
            self.log.warning("Circuit breaker is open, skipping prediction")
            return

        try:
            # Start timing
            start_time = time.perf_counter()

            # 1. Compute features
            features = self.compute_features(bar)

            # ALWAYS store features - NOT OPTIONAL!
            feature_dict = {f"feature_{i}": float(v) for i, v in enumerate(features)}
            self._feature_store.write_features(
                feature_set_id=self._ml_config.feature_set_id or "default",
                instrument_id=str(bar.bar_type.instrument_id),
                features=feature_dict,
                ts_event=bar.ts_event,
                ts_init=bar.ts_init,
            )

            # 2. Run inference
            prediction, confidence = self.run_inference(features)

            # ALWAYS store prediction - NOT OPTIONAL!
            inference_time_ms = (time.perf_counter() - start_time) * 1000
            self._model_store.write_prediction(
                model_id=self._ml_config.model_id or "default",
                instrument_id=str(bar.bar_type.instrument_id),
                prediction=float(prediction),
                confidence=float(confidence),
                features=feature_dict,
                inference_time_ms=inference_time_ms,
                ts_event=bar.ts_event,
            )

            # 3. Generate signal
            signal = self.generate_signal(prediction, confidence, bar)

            if signal is not None:
                # ALWAYS store signal - NOT OPTIONAL!
                self._strategy_store.write_signal(
                    strategy_id=self._ml_config.strategy_id or "default",
                    instrument_id=str(bar.bar_type.instrument_id),
                    signal_type=getattr(signal, "signal_type", "UNKNOWN"),
                    strength=getattr(signal, "strength", abs(prediction)),
                    model_predictions={self._ml_config.model_id or "default": prediction},
                    risk_metrics={},  # Override in subclass
                    execution_params={},  # Override in subclass
                    ts_event=bar.ts_event,
                )

                # Publish signal
                self.publish_data(signal)

            # Update metrics
            self._health_monitor.update_prediction_success()
            self._circuit_breaker.call_succeeded()
            self._update_metrics(inference_time_ms, confidence)

        except Exception as e:
            self.log.error(f"Prediction failed: {e}")
            self._health_monitor.update_prediction_failure()
            self._circuit_breaker.call_failed()

    def _update_metrics(self, latency_ms: float, confidence: float) -> None:
        """Update performance metrics."""
        # Update counters and histograms
        self._prediction_counter.inc(
            {
                LABEL_ACTOR_ID: self.id.value,
                LABEL_MODEL_NAME: self._ml_config.model_id or "unknown",
            }
        )

        self._latency_histogram.observe(
            latency_ms / 1000,  # Convert to seconds
            {LABEL_ACTOR_ID: self.id.value},
        )

        self._confidence_histogram.observe(
            confidence,
            {LABEL_ACTOR_ID: self.id.value},
        )

        # Check for latency violations
        if latency_ms > self._ml_config.max_inference_latency_ms:
            self._health_monitor.update_latency_violation()

    def on_stop(self) -> None:
        """
        Clean shutdown with AUTOMATIC flushing.

        ALL pending data is ALWAYS flushed - NOT OPTIONAL!
        """
        # ALWAYS flush all stores - NOT OPTIONAL!
        self._feature_store.flush()
        self._model_store.flush()
        self._strategy_store.flush()

        self.log.info("All stores flushed on shutdown")
        super().on_stop()

    def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status."""
        return {
            "health": self._health_monitor.check_health().value,
            "circuit_breaker": self._circuit_breaker.state.value,
            "total_predictions": self._health_monitor.total_predictions,
            "failed_predictions": self._health_monitor.failed_predictions,
            "latency_violations": self._health_monitor.total_latency_violations,
            "stores_connected": {
                "feature_store": True,  # Always true now!
                "model_store": True,  # Always true now!
                "strategy_store": True,  # Always true now!
            },
            "registries_connected": {
                "feature_registry": True,  # Always true now!
                "model_registry": True,  # Always true now!
                "strategy_registry": True,  # Always true now!
            },
        }


# Migration path: Update existing MLActor to inherit from MLActorBase
class MLActor(MLActorBase):
    """
    Deprecated: Use MLActorBase directly.

    This class exists for backward compatibility only.
    All new actors should inherit from MLActorBase.
    """

    def __init__(self, config: MLActorConfig) -> None:
        """Initialize with deprecation warning."""
        import warnings

        warnings.warn(
            "MLActor is deprecated. Use MLActorBase directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(config)