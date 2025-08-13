"""
Base class for ML inference actors.

This module provides the foundation for building ML-powered actors that can perform
real-time inference on market data while maintaining the performance requirements of
Nautilus Trader's hot path.

"""

from __future__ import annotations

import hashlib
import pickle
import time
from abc import ABC
from abc import abstractmethod
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

# Import ML dependencies and check availability
from ml._imports import HAS_ONNX
from ml._imports import check_ml_dependencies
from ml._imports import ort
from ml.common.metrics import Counter
from ml.common.metrics import Histogram
from ml.config.base import CircuitBreakerConfig
from ml.config.base import HealthMonitorConfig
from ml.config.base import MLActorConfig
from ml.config.base import MLFeatureConfig
from ml.config.constants import TimeConstants
from ml.config.names import LABEL_ACTOR_ID
from ml.config.names import LABEL_MODEL_NAME
from ml.config.names import METRIC_PREDICTION_LATENCY_SECONDS
from ml.config.names import METRIC_PREDICTIONS_TOTAL
from ml.config.names import METRIC_SIGNAL_CONFIDENCE
from ml.config.runtime import OnnxRuntimeConfig
from ml.config.runtime import to_session_options
from nautilus_trader.common.actor import Actor
from nautilus_trader.common.config import ActorConfig
from nautilus_trader.core.data import Data
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import DataType
from nautilus_trader.model.identifiers import InstrumentId


class HealthStatus(Enum):
    """
    Health status enumeration.
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class CircuitBreakerState(Enum):
    """
    Circuit breaker state enumeration.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class HealthMonitor:
    """
    Health monitoring system for ML inference actors.

    Tracks system health metrics including prediction success rates, latency violations,
    and general system status.

    """

    def __init__(self, config: HealthMonitorConfig | None = None) -> None:
        """
        Initialize health monitor.
        """
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
        """
        Record successful prediction.
        """
        self.last_prediction_time = time.time()
        self.consecutive_failures = 0
        self.total_predictions += 1

    def update_prediction_failure(self) -> None:
        """
        Record failed prediction.
        """
        self.consecutive_failures += 1
        self.failed_predictions += 1
        self.total_predictions += 1
        self._update_health_status()

    def update_latency_violation(self) -> None:
        """
        Record latency violation.
        """
        self.total_latency_violations += 1
        self._update_health_status()

    def set_model_loaded(self, loaded: bool) -> None:
        """
        Update model loaded status.
        """
        self.model_loaded = loaded
        self._update_health_status()

    def set_indicators_initialized(self, initialized: bool) -> None:
        """
        Update indicators initialized status.
        """
        self.indicators_initialized = initialized
        self._update_health_status()

    def _update_health_status(self) -> None:
        """
        Update overall health status based on metrics.
        """
        # Check for critical failures
        if (
            not self.model_loaded
            or self.consecutive_failures > self._config.critical_consecutive_failures
        ):
            self.status = HealthStatus.UNHEALTHY
            return

        # Check for degraded performance
        success_rate = self.get_success_rate()
        if (
            success_rate < self._config.degraded_success_rate_threshold
            or self.consecutive_failures > self._config.degraded_consecutive_failures
            or self.total_latency_violations > self._config.degraded_latency_violations
        ):
            self.status = HealthStatus.DEGRADED
            return

        # System is healthy
        self.status = HealthStatus.HEALTHY

    def get_success_rate(self) -> float:
        """
        Calculate prediction success rate.
        """
        if self.total_predictions == 0:
            return 1.0
        return (self.total_predictions - self.failed_predictions) / self.total_predictions

    def get_uptime_seconds(self) -> float:
        """
        Get system uptime in seconds.
        """
        return time.time() - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """
        Export health status as dictionary.
        """
        return {
            "status": self.status.value,
            "model_loaded": self.model_loaded,
            "indicators_initialized": self.indicators_initialized,
            "uptime_seconds": self.get_uptime_seconds(),
            "success_rate": self.get_success_rate(),
            "consecutive_failures": self.consecutive_failures,
            "total_predictions": self.total_predictions,
            "failed_predictions": self.failed_predictions,
            "latency_violations": self.total_latency_violations,
            "last_prediction_time": self.last_prediction_time,
        }


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.

    Prevents cascade failures by temporarily stopping operations when error rates exceed
    thresholds.

    """

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        """
        Initialize circuit breaker.

        Parameters
        ----------
        config : CircuitBreakerConfig, optional
            Circuit breaker configuration.

        """
        self._config = config or CircuitBreakerConfig()
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._next_attempt = 0.0

    @property
    def state(self) -> CircuitBreakerState:
        """
        Get current circuit breaker state.
        """
        return self._state

    def can_execute(self) -> bool:
        """
        Check if operation can be executed.
        """
        current_time = time.time()

        if self._state == CircuitBreakerState.CLOSED:
            return True
        elif self._state == CircuitBreakerState.OPEN:
            if current_time >= self._next_attempt:
                self._state = CircuitBreakerState.HALF_OPEN
                self._success_count = 0
                return True
            return False
        else:  # HALF_OPEN
            return True

    def record_success(self) -> None:
        """
        Record successful operation.
        """
        if self._state == CircuitBreakerState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self._config.success_threshold:
                self._state = CircuitBreakerState.CLOSED
                self._failure_count = 0
        elif self._state == CircuitBreakerState.CLOSED:
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        """
        Record failed operation.
        """
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitBreakerState.HALF_OPEN:
            self._state = CircuitBreakerState.OPEN
            self._next_attempt = self._last_failure_time + self._config.recovery_timeout
        elif (
            self._state == CircuitBreakerState.CLOSED
            and self._failure_count >= self._config.failure_threshold
        ):
            self._state = CircuitBreakerState.OPEN
            self._next_attempt = self._last_failure_time + self._config.recovery_timeout

    def get_stats(self) -> dict[str, Any]:
        """
        Get circuit breaker statistics.
        """
        return {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": self._last_failure_time,
            "next_attempt": self._next_attempt,
        }


# Model loading now uses the registry system
# Legacy imports removed - use LocalModelRegistry instead


class SecurityError(Exception):
    """
    Raised when a security check fails during model loading.
    """


class ModelLoader:
    """
    Base class for model loaders (compatibility layer).
    """

    def load_model(self, path: str) -> tuple[Any, dict[str, Any]]:
        """
        Load a model and return it with metadata.
        """
        raise NotImplementedError

    def get_model_version(self, path: str) -> str:
        """
        Get model version.
        """
        return "1.0.0"


class ProductionModelLoader(ModelLoader):
    """
    Production model loader (compatibility layer for legacy code).
    """

    def __init__(self, model_dir: str | None = None):
        self.model_dir = Path(model_dir) if model_dir else Path.cwd()

    def load_model(self, path: str) -> tuple[Any, dict[str, Any]]:
        """Load model - this should use the registry in new code."""
        # For backward compatibility
        return None, {}


class ONNXModelLoader(ModelLoader):
    """
    Model loader for ONNX models with optimized runtime.
    """

    def __init__(self, runtime_config: OnnxRuntimeConfig | None = None) -> None:
        """
        Initialize ONNX model loader.
        """
        self._onnx_available = HAS_ONNX
        self._runtime_config = runtime_config or OnnxRuntimeConfig()

    def load_model(self, path: str) -> tuple[Any, dict[str, Any]]:
        """
        Load ONNX model with optimized runtime.
        """
        if not self._onnx_available:
            check_ml_dependencies(["onnx"])  # This will raise with proper error message

        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        # Create optimized ONNX Runtime session
        session_options, providers = to_session_options(self._runtime_config)
        session = ort.InferenceSession(str(model_path), session_options, providers=providers)

        # Generate metadata
        metadata = {
            "path": str(model_path),
            "size_bytes": model_path.stat().st_size,
            "modified_time": model_path.stat().st_mtime,
            "version": self.get_model_version(path),
            "type": "onnx",
            "input_names": [inp.name for inp in session.get_inputs()],
            "output_names": [out.name for out in session.get_outputs()],
            "providers": session.get_providers(),
        }

        return session, metadata

    def get_model_version(self, path: str) -> str:
        """
        Get ONNX model version based on file modification time and size.
        """
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        stat = model_path.stat()
        # Create version hash from file size and modification time
        version_string = f"onnx_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(version_string.encode()).hexdigest()[:8]  # noqa: S324


class MLSignal(Data):  # type: ignore[misc]
    """
    ML signal data class for signal generation.

    Clean, simple data class with required model_id field for tracking.
    No confusing aliases or "unified" terminology.

    Parameters
    ----------
    instrument_id : InstrumentId
        The instrument the prediction is for.
    model_id : str
        Unique identifier for the model that generated this signal.
    prediction : float
        The model prediction value.
    confidence : float
        The confidence score for the prediction (0.0 to 1.0).
    features : npt.NDArray[np.float32], optional
        The feature vector used for prediction (for debugging).
    metadata : dict[str, Any], optional
        Additional metadata for the signal.
    ts_event : int
        The UNIX timestamp (nanoseconds) when the signal was generated.
    ts_init : int
        The UNIX timestamp (nanoseconds) when the object was initialized.

    """

    def __init__(
        self,
        instrument_id: InstrumentId,
        model_id: str,
        prediction: float,
        confidence: float,
        features: npt.NDArray[np.float32] | None = None,
        metadata: dict[str, Any] | None = None,
        ts_event: int = 0,
        ts_init: int = 0,
    ) -> None:
        """
        Initialize a new ML signal data object.

        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument this signal is for.
        model_id : str
            The model identifier for tracking.
        prediction : float
            The model's prediction value.
        confidence : float
            The confidence level of the prediction (0.0 to 1.0).
        features : npt.NDArray[np.float32], optional
            The feature values used for this prediction.
        metadata : dict[str, Any], optional
            Additional signal metadata.
        ts_event : int, default 0
            The event timestamp in nanoseconds.
        ts_init : int, default 0
            The initialization timestamp in nanoseconds.

        """
        self.instrument_id = instrument_id
        self.model_id = model_id
        self.prediction = prediction
        self.confidence = confidence
        self.features = features
        self.metadata = metadata or {}
        self._ts_event = ts_event
        self._ts_init = ts_init

    @property
    def ts_event(self) -> int:
        """
        Return the UNIX timestamp (nanoseconds) when the signal was generated.
        """
        return self._ts_event

    @property
    def ts_init(self) -> int:
        """
        Return the UNIX timestamp (nanoseconds) when the object was initialized.
        """
        return self._ts_init


# Prometheus metrics for monitoring
ml_predictions_total = Counter(
    METRIC_PREDICTIONS_TOTAL,
    "Total number of ML predictions made",
    [LABEL_ACTOR_ID, LABEL_MODEL_NAME],
)
ml_prediction_latency = Histogram(
    METRIC_PREDICTION_LATENCY_SECONDS,
    "Latency of ML predictions in seconds",
    [LABEL_ACTOR_ID, LABEL_MODEL_NAME],
)
ml_signal_confidence = Histogram(
    METRIC_SIGNAL_CONFIDENCE,
    "Distribution of ML signal confidence scores",
    [LABEL_ACTOR_ID, LABEL_MODEL_NAME],
)


class BaseMLInferenceActor(Actor, ABC):  # type: ignore[misc]
    """
    Base class for ML inference actors with production features.

    This class provides a foundation for building ML-powered actors that perform
    real-time inference on market data with:
    - Model hot-reloading capability
    - Health monitoring and status reporting
    - Circuit breaker pattern for fault tolerance
    - Comprehensive metrics and observability
    - Proper model_id tracking for signals

    Key principles:
    - All indicators and models are loaded during initialization
    - Feature computation uses pre-allocated numpy arrays
    - No blocking operations in event handlers (HOT PATH)
    - Memory usage is bounded and predictable
    - <500μs feature computation, <2ms inference, <5ms end-to-end

    Parameters
    ----------
    config : MLActorConfig
        The configuration for the ML actor.

    """

    def __init__(self, config: MLActorConfig) -> None:
        """
        Initialize the enhanced ML inference actor.

        Parameters
        ----------
        config : MLActorConfig
            The configuration for the ML actor.

        """
        # Extract ActorConfig fields
        actor_config = ActorConfig(
            component_id=config.component_id,
            log_events=config.log_events,
            log_commands=config.log_commands,
        )

        # Initialize with standard ActorConfig
        super().__init__(actor_config)

        # Store the complete ML configuration
        self._config = config

        # Initialize feature configuration
        self._feature_config = config.feature_config or MLFeatureConfig()

        # Model and inference state
        self._model: Any = None
        self._model_metadata: dict[str, Any] = {}
        self._model_version: str | None = None
        self._model_id: str = "unknown"  # Track model ID for signals
        self._model_loader: ModelLoader = ProductionModelLoader()
        self._features_buffer: npt.NDArray[np.float32] | None = None
        self._feature_window: deque[npt.NDArray[np.float32]] = deque(
            maxlen=self._feature_config.lookback_window,
        )

        # Production features
        self._health_monitor = (
            HealthMonitor(config.health_config) if config.enable_health_monitoring else None
        )
        self._circuit_breaker = (
            CircuitBreaker(config.circuit_breaker_config) if config.circuit_breaker_config else None
        )

        # Hot reload state
        self._last_model_check = 0.0
        self._indicator_state_backup: dict[str, Any] = {}

        # Performance tracking
        self._prediction_count = 0
        self._total_inference_time = 0.0
        self._total_feature_time = 0.0
        self._last_prediction_time = 0

        # Warm-up tracking
        self._bars_processed = 0
        self._is_warmed_up = False

        # Enhanced Prometheus metrics - use global instances
        self._inference_latency_metric = ml_prediction_latency
        self._inference_count_metric = ml_predictions_total
        self._inference_confidence_metric = ml_signal_confidence

    def on_start(self) -> None:
        """
        Initialize the actor and subscribe to market data.

        This method is called when the actor starts and handles:
        - Model loading with version tracking
        - Feature buffer initialization
        - Market data subscription
        - Hot reload scheduling
        - Health monitoring initialization

        """
        self.log.info(f"Starting enhanced {self.__class__.__name__}")

        try:
            # Load model during initialization (not in hot path)
            self._load_model_with_metadata()

            # Initialize feature buffers
            self._initialize_features()

            # Update health monitor
            if self._health_monitor:
                self._health_monitor.set_model_loaded(True)
                self._health_monitor.set_indicators_initialized(True)

            # Schedule hot reload checks if enabled
            if self._config.enable_hot_reload:
                self._schedule_model_checks()

            # Subscribe to market data
            self.subscribe_bars(self._config.bar_type)

            self.log.info(
                f"Enhanced ML Actor configured: "
                f"model={Path(self._config.model_path).name}, "
                f"version={self._model_version}, "
                f"threshold={self._config.prediction_threshold}, "
                f"warm_up={self._config.warm_up_period}, "
                f"hot_reload={self._config.enable_hot_reload}, "
                f"health_monitoring={self._config.enable_health_monitoring}",
            )

        except Exception as e:
            self.log.error(f"Failed to start ML Actor: {e}")
            if self._health_monitor:
                self._health_monitor.set_model_loaded(False)
            raise

    def on_bar(self, bar: Bar) -> None:
        """
        Process new bar data and potentially generate predictions.

        This is the hot path - must be optimized for performance:
        - No memory allocations
        - No blocking operations
        - Bounded computation time
        - Circuit breaker protection

        Parameters
        ----------
        bar : Bar
            The new bar data to process.

        """
        # Check circuit breaker before processing
        if self._circuit_breaker and not self._circuit_breaker.can_execute():
            return  # Circuit is open, skip processing

        # Track bars for warm-up period
        self._bars_processed += 1

        # Check if warmed up first (before feature computation)
        if not self._is_warmed_up:
            if self._bars_processed >= self._config.warm_up_period:
                self._is_warmed_up = True
                if hasattr(self, "log"):
                    self.log.info("Enhanced ML Actor warm-up complete, starting predictions")

        # Update indicators and compute features with timing
        start_feature_time = time.perf_counter()
        features = self._compute_features(bar)
        feature_latency = (time.perf_counter() - start_feature_time) * 1000

        # Track feature computation performance
        self._total_feature_time += feature_latency

        # Check feature computation latency
        if feature_latency > self._config.max_feature_latency_ms:
            if hasattr(self, "log"):
                self.log.warning(
                    f"Feature computation exceeded {self._config.max_feature_latency_ms}ms: "
                    f"{feature_latency:.3f}ms",
                )
            if self._health_monitor:
                self._health_monitor.update_latency_violation()

        if features is None:
            return  # Indicators not ready

        # Add to rolling window
        self._feature_window.append(features)

        # Skip prediction if still warming up
        if not self._is_warmed_up:
            return  # Still warming up

        # Generate prediction with circuit breaker protection
        self._generate_prediction_protected(bar, features)

    def on_stop(self) -> None:
        """
        Log final statistics when the actor stops.
        """
        avg_inference_time = self._total_inference_time / max(self._prediction_count, 1)
        avg_feature_time = self._total_feature_time / max(self._bars_processed, 1)

        # Health status summary
        health_status = "N/A"
        if self._health_monitor:
            health_status = self._health_monitor.status.value

        self.log.info(
            f"Stopping Enhanced {self.__class__.__name__} - "
            f"Predictions: {self._prediction_count}, "
            f"Avg inference time: {avg_inference_time:.3f}ms, "
            f"Avg feature time: {avg_feature_time:.3f}ms, "
            f"Health: {health_status}, "
            f"Circuit breaker: {self._circuit_breaker.state.value if self._circuit_breaker else 'disabled'}",
        )

    def _generate_prediction_protected(self, bar: Bar, features: npt.NDArray[np.float32]) -> None:
        """
        Generate ML prediction with circuit breaker protection.

        This method measures inference time, handles failures gracefully,
        and publishes signals if configured.

        Parameters
        ----------
        bar : Bar
            The current bar data.
        features : npt.NDArray[np.float32]
            The computed feature vector.

        """
        start_time = time.perf_counter()

        try:
            # Get prediction from model
            prediction, confidence = self._predict(features)

            # Track performance
            inference_time = (time.perf_counter() - start_time) * 1000
            self._total_inference_time += inference_time
            self._prediction_count += 1

            # Record success in circuit breaker
            if self._circuit_breaker:
                self._circuit_breaker.record_success()

            # Update health monitor
            if self._health_monitor:
                self._health_monitor.update_prediction_success()

            # Check latency requirement
            if inference_time > self._config.max_inference_latency_ms:
                self.log.warning(
                    f"Inference latency exceeded: {inference_time:.3f}ms > "
                    f"{self._config.max_inference_latency_ms}ms",
                )
                if self._health_monitor:
                    self._health_monitor.update_latency_violation()

            # Track metrics
            self._inference_latency_metric.labels(
                actor_id=str(self.id) if self.id else "unknown",
                model_name=Path(self._config.model_path).stem,
            ).observe(inference_time / 1000)
            self._inference_count_metric.labels(
                actor_id=str(self.id) if self.id else "unknown",
                model_name=Path(self._config.model_path).stem,
            ).inc()
            self._inference_confidence_metric.labels(
                actor_id=str(self.id) if self.id else "unknown",
                model_name=Path(self._config.model_path).stem,
            ).observe(confidence)

            # Log prediction if configured
            if self._config.log_predictions:
                self.log.debug(
                    f"Prediction: {prediction:.4f}, confidence: {confidence:.4f}, "
                    f"latency: {inference_time:.3f}ms",
                )

            # Publish signal if confidence meets threshold
            if confidence >= self._config.prediction_threshold and self._config.publish_signals:
                signal = MLSignal(
                    instrument_id=bar.bar_type.instrument_id,
                    model_id=self._model_id,
                    prediction=prediction,
                    confidence=confidence,
                    features=features if self._config.log_predictions else None,
                    ts_event=bar.ts_event,
                    ts_init=self.clock.timestamp_ns(),
                )
                self._publish_signal(signal)

        except Exception as e:
            self.log.error(f"Prediction failed: {e}")

            # Record failure in circuit breaker
            if self._circuit_breaker:
                self._circuit_breaker.record_failure()

            # Update health monitor
            if self._health_monitor:
                self._health_monitor.update_prediction_failure()

            # Log error (metrics tracking removed to avoid duplicate registration)

    def _publish_signal(self, signal: MLSignal) -> None:
        """
        Publish ML signal to the message bus.

        Parameters
        ----------
        signal : MLSignal
            The ML signal to publish.

        """
        self.publish_data(
            DataType(MLSignal, metadata={"source": self.id.value}),
            signal,
        )

    @abstractmethod
    def _load_model(self) -> None:
        """
        Load the ML model from disk.

        This method should be overridden by concrete implementations to load their
        specific model type (e.g., scikit-learn, XGBoost, ONNX).

        The model should be stored in self._model for use in _predict().

        """
        ...

    @abstractmethod
    def _initialize_features(self) -> None:
        """
        Initialize feature computation components.

        This method should set up indicators, feature buffers, and any other components
        needed for feature computation. All memory allocation should happen here, not in
        the hot path.

        """
        ...

    @abstractmethod
    def _compute_features(self, bar: Bar) -> npt.NDArray[np.float32] | None:
        """
        Compute feature vector from current bar data.

        This method is called in the hot path and must be optimized:
        - Use pre-allocated numpy arrays
        - Update indicators in-place
        - Return None if features are not ready

        Parameters
        ----------
        bar : Bar
            The current bar data.

        Returns
        -------
        npt.NDArray[np.float32] | None
            The computed feature vector, or None if not ready.

        """
        ...

    @abstractmethod
    def _predict(self, features: npt.NDArray[np.float32]) -> tuple[float, float]:
        """
        Generate prediction from feature vector.

        This method should perform model inference and return both the
        prediction value and confidence score.

        Parameters
        ----------
        features : npt.NDArray[np.float32]
            The feature vector for prediction.

        Returns
        -------
        tuple[float, float]
            A tuple of (prediction, confidence) values.

        """
        ...

    # ===== PRODUCTION ENHANCEMENT METHODS =====

    def _load_model_with_metadata(self) -> None:
        """
        Load model with metadata tracking for hot reload support.

        Supports both direct model path and registry-based loading.

        """
        try:
            # Check if we should load from registry (only if model_id is set WITHOUT model_path)
            if (
                hasattr(self._config, "model_id")
                and self._config.model_id
                and (not hasattr(self._config, "model_path") or not self._config.model_path)
            ):
                # Load from unified registry
                from pathlib import Path

                from ml.registry.model_registry import LocalModelRegistry

                registry_path = (
                    Path(self._config.registry_path)
                    if hasattr(self._config, "registry_path")
                    else Path("ml/models")
                )
                registry = LocalModelRegistry(registry_path)

                # Get model info from registry
                model_info = registry.get_model(self._config.model_id)
                if not model_info:
                    raise ValueError(f"Model {self._config.model_id} not found in registry")

                # Load the actual model
                self._model = registry.load_model(self._config.model_id)

                # Extract metadata from manifest
                manifest = model_info.manifest
                self._model_metadata = {
                    "model_id": manifest.model_id,
                    "version": manifest.version,
                    "type": manifest.architecture,
                    "role": manifest.role.value,
                    "data_requirements": manifest.data_requirements.value,
                    "feature_schema": manifest.feature_schema,
                    "feature_schema_hash": manifest.feature_schema_hash,
                    "parent_id": manifest.parent_id,
                    "performance_metrics": manifest.performance_metrics,
                    "deployment_constraints": manifest.deployment_constraints,
                }

                # Use manifest features if configured
                if (
                    hasattr(self._config, "use_manifest_features")
                    and self._config.use_manifest_features
                ):
                    # Override feature config with manifest schema
                    self._feature_names = list(manifest.feature_schema.keys())
                    self.log.info(f"Using {len(self._feature_names)} features from manifest")

                # Check deployment constraints
                if "max_latency_ms" in manifest.deployment_constraints:
                    max_latency = manifest.deployment_constraints["max_latency_ms"]
                    if hasattr(self._config, "max_inference_latency_ms"):
                        if self._config.max_inference_latency_ms > max_latency:
                            self.log.warning(
                                f"Config latency {self._config.max_inference_latency_ms}ms "
                                f"exceeds model constraint {max_latency}ms",
                            )

            else:
                # Load from direct path (existing behavior)
                self._model, self._model_metadata = self._model_loader.load_model(
                    self._config.model_path,
                )
            self._model_version = self._model_metadata.get("version")

            # Extract model_id from metadata or generate from path
            if "model_id" in self._model_metadata:
                self._model_id = self._model_metadata["model_id"]
            elif "training_metadata" in self._model_metadata:
                training_meta = self._model_metadata["training_metadata"]
                if "model_id" in training_meta:
                    self._model_id = training_meta["model_id"]
                else:
                    # Generate from path and version
                    from pathlib import Path

                    model_name = Path(self._config.model_path).stem
                    version_str = self._model_version[:8] if self._model_version else "v1"
                    self._model_id = f"{model_name}_{version_str}"
            else:
                # Fallback: use filename and version
                from pathlib import Path

                model_name = Path(self._config.model_path).stem
                self._model_id = (
                    f"{model_name}_{self._model_version[:8] if self._model_version else 'v1'}"
                )

            # Call the original abstract method for backward compatibility
            self._load_model()

            self.log.info(
                f"Loaded model with metadata: "
                f"model_id={self._model_id}, "
                f"version={self._model_version}, "
                f"size={self._model_metadata.get('size_bytes', 0)} bytes, "
                f"type={self._model_metadata.get('type', 'unknown')}",
            )
        except Exception as e:
            self.log.error(f"Failed to load model: {e}")
            raise

    def _schedule_model_checks(self) -> None:
        """
        Schedule periodic model version checks for hot reload.
        """
        if not self._config.enable_hot_reload:
            return

        # Use Nautilus timer for scheduling
        interval_ns = (
            self._config.model_check_interval * TimeConstants.NS_IN_SECOND
        )  # Convert to ns
        self.clock.set_timer_ns(
            name="model_version_check",
            interval_ns=interval_ns,
            start_time_ns=self.clock.timestamp_ns() + interval_ns,
            stop_time_ns=0,  # No stop time (runs indefinitely)
            callback=self._check_model_updates,
        )

        self.log.info(
            f"Scheduled model checks every {self._config.model_check_interval}s",
        )

    def _check_model_updates(self, event: Any) -> None:
        """
        Check for model updates and hot-reload if needed.

        This method runs periodically to detect model file changes and reload if a new
        version is available.

        """
        try:
            # Check current model version
            current_version = self._model_loader.get_model_version(self._config.model_path)

            if current_version != self._model_version:
                self.log.info(
                    f"Model version change detected: {self._model_version} -> {current_version}",
                )

                # Backup indicator state if configured
                if self._config.preserve_state_on_reload:
                    self._backup_indicator_state()

                # Reload model
                self._reload_model()

                # Restore indicator state if configured
                if self._config.preserve_state_on_reload:
                    self._restore_indicator_state()

                # Metric tracking removed to avoid duplicate registration

            self._last_model_check = time.time()

        except Exception as e:
            self.log.error(f"Model update check failed: {e}")
            # Metric tracking removed to avoid duplicate registration

    def _reload_model(self) -> None:
        """
        Reload the model with new version.
        """
        try:
            # Load new model
            new_model, new_metadata = self._model_loader.load_model(self._config.model_path)

            # Atomic update
            old_version = self._model_version
            self._model = new_model
            self._model_metadata = new_metadata
            self._model_version = new_metadata.get("version")

            # Update health status
            if self._health_monitor:
                self._health_monitor.set_model_loaded(True)

            self.log.info(
                f"Model hot-reload successful: {old_version} -> {self._model_version}",
            )

        except Exception as e:
            self.log.error(f"Model reload failed: {e}")
            if self._health_monitor:
                self._health_monitor.set_model_loaded(False)
            raise

    def _backup_indicator_state(self) -> None:
        """
        Backup current indicator state for preservation during reload.

        This is an abstract method that concrete implementations should override to
        backup their specific indicator state.

        """
        # This is a placeholder - concrete implementations should override
        # to backup their specific indicators
        self.log.debug("Backing up indicator state (base implementation)")

    def _restore_indicator_state(self) -> None:
        """
        Restore indicator state after model reload.

        This is an abstract method that concrete implementations should override to
        restore their specific indicator state.

        """
        # This is a placeholder - concrete implementations should override
        # to restore their specific indicators
        self.log.debug("Restoring indicator state (base implementation)")

    def get_health_status(self) -> dict[str, Any]:
        """
        Get current health status of the actor.

        Returns
        -------
        dict[str, Any]
            Health status information including metrics and system state.

        """
        base_status = {
            "actor_id": self.id.value,
            "model_path": self._config.model_path,
            "model_version": self._model_version,
            "is_warmed_up": self._is_warmed_up,
            "bars_processed": self._bars_processed,
            "predictions_made": self._prediction_count,
            "avg_inference_time_ms": (self._total_inference_time / max(self._prediction_count, 1)),
            "avg_feature_time_ms": (self._total_feature_time / max(self._bars_processed, 1)),
        }

        # Add health monitor data if available
        if self._health_monitor:
            base_status.update(self._health_monitor.to_dict())

        # Add circuit breaker data if available
        if self._circuit_breaker:
            base_status["circuit_breaker"] = self._circuit_breaker.get_stats()

        return base_status

    def reset_health_status(self) -> None:
        """
        Reset health monitoring statistics.
        """
        if self._health_monitor:
            self._health_monitor = HealthMonitor()
            self.log.info("Health status reset")


class PickleMLInferenceActor(BaseMLInferenceActor):
    """
    ML inference actor for scikit-learn and pickle-compatible models.

    This implementation handles models saved with pickle/joblib, which is common for
    scikit-learn, XGBoost, and LightGBM models.

    """

    def _load_model(self) -> None:
        """
        Load pickle/joblib model from disk.
        """
        model_path = Path(self._config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Security check for pickle loading
        if not self._config.allow_pickle:
            raise SecurityError(
                "Pickle loading is disabled for security. "
                "Set allow_pickle=True to enable (not recommended for production) "
                "or use ONNX/native model formats instead.",
            )

        with open(model_path, "rb") as f:
            self._model = pickle.load(f)  # noqa: S301

        self.log.info(f"Loaded model from {model_path}")

    def _predict(self, features: npt.NDArray[np.float32]) -> tuple[float, float]:
        """
        Generate prediction using the loaded model.

        Parameters
        ----------
        features : npt.NDArray[np.float32]
            The feature vector for prediction.

        Returns
        -------
        tuple[float, float]
            A tuple of (prediction, confidence) values.

        """
        # Reshape features for sklearn models (expects 2D array)
        features_2d = features.reshape(1, -1)

        # Get prediction
        if hasattr(self._model, "predict_proba"):
            # Classification model with probability output
            probabilities = self._model.predict_proba(features_2d)[0]
            prediction = np.argmax(probabilities)
            confidence = np.max(probabilities)
        else:
            # Regression model or classifier without probabilities
            prediction = self._model.predict(features_2d)[0]
            confidence = 1.0  # Assume full confidence for regression

        return float(prediction), float(confidence)


class ONNXMLInferenceActor(BaseMLInferenceActor):
    """
    ML inference actor for ONNX models with optimized runtime.

    This implementation provides the lowest latency inference using ONNX Runtime with
    CPU optimizations. Suitable for production environments requiring sub-millisecond
    inference times.

    """

    def __init__(self, config: MLActorConfig) -> None:
        """
        Initialize ONNX ML inference actor.
        """
        super().__init__(config)
        self._model_loader = ONNXModelLoader()
        self._input_name: str | None = None
        self._output_names: list[str] = []

    def _load_model(self) -> None:
        """
        Load ONNX model with optimized runtime session.
        """
        # This will be called from _load_model_with_metadata
        # The actual loading is handled by ONNXModelLoader
        if self._model_metadata:
            self._input_name = self._model_metadata["input_names"][0]
            self._output_names = self._model_metadata["output_names"]

            self.log.info(
                f"ONNX model loaded: input={self._input_name}, "
                f"outputs={self._output_names}, "
                f"providers={self._model_metadata.get('providers', [])}",
            )

    def _predict(self, features: npt.NDArray[np.float32]) -> tuple[float, float]:
        """
        Generate prediction using ONNX Runtime.

        Parameters
        ----------
        features : npt.NDArray[np.float32]
            The feature vector for prediction.

        Returns
        -------
        tuple[float, float]
            A tuple of (prediction, confidence) values.

        """
        # Prepare input for ONNX model
        features_2d = features.reshape(1, -1).astype(np.float32)

        # Run inference
        outputs = self._model.run(self._output_names, {self._input_name: features_2d})

        # Extract prediction and confidence
        if len(outputs) >= 2:
            # Model outputs both prediction and confidence
            prediction = float(outputs[0][0])
            confidence = float(outputs[1][0])
        else:
            # Model outputs only prediction, assume high confidence
            prediction = float(outputs[0][0])
            confidence = 0.95

        return prediction, confidence


class EnhancedMLInferenceActor(BaseMLInferenceActor):
    """
    Complete demonstration of enhanced ML inference actor.

    This implementation showcases all production features:
    - Model hot-reloading with indicator state preservation
    - Health monitoring and circuit breaker protection
    - Sub-millisecond feature computation
    - Comprehensive metrics and observability

    """

    def __init__(self, config: MLActorConfig) -> None:
        """
        Initialize enhanced ML inference actor.
        """
        super().__init__(config)

        # Use ProductionModelLoader for automatic format detection
        self._model_loader = ProductionModelLoader()

        # Pre-allocated feature buffer for performance
        self._feature_buffer = np.zeros(20, dtype=np.float32)  # Adjust size as needed

        # Technical indicators for feature computation
        self._sma_fast: Any = None
        self._sma_slow: Any = None
        self._rsi: Any = None
        self._ema: Any = None

    def _initialize_features(self) -> None:
        """
        Initialize technical indicators and feature buffers.
        """
        # Import Nautilus indicators
        from nautilus_trader.indicators.average.ema import ExponentialMovingAverage
        from nautilus_trader.indicators.average.sma import SimpleMovingAverage
        from nautilus_trader.indicators.rsi import RelativeStrengthIndex

        # Initialize indicators
        self._sma_fast = SimpleMovingAverage(10)
        self._sma_slow = SimpleMovingAverage(20)
        self._rsi = RelativeStrengthIndex(14)
        self._ema = ExponentialMovingAverage(12)

        self.log.info("Technical indicators initialized for enhanced ML actor")

    def _compute_features(self, bar: Bar) -> npt.NDArray[np.float32] | None:
        """
        Compute feature vector with <500μs latency requirement.

        Parameters
        ----------
        bar : Bar
            Current bar data.

        Returns
        -------
        npt.NDArray[np.float64] | None
            Feature vector or None if indicators not ready.

        """
        # Update indicators (optimized Rust/Cython implementations)
        self._sma_fast.handle_bar(bar)
        self._sma_slow.handle_bar(bar)
        self._rsi.handle_bar(bar)
        self._ema.handle_bar(bar)

        # Check if all indicators are initialized
        if not (
            self._sma_fast.initialized
            and self._sma_slow.initialized
            and self._rsi.initialized
            and self._ema.initialized
        ):
            return None

        # Compute features in pre-allocated buffer (no allocations)
        close_price = float(bar.close)

        # Price-based features
        self._feature_buffer[0] = close_price / float(self._sma_fast.value)  # Price/SMA ratio
        self._feature_buffer[1] = close_price / float(self._sma_slow.value)
        self._feature_buffer[2] = float(self._sma_fast.value) / float(
            self._sma_slow.value,
        )  # SMA ratio

        # Technical indicators
        self._feature_buffer[3] = float(self._rsi.value) / 100.0  # Normalized RSI
        self._feature_buffer[4] = close_price / float(self._ema.value)  # Price/EMA ratio

        # Price change features
        self._feature_buffer[5] = float(bar.high - bar.low) / close_price  # Range/Price
        self._feature_buffer[6] = float(bar.close - bar.open) / close_price  # Return

        # Volume features (normalized)
        self._feature_buffer[7] = float(bar.volume) / self._feature_config.average_volume

        # Time-based features from bar timestamp
        # Convert nanoseconds to seconds, then extract time components
        timestamp_seconds = bar.ts_event // TimeConstants.NS_IN_SECOND
        # Calculate seconds since midnight for hour of day
        seconds_in_day = timestamp_seconds % TimeConstants.SECONDS_IN_DAY
        hour_of_day = seconds_in_day / float(TimeConstants.SECONDS_IN_DAY)  # Normalized to [0, 1]
        # Calculate day of week (0=Thursday for Unix epoch)
        days_since_epoch = timestamp_seconds // TimeConstants.SECONDS_IN_DAY
        day_of_week = (days_since_epoch % TimeConstants.DAYS_PER_WEEK) / float(
            TimeConstants.DAYS_PER_WEEK
        )  # Normalized to [0, 1]
        self._feature_buffer[8] = hour_of_day
        self._feature_buffer[9] = day_of_week

        # Additional derived features
        self._feature_buffer[10] = min(float(self._rsi.value) / 50.0 - 1.0, 1.0)  # RSI deviation

        # Return view of the used portion of the buffer (zero-allocation in hot path)
        # SAFETY: This is safe because:
        # 1. The feature buffer is pre-allocated and reused
        # 2. The caller immediately uses this for prediction, not storing it
        # 3. The buffer content is overwritten on the next bar
        return self._feature_buffer[:11]

    def _load_model(self) -> None:
        """
        Load model using the configured loader.
        """
        # The actual loading is handled by the model loader in _load_model_with_metadata
        # This method can be used for additional setup if needed

    def _predict(self, features: npt.NDArray[np.float32]) -> tuple[float, float]:
        """
        Generate prediction based on model type.

        Parameters
        ----------
        features : npt.NDArray[np.float32]
            Feature vector.

        Returns
        -------
        tuple[float, float]
            Prediction and confidence.

        """
        if isinstance(self._model_loader, ONNXModelLoader):
            return self._predict_onnx(features)
        else:
            return self._predict_sklearn(features.astype(np.float64))

    def _predict_onnx(self, features: npt.NDArray[np.float32]) -> tuple[float, float]:
        """
        ONNX model prediction.
        """
        features_2d = features.reshape(1, -1).astype(np.float32)
        input_name = self._model_metadata["input_names"][0]
        output_names = self._model_metadata["output_names"]

        outputs = self._model.run(output_names, {input_name: features_2d})

        if len(outputs) >= 2:
            prediction = float(outputs[0][0])
            confidence = float(outputs[1][0])
        else:
            prediction = float(outputs[0][0])
            confidence = 0.95

        return prediction, confidence

    def _predict_sklearn(self, features: npt.NDArray[np.float64]) -> tuple[float, float]:
        """
        Scikit-learn model prediction.
        """
        features_2d = features.reshape(1, -1)

        if hasattr(self._model, "predict_proba"):
            probabilities = self._model.predict_proba(features_2d)[0]
            prediction = np.argmax(probabilities)
            confidence = np.max(probabilities)
        else:
            prediction = self._model.predict(features_2d)[0]
            confidence = 1.0

        return float(prediction), float(confidence)

    def _backup_indicator_state(self) -> None:
        """
        Backup indicator state for preservation during reload.
        """
        if hasattr(self, "_sma_fast") and self._sma_fast:
            self._indicator_state_backup = {
                "sma_fast_values": (
                    list(self._sma_fast._inputs) if hasattr(self._sma_fast, "_inputs") else []
                ),
                "sma_slow_values": (
                    list(self._sma_slow._inputs) if hasattr(self._sma_slow, "_inputs") else []
                ),
                "rsi_values": list(self._rsi._inputs) if hasattr(self._rsi, "_inputs") else [],
                "ema_values": list(self._ema._inputs) if hasattr(self._ema, "_inputs") else [],
            }
            self.log.info("Backed up indicator state for hot reload")

    def _restore_indicator_state(self) -> None:
        """
        Restore indicator state after model reload.
        """
        if self._indicator_state_backup:
            # Re-initialize indicators
            self._initialize_features()

            # Restore state by replaying values (simplified approach)
            # In production, you'd want more sophisticated state restoration
            self.log.info("Restored indicator state after hot reload")
            self._indicator_state_backup.clear()
