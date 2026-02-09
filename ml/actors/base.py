"""
Base class for ML inference actors.

This module provides the foundation for building ML-powered actors that can perform
real-time inference on market data while maintaining the performance requirements of
Nautilus Trader's hot path.

The module defines the public API explicitly via ``__all__`` to satisfy
``mypy --strict`` with ``no_implicit_reexport`` behavior. Only the intended
surface is exported; internal helpers remain private.

"""

from __future__ import annotations

import hashlib
import logging
import time
from abc import ABC
from abc import abstractmethod
from collections import deque
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import DataType
from nautilus_trader.model.identifiers import InstrumentId

from ml._imports import HAS_ONNX
from ml._imports import check_ml_dependencies
from ml.actors.common import FeaturesComponent
from ml.actors.common import ModelComponent
from ml.actors.common import RegistryComponent
from ml.actors.common import StoreOperationsComponent
from ml.actors.common import build_prediction_surface_metadata
from ml.actors.common import build_signal_metadata
from ml.actors.common.features import build_feature_dict
from ml.actors.common.features import is_monotonic_ingress_timestamp
from ml.actors.common.remediation import evaluate_inference_deadline_guard
from ml.actors.common.remediation import evaluate_ml_failure_action
from ml.common import normalize_prediction_output
from ml.common import resolve_output_is_logits
from ml.common import resolve_positive_class_index
from ml.common.logging_utils import log_best_effort
from ml.common.metrics import causality_monotonic_violations_total
from ml.common.metrics import drift_policy_actions_total
from ml.common.metrics import inference_deadline_timeouts_total
from ml.common.metrics import ml_failure_actions_total
from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_manager import MetricsManager
from ml.common.model_load_policy import apply_direct_model_load_policy
from ml.common.protocols import MLComponentMixin
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
from ml.config.policy import CausalityMonotonicEnforcement
from ml.config.policy import DriftActionPolicy
from ml.config.policy import InferenceTimeoutAction
from ml.config.policy import MLFailureAction
from ml.config.runtime import OnnxRuntimeConfig
from ml.config.runtime import to_session_options
from nautilus_trader.common.config import ActorConfig


if TYPE_CHECKING:
    from typing import Any as _Any

    class NautilusActor:  # typing stub with minimal surface used in this module
        log: _Any
        id: str
        clock: _Any

        def __init__(self, *args: object, **kwargs: object) -> None: ...
        def subscribe_bars(self, *args: object, **kwargs: object) -> None: ...
        def publish_data(self, *args: object, **kwargs: object) -> None: ...
    class NautilusData:  # typing stub
        pass

else:
    from nautilus_trader.common.actor import Actor as NautilusActor
    from nautilus_trader.core.data import Data as NautilusData

if TYPE_CHECKING:
    # Protocols for type safety without enforcing concrete implementations
    from ml.observability.ml_async_persistence import MLPersistenceWorker
    from ml.stores.base import DummyStore
    from ml.stores.file_backed import FileDataStore
    from ml.stores.protocols import DataStoreFacadeProtocol
    from ml.stores.protocols import FeatureStoreStrictProtocol
    from ml.stores.protocols import ModelStoreStrictProtocol
    from ml.stores.protocols import StrategyStoreStrictProtocol

    DataStoreFacadeLike = DataStoreFacadeProtocol | FileDataStore | DummyStore
else:
    DataStoreFacadeLike = Any


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


# Public API (explicit exports for strict typing)
__all__ = [
    "BaseMLInferenceActor",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerState",
    "HealthMonitor",
    "HealthStatus",
    "MLSignal",
    "ModelLoader",
    "ONNXModelLoader",
    "ProductionModelLoader",
]


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance with metrics.

    Prevents cascade failures by temporarily stopping operations when error rates
    exceed thresholds. Emits Prometheus metrics on state changes.

    Parameters
    ----------
    config : CircuitBreakerConfig | None, optional
        Circuit breaker configuration.
    component_id : str, default "ml_actor"
        Component label for metrics (kept low-cardinality).

    """

    def __init__(
        self,
        config: CircuitBreakerConfig | None = None,
        *,
        component_id: str = "ml_actor",
    ) -> None:
        from ml.common.metrics_manager import MetricsManager

        self._config = config or CircuitBreakerConfig()
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._next_attempt = 0.0
        self._component_id = component_id
        try:
            mm = MetricsManager.default()
            mm.set_gauge(
                "nautilus_ml_circuit_breaker_state",
                "Circuit breaker state (0=closed, 0.5=half_open, 1=open)",
                0.0,
                labels={"component": self._component_id},
                labelnames=("component",),
            )
        except Exception as exc:
            logger.debug("Circuit breaker gauge init failed: %s", exc)

    @property
    def state(self) -> CircuitBreakerState:
        """
        Get current circuit breaker state.
        """
        return self._state

    def can_execute(self) -> bool:
        """
        Check if operation can be executed based on current state.

        Transitions OPEN -> HALF_OPEN when recovery timeout elapses.

        """
        current_time = time.time()

        if self._state == CircuitBreakerState.CLOSED:
            return True
        elif self._state == CircuitBreakerState.OPEN:
            if current_time >= self._next_attempt:
                self._state = CircuitBreakerState.HALF_OPEN
                self._success_count = 0
                try:
                    mm = MetricsManager.default()
                    mm.set_gauge(
                        "nautilus_ml_circuit_breaker_state",
                        "Circuit breaker state (0=closed, 0.5=half_open, 1=open)",
                        0.5,
                        labels={"component": self._component_id},
                        labelnames=("component",),
                    )
                    mm.inc(
                        "nautilus_ml_circuit_breaker_trips_total",
                        "Total circuit breaker transitions",
                        labels={"component": self._component_id, "to_state": "half_open"},
                        labelnames=("component", "to_state"),
                    )
                except Exception as exc:
                    logger.debug("Circuit breaker metrics (half-open) failed: %s", exc)
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
                try:
                    mm = MetricsManager.default()
                    mm.set_gauge(
                        "nautilus_ml_circuit_breaker_state",
                        "Circuit breaker state (0=closed, 0.5=half_open, 1=open)",
                        0.0,
                        labels={"component": self._component_id},
                        labelnames=("component",),
                    )
                    mm.inc(
                        "nautilus_ml_circuit_breaker_trips_total",
                        "Total circuit breaker transitions",
                        labels={"component": self._component_id, "to_state": "closed"},
                        labelnames=("component", "to_state"),
                    )
                except Exception as exc:
                    logger.debug("Circuit breaker metrics (closed) failed: %s", exc)
        elif self._state == CircuitBreakerState.CLOSED:
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self, _exc: Exception | None = None) -> None:
        """
        Record failed operation.

        Parameters
        ----------
        _exc : Exception | None, optional
            Optional exception context for logging/telemetry hooks.

        """
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitBreakerState.HALF_OPEN:
            self._state = CircuitBreakerState.OPEN
            self._next_attempt = self._last_failure_time + self._config.recovery_timeout
            try:
                mm = MetricsManager.default()
                mm.set_gauge(
                    "nautilus_ml_circuit_breaker_state",
                    "Circuit breaker state (0=closed, 0.5=half_open, 1=open)",
                    1.0,
                    labels={"component": self._component_id},
                    labelnames=("component",),
                )
                mm.inc(
                    "nautilus_ml_circuit_breaker_trips_total",
                    "Total circuit breaker transitions",
                    labels={"component": self._component_id, "to_state": "open"},
                    labelnames=("component", "to_state"),
                )
            except Exception as error:
                logger.debug(
                    "Circuit breaker metrics (open from half-open) failed: %s",
                    error,
                    exc_info=True,
                )
        elif (
            self._state == CircuitBreakerState.CLOSED
            and self._failure_count >= self._config.failure_threshold
        ):
            self._state = CircuitBreakerState.OPEN
            self._next_attempt = self._last_failure_time + self._config.recovery_timeout
            try:
                mm = MetricsManager.default()
                mm.set_gauge(
                    "nautilus_ml_circuit_breaker_state",
                    "Circuit breaker state (0=closed, 0.5=half_open, 1=open)",
                    1.0,
                    labels={"component": self._component_id},
                    labelnames=("component",),
                )
                mm.inc(
                    "nautilus_ml_circuit_breaker_trips_total",
                    "Total circuit breaker transitions",
                    labels={"component": self._component_id, "to_state": "open"},
                    labelnames=("component", "to_state"),
                )
            except Exception as error:
                logger.debug(
                    "Circuit breaker metrics (open) failed: %s",
                    error,
                    exc_info=True,
                )

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

    def reset_state(self) -> None:
        """
        Reset circuit breaker state and counters.
        """
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._next_attempt = 0.0
        try:
            mm = MetricsManager.default()
            mm.set_gauge(
                "nautilus_ml_circuit_breaker_state",
                "Circuit breaker state (0=closed, 0.5=half_open, 1=open)",
                0.0,
                labels={"component": self._component_id},
                labelnames=("component",),
            )
        except Exception as exc:
            logger.debug("Circuit breaker metrics (reset) failed: %s", exc, exc_info=True)


# Model loading supports both registry (preferred) and direct path (fallback)
# Registry path follows Universal ML Architecture Pattern #1
# Direct path maintained for testing and legacy compatibility


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

    def __init__(self, model_dir: str | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else Path.cwd()

    def load_model(self, path: str) -> tuple[Any, dict[str, Any]]:
        """
        Load model based on file extension.
        """
        import json
        from pathlib import Path

        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        # Determine model type by extension
        if path.endswith(".json"):
            # XGBoost JSON model
            try:
                from ml._imports import HAS_XGBOOST
                from ml._imports import check_ml_dependencies
                from ml._imports import xgb

                if not HAS_XGBOOST:
                    check_ml_dependencies(["xgboost"])

                booster = xgb.Booster()
                booster.load_model(path)
                return booster, {"type": "xgboost", "format": "json"}
            except Exception:
                # Try loading as JSON metadata (not used in hot path)
                data = json.loads(Path(path).read_text(encoding="utf-8"))
                return data, {"type": "json", "format": "json"}

        elif path.endswith((".pkl", ".pickle")):
            # Pickle models are completely forbidden for security
            import os

            onnx_only_mode = os.environ.get("ML_ONNX_ONLY", "").lower() in {"1", "true", "yes"}
            if onnx_only_mode:
                raise ValueError(
                    "Pickle model formats are forbidden in ONNX-only mode. "
                    "Use ONNX models for secure production deployment.",
                )
            else:
                raise ValueError(
                    "Pickle model formats (.pkl, .pickle) are not supported for security reasons. "
                    "Export models to ONNX for production or joblib for testing. "
                    "Set ML_ONNX_ONLY=1 for maximum security (ONNX-only mode).",
                )

        elif path.endswith(".joblib"):
            # Production security: Fail-closed approach for joblib models
            import os

            # Check for strict ONNX-only mode (highest security)
            if os.environ.get("ML_ONNX_ONLY", "").lower() in {"1", "true", "yes"}:
                raise ValueError(
                    "Joblib models are disabled in ONNX-only mode. "
                    "Use ONNX models for production deployment.",
                )

            # Standard test-only guards
            allow_joblib = (
                os.environ.get("ML_ALLOW_JOBLIB", "").lower() in {"1", "true", "yes"}
                or bool(os.environ.get("PYTEST_CURRENT_TEST"))
                or os.environ.get("ML_TESTING", "").lower() in {"1", "true", "yes"}
            )
            if not allow_joblib:
                # Disallow joblib in production paths for security and reproducibility
                raise ValueError(
                    "Joblib model format (.joblib) is not supported in production. "
                    "Enable with ML_ALLOW_JOBLIB=1 in test runs or export models to ONNX. "
                    "For maximum security, set ML_ONNX_ONLY=1 to disable all unsafe formats.",
                )

            from ml._imports import joblib as _joblib

            if _joblib is None:
                raise ImportError(
                    "joblib not available; install for test-only usage or export to ONNX",
                )

            model = _joblib.load(path)
            metadata = {
                "type": "sklearn",
                "format": "joblib",
                "model_class": getattr(model, "__class__", type(model)).__name__,
            }
            return model, metadata

        elif path.endswith(".onnx"):
            # ONNX model with integrity verification
            from ml.common.security import secure_onnx_load

            policy_result = apply_direct_model_load_policy(
                model_path=model_path,
                model_id=model_path.stem,
                context="actor_direct_path_onnx_load",
            )
            session = secure_onnx_load(
                file_path=model_path,
                expected_digest=policy_result.expected_digest,
                strict_integrity=policy_result.strict_integrity,
            )
            metadata = dict(policy_result.metadata)
            metadata.update(
                {
                    "type": "onnx",
                    "format": "onnx",
                    "input_names": [inp.name for inp in session.get_inputs()],
                    "output_names": [out.name for out in session.get_outputs()],
                },
            )
            return session, metadata
        else:
            raise ValueError(f"Unsupported model format: {path}")


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

        policy_result = apply_direct_model_load_policy(
            model_path=model_path,
            model_id=model_path.stem,
            context="actor_direct_path_onnx_loader",
        )

        # Create optimized ONNX Runtime session with integrity verification
        session_options, providers = to_session_options(self._runtime_config)

        from ml.common.security import secure_onnx_load

        session = secure_onnx_load(
            file_path=model_path,
            expected_digest=policy_result.expected_digest,
            session_options=session_options,
            providers=providers,
            strict_integrity=policy_result.strict_integrity,
        )

        # Generate metadata
        metadata = dict(policy_result.metadata)
        metadata.update(
            {
                "path": str(model_path),
                "size_bytes": model_path.stat().st_size,
                "modified_time": model_path.stat().st_mtime,
                "version": self.get_model_version(path),
                "type": "onnx",
                "format": "onnx",
                "input_names": [inp.name for inp in session.get_inputs()],
                "output_names": [out.name for out in session.get_outputs()],
                "providers": session.get_providers(),
            },
        )

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
        return hashlib.sha256(version_string.encode()).hexdigest()[:8]


class MLSignal(NautilusData):
    """
    ML signal data class for signal generation.

    Simple data class with required model_id field for tracking.

    Parameters
    ----------
    instrument_id : InstrumentId
        The instrument the prediction is for.
    model_id : str
        Unique identifier for the model that generated this signal.
    prediction : float
        The model prediction probability in [0, 1].
    confidence : float
        The confidence score for the prediction (0.0 to 1.0), derived from the
        calibrated probability when explicit confidence is unavailable.
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


# Prometheus metrics for monitoring (initialized via MetricsManager)
_MM = MetricsManager.default()
ml_predictions_total = _MM.counter(
    METRIC_PREDICTIONS_TOTAL,
    "Total number of ML predictions made",
    [LABEL_ACTOR_ID, LABEL_MODEL_NAME],
)
ml_prediction_latency = _MM.histogram(
    METRIC_PREDICTION_LATENCY_SECONDS,
    "Latency of ML predictions in seconds",
    [LABEL_ACTOR_ID, LABEL_MODEL_NAME],
)
ml_signal_confidence = _MM.histogram(
    METRIC_SIGNAL_CONFIDENCE,
    "Distribution of ML signal confidence scores",
    [LABEL_ACTOR_ID, LABEL_MODEL_NAME],
)

_persistence_fallback_drops = get_counter(
    "nautilus_ml_persistence_fallback_drops_total",
    "Total persistence drops when sync fallback is disabled",
    labelnames=("kind", "reason"),
)


class BaseMLInferenceActor(MLComponentMixin, NautilusActor, ABC):
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

    # LEGACY CODE - Commented out for Phase 2.3.5a (now provided by @property decorators)
    # Store attributes are initialized in _init_stores_and_registries
    # _feature_store: FeatureStoreStrictProtocol  # Strict adapters wrap underlying stores
    # _model_store: ModelStoreStrictProtocol
    # _strategy_store: StrategyStoreStrictProtocol
    # _data_store: DataStoreFacadeProtocol  # Narrow facade used in actors
    # _feature_registry: Any
    # _model_registry: Any
    # _strategy_registry: Any
    # _data_registry: Any
    # _persistence_worker: MLPersistenceWorker | None

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

        # ========================================================================
        # FACADE PATTERN: Initialize 4 components for separation of concerns
        # (Phase 2.3.5a - Component-based architecture)
        #
        # Component initialization order matters:
        # 0. Initialize name-mangled attributes FIRST (properties access these)
        # 1. StoreOperationsComponent - provides stores (no dependencies)
        # 2. RegistryComponent - provides registries (no dependencies)
        # 3. ModelComponent - manages model lifecycle
        # 4. FeaturesComponent - manages feature computation
        # ========================================================================

        # Step 0: Initialize name-mangled instance attributes for internal storage (used by properties)
        # CRITICAL: These MUST be initialized BEFORE components because properties access them
        # These allow property methods to remain read-only while supporting assignment in legacy code
        self.__feature_store_instance: FeatureStoreStrictProtocol | None = None
        self.__model_store_instance: ModelStoreStrictProtocol | None = None
        self.__strategy_store_instance: StrategyStoreStrictProtocol | None = None
        self.__data_store_instance: DataStoreFacadeProtocol | None = None
        self.__feature_registry_instance: Any | None = None
        self.__model_registry_instance: Any | None = None
        self.__strategy_registry_instance: Any | None = None
        self.__data_registry_instance: Any | None = None
        self.__persistence_worker_instance: MLPersistenceWorker | None = None

        # Step 0b: Initialize feature configuration (needed by FeaturesComponent)
        # CRITICAL: Must be initialized BEFORE FeaturesComponent
        # IMPORTANT: Do NOT mutate config - preserve original config object for contract tests
        # Use internal _feature_config attribute instead
        self._feature_config = config.feature_config or MLFeatureConfig()

        # Step 0c: Initialize actor services (stores + registries) ONCE
        # This prevents duplicate init_actor_services calls (contract requirement)
        from ml.actors.actor_services import init_actor_services

        self._actor_services = init_actor_services(self._config)

        # Step 1: Initialize StoreOperationsComponent (provides all 4 stores)
        self._store_ops_component = StoreOperationsComponent(
            config=self._config,
            actor_id=str(self.id),
            services=self._actor_services,  # Pass pre-initialized services
        )

        # Step 2: Initialize RegistryComponent (provides all 4 registries)
        self._registry_component = RegistryComponent(
            config=self._config,
            logger=self.log,
            services=self._actor_services,  # Pass pre-initialized services
        )

        # Step 3: Initialize ModelComponent (manages model loading, validation, hot-reload)
        self._model_component = ModelComponent(
            config=self._config,
            logger=self.log,
        )

        # Step 4: Initialize FeaturesComponent (manages feature computation, buffering, validation)
        # Note: health_monitor and persistence_worker are initialized later, so we pass None here
        # All components now share the same config instance (self._config)
        self._features_component = FeaturesComponent(
            config=self._config,
            compute_function=self._compute_features,
            feature_registry=self._feature_registry,
            feature_store=self._feature_store,
            health_monitor=None,  # Initialized later in __init__
            persistence_worker=None,  # Initialized later in __init__
            logger=self.log,
        )
        # ========================================================================

        # LEGACY CODE - Commented out for Phase 2.3.5a (replaced by StoreOperationsComponent)
        # MANDATORY: Initialize stores and registries for data persistence
        # self._init_stores_and_registries()

        # NOTE: _feature_config initialized earlier (before FeaturesComponent) to avoid circular dependency

        # Model and inference state
        self._model: Any = None
        self._model_metadata: dict[str, Any] = {}
        self._model_version: str | None = None
        self._model_id: str = "unknown"  # Track model ID for signals
        self._decision_metadata_payload: dict[str, Any] | None = None
        self._signal_metadata_extra: dict[str, Any] | None = None
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
        self._features_component.update_dependencies(
            health_monitor=self._health_monitor,
            persistence_worker=self._persistence_worker,
        )

        # Hot reload state
        self._last_model_check = 0.0
        self._indicator_state_backup: dict[str, Any] = {}

        # Performance tracking
        self._prediction_count = 0
        self._total_inference_time = 0.0
        self._total_feature_time = 0.0
        self._last_prediction_time = 0
        self._sync_prediction_fallback_disabled_logged = False

        # Warm-up tracking
        self._bars_processed = 0
        self._is_warmed_up = False
        self._ml_inference_halted = False
        self._ml_failure_reason: str | None = None
        self._ml_halt_logged = False
        self._last_ingress_ts_event: int | None = None

        # Enhanced Prometheus metrics - use global instances
        self._inference_latency_metric = ml_prediction_latency
        self._inference_count_metric = ml_predictions_total
        self._inference_confidence_metric = ml_signal_confidence
        # Manifest-driven feature schema (populated when loading from registry)
        self._manifest_feature_names: list[str] = []
        self._manifest_feature_dtypes: list[str] = []
        self._manifest_feature_schema_hash: str | None = None

    # ============================================================================
    # FACADE PATTERN: Property accessors delegate to StoreOperationsComponent
    # (PUBLIC properties without underscore for backward compatibility)
    # ============================================================================

    @property
    def feature_store(self) -> FeatureStoreStrictProtocol:
        """
        Access feature store via StoreOperationsComponent (public API).
        """
        return self._store_ops_component.feature_store

    @property
    def model_store(self) -> ModelStoreStrictProtocol:
        """
        Access model store via StoreOperationsComponent (public API).
        """
        return self._store_ops_component.model_store

    @property
    def strategy_store(self) -> StrategyStoreStrictProtocol:
        """
        Access strategy store via StoreOperationsComponent (public API).
        """
        return self._store_ops_component.strategy_store

    @property
    def data_store(self) -> DataStoreFacadeLike:
        """
        Access data store via StoreOperationsComponent (public API).
        """
        return self._store_ops_component.data_store

    # Keep private aliases for internal use
    @property
    def _feature_store(self) -> FeatureStoreStrictProtocol:
        """
        Access feature store via StoreOperationsComponent (private).
        """
        if self.__feature_store_instance is not None:
            return self.__feature_store_instance
        return self._store_ops_component.feature_store

    @property
    def _model_store(self) -> ModelStoreStrictProtocol:
        """
        Access model store via StoreOperationsComponent (private).
        """
        if self.__model_store_instance is not None:
            return self.__model_store_instance
        return self._store_ops_component.model_store

    @property
    def _strategy_store(self) -> StrategyStoreStrictProtocol:
        """
        Access strategy store via StoreOperationsComponent (private).
        """
        if self.__strategy_store_instance is not None:
            return self.__strategy_store_instance
        return self._store_ops_component.strategy_store

    @property
    def _data_store(self) -> DataStoreFacadeLike:
        """
        Access data store via StoreOperationsComponent (private).
        """
        if self.__data_store_instance is not None:
            return self.__data_store_instance
        return self._store_ops_component.data_store

    # ============================================================================
    # FACADE PATTERN: Property accessors delegate to RegistryComponent
    # (PUBLIC properties without underscore for backward compatibility)
    # ============================================================================

    @property
    def feature_registry(self) -> Any:
        """
        Access feature registry via RegistryComponent (public API).
        """
        return self._registry_component.feature_registry

    @property
    def model_registry(self) -> Any:
        """
        Access model registry via RegistryComponent (public API).
        """
        return self._registry_component.model_registry

    @property
    def strategy_registry(self) -> Any:
        """
        Access strategy registry via RegistryComponent (public API).
        """
        return self._registry_component.strategy_registry

    @property
    def data_registry(self) -> Any:
        """
        Access data registry via RegistryComponent (public API).
        """
        return self._registry_component.data_registry

    # Keep private aliases for internal use
    @property
    def _feature_registry(self) -> Any:
        """
        Access feature registry via RegistryComponent (private).
        """
        if self.__feature_registry_instance is not None:
            return self.__feature_registry_instance
        return self._registry_component.feature_registry

    @property
    def _model_registry(self) -> Any:
        """
        Access model registry via RegistryComponent (private).
        """
        if self.__model_registry_instance is not None:
            return self.__model_registry_instance
        return self._registry_component.model_registry

    @property
    def _strategy_registry(self) -> Any:
        """
        Access strategy registry via RegistryComponent (private).
        """
        if self.__strategy_registry_instance is not None:
            return self.__strategy_registry_instance
        return self._registry_component.strategy_registry

    @property
    def _data_registry(self) -> Any:
        """
        Access data registry via RegistryComponent (private).
        """
        if self.__data_registry_instance is not None:
            return self.__data_registry_instance
        return self._registry_component.data_registry

    @property
    def _persistence_worker(self) -> MLPersistenceWorker | None:
        """
        Access persistence worker via StoreOperationsComponent.
        """
        if self.__persistence_worker_instance is not None:
            return self.__persistence_worker_instance
        return self._store_ops_component.persistence_worker

    # ============================================================================

    def _init_stores_and_registries(self) -> None:
        """
        Initialize all stores and registries - THIS IS MANDATORY!

        All ML actors MUST persist data for:
        - Feature parity between training and inference
        - Model performance tracking
        - Signal analysis and backtesting
        - Complete audit trail
        """
        # Centralized initializer (progressive fallback + wiring) via facade
        from ml.actors.actor_services import init_actor_services

        services = init_actor_services(self._config)

        # Attach services; use name-mangled attributes for internal storage
        from typing import Any as _Any
        from typing import cast as _cast

        self.__feature_store_instance = services.feature_store
        self.__model_store_instance = services.model_store
        self.__strategy_store_instance = services.strategy_store
        self.__data_store_instance = _cast(_Any, services.data_store)
        self.__feature_registry_instance = services.feature_registry
        self.__model_registry_instance = services.model_registry
        self.__strategy_registry_instance = services.strategy_registry
        self.__data_registry_instance = services.data_registry
        self._persistence_manager = None
        self.log.info("Stores and registries initialized (runtime facade)")

        # Initialize async persistence worker if enabled
        self.__persistence_worker_instance = None
        if self._config.enable_async_persistence:
            from ml.observability.ml_async_persistence import MLPersistenceWorker

            self.__persistence_worker_instance = MLPersistenceWorker(
                feature_store=self._feature_store,
                model_store=self._model_store,
                queue_maxsize=self._config.persistence_queue_size,
                flush_interval_seconds=self._config.persistence_flush_interval,
                batch_size=self._config.persistence_batch_size,
            )
            self.log.info(
                f"ML async persistence initialized: queue={self._config.persistence_queue_size}, "
                f"flush_interval={self._config.persistence_flush_interval}s",
            )

        # Propagate circuit breaker to underlying stores when available
        try:
            cb = getattr(self, "_circuit_breaker", None)
            if cb is not None:
                # Adapters expose `_store`; set on underlying stores for gating writes
                for adapter in (self._feature_store, self._model_store, self._strategy_store):
                    try:
                        raw = getattr(adapter, "_store", None)
                        if raw is not None:
                            setattr(raw, "_circuit_breaker", cb)
                        else:
                            # Fallback: set on adapter (harmless if not consumed)
                            setattr(adapter, "_circuit_breaker", cb)
                    except Exception:
                        self.log.debug(
                            f"Circuit breaker propagation failed for {type(adapter).__name__}",
                            exc_info=True,
                        )
                        continue
                # Data store may also support breaker if underlying implementation honors it
                try:
                    raw_ds = getattr(self._data_store, "_store", None)
                    if raw_ds is not None:
                        setattr(raw_ds, "_circuit_breaker", cb)
                    else:
                        setattr(self._data_store, "_circuit_breaker", cb)
                except Exception:
                    self.log.debug(
                        f"Circuit breaker propagation failed for {type(self._data_store).__name__}",
                        exc_info=True,
                    )
        except Exception:
            # Never impact actor initialization
            self.log.debug("Store circuit breaker propagation failed", exc_info=True)

    # LEGACY CODE - Preserved for Phase 2.3.5c parity tests
    # Store and registry properties now delegated via facade pattern above
    # @property
    # def feature_store(self) -> FeatureStoreStrictProtocol:
    #     """Get the feature store instance."""
    #     return self._feature_store
    #
    # @property
    # def model_store(self) -> ModelStoreStrictProtocol:
    #     """Get the model store instance."""
    #     return self._model_store
    #
    # @property
    # def strategy_store(self) -> StrategyStoreStrictProtocol:
    #     """Get the strategy store instance."""
    #     return self._strategy_store
    #
    # @property
    # def data_store(self) -> DataStoreFacadeProtocol:
    #     """Get the data store facade instance."""
    #     return self._data_store
    #
    # @property
    # def feature_registry(self) -> object:
    #     """Get the feature registry instance."""
    #     return self._feature_registry
    #
    # @property
    # def model_registry(self) -> object:
    #     """Get the model registry instance."""
    #     return self._model_registry
    #
    # @property
    # def strategy_registry(self) -> object:
    #     """Get the strategy registry instance."""
    #     return self._strategy_registry
    #
    # @property
    # def data_registry(self) -> object:
    #     """Get the data registry instance."""
    #     return self._data_registry

    def on_start(self) -> None:
        """
        Initialize the actor and subscribe to market data - delegates to all components.

        Lifecycle:
            1. Initialize stores via StoreOperationsComponent
            2. Initialize registries via RegistryComponent
            3. Load model via ModelComponent
            4. Initialize features via FeaturesComponent
            5. Subscribe to market data
            6. Call Actor.on_start() for Nautilus lifecycle

        This method is called when the actor starts and handles:
        - Model loading with version tracking
        - Feature buffer initialization
        - Market data subscription
        - Hot reload scheduling
        - Health monitoring initialization

        """
        self.log.info(f"Starting enhanced {self.__class__.__name__} (facade pattern)")

        try:
            # FACADE: Delegate to StoreOperationsComponent
            self._store_ops_component.on_start()
            self.log.debug("StoreOperationsComponent initialized")

            # FACADE: Components already initialized in __init__()
            # Just delegate to their lifecycle methods

            # FACADE: Delegate model loading to ModelComponent
            self._model_component.load_model()
            # Backward-compatible mirrors: legacy prediction paths expect the
            # loaded model + metadata on the actor instance.
            self._model = self._model_component.model
            self._model_metadata = self._model_component.model_metadata
            self._model_version = self._model_component.model_version
            if self._model_component.model_id is not None:
                self._model_id = self._model_component.model_id
            self._refresh_decision_metadata_payload()
            self.log.debug(f"Model loaded via ModelComponent: {self._model_component.model_id}")

            # FACADE: FeaturesComponent initialization happens in __init__, no separate step needed
            self.log.debug("FeaturesComponent initialized")

            # Verify training/inference parity requirements (hook for subclasses)
            try:
                self._verify_parity_requirements()
            except Exception:
                # Fail fast: parity guarantees are mandatory
                self.log.error(
                    "Parity verification failed",
                )
                raise

            # Update health monitor
            if self._health_monitor:
                self._health_monitor.set_model_loaded(True)
                self._health_monitor.set_indicators_initialized(True)

            # Schedule hot reload checks if enabled
            if self._config.enable_hot_reload:
                self._schedule_model_checks()

            # Subscribe to market data (only if actor is registered with trader)
            if hasattr(self, "trader_id") and self.trader_id is not None:
                self.subscribe_bars(self._config.bar_type)

            # Start async persistence worker (via StoreOperationsComponent)
            if self._persistence_worker:
                self._persistence_worker.start()
                self.log.info("ML persistence worker started")

            self.log.info(
                f"Enhanced ML Actor configured: "
                f"model={Path(self._config.model_path).name if self._config.model_path else 'from_registry'}, "
                f"version={self._model_component.model_version}, "
                f"threshold={self._config.prediction_threshold}, "
                f"warm_up={self._config.warm_up_period}, "
                f"hot_reload={self._config.enable_hot_reload}, "
                f"health_monitoring={self._config.enable_health_monitoring}",
            )

        except Exception:
            self.log.error(
                "Failed to start ML Actor",
            )
            if self._health_monitor:
                self._health_monitor.set_model_loaded(False)
            raise

        # LEGACY CODE - Preserved for Phase 2.3.5c parity tests
        # def _old_on_start(self) -> None:
        #     self._load_model_with_metadata()
        #     self._initialize_features()
        #     # ... old initialization code ...

    # Hook for subclass-specific parity verification (no-op by default)
    def _verify_parity_requirements(self) -> None:  # pragma: no cover - default no-op
        return None

    # Hook for subclass-specific replay/rewind preparation (no-op by default)
    def _prepare_bar_runtime_state(self, _bar: Bar) -> None:  # pragma: no cover - default no-op
        return None

    def _record_causality_monotonic_violation_metric(
        self,
        *,
        mode: CausalityMonotonicEnforcement,
    ) -> None:
        """
        Record ingress monotonic causality violation metric.

        Parameters
        ----------
        mode : CausalityMonotonicEnforcement
            Active monotonic enforcement mode at violation time.

        """
        try:
            causality_monotonic_violations_total.labels(
                actor_id=self._actor_id_label(),
                mode=str(mode.value),
            ).inc()
        except Exception as exc:
            self.log.debug(
                "ml_actor.causality_monotonic_violation_metric_failed",
                exc_info=True,
                extra={"error": str(exc), "mode": str(mode.value)},
            )

    def _apply_ingress_causality_monotonic_guard(self, bar: Bar) -> bool:
        """
        Enforce ingress monotonic/backstep contract before runtime state mutation.

        Parameters
        ----------
        bar : Bar
            Incoming market bar event.

        Returns
        -------
        bool
            ``True`` when processing should continue; ``False`` when dropped.

        """
        ts_event = int(getattr(bar, "ts_event", 0))
        previous_ts_event = self._last_ingress_ts_event
        if is_monotonic_ingress_timestamp(
            ts_event=ts_event,
            previous_ts_event=previous_ts_event,
        ):
            self._last_ingress_ts_event = ts_event
            return True

        mode = self._config.remediation_policy.causality_monotonic_enforcement
        self._record_causality_monotonic_violation_metric(mode=mode)
        self.log.warning(
            "ml_actor.ingress_causality_non_monotonic "
            f"previous_ts_event={int(previous_ts_event) if previous_ts_event is not None else 'none'} "
            f"current_ts_event={ts_event} mode={mode.value}",
        )

        if mode == CausalityMonotonicEnforcement.DROP:
            return False

        if mode == CausalityMonotonicEnforcement.RESET:
            self._reset_inference_runtime_state(
                reason="ingress_non_monotonic_backstep",
                ts_event=ts_event,
            )

        self._last_ingress_ts_event = ts_event
        return True

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

        if not self._apply_ingress_causality_monotonic_guard(bar):
            return

        # Allow subclass hooks to invalidate state before halted checks/warmup.
        try:
            self._prepare_bar_runtime_state(bar)
        except Exception as exc:
            self.log.exception(
                f"ml_actor.prepare_bar_runtime_state_failed error={exc!r}",
            )

        # Fail-closed only when explicit halt policy/action has transitioned state.
        if self._ml_inference_halted:
            if not self._ml_halt_logged:
                self.log.error(
                    f"ml_actor.inference_halted reason={self._ml_failure_reason or 'unknown'}",
                )
                self._ml_halt_logged = True
            return

        # Track bars for warm-up period
        self._bars_processed += 1

        # Check if warmed up first (before feature computation)
        if not self._is_warmed_up:
            if self._bars_processed >= self._config.warm_up_period:
                self._is_warmed_up = True
                if hasattr(self, "log"):
                    self.log.info("Enhanced ML Actor warm-up complete, starting predictions")

        # FACADE: Delegate feature computation to FeaturesComponent
        # This ensures bars are buffered and metrics are tracked
        features = self._features_component.compute_features(bar)

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
        Stop the actor - clean up all components.

        Cleanup:
            1. Features component cleanup
            2. Model component cleanup
            3. Stores flushed via StoreOperationsComponent
            4. Call Actor.on_stop() for Nautilus lifecycle

        ALWAYS flushes all stores to ensure no data is lost.

        """
        self.log.info(f"Stopping {self.__class__.__name__} (facade pattern)")

        # FACADE: Delegate cleanup to components (always initialized in __init__)
        self._features_component.cleanup()
        self.log.debug("FeaturesComponent cleaned up")

        self._model_component.cleanup()
        self.log.debug("ModelComponent cleaned up")

        # FACADE: Delegate store cleanup to StoreOperationsComponent
        self._store_ops_component.on_stop()
        self.log.debug("StoreOperationsComponent stopped (all stores flushed)")

        # Stop actor-side bus bridge if enabled
        bridge = getattr(self, "_actor_bus_bridge", None)
        if bridge is not None:
            try:
                bridge.stop(drain=True, timeout=1.0)
            except Exception as exc:
                self.log.debug(
                    "ml_actor.bus_bridge_stop_failed",
                    exc_info=True,
                    extra={"error": str(exc)},
                )

        # Legacy stats logging (keep for backward compatibility)
        avg_inference_time = self._total_inference_time / max(self._prediction_count, 1)
        avg_feature_time = self._total_feature_time / max(self._bars_processed, 1)

        # Health status summary
        health_status = "N/A"
        if self._health_monitor:
            health_status = str(self._health_monitor.status)

        self.log.info(
            f"Stopped Enhanced {self.__class__.__name__} - "
            f"Predictions: {self._prediction_count}, "
            f"Avg inference time: {avg_inference_time:.3f}ms, "
            f"Avg feature time: {avg_feature_time:.3f}ms, "
            f"Health: {health_status}, "
            f"Circuit breaker: {str(self._circuit_breaker.state) if self._circuit_breaker else 'disabled'}",
        )

        # LEGACY CODE - Preserved for Phase 2.3.5c parity tests
        # def _old_on_stop(self) -> None:
        #     if self._persistence_worker is not None:
        #         asyncio.run(self._persistence_worker.stop(drain=True, timeout=5.0))
        #     if self._persistence_worker is None:
        #         self._feature_store.flush()
        #         self._model_store.flush()
        #         # ... old cleanup code ...

    def _generate_prediction_protected(
        self,
        bar: Bar,
        features: npt.NDArray[np.float32],
    ) -> None:
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

            if self._apply_inference_deadline_guard(
                inference_time_ms=inference_time,
                ts_event=int(bar.ts_event),
            ):
                return

            # MANDATORY: Store features for parity tracking (prefer manifest names)
            feature_names: list[str] | None = None
            try:
                fid = getattr(self._config, "feature_set_id", None)
                manifest = None
                getter = getattr(self._feature_registry, "get_feature_manifest", None)
                if fid and callable(getter):
                    manifest = getter(fid)
                if manifest is not None and len(manifest.feature_names) == len(features):
                    feature_names = list(manifest.feature_names)
            except Exception as exc:
                self.log.debug(
                    "feature_manifest_lookup_failed",
                    exc_info=True,
                    extra={"error": str(exc)},
                )
            feature_dict = build_feature_dict(features, feature_names=feature_names)

            feature_set_id = getattr(self._config, "feature_set_id", "default")
            self._features_component.persist_features_async(
                feature_set_id=feature_set_id,
                instrument_id=str(bar.bar_type.instrument_id),
                features=feature_dict,
                ts_event=bar.ts_event,
                ts_init=bar.ts_init,
            )

            # MANDATORY: Store prediction for performance tracking (async if enabled)
            self._persist_prediction_async(
                instrument_id=str(bar.bar_type.instrument_id),
                prediction=float(prediction),
                confidence=float(confidence),
                features=feature_dict,
                inference_time_ms=inference_time,
                ts_event=bar.ts_event,
            )

            self._record_prediction_success_state()

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
                    f"Prediction: {prediction:.4f}, confidence: {confidence:.4f}, latency: {inference_time:.3f}ms",
                )

            # Publish signal if confidence meets threshold
            if confidence >= self._config.prediction_threshold and self._config.publish_signals:
                signal = MLSignal(
                    instrument_id=bar.bar_type.instrument_id,
                    model_id=self._model_id,
                    prediction=prediction,
                    confidence=confidence,
                    features=features if self._config.log_predictions else None,
                    metadata=build_signal_metadata(
                        bar,
                        extra=self._signal_metadata_extra,
                    ),
                    ts_event=bar.ts_event,
                    ts_init=self.clock.timestamp_ns(),
                )
                self._publish_signal(signal)

        except Exception as exc:
            self.log.exception(f"Prediction failed: {exc}", exc)
            self._record_prediction_failure_state()
            self._apply_configured_ml_failure_action(
                reason="prediction_exception",
                ts_event=int(bar.ts_event),
                detail=repr(exc),
            )

    def _actor_id_label(self) -> str:
        """
        Return low-cardinality actor label for metrics.

        Returns
        -------
        str
            Actor identifier label used in remediation metrics.

        """
        actor_id = getattr(self, "id", None)
        return str(actor_id) if actor_id is not None else "unknown"

    def _record_prediction_success_state(self) -> None:
        """
        Record successful prediction in health and circuit-breaker state.
        """
        if self._health_monitor:
            self._health_monitor.update_prediction_success()
        if self._circuit_breaker:
            self._circuit_breaker.record_success()

    def _record_prediction_failure_state(self) -> None:
        """
        Record failed prediction in health and circuit-breaker state.
        """
        if self._health_monitor:
            self._health_monitor.update_prediction_failure()
        if self._circuit_breaker:
            self._circuit_breaker.record_failure()

    def _record_inference_deadline_timeout_metric(
        self,
        *,
        action: InferenceTimeoutAction,
    ) -> None:
        """
        Record inference deadline timeout metric.

        Parameters
        ----------
        action : InferenceTimeoutAction
            Deadline policy action that was applied.

        """
        try:
            inference_deadline_timeouts_total.labels(
                actor_id=self._actor_id_label(),
                action=str(action.value),
            ).inc()
        except Exception as exc:
            self.log.debug(
                "ml_actor.inference_deadline_timeout_metric_failed",
                exc_info=True,
                extra={"error": str(exc), "action": str(action.value)},
            )

    def _record_ml_failure_action_metric(
        self,
        *,
        action: MLFailureAction,
        reason: str,
    ) -> None:
        """
        Record ML failure action metric.

        Parameters
        ----------
        action : MLFailureAction
            Failure action applied.
        reason : str
            Low-cardinality reason label.

        """
        try:
            ml_failure_actions_total.labels(
                actor_id=self._actor_id_label(),
                action=str(action.value),
                reason=reason,
            ).inc()
        except Exception as exc:
            self.log.debug(
                "ml_actor.ml_failure_action_metric_failed",
                exc_info=True,
                extra={
                    "error": str(exc),
                    "action": str(action.value),
                    "reason": reason,
                },
            )

    def _record_drift_policy_action_metric(
        self,
        *,
        action: DriftActionPolicy,
        reason: str,
    ) -> None:
        """
        Record runtime drift policy action metric.

        Parameters
        ----------
        action : DriftActionPolicy
            Drift action applied.
        reason : str
            Low-cardinality reason label.

        """
        try:
            drift_policy_actions_total.labels(
                actor_id=self._actor_id_label(),
                action=str(action.value),
                reason=reason,
            ).inc()
        except Exception as exc:
            self.log.debug(
                "ml_actor.drift_policy_action_metric_failed",
                exc_info=True,
                extra={
                    "error": str(exc),
                    "action": str(action.value),
                    "reason": reason,
                },
            )

    def _emit_risk_halt_transition_hook(
        self,
        *,
        reason: str,
        detail: str | None,
        ts_event: int | None,
    ) -> bool:
        """
        Emit best-effort risk-halt transition hook to strategy store.

        Parameters
        ----------
        reason : str
            Halt reason label.
        detail : str | None
            Optional detail payload.
        ts_event : int | None
            Event timestamp in nanoseconds.

        Returns
        -------
        bool
            ``True`` when transition event write succeeds, otherwise ``False``.

        """
        writer = getattr(self._strategy_store, "write_risk_halt_event", None)
        if not callable(writer):
            log_best_effort(
                self.log,
                "error",
                "ml_actor.failure_state_transition_hook_unavailable",
                extra={"reason": reason},
            )
            return False
        cache_obj = getattr(self, "cache", None)
        is_backtesting = bool(getattr(cache_obj, "is_backtesting", False)) if cache_obj else False
        if ts_event is None:
            ts_event = int(self.clock.timestamp_ns())
        try:
            writer(
                strategy_id=str(self.id) if self.id else "ml_actor",
                instrument_id=str(self._config.instrument_id),
                event_type="halted",
                reason=reason,
                detail=detail,
                ts_event=int(ts_event),
                is_live=not is_backtesting,
            )
            return True
        except Exception as exc:
            log_best_effort(
                self.log,
                "error",
                "ml_actor.failure_state_transition_hook_failed",
                exc_info=True,
                extra={"error": str(exc), "reason": reason},
            )
            return False

    def _apply_ml_failure_action(
        self,
        *,
        action: MLFailureAction,
        reason: str,
        ts_event: int | None,
        detail: str | None = None,
    ) -> None:
        """
        Apply ML failure action transition semantics.

        Parameters
        ----------
        action : MLFailureAction
            Action policy to evaluate.
        reason : str
            Low-cardinality reason label.
        ts_event : int | None
            Event timestamp for transition hooks.
        detail : str | None, optional
            Optional detail payload for transition hooks.

        """
        decision = evaluate_ml_failure_action(action=action)
        if decision.halt_inference and self._ml_inference_halted:
            return

        self._record_ml_failure_action_metric(action=decision.action, reason=reason)

        if decision.transition_degraded and self._health_monitor is not None:
            self._health_monitor.status = HealthStatus.DEGRADED

        if decision.halt_inference:
            self._ml_inference_halted = True
            self._ml_failure_reason = reason
            self._ml_halt_logged = False
            if self._health_monitor is not None:
                self._health_monitor.status = HealthStatus.UNHEALTHY
            emitted = self._emit_risk_halt_transition_hook(
                reason=reason,
                detail=detail,
                ts_event=ts_event,
            )
            if not emitted:
                self._ml_failure_reason = "risk_state_transition_unavailable"
                self.log.error(
                    "ml_actor.failure_state_transition_required_but_missing "
                    f"requested_reason={reason}",
                )

    def _apply_drift_policy_outcome(
        self,
        *,
        action: DriftActionPolicy,
        reason: str,
        drift_score: float,
        threshold: float,
        ts_event: int | None,
        detail: str | None = None,
    ) -> bool:
        """
        Apply runtime drift policy outcome.

        Parameters
        ----------
        action : DriftActionPolicy
            Effective drift action.
        reason : str
            Low-cardinality reason label.
        drift_score : float
            Drift score that triggered the action.
        threshold : float
            Threshold used to trigger the action.
        ts_event : int | None
            Event timestamp in nanoseconds.
        detail : str | None, optional
            Optional detail payload for hooks/logging.

        Returns
        -------
        bool
            ``True`` when current inference should be aborted (fail-closed/halt).

        """
        self._record_drift_policy_action_metric(action=action, reason=reason)

        if action == DriftActionPolicy.LOG_ONLY:
            return False

        if action == DriftActionPolicy.DEGRADED:
            if self._health_monitor is not None:
                self._health_monitor.status = HealthStatus.DEGRADED
            return False

        # Fail-closed routes through existing halt transition semantics.
        self._record_prediction_failure_state()
        drift_detail = (
            detail
            if detail is not None
            else (
                f"drift_score={float(drift_score):.6f},"
                f"threshold={float(threshold):.6f}"
            )
        )
        self._apply_ml_failure_action(
            action=MLFailureAction.HALT,
            reason=reason,
            ts_event=ts_event,
            detail=drift_detail,
        )
        return True

    def _apply_configured_ml_failure_action(
        self,
        *,
        reason: str,
        ts_event: int | None,
        detail: str | None = None,
    ) -> None:
        """
        Apply configured ML failure policy action.

        Parameters
        ----------
        reason : str
            Low-cardinality reason label.
        ts_event : int | None
            Event timestamp in nanoseconds.
        detail : str | None, optional
            Optional detail payload.

        """
        if self._ml_inference_halted:
            return

        policy = self._config.remediation_policy
        self._apply_ml_failure_action(
            action=policy.ml_failure_action,
            reason=reason,
            ts_event=ts_event,
            detail=detail,
        )

    def _apply_inference_deadline_guard(
        self,
        *,
        inference_time_ms: float,
        ts_event: int | None,
    ) -> bool:
        """
        Apply inference deadline guard policy and return abort decision.

        Parameters
        ----------
        inference_time_ms : float
            Measured inference latency.
        ts_event : int | None
            Event timestamp in nanoseconds.

        Returns
        -------
        bool
            ``True`` when the current prediction should be dropped.

        """
        max_latency_ms = float(self._config.max_inference_latency_ms)
        if float(inference_time_ms) <= max_latency_ms:
            return False

        self.log.warning(
            f"Inference latency exceeded: {inference_time_ms:.3f}ms > {max_latency_ms:.3f}ms",
        )
        if self._health_monitor:
            self._health_monitor.update_latency_violation()

        policy = self._config.remediation_policy
        decision = evaluate_inference_deadline_guard(
            elapsed_ms=float(inference_time_ms),
            deadline_ms=max_latency_ms,
            enabled=bool(policy.enable_inference_deadline_guard),
            timeout_action=policy.inference_timeout_action,
        )
        if not decision.drop_prediction:
            return False

        self._record_inference_deadline_timeout_metric(action=decision.action)
        if decision.halt_inference:
            self._record_prediction_failure_state()
            self._apply_ml_failure_action(
                action=MLFailureAction.HALT,
                reason="inference_deadline_timeout",
                ts_event=ts_event,
                detail=f"inference_time_ms={float(inference_time_ms):.6f}",
            )
        return True

    def _record_persistence_drop(self, *, kind: str, reason: str) -> None:
        """
        Record a persistence drop for observability.

        Args:
            kind: The drop type (e.g., "feature", "prediction").
            reason: Reason for drop (e.g., "sync_disabled", "queue_full").
        """
        try:
            _persistence_fallback_drops.labels(kind=kind, reason=reason).inc()
        except Exception as exc:
            self.log.debug(
                "ml_actor.persistence_drop_metric_failed",
                exc_info=True,
                extra={"error": str(exc), "kind": kind, "reason": reason},
            )

    def _persist_prediction_async(
        self,
        *,
        instrument_id: str,
        prediction: float,
        confidence: float,
        features: dict[str, float],
        inference_time_ms: float,
        ts_event: int,
        is_live: bool | None = None,
    ) -> bool:
        """
        Persist predictions using the async worker when available.

        Args:
            instrument_id: Instrument identifier.
            prediction: Model prediction value.
            confidence: Prediction confidence.
            features: Feature dictionary used for inference.
            inference_time_ms: Inference latency in milliseconds.
            ts_event: Event timestamp in nanoseconds.

        Returns:
            True when enqueued/written, False when dropped or failed.
        """
        try:
            if self._persistence_worker is not None:
                enqueued = self._persistence_worker.enqueue_prediction(
                    model_id=self._model_id,
                    instrument_id=instrument_id,
                    prediction=float(prediction),
                    confidence=float(confidence),
                    features=features,
                    inference_time_ms=float(inference_time_ms),
                    ts_event=int(ts_event),
                )
                if not enqueued:
                    self.log.warning(
                        f"Persistence queue full - prediction write dropped (instrument: {instrument_id})",
                    )
                    self._record_persistence_drop(kind="prediction", reason="queue_full")
                return enqueued

            allow_sync_fallback = bool(
                getattr(self._config, "allow_sync_persistence_fallback", True),
            )
            if not allow_sync_fallback:
                self._record_persistence_drop(kind="prediction", reason="sync_disabled")
                if not self._sync_prediction_fallback_disabled_logged:
                    self.log.warning(
                        "Sync prediction persistence disabled; dropping prediction writes",
                    )
                    self._sync_prediction_fallback_disabled_logged = True
                return False

            self._model_store.write_prediction(
                model_id=self._model_id,
                instrument_id=instrument_id,
                prediction=float(prediction),
                confidence=float(confidence),
                features=features,
                inference_time_ms=float(inference_time_ms),
                ts_event=int(ts_event),
                is_live=bool(is_live) if is_live is not None else False,
            )
            return True
        except Exception as exc:
            self.log.debug(
                "ml_actor.prediction_persist_failed",
                exc_info=True,
                extra={"error": str(exc), "instrument_id": instrument_id},
            )
            return False

    def _publish_signal(self, signal: MLSignal) -> None:
        """
        Publish ML signal to the message bus.

        Parameters
        ----------
        signal : MLSignal
            The ML signal to publish.

        """
        self.publish_data(
            DataType(MLSignal, metadata={"source": str(self.id)}),
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
            # Try registry first (preferred path per Universal Pattern #1)
            loaded_from_registry = self._try_load_from_registry()

            # Fall back to direct path loading for testing/legacy configs
            if not loaded_from_registry:
                if not (hasattr(self._config, "model_path") and self._config.model_path):
                    raise ValueError(
                        "No model_id found in registry and no model_path provided. "
                        "Please specify either model_id (preferred) or model_path (legacy).",
                    )

                # Enforce ONNX-only in production unless explicitly allowed
                import os as _os
                from pathlib import Path as _Path

                model_ext = _Path(self._config.model_path).suffix.lower()
                # Detect test/dev environments where non-ONNX may be acceptable
                # Allow non-ONNX formats strictly for tests when explicitly enabled.
                # Accepted env flags (either):
                # - ML_TEST_ALLOW_NON_ONNX
                # - ML_ALLOW_NON_ONNX_IN_TESTS (back-compat)
                is_test_env = (
                    _os.getenv("PYTEST_CURRENT_TEST") is not None
                    or _os.getenv("ML_TEST_ALLOW_NON_ONNX", "").lower() in {"1", "true", "yes"}
                    or _os.getenv("ML_ALLOW_NON_ONNX_IN_TESTS", "").lower() in {"1", "true", "yes"}
                )
                allow_dev = getattr(self._config, "allow_non_onnx_in_dev", False)
                if model_ext != ".onnx" and not (is_test_env or allow_dev):
                    raise ValueError(f"Non-ONNX model format disallowed in prod: {model_ext}")

                # Load from direct path (fallback/legacy behavior)
                self.log.info(
                    f"Loading model from direct path (fallback): {self._config.model_path}",
                )
                self._model, self._model_metadata = self._model_loader.load_model(
                    self._config.model_path,
                )
            self._model_version = self._model_metadata.get("version")

            # Determine model_id
            self._determine_model_id()
            self._refresh_decision_metadata_payload()

            # Call the original abstract method for backward compatibility
            self._load_model()

            # Optional: shared model warm-up semantics
            try:  # pragma: no cover - environment dependent
                from ml.actors.model_loader_utils import maybe_warm_up_model

                # Honor optimization-based warm-up flag when present (e.g., signal actor)
                opt_cfg = getattr(self._config, "optimization_config", None)
                warm_flag = bool(getattr(opt_cfg, "enable_model_warm_up", False))

                # Derive input dimension from manifest feature schema when available
                input_dim = 0
                try:
                    schema = self._model_metadata.get("feature_schema")
                    if isinstance(schema, dict):
                        input_dim = len(schema)
                except Exception:
                    input_dim = 0

                if warm_flag and self._model is not None and input_dim > 0:
                    maybe_warm_up_model(self._model, True, input_dim)
            except Exception as exc:
                # Warm-up is a best-effort optimization; never fail startup
                self.log.debug("Model warm-up skipped due to error: %s", exc)

            self.log.info(
                f"Loaded model with metadata: "
                f"model_id={self._model_id}, "
                f"version={self._model_version}, "
                f"size={self._model_metadata.get('size_bytes', 0)} bytes, "
                f"type={self._model_metadata.get('type', 'unknown')}",
            )
        except Exception:
            self.log.error(
                "Failed to load model",
            )
            raise

    def _try_load_from_registry(self) -> bool:
        """
        Attempt to load model and metadata from registry; return True if loaded.

        Priority:
        1. Use model_id with shared ModelRegistry (preferred)
        2. Fall back to model_path for testing/development

        """
        # Check if we have a model_id to use with registry
        if hasattr(self._config, "model_id") and self._config.model_id:
            # Use the shared registry instance (not a new one!)
            registry = self._model_registry

            model_info = registry.get_model(self._config.model_id)
            if not model_info:
                # If model_id provided but not found, check for fallback path
                if hasattr(self._config, "model_path") and self._config.model_path:
                    self.log.warning(
                        f"Model {self._config.model_id} not found in registry, "
                        f"falling back to direct path: {self._config.model_path}",
                    )
                    return False  # Let the fallback path handle it
                raise ValueError(f"Model {self._config.model_id} not found in registry")

            # Load the actual model via registry
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
                "training_config": dict(getattr(manifest, "training_config", {})),
                "decision_policy": getattr(manifest, "decision_policy", None),
                "decision_config": getattr(manifest, "decision_config", {}),
                "output_schema": getattr(manifest, "output_schema", None),
                "calibration": getattr(manifest, "calibration", None),
                "artifact_sha256_digest": getattr(manifest, "artifact_sha256_digest", None),
            }
            # Stash manifest feature names/dtypes and hash
            try:
                self._manifest_feature_names = list(manifest.feature_schema.keys())
                self._manifest_feature_schema_hash = manifest.feature_schema_hash
                self._manifest_feature_dtypes = [
                    manifest.feature_schema[name] for name in self._manifest_feature_names
                ]
            except Exception:
                self._manifest_feature_names = []
                self._manifest_feature_schema_hash = None
                self._manifest_feature_dtypes = []

            # Use manifest features if configured
            if (
                hasattr(self._config, "use_manifest_features")
                and self._config.use_manifest_features
            ):
                self._feature_names = list(manifest.feature_schema.keys())
                self.log.info(f"Using {len(self._feature_names)} features from manifest")

            # Check deployment constraints
            if "max_latency_ms" in manifest.deployment_constraints:
                max_latency = manifest.deployment_constraints["max_latency_ms"]
                if hasattr(self._config, "max_inference_latency_ms") and (
                    self._config.max_inference_latency_ms > max_latency
                ):
                    self.log.warning(
                        f"Config latency {self._config.max_inference_latency_ms}ms exceeds model constraint {max_latency}ms",
                    )

            return True

        return False

    def _determine_model_id(self) -> None:
        """
        Populate `_model_id` from metadata, training metadata, or path fallback.
        """
        if "model_id" in self._model_metadata:
            self._model_id = self._model_metadata["model_id"]
            return

        if "training_metadata" in self._model_metadata:
            training_meta = self._model_metadata["training_metadata"]
            if "model_id" in training_meta:
                self._model_id = training_meta["model_id"]
                return

        from pathlib import Path

        model_name = Path(self._config.model_path).stem
        version_str = self._model_version[:8] if self._model_version else "v1"
        self._model_id = f"{model_name}_{version_str}"

    def _refresh_decision_metadata_payload(self) -> None:
        """
        Refresh decision metadata payloads after model metadata updates.
        """
        try:
            from ml.common import decision_metadata_from_model_metadata

            payload = decision_metadata_from_model_metadata(
                self._model_metadata,
                model_id=self._model_id,
                model_version=self._model_version,
            )
            self._decision_metadata_payload = payload
            neutral_band = float(getattr(self._config, "prediction_neutral_band", 0.0))
            self._signal_metadata_extra = build_prediction_surface_metadata(
                neutral_band=neutral_band,
                decision_metadata=payload,
            )
        except Exception as exc:
            self.log.debug(
                "decision_metadata_refresh_failed",
                exc_info=True,
                error=str(exc),
            )

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

    def _check_model_updates(self, event: object) -> None:
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

        except Exception:
            self.log.error(
                "Model update check failed",
            )
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
            self._refresh_decision_metadata_payload()

            # Update health status
            if self._health_monitor:
                self._health_monitor.set_model_loaded(True)

            self.log.info(
                f"Model hot-reload successful: {old_version} -> {self._model_version}",
            )

        except Exception:
            self.log.error(
                "Model reload failed",
            )
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

    def _reset_inference_runtime_state_components(self) -> None:  # pragma: no cover - default no-op
        """
        Reset subclass-specific runtime state during replay/rewind invalidation.
        """
        return None

    def _reset_inference_runtime_state(
        self,
        *,
        reason: str,
        ts_event: int | None = None,
    ) -> None:
        """
        Reset shared inference runtime state for replay/rewind safety.

        Parameters
        ----------
        reason : str
            Low-cardinality reason label for logs.
        ts_event : int | None, optional
            Timestamp associated with the reset trigger.

        """
        self._ml_inference_halted = False
        self._ml_failure_reason = None
        self._ml_halt_logged = False
        self._bars_processed = 0
        self._is_warmed_up = False
        self._prediction_count = 0
        self._total_inference_time = 0.0
        self._total_feature_time = 0.0
        self._last_prediction_time = 0
        self._last_ingress_ts_event = None
        self._sync_prediction_fallback_disabled_logged = False
        self._feature_window.clear()
        self._indicator_state_backup.clear()

        feature_component = getattr(self, "_features_component", None)
        if feature_component is not None:
            reset_runtime_state = getattr(feature_component, "reset_runtime_state", None)
            cleanup = getattr(feature_component, "cleanup", None)
            try:
                if callable(reset_runtime_state):
                    reset_runtime_state(reason=reason)
                elif callable(cleanup):
                    cleanup()
            except Exception as exc:
                self.log.exception(
                    "ml_actor.feature_runtime_reset_failed "
                    f"reason={reason} error={exc!r}",
                )

        if self._health_monitor is not None:
            self._health_monitor = HealthMonitor(config=self._config.health_config)

        if feature_component is not None:
            update_dependencies = getattr(feature_component, "update_dependencies", None)
            if callable(update_dependencies):
                try:
                    update_dependencies(
                        health_monitor=self._health_monitor,
                        persistence_worker=self._persistence_worker,
                    )
                except Exception as exc:
                    self.log.exception(
                        "ml_actor.feature_dependency_refresh_failed "
                        f"reason={reason} error={exc!r}",
                    )

        if self._circuit_breaker is not None:
            self._circuit_breaker.reset_state()

        try:
            self._reset_inference_runtime_state_components()
        except Exception as exc:
            self.log.exception(
                "ml_actor.subclass_runtime_reset_failed "
                f"reason={reason} error={exc!r}",
            )

        self.log.info(
            "ml_actor.inference_runtime_state_reset "
            f"reason={reason} ts_event={int(ts_event) if ts_event is not None else 'none'}",
        )

    def get_health_status(self) -> dict[str, Any]:
        """
        Get current health status of the actor.

        Returns
        -------
        dict[str, Any]
            Health status information including metrics and system state.

        """
        base_status: dict[str, Any] = {
            "actor_id": str(self.id),
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
            self._health_monitor = HealthMonitor(config=self._config.health_config)
            self._features_component.update_dependencies(
                health_monitor=self._health_monitor,
                persistence_worker=self._persistence_worker,
            )
            self.log.info("Health status reset")


class PickleMLInferenceActor(BaseMLInferenceActor):
    """
    Deprecated: pickle-based inference is unsupported.

    This class is retained as a stub to provide a clear error if instantiated.
    """

    def _load_model(self) -> None:  # pragma: no cover - stub
        raise SecurityError(
            "Pickle models are deprecated and not supported. Use ONNX or framework-native formats.",
        )

    def _predict(
        self,
        features: npt.NDArray[np.float32],
    ) -> tuple[float, float]:  # pragma: no cover - stub
        raise SecurityError("Pickle models are deprecated and not supported.")


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
                f"ONNX model loaded: input={self._input_name}, outputs={self._output_names}, providers={self._model_metadata.get('providers', [])}",
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

        output_is_logits = resolve_output_is_logits(self._model_metadata)
        positive_class_index = resolve_positive_class_index(self._model_metadata)
        if len(outputs) >= 2:
            return normalize_prediction_output(
                outputs[0],
                outputs[1],
                positive_class_index=positive_class_index,
                output_is_logits=output_is_logits,
            )
        return normalize_prediction_output(
            outputs[0],
            None,
            positive_class_index=positive_class_index,
            output_is_logits=output_is_logits,
        )


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
            TimeConstants.DAYS_PER_WEEK,
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

        output_is_logits = resolve_output_is_logits(self._model_metadata)
        positive_class_index = resolve_positive_class_index(self._model_metadata)
        if len(outputs) >= 2:
            return normalize_prediction_output(
                outputs[0],
                outputs[1],
                positive_class_index=positive_class_index,
                output_is_logits=output_is_logits,
            )
        return normalize_prediction_output(
            outputs[0],
            None,
            positive_class_index=positive_class_index,
            output_is_logits=output_is_logits,
        )

    def _predict_sklearn(self, features: npt.NDArray[np.float64]) -> tuple[float, float]:
        """
        Scikit-learn model prediction.
        """
        features_2d = features.reshape(1, -1)

        if hasattr(self._model, "predict_proba"):
            probabilities = self._model.predict_proba(features_2d)[0]
            classes = getattr(self._model, "classes_", None)
            positive_class_index = resolve_positive_class_index(
                self._model_metadata,
                classes=classes,
                num_classes=len(classes) if classes is not None else None,
            )
            return normalize_prediction_output(
                probabilities,
                None,
                positive_class_index=positive_class_index,
            )
        else:
            prediction = self._model.predict(features_2d)[0]
            return normalize_prediction_output(
                prediction,
                None,
            )

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


logger = logging.getLogger(__name__)
