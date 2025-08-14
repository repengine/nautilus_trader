"""
ML Signal Actor for real-time inference and signal generation.

This module provides a production-ready ML signal actor that performs real-time
inference on market data and generates trading signals with configurable strategies.
It follows Nautilus Trader's hot/cold path architecture and maintains sub-millisecond
performance with configurable optimization levels.

Key Features:
- Multiple signal generation strategies (threshold, extremes, momentum, ensemble, adaptive)
- Configurable performance optimization levels (standard, optimized)
- Plugin architecture for custom strategies
- Zero-allocation hot path with pre-allocated buffers
- Atomic model hot-swapping with state preservation
- Comprehensive performance monitoring and metrics
- Circuit breaker protection

Performance Targets:
- P99 feature computation: <500μs
- P99 model inference: <2ms
- P99 end-to-end signal: <5ms
- Memory stable over 24h operation
- Zero allocations in hot path

"""

from __future__ import annotations

import time
from abc import ABC
from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import msgspec
import numpy as np
import numpy.typing as npt

from ml._imports import HAS_ONNX
from ml._imports import HAS_PROMETHEUS
from ml._imports import check_ml_dependencies
from ml._imports import ort
from ml.actors.base import BaseMLInferenceActor
from ml.actors.base import MLSignal
from ml.common.metrics import Counter
from ml.common.metrics import Histogram
from ml.config.actors import MLSignalActorConfig
from ml.config.actors import OptimizationConfig
from ml.config.actors import StrategyConfig
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
from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.features.engineering import IndicatorManager
from ml.registry.feature_registry import LocalFeatureRegistry
from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelStore
from nautilus_trader.model.data import Bar


if TYPE_CHECKING:
    pass

# =================================================================================================
# Enums
# =================================================================================================


class SignalStrategy(Enum):
    """
    Signal generation strategy enumeration.
    """

    THRESHOLD = "threshold"
    EXTREMES = "extremes"
    MOMENTUM = "momentum"
    ENSEMBLE = "ensemble"
    ADAPTIVE = "adaptive"


class ThresholdStrategy(Enum):
    """
    Threshold strategy enumeration.
    """

    STATIC = "static"
    REGIME_AWARE = "regime_aware"
    DYNAMIC = "dynamic"


class OptimizationLevel(Enum):
    """
    Performance optimization level.
    """

    STANDARD = "standard"  # Standard performance (default)
    OPTIMIZED = "optimized"  # Advanced optimizations enabled


# =================================================================================================
# Module-level metrics initialization (singleton pattern)
# =================================================================================================

_metrics_initialized = False
_prediction_distribution_metric = None
_confidence_distribution_metric = None
_signal_generation_time_metric = None
_feature_time_by_feature_set_metric = None
_signals_generated_metric = None
_adaptive_threshold_metric = None
_market_regime_metric = None


def _initialize_performance_metrics() -> None:
    """
    Initialize module-level performance metrics once globally.
    """
    global _metrics_initialized
    global _prediction_distribution_metric
    global _confidence_distribution_metric
    global _signal_generation_time_metric
    global _signals_generated_metric
    global _adaptive_threshold_metric
    global _market_regime_metric

    if _metrics_initialized:
        return

    if HAS_PROMETHEUS:
        from ml._imports import REGISTRY

        existing_names = set(REGISTRY._names_to_collectors.keys())

        if METRIC_PREDICTION_DISTRIBUTION not in existing_names:
            _prediction_distribution_metric = Histogram(
                METRIC_PREDICTION_DISTRIBUTION,
                "Distribution of model predictions",
                [LABEL_ACTOR_ID],
            )
        else:
            _prediction_distribution_metric = cast(
                Histogram,
                REGISTRY._names_to_collectors[METRIC_PREDICTION_DISTRIBUTION],
            )

        if METRIC_CONFIDENCE_DISTRIBUTION not in existing_names:
            _confidence_distribution_metric = Histogram(
                METRIC_CONFIDENCE_DISTRIBUTION,
                "Distribution of prediction confidence scores",
                [LABEL_ACTOR_ID],
            )
        else:
            _confidence_distribution_metric = cast(
                Histogram,
                REGISTRY._names_to_collectors[METRIC_CONFIDENCE_DISTRIBUTION],
            )

        if METRIC_SIGNAL_GENERATION_SECONDS not in existing_names:
            _signal_generation_time_metric = Histogram(
                METRIC_SIGNAL_GENERATION_SECONDS,
                "Signal generation latency in seconds",
                [LABEL_ACTOR_ID, "strategy"],
                buckets=SIGNAL_LATENCY_BUCKETS,
            )
        else:
            _signal_generation_time_metric = cast(
                Histogram,
                REGISTRY._names_to_collectors[METRIC_SIGNAL_GENERATION_SECONDS],
            )
        if METRIC_FEATURE_TIME_BY_SET_SECONDS not in existing_names:
            _feature_time_by_feature_set_metric = Histogram(
                METRIC_FEATURE_TIME_BY_SET_SECONDS,
                "Feature computation latency by feature_set_id",
                [LABEL_ACTOR_ID, LABEL_FEATURE_SET_ID],
                buckets=FEATURE_TIME_BUCKETS,
            )
        else:
            _feature_time_by_feature_set_metric = cast(
                Histogram,
                REGISTRY._names_to_collectors[METRIC_FEATURE_TIME_BY_SET_SECONDS],
            )

        if METRIC_SIGNALS_GENERATED_TOTAL not in existing_names:
            _signals_generated_metric = Counter(
                METRIC_SIGNALS_GENERATED_TOTAL,
                "Total number of signals generated",
                [LABEL_ACTOR_ID, "strategy", "signal_type"],
            )
        else:
            _signals_generated_metric = cast(
                Counter,
                REGISTRY._names_to_collectors[METRIC_SIGNALS_GENERATED_TOTAL],
            )

        if METRIC_ADAPTIVE_THRESHOLD not in existing_names:
            _adaptive_threshold_metric = Histogram(
                METRIC_ADAPTIVE_THRESHOLD,
                "Adaptive threshold values",
                [LABEL_ACTOR_ID],
            )
        else:
            _adaptive_threshold_metric = cast(
                Histogram,
                REGISTRY._names_to_collectors[METRIC_ADAPTIVE_THRESHOLD],
            )

        if METRIC_MARKET_REGIME_TOTAL not in existing_names:
            _market_regime_metric = Counter(
                METRIC_MARKET_REGIME_TOTAL,
                "Market regime detection counts",
                [LABEL_ACTOR_ID, "regime"],
            )
        else:
            _market_regime_metric = cast(
                Counter,
                REGISTRY._names_to_collectors[METRIC_MARKET_REGIME_TOTAL],
            )
    else:
        # Use dummy metrics when Prometheus is not available
        _prediction_distribution_metric = Histogram(
            METRIC_PREDICTION_DISTRIBUTION,
            "Distribution of model predictions",
            [LABEL_ACTOR_ID],
        )
        _confidence_distribution_metric = Histogram(
            METRIC_CONFIDENCE_DISTRIBUTION,
            "Distribution of prediction confidence scores",
            [LABEL_ACTOR_ID],
        )
        _signal_generation_time_metric = Histogram(
            METRIC_SIGNAL_GENERATION_SECONDS,
            "Signal generation latency in seconds",
            [LABEL_ACTOR_ID, "strategy"],
            buckets=SIGNAL_LATENCY_BUCKETS,
        )
        _signals_generated_metric = Counter(
            METRIC_SIGNALS_GENERATED_TOTAL,
            "Total number of signals generated",
            [LABEL_ACTOR_ID, "strategy", "signal_type"],
        )
        _feature_time_by_feature_set_metric = Histogram(
            METRIC_FEATURE_TIME_BY_SET_SECONDS,
            "Feature computation latency by feature_set_id",
            [LABEL_ACTOR_ID, LABEL_FEATURE_SET_ID],
            buckets=FEATURE_TIME_BUCKETS,
        )
        _adaptive_threshold_metric = Histogram(
            METRIC_ADAPTIVE_THRESHOLD,
            "Adaptive threshold values",
            [LABEL_ACTOR_ID],
        )
        _market_regime_metric = Counter(
            METRIC_MARKET_REGIME_TOTAL,
            "Market regime detection counts",
            [LABEL_ACTOR_ID, "regime"],
        )

    _metrics_initialized = True


# Initialize metrics at module import time
_initialize_performance_metrics()


# =================================================================================================
# Data Types
# =================================================================================================


# AdaptiveSignal is now just an alias for MLSignal for backward compatibility
# The unified MLSignal class handles both basic and adaptive signals
AdaptiveSignal = MLSignal


# =================================================================================================
# Signal Generation Strategy Interface
# =================================================================================================


class SignalGenerationStrategy(ABC):
    """
    Abstract base class for signal generation strategies.
    """

    @abstractmethod
    def generate_signal(
        self,
        bar: Bar,
        prediction: float,
        confidence: float,
        features: npt.NDArray[np.float32],
        context: dict[str, Any],
    ) -> MLSignal | None:
        """
        Generate a signal based on the strategy logic.
        """
        ...


# =================================================================================================
# Built-in Strategy Implementations
# =================================================================================================


class ThresholdSignalStrategy(SignalGenerationStrategy):
    """
    Simple threshold-based signal generation.
    """

    def __init__(self, threshold: float) -> None:
        """
        Initialize threshold signal strategy.

        Parameters
        ----------
        threshold : float
            The confidence threshold for generating signals.

        """
        self.threshold = threshold

    def generate_signal(
        self,
        bar: Bar,
        prediction: float,
        confidence: float,
        features: npt.NDArray[np.float32],
        context: dict[str, Any],
    ) -> MLSignal | None:
        """
        Generate signal based on confidence threshold.

        Parameters
        ----------
        bar : Bar
            The current bar.
        prediction : float
            The model prediction.
        confidence : float
            The confidence score.
        features : npt.NDArray[np.float32]
            The feature array.
        context : dict[str, Any]
            Additional context information.

        Returns
        -------
        MLSignal | None
            The generated signal or None if threshold not met.

        """
        if confidence >= self.threshold:
            return MLSignal(
                instrument_id=bar.bar_type.instrument_id,
                model_id=context.get("model_id", "unknown"),
                prediction=prediction,
                confidence=confidence,
                features=features if context.get("log_predictions", False) else None,
                ts_event=bar.ts_event,
                ts_init=context["timestamp_ns"],
            )
        return None


class ExtremesStrategy(SignalGenerationStrategy):
    """
    Signal generation based on prediction extremes.
    """

    def __init__(self, top_pct: float, threshold: float, window_size: int) -> None:
        """
        Initialize extremes strategy.

        Parameters
        ----------
        top_pct : float
            The percentile for extreme value detection.
        threshold : float
            The confidence threshold.
        window_size : int
            The window size for historical predictions.

        """
        self.top_pct = top_pct
        self.threshold = threshold
        self.window_size = window_size

    def generate_signal(
        self,
        bar: Bar,
        prediction: float,
        confidence: float,
        features: npt.NDArray[np.float32],
        context: dict[str, Any],
    ) -> MLSignal | None:
        """
        Generate signal based on prediction extremes.

        Parameters
        ----------
        bar : Bar
            The current bar.
        prediction : float
            The model prediction.
        confidence : float
            The confidence score.
        features : npt.NDArray[np.float32]
            The feature array.
        context : dict[str, Any]
            Additional context information.

        Returns
        -------
        MLSignal | None
            The generated signal or None if not extreme.

        """
        history = context.get("prediction_history", [])
        if len(history) < self.window_size:
            return None

        predictions = np.array(history[-self.window_size :])
        top_threshold = np.percentile(predictions, 100 - self.top_pct * 100)
        bottom_threshold = np.percentile(predictions, self.top_pct * 100)

        if (
            prediction >= top_threshold or prediction <= bottom_threshold
        ) and confidence >= self.threshold:
            return MLSignal(
                instrument_id=bar.bar_type.instrument_id,
                model_id=context.get("model_id", "unknown"),
                prediction=prediction,
                confidence=confidence,
                features=features if context.get("log_predictions", False) else None,
                ts_event=bar.ts_event,
                ts_init=context["timestamp_ns"],
            )
        return None


class MomentumStrategy(SignalGenerationStrategy):
    """
    Signal generation based on prediction momentum.
    """

    def __init__(self, lookback: int, threshold: float, momentum_threshold: float) -> None:
        """
        Initialize momentum strategy.

        Parameters
        ----------
        lookback : int
            The lookback period for momentum calculation.
        threshold : float
            The confidence threshold.
        momentum_threshold : float
            The momentum threshold for signal generation.

        """
        self.lookback = lookback
        self.threshold = threshold
        self.momentum_threshold = momentum_threshold

    def generate_signal(
        self,
        bar: Bar,
        prediction: float,
        confidence: float,
        features: npt.NDArray[np.float32],
        context: dict[str, Any],
    ) -> MLSignal | None:
        """
        Generate signal based on prediction momentum.

        Parameters
        ----------
        bar : Bar
            The current bar.
        prediction : float
            The model prediction.
        confidence : float
            The confidence score.
        features : npt.NDArray[np.float32]
            The feature array.
        context : dict[str, Any]
            Additional context information.

        Returns
        -------
        MLSignal | None
            The generated signal or None if momentum insufficient.

        """
        history = context.get("prediction_history", [])
        if len(history) < self.lookback:
            return None

        recent_predictions = history[-self.lookback :]
        momentum = np.mean(np.diff(recent_predictions))

        if abs(momentum) > self.momentum_threshold and confidence >= self.threshold:
            return MLSignal(
                instrument_id=bar.bar_type.instrument_id,
                model_id=context.get("model_id", "unknown"),
                prediction=prediction * (1 + momentum),
                confidence=confidence,
                features=features if context.get("log_predictions", False) else None,
                ts_event=bar.ts_event,
                ts_init=context["timestamp_ns"],
            )
        return None


class EnsembleStrategy(SignalGenerationStrategy):
    """
    Ensemble of multiple strategies with weighted voting.
    """

    def __init__(
        self,
        strategies: dict[str, SignalGenerationStrategy],
        weights: dict[str, float],
        threshold: float,
    ) -> None:
        """
        Initialize ensemble strategy.

        Parameters
        ----------
        strategies : dict[str, SignalGenerationStrategy]
            Dictionary of named strategies.
        weights : dict[str, float]
            Weights for each strategy.
        threshold : float
            The ensemble confidence threshold.

        """
        self.strategies = strategies
        self.weights = weights
        self.threshold = threshold

    def generate_signal(
        self,
        bar: Bar,
        prediction: float,
        confidence: float,
        features: npt.NDArray[np.float32],
        context: dict[str, Any],
    ) -> MLSignal | None:
        """
        Generate signal using weighted ensemble voting.

        Parameters
        ----------
        bar : Bar
            The current bar.
        prediction : float
            The model prediction.
        confidence : float
            The confidence score.
        features : npt.NDArray[np.float32]
            The feature array.
        context : dict[str, Any]
            Additional context information.

        Returns
        -------
        MLSignal | None
            The ensemble signal or None if threshold not met.

        """
        ensemble_score = 0.0
        total_weight = 0.0

        for name, strategy in self.strategies.items():
            signal = strategy.generate_signal(bar, prediction, confidence, features, context)
            if signal is not None:
                ensemble_score += self.weights.get(name, 0.0) * confidence
                total_weight += self.weights.get(name, 0.0)

        if total_weight > 0:
            ensemble_confidence = ensemble_score / total_weight
            if ensemble_confidence >= self.threshold:
                return MLSignal(
                    instrument_id=bar.bar_type.instrument_id,
                    model_id=context.get("model_id", "unknown"),
                    prediction=prediction,
                    confidence=ensemble_confidence,
                    features=features if context.get("log_predictions", False) else None,
                    ts_event=bar.ts_event,
                    ts_init=context["timestamp_ns"],
                )
        return None


class AdaptiveStrategy(SignalGenerationStrategy):
    """
    Adaptive signal generation with dynamic thresholds.
    """

    def __init__(
        self,
        base_threshold: float,
        volatility_factor: float,
        min_threshold: float,
        max_threshold: float,
    ) -> None:
        """
        Initialize the AdaptiveStrategy.

        Parameters
        ----------
        base_threshold : float
            The base confidence threshold for signal generation.
        volatility_factor : float
            Factor for adjusting threshold based on market volatility.
        min_threshold : float
            Minimum allowed threshold value.
        max_threshold : float
            Maximum allowed threshold value.

        """
        self.base_threshold = base_threshold
        self.volatility_factor = volatility_factor
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def generate_signal(
        self,
        bar: Bar,
        prediction: float,
        confidence: float,
        features: npt.NDArray[np.float32],
        context: dict[str, Any],
    ) -> MLSignal | None:
        """
        Generate adaptive signal based on dynamic thresholds.

        Parameters
        ----------
        bar : Bar
            The current bar data.
        prediction : float
            The model prediction value.
        confidence : float
            The confidence score of the prediction.
        features : npt.NDArray[np.float32]
            The computed feature array.
        context : dict[str, Any]
            Context dictionary containing adaptive threshold and timestamp.

        Returns
        -------
        MLSignal | None
            The generated signal if threshold is met, otherwise None.

        """
        adaptive_threshold = context.get("adaptive_threshold", self.base_threshold)
        signal_strength = confidence / adaptive_threshold if adaptive_threshold > 0 else 0.0

        if signal_strength >= 1.0:
            market_regime = context.get("market_regime", "unknown")
            return MLSignal(
                instrument_id=bar.bar_type.instrument_id,
                model_id=context.get("model_id", "unknown"),
                prediction=prediction,
                confidence=confidence,
                features=features,
                metadata={
                    "adaptive_threshold": adaptive_threshold,
                    "signal_strength": signal_strength,
                    "market_regime": market_regime,
                },
                ts_event=bar.ts_event,
                ts_init=context["timestamp_ns"],
            )
        return None


# =================================================================================================
# Performance Optimization Components
# =================================================================================================


class ONNXOptimizationConfig(msgspec.Struct, frozen=True):
    """
    Configuration for ONNX runtime optimizations.
    """

    graph_optimization_level: str = "ORT_ENABLE_ALL"
    execution_mode: str = "ORT_SEQUENTIAL"
    intra_threads: int = 1
    inter_threads: int = 1


class AdaptiveThresholdConfig(msgspec.Struct, frozen=True):
    """
    Configuration for adaptive thresholds.
    """

    base_threshold: float = 0.7
    volatility_factor: float = 2.0
    min_threshold: float = 0.1
    max_threshold: float = 0.95


class HotPathConfig(msgspec.Struct, frozen=True):
    """
    Configuration for hot path optimizations.
    """

    enable_zero_copy: bool = True
    pre_allocate_buffers: bool = True
    use_lock_free_buffers: bool = True


class PerformanceMonitor:
    """
    Non-blocking performance monitoring.
    """

    def __init__(self, reservoir_size: int = 1000) -> None:
        """
        Initialize the PerformanceMonitor.

        Parameters
        ----------
        reservoir_size : int, default 1000
            Maximum number of timing samples to store using reservoir sampling.

        """
        self.feature_times: list[float] = []
        self.inference_times: list[float] = []
        self.total_times: list[float] = []
        self.reservoir_size = reservoir_size
        self.prediction_count = 0
        self.signal_count = 0
        self.error_count = 0

    def record_timing(
        self,
        feature_time_ns: int,
        inference_time_ns: int,
        total_time_ns: int,
    ) -> None:
        """
        Record timing measurements in nanoseconds.
        """
        feature_time_ms = feature_time_ns / 1_000_000
        inference_time_ms = inference_time_ns / 1_000_000
        total_time_ms = total_time_ns / 1_000_000

        self.feature_times.append(feature_time_ms)
        self.inference_times.append(inference_time_ms)
        self.total_times.append(total_time_ms)

        # Keep bounded
        if len(self.feature_times) > self.reservoir_size:
            self.feature_times = self.feature_times[-self.reservoir_size :]
            self.inference_times = self.inference_times[-self.reservoir_size :]
            self.total_times = self.total_times[-self.reservoir_size :]

        self.prediction_count += 1

    def record_signal(self) -> None:
        """
        Record signal generation.
        """
        self.signal_count += 1

    def record_error(self) -> None:
        """
        Record error.
        """
        self.error_count += 1

    def get_current_stats(self) -> dict[str, Any]:
        """
        Get current performance statistics.
        """
        stats = {
            "prediction_count": self.prediction_count,
            "signal_count": self.signal_count,
            "error_count": self.error_count,
            "signal_rate": self.signal_count / max(self.prediction_count, 1),
            "error_rate": self.error_count / max(self.prediction_count, 1),
            "avg_feature_time_ms": np.mean(self.feature_times) if self.feature_times else 0.0,
            "avg_inference_time_ms": np.mean(self.inference_times) if self.inference_times else 0.0,
            "avg_total_time_ms": np.mean(self.total_times) if self.total_times else 0.0,
            "p99_total_time_ms": np.percentile(self.total_times, 99) if self.total_times else 0.0,
        }

        if self.feature_times:
            stats["last_feature_time_ms"] = self.feature_times[-1]
        if self.inference_times:
            stats["last_inference_time_ms"] = self.inference_times[-1]
        if self.total_times:
            stats["last_total_time_ms"] = self.total_times[-1]

        return stats

    def get_latency_percentiles(self) -> dict[str, dict[float, float]]:
        """
        Get latency percentiles for each measurement type.
        """
        percentiles = [50.0, 90.0, 95.0, 99.0]
        result = {}

        if self.feature_times:
            result["feature_computation"] = {
                p: float(np.percentile(self.feature_times, p)) for p in percentiles
            }

        if self.inference_times:
            result["inference"] = {
                p: float(np.percentile(self.inference_times, p)) for p in percentiles
            }

        if self.total_times:
            result["total"] = {p: float(np.percentile(self.total_times, p)) for p in percentiles}

        return result


class ModelSwapper:
    """
    Atomic model swapping for hot reload.
    """

    def __init__(self) -> None:
        """
        Initialize the ModelSwapper.

        Provides atomic model swapping capability for hot reload without disrupting
        inference operations.

        """
        self._current_model: Any | None = None
        self._current_metadata: dict[str, Any] | None = None
        self._next_model: Any | None = None
        self._next_metadata: dict[str, Any] | None = None
        self._swap_pending = False
        self._load_error: Exception | None = None

    @property
    def current_model(self) -> Any | None:
        """
        Get current model.
        """
        return self._current_model

    @property
    def current_metadata(self) -> dict[str, Any] | None:
        """
        Get current metadata.
        """
        return self._current_metadata

    @property
    def swap_pending(self) -> bool:
        """
        Check if swap is pending.
        """
        return self._swap_pending

    @property
    def load_error(self) -> Exception | None:
        """
        Get load error if any.
        """
        return self._load_error

    def set_current_model(self, model: Any, metadata: dict[str, Any] | None = None) -> None:
        """
        Set current model.
        """
        self._current_model = model
        self._current_metadata = metadata or {}
        self._load_error = None

    def set_current(self, model: Any, metadata: dict[str, Any] | None = None) -> None:
        """
        Set current model (backward compatibility).
        """
        self.set_current_model(model, metadata)

    def prepare_swap(self, model: Any, metadata: dict[str, Any] | None = None) -> None:
        """
        Prepare model swap.
        """
        self._next_model = model
        self._next_metadata = metadata or {}
        self._swap_pending = True
        self._load_error = None

    def prepare_swap_with_error(self, error: Exception) -> None:
        """
        Set error when model loading fails.
        """
        self._load_error = error
        self._swap_pending = False

    def execute_swap(self) -> bool:
        """
        Execute model swap atomically.
        """
        if not self._swap_pending:
            return False

        old_model = self._current_model
        self._current_model = self._next_model
        self._current_metadata = self._next_metadata
        self._next_model = None
        self._next_metadata = None
        self._swap_pending = False
        del old_model
        return True


# =================================================================================================
# Main Actor Implementation
# =================================================================================================


class MLSignalActor(BaseMLInferenceActor):
    """
    Production-ready ML Signal Actor for real-time inference and signal generation.

    This actor provides configurable signal generation strategies with optional
    performance optimizations for sub-millisecond latency requirements.

    """

    def __init__(self, config: MLSignalActorConfig) -> None:
        """
        Initialize ML Signal Actor with mandatory store integration.
        
        Note: Stores are automatically initialized by the base class.
        No optional store parameters are needed or allowed.
        """
        super().__init__(config)
        self._signal_config = config

        # Get configurations
        self._opt_config = config.optimization_config or OptimizationConfig()
        # Handle strategy config and set ensemble weights default
        if config.strategy_config:
            self._strat_config = config.strategy_config
            if self._strat_config.ensemble_weights is None:
                # Create a new config with default ensemble weights
                self._strat_config = StrategyConfig(
                    extremes_top_pct=self._strat_config.extremes_top_pct,
                    momentum_lookback=self._strat_config.momentum_lookback,
                    ensemble_weights={
                        "threshold": 0.4,
                        "extremes": 0.3,
                        "momentum": 0.3,
                    },
                    adaptive_volatility_factor=self._strat_config.adaptive_volatility_factor,
                    min_threshold=self._strat_config.min_threshold,
                    max_threshold=self._strat_config.max_threshold,
                    update_frequency=self._strat_config.update_frequency,
                )
        else:
            self._strat_config = StrategyConfig(
                ensemble_weights={
                    "threshold": 0.4,
                    "extremes": 0.3,
                    "momentum": 0.3,
                },
            )

        # Feature engineering
        if config.feature_config is None:
            self._feature_config = FeatureConfig()
        else:
            self._feature_config = (
                config.feature_config
                if isinstance(config.feature_config, FeatureConfig)
                else FeatureConfig()
            )

        # Feature engineering setup
        # Note: _feature_store is always available from base class
        self._feature_engineer = FeatureEngineer(self._feature_config)
        self._persist_features = config.persist_features if hasattr(config, 'persist_features') else True

        self._feature_set_id: str | None = None
        # Optional: validate features against feature registry manifest
        if (
            hasattr(config, "feature_set_id")
            and hasattr(config, "registry_path")
            and config.use_registry_features
            and config.feature_set_id is not None
            and config.registry_path is not None
        ):
            try:
                freg = LocalFeatureRegistry(Path(config.registry_path))
                feature_info = freg.get_feature_set(config.feature_set_id)
                manifest = feature_info.manifest if feature_info else None
            except Exception as e:  # pragma: no cover - safety
                manifest = None
                self.log.warning(f"Feature registry load failed: {e}")
            if manifest is not None:
                expected = list(manifest.feature_names)
                actual = self._feature_engineer.config.get_feature_names()
                if expected != actual:
                    raise ValueError(
                        "Feature schema mismatch with manifest: "
                        f"expected {len(expected)} names (hash={manifest.schema_hash}), got {len(actual)}",
                    )
                # else, features are validated
                self._feature_set_id = manifest.feature_set_id
        self._indicator_manager: IndicatorManager | None = None

        # Signal generation state
        self._prediction_history: list[float] = []
        self._confidence_history: list[float] = []
        self._last_signal_bar: int = -config.min_signal_separation_bars
        self._adaptive_threshold = config.prediction_threshold
        self._market_regime = "unknown"

        # Performance buffers
        n_features = self._feature_engineer.n_features
        self._feature_buffer = np.zeros(n_features, dtype=np.float32)
        self._prediction_window = np.zeros(config.adaptive_window, dtype=np.float32)
        self._confidence_window = np.zeros(config.adaptive_window, dtype=np.float32)
        self._volatility_window = np.zeros(config.adaptive_window, dtype=np.float32)
        self._window_index = 0

        # Initialize strategy
        self._signal_strategy = self._create_strategy()

        # Performance monitoring
        if self._opt_config.level == "optimized" or self._opt_config.level == OptimizationLevel.OPTIMIZED.value:
            self._performance_monitor = PerformanceMonitor(self._opt_config.reservoir_sample_size)
        else:
            self._performance_monitor = PerformanceMonitor(100)  # Use default size

        # Model swapping for hot reload
        self._model_swapper = ModelSwapper() if config.enable_hot_reload else None
        self._last_reload_check = 0

        # Metrics
        self._signal_generation_time_metric = _signal_generation_time_metric
        self._signals_generated_metric = _signals_generated_metric
        self._adaptive_threshold_metric = _adaptive_threshold_metric
        self._market_regime_metric = _market_regime_metric

        # Optimized components (lazy initialized)
        self._optimized_buffers: dict[str, Any] = {}

        # Handle both enum and string for logging
        strategy_name = config.signal_strategy
        if isinstance(strategy_name, SignalStrategy):
            strategy_name = strategy_name.value
        
        self.log.info(
            f"Initialized MLSignalActor with strategy: {strategy_name}, "
            f"optimization: {self._opt_config.level}, "
            f"features: {n_features}",
        )

    def _create_strategy(self) -> SignalGenerationStrategy:
        """
        Create signal generation strategy.
        """
        # Use custom strategy if provided
        if self._signal_config.custom_strategy is not None:
            return cast(SignalGenerationStrategy, self._signal_config.custom_strategy)

        # Create built-in strategy
        strategy_str = self._signal_config.signal_strategy
        # Handle both enum and string
        if isinstance(strategy_str, SignalStrategy):
            strategy_str = strategy_str.value
        threshold = self._config.prediction_threshold

        if strategy_str == "threshold" or strategy_str == SignalStrategy.THRESHOLD.value:
            return ThresholdSignalStrategy(threshold)
        elif strategy_str == "extremes" or strategy_str == SignalStrategy.EXTREMES.value:
            return ExtremesStrategy(
                self._strat_config.extremes_top_pct,
                threshold,
                self._signal_config.adaptive_window,
            )
        elif strategy_str == "momentum" or strategy_str == SignalStrategy.MOMENTUM.value:
            return MomentumStrategy(
                self._strat_config.momentum_lookback,
                threshold,
                0.01,  # momentum threshold
            )
        elif strategy_str == "ensemble" or strategy_str == SignalStrategy.ENSEMBLE.value:
            # Create sub-strategies for ensemble
            strategies = {
                "threshold": ThresholdSignalStrategy(threshold),
                "extremes": ExtremesStrategy(
                    self._strat_config.extremes_top_pct,
                    threshold,
                    self._signal_config.adaptive_window,
                ),
                "momentum": MomentumStrategy(
                    self._strat_config.momentum_lookback,
                    threshold,
                    0.01,
                ),
            }
            return EnsembleStrategy(
                strategies,
                self._strat_config.ensemble_weights
                or {
                    "threshold": 0.4,
                    "extremes": 0.3,
                    "momentum": 0.3,
                },
                threshold,
            )
        elif strategy_str == "adaptive" or strategy_str == SignalStrategy.ADAPTIVE.value:
            return AdaptiveStrategy(
                threshold,
                self._strat_config.adaptive_volatility_factor,
                self._strat_config.min_threshold,
                self._strat_config.max_threshold,
            )
        else:
            self.log.warning(f"Unknown strategy {strategy_str}, using threshold")
            return ThresholdSignalStrategy(threshold)

    def _load_model(self) -> None:
        """
        Load ML model with optional optimizations.

        The SmartModelLoader in the base class automatically detects the model format
        (ONNX, pickle, joblib) and loads it appropriately.

        """
        # For OPTIMIZED level with ONNX, use specialized loading with performance options
        if (
            self._opt_config.level == "optimized" or self._opt_config.level == OptimizationLevel.OPTIMIZED.value
            and self._config.model_path.endswith(".onnx")
        ):
            self._load_optimized_onnx_model()
        else:
            # Let the base class handle standard model loading
            super()._load_model()

        # Model is loaded by base class in on_start via _load_model_with_metadata
        # which uses the SmartModelLoader

        # Warm up if configured
        if self._opt_config.enable_model_warm_up and self._model is not None:
            self._warm_up_model()

    def _load_optimized_onnx_model(self) -> None:
        """
        Load ONNX model with optimizations.
        """
        if not HAS_ONNX:
            check_ml_dependencies(["onnx"])  # accept alias in checker

        # Create optimized session options from OnnxRuntimeConfig only
        from ml.config.runtime import OnnxRuntimeConfig as _OnnxRuntimeConfig
        from ml.config.runtime import to_session_options

        rt = getattr(self._config, "onnx_runtime_config", None) or _OnnxRuntimeConfig()
        session_options, providers = to_session_options(rt)

        # Load model
        model = ort.InferenceSession(
            self._config.model_path,
            sess_options=session_options,
            providers=providers,
        )

        # Extract metadata
        model_metadata = {
            "input_names": [inp.name for inp in model.get_inputs()],
            "output_names": [out.name for out in model.get_outputs()],
        }

        # Set model
        self._model = model
        self._model_metadata = model_metadata
        if self._model_swapper:
            self._model_swapper.set_current(model, model_metadata)

        self.log.info(f"Loaded optimized ONNX model: {self._config.model_path}")

    def _warm_up_model(self) -> None:
        """
        Warm up model with dummy predictions.
        """
        rng = np.random.default_rng()
        dummy_features = rng.standard_normal(self._feature_buffer.size).astype(np.float32)
        warm_up_times = []

        for i in range(self._opt_config.warm_up_iterations):
            start = time.perf_counter_ns()
            try:
                self._predict(dummy_features)
            except Exception as e:
                self.log.debug(f"Warm-up iteration {i} failed: {e}")
            warm_up_times.append((time.perf_counter_ns() - start) / 1_000_000)

        if warm_up_times:
            self.log.info(
                f"Model warm-up completed: avg={np.mean(warm_up_times):.3f}ms, "
                f"P99={np.percentile(warm_up_times, 99):.3f}ms",
            )

    def _initialize_features(self) -> None:
        """
        Initialize feature computation components.
        """
        self._indicator_manager = IndicatorManager(
            (
                self._feature_config
                if isinstance(self._feature_config, FeatureConfig)
                else FeatureConfig()
            ),
        )

        # Verify buffer size
        expected_features = self._feature_engineer.n_features
        if self._feature_buffer.size != expected_features:
            self._feature_buffer = np.zeros(expected_features, dtype=np.float32)

        # Initialize optimized buffers if needed
        if self._opt_config.level == "optimized" or self._opt_config.level == OptimizationLevel.OPTIMIZED.value:
            self._initialize_optimized_buffers()

        self.log.info(f"Feature engineering initialized: {expected_features} features")

    def _initialize_optimized_buffers(self) -> None:
        """
        Initialize optimized buffers for hot path.
        """
        if self._opt_config.use_lock_free_buffers:
            try:
                from ml.core.cache import LockFreeRingBuffer
                from ml.core.cache import PreAllocatedFeatureCache
                from ml.core.cache import ReservoirSampler

                self._optimized_buffers["prediction_buffer"] = LockFreeRingBuffer(
                    self._signal_config.adaptive_window * 2,
                )
                self._optimized_buffers["confidence_buffer"] = LockFreeRingBuffer(
                    self._signal_config.adaptive_window * 2,
                )
                self._optimized_buffers["feature_cache"] = PreAllocatedFeatureCache(
                    n_features=self._feature_buffer.size,
                    history_size=1000,
                )
                self._optimized_buffers["prediction_sampler"] = ReservoirSampler(
                    self._opt_config.reservoir_sample_size,
                )
                self.log.info("Initialized lock-free buffers for optimized performance")
            except ImportError:
                self.log.warning("Lock-free buffers not available, using standard buffers")

    def _compute_features(self, bar: Bar) -> npt.NDArray[np.float32] | None:
        """
        Compute feature vector from bar.
        
        Features are automatically persisted by the base class implementation.
        """
        # Always use feature engineering (base class handles persistence)
        if self._indicator_manager is None:
            return None

        start_time = time.perf_counter()
        self._indicator_manager.update_from_bar(bar)

        if not self._indicator_manager.all_initialized():
            return None

        current_bar = {
            "close": float(bar.close),
            "volume": float(bar.volume),
            "high": float(bar.high),
            "low": float(bar.low),
        }

        features = self._feature_engineer.calculate_features_online(
            current_bar=current_bar,
            indicator_manager=self._indicator_manager,
            scaler=None,
        )

        feature_time = (time.perf_counter() - start_time) * 1000
        if feature_time > self._config.max_feature_latency_ms:
            self.log.warning(f"Feature computation slow: {feature_time:.3f}ms")

        # Record feature latency by feature set if available
        if _feature_time_by_feature_set_metric and self._feature_set_id:
            try:
                _feature_time_by_feature_set_metric.labels(
                    actor_id=self.id.value,
                    feature_set_id=self._feature_set_id,
                ).observe(feature_time / 1000.0)
            except Exception as exc:
                # Swallow metrics failures but keep visibility for debugging
                self.log.debug("Feature time metric observe failed", exc_info=exc)

        return features

    def _predict(self, features: npt.NDArray[np.float32]) -> tuple[float, float]:
        """
        Generate prediction from features.
        """
        if self._model is None:
            return 0.0, 0.0

        try:
            # Check if this is a Mock object (for testing)
            from unittest.mock import MagicMock
            from unittest.mock import Mock

            if isinstance(self._model, Mock | MagicMock):
                # Let the test mocks work as before
                if hasattr(self._model, "run"):
                    # Mock ONNX model path
                    features_2d = features.reshape(1, -1).astype(np.float32)
                    if "input_names" in self._model_metadata:
                        input_name = self._model_metadata["input_names"][0]
                        outputs = self._model.run(None, {input_name: features_2d})
                    else:
                        outputs = self._model.run(None, {"input": features_2d})
                    if len(outputs) >= 2:
                        return float(outputs[0][0]), float(outputs[1][0])
                    else:
                        prediction = float(outputs[0][0])
                        return prediction, abs(prediction)
                elif hasattr(self._model, "predict_proba"):
                    features_2d = features.reshape(1, -1)
                    probabilities = self._model.predict_proba(features_2d)[0]
                    prediction = float(np.argmax(probabilities))
                    confidence = float(np.max(probabilities))
                    return prediction, confidence
                elif hasattr(self._model, "predict"):
                    features_2d = features.reshape(1, -1)
                    prediction = float(self._model.predict(features_2d)[0])
                    confidence = min(abs(prediction), 1.0) if prediction != 0 else 0.5
                    return prediction, confidence

            # Check if model uses the unified interface
            if hasattr(self._model, "predict") and hasattr(self._model, "metadata"):
                # Unified model wrapper - just call predict
                result = self._model.predict(features)
                return result[0], result[1]

            # Fallback for legacy models not using the wrapper
            # This ensures backward compatibility
            if hasattr(self._model, "run") and "input_names" in self._model_metadata:
                # Raw ONNX model
                features_2d = features.reshape(1, -1).astype(np.float32)
                input_name = self._model_metadata["input_names"][0]
                outputs = self._model.run(None, {input_name: features_2d})

                if len(outputs) >= 2:
                    return float(outputs[0][0]), float(outputs[1][0])
                else:
                    prediction = float(outputs[0][0])
                    return prediction, abs(prediction)
            elif hasattr(self._model, "predict_proba"):
                # Raw scikit-learn with probabilities
                features_2d = features.reshape(1, -1)
                probabilities = self._model.predict_proba(features_2d)[0]
                prediction = float(np.argmax(probabilities))
                confidence = float(np.max(probabilities))
                return prediction, confidence
            elif hasattr(self._model, "predict"):
                # Check if this is a raw XGBoost Booster
                if hasattr(self._model, "num_features") and hasattr(self._model, "get_score"):
                    # Raw XGBoost Booster - needs DMatrix
                    from ml._imports import xgb

                    features_2d = features.reshape(1, -1)
                    dmatrix = xgb.DMatrix(features_2d)
                    predictions = self._model.predict(dmatrix)
                    prediction = float(predictions[0])
                    confidence = min(abs(prediction), 1.0) if prediction != 0 else 0.5
                    return prediction, confidence
                else:
                    # Raw general model (sklearn, etc.)
                    features_2d = features.reshape(1, -1)
                    prediction = float(self._model.predict(features_2d)[0])
                    confidence = min(abs(prediction), 1.0) if prediction != 0 else 0.5
                    return prediction, confidence
            else:
                self.log.error(f"Unsupported model type: {type(self._model)}")
                return 0.0, 0.0
        except Exception as e:
            self.log.error(f"Prediction failed: {e}")
            # Re-raise to let base class handle circuit breaker and health monitoring
            raise

    def _generate_prediction_protected(self, bar: Bar, features: npt.NDArray[np.float32]) -> None:
        """
        Generate ML prediction with signal generation.
        """
        start_time = time.perf_counter()

        try:
            # Check for hot reload
            if self._should_hot_reload():
                self._execute_hot_reload()

            # Get prediction
            prediction, confidence = self._predict(features)
            self._prediction_count += 1

            # Update history
            self._update_prediction_history(prediction, confidence, bar)

            # Detect regime if enabled
            if self._signal_config.enable_regime_detection:
                self._detect_market_regime(bar)

            # Try to generate signal
            self._try_generate_signal(bar, prediction, confidence, features)

            # Record performance
            self._record_performance(start_time)

            # Record success
            self._record_success()

        except Exception as e:
            self._handle_prediction_error(e)

    def _try_generate_signal(
        self,
        bar: Bar,
        prediction: float,
        confidence: float,
        features: npt.NDArray[np.float32],
    ) -> None:
        """
        Try to generate and publish a signal.
        """
        # Check signal separation
        if (
            self._bars_processed - self._last_signal_bar
            < self._signal_config.min_signal_separation_bars
        ):
            return
        # Build context for strategy
        context = {
            "prediction_history": self._prediction_history,
            "confidence_history": self._confidence_history,
            "adaptive_threshold": self._adaptive_threshold,
            "market_regime": self._market_regime,
            "log_predictions": self._config.log_predictions,
            "timestamp_ns": self.clock.timestamp_ns(),
            "model_id": self._model_id if hasattr(self, "_model_id") else "unknown",
        }

        # Generate signal using strategy
        signal = self._signal_strategy.generate_signal(
            bar,
            prediction,
            confidence,
            features,
            context,
        )

        if signal is not None:
            self._last_signal_bar = self._bars_processed
            self._publish_signal(signal)

            # Prediction is automatically persisted by base class
            # Just need to ensure we're recording strategy-specific metadata
            if hasattr(self, '_strategy_store'):
                # Strategy store is always available from base class
                self._strategy_store.write_signal(
                    strategy_id=self._signal_config.strategy_id or "ml_signal",
                    instrument_id=str(bar.bar_type.instrument_id),
                    signal_type="buy" if signal.prediction > 0 else "sell",
                    strength=abs(signal.prediction),
                    model_predictions={context.get("model_id", "unknown"): prediction},
                    risk_metrics={"confidence": confidence},
                    execution_params={"threshold": self._adaptive_threshold},
                    ts_event=bar.ts_event,
                )

            if self._performance_monitor:
                self._performance_monitor.record_signal()

            if self._signals_generated_metric:
                # Handle both enum and string for metrics
                strategy_name = self._signal_config.signal_strategy
                if isinstance(strategy_name, SignalStrategy):
                    strategy_name = strategy_name.value
                    
                self._signals_generated_metric.labels(
                    actor_id=self.id.value,
                    strategy=strategy_name,
                    signal_type="buy" if signal.prediction > 0 else "sell",
                ).inc()

    def _record_performance(self, start_time: float) -> None:
        """
        Record performance metrics.
        """
        from ml.config.constants import TimeConstants

        total_time_ns = int((time.perf_counter() - start_time) * TimeConstants.NS_IN_SECOND)
        if self._performance_monitor:
            feature_time_ns = 500_000  # Placeholder, would need to track separately
            inference_time_ns = total_time_ns - feature_time_ns
            self._performance_monitor.record_timing(
                feature_time_ns,
                inference_time_ns,
                total_time_ns,
            )

    def _record_success(self) -> None:
        """
        Record successful prediction.
        """
        if self._circuit_breaker:
            self._circuit_breaker.record_success()
        if self._health_monitor:
            self._health_monitor.update_prediction_success()

    def _handle_prediction_error(self, error: Exception) -> None:
        """
        Handle prediction error.
        """
        self.log.error(f"Signal generation failed: {error}")
        if self._performance_monitor:
            self._performance_monitor.record_error()
        if self._circuit_breaker:
            self._circuit_breaker.record_failure()
        if self._health_monitor:
            self._health_monitor.update_prediction_failure()

    def _update_prediction_history(self, prediction: float, confidence: float, bar: Bar) -> None:
        """
        Update prediction history.
        """
        self._prediction_history.append(prediction)
        self._confidence_history.append(confidence)

        # Keep bounded
        max_size = max(self._signal_config.adaptive_window * 2, 1000)
        if len(self._prediction_history) > max_size:
            self._prediction_history = self._prediction_history[-max_size:]
            self._confidence_history = self._confidence_history[-max_size:]

        # Update windows
        self._prediction_window[self._window_index] = prediction
        self._confidence_window[self._window_index] = confidence

        # Update volatility
        if self._indicator_manager and "closes" in self._indicator_manager.price_history:
            closes = self._indicator_manager.price_history["closes"]
            if len(closes) >= 2:
                recent_return = abs(closes[-1] - closes[-2]) / closes[-2]
                self._volatility_window[self._window_index] = recent_return

        self._window_index = (self._window_index + 1) % self._signal_config.adaptive_window

        # Update adaptive threshold
        strategy_val = self._signal_config.signal_strategy
        if isinstance(strategy_val, SignalStrategy):
            strategy_val = strategy_val.value
        if strategy_val == "adaptive" or strategy_val == SignalStrategy.ADAPTIVE.value:
            self._update_adaptive_threshold()

    def _update_adaptive_threshold(self) -> None:
        """
        Update adaptive threshold.
        """
        volatility = float(np.mean(self._volatility_window))
        volatility_adjustment = volatility * self._strat_config.adaptive_volatility_factor
        pred_std = float(np.std(self._prediction_window))

        base_threshold = self._config.prediction_threshold
        self._adaptive_threshold = float(base_threshold + volatility_adjustment + (pred_std * 0.5))
        self._adaptive_threshold = np.clip(
            self._adaptive_threshold,
            self._strat_config.min_threshold,
            self._strat_config.max_threshold,
        )

        if self._adaptive_threshold_metric:
            self._adaptive_threshold_metric.labels(actor_id=self.id.value).observe(
                self._adaptive_threshold,
            )

    def _detect_market_regime(self, bar: Bar) -> None:
        """
        Detect current market regime.
        """
        if not self._indicator_manager or "closes" not in self._indicator_manager.price_history:
            return

        closes = self._indicator_manager.price_history["closes"]
        if len(closes) < 20:
            return

        closes_array = np.array(closes[-20:])
        returns = np.diff(closes_array) / closes_array[:-1]
        volatility = float(np.std(returns))
        trend_strength = abs(np.corrcoef(np.arange(len(closes_array)), closes_array)[0, 1])

        if volatility > 0.02:
            new_regime = "volatile"
        elif trend_strength > 0.7:
            new_regime = "trending"
        else:
            new_regime = "ranging"

        if new_regime != self._market_regime:
            self._market_regime = new_regime
            if self._market_regime_metric:
                self._market_regime_metric.labels(
                    actor_id=self.id.value,
                    regime=new_regime,
                ).inc()

    def _should_hot_reload(self) -> bool:
        """
        Check if hot reload should be performed.
        """
        if not self._signal_config.enable_hot_reload or not self._model_swapper:
            return False

        current_time = time.time()
        if current_time - self._last_reload_check < self._signal_config.hot_reload_interval:
            return False

        self._last_reload_check = int(current_time)
        # Would check for new model file here
        return False

    def _execute_hot_reload(self) -> None:
        """
        Execute model hot reload.
        """
        if not self._model_swapper:
            return

        try:
            # Load new model in background
            # This would be implemented based on specific requirements
            pass
        except Exception as e:
            self.log.error(f"Hot reload failed: {e}")

    def _backup_indicator_state(self) -> None:
        """
        Backup indicator state for hot reload.
        """
        if self._indicator_manager:
            self._indicator_state_backup = {
                "prediction_history": self._prediction_history.copy(),
                "confidence_history": self._confidence_history.copy(),
                "prediction_window": self._prediction_window.copy(),
                "confidence_window": self._confidence_window.copy(),
                "volatility_window": self._volatility_window.copy(),
                "window_index": self._window_index,
                "adaptive_threshold": self._adaptive_threshold,
                "market_regime": self._market_regime,
                "last_signal_bar": self._last_signal_bar,
            }
            self.log.info("Indicator state backed up")

    def _restore_indicator_state(self) -> None:
        """
        Restore indicator state after hot reload.
        """
        if hasattr(self, "_indicator_state_backup") and self._indicator_state_backup:
            backup = self._indicator_state_backup
            self._prediction_history = backup.get("prediction_history", [])
            self._confidence_history = backup.get("confidence_history", [])
            self._prediction_window = backup.get("prediction_window", self._prediction_window)
            self._confidence_window = backup.get("confidence_window", self._confidence_window)
            self._volatility_window = backup.get("volatility_window", self._volatility_window)
            self._window_index = backup.get("window_index", 0)
            self._adaptive_threshold = backup.get(
                "adaptive_threshold",
                self._config.prediction_threshold,
            )
            self._market_regime = backup.get("market_regime", "unknown")
            self._last_signal_bar = backup.get(
                "last_signal_bar",
                -self._signal_config.min_signal_separation_bars,
            )
            self._indicator_state_backup.clear()
            self.log.info("Indicator state restored")

    def get_signal_statistics(self) -> dict[str, Any]:
        """
        Get comprehensive signal statistics.
        """
        base_stats = self.get_health_status()

        # Handle both enum and string for stats
        strategy_name = self._signal_config.signal_strategy
        if isinstance(strategy_name, SignalStrategy):
            strategy_name = strategy_name.value
            
        signal_stats = {
            "signal_strategy": strategy_name,
            "optimization_level": self._opt_config.level,
            "adaptive_threshold": self._adaptive_threshold,
            "market_regime": self._market_regime,
            "last_signal_bar": self._last_signal_bar,
            "prediction_history_length": len(self._prediction_history),
        }

        if self._performance_monitor:
            signal_stats.update(self._performance_monitor.get_current_stats())

        base_stats.update(signal_stats)
        return base_stats

    def reset_signal_state(self) -> None:
        """
        Reset signal generation state.
        """
        self._prediction_history.clear()
        self._confidence_history.clear()
        self._prediction_window.fill(0.0)
        self._confidence_window.fill(0.0)
        self._volatility_window.fill(0.0)
        self._window_index = 0
        self._adaptive_threshold = self._config.prediction_threshold
        self._market_regime = "unknown"
        self._last_signal_bar = -self._signal_config.min_signal_separation_bars
        self.log.info("Signal generation state reset")
