# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may not use this file at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
ML Signal Actor for real-time inference and signal generation.

This module provides a production-ready ML signal actor that performs real-time
inference on market data and generates trading signals with configurable strategies. It
follows Nautilus Trader's hot/cold path architecture and maintains sub-millisecond
performance.

"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

import numpy as np

from ml.actors.base import BaseMLInferenceActor
from ml.actors.base import MLSignal
from ml.common.metrics import Counter
from ml.common.metrics import Histogram
from ml.config.base import MLActorConfig
from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.features.engineering import IndicatorManager
from nautilus_trader.common.config import NonNegativeFloat
from nautilus_trader.common.config import PositiveFloat
from nautilus_trader.common.config import PositiveInt
from nautilus_trader.core.data import Data
from nautilus_trader.model.data import Bar
from nautilus_trader.model.identifiers import InstrumentId


class SignalStrategy(Enum):
    """
    Signal generation strategy enumeration.

    Defines different approaches for converting model predictions into trading signals.

    """

    THRESHOLD = "threshold"
    EXTREMES = "extremes"
    MOMENTUM = "momentum"
    ENSEMBLE = "ensemble"
    ADAPTIVE = "adaptive"


class AdaptiveSignal(Data):
    """
    Adaptive ML signal with dynamic thresholds.

    Extends the base MLSignal with adaptive threshold information for
    sophisticated signal generation strategies.

    Parameters
    ----------
    instrument_id : InstrumentId
        The instrument the signal is for.
    prediction : float
        The model prediction value.
    confidence : float
        The base confidence score.
    adaptive_threshold : float
        The dynamically adjusted threshold.
    signal_strength : float
        The signal strength after adaptive adjustment.
    market_regime : str
        The detected market regime ("trending", "ranging", "volatile").
    ts_event : int
        The UNIX timestamp (nanoseconds) when the signal was generated.
    ts_init : int
        The UNIX timestamp (nanoseconds) when the object was initialized.

    """

    def __init__(
        self,
        instrument_id: InstrumentId,
        prediction: float,
        confidence: float,
        adaptive_threshold: float,
        signal_strength: float,
        market_regime: str,
        ts_event: int = 0,
        ts_init: int = 0,
    ) -> None:
        """
        Initialize adaptive ML signal.
        """
        self.instrument_id = instrument_id
        self.prediction = prediction
        self.confidence = confidence
        self.adaptive_threshold = adaptive_threshold
        self.signal_strength = signal_strength
        self.market_regime = market_regime
        self._ts_event = ts_event
        self._ts_init = ts_init

    @property
    def ts_event(self) -> int:
        """
        Return event timestamp.
        """
        return self._ts_event

    @property
    def ts_init(self) -> int:
        """
        Return initialization timestamp.
        """
        return self._ts_init


class MLSignalActorConfig(MLActorConfig, kw_only=True, frozen=True):
    """
    Configuration for ML Signal Actor with advanced signal generation strategies.

    Parameters
    ----------
    signal_strategy : SignalStrategy, default SignalStrategy.THRESHOLD
        The signal generation strategy to use.
    extremes_top_pct : NonNegativeFloat, default 0.1
        Percentage of top/bottom predictions to consider for extremes strategy.
    momentum_lookback : PositiveInt, default 5
        Number of bars to look back for momentum calculation.
    ensemble_weights : dict[str, float], optional
        Weights for ensemble strategy combining multiple approaches.
    adaptive_window : PositiveInt, default 20
        Window size for adaptive threshold calculation.
    adaptive_volatility_factor : PositiveFloat, default 2.0
        Factor to adjust threshold based on market volatility.
    min_signal_separation_bars : PositiveInt, default 3
        Minimum bars between signals to prevent over-trading.
    feature_importance_threshold : NonNegativeFloat, default 0.01
        Minimum feature importance to include in signal generation.
    enable_regime_detection : bool, default True
        Whether to enable market regime detection for adaptive strategies.

    """

    signal_strategy: SignalStrategy = SignalStrategy.THRESHOLD
    extremes_top_pct: NonNegativeFloat = 0.1
    momentum_lookback: PositiveInt = 5
    ensemble_weights: dict[str, float] | None = None
    adaptive_window: PositiveInt = 20
    adaptive_volatility_factor: PositiveFloat = 2.0
    min_signal_separation_bars: PositiveInt = 3
    feature_importance_threshold: NonNegativeFloat = 0.01
    enable_regime_detection: bool = True


class MLSignalActor(BaseMLInferenceActor):
    """
    Production-ready ML Signal Actor for real-time inference and signal generation.

    This actor extends BaseMLInferenceActor with sophisticated signal generation
    strategies, adaptive thresholds, and comprehensive monitoring. It maintains
    sub-millisecond performance while providing enterprise-grade features.

    Key Features:
    - Multiple signal generation strategies (threshold, extremes, momentum, ensemble, adaptive)
    - Market regime detection for adaptive thresholds
    - Feature importance analysis
    - Signal separation to prevent over-trading
    - Comprehensive metrics and monitoring
    - Circuit breaker protection
    - Model hot-reloading with state preservation

    Performance Requirements:
    - Feature computation: <500μs
    - Model inference: <2ms
    - End-to-end signal generation: <5ms
    - Memory stable over 24h operation

    """

    def __init__(self, config: MLSignalActorConfig) -> None:
        """
        Initialize ML Signal Actor.

        Parameters
        ----------
        config : MLSignalActorConfig
            Configuration for the signal actor.

        """
        super().__init__(config)
        self._signal_config = config

        # Feature engineering components
        # Use FeatureConfig directly (it inherits from MLFeatureConfig)
        if config.feature_config is None:
            self._feature_config = FeatureConfig()
        else:
            # Ensure we have a FeatureConfig instance
            if isinstance(config.feature_config, FeatureConfig):
                self._feature_config = config.feature_config
            else:
                # If it's just MLFeatureConfig, create a default FeatureConfig
                self._feature_config = FeatureConfig()
        self._feature_engineer = FeatureEngineer(self._feature_config)
        self._indicator_manager: IndicatorManager | None = None

        # Signal generation state
        self._prediction_history: list[float] = []
        self._confidence_history: list[float] = []
        self._last_signal_bar: int = -config.min_signal_separation_bars
        self._adaptive_threshold = config.prediction_threshold
        self._market_regime = "unknown"

        # Performance buffers (pre-allocated for hot path)
        self._feature_buffer = np.zeros(self._feature_engineer.n_features, dtype=np.float32)
        self._prediction_window = np.zeros(config.adaptive_window, dtype=np.float32)
        self._confidence_window = np.zeros(config.adaptive_window, dtype=np.float32)
        self._volatility_window = np.zeros(config.adaptive_window, dtype=np.float32)
        self._window_index = 0

        # Ensemble weights setup
        if config.ensemble_weights is None:
            self._ensemble_weights = {
                "threshold": 0.4,
                "extremes": 0.3,
                "momentum": 0.3,
            }
        else:
            self._ensemble_weights = config.ensemble_weights

        # Enhanced metrics for signal generation
        self._signal_generation_time_metric = Histogram(
            "nautilus_ml_signal_generation_seconds",
            "Signal generation latency in seconds",
            ["actor_id", "strategy"],
            buckets=[0.0001, 0.0005, 0.001, 0.002, 0.005],
        )
        self._signals_generated_metric = Counter(
            "nautilus_ml_signals_generated_total",
            "Total number of signals generated",
            ["actor_id", "strategy", "signal_type"],
        )
        self._adaptive_threshold_metric = Histogram(
            "nautilus_ml_adaptive_threshold",
            "Adaptive threshold values",
            ["actor_id"],
        )
        self._market_regime_metric = Counter(
            "nautilus_ml_market_regime_total",
            "Market regime detection counts",
            ["actor_id", "regime"],
        )

        self.log.info(
            f"Initialized MLSignalActor with strategy: {config.signal_strategy.value}, "
            f"features: {self._feature_engineer.n_features}, "
            f"adaptive_window: {config.adaptive_window}",
        )

    def _load_model(self) -> None:
        """
        Load ML model from configured path.

        This method is called during initialization and hot-reloads. The actual model
        loading is handled by the base class model loader.

        """
        # Model loading is handled by base class _load_model_with_metadata
        # This method can be used for additional model-specific setup
        if self._model is not None:
            self.log.info(f"Model loaded successfully: {type(self._model).__name__}")
        else:
            self.log.warning("Model is None after loading")

    def _initialize_features(self) -> None:
        """
        Initialize feature computation components.

        Sets up the indicator manager and pre-allocates all buffers needed for real-time
        feature computation.

        """
        # Initialize indicator manager with feature configuration
        # IndicatorManager expects FeatureConfig, ensure we have the right type
        if isinstance(self._feature_config, FeatureConfig):
            self._indicator_manager = IndicatorManager(self._feature_config)
        else:
            # Create a default FeatureConfig if needed
            feature_config = FeatureConfig()
            self._indicator_manager = IndicatorManager(feature_config)

        # Verify feature buffer size matches configuration
        # Use FeatureEngineer's feature count or configured feature names
        if hasattr(self._feature_config, "feature_names") and self._feature_config.feature_names:
            expected_features = len(self._feature_config.feature_names)
        else:
            expected_features = self._feature_engineer.n_features
        if self._feature_buffer.size != expected_features:
            self._feature_buffer = np.zeros(expected_features, dtype=np.float32)
            self.log.info(f"Resized feature buffer to {expected_features} features")

        self.log.info(
            f"Feature engineering initialized: {expected_features} features, "
            f"{len(self._indicator_manager.indicators)} indicators",
        )

    def _compute_features(self, bar: Bar) -> np.ndarray | None:
        """
        Compute feature vector from current bar with <500μs latency.

        This is the hot path method that must be highly optimized.
        Uses pre-allocated buffers and indicator manager for consistency.

        Parameters
        ----------
        bar : Bar
            Current bar data.

        Returns
        -------
        np.ndarray | None
            Feature vector or None if indicators not ready.

        """
        if self._indicator_manager is None:
            return None

        # Update indicators (optimized Nautilus implementations)
        start_time = time.perf_counter()
        self._indicator_manager.update_from_bar(bar)

        # Check if all indicators are ready
        if not self._indicator_manager.all_initialized():
            return None

        # Prepare current bar data
        current_bar = {
            "close": float(bar.close),
            "volume": float(bar.volume),
            "high": float(bar.high),
            "low": float(bar.low),
        }

        # Compute features using feature engineer (hot path optimized)
        features = self._feature_engineer.calculate_features_online(
            current_bar=current_bar,
            indicator_manager=self._indicator_manager,
            scaler=None,  # No scaling in hot path for performance
        )

        # Track feature computation time
        feature_time = (time.perf_counter() - start_time) * 1000
        if feature_time > self._config.max_feature_latency_ms:
            self.log.warning(f"Feature computation slow: {feature_time:.3f}ms")

        return features

    def _predict(self, features: np.ndarray) -> tuple[float, float]:
        """
        Generate prediction from feature vector with <2ms latency.

        This method performs model inference and returns both prediction
        and confidence scores optimized for different model types.

        Parameters
        ----------
        features : np.ndarray
            Feature vector for prediction.

        Returns
        -------
        tuple[float, float]
            Tuple of (prediction, confidence) values.

        """
        if self._model is None:
            return 0.0, 0.0

        try:
            # Handle different model types for optimal performance
            if hasattr(self._model, "run"):
                # ONNX model - fastest inference
                return self._predict_onnx(features)
            elif hasattr(self._model, "predict_proba"):
                # Scikit-learn model with probabilities
                return self._predict_sklearn_proba(features)
            elif hasattr(self._model, "predict"):
                # General scikit-learn or XGBoost model
                return self._predict_sklearn(features)
            else:
                self.log.error(f"Unsupported model type: {type(self._model)}")
                return 0.0, 0.0

        except Exception as e:
            self.log.error(f"Prediction failed: {e}")
            return 0.0, 0.0

    def _predict_onnx(self, features: np.ndarray) -> tuple[float, float]:
        """
        ONNX model prediction with optimal performance.
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
            confidence = abs(prediction)  # Use absolute value as confidence

        return prediction, confidence

    def _predict_sklearn_proba(self, features: np.ndarray) -> tuple[float, float]:
        """
        Scikit-learn model with probability output.
        """
        features_2d = features.reshape(1, -1)
        probabilities = self._model.predict_proba(features_2d)[0]

        # For multi-class, use the most confident prediction
        prediction = float(np.argmax(probabilities))
        confidence = float(np.max(probabilities))

        return prediction, confidence

    def _predict_sklearn(self, features: np.ndarray) -> tuple[float, float]:
        """
        General scikit-learn or XGBoost model.
        """
        features_2d = features.reshape(1, -1)
        prediction = float(self._model.predict(features_2d)[0])

        # For regression or models without probability, estimate confidence
        confidence = min(abs(prediction), 1.0) if prediction != 0 else 0.5

        return prediction, confidence

    def _generate_prediction_protected(self, bar: Bar, features: np.ndarray) -> None:
        """
        Generate ML prediction with advanced signal strategies.

        Overrides base class method to add sophisticated signal generation
        with multiple strategies and adaptive thresholds.

        Parameters
        ----------
        bar : Bar
            Current bar data.
        features : np.ndarray
            Computed feature vector.

        """
        start_time = time.perf_counter()

        try:
            # Get prediction from model
            prediction, confidence = self._predict(features)

            # Increment counter early (will also be incremented in _track_performance_metrics)
            self._prediction_count += 1

            # Update prediction history for adaptive strategies
            self._update_prediction_history(prediction, confidence, bar)

            # Detect market regime if enabled
            if self._signal_config.enable_regime_detection:
                self._detect_market_regime(bar)

            # Generate signal based on configured strategy
            signal = self._generate_signal_by_strategy(
                bar=bar,
                prediction=prediction,
                confidence=confidence,
                features=features,
            )

            # Track timing and success
            signal_time = (time.perf_counter() - start_time) * 1000
            self._signal_generation_time_metric.observe(
                signal_time / 1000,
                {"actor_id": self.id.value, "strategy": self._signal_config.signal_strategy.value},
            )

            # Record success in circuit breaker
            if self._circuit_breaker:
                self._circuit_breaker.record_success()

            # Update health monitor
            if self._health_monitor:
                self._health_monitor.update_prediction_success()

            # Publish signal if generated
            if signal is not None:
                self._publish_signal(signal)
                self._signals_generated_metric.inc(
                    1.0,
                    {
                        "actor_id": self.id.value,
                        "strategy": self._signal_config.signal_strategy.value,
                        "signal_type": "buy" if signal.prediction > 0 else "sell",
                    },
                )

            # Track performance metrics
            self._track_performance_metrics(prediction, confidence, signal_time)

        except Exception as e:
            self.log.error(f"Signal generation failed: {e}")

            # Record failure in circuit breaker
            if self._circuit_breaker:
                self._circuit_breaker.record_failure()

            # Update health monitor
            if self._health_monitor:
                self._health_monitor.update_prediction_failure()

    def _update_prediction_history(self, prediction: float, confidence: float, bar: Bar) -> None:
        """
        Update prediction history for adaptive strategies.

        Uses circular buffers for memory efficiency in long-running processes.

        Parameters
        ----------
        prediction : float
            Current prediction value.
        confidence : float
            Current confidence score.
        bar : Bar
            Current bar for volatility calculation.

        """
        # Update history lists (bounded to prevent memory growth)
        self._prediction_history.append(prediction)
        self._confidence_history.append(confidence)

        # Keep history bounded to adaptive_window size
        max_history_size = max(self._signal_config.adaptive_window * 2, 1000)
        if len(self._prediction_history) > max_history_size:
            self._prediction_history = self._prediction_history[-max_history_size:]
            self._confidence_history = self._confidence_history[-max_history_size:]

        # Update circular buffers
        self._prediction_window[self._window_index] = prediction
        self._confidence_window[self._window_index] = confidence

        # Calculate current volatility (simplified)
        if (
            self._indicator_manager is not None
            and "closes" in self._indicator_manager.price_history
            and len(self._indicator_manager.price_history["closes"]) >= 2
        ):
            closes = self._indicator_manager.price_history["closes"]
            recent_return = abs(closes[-1] - closes[-2]) / closes[-2]
            self._volatility_window[self._window_index] = recent_return

        # Advance circular buffer index
        self._window_index = (self._window_index + 1) % self._signal_config.adaptive_window

        # Update adaptive threshold
        if self._signal_config.signal_strategy == SignalStrategy.ADAPTIVE:
            self._update_adaptive_threshold()

    def _update_adaptive_threshold(self) -> None:
        """
        Update adaptive threshold based on market conditions.

        Adjusts threshold based on recent volatility and prediction distribution.

        """
        # Calculate volatility-adjusted threshold
        volatility = float(np.mean(self._volatility_window))
        volatility_adjustment = volatility * self._signal_config.adaptive_volatility_factor

        # Calculate prediction distribution metrics
        pred_std = float(np.std(self._prediction_window))

        # Adaptive threshold formula
        base_threshold = self._config.prediction_threshold
        self._adaptive_threshold = base_threshold + volatility_adjustment + (pred_std * 0.5)

        # Clamp to reasonable bounds
        self._adaptive_threshold = np.clip(self._adaptive_threshold, 0.1, 0.95)

        # Track metric
        self._adaptive_threshold_metric.observe(
            self._adaptive_threshold,
            {"actor_id": self.id.value},
        )

    def _detect_market_regime(self, bar: Bar) -> None:
        """
        Detect current market regime for adaptive strategies.

        Simple regime detection based on volatility and trend characteristics.

        Parameters
        ----------
        bar : Bar
            Current bar data.

        """
        if (
            self._indicator_manager is None
            or "closes" not in self._indicator_manager.price_history
            or len(self._indicator_manager.price_history["closes"]) < 20
        ):
            return

        closes = np.array(self._indicator_manager.price_history["closes"][-20:])

        # Calculate trend and volatility metrics
        returns = np.diff(closes) / closes[:-1]
        volatility = float(np.std(returns))
        trend_strength = abs(np.corrcoef(np.arange(len(closes)), closes)[0, 1])

        # Classify regime
        if volatility > 0.02:  # High volatility threshold
            new_regime = "volatile"
        elif trend_strength > 0.7:  # Strong trend
            new_regime = "trending"
        else:
            new_regime = "ranging"

        # Update regime if changed
        if new_regime != self._market_regime:
            self._market_regime = new_regime
            self._market_regime_metric.inc(
                1.0,
                {
                    "actor_id": self.id.value,
                    "regime": new_regime,
                },
            )
            self.log.debug(f"Market regime changed to: {new_regime}")

    def _generate_signal_by_strategy(
        self,
        bar: Bar,
        prediction: float,
        confidence: float,
        features: np.ndarray,
    ) -> MLSignal | AdaptiveSignal | None:
        """
        Generate signal based on configured strategy.

        Parameters
        ----------
        bar : Bar
            Current bar data.
        prediction : float
            Model prediction.
        confidence : float
            Prediction confidence.
        features : np.ndarray
            Feature vector.

        Returns
        -------
        MLSignal | AdaptiveSignal | None
            Generated signal or None if no signal.

        """
        # Check signal separation
        if (
            self._bars_processed - self._last_signal_bar
            < self._signal_config.min_signal_separation_bars
        ):
            return None

        strategy = self._signal_config.signal_strategy

        if strategy == SignalStrategy.THRESHOLD:
            return self._generate_threshold_signal(bar, prediction, confidence, features)
        elif strategy == SignalStrategy.EXTREMES:
            return self._generate_extremes_signal(bar, prediction, confidence, features)
        elif strategy == SignalStrategy.MOMENTUM:
            return self._generate_momentum_signal(bar, prediction, confidence, features)
        elif strategy == SignalStrategy.ENSEMBLE:
            return self._generate_ensemble_signal(bar, prediction, confidence, features)
        elif strategy == SignalStrategy.ADAPTIVE:
            return self._generate_adaptive_signal(bar, prediction, confidence, features)
        else:
            self.log.error(f"Unknown signal strategy: {strategy}")
            return None

    def _generate_threshold_signal(
        self,
        bar: Bar,
        prediction: float,
        confidence: float,
        features: np.ndarray,
    ) -> MLSignal | None:
        """
        Generate signal using simple threshold strategy.
        """
        if confidence >= self._config.prediction_threshold:
            self._last_signal_bar = self._bars_processed
            return MLSignal(
                instrument_id=bar.bar_type.instrument_id,
                prediction=prediction,
                confidence=confidence,
                features=features if self._config.log_predictions else None,
                ts_event=bar.ts_event,
                ts_init=self.clock.timestamp_ns(),
            )
        return None

    def _generate_extremes_signal(
        self,
        bar: Bar,
        prediction: float,
        confidence: float,
        features: np.ndarray,
    ) -> MLSignal | None:
        """
        Generate signal using extremes strategy (top/bottom percentile).
        """
        if len(self._prediction_history) < self._signal_config.adaptive_window:
            return None

        # Calculate percentile thresholds
        predictions = np.array(self._prediction_history[-self._signal_config.adaptive_window :])
        top_threshold = np.percentile(predictions, 100 - self._signal_config.extremes_top_pct * 100)
        bottom_threshold = np.percentile(predictions, self._signal_config.extremes_top_pct * 100)

        # Generate signal for extreme predictions
        if prediction >= top_threshold or prediction <= bottom_threshold:
            if confidence >= self._config.prediction_threshold:
                self._last_signal_bar = self._bars_processed
                return MLSignal(
                    instrument_id=bar.bar_type.instrument_id,
                    prediction=prediction,
                    confidence=confidence,
                    features=features if self._config.log_predictions else None,
                    ts_event=bar.ts_event,
                    ts_init=self.clock.timestamp_ns(),
                )
        return None

    def _generate_momentum_signal(
        self,
        bar: Bar,
        prediction: float,
        confidence: float,
        features: np.ndarray,
    ) -> MLSignal | None:
        """
        Generate signal using momentum strategy.
        """
        if len(self._prediction_history) < self._signal_config.momentum_lookback:
            return None

        # Calculate prediction momentum
        recent_predictions = self._prediction_history[-self._signal_config.momentum_lookback :]
        momentum = np.mean(np.diff(recent_predictions))

        # Generate signal based on momentum and confidence
        momentum_threshold = 0.01  # Configurable threshold
        if abs(momentum) > momentum_threshold and confidence >= self._config.prediction_threshold:
            self._last_signal_bar = self._bars_processed
            return MLSignal(
                instrument_id=bar.bar_type.instrument_id,
                prediction=prediction * (1 + momentum),  # Adjust prediction by momentum
                confidence=confidence,
                features=features if self._config.log_predictions else None,
                ts_event=bar.ts_event,
                ts_init=self.clock.timestamp_ns(),
            )
        return None

    def _generate_ensemble_signal(
        self,
        bar: Bar,
        prediction: float,
        confidence: float,
        features: np.ndarray,
    ) -> MLSignal | None:
        """
        Generate signal using ensemble of strategies.
        """
        # Get signals from different strategies
        threshold_signal = self._generate_threshold_signal(bar, prediction, confidence, features)
        extremes_signal = self._generate_extremes_signal(bar, prediction, confidence, features)
        momentum_signal = self._generate_momentum_signal(bar, prediction, confidence, features)

        # Calculate weighted ensemble score
        ensemble_score = 0.0
        total_weight = 0.0

        if threshold_signal is not None:
            ensemble_score += self._ensemble_weights.get("threshold", 0.0) * confidence
            total_weight += self._ensemble_weights.get("threshold", 0.0)

        if extremes_signal is not None:
            ensemble_score += self._ensemble_weights.get("extremes", 0.0) * confidence
            total_weight += self._ensemble_weights.get("extremes", 0.0)

        if momentum_signal is not None:
            ensemble_score += self._ensemble_weights.get("momentum", 0.0) * confidence
            total_weight += self._ensemble_weights.get("momentum", 0.0)

        # Generate signal if ensemble score is high enough
        if total_weight > 0:
            ensemble_confidence = ensemble_score / total_weight
            if ensemble_confidence >= self._config.prediction_threshold:
                self._last_signal_bar = self._bars_processed
                return MLSignal(
                    instrument_id=bar.bar_type.instrument_id,
                    prediction=prediction,
                    confidence=ensemble_confidence,
                    features=features if self._config.log_predictions else None,
                    ts_event=bar.ts_event,
                    ts_init=self.clock.timestamp_ns(),
                )
        return None

    def _generate_adaptive_signal(
        self,
        bar: Bar,
        prediction: float,
        confidence: float,
        features: np.ndarray,
    ) -> AdaptiveSignal | None:
        """
        Generate signal using adaptive strategy with dynamic thresholds.
        """
        # Calculate signal strength based on adaptive threshold
        signal_strength = (
            confidence / self._adaptive_threshold if self._adaptive_threshold > 0 else 0.0
        )

        # Generate signal if strength is sufficient
        if signal_strength >= 1.0:  # Signal strength >= 1 means above adaptive threshold
            self._last_signal_bar = self._bars_processed
            return AdaptiveSignal(
                instrument_id=bar.bar_type.instrument_id,
                prediction=prediction,
                confidence=confidence,
                adaptive_threshold=self._adaptive_threshold,
                signal_strength=signal_strength,
                market_regime=self._market_regime,
                ts_event=bar.ts_event,
                ts_init=self.clock.timestamp_ns(),
            )
        return None

    def _track_performance_metrics(
        self,
        prediction: float,
        confidence: float,
        signal_time: float,
    ) -> None:
        """
        Track detailed performance metrics.

        Parameters
        ----------
        prediction : float
            Model prediction.
        confidence : float
            Prediction confidence.
        signal_time : float
            Signal generation time in milliseconds.

        """
        # Update performance counters (already incremented in _generate_prediction_protected)
        # self._prediction_count += 1  # Already incremented earlier
        self._total_inference_time += signal_time

        # Track prediction distribution
        try:
            if not hasattr(self, "_prediction_distribution_metric"):
                self._prediction_distribution_metric = Histogram(
                    "nautilus_ml_prediction_distribution",
                    "Distribution of model predictions",
                    ["actor_id"],
                )

            self._prediction_distribution_metric.observe(
                prediction,
                {"actor_id": self.id.value},
            )

            # Track confidence distribution
            if not hasattr(self, "_confidence_distribution_metric"):
                self._confidence_distribution_metric = Histogram(
                    "nautilus_ml_confidence_distribution",
                    "Distribution of prediction confidence scores",
                    ["actor_id"],
                )

            self._confidence_distribution_metric.observe(
                confidence,
                {"actor_id": self.id.value},
            )
        except Exception as e:
            # Don't fail if metrics can't be created (e.g., in tests)
            if hasattr(self, "log"):
                self.log.debug(f"Could not create metrics: {e}")

        # Log detailed performance if enabled
        try:
            if self._config.log_predictions:
                self.log.debug(
                    f"Prediction: {prediction:.4f}, Confidence: {confidence:.4f}, "
                    f"Signal time: {signal_time:.3f}ms, Strategy: {self._signal_config.signal_strategy.value}",
                )
        except Exception as e:
            # Silently ignore logging errors to prevent disrupting trading
            _ = e  # Acknowledge exception for linting

    def _backup_indicator_state(self) -> None:
        """
        Backup indicator state for preservation during hot reload.

        Saves the current state of all indicators and prediction history for restoration
        after model reload.

        """
        if self._indicator_manager is not None:
            self._indicator_state_backup = {
                "indicators": {},
                "price_history": (
                    self._indicator_manager.price_history.copy() if self._indicator_manager else {}
                ),
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

            # Backup individual indicator states (simplified)
            for name, indicator in self._indicator_manager.indicators.items():
                if hasattr(indicator, "value") and indicator.initialized:
                    self._indicator_state_backup["indicators"][name] = {
                        "value": indicator.value,
                        "initialized": indicator.initialized,
                    }

            self.log.info("Indicator state backed up for hot reload")

    def _restore_indicator_state(self) -> None:
        """
        Restore indicator state after model reload.

        Restores all indicators and prediction history to maintain continuity after hot
        reload.

        """
        if self._indicator_state_backup and self._indicator_manager is not None:
            # Restore prediction history
            self._prediction_history = self._indicator_state_backup.get("prediction_history", [])
            self._confidence_history = self._indicator_state_backup.get("confidence_history", [])

            # Restore prediction windows
            self._prediction_window = self._indicator_state_backup.get(
                "prediction_window",
                np.zeros(self._signal_config.adaptive_window, dtype=np.float32),
            )
            self._confidence_window = self._indicator_state_backup.get(
                "confidence_window",
                np.zeros(self._signal_config.adaptive_window, dtype=np.float32),
            )
            self._volatility_window = self._indicator_state_backup.get(
                "volatility_window",
                np.zeros(self._signal_config.adaptive_window, dtype=np.float32),
            )

            # Restore state variables
            self._window_index = self._indicator_state_backup.get("window_index", 0)
            self._adaptive_threshold = self._indicator_state_backup.get(
                "adaptive_threshold",
                self._config.prediction_threshold,
            )
            self._market_regime = self._indicator_state_backup.get("market_regime", "unknown")
            self._last_signal_bar = self._indicator_state_backup.get(
                "last_signal_bar",
                -self._signal_config.min_signal_separation_bars,
            )

            # Restore price history
            price_history = self._indicator_state_backup.get("price_history", {})
            if price_history and self._indicator_manager is not None:
                self._indicator_manager.price_history = price_history

            self.log.info("Indicator state restored after hot reload")
            self._indicator_state_backup.clear()

    def get_signal_statistics(self) -> dict[str, Any]:
        """
        Get comprehensive signal generation statistics.

        Returns
        -------
        dict[str, Any]
            Dictionary containing detailed signal generation metrics.

        """
        base_stats = self.get_health_status()

        # Add signal-specific statistics
        signal_stats = {
            "signal_strategy": self._signal_config.signal_strategy.value,
            "adaptive_threshold": self._adaptive_threshold,
            "market_regime": self._market_regime,
            "signals_generated": getattr(self, "_signals_generated", 0),
            "last_signal_bar": self._last_signal_bar,
            "prediction_history_length": len(self._prediction_history),
            "feature_buffer_size": self._feature_buffer.size,
            "ensemble_weights": (
                self._ensemble_weights
                if self._signal_config.signal_strategy == SignalStrategy.ENSEMBLE
                else None
            ),
        }

        # Combine statistics
        base_stats.update(signal_stats)
        return base_stats

    def reset_signal_state(self) -> None:
        """
        Reset signal generation state.

        Clears all prediction history and resets adaptive thresholds while preserving
        indicator state.

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
