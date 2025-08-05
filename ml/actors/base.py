# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
Base class for ML inference actors.

This module provides the foundation for building ML-powered actors that can perform
real-time inference on market data while maintaining the performance requirements of
Nautilus Trader's hot path.

"""

from __future__ import annotations

import pickle
import time
from abc import ABC
from abc import abstractmethod
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

# Import metrics utilities
from ml.common.metrics import Counter
from ml.common.metrics import Histogram
from ml.config.base import MLActorConfig
from ml.config.base import MLFeatureConfig
from nautilus_trader.common.actor import Actor
from nautilus_trader.core.data import Data
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import DataType
from nautilus_trader.model.identifiers import InstrumentId


class MLSignal(Data):
    """
    Custom data type for ML predictions.

    Parameters
    ----------
    instrument_id : InstrumentId
        The instrument the prediction is for.
    prediction : float
        The model prediction value.
    confidence : float
        The confidence score for the prediction (0.0 to 1.0).
    features : np.ndarray, optional
        The feature vector used for prediction (for debugging).
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
        features: np.ndarray | None = None,
        ts_event: int = 0,
        ts_init: int = 0,
    ) -> None:
        """
        Initialize a new ML signal data object.

        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument this signal is for.
        prediction : float
            The model's prediction value.
        confidence : float
            The confidence level of the prediction (0.0 to 1.0).
        features : np.ndarray, optional
            The feature values used for this prediction.
        ts_event : int, default 0
            The event timestamp in nanoseconds.
        ts_init : int, default 0
            The initialization timestamp in nanoseconds.

        """
        self.instrument_id = instrument_id
        self.prediction = prediction
        self.confidence = confidence
        self.features = features
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
    "nautilus_ml_predictions_total",
    "Total number of ML predictions made",
    ["actor_id", "model_name"],
)
ml_prediction_latency = Histogram(
    "nautilus_ml_prediction_latency_seconds",
    "Latency of ML predictions in seconds",
    ["actor_id", "model_name"],
)
ml_signal_confidence = Histogram(
    "nautilus_ml_signal_confidence",
    "Distribution of ML signal confidence scores",
    ["actor_id", "model_name"],
)


class BaseMLInferenceActor(Actor, ABC):
    """
    Base class for ML inference actors.

    This class provides a foundation for building ML-powered actors that perform
    real-time inference on market data. It handles feature engineering, model
    loading, and signal publishing while ensuring hot path performance.

    Key principles:
    - All indicators and models are loaded during initialization
    - Feature computation uses pre-allocated numpy arrays
    - No blocking operations in event handlers
    - Memory usage is bounded and predictable

    Parameters
    ----------
    config : MLActorConfig
        The configuration for the ML actor.

    """

    def __init__(self, config: MLActorConfig) -> None:
        """
        Initialize the ML inference actor.

        Parameters
        ----------
        config : MLActorConfig
            The configuration for the ML actor.

        """
        super().__init__(config)
        self._config = config

        # Initialize feature configuration
        self._feature_config = config.feature_config or MLFeatureConfig()

        # Model and inference state
        self._model: Any = None
        self._features_buffer: np.ndarray | None = None
        self._feature_window: deque[np.ndarray] = deque(
            maxlen=self._feature_config.lookback_window,
        )

        # Performance tracking
        self._prediction_count = 0
        self._total_inference_time = 0.0
        self._last_prediction_time = 0

        # Warm-up tracking
        self._bars_processed = 0
        self._is_warmed_up = False

        # Prometheus metrics
        self._inference_latency_metric = ml_prediction_latency
        self._inference_count_metric = ml_predictions_total
        self._inference_errors_metric = Counter(
            "nautilus_ml_inference_errors_total",
            "Total number of ML inference errors",
            ["actor_id", "error_type"],
        )
        self._feature_computation_time_metric = ml_prediction_latency

    def on_start(self) -> None:
        """
        Initialize the actor and subscribe to market data.

        This method is called when the actor starts and handles:
        - Model loading
        - Feature buffer initialization
        - Market data subscription

        """
        self.log.info(f"Starting {self.__class__.__name__}")

        # Load model during initialization (not in hot path)
        self._load_model()

        # Initialize feature buffers
        self._initialize_features()

        # Subscribe to market data
        self.subscribe_bars(self._config.bar_type)

        self.log.info(
            f"ML Actor configured: model={Path(self._config.model_path).name}, "
            f"threshold={self._config.prediction_threshold}, "
            f"warm_up={self._config.warm_up_period}",
        )

    def on_bar(self, bar: Bar) -> None:
        """
        Process new bar data and potentially generate predictions.

        This is the hot path - must be optimized for performance:
        - No memory allocations
        - No blocking operations
        - Bounded computation time

        Parameters
        ----------
        bar : Bar
            The new bar data to process.

        """
        # Track bars for warm-up period
        self._bars_processed += 1

        # Update indicators and compute features
        features = self._compute_features(bar)
        if features is None:
            return  # Indicators not ready

        # Add to rolling window
        self._feature_window.append(features)

        # Check if warmed up
        if not self._is_warmed_up:
            if self._bars_processed >= self._config.warm_up_period:
                self._is_warmed_up = True
                self.log.info("ML Actor warm-up complete, starting predictions")
            else:
                return  # Still warming up

        # Generate prediction
        self._generate_prediction(bar, features)

    def on_stop(self) -> None:
        """
        Log final statistics when the actor stops.
        """
        avg_inference_time = self._total_inference_time / max(self._prediction_count, 1)

        self.log.info(
            f"Stopping {self.__class__.__name__} - "
            f"Predictions: {self._prediction_count}, "
            f"Avg inference time: {avg_inference_time:.2f}ms",
        )

    def _generate_prediction(self, bar: Bar, features: np.ndarray) -> None:
        """
        Generate ML prediction and optionally publish signal.

        This method measures inference time and publishes signals if configured.

        Parameters
        ----------
        bar : Bar
            The current bar data.
        features : np.ndarray
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

            # Check latency requirement
            if inference_time > self._config.max_inference_latency_ms:
                self.log.warning(
                    f"Inference latency exceeded: {inference_time:.2f}ms > "
                    f"{self._config.max_inference_latency_ms}ms",
                )

            # Log prediction if configured
            if self._config.log_predictions:
                self.log.debug(
                    f"Prediction: {prediction:.4f}, confidence: {confidence:.4f}, "
                    f"latency: {inference_time:.2f}ms",
                )

            # Publish signal if confidence meets threshold
            if confidence >= self._config.prediction_threshold and self._config.publish_signals:
                signal = MLSignal(
                    instrument_id=bar.bar_type.instrument_id,
                    prediction=prediction,
                    confidence=confidence,
                    features=features if self._config.log_predictions else None,
                    ts_event=bar.ts_event,
                    ts_init=self.clock.timestamp_ns(),
                )
                self._publish_signal(signal)

        except Exception as e:
            self.log.error(f"Prediction failed: {e}")

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
    def _compute_features(self, bar: Bar) -> np.ndarray | None:
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
        np.ndarray | None
            The computed feature vector, or None if not ready.

        """
        ...

    @abstractmethod
    def _predict(self, features: np.ndarray) -> tuple[float, float]:
        """
        Generate prediction from feature vector.

        This method should perform model inference and return both the
        prediction value and confidence score.

        Parameters
        ----------
        features : np.ndarray
            The feature vector for prediction.

        Returns
        -------
        tuple[float, float]
            A tuple of (prediction, confidence) values.

        """
        ...


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

        with open(model_path, "rb") as f:
            self._model = pickle.load(f)  # noqa: S301

        self.log.info(f"Loaded model from {model_path}")

    def _predict(self, features: np.ndarray) -> tuple[float, float]:
        """
        Generate prediction using the loaded model.

        Parameters
        ----------
        features : np.ndarray
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
