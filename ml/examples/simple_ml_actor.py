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
Example of a simple ML actor using the configuration adapter pattern.

This example demonstrates how to properly create and use ML actors with Nautilus
Trader's Cython components.

"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ml.actors.base import BaseMLInferenceActor
from ml.config.base import MLActorConfig
from nautilus_trader.indicators.average.ema import ExponentialMovingAverage
from nautilus_trader.indicators.average.sma import SimpleMovingAverage
from nautilus_trader.indicators.rsi import RelativeStrengthIndex
from nautilus_trader.model.data import Bar


class SimpleMLActor(BaseMLInferenceActor):
    """
    Simple ML actor for demonstration purposes.

    This actor computes basic technical indicators as features and uses a simple model
    for predictions.

    """

    def __init__(self, config: MLActorConfig) -> None:
        """
        Initialize the simple ML actor.

        Parameters
        ----------
        config : MLActorConfig
            The ML actor configuration.

        """
        super().__init__(config)

        # Technical indicators (will be initialized in on_start)
        self._sma_fast: SimpleMovingAverage | None = None
        self._sma_slow: SimpleMovingAverage | None = None
        self._rsi: RelativeStrengthIndex | None = None
        self._ema: ExponentialMovingAverage | None = None

        # Feature buffer
        self._feature_buffer = np.zeros(8, dtype=np.float32)

    def _initialize_features(self) -> None:
        """
        Initialize technical indicators.
        """
        self._sma_fast = SimpleMovingAverage(10)
        self._sma_slow = SimpleMovingAverage(20)
        self._rsi = RelativeStrengthIndex(14)
        self._ema = ExponentialMovingAverage(12)

        self.log.info("Initialized technical indicators for SimpleMLActor")

    def _load_model(self) -> None:
        """
        Load the ML model.
        """
        model_path = Path(self._config.model_path)
        if not model_path.exists():
            # For demo purposes, create a dummy model
            self.log.warning(f"Model file not found at {model_path}, using dummy model")
            self._model = DummyModel()
        else:
            with open(model_path, "rb") as f:
                self._model = pickle.load(f)  # noqa: S301
            self.log.info(f"Loaded model from {model_path}")

    def _compute_features(self, bar: Bar) -> np.ndarray | None:
        """
        Compute feature vector from bar data.

        Parameters
        ----------
        bar : Bar
            The current bar data.

        Returns
        -------
        np.ndarray | None
            The feature vector or None if indicators not ready.

        """
        # Update indicators
        if (
            self._sma_fast is None
            or self._sma_slow is None
            or self._rsi is None
            or self._ema is None
        ):
            return None

        self._sma_fast.update_raw(float(bar.close))
        self._sma_slow.update_raw(float(bar.close))
        self._rsi.update_raw(float(bar.close))
        self._ema.update_raw(float(bar.close))

        # Check if all indicators are initialized
        if not all(
            [
                self._sma_fast.initialized,
                self._sma_slow.initialized,
                self._rsi.initialized,
                self._ema.initialized,
            ],
        ):
            return None

        # Compute features
        close_price = float(bar.close)

        # Price-based features
        sma_fast_val = float(self._sma_fast.value) if self._sma_fast.value else close_price
        sma_slow_val = float(self._sma_slow.value) if self._sma_slow.value else close_price
        ema_val = float(self._ema.value) if self._ema.value else close_price
        rsi_val = float(self._rsi.value) if self._rsi.value else 50.0

        self._feature_buffer[0] = close_price / sma_fast_val
        self._feature_buffer[1] = close_price / sma_slow_val
        self._feature_buffer[2] = sma_fast_val / sma_slow_val

        # Technical indicators
        self._feature_buffer[3] = rsi_val / 100.0
        self._feature_buffer[4] = close_price / ema_val

        # Price change features
        self._feature_buffer[5] = float(bar.high - bar.low) / close_price
        self._feature_buffer[6] = float(bar.close - bar.open) / close_price

        # Volume feature
        self._feature_buffer[7] = min(float(bar.volume) / 1000000.0, 5.0)  # Normalized

        return self._feature_buffer.copy()

    def _predict(self, features: np.ndarray) -> tuple[float, float]:
        """
        Generate prediction from features.

        Parameters
        ----------
        features : np.ndarray
            The feature vector.

        Returns
        -------
        tuple[float, float]
            Prediction and confidence values.

        """
        if isinstance(self._model, DummyModel):
            return self._model.predict(features)

        # Use real model
        features_2d = features.reshape(1, -1)

        if hasattr(self._model, "predict_proba"):
            probabilities = self._model.predict_proba(features_2d)[0]
            prediction = np.argmax(probabilities)
            confidence = np.max(probabilities)
        else:
            prediction = self._model.predict(features_2d)[0]
            confidence = 0.9  # Default confidence for regression

        return float(prediction), float(confidence)


class DummyModel:
    """
    Dummy model for demonstration when no real model is available.
    """

    def predict(self, features: np.ndarray) -> tuple[float, float]:
        """
        Generate dummy predictions based on simple rules.

        Parameters
        ----------
        features : np.ndarray
            The feature vector.

        Returns
        -------
        tuple[float, float]
            Prediction and confidence.

        """
        # Simple momentum-based prediction
        # features[0] = price/sma_fast, features[3] = rsi/100

        momentum = features[0] - 1.0  # Price relative to SMA
        rsi_normalized = features[3]

        # Generate signal
        if momentum > 0.01 and rsi_normalized < 0.7:
            prediction = 1.0  # Buy signal
            confidence = min(momentum * 10, 0.95)
        elif momentum < -0.01 and rsi_normalized > 0.3:
            prediction = -1.0  # Sell signal
            confidence = min(abs(momentum) * 10, 0.95)
        else:
            prediction = 0.0  # No signal
            confidence = 0.3

        return prediction, confidence
