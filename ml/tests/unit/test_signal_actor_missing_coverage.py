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
Missing coverage tests for MLSignalActor.
"""

from unittest.mock import Mock

import numpy as np

from ml.actors.signal import MLSignalActor


def test_predict_sklearn_proba_coverage():
    """
    Test sklearn predict_proba path.
    """
    actor = Mock()
    features = np.array([0.1, 0.2, 0.3])

    # Test sklearn with predict_proba
    actor._model = Mock()
    actor._model.predict_proba = Mock(return_value=np.array([[0.3, 0.7]]))

    pred, conf = MLSignalActor._predict_sklearn_proba(actor, features)
    assert pred == 1.0
    assert conf == 0.7


def test_predict_sklearn_basic_coverage():
    """
    Test sklearn basic predict path.
    """
    actor = Mock()
    features = np.array([0.1, 0.2, 0.3])

    # Test sklearn basic predict
    actor._model = Mock()
    actor._model.predict = Mock(return_value=np.array([0.85]))

    pred, conf = MLSignalActor._predict_sklearn(actor, features)
    assert pred == 0.85
    assert conf == 0.85

    # Test zero prediction
    actor._model.predict = Mock(return_value=np.array([0.0]))
    pred, conf = MLSignalActor._predict_sklearn(actor, features)
    assert pred == 0.0
    assert conf == 0.5  # Special case


def test_signal_generation_missing_coverage():
    """
    Test missing signal generation lines.
    """
    actor = Mock()
    actor._config = Mock()
    actor._config.prediction_threshold = 0.7
    actor._config.log_predictions = True  # Test with logging enabled
    actor._signal_config = Mock()
    actor._signal_config.adaptive_window = 10
    actor._signal_config.extremes_top_pct = 0.1
    actor._signal_config.momentum_lookback = 3
    actor._bars_processed = 20
    actor._last_signal_bar = 0
    actor.clock = Mock()
    actor.clock.timestamp_ns = Mock(return_value=2000)

    bar = Mock()
    bar.bar_type = Mock()
    bar.bar_type.instrument_id = "EURUSD"
    bar.ts_event = 1000
    features = np.array([0.1, 0.2])

    # Test threshold signal with log_predictions=True (line 712)
    signal = MLSignalActor._generate_threshold_signal(actor, bar, 0.8, 0.9, features)
    assert signal is not None
    assert signal.features is not None  # Features included when log_predictions=True

    # Test extremes signal edge cases
    actor._prediction_history = list(range(100))

    # Get exact percentile values
    window = actor._prediction_history[-10:]
    predictions = np.array(window)
    top_threshold = np.percentile(predictions, 90)
    bottom_threshold = np.percentile(predictions, 10)

    # Test exactly at top threshold
    signal = MLSignalActor._generate_extremes_signal(actor, bar, top_threshold, 0.8, features)
    assert signal is not None

    # Test exactly at bottom threshold
    signal = MLSignalActor._generate_extremes_signal(actor, bar, bottom_threshold, 0.8, features)
    assert signal is not None

    # Test momentum signal with exact zero momentum
    actor._prediction_history = [0.5, 0.5, 0.5, 0.5]
    signal = MLSignalActor._generate_momentum_signal(actor, bar, 0.5, 0.8, features)
    assert signal is None  # No signal when momentum is zero


def test_ensemble_signal_partial_coverage():
    """
    Test ensemble signal with partial signals.
    """
    actor = Mock()
    actor._config = Mock()
    actor._config.prediction_threshold = 0.7
    actor._config.log_predictions = False
    actor._ensemble_weights = {"threshold": 0.4, "extremes": 0.3, "momentum": 0.3}
    actor._bars_processed = 20
    actor._last_signal_bar = 0
    actor.clock = Mock()
    actor.clock.timestamp_ns = Mock(return_value=2000)

    bar = Mock()
    bar.bar_type = Mock()
    bar.bar_type.instrument_id = "EURUSD"
    bar.ts_event = 1000
    features = None

    # Test with only some signals available
    signal1 = Mock(confidence=0.9)
    signal2 = None
    signal3 = Mock(confidence=0.7)

    actor._generate_threshold_signal = Mock(return_value=signal1)
    actor._generate_extremes_signal = Mock(return_value=signal2)
    actor._generate_momentum_signal = Mock(return_value=signal3)

    signal = MLSignalActor._generate_ensemble_signal(actor, bar, 0.8, 0.85, features)
    assert signal is not None
    # Weighted confidence = (0.4 * 0.9 + 0.3 * 0.7) / (0.4 + 0.3) = 0.814...
    assert abs(signal.confidence - 0.8142857) < 0.0001


def test_adaptive_signal_edge_cases():
    """
    Test adaptive signal generation edge cases.
    """
    actor = Mock()
    actor._config = Mock()
    actor._config.prediction_threshold = 0.7
    actor._config.log_predictions = False
    actor._adaptive_threshold = 0.6
    actor._market_regime = "ranging"
    actor._bars_processed = 20
    actor._last_signal_bar = 0
    actor.clock = Mock()
    actor.clock.timestamp_ns = Mock(return_value=2000)

    bar = Mock()
    bar.bar_type = Mock()
    bar.bar_type.instrument_id = "EURUSD"
    bar.ts_event = 1000
    features = None

    # Test exactly at threshold
    signal = MLSignalActor._generate_adaptive_signal(actor, bar, 0.6, 0.8, features)
    assert signal is not None
    assert signal.signal_strength == 1.0  # Exactly at threshold
    assert signal.market_regime == "ranging"
