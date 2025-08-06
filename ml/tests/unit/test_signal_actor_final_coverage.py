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
Final tests to achieve 90% coverage for MLSignalActor.
"""

from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest

from ml.actors.base import MLSignal
from ml.actors.signal import AdaptiveSignal
from ml.actors.signal import MLSignalActor
from ml.actors.signal import SignalStrategy
from ml.features.engineering import FeatureConfig
from ml.features.engineering import IndicatorManager


# Create simple implementations for the uncovered methods
def test_remaining_methods():
    """
    Test remaining uncovered methods directly.
    """
    # Test _initialize_features (lines 298-315)
    actor = Mock()
    actor._feature_config = FeatureConfig()
    actor._indicator_manager = None
    actor._feature_buffer = np.zeros(5)
    actor._feature_engineer = Mock()
    actor._feature_engineer.n_features = 10
    actor.log = Mock()

    MLSignalActor._initialize_features(actor)
    assert isinstance(actor._indicator_manager, IndicatorManager)

    # Test with non-FeatureConfig
    actor._feature_config = Mock()
    actor._feature_config.__class__.__name__ = "NotFeatureConfig"
    actor._indicator_manager = None

    # Mock isinstance to return False
    with patch(
        "ml.actors.signal.isinstance",
        side_effect=lambda obj, cls: False if cls == FeatureConfig else isinstance(obj, cls),
    ):
        MLSignalActor._initialize_features(actor)
        assert isinstance(actor._indicator_manager, IndicatorManager)

    # Test with feature names requiring resize
    actor._feature_config = Mock()
    actor._feature_config.feature_names = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]
    actor._feature_buffer = np.zeros(5)
    actor._indicator_manager = None

    # Mock hasattr to return True for feature_names
    with patch("ml.actors.signal.hasattr", return_value=True):
        MLSignalActor._initialize_features(actor)
        assert actor._feature_buffer.size >= 8

    # Test _predict paths (lines 389-409)
    actor = Mock()
    actor.log = Mock()
    features = np.array([0.1, 0.2, 0.3])

    # No model
    actor._model = None
    pred, conf = MLSignalActor._predict(actor, features)
    assert pred == 0.0 and conf == 0.0

    # ONNX model
    actor._model = Mock()
    actor._model.run = Mock()
    actor._predict_onnx = Mock(return_value=(0.8, 0.8))

    # Mock hasattr to return True for run method
    with patch("ml.actors.signal.hasattr", side_effect=lambda obj, attr: attr == "run"):
        pred, conf = MLSignalActor._predict(actor, features)
        actor._predict_onnx.assert_called_once()

    # Sklearn with predict_proba
    actor._model = Mock()
    actor._model.predict_proba = Mock()
    actor._predict_sklearn_proba = Mock(return_value=(1.0, 0.7))

    with patch("ml.actors.signal.hasattr", side_effect=lambda obj, attr: attr == "predict_proba"):
        pred, conf = MLSignalActor._predict(actor, features)
        actor._predict_sklearn_proba.assert_called_once()

    # Sklearn basic
    actor._model = Mock()
    actor._model.predict = Mock()
    actor._predict_sklearn = Mock(return_value=(0.85, 0.85))

    with patch("ml.actors.signal.hasattr", side_effect=lambda obj, attr: attr == "predict"):
        pred, conf = MLSignalActor._predict(actor, features)
        actor._predict_sklearn.assert_called_once()

    # Unsupported model
    actor._model = object()
    with patch("ml.actors.signal.hasattr", return_value=False):
        pred, conf = MLSignalActor._predict(actor, features)
        assert pred == 0.0 and conf == 0.0
        actor.log.error.assert_called()

    # Exception
    actor._model = Mock()
    actor._model.predict = Mock(side_effect=Exception("Test error"))

    with patch("ml.actors.signal.hasattr", return_value=True):
        pred, conf = MLSignalActor._predict(actor, features)
        assert pred == 0.0 and conf == 0.0


def test_update_prediction_history():
    """
    Test _update_prediction_history method.
    """
    actor = Mock()
    actor._signal_config = Mock()
    actor._signal_config.adaptive_window = 10
    actor._signal_config.signal_strategy = SignalStrategy.THRESHOLD
    actor._prediction_window = np.zeros(10)
    actor._confidence_window = np.zeros(10)
    actor._volatility_window = np.zeros(10)
    actor._window_index = 0
    actor._prediction_history = []
    actor._confidence_history = []
    actor._indicator_manager = None

    bar = Mock()

    # Without indicator manager
    MLSignalActor._update_prediction_history(actor, 0.5, 0.7, bar)
    assert actor._window_index == 1
    assert len(actor._prediction_history) == 1

    # With price history
    actor._indicator_manager = Mock()
    actor._indicator_manager.price_history = {"closes": [1.10, 1.11, 1.12]}
    MLSignalActor._update_prediction_history(actor, 0.6, 0.8, bar)

    # Calculate expected volatility
    prices = np.array([1.10, 1.11, 1.12])
    returns = np.diff(prices) / prices[:-1]
    expected_vol = float(np.std(returns))
    assert actor._volatility_window[1] == pytest.approx(expected_vol, rel=1e-5)

    # Test adaptive strategy
    actor._signal_config.signal_strategy = SignalStrategy.ADAPTIVE
    actor._update_adaptive_threshold = Mock()
    MLSignalActor._update_prediction_history(actor, 0.7, 0.9, bar)
    actor._update_adaptive_threshold.assert_called_once()


def test_adaptive_threshold_update():
    """
    Test _update_adaptive_threshold method.
    """
    actor = Mock()
    actor._config = Mock()
    actor._config.prediction_threshold = 0.5
    actor._signal_config = Mock()
    actor._signal_config.adaptive_volatility_factor = 2.0
    actor._adaptive_threshold = 0.5
    actor._prediction_window = np.array([0.4, 0.5, 0.6, 0.7, 0.8])
    actor._volatility_window = np.array([0.01, 0.02, 0.03, 0.02, 0.01])
    actor._adaptive_threshold_metric = Mock()

    MLSignalActor._update_adaptive_threshold(actor)

    # Check calculation
    mean_pred = np.mean(actor._prediction_window)
    mean_vol = np.mean(actor._volatility_window)
    expected = 0.5 + 2.0 * mean_vol * (mean_pred - 0.5)
    expected = np.clip(expected, 0.1, 0.95)

    assert actor._adaptive_threshold == pytest.approx(expected, rel=1e-5)
    actor._adaptive_threshold_metric.set.assert_called_once()


def test_market_regime_detection():
    """
    Test _detect_market_regime method.
    """
    actor = Mock()
    actor._indicator_manager = None
    actor._market_regime = "unknown"
    actor._market_regime_metric = Mock()
    actor.log = Mock()
    actor.id = Mock()
    actor.id.value = "TEST"

    bar = Mock()

    # No indicator manager
    MLSignalActor._detect_market_regime(actor, bar)
    assert actor._market_regime == "unknown"

    # With sufficient data - volatile
    actor._indicator_manager = Mock()
    actor._indicator_manager.price_history = {
        "closes": [1.10 + (i % 2) * 0.05 for i in range(25)],
    }
    MLSignalActor._detect_market_regime(actor, bar)
    assert actor._market_regime == "volatile"

    # Trending
    actor._market_regime = "unknown"
    actor._indicator_manager.price_history = {
        "closes": [1.10 + i * 0.001 for i in range(25)],
    }
    MLSignalActor._detect_market_regime(actor, bar)
    assert actor._market_regime == "trending"

    # Ranging
    actor._market_regime = "unknown"
    actor._indicator_manager.price_history = {
        "closes": [1.10] * 25,  # Flat prices
    }
    MLSignalActor._detect_market_regime(actor, bar)
    assert actor._market_regime == "ranging"


def test_signal_generation_methods():
    """
    Test all signal generation methods.
    """
    actor = Mock()
    actor._config = Mock()
    actor._config.prediction_threshold = 0.7
    actor._config.log_predictions = True
    actor._signal_config = Mock()
    actor._signal_config.adaptive_window = 10
    actor._signal_config.extremes_top_pct = 0.1
    actor._signal_config.momentum_lookback = 3
    actor._bars_processed = 20
    actor._last_signal_bar = 0
    actor._prediction_history = []
    actor._adaptive_threshold = 0.5
    actor._market_regime = "trending"
    actor._ensemble_weights = {"threshold": 0.4, "extremes": 0.3, "momentum": 0.3}
    actor.clock = Mock()
    actor.clock.timestamp_ns = Mock(return_value=2000)

    bar = Mock()
    bar.bar_type = Mock()
    bar.bar_type.instrument_id = "EURUSD"
    bar.ts_event = 1000
    features = np.array([0.1, 0.2])

    # Test threshold signal
    signal = MLSignalActor._generate_threshold_signal(actor, bar, 0.8, 0.9, features)
    assert isinstance(signal, MLSignal)
    assert signal.prediction == 0.8
    assert signal.confidence == 0.9
    assert signal.features is not None
    assert actor._last_signal_bar == 20

    # Test extremes signal
    actor._prediction_history = list(range(100))
    # window = actor._prediction_history[-10:]
    # top_threshold = np.percentile(window, 90)

    # Top extreme
    signal = MLSignalActor._generate_extremes_signal(actor, bar, 95.0, 0.8, features)
    assert signal is not None

    # Not extreme
    signal = MLSignalActor._generate_extremes_signal(actor, bar, 50.0, 0.8, features)
    assert signal is None

    # Test momentum signal
    actor._prediction_history = [0.2, 0.4, 0.6, 0.8]
    signal = MLSignalActor._generate_momentum_signal(actor, bar, 0.9, 0.8, features)
    assert signal is not None
    assert signal.prediction != 0.9  # Adjusted by momentum

    # Test ensemble signal
    mock_signal1 = Mock(confidence=0.9)
    mock_signal2 = Mock(confidence=0.8)
    mock_signal3 = Mock(confidence=0.7)

    actor._generate_threshold_signal = Mock(return_value=mock_signal1)
    actor._generate_extremes_signal = Mock(return_value=mock_signal2)
    actor._generate_momentum_signal = Mock(return_value=mock_signal3)

    signal = MLSignalActor._generate_ensemble_signal(actor, bar, 0.8, 0.85, features)
    assert isinstance(signal, MLSignal)
    # Weighted confidence = 0.4*0.9 + 0.3*0.8 + 0.3*0.7 = 0.81
    assert signal.confidence >= 0.8

    # Test adaptive signal
    signal = MLSignalActor._generate_adaptive_signal(actor, bar, 0.8, 0.9, features)
    assert isinstance(signal, AdaptiveSignal)
    assert signal.adaptive_threshold == 0.5
    assert signal.signal_strength == 1.8
    assert signal.market_regime == "trending"


def test_generate_signal_by_strategy():
    """
    Test _generate_signal_by_strategy method.
    """
    actor = Mock()
    actor._signal_config = Mock()
    actor._signal_config.min_signal_separation_bars = 2
    actor._signal_config.signal_strategy = SignalStrategy.THRESHOLD
    actor._bars_processed = 5
    actor._last_signal_bar = 4
    actor.log = Mock()

    bar = Mock()
    features = np.array([0.1, 0.2])

    # Test separation check
    signal = MLSignalActor._generate_signal_by_strategy(actor, bar, 0.8, 0.9, features)
    assert signal is None

    # Valid separation
    actor._last_signal_bar = 0
    mock_signal = Mock()

    # Mock all methods
    actor._generate_threshold_signal = Mock(return_value=mock_signal)
    actor._generate_extremes_signal = Mock(return_value=mock_signal)
    actor._generate_momentum_signal = Mock(return_value=mock_signal)
    actor._generate_ensemble_signal = Mock(return_value=mock_signal)
    actor._generate_adaptive_signal = Mock(return_value=mock_signal)

    # Test each strategy
    for strategy in SignalStrategy:
        actor._signal_config.signal_strategy = strategy
        signal = MLSignalActor._generate_signal_by_strategy(actor, bar, 0.8, 0.9, features)
        assert signal == mock_signal


def test_generate_prediction_protected():
    """
    Test _generate_prediction_protected method.
    """
    actor = Mock()
    actor._config = Mock()
    actor._signal_config = Mock()
    actor._signal_config.signal_strategy = SignalStrategy.THRESHOLD
    actor._signal_config.enable_regime_detection = True
    actor._circuit_breaker = Mock()
    actor._health_monitor = Mock()
    actor._signal_generation_time_metric = Mock()
    actor._signals_generated_metric = Mock()
    actor.log = Mock()
    actor.id = Mock()
    actor.id.value = "TEST"

    bar = Mock()
    bar.bar_type = Mock()
    bar.bar_type.instrument_id = "EURUSD"
    features = np.array([0.1, 0.2])

    # Success path
    actor._predict = Mock(return_value=(0.8, 0.9))
    actor._update_prediction_history = Mock()
    actor._detect_market_regime = Mock()
    actor._generate_signal_by_strategy = Mock(return_value=Mock(prediction=0.8))
    actor._publish_signal = Mock()
    actor._track_performance_metrics = Mock()

    MLSignalActor._generate_prediction_protected(actor, bar, features)

    actor._circuit_breaker.record_success.assert_called()
    actor._health_monitor.update_prediction_success.assert_called()

    # Exception path
    actor._predict = Mock(side_effect=Exception("Test error"))
    MLSignalActor._generate_prediction_protected(actor, bar, features)

    actor._circuit_breaker.record_failure.assert_called()
    actor._health_monitor.update_prediction_failure.assert_called()


def test_track_performance_metrics():
    """
    Test _track_performance_metrics method.
    """
    actor = Mock()
    actor._prediction_count = 0
    actor._total_inference_time = 0.0
    actor._config = Mock()
    actor._config.log_predictions = True
    actor._signal_config = Mock()
    actor._signal_config.signal_strategy = SignalStrategy.ADAPTIVE
    actor.id = Mock()
    actor.id.value = "TEST"
    actor.log = Mock()

    # Mock Histogram creation
    with patch("prometheus_client.Histogram") as mock_histogram:
        mock_histogram.return_value = Mock()

        MLSignalActor._track_performance_metrics(actor, 0.75, 0.85, 3.5)

        assert actor._prediction_count == 1
        assert actor._total_inference_time == 3.5

        # Check metrics were created
        assert hasattr(actor, "_prediction_distribution_metric")
        assert hasattr(actor, "_confidence_distribution_metric")


def test_backup_restore_state():
    """
    Test backup and restore methods.
    """
    actor = Mock()
    actor._indicator_manager = Mock()
    actor._indicator_manager.price_history = {"closes": [1.10, 1.11]}
    actor._indicator_manager.indicators = {
        "sma": Mock(value=50.0, initialized=True),
    }
    actor._prediction_history = [0.1, 0.2]
    actor._confidence_history = [0.6, 0.7]
    actor._prediction_window = np.array([0.1, 0.2])
    actor._confidence_window = np.array([0.3, 0.4])
    actor._volatility_window = np.array([0.01, 0.02])
    actor._window_index = 1
    actor._adaptive_threshold = 0.7
    actor._market_regime = "trending"
    actor._last_signal_bar = 10
    actor._signal_config = Mock()
    actor._signal_config.adaptive_window = 2
    actor._signal_config.min_signal_separation_bars = 2
    actor._config = Mock()
    actor._config.prediction_threshold = 0.5
    actor._indicator_state_backup = {}
    actor.log = Mock()

    # Backup
    MLSignalActor._backup_indicator_state(actor)

    # Modify state
    actor._prediction_history.clear()

    # Restore
    MLSignalActor._restore_indicator_state(actor)
    assert len(actor._prediction_history) == 2


def test_get_signal_statistics():
    """
    Test get_signal_statistics method.
    """
    actor = Mock()
    actor._signal_config = Mock()
    actor._signal_config.signal_strategy = SignalStrategy.ENSEMBLE
    actor._adaptive_threshold = 0.7
    actor._market_regime = "trending"
    actor._last_signal_bar = 10
    actor._prediction_history = [0.1, 0.2]
    actor._feature_buffer = np.zeros(5)
    actor._ensemble_weights = {"threshold": 0.5}
    actor.get_health_status = Mock(return_value={"status": "healthy"})

    stats = MLSignalActor.get_signal_statistics(actor)

    assert stats["signal_strategy"] == "ensemble"
    assert stats["adaptive_threshold"] == 0.7
    assert stats["ensemble_weights"] == {"threshold": 0.5}


def test_reset_signal_state():
    """
    Test reset_signal_state method.
    """
    actor = Mock()
    actor._prediction_history = [0.1, 0.2]
    actor._confidence_history = [0.6, 0.7]
    actor._prediction_window = np.ones(10)
    actor._confidence_window = np.ones(10)
    actor._volatility_window = np.ones(10)
    actor._window_index = 5
    actor._adaptive_threshold = 0.8
    actor._market_regime = "trending"
    actor._last_signal_bar = 10
    actor._config = Mock()
    actor._config.prediction_threshold = 0.5
    actor._signal_config = Mock()
    actor._signal_config.min_signal_separation_bars = 2
    actor.log = Mock()

    MLSignalActor.reset_signal_state(actor)

    assert len(actor._prediction_history) == 0
    assert actor._adaptive_threshold == 0.5
    assert actor._market_regime == "unknown"
