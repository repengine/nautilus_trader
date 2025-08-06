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
Final push to 90% coverage for MLSignalActor.
"""

from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np

import ml.actors.signal
from ml.actors.base import MLSignal
from ml.actors.signal import AdaptiveSignal
from ml.actors.signal import MLSignalActor
from ml.actors.signal import SignalStrategy
from ml.config.base import MLFeatureConfig
from ml.features.engineering import FeatureConfig
from ml.features.engineering import IndicatorManager


class TestMLSignalActor90Coverage:
    """
    Tests to reach 90% coverage.
    """

    def test_init_lines_212_216_242(self):
        """
        Test initialization with MLFeatureConfig base class.
        """
        with patch("ml.actors.base.BaseMLInferenceActor.__init__"):
            with patch("ml.features.engineering.FeatureEngineer") as mock_fe:
                with patch.multiple(
                    "prometheus_client",
                    Histogram=Mock(return_value=Mock()),
                    Counter=Mock(return_value=Mock()),
                    Gauge=Mock(return_value=Mock()),
                ):
                    mock_fe.return_value.n_features = 10

                    # Test with MLFeatureConfig (lines 212-216)
                    base_config = MLFeatureConfig()
                    config = Mock()
                    config.feature_config = base_config
                    config.min_signal_separation_bars = 2
                    config.prediction_threshold = 0.7
                    config.adaptive_window = 10
                    config.ensemble_weights = {"custom": 0.5}  # Line 242
                    config.signal_strategy = Mock(value="ensemble")

                    with patch.object(MLSignalActor, "log", create=True):
                        actor = MLSignalActor(config)

                        # Should convert MLFeatureConfig to FeatureConfig
                        assert isinstance(actor._feature_config, FeatureConfig)
                        assert actor._ensemble_weights == {"custom": 0.5}

    def test_initialize_features_lines_298_315(self):
        """
        Test _initialize_features method.
        """
        # Test isinstance False path (lines 303-305)
        actor = Mock()
        actor._feature_config = MagicMock()
        actor._feature_config.__class__ = object  # Not FeatureConfig
        actor._indicator_manager = None
        actor._feature_buffer = np.zeros(5)
        actor._feature_engineer = Mock()
        actor._feature_engineer.n_features = 10
        actor.log = Mock()

        # Patch isinstance to ensure it returns False for FeatureConfig check
        original_isinstance = isinstance

        def mock_isinstance(obj, cls):
            if cls == FeatureConfig:
                return False
            return original_isinstance(obj, cls)

        with patch.object(ml.actors.signal, "isinstance", mock_isinstance):
            MLSignalActor._initialize_features(actor)
            assert isinstance(actor._indicator_manager, IndicatorManager)

        # Test feature names path with resize (lines 308-315)
        actor._feature_config = MagicMock()
        actor._feature_config.feature_names = [
            "f1",
            "f2",
            "f3",
            "f4",
            "f5",
            "f6",
            "f7",
            "f8",
            "f9",
            "f10",
        ]
        actor._feature_buffer = np.zeros(5)
        actor._indicator_manager = None

        # Patch hasattr to return True
        with patch.object(ml.actors.signal, "hasattr", return_value=True):
            with patch.object(ml.actors.signal, "len", return_value=10):
                MLSignalActor._initialize_features(actor)
                assert actor._feature_buffer.size == 10
                actor.log.info.assert_called_with(
                    "Feature buffer resized from 5 to 10 to match feature configuration",
                )

    def test_predict_lines_389_409(self):
        """
        Test _predict method paths.
        """
        actor = Mock()
        actor.log = Mock()
        features = np.array([0.1, 0.2, 0.3])

        # No model (lines 389-391)
        actor._model = None
        pred, conf = MLSignalActor._predict(actor, features)
        assert pred == 0.0 and conf == 0.0

        # ONNX path (line 397)
        actor._model = MagicMock()
        actor._model_metadata = {"input_names": ["input"]}
        actor._predict_onnx = Mock(return_value=(0.8, 0.8))

        with patch.object(ml.actors.signal, "hasattr", side_effect=lambda obj, attr: attr == "run"):
            pred, conf = MLSignalActor._predict(actor, features)
            actor._predict_onnx.assert_called_once()

        # Sklearn proba path (line 399)
        actor._model = MagicMock()
        actor._predict_sklearn_proba = Mock(return_value=(1.0, 0.7))

        with patch.object(
            ml.actors.signal,
            "hasattr",
            side_effect=lambda obj, attr: attr == "predict_proba",
        ):
            pred, conf = MLSignalActor._predict(actor, features)
            actor._predict_sklearn_proba.assert_called_once()

        # Sklearn basic path (line 402)
        actor._model = MagicMock()
        actor._predict_sklearn = Mock(return_value=(0.85, 0.85))

        with patch.object(
            ml.actors.signal,
            "hasattr",
            side_effect=lambda obj, attr: attr == "predict",
        ):
            pred, conf = MLSignalActor._predict(actor, features)
            actor._predict_sklearn.assert_called_once()

        # Unsupported model (lines 404-405)
        actor._model = object()
        with patch.object(ml.actors.signal, "hasattr", return_value=False):
            pred, conf = MLSignalActor._predict(actor, features)
            assert pred == 0.0 and conf == 0.0
            actor.log.error.assert_called_with("Unsupported model type: <class 'object'>")

        # Exception (lines 406-409)
        actor._model = MagicMock()
        actor._model.predict = Mock(side_effect=Exception("Test error"))

        # Create a mock that raises exception when called
        def predict_with_error(features):
            raise Exception("Test error")

        actor._predict_onnx = predict_with_error

        with patch.object(ml.actors.signal, "hasattr", return_value=True):
            pred, conf = MLSignalActor._predict(actor, features)
            assert pred == 0.0 and conf == 0.0
            actor.log.error.assert_called_with("Prediction failed: Test error")

    def test_update_adaptive_threshold_lines_549_567(self):
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

        # Verify calculation
        # mean_pred = np.mean([0.4, 0.5, 0.6, 0.7, 0.8]) = 0.6
        # mean_vol = np.mean([0.01, 0.02, 0.03, 0.02, 0.01]) = 0.018
        # new_threshold = 0.5 + 2.0 * 0.018 * (0.6 - 0.5) = 0.5 + 0.0036 = 0.5036
        # clipped to [0.1, 0.95]

        assert abs(actor._adaptive_threshold - 0.5036) < 0.0001
        actor._adaptive_threshold_metric.set.assert_called_once()

    def test_market_regime_lines_577_591(self):
        """
        Test market regime detection edge cases.
        """
        actor = Mock()
        actor._indicator_manager = Mock()
        actor._market_regime = "unknown"
        actor._market_regime_metric = Mock()
        actor.log = Mock()
        actor.id = Mock()
        actor.id.value = "TEST"

        bar = Mock()

        # Test insufficient data (lines 582-584)
        actor._indicator_manager.price_history = {"closes": [1.10, 1.11]}
        MLSignalActor._detect_market_regime(actor, bar)
        assert actor._market_regime == "unknown"

        # Test with empty price history (lines 579-581)
        actor._indicator_manager.price_history = {}
        MLSignalActor._detect_market_regime(actor, bar)
        assert actor._market_regime == "unknown"

        # Test regime detection logic (lines 585-591)
        # Create realistic price data
        prices = [1.10 + i * 0.001 for i in range(25)]  # Trending
        actor._indicator_manager.price_history = {"closes": prices}
        MLSignalActor._detect_market_regime(actor, bar)
        assert actor._market_regime == "trending"

        # Metric mapping (line 590)
        actor._market_regime_metric.set.assert_called_with(
            0,  # trending = 0
            {"actor_id": "TEST", "regime": "trending"},
        )

    def test_signal_generation_lines_689_690_702_712(self):
        """
        Test threshold signal generation.
        """
        actor = Mock()
        actor._config = Mock()
        actor._config.prediction_threshold = 0.7
        actor._config.log_predictions = True
        actor._bars_processed = 20
        actor._last_signal_bar = 0
        actor.clock = Mock()
        actor.clock.timestamp_ns = Mock(return_value=2000)

        bar = Mock()
        bar.bar_type = Mock()
        bar.bar_type.instrument_id = "EURUSD"
        bar.ts_event = 1000
        features = np.array([0.1, 0.2])

        # Test below threshold (lines 689-690)
        signal = MLSignalActor._generate_threshold_signal(actor, bar, 0.8, 0.6, features)
        assert signal is None

        # Test above threshold (lines 702-712)
        signal = MLSignalActor._generate_threshold_signal(actor, bar, 0.8, 0.9, features)
        assert isinstance(signal, MLSignal)
        assert signal.prediction == 0.8
        assert signal.confidence == 0.9
        assert signal.features is not None  # log_predictions=True
        assert actor._last_signal_bar == 20

    def test_extremes_signal_lines_724_744(self):
        """
        Test extremes signal generation.
        """
        actor = Mock()
        actor._config = Mock()
        actor._config.prediction_threshold = 0.7
        actor._config.log_predictions = False
        actor._signal_config = Mock()
        actor._signal_config.adaptive_window = 10
        actor._signal_config.extremes_top_pct = 0.1
        actor._prediction_history = []
        actor._bars_processed = 20
        actor._last_signal_bar = 0
        actor.clock = Mock()
        actor.clock.timestamp_ns = Mock(return_value=2000)

        bar = Mock()
        bar.bar_type = Mock()
        bar.bar_type.instrument_id = "EURUSD"
        bar.ts_event = 1000
        features = None

        # Insufficient history (lines 724-726)
        signal = MLSignalActor._generate_extremes_signal(actor, bar, 0.9, 0.8, features)
        assert signal is None

        # With history
        actor._prediction_history = list(range(100))

        # Calculate thresholds
        # window = list(range(90, 100))  # Last 10 values
        # top_threshold = np.percentile(window, 90) = 98.1
        # bottom_threshold = np.percentile(window, 10) = 90.9

        # Top extreme (lines 734-744)
        signal = MLSignalActor._generate_extremes_signal(actor, bar, 99.0, 0.8, features)
        assert signal is not None
        assert signal.prediction == 99.0

        # Bottom extreme
        signal = MLSignalActor._generate_extremes_signal(actor, bar, 90.0, 0.8, features)
        assert signal is not None
        assert signal.prediction == 90.0

        # Not extreme
        signal = MLSignalActor._generate_extremes_signal(actor, bar, 95.0, 0.8, features)
        assert signal is None

        # Low confidence (line 733)
        signal = MLSignalActor._generate_extremes_signal(actor, bar, 99.0, 0.6, features)
        assert signal is None

    def test_momentum_signal_lines_756_775(self):
        """
        Test momentum signal generation.
        """
        actor = Mock()
        actor._config = Mock()
        actor._config.prediction_threshold = 0.7
        actor._config.log_predictions = False
        actor._signal_config = Mock()
        actor._signal_config.momentum_lookback = 3
        actor._prediction_history = []
        actor._bars_processed = 20
        actor._last_signal_bar = 0
        actor.clock = Mock()
        actor.clock.timestamp_ns = Mock(return_value=2000)

        bar = Mock()
        bar.bar_type = Mock()
        bar.bar_type.instrument_id = "EURUSD"
        bar.ts_event = 1000
        features = None

        # Insufficient history (lines 756-758)
        signal = MLSignalActor._generate_momentum_signal(actor, bar, 0.8, 0.9, features)
        assert signal is None

        # With momentum
        actor._prediction_history = [0.2, 0.4, 0.6, 0.8]
        signal = MLSignalActor._generate_momentum_signal(actor, bar, 0.9, 0.8, features)
        assert signal is not None

        # Calculate expected momentum
        # lookback_values = [0.4, 0.6, 0.8]  # Last 3
        momentum = (0.8 - 0.4) / 3  # = 0.133...
        expected_pred = 0.9 * (1 + momentum)  # = 1.02
        expected_pred = np.clip(expected_pred, -1, 1)  # = 1.0

        assert signal.prediction == 1.0

        # No momentum (flat values)
        actor._prediction_history = [0.5, 0.5, 0.5, 0.5]
        signal = MLSignalActor._generate_momentum_signal(actor, bar, 0.5, 0.8, features)
        assert signal is None

    def test_ensemble_signal_lines_788_821(self):
        """
        Test ensemble signal generation.
        """
        actor = Mock()
        actor._config = Mock()
        actor._config.prediction_threshold = 0.7
        actor._config.log_predictions = False
        actor._ensemble_weights = {
            "threshold": 0.4,
            "extremes": 0.3,
            "momentum": 0.3,
        }
        actor._bars_processed = 20
        actor._last_signal_bar = 0
        actor.clock = Mock()
        actor.clock.timestamp_ns = Mock(return_value=2000)

        bar = Mock()
        bar.bar_type = Mock()
        bar.bar_type.instrument_id = "EURUSD"
        bar.ts_event = 1000
        features = None

        # Test with all signals
        signal1 = Mock(confidence=0.9)
        signal2 = Mock(confidence=0.8)
        signal3 = Mock(confidence=0.7)

        actor._generate_threshold_signal = Mock(return_value=signal1)
        actor._generate_extremes_signal = Mock(return_value=signal2)
        actor._generate_momentum_signal = Mock(return_value=signal3)

        signal = MLSignalActor._generate_ensemble_signal(actor, bar, 0.8, 0.85, features)
        assert signal is not None

        # Weighted average: 0.4*0.9 + 0.3*0.8 + 0.3*0.7 = 0.81
        assert signal.confidence == 0.81

        # Test with partial signals (lines 802-807)
        actor._generate_threshold_signal = Mock(return_value=signal1)
        actor._generate_extremes_signal = Mock(return_value=None)
        actor._generate_momentum_signal = Mock(return_value=signal3)

        signal = MLSignalActor._generate_ensemble_signal(actor, bar, 0.8, 0.85, features)
        assert signal is not None

        # Only threshold and momentum: (0.4*0.9 + 0.3*0.7) / 0.7 = 0.814...
        assert abs(signal.confidence - 0.8142857) < 0.0001

        # Test no signals (lines 811-813)
        actor._generate_threshold_signal = Mock(return_value=None)
        actor._generate_extremes_signal = Mock(return_value=None)
        actor._generate_momentum_signal = Mock(return_value=None)

        signal = MLSignalActor._generate_ensemble_signal(actor, bar, 0.8, 0.85, features)
        assert signal is None

        # Test below threshold (lines 815-816)
        low_conf_signal = Mock(confidence=0.5)
        actor._generate_threshold_signal = Mock(return_value=low_conf_signal)

        signal = MLSignalActor._generate_ensemble_signal(actor, bar, 0.8, 0.85, features)
        assert signal is None

    def test_adaptive_signal_lines_834_851(self):
        """
        Test adaptive signal generation.
        """
        actor = Mock()
        actor._config = Mock()
        actor._config.prediction_threshold = 0.7
        actor._config.log_predictions = False
        actor._adaptive_threshold = 0.5
        actor._market_regime = "volatile"
        actor._bars_processed = 20
        actor._last_signal_bar = 0
        actor.clock = Mock()
        actor.clock.timestamp_ns = Mock(return_value=2000)

        bar = Mock()
        bar.bar_type = Mock()
        bar.bar_type.instrument_id = "EURUSD"
        bar.ts_event = 1000
        features = None

        # Test below adaptive threshold (lines 834-835)
        signal = MLSignalActor._generate_adaptive_signal(actor, bar, 0.4, 0.9, features)
        assert signal is None

        # Test zero threshold (lines 837-839)
        actor._adaptive_threshold = 0.0
        signal = MLSignalActor._generate_adaptive_signal(actor, bar, 0.8, 0.9, features)
        assert signal is None

        # Test valid signal (lines 840-851)
        actor._adaptive_threshold = 0.5
        signal = MLSignalActor._generate_adaptive_signal(actor, bar, 0.8, 0.9, features)
        assert isinstance(signal, AdaptiveSignal)
        assert signal.prediction == 0.8
        assert signal.confidence == 0.9
        assert signal.adaptive_threshold == 0.5
        assert signal.signal_strength == 1.8
        assert signal.market_regime == "volatile"

    def test_performance_metrics_lines_878_891(self):
        """
        Test performance metrics tracking.
        """
        actor = Mock()
        actor._prediction_count = 5
        actor._total_inference_time = 15.0
        actor._config = Mock()
        actor._config.log_predictions = True
        actor._signal_config = Mock()
        actor._signal_config.signal_strategy = SignalStrategy.ADAPTIVE
        actor.id = Mock()
        actor.id.value = "TEST"
        actor.log = Mock()

        # Mock metrics
        actor._prediction_distribution_metric = Mock()
        actor._confidence_distribution_metric = Mock()

        # Test line 878 (create metrics if not exist)
        delattr(actor, "_prediction_distribution_metric")

        with patch("prometheus_client.Histogram") as mock_histogram:
            mock_histogram.return_value = Mock()

            MLSignalActor._track_performance_metrics(actor, 0.75, 0.85, 3.5)

            # Should create metrics
            assert hasattr(actor, "_prediction_distribution_metric")
            assert hasattr(actor, "_confidence_distribution_metric")

        # Test averaging path (line 891)
        actor._prediction_count = 10
        actor._total_inference_time = 50.0

        MLSignalActor._track_performance_metrics(actor, 0.8, 0.9, 5.0)

        # Should log with average
        actor.log.debug.assert_called()
        log_msg = actor.log.debug.call_args[0][0]
        assert "Avg time: 5.00ms" in log_msg  # 55.0 / 11 = 5.0
