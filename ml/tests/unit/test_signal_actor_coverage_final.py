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
Final coverage tests for MLSignalActor targeting specific missing lines.
"""

from unittest.mock import Mock
from unittest.mock import patch

import numpy as np

from ml.actors.signal import MLSignalActor
from ml.config.base import MLFeatureConfig
from ml.features.engineering import FeatureConfig


class TestMLSignalActorCoverageFinal:
    """
    Final tests to reach 90% coverage.
    """

    def test_init_paths_coverage(self):
        """
        Test initialization paths for lines 212-216, 242.
        """
        with patch("ml.actors.base.BaseMLInferenceActor.__init__"):
            with patch("ml.features.engineering.FeatureEngineer") as mock_fe:
                mock_fe.return_value.n_features = 10

                # Test with MLFeatureConfig base class (lines 212-216)
                base_config = MLFeatureConfig()
                config = Mock()
                config.feature_config = base_config
                config.min_signal_separation_bars = 2
                config.prediction_threshold = 0.7
                config.adaptive_window = 10
                config.ensemble_weights = {"custom": 0.5}  # Line 242
                config.signal_strategy = Mock(value="threshold")

                with patch.object(MLSignalActor, "log", create=True):
                    actor = MLSignalActor(config)

                    # Should create default FeatureConfig
                    assert isinstance(actor._feature_config, FeatureConfig)
                    assert actor._ensemble_weights == {"custom": 0.5}

    def test_initialize_features_paths(self):
        """
        Test _initialize_features for lines 298-315.
        """
        actor = Mock(spec=MLSignalActor)

        # Test with base MLFeatureConfig (lines 298-303)
        actor._feature_config = Mock()
        actor._feature_config.__class__ = MLFeatureConfig  # Not FeatureConfig
        actor._indicator_manager = None
        actor._feature_buffer = np.zeros(5)
        actor._feature_engineer = Mock()
        actor._feature_engineer.n_features = 10
        actor.log = Mock()

        MLSignalActor._initialize_features(actor)

        # Test with feature names but not enough features (line 308)
        actor._feature_config = Mock()
        feature_names_mock = Mock()
        feature_names_mock.__len__ = Mock(return_value=7)
        actor._feature_config.feature_names = feature_names_mock
        actor._feature_buffer = np.zeros(5)
        actor._indicator_manager = None

        MLSignalActor._initialize_features(actor)

        # Should resize buffer
        assert actor._feature_buffer.size >= 7

    def test_compute_features_paths(self):
        """
        Test _compute_features for lines 338-369.
        """
        actor = Mock(spec=MLSignalActor)
        actor._indicator_manager = Mock()
        actor._indicator_manager.all_initialized.return_value = True
        actor._indicator_manager.update_from_bar = Mock()
        actor._feature_engineer = Mock()
        actor._feature_engineer.calculate_features_online.return_value = np.array([0.1, 0.2])
        actor._config = Mock()
        actor._config.max_feature_latency_ms = 1000
        actor.log = Mock()

        # Create a proper bar mock
        bar = Mock()
        bar.close = Mock()
        bar.close.__float__ = Mock(return_value=1.1000)
        bar.volume = Mock()
        bar.volume.__float__ = Mock(return_value=1000.0)
        bar.high = Mock()
        bar.high.__float__ = Mock(return_value=1.1001)
        bar.low = Mock()
        bar.low.__float__ = Mock(return_value=1.0999)

        # Test successful feature computation
        features = MLSignalActor._compute_features(actor, bar)
        assert features is not None
        assert len(features) == 2

        # Verify current_bar dictionary was created correctly
        call_args = actor._feature_engineer.calculate_features_online.call_args
        current_bar = call_args[1]["current_bar"]
        assert current_bar["close"] == 1.1000
        assert current_bar["volume"] == 1000.0
        assert current_bar["high"] == 1.1001
        assert current_bar["low"] == 1.0999

    def test_predict_paths(self):
        """
        Test _predict for lines 390, 397-409.
        """
        actor = Mock(spec=MLSignalActor)
        actor.log = Mock()
        features = np.array([0.1, 0.2, 0.3])

        # Test ONNX path (line 397)
        actor._model = Mock()
        actor._model.run = Mock(return_value=[[np.array([[0.8]])]])
        actor._model_metadata = {"input_names": ["input"], "output_names": ["output"]}

        pred, conf = MLSignalActor._predict(actor, features)
        # Should call _predict_onnx

        # Test sklearn with predict_proba (line 399)
        del actor._model.run
        actor._model.predict_proba = Mock(return_value=np.array([[0.3, 0.7]]))
        pred, conf = MLSignalActor._predict(actor, features)
        # Should call _predict_sklearn_proba

        # Test sklearn basic (line 402)
        del actor._model.predict_proba
        actor._model.predict = Mock(return_value=np.array([0.85]))
        pred, conf = MLSignalActor._predict(actor, features)
        # Should call _predict_sklearn

        # Test unsupported model (line 404)
        actor._model = object()  # No predict methods
        pred, conf = MLSignalActor._predict(actor, features)
        assert pred == 0.0 and conf == 0.0
        actor.log.error.assert_called_with("Unsupported model type: <class 'object'>")

    def test_generate_prediction_protected_paths(self):
        """
        Test _generate_prediction_protected for lines 481, 521-530.
        """
        actor = Mock(spec=MLSignalActor)
        actor._config = Mock()
        actor._signal_config = Mock()
        actor._signal_config.signal_strategy = Mock(value="adaptive")
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

        # Test successful path with no signal (line 521)
        actor._predict = Mock(return_value=(0.8, 0.9))
        actor._update_prediction_history = Mock()
        actor._detect_market_regime = Mock()
        actor._generate_signal_by_strategy = Mock(return_value=None)
        actor._track_performance_metrics = Mock()

        MLSignalActor._generate_prediction_protected(actor, bar, features)

        # Should still record success even without signal
        actor._circuit_breaker.record_success.assert_called()
        actor._health_monitor.update_prediction_success.assert_called()

        # Test exception path (lines 522-530)
        actor._predict.side_effect = Exception("Test error")

        MLSignalActor._generate_prediction_protected(actor, bar, features)

        actor.log.error.assert_called_with("Signal generation failed: Test error")
        actor._circuit_breaker.record_failure.assert_called()
        actor._health_monitor.update_prediction_failure.assert_called()

    def test_signal_generation_paths(self):
        """
        Test signal generation for lines 712, 724-744, 756-775, 788-821, 834-851.
        """
        actor = Mock(spec=MLSignalActor)
        actor._config = Mock()
        actor._config.prediction_threshold = 0.7
        actor._config.log_predictions = True
        actor._signal_config = Mock()
        actor._signal_config.adaptive_window = 10
        actor._signal_config.extremes_top_pct = 0.1
        actor._signal_config.momentum_lookback = 3
        actor._bars_processed = 20
        actor._last_signal_bar = 0
        actor._adaptive_threshold = 0.5
        actor._market_regime = "trending"
        actor.clock = Mock()
        actor.clock.timestamp_ns.return_value = 2000

        bar = Mock()
        bar.bar_type = Mock()
        bar.bar_type.instrument_id = "EURUSD"
        bar.ts_event = 1000
        features = np.array([0.1, 0.2])

        # Test threshold signal with features (line 712)
        signal = MLSignalActor._generate_threshold_signal(actor, bar, 0.8, 0.9, features)
        assert signal.features is not None  # Should include features when log_predictions=True

        # Test extremes signal paths (lines 724-744)
        # Insufficient history
        actor._prediction_history = [0.5, 0.6]  # Less than adaptive_window
        signal = MLSignalActor._generate_extremes_signal(actor, bar, 0.9, 0.8, features)
        assert signal is None

        # With sufficient history
        actor._prediction_history = list(range(20))  # More than adaptive_window

        # Test top extreme
        signal = MLSignalActor._generate_extremes_signal(actor, bar, 19.0, 0.8, features)
        assert signal is not None

        # Test bottom extreme
        signal = MLSignalActor._generate_extremes_signal(actor, bar, 0.0, 0.8, features)
        assert signal is not None

        # Test middle value (not extreme)
        signal = MLSignalActor._generate_extremes_signal(actor, bar, 10.0, 0.8, features)
        assert signal is None

        # Test with low confidence
        signal = MLSignalActor._generate_extremes_signal(actor, bar, 19.0, 0.5, features)
        assert signal is None  # Below threshold

        # Test momentum signal paths (lines 756-775)
        # Insufficient history
        actor._prediction_history = [0.5]
        signal = MLSignalActor._generate_momentum_signal(actor, bar, 0.8, 0.9, features)
        assert signal is None

        # With momentum
        actor._prediction_history = [0.2, 0.3, 0.4, 0.5]
        signal = MLSignalActor._generate_momentum_signal(actor, bar, 0.6, 0.8, features)
        assert signal is not None
        assert signal.prediction != 0.6  # Adjusted by momentum

        # Test ensemble signal paths (lines 788-821)
        actor._ensemble_weights = {
            "threshold": 0.4,
            "extremes": 0.3,
            "momentum": 0.3,
        }

        # Create mock signals
        mock_signal1 = Mock()
        mock_signal1.confidence = 0.9
        mock_signal2 = Mock()
        mock_signal2.confidence = 0.8
        mock_signal3 = Mock()
        mock_signal3.confidence = 0.7

        # Test with all signals present
        actor._generate_threshold_signal = Mock(return_value=mock_signal1)
        actor._generate_extremes_signal = Mock(return_value=mock_signal2)
        actor._generate_momentum_signal = Mock(return_value=mock_signal3)

        signal = MLSignalActor._generate_ensemble_signal(actor, bar, 0.8, 0.85, features)
        assert signal is not None
        # Ensemble confidence = (0.4*0.9 + 0.3*0.8 + 0.3*0.7) / 1.0 = 0.81
        assert signal.confidence >= 0.8

        # Test with partial signals
        actor._generate_threshold_signal = Mock(return_value=None)
        actor._generate_extremes_signal = Mock(return_value=mock_signal2)
        actor._generate_momentum_signal = Mock(return_value=None)

        signal = MLSignalActor._generate_ensemble_signal(actor, bar, 0.8, 0.85, features)
        assert signal is not None

        # Test adaptive signal paths (lines 834-851)
        actor._adaptive_threshold = 0.5
        actor._market_regime = "volatile"

        # Signal strength = 0.9 / 0.5 = 1.8
        signal = MLSignalActor._generate_adaptive_signal(actor, bar, 0.8, 0.9, features)
        assert signal is not None
        assert signal.adaptive_threshold == 0.5
        assert signal.signal_strength == 1.8
        assert signal.market_regime == "volatile"

        # Test with zero threshold
        actor._adaptive_threshold = 0.0
        signal = MLSignalActor._generate_adaptive_signal(actor, bar, 0.8, 0.9, features)
        assert signal is None  # signal_strength would be inf

    def test_performance_metrics_paths(self):
        """
        Test _track_performance_metrics for lines 890-904.
        """
        actor = Mock(spec=MLSignalActor)
        actor._prediction_count = 10
        actor._total_inference_time = 20.0
        actor._config = Mock()
        actor._config.log_predictions = True
        actor._signal_config = Mock()
        actor._signal_config.signal_strategy = Mock(value="ensemble")
        actor.id = Mock()
        actor.id.value = "TEST"
        actor.log = Mock()

        # Create metrics if they don't exist
        actor._prediction_distribution_metric = Mock()
        actor._confidence_distribution_metric = Mock()

        # Test with logging enabled
        MLSignalActor._track_performance_metrics(actor, 0.75, 0.85, 3.5)

        assert actor._prediction_count == 11
        assert actor._total_inference_time == 23.5

        # Verify metrics were recorded
        actor._prediction_distribution_metric.observe.assert_called_with(
            0.75,
            {"actor_id": "TEST"},
        )
        actor._confidence_distribution_metric.observe.assert_called_with(
            0.85,
            {"actor_id": "TEST"},
        )

        # Verify debug logging
        actor.log.debug.assert_called()
        log_msg = actor.log.debug.call_args[0][0]
        assert "Prediction: 0.7500" in log_msg
        assert "Confidence: 0.8500" in log_msg
        assert "Signal time: 3.500ms" in log_msg
        assert "Strategy: ensemble" in log_msg
