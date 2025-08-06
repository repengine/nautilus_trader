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
Unit tests targeting 90% coverage for MLSignalActor.
"""

from unittest.mock import Mock
from unittest.mock import patch

import numpy as np

from ml.actors.base import MLSignal
from ml.actors.signal import AdaptiveSignal
from ml.actors.signal import MLSignalActor
from ml.actors.signal import SignalStrategy
from ml.config.base import MLFeatureConfig
from ml.features.engineering import FeatureConfig


class TestMLSignalActorCoverage90:
    """
    Test MLSignalActor to achieve 90% coverage.
    """

    def test_init_with_mlfeature_config(self):
        """
        Test initialization with MLFeatureConfig (lines 212-216, 242).
        """
        with patch("ml.actors.base.BaseMLInferenceActor.__init__"):
            with patch("ml.features.engineering.FeatureEngineer") as mock_fe:
                mock_fe.return_value.n_features = 10

                # Test with MLFeatureConfig
                ml_feature_config = MLFeatureConfig()
                config = Mock()
                config.feature_config = ml_feature_config
                config.min_signal_separation_bars = 2
                config.prediction_threshold = 0.7
                config.adaptive_window = 10
                config.ensemble_weights = {"custom": 0.5}
                config.signal_strategy = Mock(value="ensemble")

                with patch.object(MLSignalActor, "log", create=True):
                    actor = MLSignalActor(config)

                    # Verify MLFeatureConfig was converted to FeatureConfig
                    assert isinstance(actor._feature_config, FeatureConfig)
                    assert actor._ensemble_weights == {"custom": 0.5}

    def test_initialize_features_all_paths(self):
        """
        Test _initialize_features method (lines 298-315).
        """
        actor = Mock()
        actor.log = Mock()

        # Test with non-FeatureConfig (lines 303-305)
        actor._feature_config = Mock(spec=object)
        actor._indicator_manager = None
        actor._feature_buffer = np.zeros(5)
        actor._feature_engineer = Mock()
        actor._feature_engineer.n_features = 10

        MLSignalActor._initialize_features(actor)
        assert actor._indicator_manager is not None

        # Test with feature names requiring resize (lines 308-315)
        actor._feature_config = Mock()
        actor._feature_config.feature_names = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]
        actor._feature_buffer = np.zeros(5)
        actor._indicator_manager = None

        # Mock hasattr to return True
        with patch("builtins.hasattr", return_value=True):
            MLSignalActor._initialize_features(actor)
            assert actor._feature_buffer.size == 8
            actor.log.info.assert_called_with(
                "Feature buffer resized from 5 to 8 to match feature configuration",
            )

    def test_compute_features_all_paths(self):
        """
        Test _compute_features method (lines 338-369).
        """
        actor = Mock()
        actor.log = Mock()
        actor._config = Mock()
        actor._config.max_feature_latency_ms = 5

        bar = Mock()
        bar.close = Mock()
        bar.close.__float__ = Mock(return_value=1.1000)
        bar.volume = Mock()
        bar.volume.__float__ = Mock(return_value=1000.0)
        bar.high = Mock()
        bar.high.__float__ = Mock(return_value=1.1001)
        bar.low = Mock()
        bar.low.__float__ = Mock(return_value=1.0999)

        # No indicator manager (line 338)
        actor._indicator_manager = None
        result = MLSignalActor._compute_features(actor, bar)
        assert result is None

        # Not initialized (lines 340-342)
        actor._indicator_manager = Mock()
        actor._indicator_manager.all_initialized = Mock(return_value=False)
        result = MLSignalActor._compute_features(actor, bar)
        assert result is None

        # Success path (lines 347-369)
        actor._indicator_manager.all_initialized = Mock(return_value=True)
        actor._indicator_manager.update_from_bar = Mock()
        actor._feature_engineer = Mock()
        actor._feature_engineer.calculate_features_online = Mock(return_value=np.array([0.1, 0.2]))

        result = MLSignalActor._compute_features(actor, bar)
        assert result is not None
        assert len(result) == 2

    def test_predict_all_paths(self):
        """
        Test _predict method (lines 389-409).
        """
        actor = Mock()
        actor.log = Mock()
        features = np.array([0.1, 0.2, 0.3])

        # No model (lines 389-391)
        actor._model = None
        pred, conf = MLSignalActor._predict(actor, features)
        assert pred == 0.0 and conf == 0.0
        actor.log.warning.assert_called_with("No model loaded, returning zero prediction")

        # ONNX model (line 397)
        actor._model = Mock()
        actor._model.run = Mock(return_value=[np.array([[0.8]])])
        actor._model_metadata = {"input_names": ["input"]}
        pred, conf = MLSignalActor._predict(actor, features)
        assert pred == 0.8 and conf == 0.8

        # Sklearn with predict_proba (line 399)
        actor._model = Mock()
        del actor._model.run
        actor._model.predict_proba = Mock(return_value=np.array([[0.3, 0.7]]))
        pred, conf = MLSignalActor._predict(actor, features)
        assert pred == 1.0 and conf == 0.7

        # Sklearn basic (line 402)
        del actor._model.predict_proba
        actor._model.predict = Mock(return_value=np.array([0.85]))
        pred, conf = MLSignalActor._predict(actor, features)
        assert pred == 0.85 and conf == 0.85

        # Unsupported model (lines 404-405)
        actor._model = object()
        pred, conf = MLSignalActor._predict(actor, features)
        assert pred == 0.0 and conf == 0.0
        actor.log.error.assert_called()

        # Exception (lines 406-409)
        actor._model = Mock()
        actor._model.predict = Mock(side_effect=Exception("Test error"))
        pred, conf = MLSignalActor._predict(actor, features)
        assert pred == 0.0 and conf == 0.0
        actor.log.error.assert_called_with("Prediction failed: Test error")

    def test_update_prediction_history_and_regime(self):
        """
        Test prediction history and regime detection (lines 470-530).
        """
        actor = Mock()
        actor._signal_config = Mock()
        actor._signal_config.adaptive_window = 10
        actor._signal_config.signal_strategy = SignalStrategy.ADAPTIVE
        actor._prediction_window = np.zeros(10)
        actor._confidence_window = np.zeros(10)
        actor._volatility_window = np.zeros(10)
        actor._window_index = 0
        actor._prediction_history = []
        actor._confidence_history = []
        actor._indicator_manager = Mock()
        actor._indicator_manager.price_history = {"closes": [1.10, 1.11, 1.12]}

        bar = Mock()

        # Update history with volatility calculation
        MLSignalActor._update_prediction_history(actor, 0.5, 0.7, bar)
        assert actor._window_index == 1
        assert len(actor._prediction_history) == 1
        assert actor._volatility_window[0] > 0  # Volatility calculated

        # Test adaptive threshold update call
        actor._update_adaptive_threshold = Mock()
        MLSignalActor._update_prediction_history(actor, 0.6, 0.8, bar)
        actor._update_adaptive_threshold.assert_called_once()

    def test_signal_generation_methods(self):
        """
        Test signal generation methods (lines 670-690, 702-712, 724-744, 756-775).
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
        actor.clock = Mock()
        actor.clock.timestamp_ns = Mock(return_value=2000)

        bar = Mock()
        bar.bar_type = Mock()
        bar.bar_type.instrument_id = "EURUSD"
        bar.ts_event = 1000
        features = np.array([0.1, 0.2])

        # Test threshold signal (lines 702-712)
        signal = MLSignalActor._generate_threshold_signal(actor, bar, 0.8, 0.9, features)
        assert isinstance(signal, MLSignal)
        assert signal.features is not None  # log_predictions=True
        assert actor._last_signal_bar == 20

        # Test extremes signal (lines 724-744)
        actor._prediction_history = list(range(100))
        signal = MLSignalActor._generate_extremes_signal(actor, bar, 95.0, 0.8, features)
        assert signal is not None

        # Test momentum signal (lines 756-775)
        actor._prediction_history = [0.2, 0.4, 0.6, 0.8]
        signal = MLSignalActor._generate_momentum_signal(actor, bar, 0.9, 0.8, features)
        assert signal is not None
        assert signal.prediction == 1.0  # Clipped to 1.0

    def test_ensemble_and_adaptive_signals(self):
        """
        Test ensemble and adaptive signal generation (lines 788-821, 834-851).
        """
        actor = Mock()
        actor._config = Mock()
        actor._config.prediction_threshold = 0.7
        actor._config.log_predictions = False
        actor._ensemble_weights = {"threshold": 0.4, "extremes": 0.3, "momentum": 0.3}
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

        # Test ensemble signal (lines 788-821)
        signal1 = Mock(confidence=0.9)
        signal2 = Mock(confidence=0.8)
        signal3 = Mock(confidence=0.7)

        actor._generate_threshold_signal = Mock(return_value=signal1)
        actor._generate_extremes_signal = Mock(return_value=signal2)
        actor._generate_momentum_signal = Mock(return_value=signal3)

        signal = MLSignalActor._generate_ensemble_signal(actor, bar, 0.8, 0.85, features)
        assert signal.confidence == 0.81  # Weighted average

        # Test adaptive signal (lines 834-851)
        signal = MLSignalActor._generate_adaptive_signal(actor, bar, 0.8, 0.9, features)
        assert isinstance(signal, AdaptiveSignal)
        assert signal.adaptive_threshold == 0.5
        assert signal.signal_strength == 1.8
        assert signal.market_regime == "volatile"

    def test_performance_metrics_tracking(self):
        """
        Test performance metrics tracking (lines 873-904).
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

        # Mock metrics
        mock_metric = Mock()

        with patch("prometheus_client.Histogram", return_value=mock_metric):
            # First call creates metrics
            MLSignalActor._track_performance_metrics(actor, 0.75, 0.85, 3.5)

            assert hasattr(actor, "_prediction_distribution_metric")
            assert hasattr(actor, "_confidence_distribution_metric")
            assert actor._prediction_count == 1
            assert actor._total_inference_time == 3.5

            # Verify logging
            actor.log.debug.assert_called()

    def test_state_backup_and_restore(self):
        """
        Test state backup and restore (lines 917-942, 952-989).
        """
        actor = Mock()
        actor._indicator_manager = Mock()
        actor._indicator_manager.price_history = {"closes": [1.10, 1.11]}
        actor._indicator_manager.indicators = {
            "sma": Mock(value=50.0, initialized=True),
            "ema": Mock(value=60.0, initialized=False),
        }
        actor._prediction_history = [0.1, 0.2, 0.3]
        actor._confidence_history = [0.6, 0.7, 0.8]
        actor._prediction_window = np.array([0.1, 0.2])
        actor._confidence_window = np.array([0.3, 0.4])
        actor._volatility_window = np.array([0.01, 0.02])
        actor._window_index = 1
        actor._adaptive_threshold = 0.75
        actor._market_regime = "trending"
        actor._last_signal_bar = 15
        actor._signal_config = Mock()
        actor._signal_config.adaptive_window = 2
        actor._signal_config.min_signal_separation_bars = 2
        actor._config = Mock()
        actor._config.prediction_threshold = 0.5
        actor._indicator_state_backup = {}
        actor.log = Mock()

        # Backup (lines 917-942)
        MLSignalActor._backup_indicator_state(actor)

        backup = actor._indicator_state_backup
        assert "indicators" in backup
        assert "sma" in backup["indicators"]
        assert "ema" not in backup["indicators"]  # Not initialized
        assert backup["adaptive_threshold"] == 0.75

        # Restore (lines 952-989)
        # Modify state
        actor._prediction_history.clear()
        actor._adaptive_threshold = 0.99

        MLSignalActor._restore_indicator_state(actor)

        assert len(actor._prediction_history) == 3
        assert actor._adaptive_threshold == 0.75
        actor.log.info.assert_called_with("Indicator state restored from backup")

    def test_get_signal_statistics(self):
        """
        Test get_signal_statistics (lines 1001-1021).
        """
        actor = Mock()
        actor._signal_config = Mock()
        actor._signal_config.signal_strategy = SignalStrategy.ENSEMBLE
        actor._adaptive_threshold = 0.7
        actor._market_regime = "trending"
        actor._last_signal_bar = 10
        actor._prediction_history = [0.1, 0.2, 0.3]
        actor._feature_buffer = np.zeros(5)
        actor._ensemble_weights = {"threshold": 0.5}
        actor.get_health_status = Mock(return_value={"status": "healthy"})

        stats = MLSignalActor.get_signal_statistics(actor)

        assert stats["signal_strategy"] == "ensemble"
        assert stats["adaptive_threshold"] == 0.7
        assert stats["ensemble_weights"] == {"threshold": 0.5}
        assert stats["health_status"]["status"] == "healthy"

    def test_reset_signal_state(self):
        """
        Test reset_signal_state (lines 1031-1041).
        """
        actor = Mock()
        actor._prediction_history = [0.1, 0.2, 0.3]
        actor._confidence_history = [0.6, 0.7, 0.8]
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
        assert np.all(actor._prediction_window == 0.0)
        assert actor._adaptive_threshold == 0.5
        assert actor._market_regime == "unknown"
        assert actor._last_signal_bar == -2
        actor.log.info.assert_called_with("Signal generation state reset")
