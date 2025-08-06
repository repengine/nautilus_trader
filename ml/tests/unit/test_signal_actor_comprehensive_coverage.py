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
Comprehensive test suite for MLSignalActor to achieve 90%+ coverage.
"""

import contextlib
import pickle
import tempfile
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np

from ml.actors.signal import AdaptiveSignal
from ml.actors.signal import MLSignalActor
from ml.actors.signal import SignalStrategy
from ml.config.base import MLFeatureConfig
from ml.features.engineering import FeatureConfig


class SimpleModel:
    """
    Simple model for testing.
    """

    def predict(self, X):
        return np.array([0.8])

    def predict_proba(self, X):
        return np.array([[0.2, 0.8]])


class TestMLSignalActorComprehensiveCoverage:
    """
    Comprehensive tests for MLSignalActor to achieve 90%+ coverage.
    """

    def setup_method(self):
        """
        Set up test fixtures.
        """
        # Clear metrics
        try:
            import gc

            from prometheus_client import REGISTRY

            collectors = list(REGISTRY._collector_to_names.keys())
            for collector in collectors:
                with contextlib.suppress(Exception):
                    REGISTRY.unregister(collector)
            gc.collect()
        except ImportError:
            pass

        # Create model file
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        with open(self.temp_file.name, "wb") as f:
            pickle.dump(SimpleModel(), f)
        self.temp_file.close()

    def teardown_method(self):
        """
        Clean up.
        """
        Path(self.temp_file.name).unlink(missing_ok=True)

    def test_initialization_with_mocks(self):
        """
        Test MLSignalActor initialization with proper mocking.
        """
        with patch("ml.actors.base.BaseMLInferenceActor.__init__"):
            with patch("ml.features.engineering.FeatureEngineer") as mock_fe:
                with patch("prometheus_client.Histogram") as mock_histogram:
                    with patch("prometheus_client.Counter") as mock_counter:
                        with patch("prometheus_client.Gauge") as mock_gauge:
                            mock_fe.return_value.n_features = 10
                            mock_histogram.return_value = Mock()
                            mock_counter.return_value = Mock()
                            mock_gauge.return_value = Mock()

                            # Test with default feature config (lines 212-216)
                            config = Mock()
                            config.feature_config = None
                            config.min_signal_separation_bars = 2
                            config.prediction_threshold = 0.7
                            config.adaptive_window = 10
                            config.ensemble_weights = None
                            config.signal_strategy = Mock(value="threshold")

                            with patch.object(MLSignalActor, "log", create=True):
                                actor = MLSignalActor(config)

                                # Verify initialization
                                assert actor._signal_config == config
                                assert actor._feature_config is not None
                                assert isinstance(actor._feature_config, FeatureConfig)
                                assert actor._adaptive_threshold == 0.7
                                assert actor._market_regime == "unknown"
                                assert actor._window_index == 0
                                assert actor._ensemble_weights["threshold"] == 0.4

                            # Test with MLFeatureConfig base class (lines 212-216)
                            base_config = MLFeatureConfig()
                            config2 = Mock()
                            config2.feature_config = base_config
                            config2.min_signal_separation_bars = 2
                            config2.prediction_threshold = 0.7
                            config2.adaptive_window = 10
                            config2.ensemble_weights = {"custom": 0.5}  # Line 242
                            config2.signal_strategy = Mock(value="ensemble")

                            with patch.object(MLSignalActor, "log", create=True):
                                actor2 = MLSignalActor(config2)

                                # Should create default FeatureConfig from MLFeatureConfig
                                assert isinstance(actor2._feature_config, FeatureConfig)
                                assert actor2._ensemble_weights == {"custom": 0.5}

    def test_all_actor_methods(self):
        """
        Test all MLSignalActor methods with proper mocking.
        """
        # Create a comprehensive mock actor
        actor = Mock(spec=MLSignalActor)

        # Setup all attributes
        actor._model = Mock()
        actor._signal_config = Mock()
        actor._signal_config.min_signal_separation_bars = 2
        actor._signal_config.adaptive_window = 10
        actor._signal_config.extremes_top_pct = 0.1
        actor._signal_config.momentum_lookback = 3
        actor._signal_config.signal_strategy = SignalStrategy.THRESHOLD
        actor._signal_config.enable_regime_detection = True
        actor._signal_config.adaptive_volatility_factor = 2.0
        actor._config = Mock()
        actor._config.prediction_threshold = 0.7
        actor._config.log_predictions = True
        actor._config.max_feature_latency_ms = 5
        actor._indicator_manager = Mock()
        actor._feature_engineer = Mock()
        actor._feature_engineer.n_features = 10
        actor._feature_engineer.calculate_features_online = Mock(return_value=np.array([0.1, 0.2]))
        actor._prediction_history = []
        actor._confidence_history = []
        actor._bars_processed = 10
        actor._last_signal_bar = 0
        actor._adaptive_threshold = 0.5
        actor._market_regime = "unknown"
        actor._window_index = 0
        actor._prediction_window = np.zeros(10)
        actor._confidence_window = np.zeros(10)
        actor._volatility_window = np.zeros(10)
        actor._ensemble_weights = {"threshold": 0.4, "extremes": 0.3, "momentum": 0.3}
        actor._model_metadata = {"input_names": ["input"], "output_names": ["pred"]}
        actor._prediction_count = 0
        actor._total_inference_time = 0.0
        actor._circuit_breaker = Mock()
        actor._circuit_breaker.record_success = Mock()
        actor._circuit_breaker.record_failure = Mock()
        actor._health_monitor = Mock()
        actor._health_monitor.update_prediction_success = Mock()
        actor._health_monitor.update_prediction_failure = Mock()
        actor._indicator_state_backup = {}
        actor._feature_config = FeatureConfig()
        actor._feature_buffer = np.zeros(10)
        actor.log = Mock()
        actor.clock = Mock()
        actor.clock.timestamp_ns = Mock(return_value=1000)
        actor.id = Mock()
        actor.id.value = "TEST"
        actor._signal_generation_time_metric = Mock()
        actor._signals_generated_metric = Mock()
        actor._prediction_distribution_metric = Mock()
        actor._confidence_distribution_metric = Mock()
        actor._adaptive_threshold_metric = Mock()
        actor._market_regime_metric = Mock()
        actor.get_health_status = Mock(return_value={"status": "healthy"})

        # Test _load_model (lines 283-286)
        MLSignalActor._load_model(actor)
        actor.log.info.assert_called()

        actor._model = None
        MLSignalActor._load_model(actor)
        actor.log.warning.assert_called()

        # Test _initialize_features (lines 298-315)
        # Test with MLFeatureConfig
        actor._feature_config = Mock()
        actor._feature_config.__class__ = MLFeatureConfig
        actor._indicator_manager = None
        MLSignalActor._initialize_features(actor)

        # Test with feature names
        actor._feature_config = FeatureConfig(
            feature_names=[
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
                "f11",
                "f12",
            ],
        )
        actor._feature_buffer = np.zeros(5)
        actor._indicator_manager = None
        MLSignalActor._initialize_features(actor)
        assert actor._feature_buffer.size >= 12

        # Test _compute_features (lines 338-369)
        bar = Mock()
        bar.close = Mock()
        bar.close.__float__ = Mock(return_value=1.1000)
        bar.volume = Mock()
        bar.volume.__float__ = Mock(return_value=1000.0)
        bar.high = Mock()
        bar.high.__float__ = Mock(return_value=1.1001)
        bar.low = Mock()
        bar.low.__float__ = Mock(return_value=1.0999)

        # No indicator manager
        actor._indicator_manager = None
        result = MLSignalActor._compute_features(actor, bar)
        assert result is None

        # Not initialized
        actor._indicator_manager = Mock()
        actor._indicator_manager.all_initialized = Mock(return_value=False)
        result = MLSignalActor._compute_features(actor, bar)
        assert result is None

        # Successful computation
        actor._indicator_manager.all_initialized = Mock(return_value=True)
        actor._indicator_manager.update_from_bar = Mock()
        result = MLSignalActor._compute_features(actor, bar)
        assert result is not None

        # Test all prediction methods (lines 389-409)
        features = np.array([0.1, 0.2, 0.3])

        # No model
        actor._model = None
        pred, conf = MLSignalActor._predict(actor, features)
        assert pred == 0.0 and conf == 0.0

        # ONNX model (line 397)
        actor._model = Mock()
        actor._model.run = Mock(return_value=[[np.array([[0.8]])]])
        pred, conf = MLSignalActor._predict(actor, features)

        # Sklearn with predict_proba (line 399)
        del actor._model.run
        actor._model.predict_proba = Mock(return_value=np.array([[0.3, 0.7]]))
        pred, conf = MLSignalActor._predict(actor, features)

        # Sklearn basic (line 402)
        del actor._model.predict_proba
        actor._model.predict = Mock(return_value=np.array([0.85]))
        pred, conf = MLSignalActor._predict(actor, features)

        # Unsupported model (line 404)
        actor._model = object()
        pred, conf = MLSignalActor._predict(actor, features)
        assert pred == 0.0 and conf == 0.0
        actor.log.error.assert_called_with("Unsupported model type: <class 'object'>")

        # Exception handling (line 406)
        actor._model = Mock()
        actor._model.predict = Mock(side_effect=Exception("Test error"))
        pred, conf = MLSignalActor._predict(actor, features)
        assert pred == 0.0 and conf == 0.0

        # Test ONNX prediction methods (lines 415-428)
        actor._model = Mock()
        actor._model.run = Mock(return_value=[np.array([[0.8]]), np.array([[0.9]])])
        actor._model_metadata = {"input_names": ["input"], "output_names": ["pred", "conf"]}
        pred, conf = MLSignalActor._predict_onnx(actor, features)
        assert pred == 0.8 and conf == 0.9

        # Single output
        actor._model.run = Mock(return_value=[np.array([[0.7]])])
        pred, conf = MLSignalActor._predict_onnx(actor, features)
        assert pred == 0.7 and conf == 0.7

        # Test sklearn prediction methods (lines 434-441, 447-453)
        actor._model = Mock()
        actor._model.predict_proba = Mock(return_value=np.array([[0.3, 0.7]]))
        pred, conf = MLSignalActor._predict_sklearn_proba(actor, features)
        assert pred == 1.0 and conf == 0.7

        actor._model.predict = Mock(return_value=np.array([0.0]))
        pred, conf = MLSignalActor._predict_sklearn(actor, features)
        assert pred == 0.0 and conf == 0.5  # Special case for zero

        # Test _update_prediction_history (lines 470-530)
        actor._indicator_manager = Mock()
        actor._indicator_manager.price_history = {"closes": [1.10, 1.11, 1.12]}
        MLSignalActor._update_prediction_history(actor, 0.5, 0.7, bar)
        assert actor._window_index == 1

        # With adaptive strategy
        actor._signal_config.signal_strategy = SignalStrategy.ADAPTIVE
        actor._update_adaptive_threshold = Mock()
        MLSignalActor._update_prediction_history(actor, 0.7, 0.9, bar)
        actor._update_adaptive_threshold.assert_called()

        # Test _update_adaptive_threshold (lines 549-567)
        MLSignalActor._update_adaptive_threshold(actor)
        actor._adaptive_threshold_metric.set.assert_called()

        # Test _detect_market_regime (lines 577-591)
        actor._indicator_manager.price_history = {
            "closes": [1.10 + (i % 2) * 0.05 for i in range(25)],
        }
        MLSignalActor._detect_market_regime(actor, bar)
        assert actor._market_regime == "volatile"

        # Test signal generation methods (lines 608-640, 670-690, 702-712, 724-744, 756-775, 788-821, 834-851)
        bar.bar_type = Mock()
        bar.bar_type.instrument_id = "EURUSD"
        bar.ts_event = 1000

        # Threshold signal (lines 702-712)
        signal = MLSignalActor._generate_threshold_signal(actor, bar, 0.8, 0.9, features)
        assert signal is not None
        assert signal.features is not None  # log_predictions=True

        # Extremes signal (lines 724-744)
        actor._prediction_history = list(range(100))
        signal = MLSignalActor._generate_extremes_signal(actor, bar, 95.0, 0.8, features)
        assert signal is not None

        # Low confidence
        signal = MLSignalActor._generate_extremes_signal(actor, bar, 95.0, 0.5, features)
        assert signal is None

        # Momentum signal (lines 756-775)
        actor._prediction_history = [0.2, 0.4, 0.6, 0.8]
        signal = MLSignalActor._generate_momentum_signal(actor, bar, 0.9, 0.8, features)
        assert signal is not None

        # Ensemble signal (lines 788-821)
        mock_signal1 = Mock(confidence=0.9)
        mock_signal2 = Mock(confidence=0.8)
        mock_signal3 = Mock(confidence=0.7)

        actor._generate_threshold_signal = Mock(return_value=mock_signal1)
        actor._generate_extremes_signal = Mock(return_value=mock_signal2)
        actor._generate_momentum_signal = Mock(return_value=mock_signal3)

        signal = MLSignalActor._generate_ensemble_signal(actor, bar, 0.8, 0.85, features)
        assert signal is not None

        # Adaptive signal (lines 834-851)
        actor._adaptive_threshold = 0.5
        actor._market_regime = "volatile"
        signal = MLSignalActor._generate_adaptive_signal(actor, bar, 0.8, 0.9, features)
        assert signal is not None
        assert isinstance(signal, AdaptiveSignal)

        # Test _generate_signal_by_strategy (lines 608-640)
        actor._signal_config.signal_strategy = SignalStrategy.THRESHOLD
        actor._bars_processed = 10
        actor._last_signal_bar = 0
        actor._generate_threshold_signal = Mock(return_value=Mock())

        signal = MLSignalActor._generate_signal_by_strategy(actor, bar, 0.8, 0.9, features)
        assert signal is not None

        # Test _generate_prediction_protected (lines 470-530)
        actor._predict = Mock(return_value=(0.8, 0.9))
        actor._update_prediction_history = Mock()
        actor._detect_market_regime = Mock()
        actor._generate_signal_by_strategy = Mock(return_value=Mock(prediction=0.8))
        actor._publish_signal = Mock()
        actor._track_performance_metrics = Mock()

        MLSignalActor._generate_prediction_protected(actor, bar, features)

        # No signal case (line 521)
        actor._generate_signal_by_strategy = Mock(return_value=None)
        MLSignalActor._generate_prediction_protected(actor, bar, features)
        actor._circuit_breaker.record_success.assert_called()

        # Exception case (lines 522-530)
        actor._predict = Mock(side_effect=Exception("Test error"))
        MLSignalActor._generate_prediction_protected(actor, bar, features)
        actor._circuit_breaker.record_failure.assert_called()
        actor._health_monitor.update_prediction_failure.assert_called()

        # Test _track_performance_metrics (lines 890-904)
        actor._config.log_predictions = True
        actor._signal_config.signal_strategy = SignalStrategy.ADAPTIVE
        MLSignalActor._track_performance_metrics(actor, 0.75, 0.85, 3.5)
        assert actor._prediction_count == 1
        assert actor._total_inference_time == 3.5
        actor.log.debug.assert_called()

        # Test backup/restore methods (lines 917-942, 952-989)
        actor._indicator_manager.indicators = {"sma": Mock(value=50.0, initialized=True)}
        actor._indicator_manager.price_history = {"closes": [1.10]}
        actor._prediction_history = [0.1, 0.2]
        actor._confidence_history = [0.6, 0.7]

        MLSignalActor._backup_indicator_state(actor)
        assert "prediction_history" in actor._indicator_state_backup

        MLSignalActor._restore_indicator_state(actor)

        # Test get_signal_statistics (lines 1001-1021)
        stats = MLSignalActor.get_signal_statistics(actor)
        assert "signal_strategy" in stats

        # Test reset_signal_state (lines 1031-1041)
        MLSignalActor.reset_signal_state(actor)
        assert len(actor._prediction_history) == 0

    def test_additional_edge_cases(self):
        """
        Test additional edge cases for complete coverage.
        """
        actor = Mock(spec=MLSignalActor)

        # Setup minimal attributes
        actor._signal_config = Mock()
        actor._signal_config.adaptive_window = 10
        actor._signal_config.extremes_top_pct = 0.1
        actor._signal_config.momentum_lookback = 3
        actor._config = Mock()
        actor._config.prediction_threshold = 0.7
        actor._config.log_predictions = False
        actor._prediction_history = []
        actor._bars_processed = 20
        actor._last_signal_bar = 0
        actor._adaptive_threshold = 0.5
        actor._market_regime = "unknown"
        actor.clock = Mock()
        actor.clock.timestamp_ns = Mock(return_value=2000)
        actor.log = Mock()

        bar = Mock()
        bar.bar_type = Mock()
        bar.bar_type.instrument_id = "TEST"
        bar.ts_event = 1000
        features = np.array([0.1, 0.2])

        # Test extremes with insufficient history
        signal = MLSignalActor._generate_extremes_signal(actor, bar, 0.9, 0.8, features)
        assert signal is None

        # Test momentum with insufficient history
        signal = MLSignalActor._generate_momentum_signal(actor, bar, 0.8, 0.9, features)
        assert signal is None

        # Test adaptive with zero threshold
        actor._adaptive_threshold = 0.0
        signal = MLSignalActor._generate_adaptive_signal(actor, bar, 0.8, 0.9, features)
        assert signal is None

        # Test ensemble with no signals
        actor._ensemble_weights = {"threshold": 1.0}
        actor._generate_threshold_signal = Mock(return_value=None)
        actor._generate_extremes_signal = Mock(return_value=None)
        actor._generate_momentum_signal = Mock(return_value=None)

        signal = MLSignalActor._generate_ensemble_signal(actor, bar, 0.8, 0.9, features)
        assert signal is None
