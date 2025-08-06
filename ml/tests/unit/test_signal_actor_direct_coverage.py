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
Direct test methods for MLSignalActor to achieve 90%+ coverage.
"""

from unittest.mock import Mock

import numpy as np

from ml.actors.signal import AdaptiveSignal
from ml.actors.signal import MLSignalActor
from ml.actors.signal import SignalStrategy
from ml.config.base import MLFeatureConfig
from ml.features.engineering import FeatureConfig
from ml.features.engineering import IndicatorManager


class TestMLSignalActorDirectCoverage:
    """
    Direct test methods for uncovered MLSignalActor code.
    """

    def test_load_model_coverage(self):
        """
        Test _load_model method for coverage.
        """
        actor = Mock()
        actor.log = Mock()

        # With model (line 284)
        actor._model = Mock()
        MLSignalActor._load_model(actor)
        actor.log.info.assert_called_with("Model loaded successfully: Mock")

        # Without model (line 286)
        actor._model = None
        MLSignalActor._load_model(actor)
        actor.log.warning.assert_called_with("Model is None after loading")

    def test_initialize_features_coverage(self):
        """
        Test _initialize_features method for coverage.
        """
        actor = Mock()
        actor.log = Mock()

        # Test with FeatureConfig instance (lines 298-303)
        actor._feature_config = FeatureConfig()
        actor._indicator_manager = None
        actor._feature_buffer = np.zeros(5)
        actor._feature_engineer = Mock()
        actor._feature_engineer.n_features = 10

        MLSignalActor._initialize_features(actor)
        assert isinstance(actor._indicator_manager, IndicatorManager)

        # Test with non-FeatureConfig (lines 303-305)
        actor._feature_config = Mock()
        actor._feature_config.__class__ = MLFeatureConfig
        actor._indicator_manager = None

        MLSignalActor._initialize_features(actor)
        assert isinstance(actor._indicator_manager, IndicatorManager)

        # Test with feature names (lines 308-310)
        actor._feature_config = Mock()
        actor._feature_config.feature_names = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]
        actor._feature_buffer = np.zeros(5)
        actor._indicator_manager = None

        MLSignalActor._initialize_features(actor)
        assert actor._feature_buffer.size >= 8

        # Test log message (line 315)
        actor.log.info.assert_called_with(
            f"Feature buffer resized from 5 to {actor._feature_buffer.size} to match feature configuration",
        )

    def test_compute_features_coverage(self):
        """
        Test _compute_features method for coverage.
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

        # Verify current_bar dictionary was created
        call_args = actor._feature_engineer.calculate_features_online.call_args
        current_bar = call_args[1]["current_bar"]
        assert current_bar["close"] == 1.1000
        assert current_bar["volume"] == 1000.0
        assert current_bar["high"] == 1.1001
        assert current_bar["low"] == 1.0999

        # Test slow computation warning (lines 366-369)
        import time

        def slow_update(bar):
            time.sleep(0.01)

        actor._indicator_manager.update_from_bar = slow_update
        result = MLSignalActor._compute_features(actor, bar)
        actor.log.warning.assert_called()

    def test_predict_methods_coverage(self):
        """
        Test prediction methods for coverage.
        """
        actor = Mock()
        actor.log = Mock()
        features = np.array([0.1, 0.2, 0.3])

        # No model (lines 389-391)
        actor._model = None
        pred, conf = MLSignalActor._predict(actor, features)
        assert pred == 0.0 and conf == 0.0
        actor.log.warning.assert_called_with("No model loaded, returning zero prediction")

        # ONNX model path (line 397)
        actor._model = Mock()
        actor._model.run = Mock(return_value=[[np.array([[0.8]])]])
        actor._model_metadata = {"input_names": ["input"], "output_names": ["output"]}
        actor._predict_onnx = Mock(return_value=(0.8, 0.8))
        pred, conf = MLSignalActor._predict(actor, features)

        # Sklearn with predict_proba (line 399)
        del actor._model.run
        actor._model.predict_proba = Mock(return_value=np.array([[0.3, 0.7]]))
        actor._predict_sklearn_proba = Mock(return_value=(1.0, 0.7))
        pred, conf = MLSignalActor._predict(actor, features)

        # Sklearn basic (line 402)
        del actor._model.predict_proba
        actor._model.predict = Mock(return_value=np.array([0.85]))
        actor._predict_sklearn = Mock(return_value=(0.85, 0.85))
        pred, conf = MLSignalActor._predict(actor, features)

        # Unsupported model (lines 404-405)
        actor._model = object()
        pred, conf = MLSignalActor._predict(actor, features)
        assert pred == 0.0 and conf == 0.0
        actor.log.error.assert_called_with("Unsupported model type: <class 'object'>")

        # Exception handling (lines 406-409)
        actor._model = Mock()
        actor._model.predict = Mock(side_effect=Exception("Test error"))
        pred, conf = MLSignalActor._predict(actor, features)
        assert pred == 0.0 and conf == 0.0
        actor.log.error.assert_called_with("Prediction failed: Test error")

    def test_predict_onnx_coverage(self):
        """
        Test _predict_onnx method for coverage.
        """
        actor = Mock()
        actor._model = Mock()
        actor._model_metadata = {"input_names": ["input"], "output_names": ["pred", "conf"]}
        features = np.array([0.1, 0.2, 0.3])

        # Dual output (lines 419-421)
        actor._model.run = Mock(return_value=[np.array([[0.8]]), np.array([[0.9]])])
        pred, conf = MLSignalActor._predict_onnx(actor, features)
        assert pred == 0.8 and conf == 0.9

        # Single output (lines 423-428)
        actor._model.run = Mock(return_value=[np.array([[0.7]])])
        pred, conf = MLSignalActor._predict_onnx(actor, features)
        assert pred == 0.7 and conf == 0.7

    def test_predict_sklearn_methods_coverage(self):
        """
        Test sklearn prediction methods for coverage.
        """
        actor = Mock()
        features = np.array([0.1, 0.2, 0.3])

        # predict_proba (lines 434-441)
        actor._model = Mock()
        actor._model.predict_proba = Mock(return_value=np.array([[0.3, 0.7]]))
        pred, conf = MLSignalActor._predict_sklearn_proba(actor, features)
        assert pred == 1.0 and conf == 0.7

        # basic predict (lines 447-453)
        actor._model.predict = Mock(return_value=np.array([0.85]))
        pred, conf = MLSignalActor._predict_sklearn(actor, features)
        assert pred == 0.85 and conf == 0.85

        # Zero prediction special case (line 450)
        actor._model.predict = Mock(return_value=np.array([0.0]))
        pred, conf = MLSignalActor._predict_sklearn(actor, features)
        assert pred == 0.0 and conf == 0.5

    def test_update_prediction_history_coverage(self):
        """
        Test _update_prediction_history method for coverage.
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

        # Without indicator manager (lines 470-478)
        MLSignalActor._update_prediction_history(actor, 0.5, 0.7, bar)
        assert actor._window_index == 1
        assert len(actor._prediction_history) == 1
        assert actor._prediction_history[0] == 0.5
        assert actor._prediction_window[0] == 0.5

        # With price history (lines 479-489)
        actor._indicator_manager = Mock()
        actor._indicator_manager.price_history = {"closes": [1.10, 1.11, 1.12]}
        MLSignalActor._update_prediction_history(actor, 0.6, 0.8, bar)
        assert actor._window_index == 2

        # Test wraparound (line 488)
        actor._window_index = 9
        MLSignalActor._update_prediction_history(actor, 0.7, 0.9, bar)
        assert actor._window_index == 0

        # Test adaptive strategy update (lines 521-530)
        actor._signal_config.signal_strategy = SignalStrategy.ADAPTIVE
        actor._update_adaptive_threshold = Mock()
        MLSignalActor._update_prediction_history(actor, 0.8, 0.9, bar)
        actor._update_adaptive_threshold.assert_called_once()

    def test_adaptive_threshold_update_coverage(self):
        """
        Test _update_adaptive_threshold method for coverage.
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

        # Test update (lines 549-567)
        MLSignalActor._update_adaptive_threshold(actor)

        # Verify threshold was updated
        assert actor._adaptive_threshold != 0.5
        assert 0.1 <= actor._adaptive_threshold <= 0.95

        # Verify metric was updated
        actor._adaptive_threshold_metric.set.assert_called_once()
        call_args = actor._adaptive_threshold_metric.set.call_args[0]
        assert call_args[0] == actor._adaptive_threshold

    def test_market_regime_detection_coverage(self):
        """
        Test _detect_market_regime method for coverage.
        """
        actor = Mock()
        actor._indicator_manager = None
        actor._market_regime = "unknown"
        actor._market_regime_metric = Mock()
        actor.log = Mock()
        actor.id = Mock()
        actor.id.value = "TEST"

        bar = Mock()

        # No indicator manager (lines 577-578)
        MLSignalActor._detect_market_regime(actor, bar)
        assert actor._market_regime == "unknown"

        # No price history (lines 579-581)
        actor._indicator_manager = Mock()
        actor._indicator_manager.price_history = {}
        MLSignalActor._detect_market_regime(actor, bar)
        assert actor._market_regime == "unknown"

        # Insufficient data (lines 582-584)
        actor._indicator_manager.price_history = {"closes": [1.10, 1.11]}
        MLSignalActor._detect_market_regime(actor, bar)
        assert actor._market_regime == "unknown"

        # Volatile regime (lines 585-591)
        actor._indicator_manager.price_history = {
            "closes": [1.10 + (i % 2) * 0.05 for i in range(25)],
        }
        MLSignalActor._detect_market_regime(actor, bar)
        assert actor._market_regime == "volatile"
        actor._market_regime_metric.set.assert_called_with(
            2,
            {"actor_id": "TEST", "regime": "volatile"},
        )

    def test_signal_generation_strategies_coverage(self):
        """
        Test all signal generation strategies for coverage.
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

        # Threshold signal (lines 702-712)
        signal = MLSignalActor._generate_threshold_signal(actor, bar, 0.8, 0.9, features)
        assert signal is not None
        assert signal.features is not None  # log_predictions=True
        assert actor._last_signal_bar == 20

        # Extremes signal (lines 724-744)
        # Insufficient history
        actor._prediction_history = [0.5, 0.6]
        signal = MLSignalActor._generate_extremes_signal(actor, bar, 0.9, 0.8, features)
        assert signal is None

        # With sufficient history - top extreme
        actor._prediction_history = list(range(100))
        signal = MLSignalActor._generate_extremes_signal(actor, bar, 95.0, 0.8, features)
        assert signal is not None

        # Bottom extreme
        signal = MLSignalActor._generate_extremes_signal(actor, bar, 5.0, 0.8, features)
        assert signal is not None

        # Middle value (not extreme)
        signal = MLSignalActor._generate_extremes_signal(actor, bar, 50.0, 0.8, features)
        assert signal is None

        # Low confidence
        signal = MLSignalActor._generate_extremes_signal(actor, bar, 95.0, 0.5, features)
        assert signal is None

        # Momentum signal (lines 756-775)
        # Insufficient history
        actor._prediction_history = [0.5]
        signal = MLSignalActor._generate_momentum_signal(actor, bar, 0.8, 0.9, features)
        assert signal is None

        # With momentum
        actor._prediction_history = [0.2, 0.4, 0.6, 0.8]
        signal = MLSignalActor._generate_momentum_signal(actor, bar, 0.9, 0.8, features)
        assert signal is not None
        assert signal.prediction != 0.9  # Adjusted by momentum

        # No momentum
        actor._prediction_history = [0.5, 0.5, 0.5, 0.5]
        signal = MLSignalActor._generate_momentum_signal(actor, bar, 0.5, 0.8, features)
        assert signal is None

        # Ensemble signal (lines 788-821)
        mock_signal1 = Mock(confidence=0.9)
        mock_signal2 = Mock(confidence=0.8)
        mock_signal3 = Mock(confidence=0.7)

        actor._generate_threshold_signal = Mock(return_value=mock_signal1)
        actor._generate_extremes_signal = Mock(return_value=mock_signal2)
        actor._generate_momentum_signal = Mock(return_value=mock_signal3)

        signal = MLSignalActor._generate_ensemble_signal(actor, bar, 0.8, 0.85, features)
        assert signal is not None
        assert signal.confidence >= 0.8

        # No signals
        actor._generate_threshold_signal = Mock(return_value=None)
        actor._generate_extremes_signal = Mock(return_value=None)
        actor._generate_momentum_signal = Mock(return_value=None)
        signal = MLSignalActor._generate_ensemble_signal(actor, bar, 0.8, 0.85, features)
        assert signal is None

        # Adaptive signal (lines 834-851)
        actor._adaptive_threshold = 0.5
        actor._market_regime = "volatile"
        signal = MLSignalActor._generate_adaptive_signal(actor, bar, 0.8, 0.9, features)
        assert signal is not None
        assert isinstance(signal, AdaptiveSignal)
        assert signal.adaptive_threshold == 0.5
        assert signal.signal_strength == 1.8
        assert signal.market_regime == "volatile"

        # Zero threshold
        actor._adaptive_threshold = 0.0
        signal = MLSignalActor._generate_adaptive_signal(actor, bar, 0.8, 0.9, features)
        assert signal is None

    def test_generate_signal_by_strategy_coverage(self):
        """
        Test _generate_signal_by_strategy method for coverage.
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

        # Signal separation check (lines 608-611)
        signal = MLSignalActor._generate_signal_by_strategy(actor, bar, 0.8, 0.9, features)
        assert signal is None
        actor.log.debug.assert_called_with(
            "Skipping signal generation due to minimum separation requirement",
        )

        # Valid separation
        actor._last_signal_bar = 0

        # Mock all strategy methods
        mock_signal = Mock()
        actor._generate_threshold_signal = Mock(return_value=mock_signal)
        actor._generate_extremes_signal = Mock(return_value=mock_signal)
        actor._generate_momentum_signal = Mock(return_value=mock_signal)
        actor._generate_ensemble_signal = Mock(return_value=mock_signal)
        actor._generate_adaptive_signal = Mock(return_value=mock_signal)

        # Test each strategy (lines 613-640)
        for strategy in SignalStrategy:
            actor._signal_config.signal_strategy = strategy
            signal = MLSignalActor._generate_signal_by_strategy(actor, bar, 0.8, 0.9, features)
            assert signal is not None

        # Unknown strategy
        actor._signal_config.signal_strategy = "unknown"
        signal = MLSignalActor._generate_signal_by_strategy(actor, bar, 0.8, 0.9, features)
        assert signal is None
        actor.log.warning.assert_called_with("Unknown signal strategy: unknown")

    def test_generate_prediction_protected_coverage(self):
        """
        Test _generate_prediction_protected method for coverage.
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

        # Success with signal (lines 470-520)
        actor._predict = Mock(return_value=(0.8, 0.9))
        actor._update_prediction_history = Mock()
        actor._detect_market_regime = Mock()
        actor._generate_signal_by_strategy = Mock(return_value=Mock(prediction=0.8))
        actor._publish_signal = Mock()
        actor._track_performance_metrics = Mock()

        MLSignalActor._generate_prediction_protected(actor, bar, features)

        actor._circuit_breaker.record_success.assert_called()
        actor._health_monitor.update_prediction_success.assert_called()
        actor._signals_generated_metric.inc.assert_called()

        # Success without signal (line 521)
        actor._generate_signal_by_strategy = Mock(return_value=None)
        MLSignalActor._generate_prediction_protected(actor, bar, features)
        actor._circuit_breaker.record_success.assert_called()

        # Regime detection disabled
        actor._signal_config.enable_regime_detection = False
        MLSignalActor._generate_prediction_protected(actor, bar, features)
        actor._detect_market_regime.assert_not_called()

        # Exception (lines 522-530)
        actor._predict = Mock(side_effect=Exception("Test error"))
        MLSignalActor._generate_prediction_protected(actor, bar, features)

        actor.log.error.assert_called_with("Signal generation failed: Test error")
        actor._circuit_breaker.record_failure.assert_called()
        actor._health_monitor.update_prediction_failure.assert_called()

    def test_track_performance_metrics_coverage(self):
        """
        Test _track_performance_metrics method for coverage.
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

        # First call creates metrics (lines 873-885)
        MLSignalActor._track_performance_metrics(actor, 0.75, 0.85, 3.5)

        assert hasattr(actor, "_prediction_distribution_metric")
        assert hasattr(actor, "_confidence_distribution_metric")
        assert actor._prediction_count == 1
        assert actor._total_inference_time == 3.5

        # Verify metrics were called
        actor._prediction_distribution_metric.observe.assert_called_with(
            0.75,
            {"actor_id": "TEST"},
        )
        actor._confidence_distribution_metric.observe.assert_called_with(
            0.85,
            {"actor_id": "TEST"},
        )

        # Verify debug logging (lines 890-904)
        actor.log.debug.assert_called()
        log_msg = actor.log.debug.call_args[0][0]
        assert "Prediction: 0.7500" in log_msg
        assert "Confidence: 0.8500" in log_msg
        assert "Signal time: 3.500ms" in log_msg
        assert "Strategy: adaptive" in log_msg

        # Test averaging (lines 902-903)
        for i in range(4):
            MLSignalActor._track_performance_metrics(actor, 0.8, 0.9, 4.0)

        actor.log.debug.assert_called()
        log_msg = actor.log.debug.call_args[0][0]
        assert "Avg time: " in log_msg

    def test_backup_restore_state_coverage(self):
        """
        Test backup and restore state methods for coverage.
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
        actor._prediction_window = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        actor._confidence_window = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        actor._volatility_window = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        actor._window_index = 3
        actor._adaptive_threshold = 0.75
        actor._market_regime = "trending"
        actor._last_signal_bar = 15
        actor._signal_config = Mock()
        actor._signal_config.adaptive_window = 5
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
        assert backup["window_index"] == 3
        assert backup["adaptive_threshold"] == 0.75

        # Restore (lines 952-989)
        # Modify state
        actor._prediction_history.clear()
        actor._adaptive_threshold = 0.99
        actor._prediction_window = np.zeros(10)  # Different size

        MLSignalActor._restore_indicator_state(actor)

        assert len(actor._prediction_history) == 3
        assert actor._adaptive_threshold == 0.75
        actor.log.info.assert_called_with("Indicator state restored from backup")

        # No backup
        actor._indicator_state_backup = None
        MLSignalActor._restore_indicator_state(actor)
        actor.log.warning.assert_called_with("No indicator state backup found")

        # No indicator manager during backup
        actor._indicator_manager = None
        actor._indicator_state_backup = {}
        MLSignalActor._backup_indicator_state(actor)
        assert "indicators" not in actor._indicator_state_backup

    def test_get_signal_statistics_coverage(self):
        """
        Test get_signal_statistics method for coverage.
        """
        actor = Mock()
        actor._signal_config = Mock()
        actor._signal_config.signal_strategy = SignalStrategy.ENSEMBLE
        actor._adaptive_threshold = 0.7
        actor._market_regime = "trending"
        actor._last_signal_bar = 10
        actor._prediction_history = [0.1, 0.2, 0.3]
        actor._feature_buffer = np.zeros(5)
        actor._ensemble_weights = {"threshold": 0.5, "extremes": 0.3, "momentum": 0.2}
        actor.get_health_status = Mock(return_value={"status": "healthy"})

        # Test with ensemble (lines 1001-1021)
        stats = MLSignalActor.get_signal_statistics(actor)

        assert stats["signal_strategy"] == SignalStrategy.ENSEMBLE.value
        assert stats["adaptive_threshold"] == 0.7
        assert stats["market_regime"] == "trending"
        assert stats["last_signal_bar"] == 10
        assert stats["prediction_history_length"] == 3
        assert stats["feature_buffer_size"] == 5
        assert stats["ensemble_weights"] == {"threshold": 0.5, "extremes": 0.3, "momentum": 0.2}
        assert stats["health_status"] == {"status": "healthy"}

        # Test without ensemble
        actor._signal_config.signal_strategy = SignalStrategy.THRESHOLD
        stats = MLSignalActor.get_signal_statistics(actor)
        assert stats["ensemble_weights"] is None

    def test_reset_signal_state_coverage(self):
        """
        Test reset_signal_state method for coverage.
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

        # Reset (lines 1031-1041)
        MLSignalActor.reset_signal_state(actor)

        assert len(actor._prediction_history) == 0
        assert len(actor._confidence_history) == 0
        assert np.all(actor._prediction_window == 0.0)
        assert np.all(actor._confidence_window == 0.0)
        assert np.all(actor._volatility_window == 0.0)
        assert actor._window_index == 0
        assert actor._adaptive_threshold == 0.5
        assert actor._market_regime == "unknown"
        assert actor._last_signal_bar == -2
        actor.log.info.assert_called_with("Signal actor state reset to initial values")
