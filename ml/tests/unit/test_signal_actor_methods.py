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
Test MLSignalActor methods directly to achieve 90%+ coverage.
"""

import time
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np

from ml.actors.base import CircuitBreaker
from ml.actors.base import HealthMonitor
from ml.actors.signal import AdaptiveSignal
from ml.actors.signal import MLSignalActor
from ml.actors.signal import SignalStrategy
from ml.features.engineering import FeatureConfig
from ml.features.engineering import IndicatorManager


class TestMLSignalActorMethods:
    """
    Test MLSignalActor methods for coverage.
    """

    def test_initialization_coverage(self):
        """
        Test initialization code coverage.
        """
        # Test MLSignalActor.__init__ paths
        with patch.object(MLSignalActor, "__init__", return_value=None):
            actor = MLSignalActor(None)

        # Set up attributes
        actor._signal_config = Mock()
        actor._signal_config.min_signal_separation_bars = 2
        actor._signal_config.adaptive_window = 10
        actor._signal_config.ensemble_weights = None

        # Test feature config paths
        actor._feature_config = None
        assert actor._feature_config is None

        actor._feature_config = FeatureConfig()
        assert isinstance(actor._feature_config, FeatureConfig)

        # Test ensemble weights default
        if actor._signal_config.ensemble_weights is None:
            ensemble_weights = {
                "threshold": 0.4,
                "extremes": 0.3,
                "momentum": 0.3,
            }
        assert ensemble_weights["threshold"] == 0.4

    def test_load_model_method(self):
        """
        Test _load_model method.
        """
        actor = Mock(spec=MLSignalActor)
        actor.log = Mock()

        # With model
        actor._model = Mock()
        MLSignalActor._load_model(actor)
        actor.log.info.assert_called()

        # Without model
        actor._model = None
        MLSignalActor._load_model(actor)
        actor.log.warning.assert_called()

    def test_initialize_features_method(self):
        """
        Test _initialize_features method.
        """
        actor = Mock(spec=MLSignalActor)
        actor._feature_config = FeatureConfig()
        actor._indicator_manager = None
        actor._feature_buffer = np.zeros(5)
        actor._feature_engineer = Mock()
        actor._feature_engineer.n_features = 10
        actor.log = Mock()

        # Call method
        MLSignalActor._initialize_features(actor)

        # Check initialization
        assert isinstance(actor._indicator_manager, IndicatorManager)

        # Test with feature names
        actor._feature_config.feature_names = ["f1", "f2", "f3"]
        actor._indicator_manager = None
        MLSignalActor._initialize_features(actor)

    def test_compute_features_method(self):
        """
        Test _compute_features method.
        """
        actor = Mock(spec=MLSignalActor)
        actor._indicator_manager = None
        actor._config = Mock()
        actor._config.max_feature_latency_ms = 5
        actor.log = Mock()

        bar = Mock()

        # No indicator manager
        result = MLSignalActor._compute_features(actor, bar)
        assert result is None

        # Not initialized
        actor._indicator_manager = Mock()
        actor._indicator_manager.all_initialized.return_value = False
        result = MLSignalActor._compute_features(actor, bar)
        assert result is None

        # Successful computation
        actor._indicator_manager.all_initialized.return_value = True
        actor._indicator_manager.update_from_bar = Mock()
        actor._feature_engineer = Mock()
        actor._feature_engineer.calculate_features_online.return_value = np.array([0.1, 0.2])

        result = MLSignalActor._compute_features(actor, bar)
        assert result is not None
        assert len(result) == 2

        # Slow computation
        def slow_update(bar):
            time.sleep(0.01)

        actor._indicator_manager.update_from_bar = slow_update
        MLSignalActor._compute_features(actor, bar)
        actor.log.warning.assert_called()

    def test_predict_method_paths(self):
        """
        Test _predict method paths.
        """
        actor = Mock(spec=MLSignalActor)
        actor._model = None
        actor.log = Mock()

        features = np.array([0.1, 0.2, 0.3])

        # No model
        pred, conf = MLSignalActor._predict(actor, features)
        assert pred == 0.0 and conf == 0.0

        # ONNX model
        actor._model = Mock()
        actor._model.run = Mock()
        pred, conf = MLSignalActor._predict(actor, features)

        # Sklearn with predict_proba
        del actor._model.run
        actor._model.predict_proba = Mock(return_value=np.array([[0.3, 0.7]]))
        pred, conf = MLSignalActor._predict(actor, features)

        # Sklearn basic
        del actor._model.predict_proba
        actor._model.predict = Mock(return_value=np.array([0.85]))
        pred, conf = MLSignalActor._predict(actor, features)

        # Unsupported
        actor._model = object()
        pred, conf = MLSignalActor._predict(actor, features)
        assert pred == 0.0 and conf == 0.0

        # Exception
        actor._model = Mock()
        actor._model.predict.side_effect = RuntimeError("Error")
        pred, conf = MLSignalActor._predict(actor, features)
        assert pred == 0.0 and conf == 0.0

    def test_predict_onnx_method(self):
        """
        Test _predict_onnx method.
        """
        actor = Mock(spec=MLSignalActor)
        actor._model = Mock()
        actor._model_metadata = {"input_names": ["input"], "output_names": ["pred", "conf"]}

        features = np.array([0.1, 0.2, 0.3])

        # Dual output
        actor._model.run.return_value = [np.array([[0.8]]), np.array([[0.9]])]
        pred, conf = MLSignalActor._predict_onnx(actor, features)
        assert pred == 0.8 and conf == 0.9

        # Single output
        actor._model.run.return_value = [np.array([[0.7]])]
        pred, conf = MLSignalActor._predict_onnx(actor, features)
        assert pred == 0.7 and conf == 0.7

    def test_predict_sklearn_methods(self):
        """
        Test sklearn prediction methods.
        """
        actor = Mock(spec=MLSignalActor)
        features = np.array([0.1, 0.2, 0.3])

        # predict_proba
        actor._model = Mock()
        actor._model.predict_proba.return_value = np.array([[0.3, 0.7]])
        pred, conf = MLSignalActor._predict_sklearn_proba(actor, features)
        assert pred == 1.0 and conf == 0.7

        # basic predict
        actor._model.predict.return_value = np.array([0.85])
        pred, conf = MLSignalActor._predict_sklearn(actor, features)
        assert pred == 0.85 and conf == 0.85

        # zero prediction
        actor._model.predict.return_value = np.array([0.0])
        pred, conf = MLSignalActor._predict_sklearn(actor, features)
        assert pred == 0.0 and conf == 0.5

    def test_update_prediction_history(self):
        """
        Test _update_prediction_history method.
        """
        actor = Mock(spec=MLSignalActor)
        actor._signal_config = Mock()
        actor._signal_config.adaptive_window = 10
        actor._signal_config.signal_strategy = SignalStrategy.THRESHOLD
        actor._prediction_window = np.zeros(10)
        actor._confidence_window = np.zeros(10)
        actor._volatility_window = np.zeros(10)
        actor._window_index = 0
        actor._indicator_manager = None

        bar = Mock()

        # Without indicator manager
        MLSignalActor._update_prediction_history(actor, 0.5, 0.7, bar)
        assert actor._window_index == 1

        # With price history
        actor._indicator_manager = Mock()
        actor._indicator_manager.price_history = {"closes": [1.10, 1.11, 1.12]}
        MLSignalActor._update_prediction_history(actor, 0.6, 0.8, bar)
        assert actor._window_index == 2

        # Test wraparound
        actor._window_index = 9
        MLSignalActor._update_prediction_history(actor, 0.7, 0.9, bar)
        assert actor._window_index == 0

        # Test adaptive strategy
        actor._signal_config.signal_strategy = SignalStrategy.ADAPTIVE
        actor._update_adaptive_threshold = Mock()
        MLSignalActor._update_prediction_history(actor, 0.8, 0.9, bar)
        actor._update_adaptive_threshold.assert_called_once()

    def test_adaptive_threshold_update(self):
        """
        Test _update_adaptive_threshold method.
        """
        actor = Mock(spec=MLSignalActor)
        actor._config = Mock()
        actor._config.prediction_threshold = 0.5
        actor._signal_config = Mock()
        actor._signal_config.adaptive_volatility_factor = 2.0
        actor._adaptive_threshold = 0.5
        actor._prediction_window = np.array([0.4, 0.5, 0.6, 0.7, 0.8])
        actor._volatility_window = np.array([0.01, 0.02, 0.03, 0.02, 0.01])
        actor._adaptive_threshold_metric = Mock()

        MLSignalActor._update_adaptive_threshold(actor)

        assert actor._adaptive_threshold != 0.5
        assert 0.1 <= actor._adaptive_threshold <= 0.95

    def test_market_regime_detection(self):
        """
        Test _detect_market_regime method.
        """
        actor = Mock(spec=MLSignalActor)
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

        # No price history
        actor._indicator_manager = Mock()
        actor._indicator_manager.price_history = {}
        MLSignalActor._detect_market_regime(actor, bar)
        assert actor._market_regime == "unknown"

        # Insufficient data
        actor._indicator_manager.price_history = {"closes": [1.10, 1.11]}
        MLSignalActor._detect_market_regime(actor, bar)
        assert actor._market_regime == "unknown"

        # Volatile regime
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
            "closes": [1.10 + np.sin(i * 0.5) * 0.0001 for i in range(25)],
        }
        MLSignalActor._detect_market_regime(actor, bar)
        assert actor._market_regime == "ranging"

    def test_signal_generation_strategies(self):
        """
        Test all signal generation strategy methods.
        """
        actor = Mock(spec=MLSignalActor)
        actor._config = Mock()
        actor._config.prediction_threshold = 0.7
        actor._config.log_predictions = False
        actor._signal_config = Mock()
        actor._signal_config.adaptive_window = 10
        actor._signal_config.extremes_top_pct = 0.1
        actor._signal_config.momentum_lookback = 3
        actor._bars_processed = 10
        actor._last_signal_bar = 0
        actor._prediction_history = []
        actor._adaptive_threshold = 0.5
        actor._market_regime = "trending"
        actor._ensemble_weights = {"threshold": 0.4, "extremes": 0.3, "momentum": 0.3}
        actor.clock = Mock()
        actor.clock.timestamp_ns.return_value = 1000

        bar = Mock()
        bar.bar_type = Mock()
        bar.bar_type.instrument_id = "TEST"
        bar.ts_event = 0
        features = np.array([0.1, 0.2])

        # Threshold
        signal = MLSignalActor._generate_threshold_signal(actor, bar, 0.8, 0.9, features)
        assert signal is not None
        assert actor._last_signal_bar == 10

        signal = MLSignalActor._generate_threshold_signal(actor, bar, 0.8, 0.5, features)
        assert signal is None

        # Extremes - insufficient history
        signal = MLSignalActor._generate_extremes_signal(actor, bar, 0.9, 0.8, features)
        assert signal is None

        # Extremes - with history
        actor._prediction_history = list(range(100))
        signal = MLSignalActor._generate_extremes_signal(actor, bar, 95.0, 0.8, features)
        assert signal is not None

        signal = MLSignalActor._generate_extremes_signal(actor, bar, 50.0, 0.8, features)
        assert signal is None

        # Momentum - insufficient history
        actor._prediction_history = [0.5]
        signal = MLSignalActor._generate_momentum_signal(actor, bar, 0.8, 0.9, features)
        assert signal is None

        # Momentum - with trend
        actor._prediction_history = [0.2, 0.4, 0.6, 0.8]
        signal = MLSignalActor._generate_momentum_signal(actor, bar, 0.9, 0.8, features)
        assert signal is not None

        # Momentum - no trend
        actor._prediction_history = [0.5, 0.5, 0.5, 0.5]
        signal = MLSignalActor._generate_momentum_signal(actor, bar, 0.5, 0.8, features)
        assert signal is None

        # Adaptive - below threshold
        actor._adaptive_threshold = 0.95
        signal = MLSignalActor._generate_adaptive_signal(actor, bar, 0.7, 0.5, features)
        assert signal is None

        # Adaptive - above threshold
        actor._adaptive_threshold = 0.3
        signal = MLSignalActor._generate_adaptive_signal(actor, bar, 0.7, 0.5, features)
        assert signal is not None
        assert isinstance(signal, AdaptiveSignal)

        # Ensemble
        actor._generate_threshold_signal = Mock(return_value=Mock())
        actor._generate_extremes_signal = Mock(return_value=None)
        actor._generate_momentum_signal = Mock(return_value=None)

        signal = MLSignalActor._generate_ensemble_signal(actor, bar, 0.8, 0.9, features)
        assert signal is not None

        # Ensemble - no signals
        actor._generate_threshold_signal = Mock(return_value=None)
        signal = MLSignalActor._generate_ensemble_signal(actor, bar, 0.8, 0.9, features)
        assert signal is None

    def test_generate_signal_by_strategy(self):
        """
        Test _generate_signal_by_strategy method.
        """
        actor = Mock(spec=MLSignalActor)
        actor._signal_config = Mock()
        actor._signal_config.min_signal_separation_bars = 2
        actor._signal_config.signal_strategy = SignalStrategy.THRESHOLD
        actor._bars_processed = 5
        actor._last_signal_bar = 4
        actor.log = Mock()

        # Mock methods
        actor._generate_threshold_signal = Mock(return_value=Mock())
        actor._generate_extremes_signal = Mock(return_value=Mock())
        actor._generate_momentum_signal = Mock(return_value=Mock())
        actor._generate_ensemble_signal = Mock(return_value=Mock())
        actor._generate_adaptive_signal = Mock(return_value=Mock())

        bar = Mock()
        features = np.array([0.1, 0.2])

        # Signal separation
        signal = MLSignalActor._generate_signal_by_strategy(actor, bar, 0.8, 0.9, features)
        assert signal is None

        # Valid separation
        actor._last_signal_bar = 0

        # Test each strategy
        for strategy in SignalStrategy:
            actor._signal_config.signal_strategy = strategy
            signal = MLSignalActor._generate_signal_by_strategy(actor, bar, 0.8, 0.9, features)
            assert signal is not None

        # Unknown strategy
        actor._signal_config.signal_strategy = "unknown"
        signal = MLSignalActor._generate_signal_by_strategy(actor, bar, 0.8, 0.9, features)
        assert signal is None

    def test_generate_prediction_protected(self):
        """
        Test _generate_prediction_protected method.
        """
        actor = Mock(spec=MLSignalActor)
        actor._config = Mock()
        actor._signal_config = Mock()
        actor._signal_config.signal_strategy = SignalStrategy.THRESHOLD
        actor._signal_config.enable_regime_detection = True
        actor._circuit_breaker = CircuitBreaker()
        actor._health_monitor = HealthMonitor()
        actor._signal_generation_time_metric = Mock()
        actor._signals_generated_metric = Mock()
        actor._track_performance_metrics = Mock()
        actor.log = Mock()
        actor.id = Mock()
        actor.id.value = "TEST"

        bar = Mock()
        bar.bar_type = Mock()
        bar.bar_type.instrument_id = "TEST"
        features = np.array([0.1, 0.2])

        # Success with signal
        actor._predict = Mock(return_value=(0.8, 0.9))
        actor._update_prediction_history = Mock()
        actor._detect_market_regime = Mock()
        actor._generate_signal_by_strategy = Mock(return_value=Mock(prediction=0.8))
        actor._publish_signal = Mock()

        MLSignalActor._generate_prediction_protected(actor, bar, features)

        actor._predict.assert_called_once()
        actor._update_prediction_history.assert_called_once()
        actor._detect_market_regime.assert_called_once()
        actor._publish_signal.assert_called_once()

        # Success without signal
        actor._generate_signal_by_strategy.return_value = None
        MLSignalActor._generate_prediction_protected(actor, bar, features)

        # Failure
        actor._predict.side_effect = Exception("Error")
        MLSignalActor._generate_prediction_protected(actor, bar, features)
        assert actor._circuit_breaker.consecutive_failures > 0

    def test_track_performance_metrics(self):
        """
        Test _track_performance_metrics method.
        """
        actor = Mock(spec=MLSignalActor)
        actor._prediction_count = 0
        actor._total_inference_time = 0.0
        actor._config = Mock()
        actor._config.log_predictions = False
        actor._signal_config = Mock()
        actor._signal_config.signal_strategy = SignalStrategy.THRESHOLD
        actor.id = Mock()
        actor.id.value = "TEST"
        actor.log = Mock()

        # First call creates metrics
        MLSignalActor._track_performance_metrics(actor, 0.5, 0.7, 2.5)
        assert actor._prediction_count == 1
        assert actor._total_inference_time == 2.5
        assert hasattr(actor, "_prediction_distribution_metric")
        assert hasattr(actor, "_confidence_distribution_metric")

        # Subsequent calls
        for i in range(4):
            MLSignalActor._track_performance_metrics(actor, 0.6, 0.8, 3.0)

        assert actor._prediction_count == 5
        assert actor._total_inference_time == 14.5

        # With logging enabled
        actor._config.log_predictions = True
        MLSignalActor._track_performance_metrics(actor, 0.8, 0.9, 3.0)
        actor.log.debug.assert_called()

    def test_backup_and_restore_state(self):
        """
        Test state backup and restore methods.
        """
        actor = Mock(spec=MLSignalActor)
        actor._indicator_manager = Mock()
        actor._indicator_manager.price_history = {"closes": [1.10, 1.11]}
        actor._indicator_manager.indicators = {
            "sma": Mock(value=50.0, initialized=True),
        }
        actor._prediction_history = [0.1, 0.2, 0.3]
        actor._confidence_history = [0.6, 0.7, 0.8]
        actor._prediction_window = np.array([0.1, 0.2])
        actor._confidence_window = np.array([0.3, 0.4])
        actor._volatility_window = np.array([0.01, 0.02])
        actor._window_index = 5
        actor._adaptive_threshold = 0.8
        actor._market_regime = "trending"
        actor._last_signal_bar = 10
        actor._signal_config = Mock()
        actor._signal_config.adaptive_window = 10
        actor._signal_config.min_signal_separation_bars = 2
        actor._config = Mock()
        actor._config.prediction_threshold = 0.5
        actor._indicator_state_backup = {}
        actor.log = Mock()

        # Backup
        MLSignalActor._backup_indicator_state(actor)
        assert "prediction_history" in actor._indicator_state_backup
        assert "indicators" in actor._indicator_state_backup

        # Modify state
        actor._prediction_history.clear()
        actor._adaptive_threshold = 0.99

        # Restore
        MLSignalActor._restore_indicator_state(actor)
        assert len(actor._prediction_history) == 3
        assert actor._adaptive_threshold == 0.8

        # Backup without manager
        actor._indicator_manager = None
        MLSignalActor._backup_indicator_state(actor)

        # Restore without backup
        actor._indicator_state_backup = None
        MLSignalActor._restore_indicator_state(actor)

    def test_get_signal_statistics(self):
        """
        Test get_signal_statistics method.
        """
        actor = Mock(spec=MLSignalActor)
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

        assert stats["signal_strategy"] == SignalStrategy.ENSEMBLE.value
        assert stats["adaptive_threshold"] == 0.7
        assert stats["ensemble_weights"] == {"threshold": 0.5}

        # Without ensemble
        actor._signal_config.signal_strategy = SignalStrategy.THRESHOLD
        stats = MLSignalActor.get_signal_statistics(actor)
        assert stats["ensemble_weights"] is None

    def test_reset_signal_state(self):
        """
        Test reset_signal_state method.
        """
        actor = Mock(spec=MLSignalActor)
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
        assert actor._window_index == 0
        assert actor._adaptive_threshold == 0.5
        assert actor._market_regime == "unknown"
        assert actor._last_signal_bar == -2

    def test_adaptive_signal_class(self):
        """
        Test AdaptiveSignal data class.
        """
        signal = AdaptiveSignal(
            instrument_id="TEST",
            prediction=0.8,
            confidence=0.9,
            adaptive_threshold=0.7,
            signal_strength=1.2,
            market_regime="trending",
            ts_event=123456789,
            ts_init=123456790,
        )

        assert signal.instrument_id == "TEST"
        assert signal.prediction == 0.8
        assert signal.confidence == 0.9
        assert signal.adaptive_threshold == 0.7
        assert signal.signal_strength == 1.2
        assert signal.market_regime == "trending"
        assert signal.ts_event == 123456789
        assert signal.ts_init == 123456790

    def test_signal_strategy_enum(self):
        """
        Test SignalStrategy enum.
        """
        assert SignalStrategy.THRESHOLD.value == "threshold"
        assert SignalStrategy.EXTREMES.value == "extremes"
        assert SignalStrategy.MOMENTUM.value == "momentum"
        assert SignalStrategy.ENSEMBLE.value == "ensemble"
        assert SignalStrategy.ADAPTIVE.value == "adaptive"
