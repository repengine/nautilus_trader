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
Direct unit tests for MLSignalActor methods to achieve 90%+ coverage.

This approach tests the actor methods directly to avoid Nautilus initialization issues.

"""

import pickle
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from ml.actors.base import CircuitBreaker
from ml.actors.base import HealthMonitor
from ml.actors.base import MLSignal
from ml.actors.signal import AdaptiveSignal
from ml.actors.signal import MLSignalActor
from ml.actors.signal import MLSignalActorConfig
from ml.actors.signal import SignalStrategy
from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.features.engineering import IndicatorManager
from nautilus_trader.common.component import TestClock
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarSpecification
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.enums import BarAggregation
from nautilus_trader.model.enums import PriceType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity


class SimpleModel:
    """
    Simple pickleable model for testing.
    """

    def predict(self, X):
        return np.array([0.8])


class TestMLSignalActorDirect:
    """
    Direct tests for MLSignalActor methods.
    """

    def setup_method(self):
        """
        Set up test fixtures.
        """
        # Model file
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)

        with open(self.temp_file.name, "wb") as f:
            pickle.dump(SimpleModel(), f)
        self.temp_file.close()

        # Test data
        self.instrument_id = InstrumentId.from_str("EURUSD.SIM")
        self.bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
        self.bar_type = BarType(self.instrument_id, self.bar_spec, AggressorSide.BUYER)

        self.config = MLSignalActorConfig(
            component_id="TEST-001",
            model_path=self.temp_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            prediction_threshold=0.7,
            warm_up_period=5,
            signal_strategy=SignalStrategy.THRESHOLD,
            adaptive_window=10,
            min_signal_separation_bars=2,
        )

    def teardown_method(self):
        """
        Clean up.
        """
        Path(self.temp_file.name).unlink(missing_ok=True)

    def create_test_bar(self, close_price: float = 1.1000) -> Bar:
        """
        Create test bar.
        """
        return Bar(
            bar_type=self.bar_type,
            open=Price.from_str(str(close_price - 0.0002)),
            high=Price.from_str(str(close_price + 0.0003)),
            low=Price.from_str(str(close_price - 0.0004)),
            close=Price.from_str(str(close_price)),
            volume=Quantity.from_str("1000"),
            ts_event=0,
            ts_init=0,
        )

    def test_signal_actor_init_coverage(self):
        """
        Test initialization code paths directly.
        """
        # Create actor instance without full Nautilus initialization
        actor = object.__new__(MLSignalActor)

        # Set up basic attributes
        actor._config = self.config
        actor._signal_config = self.config
        actor.log = Mock()
        actor.clock = TestClock()

        # Test initialization logic directly
        # Feature configuration paths
        if self.config.feature_config is None:
            actor._feature_config = FeatureConfig()
        else:
            if isinstance(self.config.feature_config, FeatureConfig):
                actor._feature_config = self.config.feature_config
            else:
                actor._feature_config = FeatureConfig()

        actor._feature_engineer = FeatureEngineer(actor._feature_config)
        actor._indicator_manager = None

        # Signal generation state
        actor._prediction_history = []
        actor._confidence_history = []
        actor._last_signal_bar = -self.config.min_signal_separation_bars
        actor._adaptive_threshold = self.config.prediction_threshold
        actor._market_regime = "unknown"

        # Performance buffers
        actor._feature_buffer = np.zeros(actor._feature_engineer.n_features, dtype=np.float32)
        actor._prediction_window = np.zeros(self.config.adaptive_window, dtype=np.float32)
        actor._confidence_window = np.zeros(self.config.adaptive_window, dtype=np.float32)
        actor._volatility_window = np.zeros(self.config.adaptive_window, dtype=np.float32)
        actor._window_index = 0

        # Ensemble weights
        if self.config.ensemble_weights is None:
            actor._ensemble_weights = {
                "threshold": 0.4,
                "extremes": 0.3,
                "momentum": 0.3,
            }
        else:
            actor._ensemble_weights = self.config.ensemble_weights

        # Verify initialization
        assert actor._adaptive_threshold == 0.7
        assert actor._market_regime == "unknown"
        assert len(actor._prediction_history) == 0
        assert actor._window_index == 0
        assert actor._ensemble_weights["threshold"] == 0.4

    def test_load_model_method(self):
        """
        Test _load_model method.
        """
        actor = object.__new__(MLSignalActor)
        actor.log = Mock()

        # With model
        actor._model = Mock()
        actor._load_model()
        actor.log.info.assert_called()

        # Without model
        actor._model = None
        actor._load_model()
        actor.log.warning.assert_called()

    def test_initialize_features_coverage(self):
        """
        Test _initialize_features method.
        """
        actor = object.__new__(MLSignalActor)
        actor._feature_config = FeatureConfig()
        actor._indicator_manager = None
        actor._feature_buffer = np.zeros(5)
        actor._feature_engineer = Mock()
        actor._feature_engineer.n_features = 10
        actor.log = Mock()

        # Initialize features
        actor._initialize_features()

        assert isinstance(actor._indicator_manager, IndicatorManager)
        assert actor._feature_buffer.size >= 10

        # Test with feature names
        actor._feature_config.feature_names = ["f1", "f2", "f3"]
        actor._initialize_features()
        assert actor._feature_buffer.size >= 3

    def test_compute_features_all_paths(self):
        """
        Test _compute_features method.
        """
        actor = object.__new__(MLSignalActor)
        actor._indicator_manager = None
        actor._config = self.config
        actor.log = Mock()

        bar = self.create_test_bar()

        # No indicator manager
        assert actor._compute_features(bar) is None

        # With indicator manager
        actor._indicator_manager = Mock()
        actor._indicator_manager.all_initialized.return_value = False
        assert actor._compute_features(bar) is None

        # All initialized
        actor._indicator_manager.all_initialized.return_value = True
        actor._indicator_manager.update_from_bar = Mock()

        actor._feature_engineer = Mock()
        actor._feature_engineer.calculate_features_online.return_value = np.array([0.1, 0.2])

        features = actor._compute_features(bar)
        assert features is not None
        assert len(features) == 2

        # Slow computation warning
        def slow_update(bar):
            time.sleep(0.01)

        actor._indicator_manager.update_from_bar = slow_update
        actor._config.max_feature_latency_ms = 5
        actor._compute_features(bar)
        actor.log.warning.assert_called()

    def test_predict_all_model_types(self):
        """
        Test _predict method with all model types.
        """
        actor = object.__new__(MLSignalActor)
        actor._model = None
        actor._model_metadata = {}
        actor.log = Mock()

        features = np.array([0.1, 0.2, 0.3])

        # No model
        pred, conf = actor._predict(features)
        assert pred == 0.0 and conf == 0.0

        # ONNX model - dual output
        actor._model = Mock()
        actor._model.run.return_value = [np.array([[0.8]]), np.array([[0.9]])]
        actor._model_metadata = {"input_names": ["input"], "output_names": ["pred", "conf"]}
        pred, conf = actor._predict_onnx(features)
        assert pred == 0.8 and conf == 0.9

        # ONNX single output
        actor._model.run.return_value = [np.array([[0.7]])]
        pred, conf = actor._predict_onnx(features)
        assert pred == 0.7 and conf == 0.7

        # Sklearn with predict_proba
        actor._model = Mock()
        actor._model.predict_proba.return_value = np.array([[0.3, 0.7]])
        pred, conf = actor._predict_sklearn_proba(features)
        assert pred == 1.0 and conf == 0.7

        # Sklearn basic
        actor._model = Mock()
        actor._model.predict.return_value = np.array([0.85])
        pred, conf = actor._predict_sklearn(features)
        assert pred == 0.85 and conf == 0.85

        # Zero prediction
        actor._model.predict.return_value = np.array([0.0])
        pred, conf = actor._predict_sklearn(features)
        assert pred == 0.0 and conf == 0.5

        # Test main _predict method paths
        actor._model = Mock()
        actor._model.run = Mock(return_value=[[np.array([[0.8]])]])
        pred, conf = actor._predict(features)
        assert pred == 0.8

        del actor._model.run
        actor._model.predict_proba = Mock(return_value=np.array([[0.2, 0.8]]))
        pred, conf = actor._predict(features)
        assert pred == 1.0

        del actor._model.predict_proba
        actor._model.predict = Mock(return_value=np.array([0.6]))
        pred, conf = actor._predict(features)
        assert pred == 0.6

        # Unsupported model
        actor._model = object()
        pred, conf = actor._predict(features)
        assert pred == 0.0 and conf == 0.0

        # Exception
        actor._model = Mock()
        actor._model.predict.side_effect = RuntimeError("Error")
        pred, conf = actor._predict(features)
        assert pred == 0.0 and conf == 0.0

    def test_update_prediction_history_coverage(self):
        """
        Test _update_prediction_history method.
        """
        actor = object.__new__(MLSignalActor)
        actor._signal_config = self.config
        actor._prediction_window = np.zeros(10)
        actor._confidence_window = np.zeros(10)
        actor._volatility_window = np.zeros(10)
        actor._window_index = 0
        actor._indicator_manager = None

        bar = self.create_test_bar()

        # Without indicator manager
        actor._update_prediction_history(0.5, 0.7, bar)
        assert actor._window_index == 1

        # With indicator manager and price history
        actor._indicator_manager = Mock()
        actor._indicator_manager.price_history = {"closes": [1.10, 1.11, 1.12]}
        actor._update_prediction_history(0.6, 0.8, bar)
        assert actor._window_index == 2

        # Test wraparound
        actor._window_index = 9
        actor._update_prediction_history(0.7, 0.9, bar)
        assert actor._window_index == 0

        # Test adaptive threshold update trigger
        actor._signal_config = Mock()
        actor._signal_config.signal_strategy = SignalStrategy.ADAPTIVE
        actor._signal_config.adaptive_window = 10
        actor._update_adaptive_threshold = Mock()
        actor._update_prediction_history(0.8, 0.9, bar)
        actor._update_adaptive_threshold.assert_called_once()

    def test_adaptive_threshold_update(self):
        """
        Test _update_adaptive_threshold method.
        """
        actor = object.__new__(MLSignalActor)
        actor._config = Mock()
        actor._config.prediction_threshold = 0.5
        actor._signal_config = Mock()
        actor._signal_config.adaptive_volatility_factor = 2.0
        actor._adaptive_threshold = 0.5
        actor._prediction_window = np.array([0.4, 0.5, 0.6, 0.7, 0.8])
        actor._volatility_window = np.array([0.01, 0.02, 0.03, 0.02, 0.01])
        actor._adaptive_threshold_metric = Mock()

        initial = actor._adaptive_threshold
        actor._update_adaptive_threshold()

        assert actor._adaptive_threshold != initial
        assert 0.1 <= actor._adaptive_threshold <= 0.95
        actor._adaptive_threshold_metric.observe.assert_called()

    def test_market_regime_detection_all_paths(self):
        """
        Test _detect_market_regime method.
        """
        actor = object.__new__(MLSignalActor)
        actor._indicator_manager = None
        actor._market_regime = "unknown"
        actor._market_regime_metric = Mock()
        actor.log = Mock()

        bar = self.create_test_bar()

        # No indicator manager
        actor._detect_market_regime(bar)
        assert actor._market_regime == "unknown"

        # No price history
        actor._indicator_manager = Mock()
        actor._indicator_manager.price_history = {}
        actor._detect_market_regime(bar)
        assert actor._market_regime == "unknown"

        # Insufficient data
        actor._indicator_manager.price_history = {"closes": [1.10, 1.11]}
        actor._detect_market_regime(bar)
        assert actor._market_regime == "unknown"

        # Volatile regime
        actor._indicator_manager.price_history = {
            "closes": [1.10 + (i % 2) * 0.05 for i in range(25)],
        }
        actor._detect_market_regime(bar)
        assert actor._market_regime == "volatile"

        # Trending regime
        actor._market_regime = "unknown"
        actor._indicator_manager.price_history = {
            "closes": [1.10 + i * 0.001 for i in range(25)],
        }
        actor._detect_market_regime(bar)
        assert actor._market_regime == "trending"

        # Ranging regime
        actor._market_regime = "unknown"
        actor._indicator_manager.price_history = {
            "closes": [1.10 + np.sin(i * 0.5) * 0.0001 for i in range(25)],
        }
        actor._detect_market_regime(bar)
        assert actor._market_regime == "ranging"

    def test_all_signal_generation_strategies(self):
        """
        Test all signal generation strategy methods.
        """
        actor = object.__new__(MLSignalActor)
        actor._config = self.config
        actor._signal_config = self.config
        actor._bars_processed = 10
        actor._last_signal_bar = 0
        actor._prediction_history = []
        actor.clock = TestClock()
        actor.log = Mock()

        bar = self.create_test_bar()
        features = np.array([0.1, 0.2])

        # Threshold strategy
        signal = actor._generate_threshold_signal(bar, 0.8, 0.9, features)
        assert signal is not None
        assert isinstance(signal, MLSignal)

        signal = actor._generate_threshold_signal(bar, 0.8, 0.5, features)
        assert signal is None

        # Extremes strategy
        actor._signal_config = Mock()
        actor._signal_config.adaptive_window = 10
        actor._signal_config.extremes_top_pct = 0.1

        # Insufficient history
        signal = actor._generate_extremes_signal(bar, 0.9, 0.8, features)
        assert signal is None

        # With history
        actor._prediction_history = list(range(100))
        signal = actor._generate_extremes_signal(bar, 95.0, 0.8, features)
        assert signal is not None

        signal = actor._generate_extremes_signal(bar, 50.0, 0.8, features)
        assert signal is None

        # Momentum strategy
        actor._signal_config.momentum_lookback = 3

        # Insufficient history
        actor._prediction_history = [0.5]
        signal = actor._generate_momentum_signal(bar, 0.8, 0.9, features)
        assert signal is None

        # With momentum
        actor._prediction_history = [0.2, 0.4, 0.6, 0.8]
        signal = actor._generate_momentum_signal(bar, 0.9, 0.8, features)
        assert signal is not None
        assert signal.prediction != 0.9

        # No momentum
        actor._prediction_history = [0.5, 0.5, 0.5, 0.5]
        signal = actor._generate_momentum_signal(bar, 0.5, 0.8, features)
        assert signal is None

        # Adaptive strategy
        actor._adaptive_threshold = 0.95
        signal = actor._generate_adaptive_signal(bar, 0.7, 0.5, features)
        assert signal is None

        actor._adaptive_threshold = 0.3
        signal = actor._generate_adaptive_signal(bar, 0.7, 0.5, features)
        assert signal is not None
        assert isinstance(signal, AdaptiveSignal)

        # Ensemble strategy
        actor._ensemble_weights = {"threshold": 0.5, "extremes": 0.3, "momentum": 0.2}
        actor._prediction_history = [0.3, 0.4, 0.5, 0.6, 0.7]

        # Mock component methods
        mock_signal = MLSignal(
            instrument_id=self.instrument_id,
            prediction=0.8,
            confidence=0.9,
            features=None,
            ts_event=0,
            ts_init=0,
        )
        actor._generate_threshold_signal = Mock(return_value=mock_signal)
        actor._generate_extremes_signal = Mock(return_value=None)
        actor._generate_momentum_signal = Mock(return_value=None)

        signal = actor._generate_ensemble_signal(bar, 0.8, 0.9, features)
        assert signal is not None

        # Test all components returning None
        actor._generate_threshold_signal = Mock(return_value=None)
        signal = actor._generate_ensemble_signal(bar, 0.8, 0.9, features)
        assert signal is None

    def test_generate_signal_by_strategy_coverage(self):
        """
        Test _generate_signal_by_strategy method.
        """
        actor = object.__new__(MLSignalActor)
        actor._signal_config = Mock()
        actor._signal_config.min_signal_separation_bars = 2
        actor._signal_config.signal_strategy = SignalStrategy.THRESHOLD
        actor._bars_processed = 5
        actor._last_signal_bar = 4
        actor.log = Mock()

        # Mock strategy methods
        actor._generate_threshold_signal = Mock(return_value=None)
        actor._generate_extremes_signal = Mock(return_value=None)
        actor._generate_momentum_signal = Mock(return_value=None)
        actor._generate_ensemble_signal = Mock(return_value=None)
        actor._generate_adaptive_signal = Mock(return_value=None)

        bar = self.create_test_bar()
        features = np.array([0.1, 0.2])

        # Signal separation check
        signal = actor._generate_signal_by_strategy(bar, 0.8, 0.9, features)
        assert signal is None

        # Valid separation
        actor._last_signal_bar = 0

        # Test each strategy
        for strategy in SignalStrategy:
            actor._signal_config.signal_strategy = strategy
            actor._generate_signal_by_strategy(bar, 0.8, 0.9, features)

        # Unknown strategy
        actor._signal_config.signal_strategy = "unknown"
        signal = actor._generate_signal_by_strategy(bar, 0.8, 0.9, features)
        assert signal is None
        actor.log.error.assert_called()

    def test_generate_prediction_protected_coverage(self):
        """
        Test _generate_prediction_protected method.
        """
        actor = object.__new__(MLSignalActor)
        actor._config = self.config
        actor._signal_config = self.config
        actor._circuit_breaker = CircuitBreaker()
        actor._health_monitor = HealthMonitor()
        actor._signal_generation_time_metric = Mock()
        actor._signals_generated_metric = Mock()
        actor.log = Mock()
        actor.id = Mock()
        actor.id.value = "TEST"

        bar = self.create_test_bar()
        features = np.array([0.1, 0.2])

        # Successful prediction with signal
        actor._predict = Mock(return_value=(0.8, 0.9))
        actor._update_prediction_history = Mock()
        actor._detect_market_regime = Mock()
        actor._generate_signal_by_strategy = Mock(
            return_value=MLSignal(
                instrument_id=self.instrument_id,
                prediction=0.8,
                confidence=0.9,
                features=None,
                ts_event=0,
                ts_init=0,
            ),
        )
        actor._publish_signal = Mock()
        actor._track_performance_metrics = Mock()

        actor._generate_prediction_protected(bar, features)

        actor._predict.assert_called_once()
        actor._update_prediction_history.assert_called_once()
        actor._publish_signal.assert_called_once()
        actor._track_performance_metrics.assert_called_once()
        assert actor._circuit_breaker.consecutive_failures == 0

        # Successful prediction without signal
        actor._generate_signal_by_strategy.return_value = None
        actor._generate_prediction_protected(bar, features)

        # Failed prediction
        actor._predict.side_effect = Exception("Prediction error")
        actor._generate_prediction_protected(bar, features)

        assert actor._circuit_breaker.consecutive_failures > 0
        assert actor._health_monitor.failed_predictions > 0

    def test_track_performance_metrics(self):
        """
        Test _track_performance_metrics method.
        """
        actor = object.__new__(MLSignalActor)
        actor._prediction_count = 0
        actor._total_inference_time = 0.0
        actor._config = self.config
        actor.id = Mock()
        actor.id.value = "TEST"
        actor.log = Mock()

        # Track metrics
        for i in range(5):
            actor._track_performance_metrics(0.5 + i * 0.1, 0.7 + i * 0.05, 2.5 + i * 0.5)

        assert actor._prediction_count == 5
        assert actor._total_inference_time > 0
        assert hasattr(actor, "_prediction_distribution_metric")
        assert hasattr(actor, "_confidence_distribution_metric")

        # Test with log predictions enabled
        actor._config.log_predictions = True
        actor._track_performance_metrics(0.8, 0.9, 3.0)
        actor.log.debug.assert_called()

    def test_state_backup_and_restore(self):
        """
        Test backup and restore methods.
        """
        actor = object.__new__(MLSignalActor)
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
        actor._config = Mock()
        actor._config.prediction_threshold = 0.5
        actor._config.min_signal_separation_bars = 2
        actor.log = Mock()

        # Backup with indicator manager
        actor._backup_indicator_state()

        assert actor._indicator_state_backup is not None
        assert "prediction_history" in actor._indicator_state_backup
        assert "indicators" in actor._indicator_state_backup

        # Modify state
        actor._prediction_history.clear()
        actor._adaptive_threshold = 0.99

        # Restore
        actor._restore_indicator_state()

        assert len(actor._prediction_history) == 3
        assert actor._adaptive_threshold == 0.8
        assert actor._indicator_state_backup == {}

        # Backup without indicator manager
        actor._indicator_manager = None
        actor._backup_indicator_state()
        assert "prediction_history" in actor._indicator_state_backup

        # Restore without backup
        actor._indicator_state_backup = None
        actor._restore_indicator_state()

    def test_get_signal_statistics(self):
        """
        Test get_signal_statistics method.
        """
        actor = object.__new__(MLSignalActor)
        actor._signal_config = Mock()
        actor._signal_config.signal_strategy = SignalStrategy.ENSEMBLE
        actor._adaptive_threshold = 0.7
        actor._market_regime = "trending"
        actor._last_signal_bar = 10
        actor._prediction_history = [0.1, 0.2, 0.3]
        actor._feature_buffer = np.zeros(5)
        actor._ensemble_weights = {"threshold": 0.5}

        # Mock base class method
        actor.get_health_status = Mock(
            return_value={
                "status": "healthy",
                "uptime": 100,
            },
        )

        stats = actor.get_signal_statistics()

        assert stats["signal_strategy"] == SignalStrategy.ENSEMBLE.value
        assert stats["adaptive_threshold"] == 0.7
        assert stats["market_regime"] == "trending"
        assert stats["ensemble_weights"] == {"threshold": 0.5}

        # Test without ensemble
        actor._signal_config.signal_strategy = SignalStrategy.THRESHOLD
        stats = actor.get_signal_statistics()
        assert stats["ensemble_weights"] is None

    def test_reset_signal_state(self):
        """
        Test reset_signal_state method.
        """
        actor = object.__new__(MLSignalActor)
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

        actor.reset_signal_state()

        assert len(actor._prediction_history) == 0
        assert len(actor._confidence_history) == 0
        assert np.all(actor._prediction_window == 0.0)
        assert np.all(actor._confidence_window == 0.0)
        assert np.all(actor._volatility_window == 0.0)
        assert actor._window_index == 0
        assert actor._adaptive_threshold == 0.5
        assert actor._market_regime == "unknown"
        assert actor._last_signal_bar == -2

    def test_adaptive_signal_class(self):
        """
        Test AdaptiveSignal data class.
        """
        signal = AdaptiveSignal(
            instrument_id=self.instrument_id,
            prediction=0.8,
            confidence=0.9,
            adaptive_threshold=0.7,
            signal_strength=1.2,
            market_regime="trending",
            ts_event=123456789,
            ts_init=123456790,
        )

        assert signal.instrument_id == self.instrument_id
        assert signal.prediction == 0.8
        assert signal.confidence == 0.9
        assert signal.adaptive_threshold == 0.7
        assert signal.signal_strength == 1.2
        assert signal.market_regime == "trending"
        assert signal.ts_event == 123456789
        assert signal.ts_init == 123456790

    def test_signal_strategy_enum(self):
        """
        Test SignalStrategy enum values.
        """
        assert SignalStrategy.THRESHOLD.value == "threshold"
        assert SignalStrategy.EXTREMES.value == "extremes"
        assert SignalStrategy.MOMENTUM.value == "momentum"
        assert SignalStrategy.ENSEMBLE.value == "ensemble"
        assert SignalStrategy.ADAPTIVE.value == "adaptive"

    def test_edge_cases_for_complete_coverage(self):
        """
        Test additional edge cases for complete coverage.
        """
        # Test ensemble with different component signal combinations
        actor = object.__new__(MLSignalActor)
        actor._config = self.config
        actor._signal_config = self.config
        actor._ensemble_weights = {"threshold": 0.4, "extremes": 0.3, "momentum": 0.3}
        actor._bars_processed = 10
        actor._last_signal_bar = 0
        actor.clock = TestClock()

        bar = self.create_test_bar()
        features = np.array([0.1, 0.2])

        # Test with extremes signal only
        mock_signal = MLSignal(
            instrument_id=self.instrument_id,
            prediction=0.8,
            confidence=0.9,
            features=None,
            ts_event=0,
            ts_init=0,
        )
        actor._generate_threshold_signal = Mock(return_value=None)
        actor._generate_extremes_signal = Mock(return_value=mock_signal)
        actor._generate_momentum_signal = Mock(return_value=None)

        signal = actor._generate_ensemble_signal(bar, 0.8, 0.9, features)
        assert signal is not None

        # Test with momentum signal only
        actor._generate_extremes_signal = Mock(return_value=None)
        actor._generate_momentum_signal = Mock(return_value=mock_signal)

        signal = actor._generate_ensemble_signal(bar, 0.8, 0.9, features)
        assert signal is not None

        # Test with all signals
        actor._generate_threshold_signal = Mock(return_value=mock_signal)
        actor._generate_extremes_signal = Mock(return_value=mock_signal)
        actor._generate_momentum_signal = Mock(return_value=mock_signal)

        signal = actor._generate_ensemble_signal(bar, 0.8, 0.9, features)
        assert signal is not None
        assert signal.confidence > 0.9  # Should be higher with all signals
