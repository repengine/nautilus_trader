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
Unit tests for MLSignalActor achieving 90%+ coverage by testing methods directly.
"""

import contextlib
import pickle
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np

from ml.actors.base import CircuitBreaker
from ml.actors.base import HealthMonitor
from ml.actors.base import MLSignal
from ml.actors.signal import AdaptiveSignal
from ml.actors.signal import MLSignalActor
from ml.actors.signal import MLSignalActorConfig
from ml.actors.signal import SignalStrategy
from ml.config.base import CircuitBreakerConfig
from ml.config.base import MLFeatureConfig
from ml.features.engineering import FeatureConfig
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarSpecification
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.enums import BarAggregation
from nautilus_trader.model.enums import PriceType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity


class MockModel:
    """
    Simple model for testing.
    """

    def predict(self, X):
        return np.array([0.8])

    def predict_proba(self, X):
        return np.array([[0.2, 0.8]])


class TestMLSignalActorCoverage:
    """
    Test suite for MLSignalActor achieving 90%+ coverage.
    """

    def setup_method(self):
        """
        Set up test fixtures.
        """
        # Clear metrics
        try:
            import gc

            from prometheus_client import REGISTRY

            for collector in list(REGISTRY._collector_to_names.keys()):
                with contextlib.suppress(Exception):
                    REGISTRY.unregister(collector)
            gc.collect()
        except ImportError:
            pass

        # Create model file
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        with open(self.temp_file.name, "wb") as f:
            pickle.dump(MockModel(), f)
        self.temp_file.close()

        # Setup test data
        self.instrument_id = InstrumentId.from_str("EURUSD.SIM")
        self.bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
        self.bar_type = BarType(self.instrument_id, self.bar_spec, AggressorSide.BUYER)

        # Basic config
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

    @patch("ml.actors.base.PickleModelLoader")
    def test_initialization_paths(self, mock_loader):
        """
        Test all initialization paths.
        """
        mock_loader.return_value.load_model.return_value = (MockModel(), {"version": "1"})

        # Test basic initialization
        actor = MLSignalActor(self.config)
        assert actor._signal_config == self.config
        assert actor._adaptive_threshold == 0.7
        assert actor._market_regime == "unknown"
        assert len(actor._prediction_history) == 0
        assert actor._last_signal_bar == -2
        assert actor._window_index == 0

        # Test with feature config
        feature_config = FeatureConfig()
        config_with_features = MLSignalActorConfig(
            component_id="TEST-002",
            model_path=self.temp_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            feature_config=feature_config,
        )
        actor2 = MLSignalActor(config_with_features)
        assert isinstance(actor2._feature_config, FeatureConfig)

        # Test with MLFeatureConfig base
        base_config = MLFeatureConfig()
        config_base = MLSignalActorConfig(
            component_id="TEST-003",
            model_path=self.temp_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            feature_config=base_config,
        )
        actor3 = MLSignalActor(config_base)
        assert isinstance(actor3._feature_config, FeatureConfig)

        # Test ensemble weights
        config_ensemble = MLSignalActorConfig(
            component_id="TEST-004",
            model_path=self.temp_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            signal_strategy=SignalStrategy.ENSEMBLE,
            ensemble_weights={"threshold": 0.6, "extremes": 0.4},
        )
        actor4 = MLSignalActor(config_ensemble)
        assert actor4._ensemble_weights["threshold"] == 0.6

    @patch("ml.actors.base.PickleModelLoader")
    def test_load_model_method(self, mock_loader):
        """
        Test _load_model method.
        """
        mock_loader.return_value.load_model.return_value = (MockModel(), {"version": "1"})

        actor = MLSignalActor(self.config)

        # With model
        actor._model = MockModel()
        actor._load_model()

        # Without model
        actor._model = None
        actor._load_model()

    @patch("ml.actors.base.PickleModelLoader")
    def test_initialize_features_paths(self, mock_loader):
        """
        Test feature initialization.
        """
        mock_loader.return_value.load_model.return_value = (MockModel(), {"version": "1"})

        actor = MLSignalActor(self.config)

        # Normal initialization
        actor._initialize_features()
        assert actor._indicator_manager is not None

        # With feature names
        feature_config = FeatureConfig(feature_names=["f1", "f2", "f3", "f4", "f5"])
        actor._feature_config = feature_config
        actor._initialize_features()
        assert actor._feature_buffer.size >= 5

        # Without feature names
        actor._feature_config.feature_names = None
        actor._feature_engineer.n_features = 10
        actor._initialize_features()
        assert actor._feature_buffer.size == 10

    @patch("ml.actors.base.PickleModelLoader")
    def test_compute_features_scenarios(self, mock_loader):
        """
        Test feature computation scenarios.
        """
        mock_loader.return_value.load_model.return_value = (MockModel(), {"version": "1"})

        actor = MLSignalActor(self.config)
        bar = self.create_test_bar()

        # No indicator manager
        actor._indicator_manager = None
        assert actor._compute_features(bar) is None

        # Initialize manager
        actor._initialize_features()

        # Not initialized
        actor._indicator_manager.all_initialized = Mock(return_value=False)
        assert actor._compute_features(bar) is None

        # Successful computation
        actor._indicator_manager.all_initialized = Mock(return_value=True)
        actor._indicator_manager.update_from_bar = Mock()
        actor._feature_engineer.calculate_features_online = Mock(return_value=np.array([0.1, 0.2]))

        features = actor._compute_features(bar)
        assert features is not None
        assert len(features) == 2

        # Slow computation
        def slow_update(bar):
            time.sleep(0.01)

        actor._indicator_manager.update_from_bar = slow_update
        actor._config.max_feature_latency_ms = 5
        actor._compute_features(bar)

    @patch("ml.actors.base.PickleModelLoader")
    def test_predict_all_paths(self, mock_loader):
        """
        Test all prediction paths.
        """
        mock_loader.return_value.load_model.return_value = (MockModel(), {"version": "1"})

        actor = MLSignalActor(self.config)
        features = np.array([0.1, 0.2, 0.3])

        # No model
        actor._model = None
        pred, conf = actor._predict(features)
        assert pred == 0.0 and conf == 0.0

        # ONNX model - dual output
        mock_onnx = Mock()
        mock_onnx.run.return_value = [np.array([[0.8]]), np.array([[0.9]])]
        actor._model = mock_onnx
        actor._model_metadata = {"input_names": ["input"], "output_names": ["pred", "conf"]}
        pred, conf = actor._predict(features)
        assert pred == 0.8 and conf == 0.9

        # ONNX model - single output
        mock_onnx.run.return_value = [np.array([[0.7]])]
        pred, conf = actor._predict(features)
        assert pred == 0.7 and conf == 0.7

        # Sklearn with predict_proba
        mock_sklearn = Mock()
        mock_sklearn.predict_proba.return_value = np.array([[0.3, 0.7]])
        del mock_sklearn.run
        actor._model = mock_sklearn
        pred, conf = actor._predict(features)
        assert pred == 1.0 and conf == 0.7

        # Sklearn with only predict
        mock_basic = Mock()
        mock_basic.predict.return_value = np.array([0.85])
        actor._model = mock_basic
        pred, conf = actor._predict(features)
        assert pred == 0.85 and conf == 0.85

        # Zero prediction
        mock_basic.predict.return_value = np.array([0.0])
        pred, conf = actor._predict(features)
        assert pred == 0.0 and conf == 0.5

        # Unsupported model
        actor._model = object()
        pred, conf = actor._predict(features)
        assert pred == 0.0 and conf == 0.0

        # Exception handling
        mock_error = Mock()
        mock_error.predict.side_effect = RuntimeError("Error")
        actor._model = mock_error
        pred, conf = actor._predict(features)
        assert pred == 0.0 and conf == 0.0

    @patch("ml.actors.base.PickleModelLoader")
    def test_update_prediction_history(self, mock_loader):
        """
        Test prediction history updates.
        """
        mock_loader.return_value.load_model.return_value = (MockModel(), {"version": "1"})

        actor = MLSignalActor(self.config)
        actor._initialize_features()

        # With price history
        actor._indicator_manager.price_history = {"closes": [1.10, 1.11, 1.12, 1.13]}

        # Multiple updates
        for i in range(15):
            bar = self.create_test_bar(1.10 + i * 0.001)
            actor._update_prediction_history(0.5 + i * 0.01, 0.7 + i * 0.01, bar)

        assert actor._window_index < 10
        assert not np.all(actor._prediction_window == 0)

        # Without price history
        actor._indicator_manager.price_history = {}
        actor._update_prediction_history(0.8, 0.9, self.create_test_bar())

        # Test adaptive threshold update
        actor._signal_config = Mock()
        actor._signal_config.signal_strategy = SignalStrategy.ADAPTIVE
        actor._signal_config.adaptive_window = 10
        actor._update_prediction_history(0.8, 0.9, self.create_test_bar())

    @patch("ml.actors.base.PickleModelLoader")
    def test_adaptive_threshold_update(self, mock_loader):
        """
        Test adaptive threshold calculation.
        """
        mock_loader.return_value.load_model.return_value = (MockModel(), {"version": "1"})

        config = MLSignalActorConfig(
            component_id="TEST",
            model_path=self.temp_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            prediction_threshold=0.5,
            signal_strategy=SignalStrategy.ADAPTIVE,
            adaptive_window=5,
            adaptive_volatility_factor=2.0,
        )

        actor = MLSignalActor(config)

        # Fill windows
        actor._prediction_window = np.array([0.4, 0.5, 0.6, 0.7, 0.8])
        actor._volatility_window = np.array([0.01, 0.02, 0.03, 0.02, 0.01])

        initial = actor._adaptive_threshold
        actor._update_adaptive_threshold()

        assert actor._adaptive_threshold != initial
        assert 0.1 <= actor._adaptive_threshold <= 0.95

    @patch("ml.actors.base.PickleModelLoader")
    def test_market_regime_detection(self, mock_loader):
        """
        Test market regime detection.
        """
        mock_loader.return_value.load_model.return_value = (MockModel(), {"version": "1"})

        actor = MLSignalActor(self.config)
        actor._initialize_features()
        bar = self.create_test_bar()

        # No indicator manager
        actor._indicator_manager = None
        actor._detect_market_regime(bar)
        assert actor._market_regime == "unknown"

        # Restore manager
        actor._initialize_features()

        # No price history
        actor._indicator_manager.price_history = {}
        actor._detect_market_regime(bar)
        assert actor._market_regime == "unknown"

        # Insufficient data
        actor._indicator_manager.price_history = {"closes": [1.10, 1.11]}
        actor._detect_market_regime(bar)
        assert actor._market_regime == "unknown"

        # Volatile
        actor._indicator_manager.price_history = {
            "closes": [1.10 + (i % 2) * 0.05 for i in range(25)],
        }
        actor._detect_market_regime(bar)
        assert actor._market_regime == "volatile"

        # Trending
        actor._indicator_manager.price_history = {
            "closes": [1.10 + i * 0.001 for i in range(25)],
        }
        actor._detect_market_regime(bar)
        assert actor._market_regime == "trending"

        # Ranging
        actor._indicator_manager.price_history = {
            "closes": [1.10 + np.sin(i * 0.5) * 0.0001 for i in range(25)],
        }
        actor._detect_market_regime(bar)
        assert actor._market_regime == "ranging"

    @patch("ml.actors.base.PickleModelLoader")
    def test_all_signal_strategies(self, mock_loader):
        """
        Test all signal generation strategies.
        """
        mock_loader.return_value.load_model.return_value = (MockModel(), {"version": "1"})

        bar = self.create_test_bar()
        features = np.array([0.1, 0.2])

        # Threshold strategy
        actor = MLSignalActor(self.config)
        signal = actor._generate_threshold_signal(bar, 0.8, 0.9, features)
        assert signal is not None

        signal = actor._generate_threshold_signal(bar, 0.8, 0.5, features)
        assert signal is None

        # Extremes strategy
        config = MLSignalActorConfig(
            component_id="TEST",
            model_path=self.temp_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            signal_strategy=SignalStrategy.EXTREMES,
            extremes_top_pct=0.1,
            adaptive_window=10,
        )
        actor = MLSignalActor(config)

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
        config.signal_strategy = SignalStrategy.MOMENTUM
        config.momentum_lookback = 3
        actor = MLSignalActor(config)

        # Insufficient history
        actor._prediction_history = [0.5]
        signal = actor._generate_momentum_signal(bar, 0.8, 0.9, features)
        assert signal is None

        # With momentum
        actor._prediction_history = [0.2, 0.4, 0.6, 0.8]
        signal = actor._generate_momentum_signal(bar, 0.9, 0.8, features)
        assert signal is not None

        # No momentum
        actor._prediction_history = [0.5, 0.5, 0.5, 0.5]
        signal = actor._generate_momentum_signal(bar, 0.5, 0.8, features)
        assert signal is None

        # Adaptive strategy
        config.signal_strategy = SignalStrategy.ADAPTIVE
        actor = MLSignalActor(config)

        actor._adaptive_threshold = 0.95
        signal = actor._generate_adaptive_signal(bar, 0.7, 0.5, features)
        assert signal is None

        actor._adaptive_threshold = 0.3
        signal = actor._generate_adaptive_signal(bar, 0.7, 0.5, features)
        assert signal is not None

        # Ensemble strategy
        config.signal_strategy = SignalStrategy.ENSEMBLE
        actor = MLSignalActor(config)
        actor._prediction_history = [0.3, 0.4, 0.5, 0.6, 0.7]

        # With component signals
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

        # No component signals
        actor._generate_threshold_signal = Mock(return_value=None)
        signal = actor._generate_ensemble_signal(bar, 0.8, 0.9, features)
        assert signal is None

    @patch("ml.actors.base.PickleModelLoader")
    def test_generate_signal_by_strategy(self, mock_loader):
        """
        Test main signal generation method.
        """
        mock_loader.return_value.load_model.return_value = (MockModel(), {"version": "1"})

        actor = MLSignalActor(self.config)
        bar = self.create_test_bar()
        features = np.array([0.1, 0.2])

        # Signal separation
        actor._bars_processed = 5
        actor._last_signal_bar = 4
        signal = actor._generate_signal_by_strategy(bar, 0.8, 0.9, features)
        assert signal is None

        # Unknown strategy
        actor._signal_config = Mock()
        actor._signal_config.signal_strategy = "invalid"
        actor._signal_config.min_signal_separation_bars = 2
        signal = actor._generate_signal_by_strategy(bar, 0.8, 0.9, features)
        assert signal is None

    @patch("ml.actors.base.PickleModelLoader")
    def test_generate_prediction_protected(self, mock_loader):
        """
        Test protected prediction generation.
        """
        mock_loader.return_value.load_model.return_value = (MockModel(), {"version": "1"})

        actor = MLSignalActor(self.config)
        actor._circuit_breaker = CircuitBreaker(CircuitBreakerConfig())
        actor._health_monitor = HealthMonitor()

        bar = self.create_test_bar()
        features = np.array([0.1, 0.2])

        # Successful prediction
        actor._predict = Mock(return_value=(0.8, 0.9))
        actor._publish_signal = Mock()
        mock_signal = MLSignal(
            instrument_id=self.instrument_id,
            prediction=0.8,
            confidence=0.9,
            features=None,
            ts_event=0,
            ts_init=0,
        )
        actor._generate_signal_by_strategy = Mock(return_value=mock_signal)

        actor._generate_prediction_protected(bar, features)

        assert actor._circuit_breaker.consecutive_failures == 0
        assert actor._health_monitor.successful_predictions > 0

        # Failed prediction
        actor._predict = Mock(side_effect=Exception("Error"))
        actor._generate_prediction_protected(bar, features)

        assert actor._circuit_breaker.consecutive_failures > 0
        assert actor._health_monitor.failed_predictions > 0

    @patch("ml.actors.base.PickleModelLoader")
    def test_performance_tracking(self, mock_loader):
        """
        Test performance metrics tracking.
        """
        mock_loader.return_value.load_model.return_value = (MockModel(), {"version": "1"})

        actor = MLSignalActor(self.config)

        # Track metrics
        for i in range(5):
            actor._track_performance_metrics(0.5 + i * 0.1, 0.7 + i * 0.05, 2.5 + i * 0.5)

        assert actor._prediction_count == 5
        assert actor._total_inference_time > 0
        assert hasattr(actor, "_prediction_distribution_metric")
        assert hasattr(actor, "_confidence_distribution_metric")

    @patch("ml.actors.base.PickleModelLoader")
    def test_state_backup_restore(self, mock_loader):
        """
        Test state backup and restoration.
        """
        mock_loader.return_value.load_model.return_value = (MockModel(), {"version": "1"})

        actor = MLSignalActor(self.config)
        actor._initialize_features()

        # Build state
        actor._prediction_history = [0.1, 0.2, 0.3]
        actor._confidence_history = [0.6, 0.7, 0.8]
        actor._adaptive_threshold = 0.8
        actor._market_regime = "trending"
        actor._window_index = 5
        actor._last_signal_bar = 10

        # Mock indicator
        mock_ind = Mock()
        mock_ind.value = 50.0
        mock_ind.initialized = True
        actor._indicator_manager.indicators = {"sma": mock_ind}
        actor._indicator_manager.price_history = {"closes": [1.10, 1.11]}

        # Backup
        actor._backup_indicator_state()

        assert actor._indicator_state_backup is not None
        assert "prediction_history" in actor._indicator_state_backup

        # Modify state
        actor._prediction_history.clear()
        actor._adaptive_threshold = 0.99

        # Restore
        actor._restore_indicator_state()

        assert len(actor._prediction_history) == 3
        assert actor._adaptive_threshold == 0.8

        # Backup without manager
        actor._indicator_manager = None
        actor._backup_indicator_state()

        # Restore without backup
        actor._indicator_state_backup = None
        actor._restore_indicator_state()

    @patch("ml.actors.base.PickleModelLoader")
    def test_get_signal_statistics(self, mock_loader):
        """
        Test signal statistics retrieval.
        """
        mock_loader.return_value.load_model.return_value = (MockModel(), {"version": "1"})

        # With ensemble
        config = MLSignalActorConfig(
            component_id="TEST",
            model_path=self.temp_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            signal_strategy=SignalStrategy.ENSEMBLE,
        )
        actor = MLSignalActor(config)

        stats = actor.get_signal_statistics()
        assert "signal_strategy" in stats
        assert "ensemble_weights" in stats
        assert stats["ensemble_weights"] is not None

        # Without ensemble
        actor2 = MLSignalActor(self.config)
        stats2 = actor2.get_signal_statistics()
        assert stats2["ensemble_weights"] is None

    @patch("ml.actors.base.PickleModelLoader")
    def test_reset_signal_state(self, mock_loader):
        """
        Test signal state reset.
        """
        mock_loader.return_value.load_model.return_value = (MockModel(), {"version": "1"})

        actor = MLSignalActor(self.config)

        # Build state
        actor._prediction_history = [0.1, 0.2, 0.3]
        actor._confidence_history = [0.6, 0.7, 0.8]
        actor._adaptive_threshold = 0.8
        actor._market_regime = "trending"
        actor._window_index = 5
        actor._prediction_window.fill(1.0)

        # Reset
        actor.reset_signal_state()

        assert len(actor._prediction_history) == 0
        assert actor._adaptive_threshold == 0.7
        assert actor._market_regime == "unknown"
        assert actor._window_index == 0
        assert np.all(actor._prediction_window == 0.0)

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
        Test SignalStrategy enum.
        """
        assert SignalStrategy.THRESHOLD.value == "threshold"
        assert SignalStrategy.EXTREMES.value == "extremes"
        assert SignalStrategy.MOMENTUM.value == "momentum"
        assert SignalStrategy.ENSEMBLE.value == "ensemble"
        assert SignalStrategy.ADAPTIVE.value == "adaptive"
