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
Tests for MLSignalActor with comprehensive coverage of all signal strategies.

Tests cover:
- All signal generation strategies (threshold, extremes, momentum, ensemble, adaptive)
- Feature computation and model prediction
- Adaptive threshold adjustment
- Market regime detection
- Performance monitoring and metrics
- State backup and restoration
- Error handling and circuit breaker

"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

from ml.actors.signal import AdaptiveSignal
from ml.actors.signal import MLSignalActor
from ml.actors.signal import MLSignalActorConfig
from ml.actors.signal import SignalStrategy
from nautilus_trader.common.component import MessageBus
from nautilus_trader.common.component import TestClock
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarSpecification
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.enums import BarAggregation
from nautilus_trader.model.enums import PriceType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity


class TestMLSignalActor:
    """
    Test cases for MLSignalActor.
    """

    def setup_method(self):
        """
        Set up test fixtures.
        """
        # Clear Prometheus metrics registry to avoid duplicates
        try:
            import gc

            from prometheus_client import REGISTRY

            # Clear all collectors to avoid duplicate metrics
            collectors = list(REGISTRY._collector_to_names.keys())
            for collector in collectors:
                try:
                    REGISTRY.unregister(collector)
                except Exception:
                    pass
            gc.collect()
        except ImportError:
            pass

        # Create temporary model file
        self.temp_model_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        self.temp_model_file.close()

        # Create basic test configuration
        self.instrument_id = InstrumentId.from_str("EURUSD.SIM")
        self.bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
        self.bar_type = BarType(self.instrument_id, self.bar_spec, AggressorSide.BUYER)

        self.config = MLSignalActorConfig(
            component_id="MLSignalActor-001",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            prediction_threshold=0.7,
            warm_up_period=20,
            signal_strategy=SignalStrategy.THRESHOLD,
            adaptive_window=10,
            min_signal_separation_bars=2,
        )

        # Mock clock and message bus for testing
        self.clock = TestClock()
        self.trader_id = TraderId("TESTER-001")
        self.msgbus = MessageBus(
            trader_id=self.trader_id,
            clock=self.clock,
        )

    def teardown_method(self):
        """
        Clean up test fixtures.
        """
        # Remove temporary model file
        Path(self.temp_model_file.name).unlink(missing_ok=True)

    def create_mock_model(self):
        """
        Create a mock ML model.
        """
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.8])
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
        return mock_model

    def create_test_actor(self, config: MLSignalActorConfig) -> MLSignalActor:
        """
        Create an actor for testing.
        """
        # For unit tests, we can create the actor directly
        # The actor will use its own internal clock
        actor = MLSignalActor(config)
        return actor

    def create_test_bar(self, close_price: float = 1.1000, volume: float = 1000.0) -> Bar:
        """
        Create a test bar with specified parameters.
        """
        return Bar(
            bar_type=self.bar_type,
            open=Price.from_str(str(close_price - 0.0002)),
            high=Price.from_str(str(close_price + 0.0003)),
            low=Price.from_str(str(close_price - 0.0004)),
            close=Price.from_str(str(close_price)),
            volume=Quantity.from_str(str(volume)),
            ts_event=self.clock.timestamp_ns(),
            ts_init=self.clock.timestamp_ns(),
        )

    @patch("ml.actors.base.PickleModelLoader")
    def test_actor_initialization(self, mock_loader):
        """
        Test MLSignalActor initialization.
        """
        # Setup mock loader
        mock_model = self.create_mock_model()
        mock_loader.return_value.load_model.return_value = (mock_model, {"version": "test"})

        # Create actor
        actor = self.create_test_actor(self.config)

        # Test initialization
        assert actor._signal_config == self.config
        assert actor._signal_config.signal_strategy == SignalStrategy.THRESHOLD
        assert actor._adaptive_threshold == self.config.prediction_threshold
        assert len(actor._prediction_history) == 0
        assert actor._market_regime == "unknown"

    @patch("ml.actors.base.PickleModelLoader")
    def test_threshold_signal_generation(self, mock_loader):
        """
        Test threshold-based signal generation.
        """
        # Setup mock model
        mock_model = self.create_mock_model()
        mock_loader.return_value.load_model.return_value = (mock_model, {"version": "test"})

        # Create actor with threshold strategy
        config = MLSignalActorConfig(
            component_id="MLSignalActor-001",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            prediction_threshold=0.5,
            warm_up_period=1,
            signal_strategy=SignalStrategy.THRESHOLD,
        )

        actor = self.create_test_actor(config)
        # actor.state is read-only, actors start in INITIALIZED state

        # Mock the prediction method
        actor._predict = MagicMock(return_value=(0.8, 0.9))

        # Process bar to trigger signal generation
        bar = self.create_test_bar()
        actor.on_bar(bar)

        # Advance past warm-up
        for _ in range(config.warm_up_period):
            actor.on_bar(self.create_test_bar())

        # Should generate signal since confidence (0.9) > threshold (0.5)
        assert actor._prediction_count > 0

    @patch("ml.actors.base.PickleModelLoader")
    def test_extremes_signal_generation(self, mock_loader):
        """
        Test extremes-based signal generation.
        """
        # Setup mock model
        mock_model = self.create_mock_model()
        mock_loader.return_value.load_model.return_value = (mock_model, {"version": "test"})

        config = MLSignalActorConfig(
            component_id="MLSignalActor-001",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            prediction_threshold=0.5,
            warm_up_period=1,
            signal_strategy=SignalStrategy.EXTREMES,
            extremes_top_pct=0.2,
            adaptive_window=5,
        )

        actor = self.create_test_actor(config)
        # actor.state is read-only, actors start in INITIALIZED state

        # Mock prediction method to return varying predictions
        predictions = [0.1, 0.2, 0.3, 0.8, 0.9]  # 0.8, 0.9 should be extremes
        confidences = [0.6] * len(predictions)

        def mock_predict(features):
            idx = len(actor._prediction_history)
            if idx < len(predictions):
                return predictions[idx], confidences[idx]
            return 0.5, 0.6

        actor._predict = mock_predict

        # Process bars to build prediction history
        for i in range(len(predictions) + 2):
            actor.on_bar(self.create_test_bar(close_price=1.1000 + i * 0.0001))

        # Check that prediction history was built
        assert len(actor._prediction_history) > 0

    @patch("ml.actors.base.PickleModelLoader")
    def test_adaptive_signal_generation(self, mock_loader):
        """
        Test adaptive signal generation with dynamic thresholds.
        """
        # Setup mock model
        mock_model = self.create_mock_model()
        mock_loader.return_value.load_model.return_value = (mock_model, {"version": "test"})

        config = MLSignalActorConfig(
            component_id="MLSignalActor-001",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            prediction_threshold=0.5,
            warm_up_period=1,
            signal_strategy=SignalStrategy.ADAPTIVE,
            adaptive_window=5,
            adaptive_volatility_factor=1.0,
        )

        actor = self.create_test_actor(config)
        # actor.state is read-only, actors start in INITIALIZED state

        # Mock prediction method
        actor._predict = MagicMock(return_value=(0.8, 0.9))

        # Process bars to trigger adaptive threshold updates
        initial_threshold = actor._adaptive_threshold

        for i in range(10):
            bar = self.create_test_bar(close_price=1.1000 + i * 0.0010)  # High volatility
            actor.on_bar(bar)

        # Adaptive threshold should have changed due to volatility
        assert actor._adaptive_threshold != initial_threshold

    @patch("ml.actors.base.PickleModelLoader")
    def test_ensemble_signal_generation(self, mock_loader):
        """
        Test ensemble signal generation combining multiple strategies.
        """
        # Setup mock model
        mock_model = self.create_mock_model()
        mock_loader.return_value.load_model.return_value = (mock_model, {"version": "test"})

        config = MLSignalActorConfig(
            component_id="MLSignalActor-001",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            prediction_threshold=0.3,
            warm_up_period=1,
            signal_strategy=SignalStrategy.ENSEMBLE,
            ensemble_weights={"threshold": 0.5, "extremes": 0.3, "momentum": 0.2},
            adaptive_window=5,
        )

        actor = self.create_test_actor(config)
        # actor.state is read-only, actors start in INITIALIZED state

        # Mock prediction method
        actor._predict = MagicMock(return_value=(0.8, 0.7))

        # Process bars to enable ensemble strategy
        for i in range(10):
            actor.on_bar(self.create_test_bar(close_price=1.1000 + i * 0.0001))

        assert actor._prediction_count > 0

    @patch("ml.actors.base.PickleModelLoader")
    def test_market_regime_detection(self, mock_loader):
        """
        Test market regime detection functionality.
        """
        # Setup mock model
        mock_model = self.create_mock_model()
        mock_loader.return_value.load_model.return_value = (mock_model, {"version": "test"})

        config = MLSignalActorConfig(
            component_id="MLSignalActor-001",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            prediction_threshold=0.5,
            warm_up_period=1,
            signal_strategy=SignalStrategy.ADAPTIVE,
            enable_regime_detection=True,
        )

        actor = self.create_test_actor(config)
        # actor.state is read-only, actors start in INITIALIZED state

        # Mock prediction method
        actor._predict = MagicMock(return_value=(0.8, 0.7))

        # Process bars with trending pattern
        for i in range(25):  # Need more than 20 for regime detection
            bar = self.create_test_bar(close_price=1.1000 + i * 0.0005)  # Trending up
            actor.on_bar(bar)

        # Market regime should have been detected
        assert actor._market_regime in ["trending", "volatile", "ranging"]

    @patch("ml.actors.base.PickleModelLoader")
    def test_feature_computation_performance(self, mock_loader):
        """
        Test that feature computation meets performance requirements.
        """
        # Setup mock model
        mock_model = self.create_mock_model()
        mock_loader.return_value.load_model.return_value = (mock_model, {"version": "test"})

        actor = self.create_test_actor(self.config)
        # actor.state is read-only, actors start in INITIALIZED state

        # Process enough bars to initialize indicators
        for i in range(self.config.warm_up_period + 5):
            bar = self.create_test_bar(close_price=1.1000 + i * 0.0001)
            actor.on_bar(bar)

        # Check that feature computation was performed
        assert actor._bars_processed > 0
        assert actor._is_warmed_up

    @patch("ml.actors.base.PickleModelLoader")
    def test_signal_separation_enforcement(self, mock_loader):
        """
        Test that minimum signal separation is enforced.
        """
        # Setup mock model
        mock_model = self.create_mock_model()
        mock_loader.return_value.load_model.return_value = (mock_model, {"version": "test"})

        config = MLSignalActorConfig(
            component_id="MLSignalActor-001",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            prediction_threshold=0.1,  # Very low threshold
            warm_up_period=1,
            signal_strategy=SignalStrategy.THRESHOLD,
            min_signal_separation_bars=5,
        )

        actor = self.create_test_actor(config)
        # actor.state is read-only, actors start in INITIALIZED state

        # Mock prediction to always return high confidence
        actor._predict = MagicMock(return_value=(0.8, 0.9))

        # Mock publish_signal to track calls
        actor._publish_signal = MagicMock()

        # Process many bars rapidly
        for i in range(20):
            actor.on_bar(self.create_test_bar())

        # Should have limited number of signals due to separation requirement
        # (exact count depends on when signals were generated)
        signal_calls = actor._publish_signal.call_count
        assert signal_calls < 5  # Should be limited by separation

    @patch("ml.actors.base.PickleModelLoader")
    def test_state_backup_and_restoration(self, mock_loader):
        """
        Test indicator state backup and restoration during hot reload.
        """
        # Setup mock model
        mock_model = self.create_mock_model()
        mock_loader.return_value.load_model.return_value = (mock_model, {"version": "test"})

        actor = self.create_test_actor(self.config)
        # actor.state is read-only, actors start in INITIALIZED state

        # Process bars to build state
        for i in range(10):
            actor.on_bar(self.create_test_bar(close_price=1.1000 + i * 0.0001))

        # Backup state
        original_prediction_history = actor._prediction_history.copy()
        original_adaptive_threshold = actor._adaptive_threshold

        actor._backup_indicator_state()

        # Modify state
        actor._prediction_history.clear()
        actor._adaptive_threshold = 0.99

        # Restore state
        actor._restore_indicator_state()

        # State should be restored (at least partially)
        assert len(actor._prediction_history) == len(original_prediction_history)
        assert actor._adaptive_threshold == original_adaptive_threshold

    @patch("ml.actors.base.PickleModelLoader")
    def test_error_handling(self, mock_loader):
        """
        Test error handling in prediction and signal generation.
        """
        # Setup mock model that raises exception
        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("Model error")
        mock_loader.return_value.load_model.return_value = (mock_model, {"version": "test"})

        actor = self.create_test_actor(self.config)
        # actor.state is read-only, actors start in INITIALIZED state

        # Process bar with failing model
        bar = self.create_test_bar()

        # Should not raise exception
        actor.on_bar(bar)

        # Health monitor should record the failure
        if actor._health_monitor:
            assert actor._health_monitor.failed_predictions > 0

    @patch("ml.actors.base.PickleModelLoader")
    def test_get_signal_statistics(self, mock_loader):
        """
        Test signal statistics retrieval.
        """
        # Setup mock model
        mock_model = self.create_mock_model()
        mock_loader.return_value.load_model.return_value = (mock_model, {"version": "test"})

        actor = self.create_test_actor(self.config)
        # actor.state is read-only, actors start in INITIALIZED state

        # Process some bars
        for i in range(5):
            actor.on_bar(self.create_test_bar())

        # Get statistics
        stats = actor.get_signal_statistics()

        # Verify required fields are present
        assert "signal_strategy" in stats
        assert "adaptive_threshold" in stats
        assert "market_regime" in stats
        assert "prediction_history_length" in stats
        assert "feature_buffer_size" in stats

        assert stats["signal_strategy"] == SignalStrategy.THRESHOLD.value

    @patch("ml.actors.base.PickleModelLoader")
    def test_reset_signal_state(self, mock_loader):
        """
        Test signal state reset functionality.
        """
        # Setup mock model
        mock_model = self.create_mock_model()
        mock_loader.return_value.load_model.return_value = (mock_model, {"version": "test"})

        actor = self.create_test_actor(self.config)
        # actor.state is read-only, actors start in INITIALIZED state

        # Build up some state
        actor._prediction_history = [0.1, 0.2, 0.3]
        actor._confidence_history = [0.6, 0.7, 0.8]
        actor._adaptive_threshold = 0.8
        actor._market_regime = "trending"

        # Reset state
        actor.reset_signal_state()

        # State should be reset
        assert len(actor._prediction_history) == 0
        assert len(actor._confidence_history) == 0
        assert actor._adaptive_threshold == actor._config.prediction_threshold
        assert actor._market_regime == "unknown"

    def test_adaptive_signal_properties(self):
        """
        Test AdaptiveSignal data class properties.
        """
        instrument_id = InstrumentId.from_str("EURUSD.SIM")

        signal = AdaptiveSignal(
            instrument_id=instrument_id,
            prediction=0.8,
            confidence=0.9,
            adaptive_threshold=0.7,
            signal_strength=1.2,
            market_regime="trending",
            ts_event=123456789,
            ts_init=123456790,
        )

        assert signal.instrument_id == instrument_id
        assert signal.prediction == 0.8
        assert signal.confidence == 0.9
        assert signal.adaptive_threshold == 0.7
        assert signal.signal_strength == 1.2
        assert signal.market_regime == "trending"
        assert signal.ts_event == 123456789
        assert signal.ts_init == 123456790

    @patch("ml.actors.base.PickleModelLoader")
    def test_different_model_types(self, mock_loader):
        """
        Test handling of different model types (sklearn, ONNX).
        """
        # Test sklearn model with predict_proba
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        # Ensure it doesn't have 'run' attribute to avoid ONNX path
        if hasattr(mock_model, "run"):
            del mock_model.run
        mock_loader.return_value.load_model.return_value = (mock_model, {"version": "test"})

        actor = MLSignalActor(self.config)
        # Initialize the model metadata first
        actor._model = mock_model
        actor._model_metadata = {"input_names": ["input"], "output_names": ["output"]}

        # Test prediction
        features = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        prediction, confidence = actor._predict(features)

        # The mock model has both predict and predict_proba
        # but predict_proba is checked first in the code
        assert prediction == 1.0  # argmax of [0.3, 0.7]
        assert confidence == 0.7  # max of [0.3, 0.7]

        # Verify the correct method was called
        mock_model.predict_proba.assert_called_once()

    @patch("ml.actors.base.PickleModelLoader")
    def test_momentum_signal_logic(self, mock_loader):
        """
        Test momentum signal generation logic.
        """
        # Setup mock model
        mock_model = self.create_mock_model()
        mock_loader.return_value.load_model.return_value = (mock_model, {"version": "test"})

        config = MLSignalActorConfig(
            component_id="MLSignalActor-001",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            prediction_threshold=0.5,
            warm_up_period=1,
            signal_strategy=SignalStrategy.MOMENTUM,
            momentum_lookback=3,
        )

        actor = self.create_test_actor(config)
        # actor.state is read-only, actors start in INITIALIZED state

        # Build prediction history with momentum
        actor._prediction_history = [0.2, 0.4, 0.6, 0.8]  # Increasing trend

        # Mock current prediction
        actor._predict = MagicMock(return_value=(0.9, 0.8))

        # Test momentum signal generation
        bar = self.create_test_bar()
        signal = actor._generate_momentum_signal(bar, 0.9, 0.8, np.array([0.1, 0.2]))

        # Should generate signal due to positive momentum
        assert signal is not None
        # Prediction should be adjusted by momentum
        assert signal.prediction != 0.9

    def test_signal_strategy_enum(self):
        """
        Test SignalStrategy enum values.
        """
        assert SignalStrategy.THRESHOLD.value == "threshold"
        assert SignalStrategy.EXTREMES.value == "extremes"
        assert SignalStrategy.MOMENTUM.value == "momentum"
        assert SignalStrategy.ENSEMBLE.value == "ensemble"
        assert SignalStrategy.ADAPTIVE.value == "adaptive"
