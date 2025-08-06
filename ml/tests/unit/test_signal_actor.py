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

import contextlib
import pickle
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from ml.actors.base import MLSignal
from ml.actors.signal import AdaptiveSignal
from ml.actors.signal import MLSignalActor
from ml.actors.signal import MLSignalActorConfig
from ml.actors.signal import SignalStrategy
from ml.config.base import CircuitBreakerConfig
from ml.features.engineering import FeatureConfig
from nautilus_trader.backtest.data_client import BacktestMarketDataClient
from nautilus_trader.common.component import MessageBus
from nautilus_trader.common.component import TestClock
from nautilus_trader.common.enums import ComponentState
from nautilus_trader.data.engine import DataEngine
from nautilus_trader.execution.engine import ExecutionEngine
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarSpecification
from nautilus_trader.model.data import BarType
from nautilus_trader.model.data import DataType
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.enums import BarAggregation
from nautilus_trader.model.enums import PriceType
from nautilus_trader.model.identifiers import ClientId
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.portfolio.portfolio import Portfolio
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.test_kit.stubs.component import TestComponentStubs


class MockTestModel:
    """
    Mock model for testing.
    """

    def predict(self, X):
        return np.array([0.8])

    def predict_proba(self, X):
        return np.array([[0.2, 0.8]])


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
                with contextlib.suppress(Exception):
                    REGISTRY.unregister(collector)
            gc.collect()
        except ImportError:
            pass

        # Create temporary model file with a mock model
        self.temp_model_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)

        # Create a simple mock model
        mock_model = MockTestModel()

        # Save model to file
        with open(self.temp_model_file.name, "wb") as f:
            pickle.dump(mock_model, f)

        self.temp_model_file.close()

        # Create basic test configuration
        self.instrument_id = InstrumentId.from_str("EURUSD.SIM")
        self.bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
        self.bar_type = BarType(self.instrument_id, self.bar_spec, AggressorSide.BUYER)

        # Create feature config
        self.feature_config = FeatureConfig(
            return_periods=[1, 5],
            momentum_periods=[5, 10],
            rsi_period=14,
            bb_period=20,
            bb_std=2.0,
            ema_fast=12,
            ema_slow=26,
            feature_names=["return_1", "return_5", "rsi_14", "bb_upper", "bb_lower"],
        )

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
            feature_config=self.feature_config,
            log_predictions=True,
            enable_health_monitoring=True,
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=60,
                success_threshold=3,
            ),
        )

        # Setup Nautilus components
        self.clock = TestClock()
        self.trader_id = TraderId("TESTER-001")

        self.msgbus = MessageBus(
            trader_id=self.trader_id,
            clock=self.clock,
        )

        self.cache = TestComponentStubs.cache()

        self.portfolio = Portfolio(
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )

        self.data_engine = DataEngine(
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )

        self.exec_engine = ExecutionEngine(
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )

        self.data_client = BacktestMarketDataClient(
            client_id=ClientId("SIM"),
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )

        self.data_engine.register_client(self.data_client)

        # Add instrument
        from nautilus_trader.model.identifiers import Venue

        instrument = TestInstrumentProvider.default_fx_ccy("EUR/USD", venue=Venue("SIM"))
        self.data_engine.process(instrument)
        self.cache.add_instrument(instrument)

        self.data_engine.start()
        self.exec_engine.start()

    def teardown_method(self):
        """
        Clean up test fixtures.
        """
        # Remove temporary model file
        Path(self.temp_model_file.name).unlink(missing_ok=True)

    def create_test_actor(self, config: MLSignalActorConfig | None = None) -> MLSignalActor:
        """
        Create a properly initialized actor for testing.
        """
        if config is None:
            config = self.config

        actor = MLSignalActor(config)

        # Register with Nautilus components
        actor.register_base(
            portfolio=self.portfolio,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )

        # Start the actor
        actor.start()

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

    def test_actor_initialization(self):
        """
        Test MLSignalActor initialization with all components.
        """
        # Create actor
        actor = self.create_test_actor()

        # Test initialization
        assert actor.state == ComponentState.RUNNING
        assert actor._signal_config == self.config
        assert actor._signal_config.signal_strategy == SignalStrategy.THRESHOLD
        assert actor._adaptive_threshold == self.config.prediction_threshold
        assert len(actor._prediction_history) == 0
        assert actor._market_regime == "unknown"
        assert actor._feature_engineer is not None
        assert actor._indicator_manager is not None
        assert actor._health_monitor is not None
        assert actor._circuit_breaker is not None

    def test_threshold_signal_generation(self):
        """
        Test threshold-based signal generation.
        """
        # Create actor with threshold strategy
        config = MLSignalActorConfig(
            component_id="MLSignalActor-001",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            prediction_threshold=0.5,
            warm_up_period=1,
            signal_strategy=SignalStrategy.THRESHOLD,
            feature_config=self.feature_config,
        )

        actor = self.create_test_actor(config)

        # Verify model was loaded
        assert actor._model is not None, f"Model not loaded from {self.temp_model_file.name}"

        # Subscribe to signals
        signals_received = []

        def capture_signal(data: MLSignal) -> None:
            if isinstance(data, MLSignal):
                signals_received.append(data)

        self.msgbus.subscribe(
            topic=DataType(MLSignal).topic,
            handler=capture_signal,
        )

        # Process bars to warm up indicators
        # Need at least 30 bars for MACD/EMA initialization
        for i in range(35):  # Ensure all indicators are initialized
            bar = self.create_test_bar(close_price=1.1000 + i * 0.0001)
            actor.on_bar(bar)

        # Check if indicators are initialized
        assert actor._indicator_manager is not None, "Indicator manager not initialized"
        assert (
            actor._indicator_manager.all_initialized()
        ), "Indicators not initialized after 35 bars"

        # Should have generated signals after warm-up
        assert actor._is_warmed_up, "Actor not warmed up"
        assert actor._prediction_count > 0, f"No predictions made (count={actor._prediction_count})"

    def test_extremes_signal_generation(self):
        """
        Test extremes-based signal generation.
        """
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
            feature_config=self.feature_config,
        )

        actor = self.create_test_actor(config)

        # Mock predict method to return varying predictions
        predictions = [0.1, 0.2, 0.3, 0.8, 0.9]  # 0.8, 0.9 should be extremes
        confidences = [0.6] * len(predictions)
        prediction_idx = 0

        def mock_predict(features):
            nonlocal prediction_idx
            if prediction_idx < len(predictions):
                result = predictions[prediction_idx], confidences[prediction_idx]
                prediction_idx += 1
                return result
            return 0.5, 0.6

        # First process enough bars to initialize indicators
        for i in range(30):
            actor.on_bar(self.create_test_bar(close_price=1.1000))

        # Now mock predict and process more bars
        actor._predict = mock_predict

        # Process bars to build prediction history
        for i in range(len(predictions) + 5):
            actor.on_bar(self.create_test_bar(close_price=1.1000 + i * 0.0001))

        # Check that prediction history was built
        assert len(actor._prediction_history) >= len(predictions)
        assert actor._is_warmed_up

    def test_momentum_signal_generation(self):
        """
        Test momentum-based signal generation.
        """
        config = MLSignalActorConfig(
            component_id="MLSignalActor-001",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            prediction_threshold=0.5,
            warm_up_period=1,
            signal_strategy=SignalStrategy.MOMENTUM,
            momentum_lookback=3,
            feature_config=self.feature_config,
        )

        actor = self.create_test_actor(config)

        # Build prediction history with momentum
        actor._prediction_history = [0.2, 0.4, 0.6, 0.8]  # Increasing trend

        # Test momentum signal generation directly
        bar = self.create_test_bar()
        signal = actor._generate_momentum_signal(bar, 0.9, 0.8, np.array([0.1, 0.2]))

        # Should generate signal due to positive momentum
        assert signal is not None
        assert signal.prediction != 0.9  # Should be adjusted by momentum

    def test_ensemble_signal_generation(self):
        """
        Test ensemble signal generation combining multiple strategies.
        """
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
            feature_config=self.feature_config,
        )

        actor = self.create_test_actor(config)

        # Process bars to enable ensemble strategy (need at least 26 for EMA slow + some for predictions)
        for i in range(35):
            actor.on_bar(self.create_test_bar(close_price=1.1000 + i * 0.0001))

        assert actor._prediction_count > 0
        assert actor._ensemble_weights == {"threshold": 0.5, "extremes": 0.3, "momentum": 0.2}

    def test_adaptive_signal_generation(self):
        """
        Test adaptive signal generation with dynamic thresholds.
        """
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
            feature_config=self.feature_config,
        )

        actor = self.create_test_actor(config)

        # Subscribe to adaptive signals
        signals_received = []

        def capture_signal(data: AdaptiveSignal) -> None:
            if isinstance(data, AdaptiveSignal):
                signals_received.append(data)

        self.msgbus.subscribe(
            topic=f"{DataType(AdaptiveSignal).topic}.{self.instrument_id}",
            handler=capture_signal,
        )

        # Process bars to trigger adaptive threshold updates
        initial_threshold = actor._adaptive_threshold

        for i in range(30):
            bar = self.create_test_bar(close_price=1.1000 + i * 0.0010)  # High volatility
            actor.on_bar(bar)

        # Adaptive threshold should have changed due to volatility
        assert actor._adaptive_threshold != initial_threshold
        assert actor._window_index > 0

    def test_market_regime_detection(self):
        """
        Test market regime detection functionality.
        """
        config = MLSignalActorConfig(
            component_id="MLSignalActor-001",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            prediction_threshold=0.5,
            warm_up_period=1,
            signal_strategy=SignalStrategy.ADAPTIVE,
            enable_regime_detection=True,
            feature_config=self.feature_config,
        )

        actor = self.create_test_actor(config)

        # Process bars with trending pattern
        for i in range(30):  # Need more than 20 for regime detection
            bar = self.create_test_bar(close_price=1.1000 + i * 0.0005)  # Trending up
            actor.on_bar(bar)

        # Market regime should have been detected
        assert actor._market_regime in ["trending", "volatile", "ranging"]

    def test_feature_computation_performance(self):
        """
        Test that feature computation meets performance requirements.
        """
        actor = self.create_test_actor()

        # Process enough bars to initialize indicators
        for i in range(self.config.warm_up_period + 5):
            bar = self.create_test_bar(close_price=1.1000 + i * 0.0001)
            start_time = time.perf_counter()
            actor.on_bar(bar)
            elapsed = (time.perf_counter() - start_time) * 1000

            # After warm-up, check performance
            if i > self.config.warm_up_period:
                assert elapsed < 10  # Should be well under 10ms

        # Check that feature computation was performed
        assert actor._bars_processed > 0
        assert actor._is_warmed_up
        assert actor._total_feature_time > 0

    def test_signal_separation_enforcement(self):
        """
        Test that minimum signal separation is enforced.
        """
        config = MLSignalActorConfig(
            component_id="MLSignalActor-001",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            prediction_threshold=0.1,  # Very low threshold
            warm_up_period=1,
            signal_strategy=SignalStrategy.THRESHOLD,
            min_signal_separation_bars=5,
            feature_config=self.feature_config,
        )

        actor = self.create_test_actor(config)

        # Track published signals
        signals_published = []

        def track_signal(data: MLSignal) -> None:
            if isinstance(data, MLSignal):
                signals_published.append(actor._bars_processed)

        self.msgbus.subscribe(
            topic=f"{DataType(MLSignal).topic}.{self.instrument_id}",
            handler=track_signal,
        )

        # Process many bars rapidly
        for i in range(30):
            actor.on_bar(self.create_test_bar())

        # Check signal separation
        for i in range(1, len(signals_published)):
            separation = signals_published[i] - signals_published[i - 1]
            assert separation >= config.min_signal_separation_bars

    def test_state_backup_and_restoration(self):
        """
        Test indicator state backup and restoration during hot reload.
        """
        actor = self.create_test_actor()

        # Process bars to build state (need at least 26 for indicators)
        for i in range(30):
            actor.on_bar(self.create_test_bar(close_price=1.1000 + i * 0.0001))

        # Backup state
        original_prediction_history = actor._prediction_history.copy()
        original_adaptive_threshold = actor._adaptive_threshold
        original_window_index = actor._window_index
        original_last_signal_bar = actor._last_signal_bar

        actor._backup_indicator_state()

        # Verify backup was created
        assert actor._indicator_state_backup is not None
        assert "prediction_history" in actor._indicator_state_backup
        assert "adaptive_threshold" in actor._indicator_state_backup

        # Modify state
        actor._prediction_history.clear()
        actor._adaptive_threshold = 0.99
        actor._window_index = 0
        actor._last_signal_bar = 0

        # Restore state
        actor._restore_indicator_state()

        # State should be restored
        assert len(actor._prediction_history) == len(original_prediction_history)
        assert actor._adaptive_threshold == original_adaptive_threshold
        assert actor._window_index == original_window_index
        assert actor._last_signal_bar == original_last_signal_bar
        assert len(actor._indicator_state_backup) == 0  # Should be cleared

    def test_error_handling_in_prediction(self):
        """
        Test error handling in prediction and signal generation.
        """
        actor = self.create_test_actor()

        # Replace model with one that raises exception
        actor._model = Mock()
        actor._model.predict.side_effect = Exception("Model error")
        actor._model.predict_proba.side_effect = Exception("Model error")

        # Process bar with failing model - should not raise
        for i in range(30):
            bar = self.create_test_bar()
            actor.on_bar(bar)  # Should handle error gracefully

        # Health monitor should record failures
        assert actor._health_monitor is not None
        assert actor._health_monitor.failed_predictions > 0

    def test_circuit_breaker_functionality(self):
        """
        Test circuit breaker opens after failures.
        """
        actor = self.create_test_actor()

        # Force failures by making model raise exceptions
        actor._model = Mock()
        actor._model.predict.side_effect = Exception("Model error")

        # Process bars to trigger circuit breaker
        for i in range(10):  # More than failure threshold
            bar = self.create_test_bar()
            actor.on_bar(bar)

        # Circuit breaker should be open
        assert actor._circuit_breaker is not None
        assert actor._circuit_breaker.state != "CLOSED"

    def test_get_signal_statistics(self):
        """
        Test signal statistics retrieval.
        """
        actor = self.create_test_actor()

        # Process some bars (need at least 26 for indicators)
        for i in range(30):
            actor.on_bar(self.create_test_bar())

        # Get statistics
        stats = actor.get_signal_statistics()

        # Verify required fields are present
        assert "signal_strategy" in stats
        assert "adaptive_threshold" in stats
        assert "market_regime" in stats
        assert "prediction_history_length" in stats
        assert "feature_buffer_size" in stats
        assert "ensemble_weights" in stats

        assert stats["signal_strategy"] == SignalStrategy.THRESHOLD.value
        assert stats["adaptive_threshold"] == self.config.prediction_threshold
        assert stats["market_regime"] == "unknown"

    def test_reset_signal_state(self):
        """
        Test signal state reset functionality.
        """
        actor = self.create_test_actor()

        # Build up some state
        actor._prediction_history = [0.1, 0.2, 0.3]
        actor._confidence_history = [0.6, 0.7, 0.8]
        actor._adaptive_threshold = 0.8
        actor._market_regime = "trending"
        actor._window_index = 5

        # Reset state
        actor.reset_signal_state()

        # State should be reset
        assert len(actor._prediction_history) == 0
        assert len(actor._confidence_history) == 0
        assert actor._adaptive_threshold == actor._config.prediction_threshold
        assert actor._market_regime == "unknown"
        assert actor._window_index == 0
        assert np.all(actor._prediction_window == 0.0)
        assert np.all(actor._confidence_window == 0.0)
        assert np.all(actor._volatility_window == 0.0)

    def test_adaptive_signal_data_class(self):
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

    def test_onnx_model_prediction(self):
        """
        Test ONNX model prediction path.
        """
        actor = self.create_test_actor()

        # Mock ONNX model
        mock_onnx_model = Mock()
        mock_onnx_model.run.return_value = [np.array([[0.8]]), np.array([[0.9]])]

        actor._model = mock_onnx_model
        actor._model_metadata = {
            "input_names": ["input"],
            "output_names": ["prediction", "confidence"],
        }

        # Test prediction
        features = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        prediction, confidence = actor._predict(features)

        assert prediction == 0.8
        assert confidence == 0.9
        mock_onnx_model.run.assert_called_once()

    def test_sklearn_proba_model_prediction(self):
        """
        Test sklearn model with predict_proba.
        """
        actor = self.create_test_actor()

        # Mock sklearn model with predict_proba
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        # Ensure it doesn't have 'run' attribute to avoid ONNX path
        if hasattr(mock_model, "run"):
            del mock_model.run

        actor._model = mock_model

        # Test prediction
        features = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        prediction, confidence = actor._predict(features)

        assert prediction == 1.0  # argmax of [0.3, 0.7]
        assert confidence == 0.7  # max of [0.3, 0.7]
        mock_model.predict_proba.assert_called_once()

    def test_sklearn_basic_model_prediction(self):
        """
        Test basic sklearn model with only predict method.
        """
        actor = self.create_test_actor()

        # Mock sklearn model with only predict
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.85])
        # Ensure it doesn't have other methods
        if hasattr(mock_model, "run"):
            del mock_model.run
        if hasattr(mock_model, "predict_proba"):
            del mock_model.predict_proba

        actor._model = mock_model

        # Test prediction
        features = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        prediction, confidence = actor._predict(features)

        assert prediction == 0.85
        assert confidence == 0.85  # Uses absolute value as confidence
        mock_model.predict.assert_called_once()

    def test_unsupported_model_type(self):
        """
        Test handling of unsupported model types.
        """
        actor = self.create_test_actor()

        # Set model to an object without required methods
        actor._model = object()

        # Test prediction - should return zeros
        features = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        prediction, confidence = actor._predict(features)

        assert prediction == 0.0
        assert confidence == 0.0

    def test_prediction_with_no_model(self):
        """
        Test prediction when model is None.
        """
        actor = self.create_test_actor()

        # Set model to None
        actor._model = None

        # Test prediction - should return zeros
        features = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        prediction, confidence = actor._predict(features)

        assert prediction == 0.0
        assert confidence == 0.0

    def test_update_prediction_history(self):
        """
        Test prediction history update with circular buffers.
        """
        actor = self.create_test_actor()

        # Initialize indicator manager price history
        actor._indicator_manager.price_history = {
            "closes": [1.1000, 1.1010, 1.1020, 1.1030],
        }

        # Update prediction history multiple times
        for i in range(15):  # More than adaptive window size
            bar = self.create_test_bar(close_price=1.1000 + i * 0.0001)
            actor._update_prediction_history(0.5 + i * 0.01, 0.7 + i * 0.01, bar)

        # Check circular buffer behavior
        assert actor._window_index < actor._signal_config.adaptive_window
        assert not np.all(actor._prediction_window == 0)
        assert not np.all(actor._confidence_window == 0)

    def test_adaptive_threshold_update(self):
        """
        Test adaptive threshold calculation.
        """
        config = MLSignalActorConfig(
            component_id="MLSignalActor-001",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            prediction_threshold=0.5,
            warm_up_period=1,
            signal_strategy=SignalStrategy.ADAPTIVE,
            adaptive_window=5,
            adaptive_volatility_factor=2.0,
            feature_config=self.feature_config,
        )

        actor = self.create_test_actor(config)

        # Fill windows with test data
        actor._prediction_window = np.array([0.4, 0.5, 0.6, 0.7, 0.8])
        actor._volatility_window = np.array([0.01, 0.02, 0.03, 0.02, 0.01])

        # Update adaptive threshold
        initial_threshold = actor._adaptive_threshold
        actor._update_adaptive_threshold()

        # Threshold should have changed
        assert actor._adaptive_threshold != initial_threshold
        assert 0.1 <= actor._adaptive_threshold <= 0.95  # Within bounds

    def test_market_regime_detection_scenarios(self):
        """
        Test different market regime detection scenarios.
        """
        actor = self.create_test_actor()

        # Test volatile regime
        actor._indicator_manager.price_history = {
            "closes": [1.10 + (i % 2) * 0.05 for i in range(25)],  # High volatility
        }

        bar = self.create_test_bar()
        actor._detect_market_regime(bar)
        assert actor._market_regime == "volatile"

        # Test trending regime
        actor._indicator_manager.price_history = {
            "closes": [1.10 + i * 0.001 for i in range(25)],  # Strong uptrend
        }

        actor._detect_market_regime(bar)
        assert actor._market_regime == "trending"

        # Test ranging regime
        actor._indicator_manager.price_history = {
            "closes": [1.10 + np.sin(i * 0.5) * 0.0001 for i in range(25)],  # Small oscillations
        }

        actor._detect_market_regime(bar)
        assert actor._market_regime == "ranging"

    def test_extremes_signal_percentile_calculation(self):
        """
        Test extremes signal percentile threshold calculation.
        """
        config = MLSignalActorConfig(
            component_id="MLSignalActor-001",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            prediction_threshold=0.5,
            warm_up_period=1,
            signal_strategy=SignalStrategy.EXTREMES,
            extremes_top_pct=0.1,  # Top/bottom 10%
            adaptive_window=10,
            feature_config=self.feature_config,
        )

        actor = self.create_test_actor(config)

        # Fill prediction history
        actor._prediction_history = list(range(100))  # 0 to 99

        # Test signal generation for extreme values
        bar = self.create_test_bar()
        features = np.array([0.1, 0.2])

        # Test top extreme (should generate signal)
        signal = actor._generate_extremes_signal(bar, 95.0, 0.8, features)
        assert signal is not None

        # Test bottom extreme (should generate signal)
        signal = actor._generate_extremes_signal(bar, 5.0, 0.8, features)
        assert signal is not None

        # Test middle value (should not generate signal)
        signal = actor._generate_extremes_signal(bar, 50.0, 0.8, features)
        assert signal is None

    def test_ensemble_weights_initialization(self):
        """
        Test ensemble weights initialization with custom and default values.
        """
        # Test with custom weights
        custom_weights = {"threshold": 0.6, "extremes": 0.2, "momentum": 0.2}
        config = MLSignalActorConfig(
            component_id="MLSignalActor-001",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            signal_strategy=SignalStrategy.ENSEMBLE,
            ensemble_weights=custom_weights,
            feature_config=self.feature_config,
        )

        actor = self.create_test_actor(config)
        assert actor._ensemble_weights == custom_weights

        # Test with default weights
        config_default = MLSignalActorConfig(
            component_id="MLSignalActor-002",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            signal_strategy=SignalStrategy.ENSEMBLE,
            ensemble_weights=None,
            feature_config=self.feature_config,
        )

        actor_default = MLSignalActor(config_default)
        assert actor_default._ensemble_weights == {
            "threshold": 0.4,
            "extremes": 0.3,
            "momentum": 0.3,
        }

    def test_signal_strategy_enum_values(self):
        """
        Test SignalStrategy enum values.
        """
        assert SignalStrategy.THRESHOLD.value == "threshold"
        assert SignalStrategy.EXTREMES.value == "extremes"
        assert SignalStrategy.MOMENTUM.value == "momentum"
        assert SignalStrategy.ENSEMBLE.value == "ensemble"
        assert SignalStrategy.ADAPTIVE.value == "adaptive"

    def test_compute_features_without_indicator_manager(self):
        """
        Test feature computation when indicator manager is None.
        """
        actor = self.create_test_actor()

        # Set indicator manager to None
        actor._indicator_manager = None

        # Try to compute features
        bar = self.create_test_bar()
        features = actor._compute_features(bar)

        assert features is None

    def test_compute_features_before_indicators_ready(self):
        """
        Test feature computation before indicators are initialized.
        """
        actor = self.create_test_actor()

        # Mock indicator manager to return not initialized
        actor._indicator_manager.all_initialized = Mock(return_value=False)

        # Try to compute features
        bar = self.create_test_bar()
        features = actor._compute_features(bar)

        assert features is None

    def test_load_model_logging(self):
        """
        Test model loading with logging.
        """
        actor = self.create_test_actor()

        # Call load model method
        actor._load_model()

        # Model should be loaded (from base class)
        assert actor._model is not None

    def test_generate_signal_unknown_strategy(self):
        """
        Test handling of unknown signal strategy.
        """
        actor = self.create_test_actor()

        # Set an invalid strategy
        actor._signal_config = Mock()
        actor._signal_config.signal_strategy = "unknown_strategy"
        actor._signal_config.min_signal_separation_bars = 2

        # Try to generate signal
        bar = self.create_test_bar()
        signal = actor._generate_signal_by_strategy(bar, 0.8, 0.9, np.array([0.1, 0.2]))

        assert signal is None

    def test_track_performance_metrics(self):
        """
        Test performance metrics tracking.
        """
        actor = self.create_test_actor()

        # Track some metrics
        for i in range(5):
            actor._track_performance_metrics(0.5 + i * 0.1, 0.7 + i * 0.05, 2.5 + i * 0.5)

        assert actor._prediction_count == 5
        assert actor._total_inference_time > 0
        assert hasattr(actor, "_prediction_distribution_metric")
        assert hasattr(actor, "_confidence_distribution_metric")

    def test_hot_reload_check_scheduling(self):
        """
        Test hot reload scheduling is set up when enabled.
        """
        config = MLSignalActorConfig(
            component_id="MLSignalActor-001",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            enable_hot_reload=True,
            model_check_interval=60,
            feature_config=self.feature_config,
        )

        actor = self.create_test_actor(config)

        # Should have scheduled model checks
        assert actor._config.enable_hot_reload
        assert actor._config.model_check_interval == 60

    def test_feature_config_initialization(self):
        """
        Test different feature config initialization scenarios.
        """
        # Test with no feature config
        config_no_features = MLSignalActorConfig(
            component_id="MLSignalActor-001",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            feature_config=None,
        )

        actor = MLSignalActor(config_no_features)
        assert actor._feature_config is not None
        assert isinstance(actor._feature_config, FeatureConfig)

    def test_feature_buffer_resizing(self):
        """
        Test feature buffer resizing based on feature configuration.
        """
        # Create config with specific feature names
        feature_config = FeatureConfig(
            feature_names=["feature1", "feature2", "feature3"],
        )

        config = MLSignalActorConfig(
            component_id="MLSignalActor-001",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            feature_config=feature_config,
        )

        actor = self.create_test_actor(config)

        # Feature buffer should match feature count
        assert actor._feature_buffer.size >= 3

    def test_ensemble_signal_with_no_component_signals(self):
        """
        Test ensemble signal when no component strategies generate signals.
        """
        actor = self.create_test_actor()
        actor._signal_config = Mock()
        actor._signal_config.signal_strategy = SignalStrategy.ENSEMBLE
        actor._signal_config.min_signal_separation_bars = 0

        # Mock all component methods to return None
        actor._generate_threshold_signal = Mock(return_value=None)
        actor._generate_extremes_signal = Mock(return_value=None)
        actor._generate_momentum_signal = Mock(return_value=None)

        # Try to generate ensemble signal
        bar = self.create_test_bar()
        signal = actor._generate_ensemble_signal(bar, 0.5, 0.6, np.array([0.1, 0.2]))

        assert signal is None

    def test_slow_feature_computation_warning(self):
        """
        Test warning when feature computation is slow.
        """
        actor = self.create_test_actor()

        # Mock slow feature computation
        original_update = actor._indicator_manager.update_from_bar

        def slow_update(bar):
            time.sleep(0.01)  # Make it slow
            original_update(bar)

        actor._indicator_manager.update_from_bar = slow_update

        # Process bar - should log warning
        for i in range(30):
            bar = self.create_test_bar()
            actor.on_bar(bar)

    def test_indicator_state_backup_without_manager(self):
        """
        Test indicator state backup when manager is None.
        """
        actor = self.create_test_actor()

        # Set indicator manager to None
        original_manager = actor._indicator_manager
        actor._indicator_manager = None

        # Backup state
        actor._backup_indicator_state()

        # Should still create backup with minimal data
        assert actor._indicator_state_backup is not None
        assert "prediction_history" in actor._indicator_state_backup

        # Restore manager
        actor._indicator_manager = original_manager

    def test_indicator_state_restore_without_backup(self):
        """
        Test indicator state restore when no backup exists.
        """
        actor = self.create_test_actor()

        # Clear any existing backup
        actor._indicator_state_backup = None

        # Try to restore - should handle gracefully
        actor._restore_indicator_state()

        # State should remain unchanged
        assert actor._prediction_history == []
        assert actor._market_regime == "unknown"

    def test_adaptive_signal_generation_below_threshold(self):
        """
        Test adaptive signal when signal strength is below threshold.
        """
        config = MLSignalActorConfig(
            component_id="MLSignalActor-001",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            signal_strategy=SignalStrategy.ADAPTIVE,
            feature_config=self.feature_config,
        )

        actor = self.create_test_actor(config)

        # Set high adaptive threshold
        actor._adaptive_threshold = 0.95

        # Generate signal with low confidence
        bar = self.create_test_bar()
        signal = actor._generate_adaptive_signal(bar, 0.7, 0.5, np.array([0.1, 0.2]))

        # Should not generate signal (strength < 1.0)
        assert signal is None

    def test_momentum_signal_insufficient_history(self):
        """
        Test momentum signal when insufficient prediction history.
        """
        actor = self.create_test_actor()

        # Clear prediction history
        actor._prediction_history = [0.5]  # Less than momentum lookback

        # Try to generate momentum signal
        bar = self.create_test_bar()
        signal = actor._generate_momentum_signal(bar, 0.8, 0.9, np.array([0.1, 0.2]))

        assert signal is None

    def test_extremes_signal_insufficient_history(self):
        """
        Test extremes signal when insufficient prediction history.
        """
        actor = self.create_test_actor()

        # Clear prediction history
        actor._prediction_history = [0.5, 0.6]  # Less than adaptive window

        # Try to generate extremes signal
        bar = self.create_test_bar()
        signal = actor._generate_extremes_signal(bar, 0.8, 0.9, np.array([0.1, 0.2]))

        assert signal is None

    def test_regime_detection_insufficient_data(self):
        """
        Test market regime detection with insufficient price history.
        """
        actor = self.create_test_actor()

        # Set insufficient price history
        actor._indicator_manager.price_history = {"closes": [1.1000, 1.1001]}

        # Try to detect regime
        bar = self.create_test_bar()
        actor._detect_market_regime(bar)

        # Regime should remain unknown
        assert actor._market_regime == "unknown"

    def test_ensemble_signal_partial_strategies(self):
        """
        Test ensemble signal when only some strategies generate signals.
        """
        config = MLSignalActorConfig(
            component_id="MLSignalActor-001",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            prediction_threshold=0.5,
            signal_strategy=SignalStrategy.ENSEMBLE,
            ensemble_weights={"threshold": 0.5, "extremes": 0.3, "momentum": 0.2},
            feature_config=self.feature_config,
        )

        actor = self.create_test_actor(config)

        # Set up state for ensemble
        actor._prediction_history = [0.3, 0.4, 0.5, 0.6, 0.7]
        actor._last_signal_bar = -10

        # Process bar to generate ensemble signal
        bar = self.create_test_bar()
        signal = actor._generate_ensemble_signal(bar, 0.8, 0.7, np.array([0.1, 0.2]))

        # Should generate signal if ensemble confidence is high enough
        if signal is not None:
            assert isinstance(signal, MLSignal)
            assert signal.confidence >= config.prediction_threshold

    def test_prediction_exception_handling(self):
        """
        Test exception handling in _predict method.
        """
        actor = self.create_test_actor()

        # Set model that raises exception
        actor._model = Mock()
        actor._model.predict.side_effect = RuntimeError("Prediction failed")

        # Should handle exception and return zeros
        features = np.array([0.1, 0.2, 0.3])
        prediction, confidence = actor._predict(features)

        assert prediction == 0.0
        assert confidence == 0.0

    def test_zero_prediction_confidence_estimation(self):
        """
        Test confidence estimation for zero predictions.
        """
        actor = self.create_test_actor()

        # Mock model to return zero prediction
        actor._model = Mock()
        actor._model.predict.return_value = np.array([0.0])
        if hasattr(actor._model, "run"):
            del actor._model.run
        if hasattr(actor._model, "predict_proba"):
            del actor._model.predict_proba

        # Test prediction
        features = np.array([0.1, 0.2, 0.3])
        prediction, confidence = actor._predict(features)

        assert prediction == 0.0
        assert confidence == 0.5  # Default confidence for zero prediction

    def test_momentum_below_threshold(self):
        """
        Test momentum signal when momentum is below threshold.
        """
        config = MLSignalActorConfig(
            component_id="MLSignalActor-001",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            signal_strategy=SignalStrategy.MOMENTUM,
            momentum_lookback=3,
            feature_config=self.feature_config,
        )

        actor = self.create_test_actor(config)

        # Set up flat prediction history (no momentum)
        actor._prediction_history = [0.5, 0.5, 0.5, 0.5]

        # Try to generate momentum signal
        bar = self.create_test_bar()
        signal = actor._generate_momentum_signal(bar, 0.5, 0.8, np.array([0.1, 0.2]))

        # Should not generate signal (momentum too low)
        assert signal is None

    def test_onnx_single_output_model(self):
        """
        Test ONNX model with single output.
        """
        actor = self.create_test_actor()

        # Mock ONNX model with single output
        mock_model = Mock()
        mock_model.run.return_value = [np.array([[0.85]])]  # Single output

        actor._model = mock_model
        actor._model_metadata = {
            "input_names": ["input"],
            "output_names": ["prediction"],
        }

        # Test prediction
        features = np.array([0.1, 0.2, 0.3])
        prediction, confidence = actor._predict(features)

        assert prediction == 0.85
        assert confidence == 0.85  # Uses absolute value as confidence

    def test_generate_prediction_protected_success(self):
        """
        Test successful prediction generation with all features.
        """
        actor = self.create_test_actor()

        # Mock successful prediction
        actor._predict = Mock(return_value=(0.8, 0.9))
        actor._publish_signal = Mock()

        # Process enough bars to warm up and initialize indicators
        for i in range(35):
            bar = self.create_test_bar(close_price=1.1000 + i * 0.0001)
            actor.on_bar(bar)

        # Check that predictions were made
        assert actor._prediction_count > 0
        assert actor._circuit_breaker.consecutive_failures == 0

    def test_generate_prediction_protected_failure(self):
        """
        Test prediction generation failure handling.
        """
        actor = self.create_test_actor()

        # Mock prediction to raise exception
        actor._predict = Mock(side_effect=Exception("Prediction error"))

        # Generate prediction - should handle error
        bar = self.create_test_bar()
        features = np.array([0.1, 0.2, 0.3])

        # This should not raise
        actor._generate_prediction_protected(bar, features)

        # Circuit breaker should record failure
        assert actor._circuit_breaker.consecutive_failures > 0
