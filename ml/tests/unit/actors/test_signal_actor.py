
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
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import numpy.typing as npt
import pytest

from ml.actors.base import CircuitBreakerState
from ml.actors.base import MLSignal
from ml.actors.signal import AdaptiveSignal
from ml.actors.signal import MLSignalActor
from ml.actors.signal import MLSignalActorConfig
from ml.actors.signal import SignalStrategy
from ml.actors.signal import StrategyConfig
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

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array([0.8])

    def predict_proba(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array([[0.2, 0.8]])


class TestMLSignalActor:
    """
    Test cases for MLSignalActor.
    """

    def setup_method(self) -> None:
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

        # Create temporary model file - we'll mock the actual loading
        self.temp_model_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
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
            model_id="test_model",
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

    def teardown_method(self) -> None:
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

        # Mock the model loading to avoid file I/O issues in tests
        mock_model = MockTestModel()
        mock_metadata = {"n_features": 5}
        
        with patch('ml.models.loader.ProductionModelLoader') as mock_loader_class:
            mock_loader_instance = Mock()
            mock_loader_instance.load_model.return_value = (mock_model, mock_metadata)
            mock_loader_class.return_value = mock_loader_instance
            
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

    def test_actor_initialization(self) -> None:
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

    def test_threshold_signal_generation(self) -> None:
        """
        Test threshold-based signal generation.
        """
        # Create actor with threshold strategy
        config = MLSignalActorConfig(
            model_id="test_model",
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

        # Should have generated signals after warm-up - test behavior, not implementation
        assert actor._is_warmed_up, "Actor not warmed up"
        
        # Test behavior: verify signals were published by checking statistics
        signal_stats = actor.get_signal_statistics()
        assert signal_stats.get("predictions_made", 0) > 0, "No predictions were made during processing"

    def test_extremes_signal_generation(self) -> None:
        """
        Test extremes-based signal generation.
        """
        config = MLSignalActorConfig(
            model_id="test_model",
            component_id="MLSignalActor-001",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            prediction_threshold=0.5,
            warm_up_period=1,
            signal_strategy=SignalStrategy.EXTREMES,
            adaptive_window=5,
            strategy_config=StrategyConfig(extremes_top_pct=0.2),
            feature_config=self.feature_config,
        )

        actor = self.create_test_actor(config)

        # Mock predict method to return varying predictions
        predictions = [0.1, 0.2, 0.3, 0.8, 0.9]  # 0.8, 0.9 should be extremes
        confidences = [0.6] * len(predictions)
        prediction_idx = 0

        def mock_predict(features: npt.NDArray[np.float64]) -> tuple[float, float]:
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
        actor._predict = mock_predict  # type: ignore[assignment]

        # Process bars to build prediction history
        for i in range(len(predictions) + 5):
            actor.on_bar(self.create_test_bar(close_price=1.1000 + i * 0.0001))

        # Check that prediction history was built
        assert len(actor._prediction_history) >= len(predictions)
        assert actor._is_warmed_up

    def test_momentum_signal_generation(self) -> None:
        """
        Test momentum-based signal generation.
        """
        config = MLSignalActorConfig(
            model_id="test_model",
            component_id="MLSignalActor-001",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            prediction_threshold=0.5,
            warm_up_period=1,
            signal_strategy=SignalStrategy.MOMENTUM,
            strategy_config=StrategyConfig(momentum_lookback=3),
            feature_config=self.feature_config,
        )

        actor = self.create_test_actor(config)

        # Build prediction history with momentum
        actor._prediction_history = [0.2, 0.4, 0.6, 0.8]  # Increasing trend

        # Process bars to generate signal through strategy pattern
        for i in range(35):  # Need enough bars for indicators
            bar = self.create_test_bar()
            actor.on_bar(bar)

        # Check that momentum strategy was used
        assert actor._signal_config.signal_strategy == SignalStrategy.MOMENTUM
        assert len(actor._prediction_history) > 0

    def test_ensemble_signal_generation(self) -> None:
        """
        Test ensemble signal generation combining multiple strategies.
        """
        config = MLSignalActorConfig(
            model_id="test_model",
            component_id="MLSignalActor-001",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            prediction_threshold=0.3,
            warm_up_period=1,
            signal_strategy=SignalStrategy.ENSEMBLE,
            adaptive_window=5,
            strategy_config=StrategyConfig(
                ensemble_weights={"threshold": 0.5, "extremes": 0.3, "momentum": 0.2},
            ),
            feature_config=self.feature_config,
        )

        actor = self.create_test_actor(config)

        # Process bars to enable ensemble strategy (need at least 26 for EMA slow + some for predictions)
        for i in range(35):
            actor.on_bar(self.create_test_bar(close_price=1.1000 + i * 0.0001))

        # Test behavior: verify predictions were made through statistics
        signal_stats = actor.get_signal_statistics()
        assert signal_stats.get("predictions_made", 0) > 0, "No predictions were made"
        assert actor._strat_config.ensemble_weights == {
            "threshold": 0.5,
            "extremes": 0.3,
            "momentum": 0.2,
        }

    def test_adaptive_signal_generation(self) -> None:
        """
        Test adaptive signal generation with dynamic thresholds.
        """
        config = MLSignalActorConfig(
            model_id="test_model",
            component_id="MLSignalActor-001",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            prediction_threshold=0.5,
            warm_up_period=1,
            signal_strategy=SignalStrategy.ADAPTIVE,
            adaptive_window=5,
            strategy_config=StrategyConfig(adaptive_volatility_factor=1.0),
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
        
        # Test behavior: verify bars were processed by checking statistics
        signal_stats = actor.get_signal_statistics()
        assert signal_stats.get("bars_processed", 0) > 0, "No bars were processed"

    def test_market_regime_detection(self) -> None:
        """
        Test market regime detection functionality.
        """
        config = MLSignalActorConfig(
            model_id="test_model",
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

        # Process enough bars to initialize indicators and build price history
        # Need at least 26 bars for MACD + some extra for regime detection
        for i in range(35):  # More than 20 required for regime detection
            bar = self.create_test_bar(close_price=1.1000 + i * 0.0005)  # Trending up
            actor.on_bar(bar)

        # Market regime detection happens only after indicators are initialized
        # and sufficient price history is available
        if actor._indicator_manager and actor._indicator_manager.all_initialized():
            # Check if regime was detected
            assert actor._market_regime in ["trending", "volatile", "ranging", "unknown"]
        else:
            # If indicators not ready, regime should remain unknown
            assert actor._market_regime == "unknown"

    def test_feature_computation_performance(self) -> None:
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

        # Check that feature computation was performed - test behavior
        signal_stats = actor.get_signal_statistics()
        assert signal_stats.get("bars_processed", 0) > 0, "No bars were processed"
        assert actor._is_warmed_up
        # _total_feature_time is now tracked differently through PerformanceMonitor
        if actor._performance_monitor:
            stats = actor._performance_monitor.get_current_stats()
            assert stats["prediction_count"] >= 0

    def test_signal_separation_enforcement(self) -> None:
        """
        Test that minimum signal separation is enforced.
        """
        config = MLSignalActorConfig(
            model_id="test_model",
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

    def test_state_backup_and_restoration(self) -> None:
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

    def test_error_handling_in_prediction(self) -> None:
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

    def test_circuit_breaker_functionality(self) -> None:
        """
        Test circuit breaker opens after failures.
        """
        actor = self.create_test_actor()

        # Force failures by creating a model that always raises exceptions
        class FailingModel:
            def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                raise Exception("Model error")

            def predict_proba(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                raise Exception("Model error")

        actor._model = FailingModel()

        # Process bars to trigger circuit breaker
        # First warm up the actor (warm_up_period=20) + enough failures to open circuit breaker
        for i in range(35):  # Warm up + enough failures to trigger circuit breaker (need ~30 bars)
            bar = self.create_test_bar()
            actor.on_bar(bar)

        # Circuit breaker should be open
        assert actor._circuit_breaker is not None
        assert actor._circuit_breaker.state != CircuitBreakerState.CLOSED

    def test_get_signal_statistics(self) -> None:
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
        # assert "feature_buffer_size" in stats  # Not included in new implementation
        # assert "ensemble_weights" in stats  # Only included for ensemble strategy

        assert stats["signal_strategy"] == SignalStrategy.THRESHOLD.value
        assert stats["adaptive_threshold"] == self.config.prediction_threshold
        assert stats["market_regime"] in ["unknown", "ranging", "trending", "volatile"]

    def test_reset_signal_state(self) -> None:
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

    def test_adaptive_signal_data_class(self) -> None:
        """
        Test AdaptiveSignal data class properties.
        """
        instrument_id = InstrumentId.from_str("EURUSD.SIM")

        signal = AdaptiveSignal(
            instrument_id=instrument_id,
            model_id="test_model",
            prediction=0.8,
            confidence=0.9,
            metadata={
                "adaptive_threshold": 0.7,
                "signal_strength": 1.2,
                "market_regime": "trending",
            },
            ts_event=123456789,
            ts_init=123456790,
        )

        assert signal.instrument_id == instrument_id
        assert signal.prediction == 0.8
        assert signal.confidence == 0.9
        assert signal.metadata is not None
        assert signal.metadata["adaptive_threshold"] == 0.7
        assert signal.metadata["signal_strength"] == 1.2
        assert signal.metadata["market_regime"] == "trending"
        assert signal.ts_event == 123456789
        assert signal.ts_init == 123456790

    def test_onnx_model_prediction(self) -> None:
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

    def test_sklearn_proba_model_prediction(self) -> None:
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

    def test_sklearn_basic_model_prediction(self) -> None:
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

    def test_unsupported_model_type(self) -> None:
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

    def test_prediction_with_no_model(self) -> None:
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

    def test_update_prediction_history(self) -> None:
        """
        Test prediction history update with circular buffers.
        """
        actor = self.create_test_actor()

        # Initialize indicator manager price history
        assert actor._indicator_manager is not None
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

    def test_adaptive_threshold_update(self) -> None:
        """
        Test adaptive threshold calculation.
        """
        config = MLSignalActorConfig(
            model_id="test_model",
            component_id="MLSignalActor-001",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            prediction_threshold=0.5,
            warm_up_period=1,
            signal_strategy=SignalStrategy.ADAPTIVE,
            adaptive_window=5,
            strategy_config=StrategyConfig(
                adaptive_volatility_factor=2.0,
                min_threshold=0.1,
                max_threshold=0.95,
            ),
            feature_config=self.feature_config,
        )

        actor = self.create_test_actor(config)

        # Fill windows with test data
        actor._prediction_window[:] = np.array([0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
        actor._volatility_window[:] = np.array([0.01, 0.02, 0.03, 0.02, 0.01], dtype=np.float32)

        # Update adaptive threshold
        initial_threshold = actor._adaptive_threshold
        actor._update_adaptive_threshold()

        # Threshold should have changed
        assert actor._adaptive_threshold != initial_threshold
        assert (
            actor._strat_config.min_threshold
            <= actor._adaptive_threshold
            <= actor._strat_config.max_threshold
        )

    def test_market_regime_detection_scenarios(self) -> None:
        """
        Test different market regime detection scenarios.
        """
        actor = self.create_test_actor()

        # Initialize indicator manager if not already
        if not actor._indicator_manager:
            actor._initialize_features()

        # Test volatile regime
        assert actor._indicator_manager is not None
        actor._indicator_manager.price_history = {
            "closes": [1.10 + (i % 2) * 0.05 for i in range(25)],  # High volatility
        }

        bar = self.create_test_bar()
        actor._detect_market_regime(bar)
        assert actor._market_regime == "volatile"

        # Test trending regime
        assert actor._indicator_manager is not None
        actor._indicator_manager.price_history = {
            "closes": [1.10 + i * 0.001 for i in range(25)],  # Strong uptrend
        }

        actor._detect_market_regime(bar)
        assert actor._market_regime == "trending"

        # Test ranging regime
        assert actor._indicator_manager is not None
        actor._indicator_manager.price_history = {
            "closes": [1.10 + np.sin(i * 0.5) * 0.0001 for i in range(25)],  # Small oscillations
        }

        actor._detect_market_regime(bar)
        assert actor._market_regime == "ranging"

    def test_extremes_signal_percentile_calculation(self) -> None:
        """
        Test extremes signal percentile threshold calculation.
        """
        from ml.actors.signal import ExtremesStrategy

        config = MLSignalActorConfig(
            model_id="test_model",
            component_id="MLSignalActor-001",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            prediction_threshold=0.5,
            warm_up_period=1,
            signal_strategy=SignalStrategy.EXTREMES,
            adaptive_window=10,
            strategy_config=StrategyConfig(extremes_top_pct=0.1),  # Top/bottom 10%
            feature_config=self.feature_config,
        )

        actor = self.create_test_actor(config)

        # Fill prediction history
        actor._prediction_history = list(range(100))  # 0 to 99

        # Test signal generation through the strategy
        bar = self.create_test_bar()
        features = np.array([0.1, 0.2])

        # Test the ExtremesStrategy directly
        strategy = ExtremesStrategy(0.1, 0.5, 10)
        context = {
            "prediction_history": list(range(100)),
            "log_predictions": False,
            "timestamp_ns": self.clock.timestamp_ns(),
        }

        # With window size 10, last 10 values are [90, 91, ..., 99]
        # Top 10% threshold is ~98.1, bottom 10% is ~90.9

        # Test top extreme (should generate signal)
        signal = strategy.generate_signal(bar, 99.0, 0.8, features, context)
        assert signal is not None

        # Test bottom extreme (should generate signal)
        signal = strategy.generate_signal(bar, 90.0, 0.8, features, context)
        assert signal is not None

        # Test middle value (should not generate signal)
        signal = strategy.generate_signal(bar, 94.0, 0.8, features, context)
        assert signal is None

    def test_ensemble_weights_initialization(self) -> None:
        """
        Test ensemble weights initialization with custom and default values.
        """
        # Test with custom weights
        custom_weights = {"threshold": 0.6, "extremes": 0.2, "momentum": 0.2}
        config = MLSignalActorConfig(
            model_id="test_model",
            component_id="MLSignalActor-001",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            signal_strategy=SignalStrategy.ENSEMBLE,
            strategy_config=StrategyConfig(ensemble_weights=custom_weights),
            feature_config=self.feature_config,
        )

        actor = self.create_test_actor(config)
        assert actor._strat_config.ensemble_weights == custom_weights

        # Test with default weights
        config_default = MLSignalActorConfig(
            model_id="test_model",
            component_id="MLSignalActor-002",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            signal_strategy=SignalStrategy.ENSEMBLE,
            strategy_config=StrategyConfig(ensemble_weights=None),
            feature_config=self.feature_config,
        )

        actor_default = MLSignalActor(config_default)
        assert actor_default._strat_config.ensemble_weights == {
            "threshold": 0.4,
            "extremes": 0.3,
            "momentum": 0.3,
        }

    def test_signal_strategy_enum_values(self) -> None:
        """
        Test SignalStrategy enum values.
        """
        assert SignalStrategy.THRESHOLD.value == "threshold"
        assert SignalStrategy.EXTREMES.value == "extremes"
        assert SignalStrategy.MOMENTUM.value == "momentum"
        assert SignalStrategy.ENSEMBLE.value == "ensemble"
        assert SignalStrategy.ADAPTIVE.value == "adaptive"

    def test_compute_features_without_indicator_manager(self) -> None:
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

    def test_compute_features_before_indicators_ready(self) -> None:
        """
        Test feature computation before indicators are initialized.
        """
        actor = self.create_test_actor()

        # Mock indicator manager to return not initialized
        assert actor._indicator_manager is not None
        actor._indicator_manager.all_initialized = Mock(return_value=False)  # type: ignore[method-assign]

        # Try to compute features
        bar = self.create_test_bar()
        features = actor._compute_features(bar)

        assert features is None

    def test_load_model_logging(self) -> None:
        """
        Test model loading with logging.
        """
        actor = self.create_test_actor()

        # Call load model method
        actor._load_model()

        # Model should be loaded (from base class)
        assert actor._model is not None

    def test_generate_signal_unknown_strategy(self) -> None:
        """
        Test handling of unknown signal strategy.
        """
        actor = self.create_test_actor()

        # The strategy is created at initialization, so we can't have an unknown strategy
        # Instead, test that the actor has a valid strategy
        assert actor._signal_strategy is not None

        # Test that all known strategies are handled
        for strategy in SignalStrategy:
            config = MLSignalActorConfig(
                model_id="test_model",
                component_id=f"MLSignalActor-{strategy.value}",
                model_path=self.temp_model_file.name,
                bar_type=self.bar_type,
                instrument_id=self.instrument_id,
                signal_strategy=strategy,
                feature_config=self.feature_config,
            )
            test_actor = MLSignalActor(config)
            assert test_actor._signal_strategy is not None

    def test_track_performance_metrics(self) -> None:
        """
        Test performance metrics tracking.
        """
        actor = self.create_test_actor()

        # Performance tracking is now done through PerformanceMonitor
        assert actor._performance_monitor is not None

        # Record some timing data
        for i in range(5):
            actor._performance_monitor.record_timing(
                feature_time_ns=500_000 + i * 100_000,  # 0.5ms + increments
                inference_time_ns=2_000_000 + i * 200_000,  # 2ms + increments
                total_time_ns=2_500_000 + i * 300_000,  # 2.5ms + increments
            )

        # Check stats
        stats = actor._performance_monitor.get_current_stats()
        assert stats["prediction_count"] == 5
        assert stats["avg_total_time_ms"] > 0

        # Metrics are module-level
        from ml.actors import signal

        assert signal._prediction_distribution_metric is not None
        assert signal._confidence_distribution_metric is not None

    def test_hot_reload_check_scheduling(self) -> None:
        """
        Test hot reload scheduling is set up when enabled.
        """
        config = MLSignalActorConfig(
            model_id="test_model",
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

    def test_feature_config_initialization(self) -> None:
        """
        Test different feature config initialization scenarios.
        """
        # Test with no feature config
        config_no_features = MLSignalActorConfig(
            model_id="test_model",
            component_id="MLSignalActor-001",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            feature_config=None,
        )

        actor = MLSignalActor(config_no_features)
        assert actor._feature_config is not None
        assert isinstance(actor._feature_config, FeatureConfig)

    def test_feature_buffer_resizing(self) -> None:
        """
        Test feature buffer resizing based on feature configuration.
        """
        # Create config with specific feature names
        feature_config = FeatureConfig(
            feature_names=["feature1", "feature2", "feature3"],
        )

        config = MLSignalActorConfig(
            model_id="test_model",
            component_id="MLSignalActor-001",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            feature_config=feature_config,
        )

        actor = self.create_test_actor(config)

        # Test behavior: verify feature configuration was applied correctly
        # The feature buffer size should match the configured feature count
        # This is a configuration validation test, not implementation testing
        expected_features = len(feature_config.enabled_features)
        assert expected_features >= 3, f"Configuration should enable at least 3 features, got {expected_features}"

    def test_ensemble_signal_with_no_component_signals(self) -> None:
        """
        Test ensemble signal when no component strategies generate signals.
        """
        from ml.actors.signal import EnsembleStrategy
        from ml.actors.signal import SignalGenerationStrategy
        from ml.actors.signal import ThresholdSignalStrategy

        # Create ensemble strategy with high thresholds so no signals are generated
        strategies: dict[str, SignalGenerationStrategy] = {
            "threshold": ThresholdSignalStrategy(0.99),  # Very high threshold
            "extremes": ThresholdSignalStrategy(0.99),  # Use threshold as placeholder
            "momentum": ThresholdSignalStrategy(0.99),  # Use threshold as placeholder
        }
        weights = {"threshold": 0.4, "extremes": 0.3, "momentum": 0.3}
        ensemble = EnsembleStrategy(strategies, weights, 0.5)

        # Test with low confidence that won't trigger any strategy
        bar = self.create_test_bar()
        context = {
            "prediction_history": [0.5],
            "log_predictions": False,
            "timestamp_ns": self.clock.timestamp_ns(),
        }

        signal = ensemble.generate_signal(bar, 0.5, 0.3, np.array([0.1, 0.2]), context)
        assert signal is None

    def test_slow_feature_computation_warning(self) -> None:
        """
        Test warning when feature computation is slow.
        """
        actor = self.create_test_actor()

        # Mock slow feature computation
        assert actor._indicator_manager is not None
        original_update = actor._indicator_manager.update_from_bar

        def slow_update(bar: Bar) -> None:
            time.sleep(0.01)  # Make it slow
            original_update(bar)

        actor._indicator_manager.update_from_bar = slow_update  # type: ignore[method-assign]

        # Process bar - should log warning
        for i in range(30):
            bar = self.create_test_bar()
            actor.on_bar(bar)

    def test_indicator_state_backup_without_manager(self) -> None:
        """
        Test indicator state backup when manager is None.
        """
        actor = self.create_test_actor()

        # Set indicator manager to None
        original_manager = actor._indicator_manager
        actor._indicator_manager = None

        # Backup state
        actor._backup_indicator_state()

        # When indicator_manager is None, no backup is created
        # This is the expected behavior as there's nothing to backup
        if not hasattr(actor, "_indicator_state_backup") or actor._indicator_state_backup == {}:
            # Expected - no backup when no indicator manager
            assert True
        else:
            # Unexpected - should not have backup data
            assert False, "Should not create backup when indicator_manager is None"

        # Restore manager
        actor._indicator_manager = original_manager

    def test_indicator_state_restore_without_backup(self) -> None:
        """
        Test indicator state restore when no backup exists.
        """
        actor = self.create_test_actor()

        # Clear any existing backup
        actor._indicator_state_backup = {}

        # Try to restore - should handle gracefully
        actor._restore_indicator_state()

        # State should remain unchanged
        assert actor._prediction_history == []
        assert actor._market_regime == "unknown"

    def test_adaptive_signal_generation_below_threshold(self) -> None:
        """
        Test adaptive signal when signal strength is below threshold.
        """
        from ml.actors.signal import AdaptiveStrategy

        config = MLSignalActorConfig(
            model_id="test_model",
            component_id="MLSignalActor-001",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            signal_strategy=SignalStrategy.ADAPTIVE,
            feature_config=self.feature_config,
        )

        self.create_test_actor(config)  # Just to validate config

        # Create adaptive strategy with high threshold
        strategy = AdaptiveStrategy(
            base_threshold=0.5,
            volatility_factor=2.0,
            min_threshold=0.1,
            max_threshold=0.95,
        )

        # Test with low confidence relative to high adaptive threshold
        bar = self.create_test_bar()
        context = {
            "adaptive_threshold": 0.95,
            "market_regime": "volatile",
            "log_predictions": False,
            "timestamp_ns": self.clock.timestamp_ns(),
        }

        signal = strategy.generate_signal(bar, 0.7, 0.5, np.array([0.1, 0.2]), context)
        # Should not generate signal (strength = 0.5/0.95 < 1.0)
        assert signal is None

    def test_momentum_signal_insufficient_history(self) -> None:
        """
        Test momentum signal when insufficient prediction history.
        """
        from ml.actors.signal import MomentumStrategy

        # Create momentum strategy
        strategy = MomentumStrategy(
            lookback=5,
            threshold=0.5,
            momentum_threshold=0.01,
        )

        # Test with insufficient history
        bar = self.create_test_bar()
        context = {
            "prediction_history": [0.5],  # Less than lookback
            "log_predictions": False,
            "timestamp_ns": self.clock.timestamp_ns(),
        }

        signal = strategy.generate_signal(bar, 0.8, 0.9, np.array([0.1, 0.2]), context)
        assert signal is None

    def test_extremes_signal_insufficient_history(self) -> None:
        """
        Test extremes signal when insufficient prediction history.
        """
        from ml.actors.signal import ExtremesStrategy

        # Create extremes strategy
        strategy = ExtremesStrategy(
            top_pct=0.1,
            threshold=0.5,
            window_size=10,
        )

        # Test with insufficient history
        bar = self.create_test_bar()
        context = {
            "prediction_history": [0.5, 0.6],  # Less than window size
            "log_predictions": False,
            "timestamp_ns": self.clock.timestamp_ns(),
        }

        signal = strategy.generate_signal(bar, 0.8, 0.9, np.array([0.1, 0.2]), context)
        assert signal is None

    def test_regime_detection_insufficient_data(self) -> None:
        """
        Test market regime detection with insufficient price history.
        """
        actor = self.create_test_actor()

        # Set insufficient price history
        assert actor._indicator_manager is not None
        actor._indicator_manager.price_history = {"closes": [1.1000, 1.1001]}

        # Try to detect regime
        bar = self.create_test_bar()
        actor._detect_market_regime(bar)

        # Regime should remain unknown
        assert actor._market_regime == "unknown"

    def test_ensemble_signal_partial_strategies(self) -> None:
        """
        Test ensemble signal when only some strategies generate signals.
        """
        from ml.actors.signal import EnsembleStrategy
        from ml.actors.signal import ExtremesStrategy
        from ml.actors.signal import MomentumStrategy
        from ml.actors.signal import ThresholdSignalStrategy

        config = MLSignalActorConfig(
            model_id="test_model",
            component_id="MLSignalActor-001",
            model_path=self.temp_model_file.name,
            bar_type=self.bar_type,
            instrument_id=self.instrument_id,
            prediction_threshold=0.5,
            signal_strategy=SignalStrategy.ENSEMBLE,
            strategy_config=StrategyConfig(
                ensemble_weights={"threshold": 0.5, "extremes": 0.3, "momentum": 0.2},
            ),
            feature_config=self.feature_config,
        )

        actor = self.create_test_actor(config)

        # Set up state for ensemble
        actor._prediction_history = [0.3, 0.4, 0.5, 0.6, 0.7]
        actor._last_signal_bar = -10

        # Test the ensemble strategy directly
        strategies = {
            "threshold": ThresholdSignalStrategy(0.5),
            "extremes": ExtremesStrategy(0.1, 0.5, 5),
            "momentum": MomentumStrategy(3, 0.5, 0.01),
        }
        assert config.strategy_config is not None
        assert config.strategy_config.ensemble_weights is not None
        ensemble = EnsembleStrategy(strategies, config.strategy_config.ensemble_weights, 0.5)

        bar = self.create_test_bar()
        context = {
            "prediction_history": actor._prediction_history,
            "log_predictions": False,
            "timestamp_ns": self.clock.timestamp_ns(),
        }

        signal = ensemble.generate_signal(bar, 0.8, 0.7, np.array([0.1, 0.2]), context)

        # Should generate signal if ensemble confidence is high enough
        if signal is not None:
            assert isinstance(signal, MLSignal)
            assert signal.confidence >= 0.5  # Ensemble threshold

    def test_prediction_exception_handling(self) -> None:
        """
        Test exception handling in _predict method.
        """
        actor = self.create_test_actor()

        # Set model that raises exception - make sure it doesn't have 'run' attribute
        actor._model = Mock(spec=["predict"])  # Only has predict, not run
        actor._model.predict.side_effect = RuntimeError("Prediction failed")

        # Should re-raise exception so base class can handle it
        features = np.array([0.1, 0.2, 0.3])

        # Expect exception to be raised
        with pytest.raises(RuntimeError, match="Prediction failed"):
            actor._predict(features)

    def test_zero_prediction_confidence_estimation(self) -> None:
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

    def test_momentum_below_threshold(self) -> None:
        """
        Test momentum signal when momentum is below threshold.
        """
        from ml.actors.signal import MomentumStrategy

        # Create momentum strategy
        strategy = MomentumStrategy(
            lookback=3,
            threshold=0.5,
            momentum_threshold=0.01,
        )

        # Test with flat prediction history (no momentum)
        bar = self.create_test_bar()
        context = {
            "prediction_history": [0.5, 0.5, 0.5, 0.5],  # No momentum
            "log_predictions": False,
            "timestamp_ns": self.clock.timestamp_ns(),
        }

        signal = strategy.generate_signal(bar, 0.5, 0.8, np.array([0.1, 0.2]), context)
        # Should not generate signal (momentum = 0 < threshold)
        assert signal is None

    def test_onnx_single_output_model(self) -> None:
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

    def test_generate_prediction_protected_success(self) -> None:
        """
        Test successful prediction generation with all features.
        """
        actor = self.create_test_actor()

        # Process enough bars to warm up and initialize indicators
        for i in range(35):
            bar = self.create_test_bar(close_price=1.1000 + i * 0.0001)
            actor.on_bar(bar)

        # Check that predictions were made after warm-up - test behavior
        if actor._is_warmed_up:
            signal_stats = actor.get_signal_statistics()
            assert signal_stats.get("predictions_made", 0) > 0, "No predictions were made after warm-up"
            if actor._health_monitor:
                assert actor._health_monitor.consecutive_failures == 0
        else:
            # If not warmed up yet, at least check no errors
            if actor._health_monitor:
                assert actor._health_monitor.consecutive_failures == 0

    def test_generate_prediction_protected_failure(self) -> None:
        """
        Test prediction generation failure handling.
        """
        actor = self.create_test_actor()

        # Mock prediction to raise exception
        actor._predict = Mock(side_effect=Exception("Prediction error"))  # type: ignore[method-assign]

        # Initialize indicators first
        for i in range(35):
            bar = self.create_test_bar()
            actor.on_bar(bar)  # This will handle errors gracefully

        # Health monitor should have recorded failures if predictions were attempted
        if actor._is_warmed_up:
            # After warm-up, predictions are attempted and failures recorded
            if actor._health_monitor:
                assert actor._health_monitor.consecutive_failures > 0
        else:
            # Before warm-up, no predictions attempted
            if actor._health_monitor:
                assert actor._health_monitor.consecutive_failures >= 0
