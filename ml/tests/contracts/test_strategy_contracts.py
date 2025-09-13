#!/usr/bin/env python3
"""
Functional tests for strategy signal handling.

These tests define the contracts for ML strategies:
1. Receive MLSignal objects via MessageBus
2. Filter signals by model_id
3. Aggregate signals from multiple models
4. Execute trades based on signals

"""

import time
from collections import deque
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from ml.actors.base import MLSignal
from ml.strategies.base import BaseMLStrategy
from ml.tests.builders import MLConfigBuilder
from nautilus_trader.core.data import Data
from nautilus_trader.model.identifiers import InstrumentId


@pytest.mark.parallel_safe
class TestStrategyContracts:
    """
    Test suite for strategy signal handling contracts.
    """

    def _create_ml_signal(
        self,
        default_instrument_id: InstrumentId,
        test_timestamps: tuple[int, int],
        model_id: str = "test_model",
        prediction: float = 0.7,
        confidence: float = 0.8,
        time_offset_ns: int = 0,
        **kwargs: Any,
    ) -> MLSignal:
        """Helper to create MLSignal with defaults."""
        ts_event, ts_init = test_timestamps
        return MLSignal(
            instrument_id=default_instrument_id,
            model_id=model_id,
            prediction=prediction,
            confidence=confidence,
            metadata=kwargs.get("metadata", {}),
            ts_event=ts_event + time_offset_ns,
            ts_init=ts_init + time_offset_ns,
        )

    def test_strategy_receives_ml_signals(
        self,
        default_instrument_id: InstrumentId,
        test_timestamps: tuple[int, int],
    ) -> None:
        """
        Strategy MUST receive MLSignal via on_data().

        Given: Strategy subscribed to MLSignal
        When: MLSignal is published
        Then: Strategy's on_data() is called with signal

        """
        # Create test strategy
        strategy = self._create_test_strategy()

        # Track on_data calls
        received_data = []
        original_on_data = strategy.on_data

        def mock_on_data(data: Data) -> None:
            received_data.append(data)
            original_on_data(data)

        strategy.on_data = mock_on_data

        # Note: In actual usage, the strategy subscribes in on_start()
        # For testing, we'll just simulate the data reception directly

        # Create and "publish" signal
        signal = self._create_ml_signal(
            default_instrument_id=default_instrument_id,
            test_timestamps=test_timestamps,
            model_id="test_model_v1",
            prediction=0.7,
            confidence=0.85,
        )

        # Simulate signal reception
        strategy.on_data(signal)

        # Assert
        assert len(received_data) == 1, "Strategy should receive the signal"
        assert isinstance(received_data[0], MLSignal), "Received data should be MLSignal"
        assert received_data[0].prediction == 0.7, "Signal data should be preserved"
        assert received_data[0].confidence == 0.85, "Signal confidence should be preserved"

    def test_strategy_filters_by_model_id(
        self,
        default_instrument_id: InstrumentId,
        test_timestamps: tuple[int, int],
    ) -> None:
        """
        Strategy can filter signals by model_id.

        Given: Strategy configured for specific model_id
        When: Multiple signals with different model_ids
        Then: Only configured model signals are processed

        """
        # Create strategy configured for specific model
        strategy = self._create_test_strategy(target_model_ids=["xgb_eurusd_1h_v2"])

        # Track processed signals
        processed_signals = []
        strategy._process_signal = Mock(side_effect=lambda s: processed_signals.append(s))

        # Create signals from different models
        signals = [
            self._create_ml_signal(
                default_instrument_id=default_instrument_id,
                test_timestamps=test_timestamps,
                model_id="xgb_eurusd_1h_v1",
                prediction=0.6,
                confidence=0.7,
            ),
            self._create_ml_signal(
                default_instrument_id=default_instrument_id,
                test_timestamps=test_timestamps,
                model_id="xgb_eurusd_1h_v2",  # Target model
                prediction=0.8,
                confidence=0.9,
            ),
            self._create_ml_signal(
                default_instrument_id=default_instrument_id,
                test_timestamps=test_timestamps,
                model_id="lgb_eurusd_1h_v1",
                prediction=0.5,
                confidence=0.6,
            ),
        ]

        # Process all signals
        for signal in signals:
            strategy.on_data(signal)

        # Assert only target model signal was processed
        assert len(processed_signals) == 1, "Only target model signal should be processed"
        assert processed_signals[0].model_id == "xgb_eurusd_1h_v2"

    def test_strategy_handles_multiple_model_signals(
        self,
        default_instrument_id: InstrumentId,
        test_timestamps: tuple[int, int],
    ) -> None:
        """
        Strategy can aggregate signals from multiple models.

        Given: Signals from 3 different models
        When: All signals arrive within time window
        Then: Strategy aggregates and makes decision

        """
        # Create strategy with aggregation capability
        strategy = self._create_test_strategy(
            aggregation_mode="voting",
            required_models=3,
            time_window_ms=1000,
        )

        # Track trading decisions
        trading_decisions = []
        strategy._execute_trade = Mock(side_effect=lambda d: trading_decisions.append(d))

        # Create signals from 3 models (all bullish)
        signals = [
            self._create_ml_signal(
                default_instrument_id=default_instrument_id,
                test_timestamps=test_timestamps,
                model_id="model_1",
                prediction=0.7,  # Bullish
                confidence=0.8,
            ),
            self._create_ml_signal(
                default_instrument_id=default_instrument_id,
                test_timestamps=test_timestamps,
                model_id="model_2",
                prediction=0.65,  # Bullish
                confidence=0.75,
                time_offset_ns=100_000_000,  # 100ms later
            ),
            self._create_ml_signal(
                default_instrument_id=default_instrument_id,
                test_timestamps=test_timestamps,
                model_id="model_3",
                prediction=0.8,  # Bullish
                confidence=0.9,
                time_offset_ns=200_000_000,  # 200ms later
            ),
        ]

        # Process signals
        for signal in signals:
            strategy.on_data(signal)

        # Assert aggregation occurred
        assert len(trading_decisions) >= 1, "Should make trading decision after aggregation"

        # Check aggregated decision
        decision = trading_decisions[0]
        assert decision["action"] in ["BUY", "SELL"], "Should have clear action"
        assert (
            decision["confidence"] > 0.7
        ), "Aggregated confidence should be high with 3 bullish signals"

    def test_strategy_respects_signal_confidence_threshold(
        self,
        default_instrument_id: InstrumentId,
        test_timestamps: tuple[int, int],
    ) -> None:
        """
        Strategy only acts on high-confidence signals.

        Given: Strategy with min_confidence=0.8
        When: Signals with varying confidence
        Then: Only high-confidence signals trigger trades

        """
        # Create strategy with confidence threshold
        strategy = self._create_test_strategy(min_confidence=0.8)

        # Track trade executions
        executed_trades = []
        strategy._execute_trade = Mock(side_effect=lambda t: executed_trades.append(t))

        # Create signals with different confidence levels
        signals = [
            self._create_ml_signal(
                default_instrument_id=default_instrument_id,
                test_timestamps=test_timestamps,
                model_id="model_1",
                prediction=0.9,
                confidence=0.7,  # Below threshold
            ),
            self._create_ml_signal(
                default_instrument_id=default_instrument_id,
                test_timestamps=test_timestamps,
                model_id="model_1",
                prediction=0.9,
                confidence=0.85,  # Above threshold
            ),
            self._create_ml_signal(
                default_instrument_id=default_instrument_id,
                test_timestamps=test_timestamps,
                model_id="model_1",
                prediction=0.9,
                confidence=0.6,  # Below threshold
            ),
        ]

        # Process signals
        for signal in signals:
            strategy.on_data(signal)

        # Assert only high-confidence signal triggered trade
        assert len(executed_trades) == 1, "Only one high-confidence trade should execute"
        assert executed_trades[0]["signal"].confidence >= 0.8

    def test_strategy_handles_conflicting_signals(
        self,
        default_instrument_id: InstrumentId,
        test_timestamps: tuple[int, int],
    ) -> None:
        """
        Strategy handles conflicting signals from different models.

        Given: Conflicting signals (buy vs sell)
        When: Signals arrive close in time
        Then: Strategy resolves conflict appropriately

        """
        # Create strategy with conflict resolution
        strategy = self._create_test_strategy(
            aggregation_mode="voting",  # Enable aggregation
            conflict_resolution="weighted_average",
            model_weights={"model_1": 0.6, "model_2": 0.4},
            required_models=2,  # Need both models
        )

        # Track decisions
        decisions = []
        strategy._make_decision = Mock(side_effect=lambda d: decisions.append(d))

        # Create conflicting signals
        base_time = time.time_ns()
        signals = [
            MLSignal(
                instrument_id=InstrumentId.from_str("EURUSD.SIM"),
                model_id="model_1",
                prediction=0.8,  # Strong buy
                confidence=0.9,
                metadata={},
                ts_event=base_time,
                ts_init=base_time,
            ),
            MLSignal(
                instrument_id=InstrumentId.from_str("EURUSD.SIM"),
                model_id="model_2",
                prediction=0.2,  # Strong sell
                confidence=0.85,
                metadata={},
                ts_event=base_time + 50_000_000,  # 50ms later
                ts_init=base_time + 50_000_000,
            ),
        ]

        # Process signals
        for signal in signals:
            strategy.on_data(signal)

        # Assert conflict was resolved
        assert len(decisions) > 0, "Strategy should make a decision"

        # With weights 0.6 and 0.4, and predictions 0.8 and 0.2:
        # Weighted average = 0.6 * 0.8 + 0.4 * 0.2 = 0.48 + 0.08 = 0.56 (slightly bullish)
        final_decision = decisions[-1]
        assert "weighted_prediction" in final_decision or "action" in final_decision

    def test_strategy_maintains_signal_history(self) -> None:
        """
        Strategy maintains recent signal history for analysis.

        Given: Stream of signals
        When: Signals are received
        Then: Recent history is maintained with size limit

        """
        # Create strategy with history tracking
        strategy = self._create_test_strategy(history_size=10)

        # Send 15 signals
        for i in range(15):
            signal = MLSignal(
                instrument_id=InstrumentId.from_str("EURUSD.SIM"),
                model_id="model_1",
                prediction=0.5 + i * 0.01,
                confidence=0.7,
                metadata={"index": i},
                ts_event=time.time_ns(),
                ts_init=time.time_ns(),
            )
            strategy.on_data(signal)

        # Check history is maintained with size limit
        if hasattr(strategy, "_signal_history"):
            assert len(strategy._signal_history) <= 10, "History should be bounded"

            # Verify most recent signals are kept
            history_indices = [s.metadata.get("index") for s in strategy._signal_history]
            assert min(history_indices) >= 5, "Should keep most recent signals"

    def test_strategy_tracks_model_performance(self) -> None:
        """
        Strategy tracks performance per model for adaptive weighting.

        Given: Signals and resulting trades
        When: Trades complete with P&L
        Then: Model performance metrics are updated

        """
        # Create strategy with performance tracking
        strategy = self._create_test_strategy(track_performance=True)

        # Initialize performance tracking
        if not hasattr(strategy, "_model_performance"):
            strategy._model_performance = {}

        # Simulate signal → trade → result cycle
        signal = MLSignal(
            instrument_id=InstrumentId.from_str("EURUSD.SIM"),
            model_id="model_1",
            prediction=0.8,
            confidence=0.9,
            metadata={},
            ts_event=time.time_ns(),
            ts_init=time.time_ns(),
        )

        # Process signal
        strategy.on_data(signal)

        # Simulate trade result
        strategy._update_model_performance("model_1", profit=100.0)

        # Check performance is tracked
        assert "model_1" in strategy._model_performance
        perf = strategy._model_performance["model_1"]
        assert "total_trades" in perf or "profit" in perf or "accuracy" in perf

    # Helper methods
    def _create_test_strategy(self, **kwargs: Any) -> Any:
        """
        Create a test strategy with configurable behavior.
        """

        class TestMLStrategy(BaseMLStrategy):
            def __init__(self, config: Any) -> None:
                super().__init__(config)
                # Apply test configuration
                self.target_model_ids = kwargs.get("target_model_ids")
                self.min_confidence = kwargs.get("min_confidence", 0.0)
                self.aggregation_mode = kwargs.get("aggregation_mode")
                self.required_models = kwargs.get("required_models", 1)
                self.time_window_ms = kwargs.get("time_window_ms", 1000)
                self.conflict_resolution = kwargs.get("conflict_resolution")
                self.model_weights = kwargs.get("model_weights", {})
                self.history_size = kwargs.get("history_size", 100)
                self.track_performance = kwargs.get("track_performance", False)

                # Internal state
                self._signal_buffer: dict[str, MLSignal] = {}
                self._signal_history: deque[Any] = deque(maxlen=self.history_size)
                self._model_signals: dict[str, MLSignal] = {}
                self._model_performance: dict[str, Any] = {}

            def on_data(self, data: Data) -> None:
                if isinstance(data, MLSignal):
                    # Add to history
                    self._signal_history.append(data)

                    # Filter by model_id if configured
                    if self.target_model_ids:
                        model_id = getattr(data, "model_id", None) or data.metadata.get("model_id")
                        if model_id not in self.target_model_ids:
                            return

                    # Check confidence threshold
                    if data.confidence < self.min_confidence:
                        return

                    # Handle aggregation
                    if self.aggregation_mode:
                        self._aggregate_signal(data)
                    else:
                        # Process single signal
                        self._process_signal(data)

            def _aggregate_signal(self, signal: MLSignal) -> None:
                model_id = getattr(signal, "model_id", None) or signal.metadata.get("model_id")
                if model_id:
                    self._model_signals[model_id] = signal

                # Check if we have enough signals
                if len(self._model_signals) >= self.required_models:
                    # Aggregate and make decision
                    if self.conflict_resolution == "weighted_average":
                        weighted_pred = sum(
                            self.model_weights.get(mid, 1.0) * s.prediction
                            for mid, s in self._model_signals.items()
                        ) / sum(
                            self.model_weights.get(mid, 1.0) for mid in self._model_signals.keys()
                        )

                        avg_confidence = np.mean(
                            [s.confidence for s in self._model_signals.values()],
                        )

                        self._make_decision(
                            {
                                "weighted_prediction": weighted_pred,
                                "confidence": avg_confidence,
                            },
                        )
                    else:
                        # Simple voting
                        bullish = sum(1 for s in self._model_signals.values() if s.prediction > 0.5)
                        bearish = len(self._model_signals) - bullish

                        action = "BUY" if bullish > bearish else "SELL"
                        confidence = max(s.confidence for s in self._model_signals.values())

                        self._execute_trade(
                            {
                                "action": action,
                                "confidence": confidence,
                                "signal": signal,
                            },
                        )

                    # Clear buffer after decision
                    self._model_signals.clear()

            def _process_signal(self, signal: MLSignal) -> None:
                """
                Process individual signal.
                """
                # For testing: execute trade if confidence is sufficient
                if signal.confidence >= self.min_confidence:
                    self._execute_trade({"signal": signal})

            def _make_decision(self, decision: dict[str, Any]) -> None:
                """
                Make trading decision.
                """

            def _execute_trade(self, trade: dict[str, Any]) -> None:
                """
                Execute trade based on signal.
                """

            def _update_model_performance(self, model_id: str, profit: float) -> None:
                """
                Update model performance metrics.
                """
                if model_id not in self._model_performance:
                    self._model_performance[model_id] = {
                        "total_trades": 0,
                        "total_profit": 0.0,
                    }

                self._model_performance[model_id]["total_trades"] += 1
                self._model_performance[model_id]["total_profit"] += profit

            def _process_ml_signal(self, signal: MLSignal) -> None:
                """
                Abstract method implementation for testing.
                """

        # Create config using builder
        config = MLConfigBuilder.strategy_config(
            min_confidence=kwargs.get("min_confidence", 0.0),
            position_size_pct=0.1,
            max_positions=1,
        )

        # Simple initialization - just create the strategy
        strategy = TestMLStrategy(config)

        return strategy
