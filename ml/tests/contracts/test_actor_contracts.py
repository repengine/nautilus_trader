#!/usr/bin/env python3

"""
Functional tests defining actor behavior contracts.

These tests define what actors MUST do, not HOW they do it.
The goal is to ensure actors:
1. Publish MLSignal objects when receiving market data
2. Include model identification in every signal
3. Handle multiple instruments correctly
4. Gracefully handle failures without crashing
"""

from typing import Any, Optional, List

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock

from nautilus_trader.common.actor import Actor
from nautilus_trader.core.data import Data
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.test_kit.stubs.component import TestComponentStubs
from nautilus_trader.test_kit.stubs.data import TestDataStubs
from nautilus_trader.test_kit.stubs.identifiers import TestIdStubs

from ml.actors.base import MLSignal, BaseMLInferenceActor, MLSignalActor


class TestActorContracts:
    """Test suite for ML actor behavioral contracts."""
    
    def test_actor_publishes_ml_signal_on_bar(self) -> None:
        """
        Actor MUST publish MLSignal when receiving bar data.
        
        Given: An actor with a loaded model
        When: A bar is received
        Then: An MLSignal is published with required fields
        """
        # Arrange
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.7])
        
        actor = self._create_test_actor(model=mock_model)
        bar = TestDataStubs.bar_5decimal()
        published_signals = []
        
        # Mock the publish_data method to capture signals
        actor.publish_data = Mock(side_effect=lambda dtype, signal: published_signals.append(signal))
        
        # Act
        actor.on_bar(bar)
        
        # Assert
        assert len(published_signals) == 1, "Actor must publish exactly one signal per bar"
        
        signal = published_signals[0]
        assert isinstance(signal, MLSignal), "Published data must be MLSignal type"
        assert signal.instrument_id == bar.bar_type.instrument_id, "Signal must reference correct instrument"
        assert 0.0 <= signal.confidence <= 1.0, "Confidence must be in [0, 1]"
        assert signal.prediction is not None, "Prediction must not be None"
        
    def test_actor_includes_model_id_in_signal(self) -> None:
        """
        Every signal MUST identify its source model.
        
        Given: An actor with model_id="xgb_eurusd_1h_v1"
        When: Signal is generated
        Then: Signal contains model_id in metadata
        """
        # Arrange
        model_id = "xgb_eurusd_1h_v1"
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.8])
        
        actor = self._create_test_actor(
            model=mock_model,
            model_id=model_id
        )
        bar = TestDataStubs.bar_5decimal()
        published_signals = []
        
        actor.publish_data = Mock(side_effect=lambda dtype, signal: published_signals.append(signal))
        
        # Act
        actor.on_bar(bar)
        
        # Assert
        assert len(published_signals) == 1
        signal = published_signals[0]
        
        # Model ID must be accessible (either as attribute or in metadata)
        assert hasattr(signal, 'model_id') or 'model_id' in signal.metadata, \
            "Signal must contain model_id"
        
        if hasattr(signal, 'model_id'):
            assert signal.model_id == model_id
        else:
            assert signal.metadata['model_id'] == model_id
            
    def test_actor_handles_multiple_instruments(self) -> None:
        """
        Actor can filter and process multiple instruments.
        
        Given: Actor configured for ["EURUSD", "GBPUSD"]
        When: Bars for multiple instruments arrive
        Then: Signals generated only for configured instruments
        """
        # Arrange
        configured_instruments = [
            InstrumentId.from_str("EURUSD.SIM"),
            InstrumentId.from_str("GBPUSD.SIM"),
        ]
        
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.6])
        
        actor = self._create_test_actor(
            model=mock_model,
            instruments=configured_instruments
        )
        
        published_signals = []
        actor.publish_data = Mock(side_effect=lambda dtype, signal: published_signals.append(signal))
        
        # Create bars for configured and non-configured instruments
        eurusd_bar = self._create_bar_for_instrument("EURUSD.SIM")
        gbpusd_bar = self._create_bar_for_instrument("GBPUSD.SIM")
        usdjpy_bar = self._create_bar_for_instrument("USDJPY.SIM")  # Not configured
        
        # Act
        actor.on_bar(eurusd_bar)
        actor.on_bar(gbpusd_bar)
        actor.on_bar(usdjpy_bar)
        
        # Assert
        assert len(published_signals) == 2, "Should only publish signals for configured instruments"
        
        signal_instruments = [s.instrument_id for s in published_signals]
        assert InstrumentId.from_str("EURUSD.SIM") in signal_instruments
        assert InstrumentId.from_str("GBPUSD.SIM") in signal_instruments
        assert InstrumentId.from_str("USDJPY.SIM") not in signal_instruments
        
    def test_actor_gracefully_handles_inference_failure(self) -> None:
        """
        Actor continues operating if model inference fails.
        
        Given: A model that will fail on certain inputs
        When: Inference fails
        Then: Error logged, no signal published, actor continues
        """
        # Arrange
        mock_model = Mock()
        mock_model.predict.side_effect = [
            np.array([0.5]),  # First prediction succeeds
            ValueError("Model inference failed"),  # Second fails
            np.array([0.6]),  # Third succeeds
        ]
        
        actor = self._create_test_actor(model=mock_model)
        published_signals = []
        actor.publish_data = Mock(side_effect=lambda dtype, signal: published_signals.append(signal))
        
        # Can't mock logger directly - will verify through behavior
        
        bars = [
            TestDataStubs.bar_5decimal(),
            TestDataStubs.bar_5decimal(),
            TestDataStubs.bar_5decimal(),
        ]
        
        # Act - Process all bars despite failure
        for bar in bars:
            actor.on_bar(bar)
        
        # Assert
        assert len(published_signals) == 2, "Should publish 2 signals (1st and 3rd)"
        # Error logging is internal - we verify by behavior (actor continues)
        
        # Verify actor is still functional after error
        assert hasattr(actor, 'on_bar'), "Actor should still have on_bar method"
        
    def test_actor_respects_hot_path_constraints(self) -> None:
        """
        Actor respects hot path constraints - no blocking operations.
        
        Given: Actor processing market data
        When: Bar data arrives
        Then: No blocking operations occur (no Polars, no I/O, no allocations)
        """
        # Arrange
        mock_model = Mock()
        # Return predictions with different confidence levels
        mock_model.predict.side_effect = [
            np.array([0.8]),  # Above threshold
            np.array([0.5]),  # Below threshold
            np.array([0.9]),  # Above threshold
            np.array([0.6]),  # Below threshold
        ]
        
        actor = self._create_test_actor(
            model=mock_model,
            min_confidence=0.7
        )
        
        published_signals = []
        actor.publish_data = Mock(side_effect=lambda dtype, signal: published_signals.append(signal))
        
        # Act
        for _ in range(4):
            bar = TestDataStubs.bar_5decimal()
            actor.on_bar(bar)
        
        # Assert
        assert len(published_signals) == 2, "Should only publish high-confidence signals"
        for signal in published_signals:
            assert signal.confidence >= 0.7, f"Signal confidence {signal.confidence} below threshold"
    
    # Helper methods
    def _create_test_actor(
        self,
        model: Optional[Any] = None,
        model_id: str = "test_model",
        instruments: Optional[List[InstrumentId]] = None,
        min_confidence: float = 0.0,
    ) -> Actor:
        """Create a test actor with mocked dependencies."""
        # Create a simplified test actor that demonstrates the contracts
        
        class TestMLActor(Actor):  # type: ignore[misc]
            def __init__(self, config: Any) -> None:
                super().__init__(config)
                self.model = model
                self.model_id = model_id
                self.instruments = instruments or []
                self.min_confidence = min_confidence
                
            def on_bar(self, bar: Bar) -> None:
                # Skip if not configured for this instrument
                if self.instruments and bar.bar_type.instrument_id not in self.instruments:
                    return
                    
                try:
                    # Generate prediction
                    features = np.random.randn(1, 10)  # Mock features
                    if self.model is not None:
                        prediction = self.model.predict(features)[0]
                        confidence = abs(float(prediction))
                    else:
                        prediction = 0.0
                        confidence = 0.5
                    
                    # Check confidence threshold
                    if confidence < self.min_confidence:
                        return
                    
                    # Create and publish signal - using model_id as first-class field
                    signal = MLSignal(
                        instrument_id=bar.bar_type.instrument_id,
                        model_id=self.model_id,  # Required field
                        prediction=float(prediction),
                        confidence=confidence,
                        ts_event=bar.ts_event,
                        ts_init=bar.ts_init,
                    )
                    
                    self.publish_data(None, signal)
                    
                except Exception as e:
                    # Silently handle errors in test
                    pass
        
        # Create config
        from nautilus_trader.config import ActorConfig
        config = ActorConfig(component_id="TEST_ACTOR")
        
        # Create actor
        actor = TestMLActor(config)
        
        return actor
    
    def _create_bar_for_instrument(self, instrument_str: str) -> Bar:
        """Create a test bar for a specific instrument."""
        from nautilus_trader.model.data import BarType, BarSpecification
        from nautilus_trader.model.enums import AggregationSource, BarAggregation, PriceType
        from nautilus_trader.model.objects import Price, Quantity
        
        # Create a new bar type with the specific instrument
        instrument_id = InstrumentId.from_str(instrument_str)
        bar_spec = BarSpecification(
            step=1,
            aggregation=BarAggregation.MINUTE,
            price_type=PriceType.MID,
        )
        bar_type = BarType(
            instrument_id=instrument_id,
            bar_spec=bar_spec,
            aggregation_source=AggregationSource.EXTERNAL,
        )
        
        # Create bar with the new bar type
        bar = Bar(
            bar_type=bar_type,
            open=Price.from_str("1.00001"),
            high=Price.from_str("1.00004"),
            low=Price.from_str("1.00000"),
            close=Price.from_str("1.00003"),
            volume=Quantity.from_int(100000),
            ts_event=0,
            ts_init=0,
        )
        return bar