# ML Migration Example: XGBoost Inference Actor

This document demonstrates the concrete migration of the `GenericInferenceActor` to a Nautilus-compatible `MLInferenceActor`.

## Original Implementation (OLD)

```python
# OLD/trade/nautilus_ml/actors/generic_inference_actor.py
from ..config.ml_config import GenericMLActorConfig as MLActorConfig
from ..registry.model_wrapper import MLflowModelWrapper
from .base_inference_actor import BaseInferenceActor

class MLPrediction(Data):
    def __init__(self, instrument_id, model_name, model_version, ...):
        # Custom initialization
        self.instrument_id = instrument_id
        self.model_name = model_name
        # ... many more fields

class GenericInferenceActor(BaseInferenceActor):
    def __init__(self, config: MLActorConfig):
        super().__init__(config)
        self._registry = None  # Lazy MLflow
        self.model = None
        # Complex initialization...
```

## Migrated Implementation (NEW)

```python
# ml/actors/ml_inference_actor.py
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
from msgspec import Struct

from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig
from nautilus_trader.core.data import Data
from nautilus_trader.model.data import Bar, DataType
from nautilus_trader.model.identifiers import InstrumentId

if TYPE_CHECKING:
    from nautilus_trader.common.clock import Clock

# Configuration using msgspec
class MLInferenceActorConfig(ActorConfig, frozen=True):
    """Configuration for ML inference actor."""

    model_name: str
    model_path: str
    instrument_ids: list[str]
    update_frequency: float = 60.0
    min_confidence: float = 0.6
    warmup_period: int = 50


# Data types following Nautilus patterns
class MLSignal(Data):
    """
    ML prediction signal.

    Attributes
    ----------
    instrument_id : InstrumentId
        The instrument identifier
    model_name : str
        Name of the model generating the signal
    prediction : int
        Prediction value (-1: sell, 0: neutral, 1: buy)
    probability : float
        Probability of the prediction (0.0 to 1.0)
    confidence : float
        Model confidence in the prediction
    ts_event : int
        UNIX timestamp (nanoseconds) of the event
    ts_init : int
        UNIX timestamp (nanoseconds) of initialization

    """

    def __init__(
        self,
        instrument_id: InstrumentId,
        model_name: str,
        prediction: int,
        probability: float,
        confidence: float,
        ts_event: int,
        ts_init: int,
    ):
        self.instrument_id = instrument_id
        self.model_name = model_name
        self.prediction = prediction
        self.probability = probability
        self.confidence = confidence
        self._ts_event = ts_event
        self._ts_init = ts_init


class MLInferenceActor(Actor):
    """
    Actor for ML model inference following Nautilus patterns.

    This actor:
    - Loads a pre-trained model at startup
    - Subscribes to market data
    - Generates ML signals
    - Publishes signals via message bus

    Parameters
    ----------
    config : MLInferenceActorConfig
        The actor configuration

    """

    def __init__(self, config: MLInferenceActorConfig) -> None:
        super().__init__(config)

        # Configuration
        self._config = config
        self._instrument_ids = [
            InstrumentId.from_str(i) for i in config.instrument_ids
        ]

        # Model components
        self._model = None
        self._feature_config = None
        self._scaler = None

        # Price history for features
        self._price_history: dict[InstrumentId, deque] = {
            inst_id: deque(maxlen=100)
            for inst_id in self._instrument_ids
        }

        # Performance tracking
        self._prediction_count = 0
        self._last_prediction_time: dict[InstrumentId, int] = {}

    def on_start(self) -> None:
        """Initialize the actor."""
        self.log.info(f"Starting ML Inference Actor: {self._config.model_name}")

        # Load model
        self._load_model()

        # Subscribe to market data
        for instrument_id in self._instrument_ids:
            # Subscribe to bars for each instrument
            bar_type = f"{instrument_id}-1-MINUTE-LAST-EXTERNAL"
            self.subscribe_bars(bar_type)

        self.log.info(
            f"Subscribed to {len(self._instrument_ids)} instruments"
        )

    def _load_model(self) -> None:
        """Load model and associated components."""
        try:
            model_path = Path(self._config.model_path)

            # Load model bundle
            with open(model_path, 'rb') as f:
                model_data = joblib.load(f)

            self._model = model_data['model']
            self._scaler = model_data.get('scaler')
            self._feature_config = model_data.get('feature_config', {})

            # Log model info
            self.log.info(
                f"Loaded model: {self._config.model_name} "
                f"from {model_path}"
            )

            # Log model metrics if available
            if 'metrics' in model_data:
                metrics = model_data['metrics']
                self.log.info(
                    f"Model metrics - "
                    f"Accuracy: {metrics.get('accuracy', 0):.3f}, "
                    f"Sharpe: {metrics.get('sharpe_ratio', 0):.3f}"
                )

        except Exception as e:
            self.log.error(f"Failed to load model: {e}")
            raise

    def on_bar(self, bar: Bar) -> None:
        """
        Process bar data and generate predictions.

        Parameters
        ----------
        bar : Bar
            The bar data

        """
        # Update price history
        self._update_price_history(bar)

        # Check if we have enough data
        history = self._price_history[bar.instrument_id]
        if len(history) < self._config.warmup_period:
            return

        # Check update frequency
        current_time = self._clock.timestamp_ns()
        last_time = self._last_prediction_time.get(bar.instrument_id, 0)

        if (current_time - last_time) / 1e9 < self._config.update_frequency:
            return

        # Generate prediction
        self._generate_prediction(bar)

        # Update last prediction time
        self._last_prediction_time[bar.instrument_id] = current_time

    def _update_price_history(self, bar: Bar) -> None:
        """Update price history for feature calculation."""
        self._price_history[bar.instrument_id].append({
            'open': float(bar.open),
            'high': float(bar.high),
            'low': float(bar.low),
            'close': float(bar.close),
            'volume': float(bar.volume),
            'timestamp': bar.ts_event,
        })

    def _generate_prediction(self, bar: Bar) -> None:
        """Generate ML prediction and publish signal."""
        try:
            # Calculate features
            features = self._calculate_features(bar.instrument_id)

            if features is None:
                return

            # Scale features if scaler available
            if self._scaler is not None:
                features = self._scaler.transform(features.reshape(1, -1))
            else:
                features = features.reshape(1, -1)

            # Make prediction
            prediction = self._model.predict(features)[0]

            # Get probability if available
            if hasattr(self._model, 'predict_proba'):
                probabilities = self._model.predict_proba(features)[0]
                # For binary classification
                if len(probabilities) == 2:
                    probability = probabilities[1]
                else:
                    # Multi-class - use max probability
                    probability = np.max(probabilities)
            else:
                probability = 0.5  # Default for non-probabilistic models

            # Calculate confidence (simplified)
            confidence = abs(probability - 0.5) * 2

            # Skip low confidence predictions
            if confidence < self._config.min_confidence:
                return

            # Map to trading signal
            if prediction == 1 and probability > 0.5:
                signal_prediction = 1  # Buy
            elif prediction == 0 and probability < 0.5:
                signal_prediction = -1  # Sell
            else:
                signal_prediction = 0  # Neutral

            # Create ML signal
            signal = MLSignal(
                instrument_id=bar.instrument_id,
                model_name=self._config.model_name,
                prediction=signal_prediction,
                probability=probability,
                confidence=confidence,
                ts_event=bar.ts_event,
                ts_init=self._clock.timestamp_ns(),
            )

            # Publish signal
            self.publish_data(
                DataType(MLSignal),
                signal,
            )

            self._prediction_count += 1

            # Log significant signals
            if confidence > 0.7:
                self.log.info(
                    f"{self._config.model_name} signal for {bar.instrument_id}: "
                    f"prediction={signal_prediction}, "
                    f"confidence={confidence:.3f}"
                )

        except Exception as e:
            self.log.error(
                f"Error generating prediction for {bar.instrument_id}: {e}"
            )

    def _calculate_features(self, instrument_id: InstrumentId) -> np.ndarray | None:
        """
        Calculate features from price history.

        Simple feature engineering for demonstration.
        In production, this would use the FeatureEngine.

        """
        history = list(self._price_history[instrument_id])

        if len(history) < 20:  # Need minimum history
            return None

        # Extract price arrays
        closes = np.array([h['close'] for h in history])
        volumes = np.array([h['volume'] for h in history])

        # Simple features
        features = []

        # Price returns
        features.append(self._calculate_return(closes, 1))
        features.append(self._calculate_return(closes, 5))
        features.append(self._calculate_return(closes, 20))

        # Simple moving averages
        features.append(closes[-1] / np.mean(closes[-20:]) - 1)
        features.append(closes[-1] / np.mean(closes[-50:]) - 1)

        # Volume ratio
        features.append(volumes[-1] / np.mean(volumes[-20:]))

        # RSI (simplified)
        rsi = self._calculate_rsi(closes, 14)
        features.append(rsi / 100.0)

        # Volatility
        returns = np.diff(closes) / closes[:-1]
        features.append(np.std(returns[-20:]))

        return np.array(features)

    def _calculate_return(self, prices: np.ndarray, period: int) -> float:
        """Calculate return over period."""
        if len(prices) < period + 1:
            return 0.0
        return (prices[-1] / prices[-period-1] - 1)

    def _calculate_rsi(self, prices: np.ndarray, period: int) -> float:
        """Calculate RSI (simplified)."""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def on_stop(self) -> None:
        """Clean up resources."""
        self.log.info(
            f"Stopping ML Inference Actor: {self._config.model_name} - "
            f"Generated {self._prediction_count} predictions"
        )


# Factory function for creating actors
def create_ml_actor(
    model_name: str,
    model_path: str,
    instruments: list[str],
) -> MLInferenceActor:
    """
    Create ML inference actor.

    Parameters
    ----------
    model_name : str
        Name of the model
    model_path : str
        Path to model file
    instruments : list[str]
        List of instrument IDs

    Returns
    -------
    MLInferenceActor
        Configured ML actor

    """
    config = MLInferenceActorConfig(
        actor_id=f"ML-{model_name.upper()}",
        model_name=model_name,
        model_path=model_path,
        instrument_ids=instruments,
    )

    return MLInferenceActor(config)
```

## Key Migration Changes

### 1. Configuration

- **OLD**: Pydantic `BaseModel`
- **NEW**: msgspec `Struct` with `frozen=True`

### 2. Data Types

- **OLD**: Complex `MLPrediction` with many fields
- **NEW**: Simple `MLSignal` following Nautilus `Data` pattern

### 3. Model Loading

- **OLD**: Complex MLflow registry with retries
- **NEW**: Simple joblib load at startup

### 4. Message Publishing

- **OLD**: `self.publish_data(type(signal), signal)`
- **NEW**: `self.publish_data(DataType(MLSignal), signal)`

### 5. Error Handling

- **OLD**: Complex retry logic with fallbacks
- **NEW**: Simple try/except with logging

### 6. Feature Calculation

- **OLD**: External `FeatureEngineerV2` with complex state
- **NEW**: Simplified inline calculation (or use FeatureActor)

## Integration Example

```python
# ml/examples/run_ml_inference.py
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig

# Import our ML components
from ml.actors.ml_inference_actor import create_ml_actor
from ml.strategies.ml_trading_strategy import MLTradingStrategy

def run_ml_backtest():
    """Run backtest with ML inference actor."""

    # Create engine
    engine = BacktestEngine(config=BacktestEngineConfig())

    # Create ML actor
    ml_actor = create_ml_actor(
        model_name="xgboost_momentum",
        model_path="models/xgboost_latest.pkl",
        instruments=["AAPL.NASDAQ", "MSFT.NASDAQ"],
    )

    # Create strategy that listens to ML signals
    strategy = MLTradingStrategy(
        config=MLTradingStrategyConfig(
            strategy_id="ML-STRATEGY-001",
            instrument_ids=["AAPL.NASDAQ", "MSFT.NASDAQ"],
            trade_size=Decimal("100"),
            signal_threshold=0.7,
        )
    )

    # Add to engine
    engine.add_actor(ml_actor)
    engine.add_strategy(strategy)

    # Load data and run
    # ... (data loading code)

    engine.run()

    # Get results
    results = engine.trader.generate_account_report()
    print(results)
```

## Testing the Migration

```python
# ml/tests/unit/test_ml_inference_actor.py
import numpy as np
import pytest
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.test_kit.stubs.data import TestDataStubs

from ml.actors.ml_inference_actor import MLInferenceActor, MLSignal

class TestMLInferenceActor:
    """Test ML inference actor."""

    def test_actor_initialization(self):
        """Test actor initializes correctly."""
        # Create config
        config = MLInferenceActorConfig(
            actor_id="TEST-ML-001",
            model_name="test_model",
            model_path="tests/fixtures/test_model.pkl",
            instrument_ids=["AAPL.NASDAQ"],
        )

        # Create actor
        actor = MLInferenceActor(config)

        assert actor.id.value == "TEST-ML-001"
        assert actor._config.model_name == "test_model"

    def test_signal_generation(self):
        """Test ML signal generation."""
        # Setup test data
        bar = TestDataStubs.bar_5decimal()

        # Create mock actor with loaded model
        actor = self._create_test_actor()

        # Process enough bars for warmup
        for _ in range(60):
            actor.on_bar(bar)

        # Check signal was generated
        assert actor._prediction_count > 0

    def test_feature_calculation(self):
        """Test feature calculation consistency."""
        actor = self._create_test_actor()

        # Add price history
        for i in range(100):
            actor._price_history[InstrumentId("AAPL.NASDAQ")].append({
                'close': 100 + i * 0.1,
                'volume': 1000000,
                'timestamp': i * 1000000000,
            })

        # Calculate features
        features = actor._calculate_features(InstrumentId("AAPL.NASDAQ"))

        assert features is not None
        assert len(features) == 8  # Number of features
        assert not np.any(np.isnan(features))
```

## Summary

This example demonstrates:

1. **Simplified architecture** - Removed complex MLflow dependencies from hot path
2. **Nautilus patterns** - Proper use of Actor, Data, and message bus
3. **Type safety** - Full type hints with msgspec
4. **Testing** - Unit tests following Nautilus patterns
5. **Performance** - Optimized for low-latency inference

The migrated component is cleaner, more maintainable, and fully integrated with Nautilus Trader's architecture.
