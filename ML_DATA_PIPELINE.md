# ML Data Pipeline: Training to Execution

## Overview

This document describes the concrete data transformation pipeline from ML model training to live execution in Nautilus Trader, addressing the Rust/Python boundary issues.

## Key Architecture Decisions

### 1. **Separate ML Package Structure**
```
/home/nate/projects/
├── nautilus_trader/          # Core Nautilus (don't modify)
└── nautilus_ml/              # Your ML package (separate project)
    ├── setup.py
    ├── pyproject.toml
    └── nautilus_ml/
        ├── __init__.py
        ├── actors/
        ├── strategies/
        ├── features/
        ├── models/
        └── training/
```

This avoids Rust import issues by keeping ML code separate and importing only the Python API.

### 2. **Data Flow Architecture**

```
┌─────────────────────────────────────────────────────────────────────┐
│                          COLD PATH (Training)                        │
├─────────────────────────────────────────────────────────────────────┤
│  Historical Data → Feature Engineering → Model Training → MLflow     │
│  (Parquet/CSV)     (Polars/Pandas)      (XGBoost/etc)   (Registry) │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
                            Model Artifacts
                          (model.pkl, config.json)
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                          HOT PATH (Execution)                        │
├─────────────────────────────────────────────────────────────────────┤
│  Market Data → Feature Extract → Model Inference → Strategy Action  │
│  (Bar/Tick)    (Incremental)     (Cached Model)    (Buy/Sell)      │
└─────────────────────────────────────────────────────────────────────┘
```

## Concrete Implementation

### 1. **Training Pipeline (Cold Path)**

```python
# nautilus_ml/training/train_model.py
import polars as pl
import pandas as pd
from pathlib import Path
import joblib
import json
from datetime import datetime

from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import Bar
from lightgbm import LGBMClassifier
import mlflow


class ModelTrainer:
    """Trains ML models on historical Nautilus data."""
    
    def __init__(self, catalog_path: str):
        self.catalog = ParquetDataCatalog(catalog_path)
        
    def load_training_data(self, instruments: list[str], start_date: str, end_date: str) -> pl.DataFrame:
        """Load historical bars from Nautilus catalog."""
        # Load bars from catalog
        bars_list = []
        for instrument in instruments:
            bars = self.catalog.bars(
                instrument_ids=[instrument],
                start=start_date,
                end=end_date
            )
            bars_list.extend(bars)
        
        # Convert to DataFrame for ML
        df = self._bars_to_dataframe(bars_list)
        return df
    
    def _bars_to_dataframe(self, bars: list[Bar]) -> pl.DataFrame:
        """Convert Nautilus Bar objects to Polars DataFrame."""
        records = []
        for bar in bars:
            records.append({
                'timestamp': bar.ts_event,
                'instrument_id': str(bar.instrument_id),
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': float(bar.volume),
            })
        
        return pl.DataFrame(records)
    
    def engineer_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create ML features from price data."""
        # Sort by instrument and time
        df = df.sort(['instrument_id', 'timestamp'])
        
        # Calculate features using Polars expressions
        df = df.with_columns([
            # Returns
            (pl.col('close') / pl.col('close').shift(1) - 1).alias('returns_1'),
            (pl.col('close') / pl.col('close').shift(5) - 1).alias('returns_5'),
            (pl.col('close') / pl.col('close').shift(20) - 1).alias('returns_20'),
            
            # Price ratios
            (pl.col('high') / pl.col('low') - 1).alias('high_low_ratio'),
            (pl.col('close') / pl.col('open') - 1).alias('close_open_ratio'),
            
            # Volume features
            (pl.col('volume') / pl.col('volume').shift(1) - 1).alias('volume_ratio'),
            pl.col('volume').rolling_mean(window_size=20).alias('volume_sma_20'),
            
            # Technical indicators
            pl.col('close').rolling_mean(window_size=20).alias('sma_20'),
            pl.col('close').rolling_mean(window_size=50).alias('sma_50'),
            (pl.col('close') / pl.col('close').rolling_mean(window_size=20) - 1).alias('price_to_sma_20'),
            
            # Volatility
            pl.col('returns_1').rolling_std(window_size=20).alias('volatility_20'),
        ]).over('instrument_id')
        
        # Create target (next period return)
        df = df.with_columns([
            pl.col('returns_1').shift(-1).alias('target_return')
        ]).over('instrument_id')
        
        # Create classification target
        df = df.with_columns([
            pl.when(pl.col('target_return') > 0.001).then(1)
            .when(pl.col('target_return') < -0.001).then(-1)
            .otherwise(0)
            .alias('target_direction')
        ])
        
        return df.drop_nulls()
    
    def train_model(self, df: pl.DataFrame) -> dict:
        """Train ML model and save artifacts."""
        # Feature columns
        feature_cols = [
            'returns_1', 'returns_5', 'returns_20',
            'high_low_ratio', 'close_open_ratio',
            'volume_ratio', 'volume_sma_20',
            'price_to_sma_20', 'volatility_20'
        ]
        
        # Split data
        train_size = int(len(df) * 0.8)
        train_df = df[:train_size]
        test_df = df[train_size:]
        
        # Prepare data
        X_train = train_df[feature_cols].to_pandas()
        y_train = train_df['target_direction'].to_pandas()
        X_test = test_df[feature_cols].to_pandas()
        y_test = test_df['target_direction'].to_pandas()
        
        # Train model
        model = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        # Save artifacts
        artifacts = {
            'model': model,
            'feature_columns': feature_cols,
            'model_config': {
                'type': 'lightgbm_classifier',
                'version': '1.0',
                'trained_at': datetime.now().isoformat(),
                'train_score': train_score,
                'test_score': test_score,
            }
        }
        
        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_params(model.get_params())
            mlflow.log_metrics({
                'train_accuracy': train_score,
                'test_accuracy': test_score
            })
            
            # Save model
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name="nautilus_direction_predictor"
            )
            
            # Save feature config
            mlflow.log_dict(artifacts['model_config'], "model_config.json")
            
        return artifacts
```

### 2. **Feature Engineering Bridge (Shared)**

```python
# nautilus_ml/features/feature_engine.py
from collections import deque
import numpy as np
from typing import Dict, Any

from nautilus_trader.model.data import Bar
from nautilus_trader.indicators.average.sma import SimpleMovingAverage
from nautilus_trader.indicators.atr import AverageTrueRange


class RealtimeFeatureEngine:
    """
    Computes features incrementally for real-time inference.
    MUST produce identical features to training pipeline.
    """
    
    def __init__(self, feature_config: dict):
        self.feature_config = feature_config
        self.feature_names = feature_config['feature_columns']
        
        # Initialize rolling windows
        self.price_buffer = deque(maxlen=50)  # For SMA calculations
        self.volume_buffer = deque(maxlen=20)
        self.returns_buffer = deque(maxlen=20)
        
        # Initialize Nautilus indicators for consistency
        self.sma_20 = SimpleMovingAverage(20)
        self.sma_50 = SimpleMovingAverage(50)
        self.atr = AverageTrueRange(14)
        
        # State for incremental calculations
        self.last_close = None
        self.last_volume = None
        
    def update(self, bar: Bar) -> Dict[str, float]:
        """Update features with new bar data."""
        close = float(bar.close)
        high = float(bar.high)
        low = float(bar.low)
        open_ = float(bar.open)
        volume = float(bar.volume)
        
        # Update buffers
        self.price_buffer.append(close)
        self.volume_buffer.append(volume)
        
        # Calculate returns
        if self.last_close is not None:
            returns_1 = (close / self.last_close - 1)
            self.returns_buffer.append(returns_1)
        else:
            returns_1 = 0.0
            
        # Update indicators
        self.sma_20.update(bar.close)
        self.sma_50.update(bar.close)
        self.atr.update(bar)
        
        # Calculate features (MUST match training)
        features = {}
        
        # Returns
        features['returns_1'] = returns_1
        features['returns_5'] = self._calculate_return(5) if len(self.price_buffer) >= 5 else 0.0
        features['returns_20'] = self._calculate_return(20) if len(self.price_buffer) >= 20 else 0.0
        
        # Price ratios
        features['high_low_ratio'] = (high / low - 1) if low > 0 else 0.0
        features['close_open_ratio'] = (close / open_ - 1) if open_ > 0 else 0.0
        
        # Volume features
        features['volume_ratio'] = (volume / self.last_volume - 1) if self.last_volume and self.last_volume > 0 else 0.0
        features['volume_sma_20'] = np.mean(self.volume_buffer) if len(self.volume_buffer) >= 20 else volume
        
        # Technical indicators
        features['price_to_sma_20'] = (close / float(self.sma_20.value) - 1) if self.sma_20.initialized else 0.0
        
        # Volatility
        features['volatility_20'] = np.std(self.returns_buffer) if len(self.returns_buffer) >= 20 else 0.0
        
        # Update state
        self.last_close = close
        self.last_volume = volume
        
        # Return only the features used in training
        return {k: features[k] for k in self.feature_names}
    
    def _calculate_return(self, periods: int) -> float:
        """Calculate return over n periods."""
        if len(self.price_buffer) < periods + 1:
            return 0.0
        return self.price_buffer[-1] / self.price_buffer[-periods-1] - 1
```

### 3. **ML Inference Actor (Hot Path)**

```python
# nautilus_ml/actors/ml_inference_actor.py
import joblib
import numpy as np
from pathlib import Path

from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig
from nautilus_trader.core.data import Data
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId

from nautilus_ml.features.feature_engine import RealtimeFeatureEngine


class MLSignal(Data):
    """Custom data type for ML predictions."""
    
    def __init__(
        self,
        instrument_id: InstrumentId,
        prediction: int,  # -1, 0, 1
        probability: float,
        features: dict,
        ts_event: int,
        ts_init: int,
    ):
        super().__init__()
        self.instrument_id = instrument_id
        self.prediction = prediction
        self.probability = probability
        self.features = features
        self.ts_event = ts_event
        self.ts_init = ts_init


class MLInferenceActorConfig(ActorConfig, frozen=True):
    """Configuration for ML inference actor."""
    
    model_path: str
    config_path: str
    bar_type: str
    min_probability: float = 0.6


class MLInferenceActor(Actor):
    """
    Real-time ML inference actor.
    Processes market data and generates ML signals.
    """
    
    def __init__(self, config: MLInferenceActorConfig):
        super().__init__(config)
        
        # Load model and config
        self.model = joblib.load(config.model_path)
        self.model_config = joblib.load(config.config_path)
        self.bar_type = BarType.from_str(config.bar_type)
        self.min_probability = config.min_probability
        
        # Initialize feature engine
        self.feature_engine = RealtimeFeatureEngine(self.model_config)
        
        # Performance tracking
        self.predictions_made = 0
        self.signals_sent = 0
        
    def on_start(self) -> None:
        """Subscribe to market data."""
        self.subscribe_bars(self.bar_type)
        self.log.info(f"ML Inference Actor started, model: {self.config.model_path}")
        
    def on_bar(self, bar: Bar) -> None:
        """Process new bar and generate predictions."""
        # Extract features
        features = self.feature_engine.update(bar)
        
        # Prepare for model
        feature_array = np.array([
            [features[col] for col in self.model_config['feature_columns']]
        ])
        
        # Get prediction and probability
        prediction = self.model.predict(feature_array)[0]
        probabilities = self.model.predict_proba(feature_array)[0]
        max_prob = np.max(probabilities)
        
        self.predictions_made += 1
        
        # Only send high-confidence signals
        if max_prob >= self.min_probability and prediction != 0:
            signal = MLSignal(
                instrument_id=bar.instrument_id,
                prediction=int(prediction),
                probability=float(max_prob),
                features=features,
                ts_event=bar.ts_event,
                ts_init=self.clock.timestamp_ns(),
            )
            
            # Publish signal
            self.publish_signal(
                signal,
                channel=f"ml.signals.{bar.instrument_id}"
            )
            
            self.signals_sent += 1
            
            self.log.info(
                f"ML Signal: {bar.instrument_id} "
                f"pred={prediction} prob={max_prob:.3f}"
            )
```

### 4. **ML Strategy (Execution Layer)**

```python
# nautilus_ml/strategies/ml_strategy.py
from decimal import Decimal

from nautilus_trader.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.objects import Quantity

from nautilus_ml.actors.ml_inference_actor import MLSignal


class MLStrategyConfig(StrategyConfig, frozen=True):
    """Configuration for ML trading strategy."""
    
    instrument_id: InstrumentId
    trade_size: Decimal
    ml_signal_channel: str = "ml.signals.*"
    position_limit: int = 1


class MLTradingStrategy(Strategy):
    """
    Executes trades based on ML signals.
    """
    
    def __init__(self, config: MLStrategyConfig):
        super().__init__(config)
        
        # Configuration
        self.instrument_id = config.instrument_id
        self.trade_size = config.trade_size
        self.signal_channel = config.ml_signal_channel
        self.position_limit = config.position_limit
        
        # State
        self.last_signal = None
        self.signals_received = 0
        self.trades_executed = 0
        
    def on_start(self) -> None:
        """Subscribe to ML signals."""
        # Subscribe to ML signals for our instrument
        self.subscribe_data(
            data_type=MLSignal,
            channel=self.signal_channel
        )
        
        # Subscribe to bars for position management
        self.subscribe_bars(self.config.bar_type)
        
        self.log.info(
            f"ML Strategy started for {self.instrument_id}, "
            f"listening to {self.signal_channel}"
        )
        
    def on_data(self, data: Data) -> None:
        """Handle ML signals."""
        if not isinstance(data, MLSignal):
            return
            
        # Filter for our instrument
        if data.instrument_id != self.instrument_id:
            return
            
        self.signals_received += 1
        self.last_signal = data
        
        # Check if we should trade
        if self._should_trade(data):
            self._execute_trade(data)
            
    def _should_trade(self, signal: MLSignal) -> bool:
        """Determine if we should act on this signal."""
        # Check position limits
        positions = self.cache.positions_for_instrument(self.instrument_id)
        if len(positions) >= self.position_limit:
            return False
            
        # Check if we have an open position in same direction
        for position in positions:
            if position.side == OrderSide.BUY and signal.prediction == 1:
                return False
            if position.side == OrderSide.SELL and signal.prediction == -1:
                return False
                
        return True
        
    def _execute_trade(self, signal: MLSignal) -> None:
        """Execute trade based on ML signal."""
        # Determine order side
        if signal.prediction == 1:
            order_side = OrderSide.BUY
        elif signal.prediction == -1:
            order_side = OrderSide.SELL
        else:
            return  # Neutral signal
            
        # Create market order
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=order_side,
            quantity=Quantity.from_str(str(self.trade_size)),
            time_in_force=TimeInForce.GTC,
        )
        
        # Submit order
        self.submit_order(order)
        self.trades_executed += 1
        
        self.log.info(
            f"ML Trade: {order_side} {self.trade_size} {self.instrument_id} "
            f"(signal: {signal.prediction}, prob: {signal.probability:.3f})"
        )
```

### 5. **Integration Example**

```python
# examples/ml_trading_system.py
from pathlib import Path

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.model.identifiers import InstrumentId

from nautilus_ml.actors.ml_inference_actor import MLInferenceActor, MLInferenceActorConfig
from nautilus_ml.strategies.ml_strategy import MLTradingStrategy, MLStrategyConfig


def run_ml_backtest():
    """Run backtest with ML strategy."""
    
    # Configure backtest engine
    engine = BacktestEngine(config=BacktestEngineConfig())
    
    # Add ML inference actor
    ml_actor = MLInferenceActor(
        config=MLInferenceActorConfig(
            actor_id="ML-INFERENCE-001",
            model_path="models/direction_predictor.pkl",
            config_path="models/model_config.pkl",
            bar_type="AAPL.NASDAQ-1-MINUTE-LAST-EXTERNAL",
            min_probability=0.65,
        )
    )
    
    # Add ML strategy
    ml_strategy = MLTradingStrategy(
        config=MLStrategyConfig(
            strategy_id="ML-STRATEGY-001",
            instrument_id=InstrumentId.from_str("AAPL.NASDAQ"),
            trade_size=Decimal("100"),
            ml_signal_channel="ml.signals.AAPL.NASDAQ",
        )
    )
    
    # Add to engine
    engine.add_actor(ml_actor)
    engine.add_strategy(ml_strategy)
    
    # Load data and run
    engine.add_data(...)
    engine.run()
```

## Key Integration Points

### 1. **Data Type Conversions**
- **Training**: Nautilus Bar → Polars DataFrame → Pandas (for sklearn)
- **Inference**: Nautilus Bar → Feature Dict → Numpy Array

### 2. **Feature Consistency**
- Use same feature engineering code (or validate parity)
- Use Nautilus indicators where possible
- Test feature parity between batch and real-time

### 3. **Model Artifacts**
- Save model + feature config together
- Use MLflow for versioning
- Load at actor initialization (not per prediction)

### 4. **Communication Pattern**
- Actor publishes MLSignal via message bus
- Strategy subscribes to signals
- Clean separation of inference and execution

### 5. **Performance Optimization**
- Incremental feature updates (not full recalculation)
- Model loaded once at startup
- Efficient numpy arrays for prediction

## Common Pitfalls Avoided

1. **No Rust imports**: Only use Python API
2. **No blocking operations**: All calculations are fast
3. **Clean data flow**: Market data → Features → Prediction → Execution
4. **Proper typing**: Use Nautilus types correctly
5. **Efficient updates**: Incremental calculations in hot path

This architecture provides a clean, efficient pipeline from ML training to live execution while respecting Nautilus Trader's design principles and avoiding the Rust boundary issues.