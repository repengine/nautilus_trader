# Solving the Polars vs msgspec Conflict

## The Problem

In the last prototype, there was a fundamental incompatibility:
- **Nautilus Trader** uses `msgspec` for configuration and data serialization
- **ML Pipeline** uses `polars` for data processing
- **Conflict**: Polars DataFrames cannot be serialized by msgspec
- **Result**: Runtime errors when trying to pass data between components

## The Root Cause

```python
# ❌ THIS FAILED in last prototype
from nautilus_trader.config import ActorConfig
import polars as pl

class MLActorConfig(ActorConfig, frozen=True):
    """This breaks because msgspec can't serialize Polars DataFrame"""
    training_data: pl.DataFrame  # ❌ msgspec cannot handle this!
    feature_columns: list[str]
```

## The Solution: Clear Separation

### 1. **Configuration (msgspec domain)**
Only use msgspec-compatible types in configurations:

```python
# ✅ CORRECT: Use only msgspec-compatible types in configs
from nautilus_trader.config import ActorConfig
from nautilus_trader.model.identifiers import InstrumentId
import msgspec

class MLActorConfig(ActorConfig, frozen=True):
    """ML Actor configuration with msgspec-compatible types only."""
    
    # Primitives and basic types only
    model_path: str
    feature_config_path: str
    instruments: list[str]  # Not InstrumentId objects
    lookback_periods: int = 20
    update_frequency: int = 60
    
    # Don't put DataFrames, numpy arrays, or complex objects here!
```

### 2. **Data Processing (Polars domain)**
Keep Polars operations completely separate from Nautilus configurations:

```python
# ✅ CORRECT: Polars used only in data processing, not configuration
class MLDataProcessor:
    """Handles all Polars operations separately from Nautilus."""
    
    def __init__(self, config: dict):  # Plain dict, not msgspec
        self.config = config
        self._df_cache = {}  # Internal state, not exposed
        
    def process_training_data(self, file_path: str) -> None:
        """Load and process data with Polars - internal only."""
        # Polars operations stay here
        df = pl.read_parquet(file_path)
        df = self._engineer_features(df)
        
        # Convert to numpy for model training
        X = df.select(self.config['features']).to_numpy()
        y = df.select('target').to_numpy()
        
        # Don't return DataFrame - return numpy or save to disk
        return X, y
```

### 3. **Bridge Pattern: Nautilus ↔ ML**

Create clear interfaces that convert between domains:

```python
# ✅ CORRECT: Clear conversion points
from nautilus_trader.model.data import Bar
import numpy as np

class NautilusMLBridge:
    """Converts between Nautilus and ML data types."""
    
    @staticmethod
    def bars_to_dict(bars: list[Bar]) -> list[dict]:
        """Convert Nautilus Bars to plain dicts for Polars."""
        return [
            {
                'timestamp': bar.ts_event,
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': float(bar.volume),
            }
            for bar in bars
        ]
    
    @staticmethod
    def features_to_array(features: dict) -> np.ndarray:
        """Convert feature dict to numpy array for model."""
        return np.array(list(features.values()))
```

### 4. **Correct ML Actor Implementation**

```python
# ✅ CORRECT: Proper separation of concerns
from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig
from nautilus_trader.model.data import Bar
import joblib
import json

class MLInferenceActorConfig(ActorConfig, frozen=True):
    """Config with only msgspec-compatible types."""
    model_path: str
    feature_config_path: str
    min_confidence: float = 0.6

class MLInferenceActor(Actor):
    def __init__(self, config: MLInferenceActorConfig):
        super().__init__(config)
        
        # Load model and config from disk (not from msgspec config)
        self.model = joblib.load(config.model_path)
        
        # Load feature config as plain dict
        with open(config.feature_config_path, 'r') as f:
            self.feature_config = json.load(f)
            
        # Initialize feature engine (no Polars here!)
        self.feature_engine = RealtimeFeatureEngine(self.feature_config)
        
    def on_bar(self, bar: Bar) -> None:
        """Process bar without Polars."""
        # Use plain Python/numpy for real-time processing
        features = self.feature_engine.update(bar)  # Returns dict
        
        # Convert to numpy for model
        X = np.array([features[col] for col in self.feature_config['columns']])
        
        # Predict
        prediction = self.model.predict(X.reshape(1, -1))
```

### 5. **Training Pipeline (Offline)**

Keep Polars in the training pipeline, completely separate:

```python
# ✅ CORRECT: Training script uses Polars freely
# train_model.py (separate from Nautilus runtime)

import polars as pl
import pandas as pd
from pathlib import Path

def train_model(data_path: str, output_path: str):
    """Training happens offline - use Polars freely here."""
    
    # Load with Polars
    df = pl.read_parquet(data_path)
    
    # Feature engineering with Polars
    df = df.with_columns([
        (pl.col('close') / pl.col('close').shift(1) - 1).alias('returns'),
        pl.col('close').rolling_mean(20).alias('sma_20'),
        # ... more features
    ])
    
    # Train model
    # ... training code ...
    
    # Save artifacts (model + config)
    joblib.dump(model, output_path / 'model.pkl')
    
    # Save feature config as JSON (msgspec compatible)
    feature_config = {
        'columns': ['returns', 'sma_20'],
        'lookback': 20,
    }
    with open(output_path / 'config.json', 'w') as f:
        json.dump(feature_config, f)
```

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING (Offline)                          │
│                   Uses Polars freely                             │
│                                                                  │
│  Historical Data → Polars DataFrame → Model Training → Artifacts│
│                                                         ↓        │
└─────────────────────────────────────────────────────────────────┘
                                                          ↓
                                              model.pkl + config.json
                                                          ↓
┌─────────────────────────────────────────────────────────────────┐
│                      INFERENCE (Online)                          │
│                   No Polars! Only numpy/dicts                    │
│                                                                  │
│  Bar → Feature Dict → Numpy Array → Model → Signal → Strategy  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Rules to Follow

### ✅ DO:
1. Use Polars only in offline training scripts
2. Save trained models and configs to disk
3. Load models from disk in actors
4. Use numpy arrays for model inference
5. Keep configurations simple (strings, numbers, lists)

### ❌ DON'T:
1. Put DataFrames in msgspec configs
2. Import polars in Actors or Strategies
3. Try to serialize complex objects with msgspec
4. Mix data processing frameworks

## Example: Feature Engineering Without Polars

```python
# ✅ Real-time feature engineering without Polars
class RealtimeFeatureEngine:
    """Computes features incrementally without Polars."""
    
    def __init__(self, config: dict):
        self.lookback = config['lookback']
        self.price_history = deque(maxlen=self.lookback)
        
    def update(self, bar: Bar) -> dict:
        """Update features with new bar."""
        close = float(bar.close)
        self.price_history.append(close)
        
        if len(self.price_history) < 2:
            return {'returns': 0.0, 'sma_20': close}
            
        # Calculate features with pure Python
        returns = (close - self.price_history[-2]) / self.price_history[-2]
        sma_20 = sum(list(self.price_history)[-20:]) / min(20, len(self.price_history))
        
        return {
            'returns': returns,
            'sma_20': sma_20,
        }
```

## Testing for Compatibility

```python
# ✅ Test that your config is msgspec-compatible
import msgspec

def test_config_serialization():
    """Ensure configs can be serialized."""
    config = MLInferenceActorConfig(
        actor_id="TEST",
        model_path="/path/to/model.pkl",
        feature_config_path="/path/to/config.json",
    )
    
    # This should work without errors
    encoded = msgspec.msgpack.encode(config)
    decoded = msgspec.msgpack.decode(encoded, type=MLInferenceActorConfig)
    assert config == decoded
```

## Summary

The solution is **strict separation**:
- **Training**: Use Polars/Pandas freely (offline)
- **Configuration**: Only msgspec-compatible types
- **Inference**: Use numpy/dicts (no Polars)
- **Bridge**: Clear conversion points between domains

This avoids the serialization conflicts while maintaining the benefits of both frameworks in their appropriate domains.