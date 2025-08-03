#!/usr/bin/env python3
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
Correct ML integration avoiding Polars/msgspec conflicts.

This example shows the proper separation of concerns:
1. Training uses Polars (offline script)
2. Configuration uses only msgspec-compatible types
3. Inference uses numpy/dicts (no Polars)
"""

import json
import joblib
import numpy as np
from collections import deque
from pathlib import Path
from typing import Optional

from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig
from nautilus_trader.core.data import Data
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price

# ============================================================================
# CONFIGURATION (msgspec domain)
# ============================================================================

class MLInferenceActorConfig(ActorConfig, frozen=True):
    """
    Configuration for ML inference actor.
    
    IMPORTANT: Only msgspec-compatible types!
    No DataFrames, numpy arrays, or complex objects.
    """
    
    model_path: str  # Path to saved model file
    feature_config_path: str  # Path to feature configuration
    bar_type_str: str  # String representation of bar type
    prediction_threshold: float = 0.6
    max_lookback: int = 100


# ============================================================================
# CUSTOM DATA TYPES (msgspec compatible)
# ============================================================================

class MLPrediction(Data):
    """
    ML prediction data type.
    
    Uses only primitive types that msgspec can handle.
    """
    
    def __init__(
        self,
        instrument_id_str: str,  # String, not InstrumentId object
        prediction: float,
        confidence: float,
        direction: int,  # -1, 0, 1
        feature_values: dict[str, float],  # Simple dict
        model_version: str,
        ts_event: int,
        ts_init: int,
    ):
        self.instrument_id_str = instrument_id_str
        self.prediction = prediction
        self.confidence = confidence
        self.direction = direction
        self.feature_values = feature_values
        self.model_version = model_version
        self.ts_event = ts_event
        self.ts_init = ts_init


# ============================================================================
# FEATURE ENGINEERING (no Polars in hot path!)
# ============================================================================

class RealtimeFeatureEngine:
    """
    Computes features incrementally for real-time inference.
    
    NO POLARS! Only numpy and standard Python for performance.
    """
    
    def __init__(self, feature_config: dict):
        """Initialize with configuration dictionary."""
        self.feature_names = feature_config['feature_names']
        self.lookback_periods = feature_config['lookback_periods']
        
        # Use standard Python collections for efficiency
        self.price_buffer = deque(maxlen=max(self.lookback_periods.values()))
        self.volume_buffer = deque(maxlen=20)
        
        # Pre-compute feature indices for fast access
        self.feature_indices = {
            name: idx for idx, name in enumerate(self.feature_names)
        }
        
        # State tracking
        self.last_price = None
        self.initialized = False
        
    def update(self, bar: Bar) -> Optional[np.ndarray]:
        """
        Update features with new bar data.
        
        Returns numpy array ready for model prediction.
        """
        # Extract values (these are already Python floats via Nautilus)
        close = float(bar.close)
        high = float(bar.high)
        low = float(bar.low)
        volume = float(bar.volume)
        
        # Update buffers
        self.price_buffer.append(close)
        self.volume_buffer.append(volume)
        
        # Need minimum data
        if len(self.price_buffer) < 2:
            return None
            
        # Calculate features using numpy where beneficial
        prices = np.array(self.price_buffer)
        
        features = np.zeros(len(self.feature_names))
        
        # Returns
        if 'return_1' in self.feature_indices:
            features[self.feature_indices['return_1']] = (
                (prices[-1] - prices[-2]) / prices[-2] if prices[-2] != 0 else 0
            )
            
        if 'return_5' in self.feature_indices and len(prices) >= 6:
            features[self.feature_indices['return_5']] = (
                (prices[-1] - prices[-6]) / prices[-6] if prices[-6] != 0 else 0
            )
            
        # Moving averages
        if 'sma_20' in self.feature_indices and len(prices) >= 20:
            features[self.feature_indices['sma_20']] = np.mean(prices[-20:])
            
        # Volatility
        if 'volatility_20' in self.feature_indices and len(prices) >= 20:
            returns = np.diff(prices[-21:]) / prices[-21:-1]
            features[self.feature_indices['volatility_20']] = np.std(returns)
            
        # Price ratios
        if 'high_low_ratio' in self.feature_indices:
            features[self.feature_indices['high_low_ratio']] = (
                (high - low) / low if low != 0 else 0
            )
            
        # Volume features
        if 'volume_ratio' in self.feature_indices and self.last_price is not None:
            prev_vol = self.volume_buffer[-2] if len(self.volume_buffer) >= 2 else volume
            features[self.feature_indices['volume_ratio']] = (
                volume / prev_vol if prev_vol != 0 else 1
            )
            
        self.last_price = close
        self.initialized = True
        
        return features
    
    def get_feature_dict(self, features: np.ndarray) -> dict[str, float]:
        """Convert feature array to dictionary for debugging."""
        return {
            name: float(features[idx])
            for name, idx in self.feature_indices.items()
        }


# ============================================================================
# ML INFERENCE ACTOR
# ============================================================================

class MLInferenceActor(Actor):
    """
    ML inference actor that generates predictions from market data.
    
    Key design:
    - Loads model and config from disk (not from msgspec config)
    - Uses numpy for inference (not Polars)
    - Publishes simple data types via message bus
    """
    
    def __init__(self, config: MLInferenceActorConfig):
        super().__init__(config)
        
        # Store configuration
        self.model_path = Path(config.model_path)
        self.feature_config_path = Path(config.feature_config_path)
        self.bar_type = BarType.from_str(config.bar_type_str)
        self.prediction_threshold = config.prediction_threshold
        
        # Load model and configuration from disk
        self._load_model_artifacts()
        
        # Initialize feature engine
        self.feature_engine = RealtimeFeatureEngine(self.feature_config)
        
        # Performance tracking
        self.predictions_made = 0
        self.signals_published = 0
        
    def _load_model_artifacts(self) -> None:
        """Load model and configuration from disk."""
        # Load model using joblib
        self.model = joblib.load(self.model_path)
        self.log.info(f"Loaded model from {self.model_path}")
        
        # Load feature configuration as plain dict
        with open(self.feature_config_path, 'r') as f:
            self.feature_config = json.load(f)
        self.log.info(f"Loaded feature config with {len(self.feature_config['feature_names'])} features")
        
        # Extract model metadata
        self.model_version = self.feature_config.get('model_version', 'unknown')
        
    def on_start(self) -> None:
        """Subscribe to market data on start."""
        self.subscribe_bars(self.bar_type)
        self.log.info(
            f"ML Inference Actor started: "
            f"model_version={self.model_version}, "
            f"bar_type={self.bar_type}"
        )
        
    def on_bar(self, bar: Bar) -> None:
        """Process new bar and generate predictions."""
        # Extract features (returns numpy array)
        features = self.feature_engine.update(bar)
        
        if features is None:
            return  # Not enough data yet
            
        # Run inference
        try:
            # Reshape for sklearn (expects 2D array)
            X = features.reshape(1, -1)
            
            # Get prediction and probability
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            # Find the predicted class and its probability
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
            
            # Map to direction (-1, 0, 1)
            direction_map = {0: -1, 1: 0, 2: 1}  # Assuming 3 classes
            direction = direction_map.get(predicted_class, 0)
            
            self.predictions_made += 1
            
            # Only publish high-confidence predictions
            if confidence >= self.prediction_threshold:
                # Create prediction object
                ml_prediction = MLPrediction(
                    instrument_id_str=str(bar.instrument_id),
                    prediction=float(prediction),
                    confidence=float(confidence),
                    direction=direction,
                    feature_values=self.feature_engine.get_feature_dict(features),
                    model_version=self.model_version,
                    ts_event=bar.ts_event,
                    ts_init=self.clock.timestamp_ns(),
                )
                
                # Publish to message bus
                self.publish_data(
                    data_type=MLPrediction,
                    data=ml_prediction,
                )
                
                self.signals_published += 1
                
                if self.signals_published % 100 == 0:
                    self.log.info(
                        f"ML Stats: {self.predictions_made} predictions, "
                        f"{self.signals_published} signals published"
                    )
                    
        except Exception as e:
            self.log.error(f"Prediction failed: {e}")


# ============================================================================
# TRAINING SCRIPT (separate file in production)
# ============================================================================

def train_model_offline():
    """
    Example training script - runs offline, uses Polars freely.
    
    This would be in a separate file/module in production.
    """
    import polars as pl
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # Load data with Polars (offline is fine!)
    df = pl.read_parquet("historical_data.parquet")
    
    # Feature engineering with Polars
    df = df.with_columns([
        (pl.col('close') / pl.col('close').shift(1) - 1).alias('return_1'),
        (pl.col('close') / pl.col('close').shift(5) - 1).alias('return_5'),
        pl.col('close').rolling_mean(window_size=20).alias('sma_20'),
        ((pl.col('high') - pl.col('low')) / pl.col('low')).alias('high_low_ratio'),
    ])
    
    # Create target
    df = df.with_columns([
        pl.when(pl.col('return_1').shift(-1) > 0.001).then(2)
        .when(pl.col('return_1').shift(-1) < -0.001).then(0)
        .otherwise(1)
        .alias('target')
    ])
    
    # Drop nulls and convert to numpy for sklearn
    df_clean = df.drop_nulls()
    
    feature_names = ['return_1', 'return_5', 'sma_20', 'high_low_ratio']
    X = df_clean.select(feature_names).to_numpy()
    y = df_clean.select('target').to_numpy().ravel()
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, 'models/rf_model.pkl')
    
    # Save feature configuration (msgspec compatible!)
    feature_config = {
        'feature_names': feature_names,
        'lookback_periods': {
            'return_1': 2,
            'return_5': 6,
            'sma_20': 20,
            'high_low_ratio': 1,
        },
        'model_version': 'rf_v1.0',
        'model_type': 'RandomForestClassifier',
        'train_score': float(model.score(X_train, y_train)),
        'test_score': float(model.score(X_test, y_test)),
    }
    
    with open('models/feature_config.json', 'w') as f:
        json.dump(feature_config, f, indent=2)
    
    print(f"Model trained: train_score={feature_config['train_score']:.3f}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def create_ml_actor():
    """Create ML inference actor with proper configuration."""
    config = MLInferenceActorConfig(
        actor_id="ML-INFERENCE-001",
        model_path="models/rf_model.pkl",
        feature_config_path="models/feature_config.json",
        bar_type_str="AAPL.NASDAQ-1-MINUTE-LAST-EXTERNAL",
        prediction_threshold=0.65,
        max_lookback=100,
    )
    
    return MLInferenceActor(config)


if __name__ == "__main__":
    # This shows the structure - in practice:
    # 1. Run train_model_offline() separately
    # 2. Create and add actor to Nautilus engine
    print("See train_model_offline() for training")
    print("See create_ml_actor() for inference")