# Float32 vs Float64: The Nuanced Policy

## The Real Answer: It Depends on the Path

You're absolutely right to question blanket float32 standardization. Here's the more sophisticated policy:

## Recommended Dtype Policy

### 🧊 COLD PATH (Training)
**Use float64 where it matters:**

```python
class LightGBMTrainer:
    def prepare_data(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        # ✅ KEEP float64 for training stability
        X = df.select(self.feature_columns).to_numpy()  # float64 default
        y = df.select(self.target_column).to_numpy()   # float64 default
        
        # LightGBM/XGBoost handle float64 natively
        # Better numerical stability for:
        # - Large feature sets (100+ features)
        # - Features with different scales
        # - Gradient computations
        # - Tree splitting decisions
        
        return X, y  # Keep as float64
    
    def save_model(self, model, path):
        # Model internally uses float64 splits
        # That's fine - precision matters here
        model.save_model(path)
```

**Why float64 is better for training:**
1. **Numerical Stability**: Gradient descent accumulates small errors
2. **Feature Scales**: When mixing features like price (1000s) with ratios (0.001)
3. **Tree Splits**: More precise split points = better model
4. **Large Feature Sets**: 100+ features accumulate rounding errors
5. **No Speed Penalty**: Training is offline, precision > speed

### 🔥 HOT PATH (Inference)
**Must use float32:**

```python
class MLSignalActor:
    def __init__(self):
        # ✅ MUST be float32 for hot path
        self._feature_buffer = np.zeros(n_features, dtype=np.float32)
    
    def _compute_features(self, bar) -> np.ndarray:
        # All features computed as float32
        close = np.float32(bar.close)
        volume = np.float32(bar.volume)
        
        # Indicators already use appropriate precision
        sma = np.float32(self.sma.value)
        rsi = np.float32(self.rsi.value)
        
        return self._feature_buffer  # float32
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        # Models MUST return float32 predictions
        if features.dtype != np.float32:
            features = features.astype(np.float32)
        
        predictions = self._model.predict(features)
        return predictions.astype(np.float32)
```

**Why float32 for inference:**
1. **ONNX Optimization**: ONNX Runtime optimized for float32
2. **Memory**: Half the cache pressure (critical in hot path)
3. **SIMD**: CPU vectorization works better with float32
4. **Sufficient Precision**: Predictions are probabilities [0,1]
5. **2x Faster**: Especially with batch predictions

## The Sophisticated Approach

### Model Training & Saving

```python
class XGBoostTrainer:
    def train(self, X_train, y_train):
        # ✅ Train with float64 for stability
        assert X_train.dtype == np.float64
        
        # XGBoost internally uses float32 for predictions anyway
        # but float64 for gradient computations
        model = xgb.train(params, dtrain)
        
        # Save with metadata about expected dtypes
        metadata = {
            'training_dtype': 'float64',
            'inference_dtype': 'float32',  # What we expect at inference
            'features': feature_names,
        }
        
        return model, metadata
```

### Model Loading & Inference

```python
class XGBoostModel:
    def __init__(self, model, metadata):
        self._model = model
        self._inference_dtype = np.float32  # Always float32 for inference
        
    def predict(self, features: np.ndarray) -> np.ndarray:
        # Convert if needed (but actor should provide float32)
        if features.dtype != np.float32:
            features = features.astype(np.float32)
        
        # XGBoost/LightGBM handle float32 inputs efficiently
        # Even if trained on float64
        predictions = self._model.predict(features)
        
        # Always return float32
        return predictions.astype(np.float32)
```

## ONNX Conversion Bridge

```python
def export_to_onnx(model, example_input, metadata):
    # ONNX uses float32 by default
    # This is where we bridge from float64 training to float32 inference
    
    # Create float32 example for ONNX export
    example_float32 = example_input.astype(np.float32)
    
    # Export with float32 inputs/outputs
    onnx_model = convert_to_onnx(
        model,
        initial_types=[('input', FloatTensorType([None, n_features]))],  # float32
        target_opset=12
    )
    
    # Metadata indicates the dtype expectation
    metadata['onnx_input_dtype'] = 'float32'
    metadata['onnx_output_dtype'] = 'float32'
    
    return onnx_model, metadata
```

## Feature Engineering: The Bridge

```python
class FeatureEngineer:
    def __init__(self, mode='inference'):
        self.mode = mode
        
    def compute_features(self, data):
        if self.mode == 'training':
            # ✅ Use float64 for training (Polars DataFrame)
            features = data.select([
                pl.col('close').cast(pl.Float64),
                pl.col('volume').cast(pl.Float64),
                # Complex feature engineering...
            ]).to_numpy()  # Returns float64
            
        else:  # inference
            # ✅ Use float32 for inference (numpy only)
            features = np.array([
                np.float32(data.close),
                np.float32(data.volume),
                # Simple feature computation...
            ], dtype=np.float32)
            
        return features
```

## When Precision Really Matters

### Cases where float64 is essential (training):

1. **Accumulated Statistics**
```python
# Running statistics over millions of samples
running_mean = np.float64(0)
running_var = np.float64(0)
for i in range(1_000_000):
    running_mean += (value - running_mean) / (i + 1)  # Needs float64
```

2. **Cross-Product Features**
```python
# Many multiplied features
feature = price * volume * time_weight * decay_factor  # Can overflow float32
```

3. **Log/Exp Transformations**
```python
# Log of small probabilities
log_prob = np.log(1e-10)  # float32 might underflow
```

### Cases where float32 is sufficient (inference):

1. **Final Predictions**
```python
probability = 0.7234  # float32 precision is plenty
```

2. **Normalized Features**
```python
rsi = 45.2  # [0, 100] range, float32 is fine
z_score = 1.35  # Standardized, float32 is fine
```

3. **Indicator Values**
```python
sma_20 = 1825.43  # float32 handles this perfectly
```

## Updated Recommendation

### Training Pipeline (Cold Path)
```python
# ✅ CORRECT: Preserve precision where it matters
X_train = df.to_numpy()  # float64 - good for stability
model.fit(X_train, y_train)  # Train with float64

# But save metadata about inference expectations
metadata['inference_dtype'] = 'float32'
```

### Inference Pipeline (Hot Path)
```python
# ✅ CORRECT: Optimize for speed
features = np.zeros(n, dtype=np.float32)  # Pre-allocate float32
predictions = model.predict(features)  # Returns float32
```

### The Bridge (Model Wrapper)
```python
class ModelWrapper:
    def predict(self, features):
        # Handle both gracefully
        if features.dtype == np.float64:
            # Training/backtesting might send float64
            features = features.astype(np.float32)
        
        # Model trained on float64 can predict with float32
        return self._model.predict(features).astype(np.float32)
```

## Summary

**You were right to question this!** The correct policy is:

1. **Training**: Use float64 for numerical stability (no precision loss)
2. **Inference**: Convert to float32 for speed (sufficient precision)
3. **Bridge**: Models trained on float64 can inference with float32
4. **ONNX**: Always uses float32 (handles conversion automatically)

This gives you the best of both worlds:
- **Training accuracy** from float64 precision
- **Inference speed** from float32 optimization
- **No precision loss** where it matters (training)
- **2x speedup** where it counts (hot path)

The key insight: Tree models (XGBoost/LightGBM) trained on float64 store their split points internally and can accurately score float32 inputs. You don't lose model quality by using float32 at inference time.