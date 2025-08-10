# Consolidated Issues from All ISSUES.txt Files

## Overview
This document consolidates ALL specific technical issues found across the ML module, with concrete fixes and code snippets.

## 1. XGBoost Issues (training/ISSUES_XGBOOST.txt)

### 🔴 Critical Performance Issues

#### Issue: DMatrix Creation Per Tick (Lines 23-30)
**Problem**: Creating DMatrix on every prediction is expensive
```python
# ❌ Current (SLOW)
dmatrix = xgb.DMatrix(features)
preds = booster.predict(dmatrix)
```

**Fix**: Use inplace_predict
```python
# ✅ Fixed (FAST)
preds = booster.inplace_predict(features, validate_features=False)
# Also apply best iteration if available:
if self.best_iteration:
    preds = booster.inplace_predict(
        features, 
        iteration_range=(0, self.best_iteration + 1),
        validate_features=False
    )
```

#### Issue: Model Type Detection Fails (Lines 12-18)
**Problem**: Detecting Booster by absence of predict_proba misclassifies XGBRegressor
```python
# ❌ Current (BROKEN)
if not hasattr(model, 'predict_proba'):  # XGBRegressor also lacks this!
    # Assumes it's a Booster, but could be sklearn regressor
```

**Fix**: Use explicit type checks
```python
# ✅ Fixed
import xgboost as xgb

if isinstance(model, xgb.Booster):
    # Booster-specific logic
elif isinstance(model, (xgb.XGBClassifier, xgb.XGBRegressor)):
    # sklearn-style logic
else:
    raise TypeError(f"Unsupported model type: {type(model)}")
```

### 🟡 Important Issues

#### Issue: Not Using Best Iteration (Lines 32-34, 70-72)
**Problem**: Predictions use all trees, not early-stopped point
```python
# ❌ Current
model.predict(X)  # Uses all trees
```

**Fix**: Store and use best_iteration
```python
# ✅ Fixed
# In trainer, save to metadata:
metadata['best_iteration'] = model.best_iteration

# In inference:
if hasattr(model, 'best_iteration') and model.best_iteration:
    preds = model.predict(X, iteration_range=(0, model.best_iteration + 1))
```

#### Issue: Returns Hard Labels Instead of Probabilities (Lines 64-66)
**Problem**: Trainer predict() returns binary labels, hiding probability info
```python
# ❌ Current
def predict(self, X):
    preds = model.predict(X)
    return (preds > 0.5).astype(int)  # Lost probability!
```

**Fix**: Return probabilities by default
```python
# ✅ Fixed
def predict(self, X, return_proba=True):
    preds = model.predict(X, iteration_range=(0, self.best_iteration + 1))
    if return_proba:
        return preds.astype(np.float32)  # Probabilities
    else:
        return (preds > 0.5).astype(int)  # Labels only if requested
```

#### Issue: GPU Version Compatibility (Lines 68-75)
**Problem**: GPU params differ between XGBoost 1.x and 2.x
```python
# ❌ Current (only works on 2.x)
params['tree_method'] = 'hist'
params['device'] = 'cuda:0'
```

**Fix**: Version-aware GPU setup
```python
# ✅ Fixed
import xgboost as xgb

xgb_version = tuple(map(int, xgb.__version__.split('.')[:2]))

if xgb_version >= (2, 0):
    params['tree_method'] = 'hist'
    params['device'] = f'cuda:{gpu_id}'
else:  # XGBoost 1.x
    params['tree_method'] = 'gpu_hist'
    params['predictor'] = 'gpu_predictor'
```

## 2. LightGBM Issues (training/ISSUES_LIGHTGBM.txt)

### 🔴 Critical Issues

#### Issue: Model Type Misclassification (Lines 10-20)
**Problem**: Same as XGBoost - LGBMRegressor misidentified as Booster
```python
# ❌ Current
if hasattr(model, 'predict') and not hasattr(model, 'predict_proba'):
    # LGBMRegressor wrongly treated as Booster
```

**Fix**: Explicit type checks
```python
# ✅ Fixed
from lightgbm import Booster, LGBMClassifier, LGBMRegressor

if isinstance(model, Booster):
    # Booster logic
elif isinstance(model, (LGBMClassifier, LGBMRegressor)):
    # sklearn API logic
```

### 🟡 Important Issues

#### Issue: Returns Hard Classes by Default (Lines 55-64)
**Problem**: Like XGBoost, returns labels instead of probabilities
```python
# ❌ Current
if objective == "binary":
    return (proba > 0.5).astype(int)
```

**Fix**: Return probabilities
```python
# ✅ Fixed
def predict(self, X, return_labels=False):
    proba = model.predict(X, num_iteration=model.best_iteration)
    proba = proba.astype(np.float32)  # Ensure float32
    
    if return_labels:
        return (proba > 0.5).astype(int)
    return proba  # Default: return probabilities
```

#### Issue: Data Type Inconsistency (Lines 78-82)
**Problem**: Training uses float64, inference expects float32
```python
# ❌ Current
X = df.to_numpy()  # Returns float64
# Later in inference: expects float32
```

**Fix**: Consistent float32
```python
# ✅ Fixed
# In prepare_data:
X = df.to_numpy().astype(np.float32)

# Record in metadata:
metadata['dtype'] = 'float32'
```

## 3. Actor Issues (actors/ISSUES.txt)

### 🔴 Critical Duplication & Performance Issues

#### Issue: Two Classes Named MLSignalActor (Lines 18-25)
**Problem**: Name collision causes import confusion
```python
# ❌ Current
# In base.py:
class MLSignalActor(Actor): ...

# In signal.py:
class MLSignalActor(Actor): ...  # SAME NAME!
```

**Fix**: Rename for clarity
```python
# ✅ Fixed
# In base.py:
class SimpleMLSignalActor(Actor):
    """Basic ML signal actor for examples"""

# In signal.py:
class MLSignalActor(Actor):
    """Production ML signal actor with full features"""
```

#### Issue: Model Loaded Twice (Lines 14-17)
**Problem**: Base class loads model, then subclass loads again
```python
# ❌ Current
class BaseMLInferenceActor:
    def _load_model_with_metadata(self):
        self._model = ProductionModelLoader.load(...)  # First load
        self._load_model()  # Calls subclass method
        
class PickleMLInferenceActor(BaseMLInferenceActor):
    def _load_model(self):
        self._model = pickle.load(...)  # Second load (overwrites!)
```

**Fix**: Conditional loading
```python
# ✅ Fixed
class BaseMLInferenceActor:
    def _load_model_with_metadata(self):
        if not getattr(self, '_skip_base_load', False):
            self._model = ProductionModelLoader.load(...)
        self._load_model()  # For subclass-specific setup

class PickleMLInferenceActor(BaseMLInferenceActor):
    _skip_base_load = True  # Skip base loading
    
    def _load_model(self):
        if self.config.allow_pickle:  # Security check
            self._model = pickle.load(...)
        else:
            raise SecurityError("Pickle disabled in production")
```

#### Issue: Pickle in Production (Lines 18-20)
**Problem**: Direct pickle.load is a security risk
```python
# ❌ Current
with open(path, 'rb') as f:
    self._model = pickle.load(f)  # SECURITY RISK!
```

**Fix**: Gate behind config flag
```python
# ✅ Fixed
class MLActorConfig:
    allow_pickle: bool = False  # Default: disabled

class PickleMLInferenceActor:
    def _load_model(self):
        if not self.config.allow_pickle:
            raise SecurityError(
                "Pickle loading disabled for security. "
                "Use ONNX or native formats in production."
            )
        self.log.warning("Loading pickle model - security risk!")
        # ... pickle loading ...
```

### 🟡 Important Issues

#### Issue: Fake Latency Measurements (Lines 40-42)
**Problem**: Using hardcoded latency instead of measuring
```python
# ❌ Current
feature_time_ns = 500_000  # Hardcoded!
inference_time_ns = total_time - feature_time_ns  # Meaningless
```

**Fix**: Measure actual times
```python
# ✅ Fixed
def _predict_with_timing(self, features):
    # Measure feature computation
    feature_start = time.perf_counter_ns()
    features = self._compute_features(bar)
    feature_time_ns = time.perf_counter_ns() - feature_start
    
    # Measure inference
    inference_start = time.perf_counter_ns()
    prediction = self._model.predict(features)
    inference_time_ns = time.perf_counter_ns() - inference_start
    
    # Record real measurements
    self._record_performance(feature_time_ns, inference_time_ns)
    return prediction
```

#### Issue: Confidence Semantics Wrong (Lines 31-38)
**Problem**: Using abs(prediction) as confidence for regression
```python
# ❌ Current
if model_type == "regression":
    confidence = abs(prediction)  # Meaningless!
```

**Fix**: Proper confidence handling
```python
# ✅ Fixed
def _get_confidence(self, prediction, model_type):
    if model_type == "classification":
        # For binary: confidence is the probability itself
        confidence = prediction if isinstance(prediction, float) else max(prediction)
    elif model_type == "regression":
        # For regression: need separate calibration
        if self.calibrator:
            confidence = self.calibrator.predict_proba(prediction)
        else:
            confidence = 1.0  # Or use uncertainty quantification
    return confidence
```

## 4. Feature Engineering Issues (features/ISSUES.txt)

### 🔴 Critical Bar Construction Error

#### Issue: Wrong Bar Construction (Lines 39-42)
**Problem**: Creating Bar with raw floats instead of proper objects
```python
# ❌ Current (RUNTIME ERROR!)
bar = Bar(open, high, low, close, volume)  # Wrong! Expects Price/Quantity objects
```

**Fix**: Use proper Bar construction or raw updates
```python
# ✅ Fixed Option 1: Proper Bar objects
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.data import Bar

bar = Bar(
    bar_type=self.bar_type,
    open=Price.from_str(str(open)),
    high=Price.from_str(str(high)),
    low=Price.from_str(str(low)),
    close=Price.from_str(str(close)),
    volume=Quantity.from_str(str(volume)),
    ts_event=ts_event,
    ts_init=ts_init
)

# ✅ Fixed Option 2: Use raw indicator updates (faster)
# Don't create Bar at all, use:
atr.update_raw(high, low, close)
bb.update_raw(close)
```

### 🟡 Important Issues

#### Issue: Feature Name Mismatch (Lines 43-48)
**Problem**: Internal names don't match exported names
```python
# ❌ Current
# Registry says: "spread_tightness"
# But code returns: "spread_tightness_raw"
```

**Fix**: Consistent naming
```python
# ✅ Fixed
def get_values(self):
    raw_values = indicator.get_values()
    # Map internal to external names
    return {
        "spread_tightness": raw_values["spread_tightness_raw"],
        "spread_trend": raw_values["spread_trend_raw"]
    }
```

#### Issue: Dtype Inconsistency (Lines 49-52)
**Problem**: Buffer allocated as float64, but inference needs float32
```python
# ❌ Current
self._feature_buffer = np.zeros(n_features, dtype=np.float64)
```

**Fix**: Use float32 everywhere
```python
# ✅ Fixed
self._feature_buffer = np.zeros(n_features, dtype=np.float32)
# And ensure scaler outputs are cast:
scaled = self.scaler.transform(features).astype(np.float32)
```

## 5. Strategy Issues (strategies/ISSUES.txt)

### 🔴 Critical Issues

#### Issue: Missing Feature Engineer in OptimizedMLSignalActor (Lines 27-30)
**Problem**: References self._feature_engineer without creating it
```python
# ❌ Current
class OptimizedMLSignalActor(BaseMLInferenceActor):
    def _compute_features(self):
        return self._feature_engineer.compute()  # AttributeError!
```

**Fix**: Initialize feature engineer
```python
# ✅ Fixed
class OptimizedMLSignalActor(BaseMLInferenceActor):
    def __init__(self, config):
        super().__init__(config)
        self._feature_engineer = FeatureEngineer(config.feature_config)
        self._indicator_manager = IndicatorManager()
```

## 6. Consolidated Priority Fix List

### 🔴 Must Fix NOW (Breaking/Security)
1. **Bar construction error** - Runtime crash
2. **Duplicate MLSignalActor** - Import confusion  
3. **Pickle in production** - Security vulnerability
4. **Missing feature engineer** - AttributeError

### 🟡 Fix This Week (Performance/Correctness)
1. **XGBoost inplace_predict** - 10x+ speedup
2. **Best iteration not used** - Wrong predictions
3. **Model type detection** - Misclassification
4. **Return probabilities** - Not labels
5. **Float32 consistency** - Memory/speed
6. **Real latency measurement** - Not placeholders

### 🟢 Fix Soon (Polish/Monitoring)
1. **Schema validation** - Feature parity
2. **GPU version compatibility** - Cross-version support
3. **Calibration** - Better probabilities
4. **Hot reload** - Model updates
5. **Unified metrics** - Consistent monitoring

## 7. File-by-File Changes Needed

### ml/models/xgboost_model.py
- Fix isinstance checks
- Add inplace_predict
- Use best_iteration
- Return float32

### ml/models/lightgbm_model.py
- Fix isinstance checks  
- Use best_iteration
- Return float32

### ml/training/xgboost.py
- Return probabilities by default
- GPU version compatibility
- Save best_iteration to metadata
- Add calibration option

### ml/training/lightgbm.py
- Return probabilities by default
- Cast to float32 in prepare_data
- Save complete metadata

### ml/actors/base.py
- Rename SimpleMLSignalActor
- Fix double loading
- Add allow_pickle config
- Measure real latencies

### ml/actors/signal.py
- Keep as MLSignalActor (production)
- Fix feature engineer init
- Real latency measurements
- Proper confidence handling

### ml/features/engineering.py
- Fix Bar construction or use raw updates
- Consistent feature naming
- Float32 buffers
- Handle missing columns in Polars

## 8. Quick Win Implementation Order

### Day 1: Critical Fixes (2-3 hours)
```bash
# 1. Rename duplicate classes
# 2. Fix Bar construction 
# 3. Add pickle security check
# 4. Initialize feature engineer
```

### Day 2: Performance (3-4 hours)
```bash
# 1. Implement inplace_predict
# 2. Add best_iteration handling
# 3. Fix model type detection
# 4. Standardize float32
```

### Day 3: Correctness (2-3 hours)
```bash
# 1. Return probabilities
# 2. Real latency measurement
# 3. GPU compatibility
# 4. Feature name consistency
```

### Day 4: Validation (2-3 hours)
```bash
# 1. Schema validation
# 2. Metadata completeness
# 3. Testing updates
# 4. Documentation
```

## 9. Testing After Fixes

```python
# Test suite to verify fixes
def test_all_fixes():
    # 1. Test XGBoost inplace_predict
    assert inference_time < 2.0  # ms
    
    # 2. Test best iteration
    assert model.predict with best_iter != without
    
    # 3. Test probability output
    assert 0 <= prediction <= 1
    
    # 4. Test float32 consistency
    assert features.dtype == np.float32
    assert predictions.dtype == np.float32
    
    # 5. Test no pickle in prod
    with pytest.raises(SecurityError):
        PickleMLInferenceActor(allow_pickle=False)
    
    # 6. Test feature parity
    assert max_diff < 1e-10
```

---

*This consolidation covers ALL technical issues from the ISSUES.txt files with concrete, actionable fixes.*