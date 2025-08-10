# Critical Fixes Checklist - ML Module

## 🔴 STOP! Fix These First (Runtime Errors & Security)

### 1. ❌ Bar Construction Runtime Error
**File**: `ml/features/engineering.py` (Version B)
**Line**: ~40
```python
# BROKEN - Will crash at runtime!
bar = Bar(open, high, low, close, volume)  # ❌ TypeError
```
**Fix**:
```python
# Option A: Don't create Bar, use raw updates
atr.update_raw(high, low, close)
bb.update_raw(close)

# Option B: Create proper Bar (slower)
from nautilus_trader.model.objects import Price, Quantity
bar = Bar(
    bar_type=self.bar_type,
    open=Price.from_str(str(open)),
    high=Price.from_str(str(high)),
    # ... etc
)
```

### 2. ❌ Duplicate MLSignalActor Classes
**Files**: `ml/actors/base.py` AND `ml/actors/signal.py`
```python
# Both files have:
class MLSignalActor(Actor):  # NAME COLLISION!
```
**Fix**:
```python
# In base.py:
class SimpleMLSignalActor(Actor):  # Renamed

# In signal.py:
class MLSignalActor(Actor):  # Keep for production
```

### 3. ❌ Pickle Security Vulnerability
**File**: `ml/actors/base.py` - PickleMLInferenceActor
```python
# SECURITY RISK!
self._model = pickle.load(f)  # Arbitrary code execution
```
**Fix**:
```python
class MLActorConfig:
    allow_pickle: bool = False  # Default OFF

def _load_model(self):
    if not self.config.allow_pickle:
        raise SecurityError("Pickle disabled in production")
```

### 4. ❌ Missing Feature Engineer (AttributeError)
**File**: `ml/actors/signal.py` - OptimizedMLSignalActor
```python
# BROKEN - Will crash!
def _compute_features(self):
    return self._feature_engineer.compute()  # AttributeError!
```
**Fix**:
```python
def __init__(self, config):
    super().__init__(config)
    self._feature_engineer = FeatureEngineer(config.feature_config)  # Add this!
```

## 🟡 Fix These Next (10x Performance & Wrong Results)

### 5. ⚠️ XGBoost DMatrix Every Tick (10x+ Slower)
**Files**: `ml/models/xgboost_model.py`, `ml/actors/signal.py`
```python
# SLOW (current):
dmatrix = xgb.DMatrix(features)  # Allocation every tick!
preds = booster.predict(dmatrix)
```
**Fix**:
```python
# FAST:
preds = booster.inplace_predict(features, validate_features=False)
if self.best_iteration:
    preds = booster.inplace_predict(
        features,
        iteration_range=(0, self.best_iteration + 1)
    )
```

### 6. ⚠️ Model Type Detection Wrong
**Files**: `ml/models/xgboost_model.py`, `ml/models/lightgbm_model.py`
```python
# BROKEN:
if not hasattr(model, 'predict_proba'):
    # XGBRegressor wrongly treated as Booster!
```
**Fix**:
```python
# Use explicit type checks:
if isinstance(model, xgb.Booster):
    # Booster logic
elif isinstance(model, (xgb.XGBClassifier, xgb.XGBRegressor)):
    # sklearn API logic
```

### 7. ⚠️ Best Iteration Ignored (Wrong Predictions)
**Files**: All trainer and model files
```python
# Using all trees instead of early-stopped point
model.predict(X)  # Should use best_iteration!
```
**Fix**:
```python
# Save in trainer:
metadata['best_iteration'] = model.best_iteration

# Use in inference:
preds = model.predict(X, iteration_range=(0, best_iteration + 1))
```

### 8. ⚠️ Returns Labels Instead of Probabilities
**Files**: `ml/training/xgboost.py`, `ml/training/lightgbm.py`
```python
# WRONG - Loses information:
return (preds > 0.5).astype(int)  # Binary labels
```
**Fix**:
```python
def predict(self, X, return_proba=True):
    preds = model.predict(X)
    if return_proba:
        return preds.astype(np.float32)  # Probabilities!
    return (preds > 0.5).astype(int)  # Only if requested
```

## 🟢 Important But Not Urgent


### 10. Fake Latency Measurements
- Using hardcoded 500μs instead of measuring
- Fix: Use actual time.perf_counter_ns()

### 11. No Schema Validation
- Features could mismatch silently
- Fix: Validate names, order, dtype, hash

### 12. GPU Version Compatibility
- XGBoost 1.x vs 2.x params differ
- Fix: Version-aware configuration

## Quick Test to Verify Fixes

```bash
# After making changes, run this test script:
python -c "
from ml.actors.base import SimpleMLSignalActor  # Should work
from ml.actors.signal import MLSignalActor      # Different class

# Test XGBoost speed
import time
start = time.time()
model.predict(features)  # Should use inplace_predict
assert time.time() - start < 0.002  # <2ms

# Test probabilities
preds = trainer.predict(X)
assert 0 <= preds.min() <= preds.max() <= 1  # Probabilities!

# Test security
try:
    PickleMLInferenceActor(allow_pickle=False)
    assert False, 'Should have raised SecurityError'
except SecurityError:
    pass  # Good!

print('✅ All critical fixes verified!')
"
```

## Files to Change (Priority Order)

1. **ml/features/engineering.py** - Fix Bar construction
2. **ml/actors/base.py** - Rename class, add pickle security
3. **ml/actors/signal.py** - Initialize feature engineer
4. **ml/models/xgboost_model.py** - inplace_predict, type checks
5. **ml/models/lightgbm_model.py** - Type checks, best_iteration
6. **ml/training/xgboost.py** - Return probabilities, GPU compat
7. **ml/training/lightgbm.py** - Return probabilities, float32

## Time Estimate

- Critical fixes (1-4): **2 hours**
- Performance fixes (5-8): **3 hours**
- Nice-to-have (9-12): **2 hours**

**Total: 7 hours to fix all issues**

---

*Start with the critical fixes - they're blocking everything else!*
