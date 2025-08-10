# ML Integration Overview - Nautilus Trader

## Executive Summary

This document consolidates and organizes the ML integration architecture for Nautilus Trader, addressing the teacher-student distillation pipeline, implementation issues, and providing a clear path forward.

## 1. Core Architecture: Teacher-Student Distillation Pipeline

### The Pipeline Flow
```
1. Teacher Training (Offline, T+1)
   ↓
2. Student Training (Distillation)
   ↓
3. ONNX Export & Validation
   ↓
4. Registry & Deployment
   ↓
5. Inference Actor (Hot Path)
   ↓
6. Signal Publishing
   ↓
7. Strategy Execution
```

### Why Teacher-Student Distillation?

**Teacher Models** (Cold Path):
- Can use rich features (L2/L3 depth, cross-asset data, long sequences)
- Can be computationally expensive (TFT, Transformers, PyTorch Geometric)
- Run offline with full historical context
- Output: High-quality predictions/probabilities

**Student Models** (Hot Path):
- Must use only L1 features available in real-time
- Must be fast (<2ms inference)
- Typically LightGBM, XGBoost, or small MLPs
- Learn to approximate teacher outputs
- Export to ONNX for deployment

### Three Distillation Recipes

#### 1. Soft-Label Cross-Entropy (Cleanest)
```python
# Teacher probability q ∈ (0,1) as target
Loss = -[q*log(p) + (1-q)*log(1-p)]
```
- Best for neural students
- Good calibration
- LightGBM may need label smoothing trick

#### 2. Logit Regression (Best for Trees)
```python
# Train on teacher logits
z_T = logit(q) = log(q/(1-q))
Loss = MSE(z_student, z_teacher)
# At inference: p = sigmoid(z_student)
```
- Perfect for LightGBM/XGBoost
- Very stable training
- Need final calibration step

#### 3. Joint Loss (Reduces Bias)
```python
Loss = α*CE(p_student, y_true) + (1-α)*CE(p_student, q_teacher)
```
- Combines distillation with ground truth
- Improves real-world calibration
- More complex implementation

## 2. Critical Architecture Decisions

### Hot/Cold Path Separation (MUST FOLLOW)

**HOT PATH** (actors, signal generation):
- Numpy-only, NO Polars/pandas
- Pre-allocated float32 arrays
- Zero allocations per tick
- <500μs features, <2ms inference, <5ms end-to-end
- Models loaded once at initialization
- Use Nautilus's optimized indicators

**COLD PATH** (training, data prep):
- Polars/pandas allowed
- Heavy computations OK
- Batch processing patterns
- Feature engineering flexibility

### Security: No Pickle in Production
- **ProductionModelLoader**: ONNX, XGBoost JSON, LightGBM text
- **Pickle explicitly forbidden** with SecurityError
- Deprecation warnings guide migration
- Model validation before deployment

### Float32 Dtype Policy
- **Training**: Can use float64 internally for stability
- **Inference**: MUST return float32
- **ONNX**: Natively uses float32
- **Buffers**: Pre-allocated as float32

## 3. Current Implementation Issues (Prioritized)

### 🔴 Critical (Fix Immediately)

1. **Duplicate MLSignalActor Classes**
   - Two classes with same name in different modules
   - **Fix**: Rename to SimpleMLSignalActor and MLSignalActor

2. **XGBoost DMatrix Per Tick**
   - Creating DMatrix on every prediction (slow)
   - **Fix**: Use `booster.inplace_predict()` with `iteration_range`

3. **Wrong Bar Construction**
   - Creating Bar with raw floats instead of proper objects
   - **Fix**: Use proper Bar objects or stay with raw indicator updates

4. **Schema Validation Missing**
   - No feature validation between training and inference
   - **Fix**: Validate feature names, order, dtype, hash before scoring

### 🟡 Important (Fix Soon)

1. **Prediction Semantics**
   - Returning labels instead of probabilities
   - **Fix**: Always return probabilities, threshold in strategy

2. **Best Iteration Not Used**
   - Models use all trees instead of early-stopped point
   - **Fix**: Store and use best_iteration in metadata

3. **Feature Parity Issues**
   - Different indicator update methods in batch vs online
   - **Fix**: Use same API (handle_bar) in both paths

4. **Dtype Inconsistency**
   - Mix of float32/float64 across pipeline
   - **Fix**: Standardize on float32 end-to-end

### 🟢 Nice to Have

1. **Calibration Missing**
   - No Platt/Isotonic calibration
   - **Fix**: Add optional calibration step

2. **Hot Reload Stubbed**
   - Version checking returns False
   - **Fix**: Implement atomic symlink swapping

3. **Metrics Duplication**
   - Different metric names across actors
   - **Fix**: Unify namespace (nautilus_ml_*)

## 4. Feature Engineering Issues

### Key Problems
1. **Indicator API Mismatch**: `update_raw()` vs `handle_bar()`
2. **Polars Missing Columns**: Crashes on select during warmup
3. **Return Type Drift**: float32 vs float64 inconsistency
4. **Feature Name Mismatch**: spread_tightness vs spread_tightness_raw

### Solutions
```python
# Unified approach
- Use handle_bar(bar) consistently
- Pre-allocate float32 buffers
- Validate schema hash at load
- Store capability flags in metadata
```

## 5. Model Management Architecture

### Model Registry Structure
```yaml
model_id: es_dir_3s_student_v3
onnx_path: s3://models/es_dir_3s_student_v3/model.onnx
meta_path: s3://models/es_dir_3s_student_v3/meta.json
feature_schema_hash: "a94a8fe5..."
output_schema: {kind: binary_proba, shape: [None, 1]}
best_iteration: 150
deployed_at: "2025-08-06T01:30:00Z"
```

### Required Metadata
- feature_names (ordered)
- feature_dtypes
- schema_hash
- output_schema
- best_iteration (trees)
- calibration_params
- training_date_range
- instruments

## 6. Actionable Next Steps

### Week 1: Critical Fixes
1. Fix duplicate MLSignalActor names
2. Implement XGBoost inplace_predict
3. Fix Bar construction or use raw updates
4. Add schema validation at model load

### Week 2: Pipeline Completion
1. Implement teacher model (TFT/N-HiTS)
2. Implement student distillation (LightGBM)
3. Add ONNX export with metadata
4. Create acceptance tests (framework vs ORT)

### Week 3: Production Readiness
1. Standardize float32 everywhere
2. Add feature parity validation
3. Implement hot reload with atomic swaps
4. Unify metrics and monitoring

### Week 4: Advanced Features
1. Add calibration (Platt/Isotonic)
2. Implement multi-horizon support
3. Add A/B testing capability
4. Complete registry implementation

## 7. Testing Protocol Summary

### Test Categories
1. **Contract Tests**: Behavioral requirements all implementations must follow
2. **Unit Tests**: Individual components in isolation (<100ms)
3. **Integration Tests**: Component interactions with real models
4. **Performance Tests**: Latency and throughput validation

### Key Requirements
- ≥90% coverage for ML module
- Feature parity tolerance: 1e-10
- P99 latency < 5ms
- Zero allocations in hot path
- Security: Reject pickle files

## 8. Quick Reference Commands

```bash
# Run ML tests
pytest ml/tests/unit/ -v
pytest ml/tests/contracts/ -v
pytest ml/tests/integration/ -v

# Check feature parity
python -m ml.features.validation

# Performance benchmarks
pytest ml/tests/performance/ --benchmark-only

# Pre-commit checks
make pre-commit
make format
```

## 9. Common Pitfalls to Avoid

1. **Never** use pandas in hot path
2. **Never** load models during runtime
3. **Never** use pickle in production
4. **Always** validate feature schema
5. **Always** use float32 for inference
6. **Always** test feature parity

## 10. Model Support Matrix

| Library | ONNX Export | Deployment Path |
|---------|-------------|-----------------|
| XGBoost | ✅ Native + ONNX | Direct or ONNX |
| LightGBM | ✅ Native + ONNX | Direct or ONNX |
| PyTorch | ✅ torch.onnx.export | ONNX Runtime |
| PyTorch Forecasting | ✅ Via PyTorch | Export nn.Module |
| Statsmodels | ❌ | Teacher → Distill |
| PyTorch Geometric | ❌ | Teacher → Distill |
| River | ❌ | Online adapter |

---

*This document consolidates insights from the testing protocol, architectural decisions, and implementation issues. Use it as your primary reference for ML integration.*
