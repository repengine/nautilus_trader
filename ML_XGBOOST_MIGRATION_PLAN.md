# XGBoost Trainer Migration Plan

## Executive Summary

This document outlines the migration of the XGBoost trainer from the OLD implementation to the new ML architecture, maintaining functionality while conforming to Nautilus patterns.

## 1. OLD XGBoost Trainer Analysis

### Key Features

1. **Unified Single/Multi-Asset Support**
   - Single asset: Direct DataFrame processing
   - Multi-asset: Dictionary of DataFrames with cross-sectional features
   - Sector-based relative features

2. **Advanced ML Features**
   - SHAP value computation with interaction analysis
   - Multiple feature importance methods (native, SHAP, permutation)
   - Feature decay detection over time
   - Monotonic constraints for interpretability
   - GPU acceleration support

3. **Hyperparameter Optimization**
   - Optuna integration with custom objectives
   - TimeSeriesSplit cross-validation
   - Trading metrics (Sharpe ratio) as optimization target

4. **MLflow Integration**
   - Experiment tracking
   - Model versioning
   - Artifact storage
   - Feature importance reports

5. **Resource Management**
   - GPU/CPU allocation
   - Memory management
   - Training time limits

### Dependencies

```python
# Core ML Libraries
xgboost >= 1.7.0
shap >= 0.40.0
scikit-learn >= 1.0.0
optuna >= 3.0.0

# Data Processing
polars >= 0.15.0
pandas >= 1.5.0  # Used for compatibility
numpy >= 1.23.0

# Visualization
matplotlib >= 3.5.0

# ML Operations
mlflow >= 2.0.0
```

## 2. Migration Strategy

### Phase 1: Core XGBoost Trainer (Week 1)

#### Step 1.1: Create XGBoostTrainer Class

```python
# ml/training/xgboost.py
from ml.training.base import BaseMLTrainer
from ml.config.base import MLTrainingConfig, MLFeatureConfig

class XGBoostTrainer(BaseMLTrainer):
    """XGBoost trainer for Nautilus ML."""
```

#### Step 1.2: Migrate Core Training Logic

- Implement `prepare_data()` method
- Implement `_train_model()` method
- Remove MLflow dependencies (make optional)
- Replace pandas with polars throughout
- Use msgspec configurations

#### Step 1.3: Implement Feature Engineering

- Port single-asset feature preparation
- Ensure compatibility with `ml.features.engineering`
- Maintain feature parity checks

### Phase 2: Multi-Asset Support (Week 2)

#### Step 2.1: Port Cross-Sectional Features

- Migrate `_prepare_multi_asset_data()`
- Migrate `_add_cross_sectional_features()`
- Implement sector-relative calculations

#### Step 2.2: Portfolio-Level Metrics

- Port `_calculate_multi_asset_sharpe()`
- Implement metadata tracking for alignment
- Add correlation-aware position sizing

### Phase 3: Advanced Features (Week 3)

#### Step 3.1: SHAP Integration

- Make SHAP optional dependency
- Port `_compute_shap_values()`
- Port `_compute_feature_interactions()`
- Migrate visualization code

#### Step 3.2: Feature Importance Tracking

- Port `_track_feature_importance()`
- Port `_check_feature_decay()`
- Implement importance report generation

#### Step 3.3: Monotonic Constraints

- Port constraint handling
- Ensure XGBoost parameter compatibility

### Phase 4: Optimization & Resource Management (Week 4)

#### Step 4.1: Optuna Integration

- Create separate OptunaTuner class
- Implement objective functions
- Port multi-asset optimization

#### Step 4.2: Resource Management

- Simplify resource allocation
- Remove complex GPU management
- Use context managers for cleanup

## 3. Key Changes for New Architecture

### Configuration Changes

```python
# OLD: Dictionary-based config
config = {
    "instrument": "AAPL",
    "multi_asset": False,
    "enable_shap": True,
}

# NEW: msgspec-based config
from ml.config.xgboost import XGBoostTrainingConfig

config = XGBoostTrainingConfig(
    data_source="path/to/data",
    instrument_id=InstrumentId("AAPL.NASDAQ"),
    enable_shap=True,
)
```

### Hot/Cold Path Separation

```python
# COLD PATH (Training) - OK to use heavy libraries
- Polars for data processing
- SHAP for explainability
- Optuna for optimization
- MLflow for tracking

# HOT PATH (Inference) - Must be lightweight
- NumPy only
- Pre-computed features
- ONNX runtime
- No blocking operations
```

### Integration with BaseMLTrainer

```python
class XGBoostTrainer(BaseMLTrainer):
    def prepare_data(self, data, target_col="target"):
        # Reuse feature engineering from base
        # Add XGBoost-specific preprocessing
        pass

    def _train_model(self, X_train, y_train, X_val, y_val, **kwargs):
        # Core XGBoost training logic
        # Return model and metrics
        pass
```

## 4. Configuration Classes

```python
# ml/config/xgboost.py
from ml.config.base import MLTrainingConfig
from nautilus_trader.common.config import PositiveInt, NonNegativeFloat

class XGBoostTrainingConfig(MLTrainingConfig, kw_only=True, frozen=True):
    """Configuration for XGBoost training."""

    # XGBoost parameters
    n_estimators: PositiveInt = 100
    max_depth: PositiveInt = 6
    learning_rate: PositiveFloat = 0.3
    subsample: PositiveFloat = 1.0
    colsample_bytree: PositiveFloat = 1.0

    # GPU support
    tree_method: str = "hist"  # "hist" or "gpu_hist"
    gpu_id: NonNegativeInt = 0

    # Advanced features
    enable_shap: bool = False
    monotonic_constraints: dict[str, int] | None = None

    # Multi-asset
    multi_asset: bool = False
    sector_map: dict[str, str] | None = None

    # Optimization
    optimize_hyperparams: bool = False
    n_trials: PositiveInt = 100
```

## 5. Implementation Priority

### Must Have (P0)

1. Basic XGBoost training
2. Single-asset support
3. Standard ML metrics
4. Model serialization
5. Feature engineering integration

### Should Have (P1)

1. Multi-asset support
2. Cross-sectional features
3. Trading metrics (Sharpe)
4. GPU acceleration
5. Cross-validation

### Nice to Have (P2)

1. SHAP explainability
2. Feature decay detection
3. Optuna optimization
4. MLflow integration
5. Interaction plots

## 6. Testing Requirements

### Unit Tests

```python
# ml/tests/unit/test_xgboost_trainer.py
- Test single-asset training
- Test multi-asset training
- Test feature preparation
- Test metric calculations
- Test model serialization
```

### Integration Tests

```python
# ml/tests/integration/test_xgboost_integration.py
- Test with real Nautilus data
- Test feature parity
- Test inference latency
- Test memory stability
```

### Performance Tests

```python
# ml/tests/performance/test_xgboost_performance.py
- Training time benchmarks
- Inference latency < 2ms
- Memory usage tracking
- Feature computation < 500μs
```

## 7. Migration Risks & Mitigations

### Risk 1: Feature Parity

- **Risk**: Training/inference feature mismatch
- **Mitigation**: Strict parity tests with 1e-10 tolerance

### Risk 2: Performance Regression

- **Risk**: New implementation slower
- **Mitigation**: Benchmark against OLD implementation

### Risk 3: Dependency Conflicts

- **Risk**: Version incompatibilities
- **Mitigation**: Pin all dependency versions

### Risk 4: API Breaking Changes

- **Risk**: Existing users affected
- **Mitigation**: Provide migration guide and compatibility layer

## 8. Success Criteria

1. **Functional Parity**: All OLD features work in NEW
2. **Performance**: No regression > 20%
3. **Test Coverage**: ≥ 90% for ML modules
4. **Documentation**: Complete API docs and examples
5. **Integration**: Works with ML actors/strategies

## 9. Timeline

- **Week 1**: Core trainer implementation
- **Week 2**: Multi-asset support
- **Week 3**: Advanced features
- **Week 4**: Optimization & testing
- **Week 5**: Documentation & examples

## 10. Next Steps

1. Create `ml/training/xgboost.py`
2. Create `ml/config/xgboost.py`
3. Implement basic single-asset training
4. Add comprehensive tests
5. Benchmark against OLD implementation
