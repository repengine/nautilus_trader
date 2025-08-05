# ML Migration Plan: nautilus_ml → Nautilus Trader ml/

## Executive Summary

This plan outlines the migration of the sophisticated ML training infrastructure from `OLD/trade/nautilus_ml` into a new `ml/` folder that follows Nautilus Trader architectural principles while preserving the advanced MLOps capabilities.

**Key Principles:**

- Strict hot/cold path separation
- Feature engineering consistency between training and inference
- MLflow integration for model lifecycle management
- Comprehensive testing with 90%+ coverage requirement

---

## Phase 1: Foundation Setup (Week 1-2)

### 1.1 Create ML Directory Structure

```bash
ml/
├── __init__.py
├── actors/                    # Hot path - Real-time inference
│   ├── __init__.py
│   ├── base.py               # BaseInferenceActor (from GenericInferenceActor)
│   ├── signal_actor.py       # ML signal generation
│   └── portfolio_actor.py    # Portfolio construction
├── strategies/               # Hot path - Signal aggregation
│   ├── __init__.py
│   ├── ensemble.py          # EnsembleMLStrategy (migrate)
│   └── base.py              # Base ML strategy
├── training/                # Cold path - Model development
│   ├── __init__.py
│   ├── base_trainer.py      # BaseTrainer (migrate)
│   ├── xgboost_trainer.py   # UnifiedXGBoostTrainer (migrate)
│   ├── lightgbm_trainer.py  # UnifiedLightGBMTrainer (migrate)
│   └── neural_trainer.py    # NeuralForecastTrainer (migrate)
├── features/                # Shared - Feature engineering
│   ├── __init__.py
│   ├── engineering.py       # FeatureEngineerV2 (migrate)
│   └── validation.py        # Feature parity tests
├── data/                    # Shared - Data access
│   ├── __init__.py
│   └── loaders.py          # UnifiedNautilusDataLoader (adapt)
├── models/                  # Shared - Model artifacts
│   ├── __init__.py
│   └── registry.py         # ModelRegistry (migrate)
├── config/                  # Shared - Configuration
│   ├── __init__.py
│   └── ml_config.py        # ML configurations using msgspec
├── utils/                   # Shared utilities
│   ├── __init__.py
│   ├── mlflow.py           # MLflowManager (migrate)
│   └── metrics.py          # Trading metrics
└── tests/                  # Comprehensive test suite
    ├── test_feature_parity.py  # CRITICAL: Feature validation
    ├── test_trainers.py
    └── test_integration.py
```

### 1.2 Core Infrastructure Tasks

#### Delete Problematic Code

- [ ] Remove `/trainers/` directory (contains only stubs)
- [ ] Clean up duplicate/conflicting implementations

#### Create Base Classes

- [ ] Port `BaseTrainer` with Nautilus-compatible modifications
- [ ] Create `BaseInferenceActor` from `GenericInferenceActor`
- [ ] Implement `BaseMLStrategy` following Nautilus patterns

#### Set Up Configuration

- [ ] Convert Pydantic configs to msgspec
- [ ] Create ML-specific configuration classes
- [ ] Set up environment-based configuration loading

---

## Phase 2: Feature Engineering Migration (Week 2-3)

### 2.1 Critical Components

```python
# /ml/features/engineering.py
class FeatureEngineer:
    """
    CRITICAL: This class ensures feature parity between training and inference.
    Uses Nautilus indicators for perfect consistency.
    """

    def __init__(self, config: FeatureConfig):
        # Initialize Nautilus indicators
        self.indicators = self._init_indicators(config)

    def compute_batch(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Batch computation for training (cold path)."""
        # Use Polars for performance
        # Apply Nautilus indicators consistently

    def compute_realtime(self, bar: Bar) -> dict:
        """Real-time computation for inference (hot path)."""
        # Update indicators incrementally
        # Return feature dict
```

### 2.2 Feature Validation

- [ ] Migrate feature parity tests (CRITICAL!)
- [ ] Ensure tolerance < 1e-10 for all features
- [ ] Add CI/CD integration for automatic validation

### 2.3 Indicator Integration

- [ ] Use Nautilus native indicators exclusively
- [ ] Implement incremental updates for real-time
- [ ] Add feature caching for performance

---

## Phase 3: Training Infrastructure (Week 3-4)

### 3.1 Trainer Migration Priority

1. **XGBoost Trainer** (Most mature, 841+ lines)
   - [ ] Migrate `UnifiedXGBoostTrainer`
   - [ ] Preserve GPU acceleration
   - [ ] Maintain Optuna integration
   - [ ] Keep SHAP explainability

2. **LightGBM Trainer** (Second priority)
   - [ ] Migrate `UnifiedLightGBMTrainer`
   - [ ] Preserve multi-asset support
   - [ ] Maintain cross-sectional features

3. **Neural Trainers** (Optional, based on needs)
   - [ ] Evaluate which models to migrate
   - [ ] Create conditional imports
   - [ ] Document dependencies

### 3.2 MLOps Integration

```python
# /ml/utils/mlflow.py
class MLflowManager:
    """Centralized MLflow operations for Nautilus ML."""

    def log_model(self, model, trainer_type: str, metrics: dict):
        # Standard model logging
        # Include feature config
        # Add Nautilus-specific metadata

    def load_model(self, model_uri: str) -> Any:
        # Load with proper error handling
        # Validate feature compatibility
```

### 3.3 Model Registry

- [ ] Port `ModelRegistry` class
- [ ] Implement model versioning
- [ ] Add promotion workflows
- [ ] Create model validation pipeline

---

## Phase 4: Hot Path Integration (Week 4-5)

### 4.1 ML Actor Implementation

```python
# /ml/actors/signal_actor.py
from nautilus_trader.common.actor import Actor
from nautilus_trader.model.data import Bar

class MLSignalActor(Actor):
    """
    Real-time ML inference actor.
    Follows Nautilus hot path requirements.
    """

    def __init__(self, config: MLActorConfig):
        super().__init__(config)
        self.model = self._load_model(config.model_uri)
        self.feature_engineer = FeatureEngineer(config.feature_config)

    def on_bar(self, bar: Bar) -> None:
        # Compute features (optimized for latency)
        features = self.feature_engineer.compute_realtime(bar)

        # Generate prediction
        signal = self.model.predict(features)

        # Publish via message bus
        self.publish_signal(signal)
```

### 4.2 Strategy Integration

- [ ] Implement `EnsembleMLStrategy`
- [ ] Add signal aggregation logic
- [ ] Integrate risk management
- [ ] Create order generation logic

### 4.3 Message Bus Integration

- [ ] Define ML-specific message types
- [ ] Set up actor-strategy communication
- [ ] Implement signal publishing/subscription

---

## Phase 5: Testing & Validation (Week 5-6)

### 5.1 Test Migration

- [ ] Port all existing tests
- [ ] Add Nautilus-specific integration tests
- [ ] Ensure 90%+ coverage for all modules
- [ ] Add performance benchmarks

### 5.2 Critical Test Areas

1. **Feature Parity** (MUST PASS)
   - Batch vs real-time consistency
   - Indicator accuracy
   - Data type conversions

2. **Model Lifecycle**
   - Training → Registry → Loading
   - Version management
   - Performance validation

3. **Integration Tests**
   - End-to-end pipeline
   - Backtest validation
   - Live simulation

### 5.3 CI/CD Setup

- [ ] Add pre-commit hooks for ML code
- [ ] Set up automated testing
- [ ] Add performance regression tests
- [ ] Create deployment validation

---

## Phase 6: Documentation & Examples (Week 6)

### 6.1 Documentation

- [ ] API documentation for all classes
- [ ] Integration guide
- [ ] Migration guide from old system
- [ ] Performance tuning guide

### 6.2 Examples

- [ ] Simple ML backtest example
- [ ] Multi-model ensemble example
- [ ] Live trading simulation
- [ ] Custom trainer example

### 6.3 Best Practices

- [ ] Hot/cold path guidelines
- [ ] Feature engineering patterns
- [ ] Model deployment checklist
- [ ] Performance optimization tips

---

## Migration Execution Plan

### Week 1-2: Foundation

- Set up directory structure
- Create base classes
- Configure development environment
- Start feature engineering migration

### Week 3-4: Core Components

- Migrate trainers (XGBoost first)
- Port MLflow integration
- Implement model registry
- Begin hot path components

### Week 5-6: Integration & Testing

- Complete actor/strategy integration
- Migrate all tests
- Perform integration testing
- Create documentation

### Success Criteria

- [ ] All tests pass with 90%+ coverage
- [ ] Feature parity validation passes (tolerance < 1e-10)
- [ ] Successful backtest with ML strategy
- [ ] Performance benchmarks meet requirements
- [ ] Zero mypy errors
- [ ] Documentation complete

---

## Risk Mitigation

### Technical Risks

1. **Feature Drift**: Mitigated by parity tests
2. **Performance Issues**: Addressed by hot path optimization
3. **Integration Bugs**: Covered by comprehensive testing

### Process Risks

1. **Scope Creep**: Stick to phased approach
2. **Dependency Conflicts**: Use conditional imports
3. **Breaking Changes**: Maintain backward compatibility

---

## Long-term Vision

### Future Enhancements

1. **Distributed Training**: Ray/Dask integration
2. **AutoML**: Automated pipeline optimization
3. **Real-time Learning**: Online model updates
4. **Multi-asset Portfolio**: Advanced optimization

### Maintenance Plan

1. **Quarterly Reviews**: Performance and accuracy
2. **Continuous Integration**: Automated testing
3. **Model Monitoring**: Drift detection
4. **Documentation Updates**: Keep current

This migration plan ensures a smooth transition while maintaining the sophisticated ML capabilities and following Nautilus Trader's architectural principles.
