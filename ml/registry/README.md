# Model Registry - ML Lifecycle Orchestrator

## Overview

The Model Registry acts as the central orchestrator for all ML components in Nautilus Trader, tracking and managing the complete lifecycle of machine learning models from training to production deployment.

## Architecture

```
ml/registry/
├── base.py              # Abstract registry interface & data structures
├── local_registry.py    # JSON-based local registry implementation
├── mlflow_registry.py   # MLflow integration (optional)
├── deployment.py        # Deployment orchestration & management
└── example_workflow.py  # Complete workflow demonstration
```

## How It Orchestrates Other Agents' Work

### 1. Training (Agent 3) → Registry

When `XGBoostTrainer` or `LightGBMTrainer` completes training:

```python
# Training saves model with metadata
trainer.save_model("models/xgb_v1.json")

# Registry registers the trained model
model_id = registry.register_model(
    model_path=Path("models/xgb_v1.json"),
    metadata={
        "features": trainer.get_feature_names(),
        "metrics": trainer.get_metrics(),
    }
)
```

### 2. Registry → Models (Agent 2)

Registry ensures models are loadable by `ProductionModelLoader`:

- Validates model file exists
- Stores metadata compatible with loader
- Tracks model format (ONNX, XGBoost, LightGBM)

### 3. Registry → Actors (Agent 1)

Registry deploys models to `MLSignalActor`:

```python
registry.deploy_model(
    model_id=model_id,
    target="ml_signal_actor",
    config={"instruments": ["EURUSD"]}
)
```

Actor loads model via registry metadata:

- Model path from registry
- Feature configuration
- Model version for signal tracking

### 4. Registry → Strategies (Agent 4)

Registry enables multi-model strategies:

```python
# A/B test configuration
registry.configure_ab_test(
    models=[model_v1, model_v2],
    split_ratio=0.5,
    target="ml_signal_actor"
)
```

Strategy receives signals with `model_id` for tracking.

## Key Features

### Model Registration

- Automatic versioning (1.0.0, 1.0.1, etc.)
- Metadata preservation
- Thread-safe operations
- Persistent storage (JSON)

### Deployment Management

- Zero-downtime hot reload
- Gradual rollout (10% → 25% → 50% → 100%)
- Multi-target deployments
- Configuration tracking

### Performance Monitoring

```python
registry.track_performance(model_id, {
    "live_accuracy": 0.93,
    "pnl": 1500.0,
    "sharpe_ratio": 1.8
})
```

### A/B Testing

- Traffic splitting between models
- Performance comparison
- Automatic winner selection
- Rollback capability

### Model Lifecycle States

- `INACTIVE`: Registered but not deployed
- `ACTIVE`: Currently serving predictions
- `TESTING`: In A/B test or shadow mode
- `RETIRED`: No longer in production

## Usage Example

```python
from pathlib import Path
from ml.registry import LocalModelRegistry, ModelDeploymentManager

# Initialize registry
registry = LocalModelRegistry(Path("ml/tests/data/model_registry"))
deployment_manager = ModelDeploymentManager(registry)

# Register trained model
model_id = registry.register_model(
    model_path=Path("models/xgb_model.json"),
    metadata={
        "features": ["sma_10", "rsi_14"],
        "accuracy": 0.92
    }
)

# Deploy to production
deployment_id = deployment_manager.deploy(
    model_id=model_id,
    config={
        "target": "ml_signal_actor",
        "instruments": ["EURUSD", "GBPUSD"]
    }
)

# Track performance
registry.track_performance(model_id, {
    "live_accuracy": 0.91,
    "pnl": 1500.0
})

# Hot reload new version
new_model_id = registry.register_model(...)
deployment_manager.hot_reload(deployment_id, new_model_id)

# Rollback if needed
registry.rollback("ml_signal_actor", model_id)
```

## Integration Points

### With Training Pipeline

- Registers models after training
- Stores training metadata
- Tracks model lineage

### With Inference Actors

- Provides model paths
- Manages model versions
- Coordinates updates

### With Trading Strategies

- Enables model selection
- Supports A/B testing
- Tracks per-model performance

### With Monitoring

- Performance metrics collection
- Drift detection support
- Alert triggering

## Thread Safety

All registry operations are thread-safe:

- File locking for JSON persistence
- Thread locks for concurrent access
- Atomic operations for state changes

## Production Considerations

### Local Registry (Default)

- JSON-based persistence
- No external dependencies
- Suitable for single-instance deployments
- File-based backup/restore

### MLflow Registry (Optional)

- Enterprise-grade model management
- Centralized tracking server
- Team collaboration features
- Requires MLflow installation

## Testing

```bash
# Run registry tests
python -m pytest ml/tests/test_registry_contracts.py -xvs

# Type checking (must be 0 errors)
mypy ml/registry --strict
```

## Performance

- Model registration: < 10ms
- Deployment update: < 5ms
- Performance tracking: < 1ms
- Registry load time: < 100ms for 1000 models

## Future Enhancements

1. **Model Versioning**: Git-like branching and merging
2. **Auto-rollback**: Automatic rollback on performance degradation
3. **Model Compression**: Store compressed model artifacts
4. **Distributed Registry**: Multi-node registry synchronization
5. **Model Marketplace**: Share models across teams
6. **Compliance Tracking**: Audit logs and regulatory compliance
