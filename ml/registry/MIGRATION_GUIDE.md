# Model Registry Migration Guide

## Overview
This guide helps you migrate from the legacy `ml.tracking.model_registry.ModelRegistry` to the new `ml.registry` module.

## Key Changes

### 1. Module Structure
- **OLD**: Single monolithic `ModelRegistry` class with MLflow dependency
- **NEW**: Modular design with separate concerns:
  - `ml.registry.base.ModelRegistry` - Abstract interface
  - `ml.registry.local_registry.LocalModelRegistry` - File-based implementation
  - `ml.registry.statistics` - Statistical testing utilities
  - `ml.registry.canary` - Canary deployment functionality
  - `ml.registry.deployment` - Deployment management

### 2. MLflow Decoupling
- **OLD**: Registry inherits from MLflowManager, tightly coupled
- **NEW**: Registry is independent, MLflow integration separate

## Migration Steps

### Step 1: Update Imports

```python
# OLD
from ml.tracking.model_registry import ModelRegistry
from ml.config.shared import MLflowConfig

# NEW
from ml.registry import LocalModelRegistry, ModelDeploymentManager
from ml.registry.statistics import welch_t_test
from ml.registry.canary import CanaryConfig, CanaryDeployment
```

### Step 2: Initialize Registry

```python
# OLD
config = MLflowConfig(
    tracking_uri="http://localhost:5000",
    experiment_name="my_experiment"
)
registry = ModelRegistry(config)

# NEW
from pathlib import Path
registry_path = Path("models/registry")
registry = LocalModelRegistry(registry_path)
```

### Step 3: Register Models

```python
# OLD - Uses MLflow model versioning
model_name = "my_model"
registry.register_model(
    model_uri=f"runs:/{run_id}/model",
    name=model_name
)

# NEW - Direct file-based registration
model_id = registry.register_model(
    model_path=Path("models/my_model.onnx"),
    metadata={
        "features": ["sma_10", "rsi_14"],
        "training_metrics": {"accuracy": 0.92},
    },
    version="1.0.0"
)
```

### Step 4: A/B Testing

```python
# OLD - Built into registry
test_id = registry.setup_ab_test(
    test_name="model_comparison",
    model_a_name="model_v1",
    model_b_name="model_v2",
    traffic_split=0.5,
    success_metric="accuracy"
)

# NEW - Use statistics module for analysis
from ml.registry.statistics import welch_t_test

# Configure A/B test in registry
ab_config = registry.configure_ab_test(
    models=[model_a_id, model_b_id],
    split_ratio=0.5,
    duration_hours=24,
    target="ml_signal_actor"
)

# Analyze results with statistics
results_a = np.array([...])  # Model A performance
results_b = np.array([...])  # Model B performance
test_result = welch_t_test(results_a, results_b)
```

### Step 5: Canary Deployments

```python
# OLD - Built into registry
deployment_id = registry.setup_canary_deployment(
    deployment_name="v2_canary",
    model_name="my_model",
    model_version="2",
    traffic_percentage=5.0
)

# NEW - Use dedicated canary module
from ml.registry.canary import CanaryDeployment, CanaryConfig

config = CanaryConfig(
    traffic_percentage=5.0,
    baseline_threshold=0.95,
    monitoring_duration_hours=24
)

canary = CanaryDeployment(
    deployment_id="canary_001",
    model_id=model_id,
    config=config,
    baseline_performance=0.90
)

# Record metrics
canary.record_metric(0.92, latency_ms=15.0)

# Check status
should_promote, reason = canary.should_promote()
should_rollback, reason = canary.should_rollback()
```

### Step 6: Model Deployment

```python
# OLD - Through MLflow stages
registry.transition_model_stage(
    model_name="my_model",
    version="2",
    stage=ModelStage.PRODUCTION
)

# NEW - Direct deployment management
deployment_manager = ModelDeploymentManager(registry)

deployment_id = deployment_manager.deploy(
    model_id=model_id,
    config={
        "target": "ml_signal_actor",
        "instruments": ["EURUSD"],
    }
)

# Hot reload to new version
deployment_manager.hot_reload(
    deployment_id=deployment_id,
    new_model_id=new_model_id
)
```

## Feature Mapping

| Legacy Feature | New Implementation |
|---------------|-------------------|
| `setup_ab_test()` | `registry.configure_ab_test()` + `statistics.welch_t_test()` |
| `setup_canary_deployment()` | `canary.CanaryDeployment` |
| `rollback_model()` | `registry.rollback()` |
| `validate_model_quality()` | Implement quality gates in training pipeline |
| `get_deployment_history()` | `registry.get_performance_history()` |
| MLflow stage transitions | `registry.deploy_model()` / `retire_model()` |

## Benefits of Migration

1. **No External Dependencies**: Registry works without MLflow
2. **Better Separation**: MLflow for experiment tracking, Registry for deployment
3. **Cleaner API**: Each module has single responsibility
4. **Type Safety**: Full type annotations with mypy strict compliance
5. **Thread Safe**: Built-in thread safety for concurrent operations
6. **Testable**: Comprehensive test coverage without MLflow mocks

## Backward Compatibility

The legacy `ModelRegistry` will show deprecation warnings but remain functional until version 2.0.0. This gives you time to migrate gradually.

## Need Help?

- Check the test files for usage examples:
  - `ml/tests/test_registry_contracts.py`
  - `ml/tests/test_registry_statistics.py`
  - `ml/tests/test_registry_canary.py`
- Review the example workflow: `ml/registry/example_workflow.py`