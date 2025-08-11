# Test Migration Guide: From Metadata to ModelManifest API

## Overview

This guide explains how to migrate tests from the old metadata-based API to the new ModelManifest-based API for the ML registry system.

## Quick Reference

### Old API → New API Mapping

| Old API | New API |
|---------|---------|
| `metadata={"features": [...]}` | `manifest=create_test_manifest(features=[...])` |
| `model_info.metadata["features"]` | `list(model_info.manifest.feature_schema.keys())` |
| `model_info.model_id` | `model_info.manifest.model_id` |
| `model_info.version` | `model_info.manifest.version` |
| `register_model(path, metadata=..., version=...)` | `register_model(path, manifest)` |

## Step-by-Step Migration Process

### 1. Add Helper Imports

Add these imports to your test file:

```python
from ml.tests.helpers.manifest_helpers import create_test_manifest
from ml.tests.helpers.manifest_helpers import metadata_to_manifest
```

### 2. Update Model Registration

#### Old API:
```python
model_id = registry.register_model(
    model_path=model_path,
    metadata={
        "features": ["sma_10", "rsi_14"],
        "training_metrics": {"accuracy": 0.92},
        "trainer_class": "XGBoostTrainer",
    },
    version="1.0.0"
)
```

#### New API:
```python
manifest = metadata_to_manifest(
    metadata={
        "features": ["sma_10", "rsi_14"],
        "training_metrics": {"accuracy": 0.92},
        "trainer_class": "XGBoostTrainer",
    },
    version="1.0.0"
)
model_id = registry.register_model(model_path, manifest)
```

#### Simplified (using helper):
```python
manifest = create_test_manifest(
    features=["sma_10", "rsi_14"],
    metrics={"accuracy": 0.92}
)
model_id = registry.register_model(model_path, manifest)
```

### 3. Update ModelInfo Creation

#### Old API:
```python
model_info = ModelInfo(
    model_id="test_model",
    model_path=Path("/models/test.onnx"),
    version="1.0.0",
    metadata={"features": ["sma_10"]},
    deployment_status=DeploymentStatus.INACTIVE,
    deployed_to=[],
    created_at=time.time(),
    last_modified=time.time(),
)
```

#### New API:
```python
manifest = create_test_manifest(
    features=["sma_10"],
    model_id="test_model"
)
manifest.version = "1.0.0"

model_info = ModelInfo(
    manifest=manifest,
    model_path=Path("/models/test.onnx"),
    deployment_status=DeploymentStatus.INACTIVE,
    deployed_to=[],
    last_modified=time.time(),
)
```

### 4. Update Assertions

#### Accessing Model ID
- Old: `model_info.model_id`
- New: `model_info.manifest.model_id`

#### Accessing Version
- Old: `model_info.version`
- New: `model_info.manifest.version`

#### Accessing Features
- Old: `model_info.metadata["features"]`
- New: `list(model_info.manifest.feature_schema.keys())`

#### Accessing Metrics
- Old: `model_info.metadata["training_metrics"]`
- New: `model_info.manifest.performance_metrics`

## Migration Patterns by Test Type

### Contract Tests (test_registry_contracts.py)

These tests verify the registry API contract. Migration involves:

1. Replace metadata dict with ModelManifest
2. Update assertions to access fields through manifest
3. Keep the same test logic and flow

**Example**: See `test_registry_contracts_migrated.py` for complete migration

### Property Tests (test_registry_hypothesis.py)

These tests use Hypothesis for property-based testing. Migration involves:

1. Update strategies to generate ModelManifest objects
2. Modify property assertions to use manifest fields

```python
# Old strategy
@given(
    metadata=st.fixed_dictionaries({
        "features": st.lists(st.text(), min_size=1),
        "metrics": st.dictionaries(st.text(), st.floats()),
    })
)

# New strategy
@given(
    features=st.lists(st.text(), min_size=1),
    metrics=st.dictionaries(st.text(), st.floats()),
)
def test_property(features, metrics):
    manifest = create_test_manifest(features=features, metrics=metrics)
    # ... rest of test
```

### Integration Tests (test_registry_integration.py)

These tests verify end-to-end workflows. Migration involves:

1. Update training pipeline to output ModelManifest
2. Modify deployment flows to use manifest
3. Keep integration logic intact

```python
# Old flow
def test_training_to_deployment_flow():
    # Train model
    metadata = trainer.train()
    model_id = registry.register_model(path, metadata=metadata)

# New flow
def test_training_to_deployment_flow():
    # Train model
    metadata = trainer.train()
    manifest = metadata_to_manifest(metadata)
    model_id = registry.register_model(path, manifest)
```

### Comprehensive ML Tests

These test the entire ML system. Migration involves:

1. Update model selection logic to use manifest
2. Modify registry queries to use new methods

```python
# Old: Get best model by accuracy
best_model = max(
    registry.get_all_models(),
    key=lambda m: m.metadata.get("training_metrics", {}).get("accuracy", 0)
)

# New: Get best model by accuracy
best_model = max(
    registry.get_all_models(),
    key=lambda m: m.manifest.performance_metrics.get("accuracy", 0)
)
```

## Common Pitfalls and Solutions

### 1. Missing Model ID

**Problem**: Old tests might not set model_id
**Solution**: Let the manifest auto-generate it or use `model_id=f"test_{index}"`

### 2. Feature Type Inference

**Problem**: Old API didn't specify feature types
**Solution**: The helper functions infer types from feature names, or you can specify explicitly:

```python
manifest.feature_schema = {
    "close": "float64",
    "volume": "int64",
    "sma_20": "float32",
}
```

### 3. Accessing Original Metadata

If you need the exact original metadata dict:

```python
original = manifest.training_config.get("original_metadata", {})
```

### 4. Role and Data Requirements

The new API requires explicit role and data requirements. The helpers provide sensible defaults:

- Default role: `ModelRole.INFERENCE`
- Default data requirements: `DataRequirements.L1_ONLY`

Override as needed:

```python
manifest = create_test_manifest(
    features=["orderbook_imbalance"],
    role=ModelRole.TEACHER,
)
# Will auto-infer DataRequirements.L1_L2_L3 from "orderbook" feature
```

## Testing the Migration

After migrating a test file:

1. Remove the `pytestmark = pytest.mark.skip(...)` decorator
2. Run the specific test file:
   ```bash
   python -m pytest ml/tests/contracts/test_registry_contracts_migrated.py -xvs
   ```
3. Fix any remaining issues
4. Run all ML tests to ensure no regressions:
   ```bash
   python -m pytest ml/tests -q
   ```

## Example: Complete Test Migration

Here's a complete before/after example:

### Before (Old API):
```python
def test_model_lifecycle():
    # Register
    model_id = registry.register_model(
        model_path=path,
        metadata={"features": ["sma", "rsi"], "accuracy": 0.9},
        version="1.0"
    )

    # Deploy
    registry.deploy_model(model_id, "actor")

    # Check
    model = registry.get_model(model_id)
    assert model.metadata["accuracy"] == 0.9
    assert model.version == "1.0"
```

### After (New API):
```python
def test_model_lifecycle():
    # Register
    manifest = create_test_manifest(
        features=["sma", "rsi"],
        metrics={"accuracy": 0.9}
    )
    manifest.version = "1.0"
    model_id = registry.register_model(path, manifest)

    # Deploy
    registry.deploy_model(model_id, "actor")

    # Check
    model = registry.get_model(model_id)
    assert model.manifest.performance_metrics["accuracy"] == 0.9
    assert model.manifest.version == "1.0"
```

## Next Steps

1. Start with simple contract tests
2. Move to property tests
3. Finally tackle integration tests
4. Remove skip decorators as you complete each file
5. Run full test suite to verify

## Questions?

Refer to:
- `test_unified_registry.py` - Reference implementation using new API
- `manifest_helpers.py` - Helper functions for migration
- `test_registry_contracts_migrated.py` - Complete migration example
