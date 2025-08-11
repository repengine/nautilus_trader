# Unified Model Registry - Integration Summary

## What We Built

A unified, self-describing model registry that integrates seamlessly with the existing `MLSignalActor`, supporting ALL model types (teachers, students, inference) through comprehensive manifests.

## Key Changes

### 1. Extended MLInferenceConfig
```python
class MLInferenceConfig:
    # Existing (backward compatible)
    model_path: str | None

    # New registry support
    model_id: str | None
    registry_path: str | None
    use_manifest_features: bool = True
```

### 2. Self-Describing Models
Every model carries a `ModelManifest` with:
- **Role**: Teacher, Student, Inference
- **Data Requirements**: L1-only, L1+L2, L1+L2+L3
- **Feature Schema**: Exact features with validation hash
- **Constraints**: Latency, memory requirements
- **Lineage**: Parent-child relationships

### 3. Enhanced LocalModelRegistry
- Manifest-based registration
- Role and data requirement queries
- Lineage tracking
- LRU caching for performance
- Auto-deployment with validation

### 4. MLSignalActor Integration
The existing `MLSignalActor` now:
- Loads models from registry when `model_id` provided
- Uses manifest features automatically
- Validates deployment constraints
- Tracks performance back to registry

## Usage Examples

### Register Models
```python
# Teacher (offline, L2/L3 data)
teacher_manifest = ModelManifest(
    model_id="tft_teacher_001",
    role=ModelRole.TEACHER,
    data_requirements=DataRequirements.L1_L2_L3,
    architecture="TFT",
    feature_schema={...},  # 20+ L2/L3 features
    deployment_constraints={"max_latency_ms": 1000}
)

# Student (live trading, L1-only)
student_manifest = ModelManifest(
    model_id="lgb_student_001",
    role=ModelRole.STUDENT,
    data_requirements=DataRequirements.L1_ONLY,
    parent_id="tft_teacher_001",
    feature_schema={...},  # L1-only features
    deployment_constraints={"max_latency_ms": 5}
)

registry.register_model(path, manifest, auto_deploy=True)
```

### Use in MLSignalActor
```python
# Direct path (existing)
config = MLSignalActorConfig(
    model_path="models/my_model.onnx"
)

# Registry-based (new)
config = MLSignalActorConfig(
    model_id="lgb_student_001",
    registry_path="ml/models",
    use_manifest_features=True
)

actor = MLSignalActor(config)  # Same actor, now manifest-aware
```

## Benefits

1. **No New Modules**: All functionality integrated into existing components
2. **Backward Compatible**: Existing model_path approach still works
3. **Self-Describing**: Models carry all needed metadata
4. **Validation**: Automatic contract validation (students must be <5ms, L1-only)
5. **Lineage**: Complete teacher-student tracking
6. **Performance**: LRU caching, lazy loading
7. **Simple Names**: MLSignalActor, not UnifiedMLInferenceActor

## Test Contracts

Each model type has enforced contracts:

**Teachers**:
- Must use L2/L3 data
- 20+ features minimum
- Can have higher latency

**Students**:
- Must use L1-only data
- Must reference teacher
- Must have <5ms latency
- Feature parity validation

## Files Modified

1. `ml/registry/base.py` - Added ModelManifest, roles, data requirements
2. `ml/registry/local_registry.py` - Enhanced with manifest support
3. `ml/config/base.py` - Extended MLInferenceConfig for registry
4. `ml/actors/base.py` - Added registry loading to _load_model_with_metadata
5. `ml/tests/unit/registry/test_model_contracts.py` - Test contracts
6. `ml/tests/unit/registry/test_unified_registry.py` - Registry tests

## Summary

The unified model registry provides a clean, integrated solution for managing all ML models in Nautilus Trader. It maintains backward compatibility while adding powerful manifest-based capabilities, all without introducing new wrapper modules or complex naming.
