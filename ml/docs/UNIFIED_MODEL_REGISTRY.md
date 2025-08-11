# Unified Model Registry Architecture

## Overview

The Unified Model Registry provides a single, self-describing model management system that handles ALL model types (teachers, students, inference models) through comprehensive manifests. This design follows test contract driven development principles with simple, non-aliased method names.

## Key Design Principles

### 1. Self-Describing Models
Every model carries a `ModelManifest` that completely describes:
- **Role**: Teacher, Student, Inference, Ensemble, Feature
- **Data Requirements**: L1-only, L1+L2, L1+L2+L3, Historical, Streaming
- **Architecture**: XGBoost, LightGBM, TFT, ONNX, etc.
- **Feature Schema**: Exact features with types and validation hash
- **Lineage**: Parent-child relationships for distillation
- **Constraints**: Deployment requirements (latency, memory)
- **Performance**: Metrics tracked over time

### 2. Simple Method Names
Following the user's preference for simplicity:
- `MLInferenceActor` (not UnifiedMLInferenceActor)
- `register_model()` (not register_model_with_manifest)
- `get_model()` (not retrieve_model_info)

### 3. Test Contract Driven Development
Each model type has a contract that defines expected behaviors:

```python
class TeacherModelContract:
    - Must use L2/L3 data
    - Must have 20+ features
    - Can have higher latency (<1s)
    - Focus on accuracy over speed

class StudentModelContract:
    - Must use L1-only data
    - Must reference teacher (parent_id)
    - Must have <5ms latency
    - Must maintain feature parity
```

## Architecture Components

### ModelManifest
The core self-describing structure:

```python
@dataclass
class ModelManifest:
    model_id: str
    role: ModelRole
    data_requirements: DataRequirements
    architecture: str
    feature_schema: dict[str, str]
    feature_schema_hash: str
    parent_id: Optional[str]
    children_ids: list[str]
    training_config: dict[str, Any]
    performance_metrics: dict[str, float]
    deployment_constraints: dict[str, Any]
    version: str
    created_at: float
    last_modified: float
```

### LocalModelRegistry
Enhanced registry with caching and manifest support:

```python
class LocalModelRegistry:
    def register_model(
        model_path: Path,
        manifest: ModelManifest,
        auto_deploy: bool = False
    ) -> str:
        """Register any model type with manifest."""

    def get_models_by_role(role: ModelRole) -> list[ModelInfo]:
        """Query models by their role."""

    def get_models_by_data_requirements(
        requirements: DataRequirements
    ) -> list[ModelInfo]:
        """Query models by data needs."""

    def get_model_lineage(model_id: str) -> list[ModelInfo]:
        """Get complete parent-child lineage."""

    def load_model(model_id: str) -> Any:
        """Load with LRU caching."""
```

### MLSignalActor Integration
The existing MLSignalActor now supports manifest-based models:

```python
# Configuration extended to support both approaches
class MLInferenceConfig:
    model_path: str | None  # Direct path (existing)
    model_id: str | None    # Registry ID (new)
    registry_path: str | None  # Registry location
    use_manifest_features: bool  # Use manifest schema

# MLSignalActor automatically:
# - Loads from registry when model_id provided
# - Uses manifest features if configured
# - Validates deployment constraints
# - Tracks performance back to registry
```

## Usage Examples

### 1. Register a Teacher Model

```python
# Create teacher manifest
teacher_manifest = ModelManifest(
    model_id="tft_teacher_001",
    role=ModelRole.TEACHER,
    data_requirements=DataRequirements.L1_L2_L3,
    architecture="TFT",
    feature_schema={
        "close": "float32",
        "volume": "float32",
        "bid_ask_spread": "float32",
        "order_book_imbalance": "float32",
        # ... 20+ L2/L3 features
    },
    feature_schema_hash=compute_hash(feature_schema),
    performance_metrics={
        "accuracy": 0.72,
        "sharpe_ratio": 1.2,
    },
    deployment_constraints={
        "max_latency_ms": 1000,  # Teachers can be slower
        "max_memory_mb": 2000,
    }
)

# Register
registry = LocalModelRegistry(Path("ml/models"))
teacher_id = registry.register_model(
    model_path=Path("models/tft_teacher.pkl"),
    manifest=teacher_manifest,
    auto_deploy=False  # Teachers don't deploy directly
)
```

### 2. Register a Student Model

```python
# Create student manifest
student_manifest = ModelManifest(
    model_id="lgb_student_001",
    role=ModelRole.STUDENT,
    data_requirements=DataRequirements.L1_ONLY,
    architecture="LightGBM",
    parent_id=teacher_id,  # Links to teacher
    feature_schema={
        "close": "float32",
        "volume": "float32",
        "rsi_14": "float32",
        "sma_20": "float32",
        # L1-only features
    },
    feature_schema_hash=compute_hash(feature_schema),
    performance_metrics={
        "accuracy": 0.68,
        "inference_latency_ms": 2.5,  # Critical for students
        "distillation_loss": 0.05,
        "feature_parity_error": 1e-11,
    },
    deployment_constraints={
        "max_latency_ms": 5,  # Strict for live trading
        "max_memory_mb": 256,
    }
)

# Register with auto-deployment
student_id = registry.register_model(
    model_path=Path("models/lgb_student.onnx"),
    manifest=student_manifest,
    auto_deploy=True  # Auto-validate and deploy
)
```

### 3. Query Models

```python
# Get all students ready for deployment
students = registry.get_models_by_role(ModelRole.STUDENT)
l1_models = registry.get_models_by_data_requirements(
    DataRequirements.L1_ONLY
)

# Get complete lineage
lineage = registry.get_model_lineage(student_id)
# Returns: [grandparent_teacher, parent_teacher, student, child_models...]

# Load with caching
model = registry.load_model(student_id)  # First load from disk
model = registry.load_model(student_id)  # Second load from cache
```

### 4. Deploy with MLSignalActor

```python
# Using model from registry (manifest-based)
from ml.config.signal import MLSignalActorConfig

config = MLSignalActorConfig(
    model_id=student_id,  # Load from registry
    registry_path="ml/models",
    use_manifest_features=True,  # Use features from manifest
    prediction_threshold=0.7,
    warm_up_period=100,
    bar_type=bar_type,
)

# Or using direct model path (backward compatible)
config = MLSignalActorConfig(
    model_path="models/my_model.onnx",  # Direct path
    prediction_threshold=0.7,
    feature_config=feature_config,  # Manual feature config
    bar_type=bar_type,
)

# MLSignalActor adapts based on manifest when using model_id
actor = MLSignalActor(config)
# - Reads manifest and configures features automatically
# - Validates against deployment constraints
# - Tracks performance metrics back to registry
```

## Contract Validation

Models are validated against their contracts automatically:

```python
validator = ModelContractValidator()

# Validate manifest
is_valid, errors = validator.validate(manifest)

# Validate relationships
is_valid, errors = validator.validate_relationship(
    child_manifest=student_manifest,
    parent_manifest=teacher_manifest
)

# Auto-validation on registration
registry.register_model(
    model_path=path,
    manifest=manifest,
    auto_deploy=True  # Only deploys if validation passes
)
```

## Performance Features

### 1. In-Memory Caching
- LRU cache with configurable size
- Automatic eviction of least recently used
- Significant performance boost for frequently accessed models

### 2. Lazy Loading
- Models loaded only when needed
- Manifests always in memory for fast queries
- Feature initialization deferred until first use

### 3. Performance Tracking
```python
registry.track_performance(model_id, {
    "predictions": 10000,
    "avg_latency_ms": 2.3,
    "p99_latency_ms": 4.8,
    "accuracy": 0.67,
})

history = registry.get_performance_history(model_id)
```

## Benefits of This Architecture

1. **Simplicity**: One registry, one inference actor, simple method names
2. **Flexibility**: Handles any model type through manifests
3. **Safety**: Contract validation ensures correctness
4. **Performance**: Caching, lazy loading, optimized for each role
5. **Traceability**: Complete lineage tracking for distillation
6. **Automation**: Auto-deployment with validation
7. **Extensibility**: Easy to add new roles or requirements

## Integration with Existing System

```python
# Direct model loading (existing approach)
config = MLSignalActorConfig(
    model_path="models/my_model.onnx",
    feature_config=feature_config,
)
actor = MLSignalActor(config)

# Registry-based loading (new approach)
config = MLSignalActorConfig(
    model_id="lgb_student_001",
    registry_path="ml/models",
    use_manifest_features=True,
)
actor = MLSignalActor(config)  # Same actor, manifest-aware

# The registry provides:
# - Self-describing models with manifests
# - Automatic validation against contracts
# - Lineage tracking for distillation
# - Performance monitoring
# - Model versioning and caching
```

## Future Enhancements

1. **Distributed Registry**: Support for cloud storage backends
2. **Model Serving**: REST/gRPC endpoints based on manifest
3. **Automated Retraining**: Trigger based on performance degradation
4. **Feature Store Integration**: Link manifests to feature definitions
5. **Multi-Model Ensembles**: Compose models based on manifests

## Summary

The Unified Model Registry simplifies ML model management in Nautilus Trader by:
- Using self-describing manifests for ALL model types
- Providing a single registry with role-aware capabilities
- Implementing test contracts for validation
- Supporting teacher-student distillation workflows
- Enabling automatic deployment with validation
- Maintaining simple, comprehensive method names

This architecture ensures that models are properly validated, efficiently cached, and automatically deployed while maintaining the strict performance requirements for live trading.
