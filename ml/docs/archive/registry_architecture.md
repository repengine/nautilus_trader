# Registry Architecture - Unified ML Model and Feature Management

## Executive Summary

The Nautilus Trader ML registry system provides a unified, self-describing approach to managing both models and features through comprehensive manifests. This architecture supports ALL model types (teachers, students, inference models) and feature sets with strict validation, lineage tracking, and performance optimization. The design follows test contract driven development principles with simple, non-aliased method names and seamless integration with existing components.

### Key Benefits

1. **Unified Management**: Single system for both model and feature registry
2. **Self-Describing**: Complete metadata through manifests
3. **Backward Compatible**: Existing approaches continue to work
4. **Contract Validation**: Automatic enforcement of model/feature contracts
5. **Performance Optimized**: LRU caching, lazy loading, hot path optimization
6. **Lineage Tracking**: Complete teacher-student and feature relationships
7. **Simple Integration**: No new wrapper modules required

## Architecture Overview

### Core Components

The registry system consists of two primary registries working in tandem:

1. **ModelRegistry**: Manages ML models with role-based contracts
2. **FeatureRegistry**: Manages feature sets with schema validation

Both registries use manifest-based storage with identical architectural patterns:

```
ml/
├── registry/
│   ├── base.py              # Common types and interfaces
│   ├── local_registry.py    # File-based model registry
│   └── feature_registry.py  # File-based feature registry
├── models/                  # Model storage
│   ├── manifests/          # Model manifest files
│   └── artifacts/          # Model binary files
└── features/               # Feature storage
    ├── manifests/          # Feature manifest files
    └── schemas/            # Feature schema definitions
```

### Design Principles

1. **Self-Describing**: Every artifact carries complete metadata
2. **Contract-Driven**: Enforced validation for different roles
3. **Simple Names**: No complex aliases or wrapper classes
4. **Performance First**: Optimized for hot path requirements
5. **Lineage Aware**: Track relationships for distillation workflows

## Model Registry Details

### ModelManifest Structure

The core self-describing structure for all models:

```python
@dataclass
class ModelManifest:
    model_id: str                           # Unique identifier
    role: ModelRole                         # TEACHER, STUDENT, INFERENCE
    data_requirements: DataRequirements     # L1_ONLY, L1_L2, L1_L2_L3
    architecture: str                       # XGBoost, LightGBM, TFT, ONNX
    feature_schema: dict[str, str]          # Feature names -> dtypes
    feature_schema_hash: str                # Validation hash
    parent_id: Optional[str]                # Parent model for lineage
    children_ids: list[str]                 # Child models
    training_config: dict[str, Any]         # Training parameters
    performance_metrics: dict[str, float]   # Historical performance
    deployment_constraints: dict[str, Any]  # Latency, memory limits
    version: str                           # Semantic version
    created_at: float                      # Creation timestamp
    last_modified: float                   # Last update timestamp
```

### Model Roles and Contracts

#### Teacher Models

```python
class TeacherModelContract:
    """Contract for teacher models (offline training)."""

    # Requirements:
    - Must use L2/L3 data for rich features
    - Must have 20+ features minimum
    - Can have higher latency (<1000ms)
    - Focus on accuracy over speed
    - Generate knowledge for distillation
```

#### Student Models

```python
class StudentModelContract:
    """Contract for student models (live trading)."""

    # Requirements:
    - Must use L1-only data
    - Must reference teacher (parent_id required)
    - Must have <5ms inference latency
    - Must maintain feature parity validation
    - Optimized for production deployment
```

#### Inference Models

```python
class InferenceModelContract:
    """Contract for general inference models."""

    # Requirements:
    - Flexible data requirements
    - Performance constraints based on role
    - Can be standalone or ensemble components
```

### ModelRegistry API

```python
class ModelRegistry:
    """Enhanced registry with caching and manifest support."""

    def register_model(
        self,
        model_path: Path,
        manifest: ModelManifest,
        auto_deploy: bool = False
    ) -> str:
        """Register any model type with manifest."""

    def get_models_by_role(self, role: ModelRole) -> list[ModelInfo]:
        """Query models by their role."""

    def get_models_by_data_requirements(
        self,
        requirements: DataRequirements
    ) -> list[ModelInfo]:
        """Query models by data needs."""

    def get_model_lineage(self, model_id: str) -> list[ModelInfo]:
        """Get complete parent-child lineage."""

    def load_model(self, model_id: str) -> Any:
        """Load with LRU caching."""

    def track_performance(
        self,
        model_id: str,
        metrics: dict[str, float]
    ) -> None:
        """Track performance metrics over time."""
```

## Feature Registry Details

### FeatureManifest Structure

The comprehensive feature set descriptor:

```python
@dataclass
class FeatureManifest:
    feature_set_id: str                    # Unique identifier
    name: str                             # Human-readable name
    version: str                          # Semantic version
    role: FeatureRole                     # TEACHER, STUDENT, INFERENCE_SUPPORT
    data_requirements: DataRequirements   # Data level requirements
    feature_names: list[str]              # Ordered feature names
    feature_dtypes: list[str]             # Corresponding data types
    schema_hash: str                      # Deterministic validation hash
    pipeline_signature: str               # Pipeline configuration hash
    pipeline_version: str                 # Pipeline version
    capability_flags: dict[str, bool]     # Available transformations
    constraints: dict[str, Any]           # Performance constraints
    parity_tolerance: float               # Batch/online tolerance
    parity_digest: str                    # Parity validation hash
    perf_digest: str                      # Performance validation hash
    parent_feature_set_id: Optional[str]  # Parent feature set
    stage: FeatureStage                   # Lifecycle stage
    created_at: float                     # Creation timestamp
    last_modified: float                  # Last update timestamp
```

Canonical source of feature names:

- Feature names and pipeline signatures are computed by the declarative
  pipeline (`PipelineRunner`) built from `FeatureConfig`. Both
  `FeatureConfig.get_feature_names()` and `FeatureEngineer.generate_feature_manifest()`
  delegate to this pipeline to prevent drift between training/inference paths and
  storage schemas.

### Feature Roles

```python
class FeatureRole(Enum):
    TEACHER = "teacher"                   # Rich features for offline training
    STUDENT = "student"                   # Simplified features for live trading
    INFERENCE_SUPPORT = "inference"       # General inference features
```

### Feature Lifecycle Stages

```python
class FeatureStage(Enum):
    CANDIDATE = "candidate"               # Under development
    STAGING = "staging"                   # Testing phase
    PROD = "prod"                        # Production ready
    DEPRECATED = "deprecated"             # Being phased out
    SCRAPPED = "scrapped"                # Removed but tracked
```

### FeatureRegistry API

```python
class FeatureRegistry:
    """Registry for feature set management."""

    def register_feature_set(self, manifest: FeatureManifest) -> str:
        """Register a feature set with manifest."""

    def promote(self, feature_set_id: str, stage: FeatureStage) -> None:
        """Promote feature set to new lifecycle stage."""

    def deprecate(self, feature_set_id: str) -> None:
        """Mark feature set as deprecated."""

    def scrap(self, feature_set_id: str) -> None:
        """Mark feature set as scrapped (audit trail preserved)."""

    def get_feature_set(self, feature_set_id: str) -> FeatureManifest:
        """Retrieve feature set manifest."""

    def list_by_stage(self, stage: FeatureStage) -> list[FeatureManifest]:
        """List feature sets by lifecycle stage."""
```

## Integration Patterns

### MLSignalActor Integration

The existing `MLSignalActor` seamlessly supports both direct model loading and registry-based loading:

```python
# Enhanced configuration supporting both approaches
class MLInferenceConfig:
    # Existing (backward compatible)
    model_path: str | None                # Direct model file path
    feature_config: FeatureConfig | None  # Manual feature configuration

    # New registry support
    model_id: str | None                  # Registry model identifier
    feature_set_id: str | None            # Registry feature set identifier
    registry_path: str | None             # Registry base path
    use_manifest_features: bool = True    # Use manifest-defined features
    use_registry_features: bool = False   # Use registry feature sets
```

### Backward Compatibility

```python
# Existing approach continues to work unchanged
config = MLSignalActorConfig(
    model_path="models/my_model.onnx",    # Direct path
    feature_config=FeatureConfig(),       # Manual configuration
    prediction_threshold=0.7,
    bar_type=bar_type,
)
actor = MLSignalActor(config)
```

### Registry-Based Approach

```python
# New manifest-aware approach
config = MLSignalActorConfig(
    model_id="lgb_student_001",           # Registry lookup
    feature_set_id="features_v1_0_0",     # Registry feature set
    registry_path="ml/models",            # Registry location
    use_manifest_features=True,           # Automatic configuration
    bar_type=bar_type,
)
actor = MLSignalActor(config)

# Actor automatically:
# - Loads model from registry
# - Configures features from manifest
# - Validates deployment constraints
# - Tracks performance metrics
```

## Code Examples

### Model Registration Workflow

#### 1. Register Teacher Model

```python
from ml.registry.base import ModelManifest, ModelRole, DataRequirements
from ml.registry.local_registry import ModelRegistry
from pathlib import Path

# Create teacher manifest with rich L2/L3 features
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
        "order_flow_toxicity": "float32",
        "microstructure_signal": "float32",
        # ... 15+ additional L2/L3 features
    },
    feature_schema_hash=compute_hash(feature_schema),
    training_config={
        "epochs": 100,
        "batch_size": 512,
        "learning_rate": 1e-3,
        "dropout": 0.1,
    },
    performance_metrics={
        "accuracy": 0.72,
        "sharpe_ratio": 1.2,
        "information_ratio": 0.8,
    },
    deployment_constraints={
        "max_latency_ms": 1000,      # Teachers can be slower
        "max_memory_mb": 2000,
        "min_gpu_memory_mb": 4000,
    },
    version="1.0.0",
)

# Register teacher
registry = ModelRegistry(Path("ml/models"))
teacher_id = registry.register_model(
    model_path=Path("models/tft_teacher.pkl"),
    manifest=teacher_manifest,
    auto_deploy=False  # Teachers don't deploy to production
)
```

#### 2. Register Student Model

```python
# Create student manifest with L1-only features
student_manifest = ModelManifest(
    model_id="lgb_student_001",
    role=ModelRole.STUDENT,
    data_requirements=DataRequirements.L1_ONLY,
    architecture="LightGBM",
    parent_id=teacher_id,  # Links to teacher for lineage
    feature_schema={
        "close": "float32",
        "volume": "float32",
        "rsi_14": "float32",
        "sma_20": "float32",
        "ema_12": "float32",
        "bb_upper": "float32",
        "bb_lower": "float32",
        # Simplified L1-only features
    },
    feature_schema_hash=compute_hash(feature_schema),
    training_config={
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.1,
        "feature_fraction": 0.8,
    },
    performance_metrics={
        "accuracy": 0.68,               # Slightly lower than teacher
        "inference_latency_ms": 2.5,    # Critical metric
        "distillation_loss": 0.05,      # Knowledge transfer quality
        "feature_parity_error": 1e-11,  # Numerical consistency
    },
    deployment_constraints={
        "max_latency_ms": 5,        # Strict for live trading
        "max_memory_mb": 256,       # Memory efficient
        "requires_gpu": False,      # CPU inference only
    },
    version="1.0.0",
)

# Register student with auto-deployment
student_id = registry.register_model(
    model_path=Path("models/lgb_student.onnx"),
    manifest=student_manifest,
    auto_deploy=True  # Validates and deploys if contracts pass
)
```

### Feature Registration Workflow

#### 1. Generate Features from FeatureEngineer

```python
from ml.features.engineering import FeatureConfig, FeatureEngineer
from ml.registry.feature_registry import FeatureRegistry, FeatureRole
from ml.registry.base import DataRequirements

# Configure features for student model
feature_config = FeatureConfig(
    enable_technical_indicators=True,
    enable_microstructure=False,  # L1-only
    rsi_periods=[14],
    sma_periods=[20, 50],
    ema_periods=[12, 26],
)

# Generate manifest from engineer
engineer = FeatureEngineer(feature_config)
feature_manifest = engineer.generate_feature_manifest(
    name="student_features",
    version="1.0.0",
    role=FeatureRole.STUDENT,
    data_requirements=DataRequirements.L1_ONLY,
    constraints={"max_latency_ms_hot_path": 1.0},
)

# Register feature set
feature_registry = FeatureRegistry(Path("ml/features"))
feature_set_id = feature_registry.register_feature_set(feature_manifest)

# Promote through lifecycle
feature_registry.promote(feature_set_id, FeatureStage.STAGING)
# After validation...
feature_registry.promote(feature_set_id, FeatureStage.PROD)
```

#### 2. Training Integration

```python
from ml.training.feature_export import register_feature_set_from_engineer

# Register features with performance validation
feature_set_id = register_feature_set_from_engineer(
    registry_path=Path("ml/features"),
    name="student_features",
    version="1.0.0",
    role=FeatureRole.STUDENT,
    data_requirements=DataRequirements.L1_ONLY,
    parity_report={
        "tolerance": 1e-10,
        "max_difference": 3e-12,
        "test_samples": 10000,
    },
    perf_report={
        "p50_feature_ms": 0.2,
        "p99_feature_ms": 0.4,
        "memory_usage_mb": 12,
    },
)
```

### Query and Discovery

```python
# Query models by role
teachers = registry.get_models_by_role(ModelRole.TEACHER)
students = registry.get_models_by_role(ModelRole.STUDENT)

# Query by data requirements
l1_models = registry.get_models_by_data_requirements(
    DataRequirements.L1_ONLY
)
l2_l3_models = registry.get_models_by_data_requirements(
    DataRequirements.L1_L2_L3
)

# Explore lineage
lineage = registry.get_model_lineage(student_id)
# Returns: [teacher_model, student_model, ...]

# Performance tracking
registry.track_performance(student_id, {
    "predictions_count": 10000,
    "avg_latency_ms": 2.3,
    "p99_latency_ms": 4.8,
    "accuracy": 0.67,
    "sharpe_ratio": 1.1,
})

# Get performance history
history = registry.get_performance_history(student_id)
```

### Actor Validation and Deployment

```python
# Validate feature schema at actor startup
from ml.actors.signal import MLSignalActorConfig

config = MLSignalActorConfig(
    model_id="lgb_student_001",
    feature_set_id="features_v1_0_0",
    registry_path="ml/models",
    use_registry_features=True,
    prediction_threshold=0.7,
    bar_type=bar_type,
    instrument_id=instrument_id,
)

# Actor automatically validates:
# - Model deployment constraints
# - Feature schema compatibility
# - Performance requirements
# - Lineage integrity
actor = MLSignalActor(config)
```

## Best Practices

### 1. Schema Management

```python
# Always compute deterministic hashes
from ml.registry.base import compute_schema_hash

feature_schema = {
    "close": "float32",
    "volume": "float32",
    "rsi_14": "float32",
}
schema_hash = compute_schema_hash(feature_schema)

# Validate schema compatibility
def validate_compatibility(student_schema: dict, teacher_schema: dict) -> bool:
    """Ensure student features are subset of teacher features."""
    student_features = set(student_schema.keys())
    teacher_features = set(teacher_schema.keys())
    return student_features.issubset(teacher_features)
```

### 2. Performance Optimization

```python
# Use LRU caching for frequently accessed models
registry = ModelRegistry(
    base_path=Path("ml/models"),
    cache_size=32,  # Keep 32 models in memory
)

# Batch operations for efficiency
models_to_register = [
    (path1, manifest1),
    (path2, manifest2),
    (path3, manifest3),
]

for model_path, manifest in models_to_register:
    registry.register_model(model_path, manifest)
```

### 3. Contract Validation

```python
# Validate before registration
from ml.registry.validation import ModelContractValidator

validator = ModelContractValidator()

# Validate individual manifest
is_valid, errors = validator.validate(student_manifest)
if not is_valid:
    raise ValueError(f"Contract validation failed: {errors}")

# Validate teacher-student relationship
is_valid, errors = validator.validate_relationship(
    child_manifest=student_manifest,
    parent_manifest=teacher_manifest
)
```

### 4. Lifecycle Management

```python
# Progressive deployment
feature_registry.register_feature_set(manifest)  # CANDIDATE
run_validation_tests(feature_set_id)
feature_registry.promote(feature_set_id, FeatureStage.STAGING)

run_integration_tests(feature_set_id)
feature_registry.promote(feature_set_id, FeatureStage.PROD)

# Deprecation workflow
feature_registry.deprecate(old_feature_set_id)
deploy_new_version(new_feature_set_id)
feature_registry.scrap(old_feature_set_id)  # Preserve audit trail
```

### 5. Error Handling

```python
# Robust error handling
try:
    model = registry.load_model(model_id)
except ModelNotFoundError as e:
    logger.error(f"Model {model_id} not found: {e}")
    # Fallback to default model
    model = registry.load_model("default_model")
except ModelValidationError as e:
    logger.error(f"Model validation failed: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error loading model: {e}")
    # Report to monitoring
    metrics.registry_errors.inc()
    raise
```

### 6. Monitoring and Metrics

```python
# Expose comprehensive metrics
from ml.common.metrics import (
    registry_operations,
    model_loading_latency,
    validation_failures,
)

# Track operations
@registry_operations.time()
def register_model(self, model_path: Path, manifest: ModelManifest) -> str:
    # Registration logic
    pass

@model_loading_latency.time()
def load_model(self, model_id: str) -> Any:
    # Loading logic
    pass

# Track validation failures
try:
    validator.validate(manifest)
except ValidationError:
    validation_failures.inc(labels={"type": "manifest"})
    raise
```

## Architecture Benefits

### 1. Unified Management

- Single system for both models and features
- Consistent manifest-based approach
- Shared validation and lifecycle patterns

### 2. Performance Optimized

- LRU caching for hot path efficiency
- Lazy loading to minimize memory usage
- Contract-driven optimization for each role

### 3. Safety and Validation

- Automatic contract enforcement
- Schema validation at deployment
- Lineage integrity checks

### 4. Developer Experience

- Simple, non-aliased method names
- Backward compatibility maintained
- Clear separation of concerns

### 5. Operational Excellence

- Complete lineage tracking
- Performance monitoring
- Automated deployment validation

## Integration with Existing System

The registry architecture integrates seamlessly with existing Nautilus Trader components:

### MLSignalActor Enhancement

- Supports both direct model loading and registry-based loading
- Automatic feature configuration from manifests
- Runtime validation against deployment constraints
- Performance tracking back to registry

### FeatureEngineer Integration

- Generates manifests from feature configurations
- Validates batch/online parity
- Supports hot path optimization

### Training Pipeline Integration

- Registers models and features during training
- Tracks lineage relationships
- Validates distillation quality

### Monitoring Integration

- Exposes Prometheus metrics
- Tracks registry operations
- Reports validation failures

This unified registry architecture provides a robust foundation for ML model and feature management in Nautilus Trader while maintaining the platform's high performance and reliability standards.
