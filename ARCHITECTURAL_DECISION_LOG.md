# Architectural Decision Log

## Date: 2025-08-10

### Decision: Standardized Float32 Dtype Policy for ML Inference

#### Context
The ML module had inconsistent dtype usage across training and inference paths:
- Training code used `np.float64` for numerical stability
- Inference code used `np.float32` for performance
- This created potential precision mismatches and confusion

#### Decision
Established a clear dtype policy for the ML module:

1. **Training Path (Cold)**:
   - Internal computations can use `np.float64` for numerical stability
   - Training data (`X_train`, `y_train`) can remain `np.float64`
   - Model fitting uses native precision

2. **Inference Path (Hot)**:
   - ALL model predictions MUST return `np.float32`
   - Feature arrays for inference use `np.float32`
   - ONNX models natively use float32
   - Pre-allocated buffers use `np.float32`

3. **Implementation**:
   ```python
   # Training (cold path) - float64 internally OK
   X_train = data.astype(np.float64)  # For stability
   model.fit(X_train, y_train)

   # Inference (hot path) - MUST use float32
   X_inference = features.astype(np.float32)
   predictions = model.predict(X_inference)  # Returns np.float32
   ```

#### Rationale
- **ONNX Optimization**: ONNX Runtime is optimized for float32
- **Hardware Acceleration**: GPUs and modern CPUs have better float32 throughput
- **Memory Efficiency**: Half the memory usage vs float64
- **Compatibility**: Most production ML systems use float32
- **Performance**: ~2x faster inference with float32 on most hardware

#### Migration
- Updated all `predict()` methods to return `np.float32`
- Feature buffers already use `np.float32`
- SHAP values converted to `np.float32` for consistency
- Training internals unchanged (can still use float64 for stability)

#### Validation
- Type hints updated to reflect dtype policy
- Tests verify dtype consistency
- MyPy catches type mismatches at build time

## Date: 2025-08-09

### Decision: Production Model Loading Architecture - ONNX First, No Pickle

#### Context
The ML module was using pickle for model serialization in tests and had inconsistent model loading patterns across the codebase. This created several issues:
- **Security risk**: Pickle can execute arbitrary code during deserialization
- **Version fragility**: Pickle breaks between Python versions
- **Inconsistent metadata**: Different model types had different metadata structures
- **Test failures**: Dimension mismatches between training and inference

#### Decision
Implemented a production-grade model loading system with the following architecture:

1. **`ProductionModelLoader`** as the primary loader:
   - ONNX as the primary format for cross-platform compatibility
   - Native format support for XGBoost (`.json`, `.xgb`) and LightGBM (`.txt`, `.lgb`)
   - Explicit rejection of pickle files with clear error messages
   - Standardized metadata structure across all model types

2. **Deprecated `PickleModelLoader`**:
   - Only for backward compatibility in existing tests
   - Shows deprecation warnings
   - Delegates non-pickle files to `ProductionModelLoader`

3. **Unified model interface** (in `ml/models/`):
   - `BaseModel` abstract class with consistent `predict()` method
   - Type-specific wrappers (ONNXModel, XGBoostModel, LightGBMModel)
   - Automatic model type detection
   - Standardized metadata via `ModelMetadata` class

#### Rationale
- **ONNX advantages**:
  - No arbitrary code execution (secure)
  - Hardware acceleration via ONNX Runtime
  - Smaller file sizes
  - Cross-platform compatibility
  - Version stability

- **Native format benefits**:
  - Faster loading for same-framework inference
  - No conversion overhead
  - Framework-specific optimizations

- **Security first**:
  - Pickle explicitly forbidden in production
  - Clear migration path for legacy code
  - Deprecation warnings guide users to safer alternatives

#### Consequences
- **Positive**:
  - Eliminated security vulnerabilities from pickle
  - Consistent model loading across all ML components
  - Better performance with ONNX optimizations
  - Clear separation between test and production code
  - All 539 ML tests pass

- **Negative**:
  - Existing pickle models need conversion to ONNX or native formats
  - Slight learning curve for ONNX conversion

#### Implementation Files
- `ml/actors/base.py`: ProductionModelLoader implementation
- `ml/models/__init__.py`: Unified model abstraction layer
- `ml/models/saver.py`: Model saving utilities with metadata

---

## Date: 2025-08-09 (continued)

### Decision: Training-Inference Contract for ML Pipeline

#### Context
After implementing the production model loading architecture, identified that training modules were still using pickle and lacked a standardized interface to ensure compatibility with inference actors. This gap could lead to:
- Models that can't be loaded by ProductionModelLoader
- Feature mismatches between training and inference
- Inconsistent metadata across the pipeline
- No validation that trained models work with inference actors

#### Decision
Created abstract interfaces to ensure training-inference compatibility:

1. **`ModelExportMixin`**: Provides standardized export methods
   - `save_for_production()`: Saves in ONNX or native format (never pickle)
   - `validate_inference_compatibility()`: Tests the full save-load-predict pipeline
   - Automatic format selection based on model type
   - Enforces metadata standards

2. **`TrainingActorContract`**: Ensures training outputs match actor requirements
   - `get_required_features()`: Exact feature names for inference
   - `get_model_input_shape()`: Expected input dimensions
   - `export_for_actor()`: Complete export with model and config
   - `generate_actor_config()`: Creates MLSignalActor configuration

3. **Updated `BaseModelTrainer`**: Uses production save/load methods
   - Replaced pickle with `save_model_with_metadata()`
   - Uses `ProductionModelLoader` for loading
   - Exports to ONNX via `convert_to_onnx()`

#### Rationale
- **Consistency**: Single source of truth for model formats
- **Safety**: Compile-time contract enforcement
- **Validation**: Test compatibility before deployment
- **Documentation**: Clear interface requirements

#### Implementation
- `ml/training/model_exporter.py`: Abstract interfaces and mixins
- `ml/training/base.py`: Updated to use production methods
- All training classes should inherit from `ModelExportMixin`
- All trainers targeting actors should implement `TrainingActorContract`

---

## Date: 2025-08-10

### Decision: Comprehensive ML Module Refactoring - Test-Driven Architecture with Clear Separation of Concerns

#### Context
The ML module had accumulated significant technical debt from a previous failed integration attempt:
- **88.8% of modules had zero test coverage**
- **1,047 mypy strict errors** indicating poor type safety
- **No clear module responsibilities** - actors were loading models, training code mixed with inference
- **Confusing terminology** like "unified" and "adaptive" that obscured actual functionality
- **No model tracking** - couldn't identify which model generated which signal
- **Hot/cold path violations** - ML predictions blocked the trading loop
- **Security vulnerabilities** - pickle files used throughout production paths

#### Decision
Implemented a comprehensive test-driven refactoring with 5 clearly separated modules:

1. **`ml/actors/`** - Signal Generation & Publishing (HOT PATH)
   - Purpose: Generate ML signals from models and publish to Nautilus MessageBus
   - Input: Market data (bars, ticks), loaded models
   - Output: MLSignal objects with model_id field
   - Constraints: Numpy-only, pre-allocated arrays, <5ms latency

2. **`ml/models/`** - Model Abstraction & Loading
   - Purpose: Provide unified interface for different model formats
   - Input: Model files (ONNX, XGBoost, LightGBM)
   - Output: Standardized model objects with predict() interface
   - Security: Explicit rejection of pickle files with SecurityError

3. **`ml/training/`** - Model Training & Export (COLD PATH)
   - Purpose: Train models and export to production formats
   - Input: Training data (pandas/polars DataFrames)
   - Output: Models in ONNX/native format with .meta.json files
   - Constraints: Heavy compute OK, Polars OK, batch processing allowed

4. **`ml/strategies/`** - Signal Consumption & Trading
   - Purpose: Consume ML signals and execute trades
   - Input: MLSignal objects from MessageBus
   - Output: Trading commands (orders)
   - Features: Multi-model aggregation, filtering by model_id, performance tracking

5. **`ml/registry/`** - Model Registry & Lifecycle (NEW)
   - Purpose: Track and manage deployed models
   - Input: Model metadata, deployment configurations
   - Output: Model deployment status, versioning, A/B test configs
   - Features: Hot reload, rollback capability, performance monitoring

#### Key Architectural Principles

**Hot/Cold Path Separation:**
- **HOT PATH** (actors, signal generation):
  - Numpy-only, no Polars/pandas
  - Pre-allocated arrays, zero allocations
  - <500μs feature computation, <2ms inference, <5ms end-to-end
  - Model loaded once at initialization
  - Nautilus's optimized indicators (Rust/Cython)

- **COLD PATH** (training, data preparation):
  - Polars/pandas allowed
  - Heavy computations acceptable
  - Batch processing patterns
  - Feature engineering flexibility

**Clean Abstractions:**
- `BaseModel.predict()` - all models have same interface
- `MLSignal` with required `model_id` field - standardized signal format
- `ModelRegistry` - single source of truth for deployments
- `ProductionModelLoader` - secure model loading without pickle

**Production-Ready Security:**
- No pickle files in production paths
- Model validation before deployment
- Rollback capability for failed models
- Comprehensive audit trail via model_id tracking

#### Rationale
- **Clear ownership**: Each module has single, well-defined responsibility
- **Type safety**: Zero mypy strict errors after refactoring
- **Testability**: Functional tests define behavior, not implementation
- **Performance**: Explicit hot/cold path separation prevents blocking
- **Security**: Eliminated arbitrary code execution vulnerabilities
- **Observability**: model_id enables complete signal traceability

#### Consequences
**Positive:**
- All functional contract tests passing
- Zero mypy strict errors (down from 1,047)
- Clear data flow between modules
- Performance: P99 latency < 1ms (requirement: < 5ms)
- Enabled multi-model deployments and A/B testing
- Complete audit trail from training to trading

**Negative:**
- Breaking changes to existing ML code
- Need to migrate from confusing terminology
- Existing models need model_id added

#### Implementation Details
- **Test files created**:
  - `test_actor_contracts.py` - Actor behavior requirements
  - `test_model_contracts.py` - Model abstraction requirements
  - `test_training_contracts.py` - Training pipeline requirements
  - `test_strategy_contracts.py` - Strategy signal handling
  - `test_integration_pipeline.py` - End-to-end verification
  - `test_registry_contracts.py` - Registry functionality

- **Key classes introduced**:
  - `MLSignal` - Standardized signal with model_id
  - `ProductionModelLoader` - Secure model loading
  - `ModelRegistry` - Lifecycle management
  - `BaseMLStrategy` - Multi-model signal handling
  - `ModelDeploymentManager` - Deployment orchestration

#### Migration Path
1. Add model_id to all existing models
2. Convert pickle models to ONNX or native formats
3. Update actors to use new MLSignal format
4. Register all models with the registry
5. Update strategies for multi-model support

---

## Date: 2025-01-11

### Decision: Unified Model Registry with Self-Describing Manifests

#### Context
The ML module had multiple challenges with model management:
- **No unified model tracking**: Different model types (teachers, students, inference) handled separately
- **Unclear model requirements**: No way to declare what data/features a model needs
- **Missing lineage tracking**: No connection between teacher and student models in distillation
- **Manual deployment validation**: No automatic checks for deployment constraints
- **Inconsistent model metadata**: Different structures for different model types

The teacher-student distillation workflow required:
- Teachers using rich L2/L3 order book data offline
- Students distilled to use L1-only data for live trading
- Strict latency requirements (<5ms) for production deployment
- Feature parity validation between training and inference

#### Decision
Implemented a unified model registry with self-describing manifests that handle ALL model types through comprehensive declarations:

1. **`ModelManifest` - Self-Describing Model Identity**:
   ```python
   @dataclass
   class ModelManifest:
       # Identity & Role
       model_id: str
       role: ModelRole  # TEACHER, STUDENT, INFERENCE
       data_requirements: DataRequirements  # L1_ONLY, L1_L2, L1_L2_L3

       # Schema & Validation
       feature_schema: dict[str, str]  # Exact features required
       feature_schema_hash: str  # Hash for validation

       # Relationships
       parent_id: Optional[str]  # Teacher for students
       children_ids: list[str]  # Students from teacher

       # Constraints & Performance
       deployment_constraints: dict  # {"max_latency_ms": 5}
       performance_metrics: dict  # {"accuracy": 0.68}
   ```

2. **Enhanced `LocalModelRegistry`**:
   - Manifest-based registration with validation
   - Role-based queries (`get_models_by_role()`)
   - Data requirement filtering (`get_models_by_data_requirements()`)
   - Complete lineage tracking (`get_model_lineage()`)
   - LRU caching for performance
   - Auto-deployment with contract validation

3. **Integration with Existing `MLSignalActor`**:
   - Extended `MLInferenceConfig` to support both:
     - `model_path` (existing direct loading)
     - `model_id` + `registry_path` (new registry-based)
   - Modified `BaseMLInferenceActor._load_model_with_metadata()` to load from registry
   - Automatic feature configuration from manifest
   - Deployment constraint validation

4. **Test Contract Driven Development**:
   - Each model role has enforced contracts
   - Teachers: Must use L2/L3 data, 20+ features
   - Students: Must use L1-only, <5ms latency, have parent
   - Automatic validation on registration

#### Key Architectural Principles

**Manifest vs Metadata Separation:**
- **Manifest**: Core identity and requirements (validated, required)
  - WHO: Role in the system
  - WHAT: Data and features needed
  - HOW: Performance characteristics
  - WHERE: Lineage relationships
  - WHEN: Deployment constraints

- **Metadata**: Additional context (flexible, optional)
  - Training notes, experiment tracking
  - Runtime statistics, deployment history
  - User annotations, custom metrics

**No New Wrapper Modules:**
- All functionality integrated into existing components
- Configuration-driven approach
- Backward compatible with existing `model_path` usage

#### Rationale
- **Self-Describing**: Models carry all requirements with them
- **Type Safety**: Strongly typed manifests prevent errors
- **Validation**: Automatic contract enforcement
- **Simplicity**: Single registry for all model types
- **Performance**: LRU caching, lazy loading
- **Lineage**: Complete teacher-student tracking

#### Implementation Details
- **Files Modified**:
  - `ml/registry/base.py`: Added `ModelManifest`, `ModelRole`, `DataRequirements`
  - `ml/registry/local_registry.py`: Enhanced with manifest support
  - `ml/config/base.py`: Extended `MLInferenceConfig`
  - `ml/actors/base.py`: Added registry loading support

- **New Test Infrastructure**:
  - `test_model_contracts.py`: Contract definitions for each role
  - `test_unified_registry.py`: Registry functionality tests
  - `ModelContractValidator`: Validates manifests against contracts

#### Consequences
**Positive:**
- Unified model management across all types
- Automatic validation prevents deployment errors
- Self-documenting models with manifests
- Complete audit trail via lineage
- Backward compatible with existing code
- Enables sophisticated teacher-student workflows

**Negative:**
- Existing models need manifest creation
- Some hypothesis tests need updating for new API
- Additional validation overhead (minimal)

#### Example Usage
```python
# Register teacher model (offline, L2/L3 data)
teacher_manifest = ModelManifest(
    model_id="tft_teacher_001",
    role=ModelRole.TEACHER,
    data_requirements=DataRequirements.L1_L2_L3,
    feature_schema={...},  # 20+ L2/L3 features
    deployment_constraints={"max_latency_ms": 1000}
)

# Register student model (live trading, L1-only)
student_manifest = ModelManifest(
    model_id="lgb_student_001",
    role=ModelRole.STUDENT,
    data_requirements=DataRequirements.L1_ONLY,
    parent_id="tft_teacher_001",
    feature_schema={...},  # L1-only features
    deployment_constraints={"max_latency_ms": 5}
)

registry.register_model(path, manifest, auto_deploy=True)

# Use in MLSignalActor (backward compatible)
config = MLSignalActorConfig(
    model_id="lgb_student_001",  # Load from registry
    registry_path="ml/models",
    use_manifest_features=True  # Use manifest schema
)
```

---

## 2024-12-10: ML Testing Protocol and Infrastructure

### Context
After the ML module refactoring, we discovered 53 test failures primarily due to:
- XGBoost model loading errors from invalid/empty test files
- Mock objects missing required methods
- Tests checking implementation details rather than behavior

### Decision
Established comprehensive testing protocol and infrastructure:

1. **Testing Protocol Document** (`ml/tests/TESTING_PROTOCOL.md`)
   - Core principles: Test behavior, not implementation
   - Test categories: Contract, Unit, Integration, Performance
   - Coverage requirements: ≥90% for ML module
   - Security testing: Reject pickle, validate inputs

2. **Test Model Factory** (`ml/tests/fixtures/model_factory.py`)
   - Creates minimal but valid models for testing
   - Supports XGBoost, LightGBM, ONNX, sklearn
   - All models saved in production-safe formats (no pickle)
   - Includes metadata for validation

3. **Key Testing Principles**
   - Use real components where possible (minimal models, not empty files)
   - Test observable behavior through public interfaces
   - Performance requirements: Feature <500μs, Inference <2ms, E2E <5ms
   - Zero-allocation verification for hot path

### Rationale
- **Real Models**: Using valid (but minimal) models catches actual integration issues
- **Behavior Focus**: Testing implementation details makes tests brittle
- **Security First**: Enforcing no-pickle policy even in tests
- **Performance Critical**: ML inference is on hot path, must verify latency

### Implementation Details
```python
# Test behavior, not implementation
# ❌ BAD
assert actor._bars_processed == 10

# ✅ GOOD
stats = actor.get_statistics()
assert stats["bars_processed"] == 10

# Use factory for consistent test models
model_path = TestModelFactory.create_minimal_xgboost_model(
    n_features=10,
    model_type="classification"
)
```

### Consequences
- **Positive**: Tests validate real functionality, not mocks
- **Positive**: Consistent test data across all tests
- **Positive**: Security enforced even in test environment
- **Negative**: Slightly slower tests due to real model creation
- **Mitigation**: Cache test models where appropriate

---
