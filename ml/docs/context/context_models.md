# ML Models Framework Context

## Executive Summary

The ML models framework in Nautilus Trader provides a production-ready infrastructure for training, exporting, and deploying machine learning models optimized for financial time series prediction. The framework emphasizes hot-path performance, ONNX export capabilities, and seamless integration with the broader ML pipeline including registries, stores, and inference actors.

**Key Design Principles:**

- **Production-First**: All models export to ONNX for hot-path inference
- **Framework Agnostic**: Supports XGBoost, LightGBM, PyTorch, and custom models
- **Registry Integration**: Models are managed through the ModelRegistry system
- **Security**: No pickle support - only safe, inspectable formats
- **Performance**: Sub-millisecond inference through ONNX Runtime optimization

## Available Model Architectures

### 1. Tree-Based Models (Production Ready)

#### XGBoost Models (`ml/training/non_distilled/xgboost.py`)
**Implementation**: `XGBoostTrainer(BaseMLTrainer, ModelExportMixin)`

**Features:**

- Native XGBoost JSON format support
- ONNX export with optimized conversion
- GPU acceleration support (CUDA)
- Monotonic constraints for interpretability
- SHAP value computation for explainability
- Optuna hyperparameter optimization
- MLflow experiment tracking

**Configuration**: `XGBoostTrainingConfig`

- Comprehensive hyperparameter control
- GPU configuration options
- Regularization parameters
- Early stopping and cross-validation

**Export Formats:**

- Native: `.json`, `.xgb`
- Production: `.onnx` (preferred)
- Metadata: `.meta.json` sidecars

#### LightGBM Models (`ml/training/non_distilled/lightgbm.py`)
**Implementation**: `LightGBMTrainer(BaseMLTrainer, ModelExportMixin)`

**Features:**

- Advanced sampling strategies (GOSS, DART)
- Categorical feature handling
- GPU acceleration support
- Exclusive Feature Bundling (EFB)
- Native ONNX export
- Feature importance analysis

**Configuration**: `LightGBMTrainingConfig`

- GOSS (Gradient-based One-Side Sampling)
- DART (Dropouts meet Multiple Additive Regression Trees)
- GPU and EFB configurations
- Comprehensive hyperparameter tuning

**Export Formats:**

- Native: `.txt`, `.lgb`
- Production: `.onnx` (preferred)
- Metadata: `.meta.json` sidecars

### 2. Deep Learning Models (Teacher-Student Framework)

#### Temporal Fusion Transformer (TFT)
**Implementation**: `TFTTeacher(BaseTeacher)` (Placeholder)

**Framework**: PyTorch-based teacher models in the teacher-student architecture

- **Teacher Path**: Heavy models for high-quality predictions
- **Student Path**: Lightweight models distilled for hot-path inference
- **Calibration**: Platt scaling for probability calibration

**Current Status**: Framework defined, awaiting full PyTorch Forecasting integration

### 3. Dummy Models (Testing/Development)

#### Simple Test Models (`ml/models/save_dummy_model.py`)
**Implementation**: `DummyModel` class

**Purpose**:

- Infrastructure testing
- CI/CD pipeline validation
- Performance benchmarking baseline

**Features:**

- Linear combination with sigmoid activation
- Configurable bias (bullish/bearish/neutral)
- Pickle format (development only; production actors intentionally reject pickle for security — export ONNX for production inference)

## Configuration Management

### Base Configuration Hierarchy

```python
BaseMLTrainer(ABC)
├── MLTrainingConfig (base)
├── XGBoostTrainingConfig
├── LightGBMTrainingConfig
└── TFTTeacherConfig
```

### Key Configuration Classes

1. **MLTrainingConfig** (`ml/config/base.py`)
   - Target column specification
   - Train/test split ratios
   - Cross-validation strategy
   - Model save paths

2. **XGBoostTrainingConfig** (`ml/config/xgboost.py`)
   - Objective functions (regression, classification)
   - Regularization parameters
   - GPU configuration
   - Monotonic constraints

3. **LightGBMTrainingConfig** (`ml/config/lightgbm.py`)
   - Boosting strategies
   - Advanced sampling (GOSS, DART)
   - GPU acceleration
   - Categorical feature handling

4. **OnnxRuntimeConfig** (`ml/config/runtime.py`)
   - Provider configurations (CPU, CUDA, TensorRT)
   - Session optimization
   - Memory management

## Training/Inference Patterns

### Training Pipeline Architecture

```python
# 1. Data Preparation
X, y, metadata = trainer.prepare_data(data, target_col="target")

# 2. Feature Engineering (via FeatureStore integration)
features = trainer.prepare_data_with_feature_store(
    instrument_id="EUR/USD.SIM",
    start=start_date,
    end=end_date
)

# 3. Model Training
training_results = trainer.train(
    data=training_data,
    validation_data=validation_data
)

# 4. Export for Production
trainer.save_for_production(
    path="model.onnx",
    format="onnx",
    include_metadata=True
)
```

### Inference Pipeline Architecture

```python
# 1. Model Loading (via Registry)
model_manifest = model_registry.get_latest_model(model_name="xgb_signals")
model, metadata = loader.load_model(model_manifest.path)

# 2. Feature Computation (Hot Path)
features = feature_engineer.compute_features(bar_data)

# 3. Prediction (Sub-millisecond)
predictions = ort_session.run(None, {"features": features})

# 4. Signal Generation
signal = MLSignal(
    instrument_id=instrument_id,
    model_id=model_manifest.model_id,
    prediction=predictions[0],
    confidence=confidence_score
)
```

## Export Capabilities

### ONNX Export Framework (`ml/training/export.py`)

**Core Functions:**

- `convert_to_onnx()`: Framework-agnostic ONNX conversion
- `convert_to_torchscript()`: PyTorch model export
- `save_model_with_metadata()`: Unified model saving
- `detect_model_type()`: Automatic format detection

**Model Type Support:**

```python
class ModelType(Enum):
    ONNX = "onnx"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    SKLEARN = "sklearn"
    UNKNOWN = "unknown"
```

**Metadata Schema:**

```json
{
    "model_type": "xgboost",
    "path": "/path/to/model.onnx",
    "version": "abc12345",
    "size_bytes": 1048576,
    "input_shape": [null, 42],
    "output_shape": [null, 1],
    "training_metadata": {
        "feature_names": ["feature_1", "feature_2"],
        "trainer_class": "XGBoostTrainer"
    }
}
```

### Export Contracts

**ModelExportMixin**: Ensures production compatibility

```python
class ModelExportMixin(ABC):
    @abstractmethod
    def get_model(self) -> Any

    @abstractmethod
    def get_feature_names(self) -> list[str]

    @abstractmethod
    def get_training_metadata(self) -> dict[str, Any]

    def save_for_production(self, path, format="auto") -> Path
    def validate_inference_compatibility(self, model_path) -> bool
```

**TrainingActorContract**: Actor integration interface

```python
class TrainingActorContract(ABC):
    @abstractmethod
    def get_required_features(self) -> list[str]

    @abstractmethod
    def get_model_input_shape(self) -> tuple[int, ...]

    @abstractmethod
    def export_for_actor(self, actor_model_path, actor_config_path) -> dict
```

## Performance Characteristics

### Hot Path Requirements

- **Feature Computation**: <500μs
- **Model Inference**: <2ms
- **End-to-End Latency**: <5ms
- **Memory**: Bounded, pre-allocated arrays

### Model Loading Performance

- **ONNX Runtime**: Optimized with session options
- **Provider Priority**: CUDA → TensorRT → CPU
- **Memory Management**: Configurable arena allocation
- **Threading**: Optimized for inference workloads

### ONNX Runtime Optimization

```python
# Session Options
SessionOptions:
    - intra_op_num_threads: CPU parallelism
    - inter_op_num_threads: Graph parallelism
    - enable_mem_pattern: Memory optimization
    - enable_cpu_mem_arena: Arena allocation

# Execution Providers
Providers = [
    "TensorrtExecutionProvider",  # GPU acceleration
    "CUDAExecutionProvider",      # CUDA fallback
    "CPUExecutionProvider"        # CPU fallback
]
```

## Integration Points

### Registry System Integration

- **ModelRegistry**: Semantic versioning, rollback capabilities
- **FeatureRegistry**: Feature schema validation
- **StrategyRegistry**: Strategy compatibility tracking

### Store System Integration

- **ModelStore**: Prediction persistence and performance tracking
- **FeatureStore**: Training/inference parity guarantees
- **StrategyStore**: Trading decision audit trail

### Actor System Integration

- **BaseMLInferenceActor**: Production inference base class
- **MLSignalActor**: Signal generation with built-in features
- **ProductionModelLoader**: Safe model loading (no pickle)
- **ONNXModelLoader**: Optimized ONNX runtime integration

## Current State Assessment

### Production Ready ✅

- **XGBoost Training & Export**: Full ONNX pipeline
- **LightGBM Training & Export**: Complete implementation
- **ONNX Runtime Loading**: Optimized for hot-path
- **Model Registry Integration**: Semantic versioning
- **Security**: No pickle, safe formats only

### In Development 🚧

- **TFT Teacher Implementation**: PyTorch Forecasting integration
- **Student Model Distillation**: Teacher → Student pipeline
- **Advanced Architectures**: N-BEATS, DeepLOB, Graph Neural Networks
- **Knowledge Distillation**: Heavy → Lightweight model transfer

### Framework Available 📋

- **Model Export Contracts**: Standardized interfaces
- **Training Base Classes**: Extensible architecture
- **Configuration System**: Comprehensive hyperparameter management
- **Performance Monitoring**: Latency and accuracy tracking

## Critical Implementation Notes

### Security Considerations

- **No Pickle Support**: Enforced across all loaders
- **Safe Formats Only**: ONNX, JSON, joblib, native formats
- **Model Validation**: Schema and compatibility checks
- **Version Control**: Immutable model artifacts

### Performance Guidelines

- **Pre-allocation**: Feature buffers allocated at startup
- **Hot Path Isolation**: No I/O or heavy computation
- **Model Caching**: Load once, reuse efficiently
- **Memory Bounds**: Configurable limits and monitoring

### Production Deployment

- **ONNX Requirement**: All production models must export to ONNX
- **Metadata Compliance**: Complete feature schemas required
- **Registry Integration**: Models must be registered for deployment
- **Health Monitoring**: Circuit breakers and performance tracking

### Future Extensibility

- **Plugin Architecture**: New model types via BaseMLTrainer
- **Custom Exporters**: Framework-specific ONNX converters
- **Advanced Features**: Ensemble methods, multi-model strategies
- **Cloud Integration**: Model serving and auto-scaling capabilities

This framework provides a robust foundation for production ML in financial markets, emphasizing performance, security, and operational excellence while maintaining flexibility for future model architectures and deployment scenarios.
## Cross-Module References

- **Data Pipeline**: See `context_data.md` for data ingestion and collection
- **Feature Engineering**: See `context_features.md` for feature computation
- **Stores**: See `context_stores.md` for persistence layer
- **Training**: See `context_training.md` for model training pipelines
- **Registry**: See `context_registry.md` for lifecycle management
- **Strategies**: See `context_strategies.md` for trading strategy framework
- **Deployment**: See `context_deployment.md` for containerization
- **Monitoring**: See `context_monitoring.md` for observability
- **Actors**: See `context_actors.md` for inference actors
- **Models**: See `context_models.md` for model implementations
