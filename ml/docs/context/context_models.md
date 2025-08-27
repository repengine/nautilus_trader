# ML Models Framework Context

## Executive Summary

The ML models framework in Nautilus Trader provides a production-ready infrastructure for training, exporting, and deploying machine learning models optimized for financial time series prediction. The framework emphasizes hot-path performance, ONNX export capabilities, and seamless integration with the broader ML pipeline including registries, stores, and inference actors.

**Key Design Principles:**

- **Production-First**: All models export to ONNX for hot-path inference
- **Framework Agnostic**: Supports XGBoost, LightGBM, PyTorch, and custom models
- **Registry Integration**: Models are managed through the ModelRegistry system
- **Security**: No pickle support in production - only safe, inspectable formats (ONNX, JSON, joblib)
- **Performance**: Sub-millisecond inference through ONNX Runtime optimization

## Current Implementation State

### Models Directory Structure

The `/ml/models/` directory currently contains:
- **Dummy Models**: Pre-generated test models in pickle format (for testing only)
  - `dummy_bullish_model.pkl` - Bullish bias test model
  - `dummy_bearish_model.pkl` - Bearish bias test model
  - `dummy_neutral_model.pkl` - Neutral bias test model
- **Model Generator**: `save_dummy_model.py` - Script to create test models

**Note**: These pickle models are for development/testing only. Production systems use the `ProductionModelLoader` which explicitly rejects pickle formats for security.

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

#### Teacher-Student Architecture (`ml/training/teacher/`)
**Base Implementation**: `BaseTeacher` abstract class in `ml/training/teacher/base.py`

**Components:**
- **BaseTeacher**: Abstract interface for teacher models
- **TeacherConfig**: Configuration dataclass for teacher models
- **TFT Teacher**: Temporal Fusion Transformer implementation (in development)

**Features:**
- Platt calibration for probability adjustment
- Soft label generation for student distillation
- Feature schema validation
- Cold-path optimization (heavy computation allowed)

**Current Status**: Framework established, TFT implementation in progress

### 3. Test Models (Development/Testing)

#### Dummy Models (`ml/models/` and `ml/examples/`)
**Implementations**:
- `DummyModel` class in `ml/models/save_dummy_model.py`
- Extended version in `ml/examples/create_dummy_model.py`

**Purpose**:

- Infrastructure testing and validation
- CI/CD pipeline verification
- Performance benchmarking baseline
- Dry-run testing without actual training

**Features:**

- Linear combination with sigmoid activation
- Configurable bias (bullish/bearish/neutral)
- Deterministic random number generation for reproducibility
- Support for both `predict()` and `predict_proba()` methods

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

## Training Infrastructure

### Base Training Framework (`ml/training/base.py`)

The `BaseMLTrainer` abstract class provides a comprehensive training pipeline with:

**Core Features:**
- Standardized data preparation pipeline
- Feature engineering integration via FeatureStore
- Cross-validation support (time-series and k-fold)
- Optuna hyperparameter optimization
- MLflow experiment tracking
- Automatic ONNX export
- Trading-specific metrics calculation

**Training Pipeline Methods:**
- `train()`: Orchestrates complete training pipeline
- `prepare_data()`: Abstract method for data preparation
- `prepare_data_with_feature_store()`: FeatureStore integration for training/inference parity
- `evaluate()`: Standard ML metrics computation
- `calculate_trading_metrics()`: Financial metrics (Sharpe, drawdown, etc.)
- `get_feature_importance()`: Extract feature importance from models

**Cross-Validation Support:**
- Time-series CV for temporal data
- Standard k-fold CV with sklearn fallback
- Purged walk-forward validation integration

### Training/Inference Patterns

#### Training Pipeline Architecture

```python
# 1. Initialize trainer with config
trainer = XGBoostTrainer(config=training_config)

# 2. Option A: Prepare data with FeatureStore (recommended for parity)
X, y, feature_names = trainer.prepare_data_with_feature_store(
    instrument_id="EUR/USD.SIM",
    start=start_date,
    end=end_date
)

# 2. Option B: Manual data preparation
X, y, metadata = trainer.prepare_data(data, target_col="target")

# 3. Train model with automatic optimization and tracking
training_results = trainer.train(
    data=training_data,
    validation_data=validation_data  # Optional
)

# 4. Export for production (automatic ONNX conversion)
trainer.save_for_production(
    path="model.onnx",
    format="onnx",  # or "auto" for automatic detection
    include_metadata=True
)
```

#### Inference Pipeline Architecture

```python
# 1. Model Loading (via loaders)
loader = ProductionModelLoader()  # or ONNXModelLoader()
model, metadata = loader.load_model("model.onnx")

# 2. Feature Computation (Hot Path)
features = feature_engineer.compute_features(bar_data)

# 3. Prediction (Sub-millisecond with ONNX)
if isinstance(model, ort.InferenceSession):
    predictions = model.run(None, {"features": features})[0]
else:
    predictions = model.predict(features)

# 4. Signal Generation
signal = MLSignal(
    instrument_id=instrument_id,
    model_id=metadata.get("version", "unknown"),
    prediction=float(predictions[0]),
    confidence=confidence_score,
    ts_event=ts_event,
    ts_init=ts_init
)
```

## Model Loading Infrastructure

### Production Model Loaders (`ml/actors/base.py`)

The framework provides two main model loaders for production use:

#### ProductionModelLoader
**Purpose**: General-purpose model loader with security restrictions

**Supported Formats:**
- **ONNX** (`.onnx`) - Preferred for production
- **XGBoost JSON** (`.json`) - Native XGBoost format
- **Joblib** (`.joblib`) - For sklearn models
- **LightGBM** (`.lgb`, `.txt`) - Native LightGBM format

**Security Policy:**
- **Pickle formats explicitly rejected** - Raises `ValueError` for `.pkl` or `.pickle` files
- Only safe, inspectable formats allowed in production

#### ONNXModelLoader
**Purpose**: Optimized ONNX model loading with runtime configuration

**Features:**
- Configurable execution providers (CUDA, TensorRT, CPU)
- Session optimization options
- Memory arena configuration
- Threading optimization for inference

**Configuration**: Uses `OnnxRuntimeConfig` for fine-tuning:
- Provider selection and fallback chain
- Thread pool configuration
- Memory optimization settings
- Graph optimization level

## Export Capabilities

### ONNX Export Framework (`ml/training/export.py`)

**Core Functions:**

- `convert_to_onnx()`: Framework-agnostic ONNX conversion
- `convert_to_torchscript()`: PyTorch model export
- `save_model_with_metadata()`: Unified model saving with metadata sidecars
- `detect_model_type()`: Automatic format detection

**Model Type Detection:**

```python
class ModelType(Enum):
    ONNX = "onnx"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    SKLEARN = "sklearn"
    UNKNOWN = "unknown"
```

**Metadata Sidecar Schema:**

Models saved with `save_model_with_metadata()` include a `.meta.json` file:

```json
{
    "model_type": "xgboost",
    "path": "/path/to/model.xgb",
    "version": "abc12345",
    "size_bytes": 1048576,
    "modified_time": 1234567890.0,
    "input_shape": [null, 42],
    "output_shape": [null, 1],
    "training_metadata": {
        "feature_names": ["feature_1", "feature_2"],
        "trainer_class": "XGBoostTrainer"
    }
}
```

### Export Contracts

**ModelExportMixin**: Ensures production compatibility (`ml/training/export.py`)

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

**TrainingActorContract**: Actor integration interface (`ml/training/export.py`)

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
- **PersistenceManager**: Unified backend for registry storage

### Store System Integration

All ML actors inherit from `BaseMLInferenceActor` which automatically initializes three mandatory stores:

- **ModelStore**: Prediction persistence and performance tracking
- **FeatureStore**: Training/inference parity guarantees
- **StrategyStore**: Trading decision audit trail

### Actor System Integration

- **BaseMLInferenceActor**: Production inference base class with mandatory stores
- **MLSignalActor**: Signal generation with built-in features
- **MLSignal**: Clean data class for ML signals with model_id tracking
- **HealthMonitor**: System health tracking for ML actors
- **CircuitBreaker**: Failure protection for inference pipeline

### Data Signal Classes

- **MLSignal**: Core signal class with required fields:
  - `instrument_id`: Target instrument
  - `model_id`: Model identifier for tracking
  - `prediction`: Model output value
  - `confidence`: Confidence score (0.0 to 1.0)
  - `ts_event`, `ts_init`: Nautilus-standard timestamps

## Current State Assessment

### Production Ready ✅

- **XGBoost Training & Export**: Full ONNX pipeline with native JSON support
- **LightGBM Training & Export**: Complete implementation with native format support
- **ONNX Runtime Loading**: Optimized for hot-path with configurable providers
- **Production Model Loaders**: Security-hardened with pickle rejection
- **Base Training Infrastructure**: Complete with CV, Optuna, MLflow integration
- **Model Export Framework**: Unified export with metadata sidecars
- **Test Models**: Dummy models for infrastructure testing

### In Development 🚧

- **TFT Teacher Implementation**: Framework defined, awaiting PyTorch Forecasting integration
- **Student Model Distillation**: Teacher → Student pipeline in progress
- **Advanced Architectures**: N-BEATS, DeepLOB, Graph Neural Networks planned
- **Knowledge Distillation**: Teacher-student framework established, implementation ongoing

### Framework Available 📋

- **Model Export Contracts**: `ModelExportMixin` and `TrainingActorContract` interfaces
- **Training Base Classes**: `BaseMLTrainer` with full pipeline support
- **Teacher-Student Framework**: `BaseTeacher` abstract class ready for implementation
- **Configuration System**: Comprehensive config classes for all components
- **Health Monitoring**: `HealthMonitor` and `CircuitBreaker` for production reliability

## Critical Implementation Notes

### Security Considerations

- **No Pickle in Production**: `ProductionModelLoader` explicitly rejects `.pkl` and `.pickle` files
- **Safe Formats Only**: ONNX, JSON, joblib, native framework formats (XGBoost JSON, LightGBM text)
- **Test Models Exception**: Pickle models in `/ml/models/` are for testing only, never for production
- **Model Validation**: Type detection and compatibility checks before loading
- **Version Control**: Metadata sidecars track model versions and modifications

### Performance Guidelines

- **Pre-allocation**: Feature buffers allocated at actor initialization
- **Hot Path Isolation**: No training or heavy computation in inference path
- **Model Caching**: Models loaded once at startup, reused for all predictions
- **ONNX Optimization**: Configurable runtime with provider fallback chain
- **Memory Bounds**: Arena allocation and thread pool configuration

### Production Deployment Requirements

- **ONNX Export**: Strongly recommended for all production models
- **Metadata Sidecars**: `.meta.json` files track model provenance
- **Registry Integration**: Use ModelRegistry for lifecycle management (optional but recommended)
- **Health Monitoring**: HealthMonitor and CircuitBreaker protect against failures
- **Store Integration**: All actors must inherit from BaseMLInferenceActor for mandatory stores

### Development and Testing

- **Dummy Models**: Use provided test models for infrastructure validation
- **Model Generator Scripts**: `save_dummy_model.py` and `create_dummy_model.py` for test data
- **Dry-Run Testing**: Test infrastructure without training actual models
- **Deterministic RNG**: Test models use seeded random numbers for reproducibility

### Future Extensibility

- **Plugin Architecture**: Extend `BaseMLTrainer` for new model types
- **Custom Exporters**: Add framework-specific ONNX converters in `export.py`
- **Teacher Models**: Implement `BaseTeacher` for heavy models
- **Student Distillation**: Use teacher outputs for lightweight student training
- **Advanced Architectures**: Framework ready for N-BEATS, DeepLOB, GNNs, etc.

This framework provides a robust foundation for production ML in financial markets, with a clear separation between development (allowing pickle for convenience) and production (enforcing security through format restrictions), while maintaining high performance and operational excellence.
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
