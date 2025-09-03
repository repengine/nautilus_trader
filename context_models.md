# ML Models Module Context

## Overview
The `ml/models/` module provides model artifacts, training infrastructure, and production-ready export utilities for ML trading systems. It follows strict security and compatibility standards, supporting only safe, production-ready model formats.

## Core Components

### Model Artifacts

**save_dummy_model.py** - Test Model Generation
```python
class DummyModel:
    """A simple dummy model for testing ML infrastructure."""
```

**Key Features:**
- **⚠️ CORRECTION:** Generates simple linear models with sigmoid activation
- Support for n_features configuration (default: 10)
- Deterministic predictions using fixed random seed (42)
- Three bias variants: bullish (0.55), bearish (0.45), neutral (0.5)
- Compatible predict() and predict_proba() interfaces
- **📝 ADDITION:** Used for infrastructure testing, not production trading

**examples/create_dummy_model.py** - Enhanced Test Model Factory
```python
def create_dummy_models() -> Path:
    """Create several dummy models for testing different scenarios."""
```

**Enhanced Features:**
- **✨ ENHANCEMENT:** Includes feature_names list for better metadata
- **✨ ENHANCEMENT:** Adds controlled noise for realistic prediction variability
- **📝 ADDITION:** Creates models in ml/models/ directory with proper structure
- **📝 ADDITION:** Provides usage documentation for each model variant

### Training Infrastructure

**training/base.py** - Base ML Trainer Framework
```python
class BaseMLTrainer(ABC):
    """Base class for ML model trainers."""
```

**Core Training Pipeline:**
1. **Data Preparation**: Feature engineering and validation set creation
2. **Hyperparameter Optimization**: Optional Optuna integration
3. **Cross-Validation**: Time series and standard K-fold support  
4. **Model Training**: Framework-specific implementation
5. **Evaluation**: Classification and regression metrics
6. **Trading Metrics**: Sharpe ratio, max drawdown, win rate
7. **MLflow Tracking**: Optional experiment tracking
8. **Model Serialization**: Production-ready format export

**Key Features:**
- **✨ ENHANCEMENT:** FeatureStore integration for training/inference parity
- **📝 ADDITION:** Automatic model type detection (XGBoost, LightGBM, sklearn, ONNX)
- **📝 ADDITION:** Trading-specific performance metrics calculation
- **🔄 UPDATE:** Mandatory ONNX export for production deployment
- **⚠️ CORRECTION:** No pickle model support for security compliance

### Model Export and Conversion

**training/export.py** - Production Model Export
```python
def save_model_with_metadata(
    model: Any,
    path: str | Path,
    training_metadata: dict[str, Any] | None = None,
) -> Path:
```

**Export Formats Supported:**
- **ONNX**: Universal inference format (preferred for production)
- **XGBoost**: Native .xgb format for XGBoost models
- **LightGBM**: Native .lgb format for LightGBM models  
- **TorchScript**: PyTorch .pt format for deep learning models

**Key Features:**
- **⚠️ CORRECTION:** No pickle format support (security risk eliminated)
- **📝 ADDITION:** Automatic model type detection from objects and file extensions
- **📝 ADDITION:** Technical metadata sidecar (.meta.json) with file-level context
- **✨ ENHANCEMENT:** ONNX conversion with configurable opset versions
- **📝 ADDITION:** Inference compatibility validation

**Model Type Detection:**
```python
class ModelType(Enum):
    ONNX = "onnx"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    SKLEARN = "sklearn"
    UNKNOWN = "unknown"
```

### Advanced Training Components

**training/teacher/tft_model.py** - Temporal Fusion Transformer
```python
class TFTTeacher(BaseTeacher):
    """Temporal Fusion Transformer (TFT) teacher model."""
```

**Features:**
- **📝 ADDITION:** Placeholder implementation for TFT architecture
- **📝 ADDITION:** Focus on calibration rather than training (assumes pre-trained model)
- **📝 ADDITION:** Designed for knowledge distillation workflows
- **🔄 UPDATE:** Returns input as logits for calibration pipeline

**training/optuna_optimizer.py** - Hyperparameter Optimization
- **📝 ADDITION:** Optuna integration for automated hyperparameter tuning
- **📝 ADDITION:** TPE sampling with deterministic seeding
- **📝 ADDITION:** Objective metric calculation for classification/regression

## Architecture Patterns

### Security-First Model Handling
**⚠️ CORRECTION:** Complete elimination of pickle format support:
```python
# DEPRECATED: Pickle models no longer supported
# raise RuntimeError("Dummy pickle models are no longer supported.")

# CURRENT: ONNX and framework-native only
save_model_with_metadata(model, path, force_pickle=False)  # force_pickle removed
```

### Production Export Pipeline
All models follow the standardized export process:
```python
class ModelExportMixin(ABC):
    def save_for_production(
        self,
        path: str | Path,
        format: str = "auto",  # auto, onnx, native
        include_metadata: bool = True,
        opset: int = DEFAULT_ONNX_OPSET,
    ) -> Path:
```

### Training/Inference Compatibility
**📝 ADDITION:** Mandatory compatibility validation:
```python
def validate_inference_compatibility(
    self,
    model_path: str | Path,
    test_features: NDArray[np.float32] | None = None,
) -> bool:
```

### Hot/Cold Path Separation
- **Cold Path**: Model training, hyperparameter optimization, export
- **Hot Path**: ONNX Runtime inference with <5ms P99 latency
- **✨ ENHANCEMENT:** Models loaded once at actor startup, never in inference loops

## Integration Points

### Store and Registry Integration
Training components integrate with the mandatory 4-store pattern:
- **FeatureStore**: Training/inference parity via feature caching
- **ModelStore**: Performance metrics and prediction tracking
- **ModelRegistry**: Model version management and deployment tracking
- **DataStore**: Unified training data access

### Framework Support
**Central Imports Pattern** (ml/_imports.py):
```python
from ml._imports import HAS_XGBOOST, xgb, check_ml_dependencies

if not HAS_XGBOOST:
    check_ml_dependencies(["xgboost"])
model = xgb.XGBClassifier()
```

**Supported Frameworks:**
- **XGBoost**: Gradient boosting with native .xgb export
- **LightGBM**: Fast gradient boosting with .lgb export
- **scikit-learn**: Classical ML with ONNX conversion
- **ONNX Runtime**: Universal inference engine
- **PyTorch**: Deep learning with TorchScript export (TFT support)

### MLflow Integration
**📝 ADDITION:** Optional experiment tracking:
- Parameter logging during training
- Metric tracking across CV folds
- Model artifacts and metadata storage
- Automated run management with graceful fallbacks

## Training Workflows

### Standard Training Pipeline
```python
trainer = ConcreteMLTrainer(config)
results = trainer.train(
    data=training_data,
    validation_data=validation_data
)

# Automatic exports:
# - model.onnx (production inference)
# - model.meta.json (technical metadata)
# - MLflow tracking (if configured)
```

### Feature Store Integration
**✨ ENHANCEMENT:** Guaranteed training/inference parity:
```python
X, y, feature_names = trainer.prepare_data_with_feature_store(
    instrument_id="SPY.XNAS",
    start=start_date,
    end=end_date,
    compute_if_missing=True
)
```

### Cross-Validation Strategies
**📝 ADDITION:** Time series aware validation:
- **Time Series CV**: Respects temporal order, prevents lookahead bias
- **Purged CV**: Gaps between training/validation to prevent leakage
- **Standard K-Fold**: Available but not recommended for financial data

### Hyperparameter Optimization
**📝 ADDITION:** Optuna-powered optimization:
```python
# Automatic hyperparameter tuning
best_params = trainer._optimize_hyperparameters(X_train, y_train, X_val, y_val)
```

## Model Lifecycle Management

### Version Control
**🔄 UPDATE:** Semantic versioning through ModelRegistry:
- Model artifacts use content-based hashing
- Registry tracks semantic versions (v1.0.0, v1.1.0, etc.)
- No versioned filenames (tft_model_v2.py → tft_model.py)

### Deployment Process
1. **Training**: Model trained with BaseMLTrainer
2. **Export**: Automatic ONNX/native format conversion
3. **Registration**: ModelRegistry records manifest with metadata
4. **Validation**: Inference compatibility testing
5. **Deployment**: Actor loads via ProductionModelLoader

### A/B Testing Support
**📝 ADDITION:** Multiple model deployment:
- Registry supports multiple active model versions
- Actor configuration specifies model selection criteria
- Performance tracking enables automated model switching

## Performance Requirements

### Training Performance (Cold Path)
- **Hyperparameter Optimization**: Parallel trial execution via Optuna
- **Cross-Validation**: Configurable fold counts with early stopping
- **Feature Engineering**: Polars-based data processing
- **Memory Management**: Lazy evaluation and batch processing

### Inference Performance (Hot Path)
- **ONNX Runtime**: Optimized inference with SIMD acceleration
- **Model Loading**: Once at startup, cached in memory
- **Latency Target**: <5ms P99 for prediction calls
- **Memory Usage**: Pre-allocated arrays, zero garbage collection

## Testing and Validation

### Model Testing Framework
**📝 ADDITION:** Comprehensive test coverage:
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end training workflows  
- **Property Tests**: Hypothesis-based validation
- **Performance Tests**: Latency and memory benchmarks

### Dummy Model Usage
**✨ ENHANCEMENT:** Infrastructure testing without real models:
```python
# Create test models for CI/CD
models_dir = create_dummy_models()  # Creates bullish/bearish/neutral variants
```

### Compatibility Validation
**📝 ADDITION:** Automated compatibility checks:
```python
# Validate ONNX model can be loaded and run
is_compatible = trainer.validate_inference_compatibility(
    model_path="model.onnx",
    test_features=sample_features
)
```

## Best Practices

### Security Guidelines
- **⚠️ CORRECTION:** Never use pickle format (arbitrary code execution risk)
- **✨ ENHANCEMENT:** ONNX format preferred for universal compatibility
- **📝 ADDITION:** Model path validation and sanitization
- **📝 ADDITION:** Framework-native formats for gradient boosting

### Performance Guidelines
- **Model Size**: Prefer smaller models for faster loading
- **Feature Count**: Optimize feature selection for inference speed
- **Batch Size**: Configure optimal batch sizes for training/inference
- **Memory Usage**: Monitor peak memory consumption during training

### Maintenance Guidelines
- **Model Registry**: Always register models with proper metadata
- **Version Control**: Use semantic versioning for model releases
- **Documentation**: Include training configuration and performance metrics
- **Monitoring**: Track model performance degradation in production

This models module ensures secure, performant, and maintainable ML model lifecycle management while enforcing strict compatibility requirements for production deployment.