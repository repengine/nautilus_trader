# ML Training Infrastructure Context Document

## Executive Summary

The ml/training/ directory implements a sophisticated model training infrastructure centered around a teacher-student knowledge distillation architecture. This system supports both traditional (non-distilled) and distilled model training, with comprehensive export capabilities and registry integration.

### Key Components

- **Teacher Models**: Heavy models (TFT) that generate high-quality soft labels
- **Student Models**: Lightweight models (LightGBM) trained on teacher outputs
- **Export System**: Unified ONNX export with production compatibility
- **Registry Integration**: Seamless model and feature registry integration
- **Training Pipeline**: Complete training, validation, and deployment workflow

## Architecture Overview

### Core Training Infrastructure

#### BaseMLTrainer (base.py)
The foundational abstract base class providing:

**Key Features:**

- Standardized training pipeline (data prep → HPO → training → evaluation → export)
- Feature engineering integration via FeatureStore
- Cross-validation support (time-series and standard K-fold)
- Optuna hyperparameter optimization
- MLflow experiment tracking
- ONNX model export capabilities
- Trading-specific performance metrics

**Critical Methods:**

- `train()`: Main orchestration method handling complete workflow
- `prepare_data()`: Abstract method for data preprocessing
- `_train_model()`: Abstract model-specific training logic
- `predict()`: Abstract prediction interface
- `prepare_data_with_feature_store()`: FeatureStore integration for guaranteed parity

**Integration Points:**

- FeatureStore for feature computation and retrieval
- ModelRegistry for artifact persistence
- Prometheus metrics for monitoring
- MLflow for experiment tracking

#### Export System (export.py)
Unified model export infrastructure ensuring production compatibility:

**ModelType Detection:**

- ONNX, XGBoost, LightGBM, sklearn model type detection
- File extension and object-based inference

**Export Functions:**

- `save_model_with_metadata()`: Native format export with sidecar metadata
- `convert_to_onnx()`: Cross-framework ONNX conversion
- `convert_to_torchscript()`: PyTorch model tracing/scripting

**Production Contracts:**

- `ModelExportMixin`: Interface for production-ready exports
- `TrainingActorContract`: Actor compatibility contract
- `validate_inference_compatibility()`: ONNX runtime validation

### Teacher-Student Distillation Architecture

#### Teacher Models (teacher/)

**BaseTeacher (teacher/base.py):**

- Abstract interface for teacher models
- Platt calibration support for probability calibration
- Feature schema definition requirements

**TFTTeacher (teacher/tft_teacher.py):**

- Temporal Fusion Transformer implementation using pytorch-forecasting
- Binary classification focus with BCEWithLogitsLoss
- Configurable architecture parameters (encoder length, hidden size, LSTM layers)
- Time series dataset preparation and validation
- Raw logits and calibrated probability output

**TFT CLI (teacher/tft_cli.py):**

- Comprehensive CLI for TFT teacher training and calibration
- Feature registry integration for schema enforcement
- Multiple input modes (training CSV, precomputed logits, ONNX inference)
- Teacher registration with non-serveable status
- Export options: TorchScript, SafeTensors, pickle
- Platt calibration on validation data

**TorchScript Export (teacher/tft_torchscript.py):**

- `TFTScriptAdapter`: Wrapper for dict-input models to tensor-input
- `export_tft_to_torchscript_from_batch()`: Batch-based tracing
- Production-ready PyTorch model serialization

#### Student Models (student/)

**LightGBMStudentDistiller (student/lightgbm.py):**

- Production-oriented student training on teacher soft labels
- Multiple objectives: logit_mse, soft_ce, hybrid (custom gradient)
- Platt calibration on raw scores against true labels
- ONNX export with baked-in Sigmoid and calibration layers
- Strict metadata emission for train-serve parity

**Objectives:**

- `logit_mse`: MSE on teacher logits (regression setup)
- `soft_ce`: Binary cross-entropy on teacher probabilities
- `hybrid`: Custom gradient combining CE and MSE with lambda weighting

**Export Features:**

- ONNX graph modification for calibration integration
- Metadata schema hash for feature parity validation
- Production metadata (feature_names, dtypes, calibration params)

**Student CLI (student/lightgbm_cli.py):**

- End-to-end student training and registry integration
- Feature registry schema validation
- Model manifest generation with parent linkage
- Automatic deployment flag support

### Non-Distilled Trainers (non_distilled/)

#### LightGBMTrainer (non_distilled/lightgbm.py)

- Traditional LightGBM training without distillation
- Advanced features: GPU, GOSS, DART, EFB support
- Feature importance analysis and visualization
- Early stopping and categorical feature handling
- ONNX conversion via onnxmltools

#### XGBoostTrainer (non_distilled/xgboost.py)

- Comprehensive XGBoost training implementation
- GPU acceleration and monotonic constraints
- SHAP value computation for interpretability
- Optuna hyperparameter optimization
- Production-ready JSON format export

### Hyperparameter Optimization (optuna_optimizer.py)

**XGBoostOptunaOptimizer:**

- Sophisticated HPO with multiple sampling strategies (TPE, Random, CMA-ES)
- Advanced pruning (Median, Percentile, Hyperband)
- Financial data-optimized parameter ranges
- GPU-aware optimization
- Study persistence and resumption

### Distillation CLI (distillation/)

**CLI Interface (distillation/cli.py):**

- Streamlined student training from NPZ files
- Registry integration with feature schema validation
- Parent model lineage tracking
- Cold-path only operation (numpy arrays)

## Training Pipeline Specifications

### Cold Path vs Hot Path Architecture

**Cold Path (Training):**

- Heavy computations (model training, HPO, feature engineering)
- Full precision (float64) for numerical stability
- Complex frameworks (PyTorch, TensorFlow)
- Comprehensive logging and experimentation

**Hot Path (Inference):**

- Lightweight operations (< 5ms P99 latency)
- Optimized precision (float32)
- Minimal dependencies (ONNX Runtime, NumPy)
- Pre-allocated arrays and cached models

### Data Requirements and Schemas

**Input Formats:**

- Polars DataFrames for training data
- NPZ files for teacher-student handoff
- CSV for TFT training with time series structure

**Schema Requirements:**

- All data must include: instrument_id, ts_event, ts_init
- Nanosecond timestamps (Nautilus standard)
- Feature schema hashing for train-serve parity
- Pipeline signature tracking for reproducibility

**Validation Requirements:**

- Point-in-time correctness verification
- Feature-data matching protocols
- Schema hash validation across training/inference

### Model Export and Deployment

**Export Formats:**

- **ONNX**: Default cross-platform format (opset 17)
- **Native**: Framework-specific (XGBoost JSON, LightGBM TXT)
- **TorchScript**: PyTorch production format
- **SafeTensors**: Weight-only serialization

**Metadata Sidecars:**

- Technical metadata (size, modified time, input/output hints)
- Feature schemas and calibration parameters
- Training configuration snapshots
- Performance benchmarks

**Production Readiness:**

- ONNX Runtime validation
- Float32 parity testing
- Latency benchmarking
- Memory footprint analysis

### Registry Integration Patterns

**Model Registry Integration:**

- Semantic versioning for models
- Role-based categorization (TEACHER, STUDENT, PRODUCTION)
- Data requirements specification (L1_ONLY, L1_L2, HISTORICAL)
- Performance metrics tracking

**Feature Registry Integration:**

- Schema hash enforcement
- Pipeline signature matching
- Feature set versioning
- Backward compatibility management

**Deployment Integration:**

- Auto-deployment flags for students
- Production readiness validation
- Artifact path management
- Manifest generation automation

## Current Implementation Status

### Completed Components

✅ **Base Training Infrastructure**

- BaseMLTrainer with complete pipeline
- Export system with ONNX conversion
- MLflow and Optuna integration
- Trading metrics calculation

✅ **Teacher Architecture**

- TFT teacher implementation
- CLI with registry integration
- TorchScript export capability
- Platt calibration system

✅ **Student Architecture**

- LightGBM student distiller
- Multiple objective functions
- ONNX export with calibration
- Registry integration

✅ **Non-Distilled Trainers**

- LightGBM trainer with advanced features
- XGBoost trainer with SHAP support
- Comprehensive hyperparameter optimization

### Component Status

✅ **ModelExportMixin**

- Fully implemented in `export.py:325`
- Provides complete model export functionality
- No separate model_exporter.py needed (functionality in export.py)

✅ **TFT Trainer**

- Implemented in `teacher/tft_teacher.py`
- Complete teacher model implementation with calibration
- No need for duplicate in main training/ directory

### Integration Health

🟢 **Registry Integration**: Fully functional with feature/model registries
🟢 **Export System**: Complete ONNX and native format support with ModelExportMixin
🟢 **Distillation Pipeline**: End-to-end teacher→student workflow
🟢 **Documentation**: Comprehensive README with all components implemented
🟢 **Import Dependencies**: All exports available from training/export.py

## Critical Implementation Notes

### Training Best Practices

1. **Feature Parity Enforcement**
   - Always use FeatureStore for training data preparation
   - Validate schema hashes between training and inference
   - Test float32 parity with np.testing.assert_allclose(rtol=1e-10)

2. **Model Calibration**
   - Apply Platt calibration to all classification models
   - Use validation data disjoint from training
   - Bake calibration into ONNX graphs for production

3. **Registry Integration**
   - Always register models with proper manifests
   - Include parent_id for student models
   - Set appropriate data_requirements and serveable flags

4. **Performance Validation**
   - Benchmark inference latency (target: <5ms P99)
   - Validate ONNX Runtime compatibility
   - Test memory usage under load

### Error Handling Patterns

1. **Dependency Management**
   - Use ml.*imports feature flags (HAS**)
   - Graceful degradation when optional deps missing
   - Clear error messages with installation instructions

2. **Training Failures**
   - Comprehensive try-catch in objective functions
   - Return worst possible scores for failed trials
   - Log errors for debugging without breaking optimization

3. **Export Validation**
   - Smoke test ONNX models post-export
   - Validate metadata completeness
   - Check file integrity and permissions

### Production Deployment Checklist

1. **Model Artifacts**
   - [ ] ONNX format with correct opset
   - [ ] Metadata sidecar with feature schema
   - [ ] Calibration parameters embedded
   - [ ] Performance benchmarks available

2. **Registry State**
   - [ ] Model manifest registered
   - [ ] Feature schema hash validated
   - [ ] Parent lineage established (for students)
   - [ ] Deployment status set correctly

3. **Compatibility**
   - [ ] ONNX Runtime validation passed
   - [ ] Float32 parity confirmed
   - [ ] Feature ordering verified
   - [ ] Input/output shapes documented

This training infrastructure provides a robust foundation for ML model development in the Nautilus ecosystem, with strong emphasis on production readiness, reproducibility, and performance optimization.
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
