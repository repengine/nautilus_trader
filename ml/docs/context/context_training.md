# ML Training Infrastructure Context Document

## Executive Summary

The ml/training/ directory implements a comprehensive model training infrastructure with both traditional and teacher-student knowledge distillation architectures. The system provides production-ready training pipelines with strict feature parity enforcement, ONNX export capabilities, and full registry integration.

Operational notes:
- Feature parity and persistence depend on `FeatureStore` reading/writing with UNIX nanosecond timestamps. Stores defensively normalize timestamp units to ns with warnings. See `context_stores.md` → "Timestamp Policy & Normalization".
- For integration tests and pipelines that hit the DB, apply migrations and run the DB preflight. See `context_deployment.md`.

### Key Components

- **Base Training Infrastructure**: Abstract trainer with MLflow, Optuna, and cross-validation support
- **Teacher Models**: TFT (Temporal Fusion Transformer) for generating high-quality soft labels
- **Student Models**: LightGBM students trained via knowledge distillation
- **Non-Distilled Models**: Traditional XGBoost and LightGBM trainers
- **Export System**: Unified ONNX/TorchScript export with production compatibility
- **Registry Integration**: Feature and model registry integration for lifecycle management

## Architecture Overview

### Core Training Infrastructure

#### BaseMLTrainer (base.py)
The foundational abstract base class providing a complete training framework:

**Key Features:**

- Standardized training pipeline (data prep → HPO → training → evaluation → export)
- FeatureStore integration for guaranteed train-serve parity
- Cross-validation support (time-series and standard K-fold)
- Optuna hyperparameter optimization with customizable objectives
- MLflow experiment tracking with automatic parameter logging
- ONNX model export via framework-specific converters
- Trading-specific performance metrics (Sharpe, Information ratio, drawdown)

**Critical Methods:**

- `train()`: Main orchestration method handling complete workflow
- `prepare_data()`: Abstract method for data preprocessing
- `_train_model()`: Abstract model-specific training logic
- `predict()`: Abstract prediction interface with float32 output
- `prepare_data_with_feature_store()`: FeatureStore integration for guaranteed parity
- `evaluate()`: Model evaluation with classification/regression metrics
- `calculate_trading_metrics()`: Trading performance evaluation
- `get_feature_importance()`: Feature importance extraction

**Integration Points:**

- FeatureStore for feature computation and storage
- MLflow for experiment tracking (optional via config)
- Optuna for hyperparameter optimization (optional via config)
- ProductionModelLoader for model loading compatibility

#### Export System (export.py)
Unified model export infrastructure ensuring production compatibility:

**ModelType Detection:**

- Automatic detection of ONNX, XGBoost, LightGBM, sklearn models
- File extension (.onnx, .xgb, .lgb) and object-based inference
- Framework-agnostic export handling

**Export Functions:**

- `save_model_with_metadata()`: Native format export with technical metadata sidecar
- `convert_to_onnx()`: Cross-framework ONNX conversion with opset 17 default
- `convert_to_torchscript()`: PyTorch model tracing/scripting for production
- `detect_model_type()`: Intelligent model type detection

**Production Contracts:**

- `ModelExportMixin`: Mixin class ensuring production-ready exports
  - `save_for_production()`: Unified save method with format auto-detection
  - `validate_inference_compatibility()`: ONNX runtime validation
  - `get_model()`, `get_feature_names()`, `get_training_metadata()`: Required interfaces
- `TrainingActorContract`: Contract for training-inference compatibility
  - `get_required_features()`: Feature requirements specification
  - `export_for_actor()`: Actor-specific export
  - `generate_actor_config()`: MLSignalActor configuration generation

**Key Constants:**

- `DEFAULT_ONNX_OPSET`: Centralized opset version (17) from Versions.ONNX_OPSET
- `ONNX_INPUT_NAME`: Standard input name for ONNX models

### Teacher-Student Distillation Architecture

#### Teacher Models (teacher/)

**BaseTeacher (teacher/base.py):**

- Abstract interface for all teacher models
- Built-in Platt calibration for probability calibration
- Methods: `fit()`, `predict_logits()`, `predict_proba()`, `calibrate()`
- Feature schema definition via `feature_schema()`
- TeacherConfig dataclass for configuration

**TFTTeacher (teacher/tft_teacher.py):**

- Temporal Fusion Transformer implementation using pytorch-forecasting
- Binary classification with BCEWithLogitsLoss
- Configurable architecture:
  - `max_encoder_length`: Historical context window (default: 30)
  - `max_prediction_length`: Forecast horizon (default: 1)
  - `hidden_size`: Network width (default: 16)
  - `lstm_layers`: Recurrent depth (default: 1)
  - `attention_head_size`: Multi-head attention (default: 2)
- Automatic train/validation split (80/20)
- Static and time-varying feature support
- Raw logits output with optional Platt calibration

**TFT CLI (teacher/tft_cli.py):**

- Complete CLI for TFT teacher training and calibration
- Feature registry integration for schema enforcement
- Input modes:
  - Training from CSV with full TFT training
  - Calibration from precomputed logits (.npz)
  - Inference using existing ONNX model
- Export capabilities:
  - TorchScript (.pt) via `--export_torchscript`
  - SafeTensors weights via `--export_safetensors`
  - Teacher registration via `--register_teacher`
- Interpretability artifacts via `--save_interpretability`
- Seed control for reproducibility

**TorchScript Export (teacher/tft_torchscript.py):**

- `TFTScriptAdapter`: Wrapper converting dict inputs to tensor inputs
- `export_tft_to_torchscript_from_batch()`: Batch-based model tracing
- Production-ready serialization for deployment

#### Student Models (student/)

**LightGBMStudentDistiller (student/lightgbm.py):**

- Production-oriented knowledge distillation from teacher soft labels
- Three distillation objectives:
  - `logit_mse`: MSE loss on raw teacher logits (regression mode)
  - `soft_ce`: Binary cross-entropy on teacher probabilities
  - `hybrid`: Custom gradient combining CE and MSE with kd_lambda weighting
- Platt calibration on raw scores against true labels
- ONNX export with calibration baked into graph:
  - Automatic Sigmoid layer addition
  - Optional Platt scaling (Mul/Add nodes)
  - Graph modification for end-to-end inference
- Strict metadata emission via StudentMeta dataclass:
  - Feature schema hash for parity validation
  - Calibration parameters storage
  - Training metadata preservation

**Key Methods:**

- `fit()`: Train on teacher soft labels with optional true labels for calibration
- `_fit_platt_on_raw()`: Fit Platt calibration using sklearn LogisticRegression
- `predict_proba()`: Calibrated probability predictions
- `export_onnx()`: Production ONNX export with metadata sidecar

**Student CLI (student/lightgbm_cli.py):**

- Complete CLI for student distillation workflow
- Required inputs:
  - `--features_npz`: Features with train/val splits
  - `--teacher_npz`: Teacher predictions and optional true labels
  - `--feature_registry_dir`: Feature registry path
  - `--model_registry_dir`: Model registry path
- Registry integration:
  - Feature schema validation via FeatureRegistry
  - Parent model lineage tracking
  - Automatic model registration with `auto_deploy=True`
- Output artifacts:
  - student.onnx: Production model
  - student.meta.json: Metadata sidecar

### Non-Distilled Trainers (non_distilled/)

#### LightGBMTrainer (non_distilled/lightgbm.py)

**Core Features:**

- Traditional LightGBM training extending BaseMLTrainer and ModelExportMixin
- Advanced boosting configurations:
  - GPU acceleration via `gpu_config`
  - GOSS (Gradient-based One-Side Sampling) via `goss_config`
  - DART (Dropouts meet Multiple Additive Regression Trees) via `dart_config`
  - EFB (Exclusive Feature Bundling) via `efb_config`
- Automatic categorical feature detection and handling
- Feature importance analysis with gain-based ranking
- Early stopping with configurable rounds
- Visualization support via matplotlib

**Key Methods:**

- `prepare_data()`: Polars DataFrame processing with categorical encoding
- `_train_model()`: LightGBM-specific training with callbacks
- `predict()`: Predictions with optional label conversion for classification
- `plot_importance()`: Feature importance visualization
- `save_model()`: Native .txt/.lgb format with metadata sidecar
- `load_model()`: Model loading with metadata restoration

**ONNX Export:**

- Conversion via onnxmltools with FloatTensorType specification
- Fallback to native text format if onnxmltools unavailable

#### XGBoostTrainer (non_distilled/xgboost.py)

**Core Features:**

- Comprehensive XGBoost training extending BaseMLTrainer and ModelExportMixin
- GPU acceleration support via `gpu_config`
- Monotonic constraints for interpretable models
- Missing value handling with configurable strategies
- DMatrix optimization for efficient data handling
- SHAP value computation for model interpretability

**Key Methods:**

- `prepare_data()`: Polars processing with missing value handling
- `_train_model()`: XGBoost training with early stopping
- `predict()`: Flexible prediction with probability/label conversion
- `get_feature_importance()`: Gain-based importance extraction
- `get_shap_values()`: SHAP value calculation with interaction support

**ONNX Export Handling:**

- Special handling for XGBoost feature naming (f0, f1, f2...)
- Temporary model serialization to avoid feature name conflicts
- Conversion via onnxmltools with proper input type specification
- Fallback to native JSON format if conversion fails

### Hyperparameter Optimization (optuna_optimizer.py)

**XGBoostOptunaOptimizer:**

- Sophisticated hyperparameter optimization for XGBoost models
- Multiple sampling strategies:
  - TPE (Tree-structured Parzen Estimator) with multivariate support
  - Random sampling for baseline comparison
  - CMA-ES with IPOP restart strategy
  - Grid search for exhaustive exploration
- Advanced pruning strategies:
  - Median pruner with warmup steps
  - Percentile pruner (25th percentile default)
  - Hyperband pruner with reduction factor 3
- Financial data-optimized parameter ranges:
  - `n_estimators`: 50-1000 (step 50)
  - `max_depth`: 3-12 for interpretability
  - `learning_rate`: 0.005-0.3 (log scale)
  - `subsample`/`colsample`: 0.6-1.0
  - Regularization: alpha/lambda up to 50
- GPU-aware optimization with memory-efficient settings
- Study persistence via RDBStorage with connection pooling
- Comprehensive study summary with parameter importance analysis

### Distillation CLI (distillation/)

**CLI Interface (distillation/cli.py):**

- Streamlined student training from NPZ files (cold-path only)
- Required inputs:
  - `--features_npz`: X_train, X_val, feature_names arrays
  - `--teacher_npz`: q_train (probabilities), optional y_val_true
  - `--feature_registry_dir`: Feature registry location
  - `--model_registry_dir`: Model registry location
- Configurable distillation:
  - `--objective`: Choice of logit_mse, soft_ce, hybrid
  - `--kd_lambda`: Hybrid objective weighting (default: 0.5)
  - `--early_stopping`: Rounds for early stopping (default: 200)
  - `--opset`: ONNX opset version (default: 17)
- Registry integration:
  - Feature schema validation via FeatureRegistry
  - Parent model lineage tracking via `--parent_id`
  - Automatic model manifest generation
  - Pipeline signature and version tracking
- Output structure:
  - student.onnx: Production model
  - student.meta.json: Technical metadata
  - Registry entry with auto_deploy flag

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

- BaseMLTrainer with complete training pipeline orchestration
- Unified export system with ONNX/TorchScript conversion
- MLflow experiment tracking with automatic parameter logging
- Optuna hyperparameter optimization with custom objectives
- Trading-specific metrics (Sharpe, Information ratio, drawdown)
- FeatureStore integration for train-serve parity

✅ **Teacher Architecture**

- BaseTeacher abstract interface with Platt calibration
- TFTTeacher implementation with pytorch-forecasting backend
- Comprehensive TFT CLI with multiple input/output modes
- TorchScript export via TFTScriptAdapter
- SafeTensors weight export support
- Teacher registration in model registry

✅ **Student Architecture**

- LightGBMStudentDistiller with three distillation objectives
- ONNX export with calibration baked into graph
- StudentMeta dataclass for strict metadata tracking
- Complete CLI with registry integration
- Automatic deployment flag support

✅ **Non-Distilled Trainers**

- LightGBMTrainer with GPU/GOSS/DART/EFB support
- XGBoostTrainer with SHAP interpretability
- Both trainers extend BaseMLTrainer and ModelExportMixin
- Native format export with metadata sidecars
- ONNX conversion with framework-specific handling

✅ **Hyperparameter Optimization**

- XGBoostOptunaOptimizer with financial-optimized ranges
- Multiple sampling and pruning strategies
- Study persistence and parameter importance analysis

### Module Organization

| Module | Location | Purpose |
|--------|----------|---------|
| BaseMLTrainer | training/base.py | Abstract training framework |
| Export System | training/export.py | Model export and contracts |
| XGBoostOptunaOptimizer | training/optuna_optimizer.py | HPO for XGBoost |
| TFT Teacher | training/teacher/tft_teacher.py | Teacher implementation |
| TFT CLI | training/teacher/tft_cli.py | Teacher training CLI |
| Student Distiller | training/student/lightgbm.py | Student training |
| Student CLI | training/student/lightgbm_cli.py | Student CLI |
| LightGBMTrainer | training/non_distilled/lightgbm.py | Traditional LightGBM |
| XGBoostTrainer | training/non_distilled/xgboost.py | Traditional XGBoost |
| Distillation CLI | training/distillation/cli.py | Unified distillation |

### Integration Health

🟢 **Registry Integration**: Full FeatureRegistry and ModelRegistry integration
🟢 **Export System**: Complete ONNX/TorchScript/native format support
🟢 **Distillation Pipeline**: End-to-end teacher→student workflow
🟢 **Configuration System**: Config classes with validation and defaults
🟢 **Import Management**: Centralized imports via ml/_imports.py

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
   - [ ] ONNX format with correct opset (17 default)
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

## Additional Components

### Teacher Model Variations

**TFT Model Placeholder (teacher/tft_model.py):**
- Simplified TFTTeacher implementation for testing
- Placeholder for when pytorch-forecasting unavailable
- Pass-through predict_logits for pre-computed values

**CLI Compatibility Shim (teacher/cli.py):**
- CalibratingTeacher for simple calibration workflows
- Dual-mode operation:
  - Legacy mode: Direct calibration from NPZ files
  - Registry mode: Forwards to tft_cli.py for full integration
- Backward compatibility for existing pipelines

### Console Script Entry Points

Defined in `pyproject.toml`:
- `ml-teacher-tft` → `ml.training.teacher.tft_cli:main`
- `ml-student-lightgbm` → `ml.training.student.lightgbm_cli:main`

### Export Format Specifications

**ONNX Export:**
- Default opset: 17 (from Versions.ONNX_OPSET)
- Input name: Standardized via ONNX_INPUT_NAME
- FloatTensorType with [None, n_features] shape
- Metadata sidecar (.onnx.meta.json) with technical details

**Native Formats:**
- LightGBM: .txt or .lgb with best_iteration
- XGBoost: .json or .xgb format
- Metadata sidecar with training configuration

**TorchScript Export:**
- .pt format for PyTorch models
- Supports both tracing and scripting
- TFTScriptAdapter for dict→tensor conversion

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
