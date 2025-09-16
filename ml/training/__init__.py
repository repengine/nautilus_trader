"""
ML model training infrastructure for Nautilus Trader.

This module provides a comprehensive training framework for financial machine learning models,
implementing cold path patterns with proper separation from hot path inference. All training
operates in the cold path and can take hours, while inference must be <5ms P99 latency.

Universal ML Architecture Patterns Compliance:
- Pattern 1: Uses 4-Store + 4-Registry integration via BaseMLTrainer
- Pattern 2: Protocol-first design with abstract base classes
- Pattern 3: Cold path only - no hot path operations
- Pattern 4: Progressive fallback with optional dependencies
- Pattern 5: Centralized metrics via ml.common.metrics_bootstrap

Core Components:

Base Classes:
    BaseMLTrainer: Abstract base class for all ML trainers with standardized interface
    ModelExportMixin: Mixin for production-ready model export (ONNX, native formats)
    TrainingActorContract: Contract ensuring training outputs are actor-compatible

Specific Trainers:
    XGBoostTrainer: XGBoost trainer with GPU support, monotonic constraints, SHAP values
    LightGBMTrainer: LightGBM trainer with categorical features, GOSS, DART boosting
    TFTTeacher: Temporal Fusion Transformer teacher for complex time series patterns

Distillation Framework:
    LightGBMStudentDistiller: Teacher-student distillation for production deployment
    BaseTeacher: Abstract teacher interface for soft label generation

Export and Utilities:
    convert_to_onnx: Convert models to ONNX format for production inference
    convert_to_torchscript: Convert PyTorch models to TorchScript
    save_model_with_metadata: Save models with comprehensive metadata
    XGBoostOptunaOptimizer: Advanced hyperparameter optimization with Optuna
    safe_torch_load: Hardened PyTorch model loading with checksum validation

Training Pipeline Architecture:

1. Cold Path Training:
   - Feature engineering with FeatureStore integration
   - Cross-validation with purged time series splits
   - Hyperparameter optimization with Optuna
   - MLflow experiment tracking
   - Model export to production formats

2. Teacher-Student Distillation:
   - Heavy teacher models (TFT) generate soft labels
   - Lightweight student models (LightGBM/XGBoost) for inference
   - Calibration for probability alignment
   - ONNX export with embedded preprocessing

3. Production Integration:
   - Registry-based model deployment
   - Metadata tracking for train-serve parity
   - Health monitoring and performance metrics
   - Fallback strategies for model loading

Example Usage:

Basic Training:
    >>> from ml.training import XGBoostTrainer
    >>> from ml.config.xgboost import XGBoostTrainingConfig
    >>>
    >>> # Configure training
    >>> config = XGBoostTrainingConfig(
    ...     objective="binary:logistic",
    ...     n_estimators=1000,
    ...     max_depth=6,
    ...     learning_rate=0.1,
    ...     early_stopping_rounds=50
    ... )
    >>>
    >>> # Train model
    >>> trainer = XGBoostTrainer(config)
    >>> results = trainer.train(train_data, validation_data)
    >>>
    >>> # Export for production
    >>> trainer.save_for_production("models/xgb_model.onnx", format="onnx")

Advanced Training with Optuna:
    >>> from ml.training import XGBoostTrainer, XGBoostOptunaOptimizer
    >>> from ml.config.shared import OptunaConfig
    >>>
    >>> # Configure hyperparameter optimization
    >>> optuna_config = OptunaConfig(
    ...     study_name="xgb_optimization",
    ...     n_trials=100,
    ...     direction="maximize",
    ...     sampler="tpe",
    ...     pruner="median"
    ... )
    >>>
    >>> # Configure training with optimization
    >>> config = XGBoostTrainingConfig(
    ...     objective="binary:logistic",
    ...     optuna_config=optuna_config,
    ...     cv_folds=5,
    ...     cv_strategy="time_series"
    ... )
    >>>
    >>> trainer = XGBoostTrainer(config)
    >>> results = trainer.train(train_data, validation_data)

Teacher-Student Distillation:
    >>> from ml.training import TFTTeacher, LightGBMStudentDistiller
    >>> from ml.training.teacher.base import TFTTeacherConfig
    >>>
    >>> # Train teacher model
    >>> teacher_config = TFTTeacherConfig(loss_name="bce")
    >>> teacher = TFTTeacher(teacher_config)
    >>> teacher.fit(time_series_data)
    >>>
    >>> # Generate soft labels
    >>> soft_labels = teacher.predict_proba(validation_features)
    >>>
    >>> # Train student model
    >>> student = LightGBMStudentDistiller(
    ...     objective="soft_ce",
    ...     kd_lambda=0.7,
    ...     early_stopping=200
    ... )
    >>> student.fit(train_features, soft_labels, val_features, true_labels)
    >>>
    >>> # Export student for production
    >>> onnx_path, meta_path = student.export_onnx(
    ...     feature_names=feature_names,
    ...     out_dir="models/student",
    ...     model_id="lightgbm_student_v1"
    ... )

Cross-Validation and Evaluation:
    >>> from ml.training import BaseMLTrainer
    >>>
    >>> # Configure time series cross-validation
    >>> config = XGBoostTrainingConfig(
    ...     cv_folds=5,
    ...     cv_strategy="time_series",
    ...     train_test_split=0.8
    ... )
    >>>
    >>> trainer = XGBoostTrainer(config)
    >>> results = trainer.train(data)
    >>>
    >>> # Access cross-validation results
    >>> cv_scores = results["metrics"]["cv_scores"]
    >>> trading_metrics = trainer.calculate_trading_metrics(returns, predictions)
    >>>
    >>> # Get feature importance
    >>> importance = trainer.get_feature_importance()

Model Export and Deployment:
    >>> from ml.training.export import convert_to_onnx, save_model_with_metadata
    >>>
    >>> # Convert existing model to ONNX
    >>> sample_input = np.random.randn(1, n_features).astype(np.float32)
    >>> onnx_path = convert_to_onnx(
    ...     model=trained_model,
    ...     sample_input=sample_input,
    ...     output_path="models/converted.onnx",
    ...     opset_version=17
    ... )
    >>>
    >>> # Save with comprehensive metadata
    >>> model_path = save_model_with_metadata(
    ...     model=trained_model,
    ...     path="models/xgb_with_meta",
    ...     input_shape=(1, n_features),
    ...     training_metadata=training_results
    ... )

Integration with Stores and Registries:
    >>> # Training with FeatureStore integration
    >>> config = XGBoostTrainingConfig(
    ...     db_connection="postgresql://user:pass@host/db",
    ...     feature_config=feature_config,
    ...     pipeline_spec=pipeline_spec
    ... )
    >>>
    >>> trainer = XGBoostTrainer(config)
    >>>
    >>> # Use FeatureStore for train-serve parity
    >>> X, y, feature_names = trainer.prepare_data_with_feature_store(
    ...     instrument_id="EUR/USD.SIM",
    ...     start=datetime(2023, 1, 1),
    ...     end=datetime(2023, 12, 31),
    ...     compute_if_missing=True
    ... )

Notes:
    - All training operates in the cold path and can be resource-intensive
    - Models are automatically exported in production-ready formats (ONNX preferred)
    - Feature engineering must maintain parity between training and inference
    - Use registry system for model deployment and versioning
    - Implement proper cross-validation to avoid overfitting on financial data
    - GPU acceleration available for XGBoost and LightGBM where supported

Dependencies:
    Required: numpy, polars
    Optional: xgboost, lightgbm, pytorch, pytorch-forecasting, optuna, mlflow,
             onnx, onnxmltools, skl2onnx, sklearn (for calibration and metrics)

Warning:
    Never import training modules in hot path code (actors, strategies).
    Training is cold path only - use lazy imports via __getattr__ when needed.

"""

from typing import Any


# Core exports - these are always available
__all__ = [
    "BaseMLTrainer",
    "BaseTeacher",
    "LightGBMStudentDistiller",
    "LightGBMTrainer",
    "ModelExportMixin",
    "ModelType",
    "TFTScriptAdapter",
    "TFTTeacher",
    "TFTTeacherConfig",
    "TrainingActorContract",
    "XGBoostOptunaOptimizer",
    "XGBoostTrainer",
    "convert_to_onnx",
    "convert_to_torchscript",
    "detect_model_type",
    "export_tft_to_torchscript_from_batch",
    "safe_torch_load",
    "save_model_with_metadata",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - import side-effect utility
    """
    Lazy loading of training components to avoid heavy dependencies at import time.

    This follows the Universal ML Architecture Pattern of progressive fallback -
    components are only loaded when needed and fail gracefully if dependencies are
    missing.

    """
    # Base classes - always available
    if name == "BaseMLTrainer":
        from .base import BaseMLTrainer

        return BaseMLTrainer

    # Export utilities and contracts
    elif name == "ModelExportMixin":
        from .export import ModelExportMixin

        return ModelExportMixin
    elif name == "TrainingActorContract":
        from .export import TrainingActorContract

        return TrainingActorContract
    elif name == "convert_to_onnx":
        from .export import convert_to_onnx

        return convert_to_onnx
    elif name == "convert_to_torchscript":
        from .export import convert_to_torchscript

        return convert_to_torchscript
    elif name == "save_model_with_metadata":
        from .export import save_model_with_metadata

        return save_model_with_metadata
    elif name == "ModelType":
        from .export import ModelType

        return ModelType
    elif name == "detect_model_type":
        from .export import detect_model_type

        return detect_model_type

    # Specific trainers - loaded on demand
    elif name == "XGBoostTrainer":
        from .non_distilled.xgboost import XGBoostTrainer

        return XGBoostTrainer
    elif name == "LightGBMTrainer":
        from .non_distilled.lightgbm import LightGBMTrainer

        return LightGBMTrainer

    # Teacher models - loaded on demand
    elif name == "BaseTeacher":
        from .teacher.base import BaseTeacher

        return BaseTeacher
    elif name == "TFTTeacher":
        from .teacher.tft_teacher import TFTTeacher

        return TFTTeacher
    elif name == "TFTTeacherConfig":
        from .teacher.tft_teacher import TFTTeacherConfig

        return TFTTeacherConfig

    # Distillation framework - loaded on demand
    elif name == "LightGBMStudentDistiller":
        from .student.lightgbm import LightGBMStudentDistiller

        return LightGBMStudentDistiller

    # Optimization utilities - loaded on demand
    elif name == "XGBoostOptunaOptimizer":
        from .optuna_optimizer import XGBoostOptunaOptimizer

        return XGBoostOptunaOptimizer

    # Safe loading utilities - loaded on demand
    elif name == "safe_torch_load":
        from .safe_torch import safe_torch_load

        return safe_torch_load

    # TorchScript utilities - loaded on demand
    elif name == "export_tft_to_torchscript_from_batch":
        from .teacher.tft_torchscript import export_tft_to_torchscript_from_batch

        return export_tft_to_torchscript_from_batch
    elif name == "TFTScriptAdapter":
        from .teacher.tft_torchscript import TFTScriptAdapter

        return TFTScriptAdapter

    # Backward compatibility - legacy names
    elif name == "UnifiedXGBoostTrainer":
        from .non_distilled.xgboost import XGBoostTrainer

        return XGBoostTrainer
    elif name == "UnifiedLightGBMTrainer":
        from .non_distilled.lightgbm import LightGBMTrainer

        return LightGBMTrainer

    # Unknown attribute
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
