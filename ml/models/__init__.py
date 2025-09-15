# ruff: noqa: RUF022
"""
ML Models Package.

This package provides model abstractions, utilities, and implementations for the Nautilus Trader
ML pipeline. It follows cold path patterns - all functionality here is for training, loading,
and offline operations. Hot path inference uses pre-loaded models through actors.

Key Components:
- Model type detection and classification
- Model export and conversion utilities
- Training contracts and mixins
- Model loaders for different formats
- Dummy models for testing

Architecture Notes:
- All model training and loading operations are cold path only
- Hot path inference actors use pre-loaded models from registries
- Model exports follow ONNX-first approach for production inference
- Security: No pickle loading in production, safe formats only
- Progressive fallback: ONNX → native framework formats → errors

Usage Examples:
    # Model type detection
    from ml.models import ModelType, detect_model_type
    model_type = detect_model_type(my_model, Path("model.onnx"))

    # Export for production
    from ml.models import convert_to_onnx, save_model_with_metadata
    onnx_path = convert_to_onnx(model, sample_input, "output.onnx")

    # Loading in actors (cold path initialization only)
    from ml.models import ProductionModelLoader
    loader = ProductionModelLoader("/models")
    model, metadata = loader.load_model("my_model.onnx")

    # Testing with dummy models
    from ml.models import DummyModel
    dummy = DummyModel(n_features=10)
    predictions = dummy.predict(test_features)

Security Policy:
- Pickle models (.pkl) are NEVER loaded in production
- Joblib models only allowed in explicit testing contexts
- ONNX models preferred for inference (secure, optimized)
- All model files validated before loading

"""

from __future__ import annotations

# Model loaders (for cold path initialization)
from ml.actors.base import ModelLoader
from ml.actors.base import ONNXModelLoader
from ml.actors.base import ProductionModelLoader

# Dummy model for testing
from ml.models.save_dummy_model import DummyModel

# Training base classes (cold path)
from ml.training.base import BaseMLTrainer
from ml.training.export import DEFAULT_ONNX_OPSET
from ml.training.export import ModelExportMixin

# Core model abstractions and utilities (cold path)
from ml.training.export import ModelType
from ml.training.export import TrainingActorContract
from ml.training.export import convert_to_onnx
from ml.training.export import convert_to_torchscript
from ml.training.export import detect_model_type
from ml.training.export import save_model_with_metadata


# ruff: noqa: RUF022
__all__ = [
    "BaseMLTrainer",
    "DEFAULT_ONNX_OPSET",
    "DummyModel",
    "ModelExportMixin",
    "ModelLoader",
    "ModelType",
    "ONNXModelLoader",
    "ProductionModelLoader",
    "TrainingActorContract",
    "convert_to_onnx",
    "convert_to_torchscript",
    "detect_model_type",
    "save_model_with_metadata",
]


# Development notes for maintainers:
#
# 1. Cold Path Only: This package is for training, loading, and offline operations.
#    Hot path inference should use pre-loaded models in actors.
#
# 2. Security First: Never expose pickle loading functionality. Use safe formats:
#    - ONNX (preferred for production inference)
#    - Native framework formats (XGBoost JSON, LightGBM text)
#    - Joblib only in testing contexts
#
# 3. Universal Patterns Compliance:
#    - Pattern 1: Not applicable (no stores/registries in models package)
#    - Pattern 2: Use protocols for model interfaces where applicable
#    - Pattern 3: Strictly cold path - no hot path operations
#    - Pattern 4: Progressive fallback in model loading
#    - Pattern 5: Use metrics_bootstrap if adding metrics
#
# 4. Adding New Model Types:
#    - Update ModelType enum in training/export.py
#    - Add detection logic in detect_model_type()
#    - Implement conversion to ONNX for production inference
#    - Add security validation in ProductionModelLoader
#    - Write comprehensive tests including security tests
#
# 5. Import Strategy:
#    - Re-export from other ml/ modules rather than implementing here
#    - Keep this package focused on coordination and testing utilities
#    - Heavy implementations belong in training/ or actors/ packages
#
# 6. Testing:
#    - DummyModel provides simple testing functionality
#    - Use model_factories from tests/fixtures/ for complex test scenarios
#    - Test all model loading paths including security failures
#
# 7. Documentation:
#    - All public classes must have comprehensive docstrings
#    - Include usage examples for common patterns
#    - Document security implications of each loader
#    - Cross-reference with training and actors documentation
