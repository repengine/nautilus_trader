"""
Nautilus ML - Machine Learning Integration for Nautilus Trader.

This package provides a comprehensive ML framework that integrates seamlessly
with Nautilus Trader's high-performance trading infrastructure.

Key Features
------------
- Strict hot/cold path separation for optimal performance
- Feature engineering consistency between training and inference
- MLflow integration for model lifecycle management
- Actor-based real-time inference architecture
- Comprehensive testing with 90%+ coverage requirement

Architecture
------------
The ML package follows a clear separation of concerns:

Cold Path (Training):
    - Data loading and feature engineering with Polars
    - Model training with XGBoost, LightGBM, Neural Networks
    - Hyperparameter optimization with Optuna
    - Model registry and versioning with MLflow

Hot Path (Inference):
    - Real-time feature computation with numpy
    - Low-latency model inference (<5ms requirement)
    - Actor-based signal generation
    - Message bus integration for signal distribution

"""

from ml._imports import HAS_LIGHTGBM
from ml._imports import HAS_ONNX
from ml._imports import HAS_OPTUNA
from ml._imports import HAS_POLARS
from ml._imports import HAS_XGBOOST
from ml._imports import check_ml_dependencies


__version__ = "0.1.0"

__all__ = [
    "HAS_LIGHTGBM",
    "HAS_ONNX",
    "HAS_OPTUNA",
    "HAS_POLARS",
    "HAS_XGBOOST",
    "check_ml_dependencies",
]
