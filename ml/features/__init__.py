
"""
Feature engineering and validation for ML components.

This module provides feature engineering capabilities with perfect consistency between
batch (training) and real-time (inference) computation paths.

"""

from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.features.engineering import IndicatorManager
from ml.features.validation import FeatureParityValidator
from ml.features.validation import validate_feature_parity


__all__ = [
    "FeatureConfig",
    "FeatureEngineer",
    "FeatureParityValidator",
    "IndicatorManager",
    "validate_feature_parity",
]
