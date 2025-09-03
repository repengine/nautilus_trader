"""
ML configuration classes for Nautilus Trader.
"""

from ml.config.actors import MLSignalActorConfig
from ml.config.actors import OptimizationConfig
from ml.config.actors import StrategyConfig
from ml.config.base import CanaryDeploymentConfig
from ml.config.base import CircuitBreakerConfig
from ml.config.base import MLActorConfig
from ml.config.base import MLFeatureConfig
from ml.config.base import MLInferenceConfig
from ml.config.base import MLStrategyConfig
from ml.config.base import MLTrainingConfig
from ml.config.base import ModelDeploymentConfig
from ml.config.base import MultiModelStrategyConfig
from ml.config.constants import FeatureColumns
from ml.config.constants import IndicatorNames
from ml.config.constants import MLConstants
from ml.config.constants import SystemConstants
from ml.config.constants import TechnicalIndicatorPeriods
from ml.config.constants import TimeConstants
from ml.config.lightgbm import LightGBMTrainingConfig
from ml.config.lightgbm import UnifiedLightGBMConfig  # Backward compatibility
from ml.config.registry import ModelRegistryConfig
from ml.config.runtime import OnnxRuntimeConfig
from ml.config.shared import AdvancedTrainingConfig
from ml.config.shared import BaseGPUConfig
from ml.config.shared import LightGBMGPUConfig

# MLflowConfig removed - deprecated
from ml.config.shared import OptunaConfig
from ml.config.shared import XGBoostGPUConfig
from ml.config.xgboost import UnifiedXGBoostConfig  # Backward compatibility
from ml.config.xgboost import XGBoostTrainingConfig


__all__ = [
    "AdvancedTrainingConfig",
    "BaseGPUConfig",
    "CanaryDeploymentConfig",
    "CircuitBreakerConfig",
    "FeatureColumns",
    "IndicatorNames",
    "LightGBMGPUConfig",
    "LightGBMTrainingConfig",
    "MLActorConfig",
    "MLConstants",
    "MLFeatureConfig",
    "MLInferenceConfig",
    "MLSignalActorConfig",
    "MLStrategyConfig",
    "MLTrainingConfig",
    "ModelDeploymentConfig",
    "ModelRegistryConfig",
    "MultiModelStrategyConfig",
    "OnnxRuntimeConfig",
    "OptimizationConfig",
    "OptunaConfig",
    "StrategyConfig",
    "SystemConstants",
    "TechnicalIndicatorPeriods",
    "TimeConstants",
    "UnifiedLightGBMConfig",
    "UnifiedXGBoostConfig",
    "XGBoostGPUConfig",
    "XGBoostTrainingConfig",
]
