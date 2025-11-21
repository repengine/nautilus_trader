"""
ML configuration classes for Nautilus Trader.

This module provides centralized access to all ML configuration classes following the
Universal ML Architecture Patterns. Configuration classes are organized into logical
groups with clear boundaries between core, training, and specialized configurations.

Configuration Groups
-------------------
- Core Configs: Base actor and inference configurations
- Training Configs: Framework-specific training configurations
- Constants: System constants and predefined values
- Specialized: Bus, observability, and runtime configurations
- Validation: Configuration validators and helpers

All configuration classes are designed to be:
- Immutable (frozen=True)
- Type-safe with complete annotations
- Environment-aware with validation
- Compatible with msgspec serialization

"""

from __future__ import annotations

# Actor configurations
from ml.config.actors import MLSignalActorConfig
from ml.config.actors import OptimizationConfig
from ml.config.actors import StrategyConfig

# Deployment and reliability
from ml.config.base import CanaryDeploymentConfig
from ml.config.base import CircuitBreakerConfig

# =============================================================================
# CORE CONFIGURATION CLASSES
# =============================================================================
# Base ML configurations
from ml.config.base import MLActorConfig
from ml.config.base import MLFeatureConfig
from ml.config.base import MLInferenceConfig
from ml.config.base import MLStrategyConfig
from ml.config.base import MLTrainingConfig
from ml.config.base import ModelDeploymentConfig
from ml.config.base import MultiModelStrategyConfig

# Message bus and observability
from ml.config.bus import MessageBusConfig

# =============================================================================
# CONSTANTS AND ENUMS
# =============================================================================
# ML constants
from ml.config.constants import FeatureColumns
from ml.config.constants import IndicatorNames
from ml.config.constants import MLConstants
from ml.config.constants import SystemConstants
from ml.config.constants import TechnicalIndicatorPeriods
from ml.config.constants import TimeConstants

# Dataset ID constants
from ml.config.dataset_ids import EARNINGS_ACTUALS_DATASET_ID
from ml.config.dataset_ids import EARNINGS_ESTIMATES_DATASET_ID
from ml.config.dataset_ids import EVENTS_CALENDAR_DATASET_ID
from ml.config.dataset_ids import L2_MINUTE_DATASET_ID
from ml.config.dataset_ids import MACRO_OBSERVATIONS_DATASET_ID
from ml.config.dataset_ids import MACRO_RELEASES_DATASET_ID
from ml.config.dataset_ids import MICRO_MINUTE_DATASET_ID

# Event and message constants
from ml.config.events import EventStatus
from ml.config.events import Source as EventSource
from ml.config.events import Stage as EventStage

# Framework-specific training
from ml.config.lightgbm import LightGBMTrainingConfig

# Backward compatibility aliases
from ml.config.lightgbm import UnifiedLightGBMConfig

# =============================================================================
# CONFIGURATION VALIDATORS AND HELPERS
# =============================================================================
from ml.config.loader import load_from_file
from ml.config.loader import merge_env
from ml.config.market_data import MarketDatasetInput
from ml.config.market_data import MarketFeedDescriptor
from ml.config.market_data import MarketFeedDescriptorSet
from ml.config.market_data import load_market_feed_descriptors
from ml.config.observability import ObservabilityConfig
from ml.config.playground import LiquidityScalingDefaults
from ml.config.playground import ThreeDRiskBacktestDefaults

# =============================================================================
# SPECIALIZED CONFIGURATIONS
# =============================================================================
# Registry and storage
from ml.config.registry import ModelRegistryConfig
from ml.config.registry import RegistryPolicyConfig

# Runtime and inference
from ml.config.runtime import OnnxRuntimeConfig

# Data collection and scheduling
from ml.config.scheduler_config import DatabentoConfig
from ml.config.scheduler_config import SchedulerConfig
from ml.config.scheduler_config import UniverseConfig

# =============================================================================
# TRAINING CONFIGURATIONS
# =============================================================================
# Advanced training features
from ml.config.shared import AdvancedTrainingConfig
from ml.config.shared import BaseGPUConfig

# GPU acceleration
from ml.config.shared import LightGBMGPUConfig
from ml.config.shared import OptunaConfig
from ml.config.shared import XGBoostGPUConfig
from ml.config.streaming_pipeline import DatasetServiceConfig
from ml.config.streaming_pipeline import StreamingPersistenceConfig
from ml.config.streaming_pipeline import StreamingWorkerConfig
from ml.config.streaming_pipeline import TrainingOrchestratorConfig
from ml.config.xgboost import UnifiedXGBoostConfig
from ml.config.xgboost import XGBoostTrainingConfig


# =============================================================================
# CONFIGURATION VALIDATION UTILITIES
# =============================================================================


def validate_ml_config(config: MLActorConfig | MLInferenceConfig) -> list[str]:
    """
    Validate ML configuration for compliance with Universal Patterns.

    Parameters
    ----------
    config : MLActorConfig | MLInferenceConfig
        The configuration to validate.

    Returns
    -------
    list[str]
        List of validation issues. Empty list indicates valid configuration.

    """
    issues = []

    # Pattern 1: Validate model specification
    if hasattr(config, "model_path") and hasattr(config, "model_id"):
        if not config.model_path and not config.model_id:
            issues.append("Either model_path or model_id must be provided")
        if config.model_path and config.model_id:
            issues.append("Cannot specify both model_path and model_id")

    # Pattern 3: Validate hot path constraints
    if hasattr(config, "max_inference_latency_ms"):
        if config.max_inference_latency_ms > 5.0:
            issues.append(
                f"max_inference_latency_ms ({config.max_inference_latency_ms}) exceeds 5ms SLA",
            )

    if hasattr(config, "max_feature_latency_ms"):
        if config.max_feature_latency_ms > 0.5:
            issues.append(
                f"max_feature_latency_ms ({config.max_feature_latency_ms}) exceeds 0.5ms SLA",
            )

    # Pattern 4: Validate fallback configuration
    if hasattr(config, "use_dummy_stores") and hasattr(config, "db_connection"):
        if not config.use_dummy_stores and not config.db_connection:
            issues.append("db_connection required when use_dummy_stores=False")

    return issues


def get_config_defaults() -> dict[str, object]:
    """
    Get default configuration values for common ML configurations.

    Returns
    -------
    dict[str, object]
        Dictionary of default configuration instances.

    """
    return {
        "ml_feature": MLFeatureConfig(),
        "ml_inference": MLInferenceConfig(model_path="./models/default.onnx"),
        "optimization": OptimizationConfig(),
        "strategy": StrategyConfig(),
        "onnx_runtime": OnnxRuntimeConfig(),
        "model_registry": ModelRegistryConfig(),
        "observability": ObservabilityConfig(),
        "message_bus": MessageBusConfig(),
    }


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "EARNINGS_ACTUALS_DATASET_ID",
    "EARNINGS_ESTIMATES_DATASET_ID",
    "EVENTS_CALENDAR_DATASET_ID",
    "L2_MINUTE_DATASET_ID",
    "MACRO_OBSERVATIONS_DATASET_ID",
    "MACRO_RELEASES_DATASET_ID",
    "MICRO_MINUTE_DATASET_ID",
    "AdvancedTrainingConfig",
    "BaseGPUConfig",
    "CanaryDeploymentConfig",
    "CircuitBreakerConfig",
    "DatabentoConfig",
    "DatasetServiceConfig",
    "EventSource",
    "EventStage",
    "EventStatus",
    "FeatureColumns",
    "IndicatorNames",
    "LightGBMGPUConfig",
    "LightGBMTrainingConfig",
    "LiquidityScalingDefaults",
    "MLActorConfig",
    "MLConstants",
    "MLFeatureConfig",
    "MLInferenceConfig",
    "MLSignalActorConfig",
    "MLStrategyConfig",
    "MLTrainingConfig",
    "MarketDatasetInput",
    "MarketFeedDescriptor",
    "MarketFeedDescriptorSet",
    "MessageBusConfig",
    "ModelDeploymentConfig",
    "ModelRegistryConfig",
    "MultiModelStrategyConfig",
    "ObservabilityConfig",
    "OnnxRuntimeConfig",
    "OptimizationConfig",
    "OptunaConfig",
    "RegistryPolicyConfig",
    "SchedulerConfig",
    "StrategyConfig",
    "StreamingPersistenceConfig",
    "StreamingWorkerConfig",
    "SystemConstants",
    "TechnicalIndicatorPeriods",
    "ThreeDRiskBacktestDefaults",
    "TimeConstants",
    "TrainingOrchestratorConfig",
    "UnifiedLightGBMConfig",
    "UnifiedXGBoostConfig",
    "UniverseConfig",
    "XGBoostGPUConfig",
    "XGBoostTrainingConfig",
    "get_config_defaults",
    "load_from_file",
    "load_market_feed_descriptors",
    "merge_env",
    "validate_ml_config",
]
