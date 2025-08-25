#!/usr/bin/env python3

"""
Model Registry for orchestrating ML model lifecycle.

This module provides:
- Model registration and versioning
- Deployment tracking and management
- Performance monitoring
- A/B testing support
- Hot reload capabilities
- Rollback functionality

The registry acts as the central orchestrator for all ML components,
tracking which models are deployed where and their performance over time.
"""

from ml.registry.base import DataRequirements
from ml.registry.base import DeploymentStatus
from ml.registry.base import ModelInfo
from ml.registry.base import ModelManifest
from ml.registry.base import ModelRole
from ml.registry.dataclasses import CanaryConfig
from ml.registry.dataclasses import CanaryDeployment
from ml.registry.dataclasses import DataContract
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import QualityFlag
from ml.registry.dataclasses import QualityGate
from ml.registry.dataclasses import RolloutPlan
from ml.registry.dataclasses import StorageKind
from ml.registry.dataclasses import ValidationResult
from ml.registry.dataclasses import ValidationRule
from ml.registry.dataclasses import ValidationRuleType
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.model_registry import ModelRegistry
from ml.registry.statistics import calculate_sample_size
from ml.registry.statistics import welch_t_test
from ml.registry.strategy_registry import StrategyRegistry


__all__ = [
    "CanaryConfig",
    "CanaryDeployment",
    "DataContract",
    "DataRequirements",
    "DatasetManifest",
    "DatasetType",
    "DeploymentStatus",
    "FeatureRegistry",
    "ModelInfo",
    "ModelManifest",
    "ModelRegistry",
    "ModelRole",
    "QualityFlag",
    "QualityGate",
    "RolloutPlan",
    "StorageKind",
    "StrategyRegistry",
    "ValidationResult",
    "ValidationRule",
    "ValidationRuleType",
    "calculate_sample_size",
    "welch_t_test",
]
