#!/usr/bin/env python3

"""
ML Registry Package - 4 Mandatory Registries + Supporting Components

This module provides the 4 mandatory registries required by Pattern 1 (Universal ML Architecture):
- FeatureRegistry: Feature schema validation and lifecycle management
- ModelRegistry: Model deployment tracking and A/B testing
- StrategyRegistry: Strategy compatibility and requirement validation
- DataRegistry: Dataset manifest management and lineage tracking

All actors MUST use these 4 registries via BaseMLInferenceActor inheritance.

Usage:
    from ml.registry import FeatureRegistry, ModelRegistry, StrategyRegistry, DataRegistry
    from ml.registry import RegistryProtocol, TypedRegistryProtocol

Registry System Components:
    - Protocols: RegistryProtocol, TypedRegistryProtocol (structural typing interfaces)
    - Base Classes: AbstractRegistry, DummyRegistry (implementation foundation)
    - Data Classes: Model/Feature/Strategy/Dataset manifests and metadata
    - Support: Statistical utilities, persistence layer, validation helpers

Thread-safety: All registries are thread-safe for concurrent operations.
Backends: Configurable JSON (development) or PostgreSQL (production) persistence.
"""

# =============================================================================
# 4 MANDATORY REGISTRIES (Pattern 1 Requirement)
# =============================================================================

from ml.registry.abstract_registry import AbstractRegistry
from ml.registry.base import DataRequirements

# =============================================================================
# CORE DATA CLASSES & MANIFESTS
# =============================================================================
# Model Registry Types
from ml.registry.base import DeploymentStatus
from ml.registry.base import DummyRegistry
from ml.registry.base import ModelInfo
from ml.registry.base import ModelManifest
from ml.registry.base import ModelRole
from ml.registry.data_registry import DataRegistry
from ml.registry.data_registry import Watermark

# =============================================================================
# DEPLOYMENT & TESTING SUPPORT
# =============================================================================
# A/B Testing & Canary Deployments
from ml.registry.dataclasses import CanaryConfig
from ml.registry.dataclasses import CanaryDeployment

# Data Registry Types
from ml.registry.dataclasses import DataContract
from ml.registry.dataclasses import DatasetLineageRecord
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import QualityFlag
from ml.registry.dataclasses import QualityGate
from ml.registry.dataclasses import RolloutPlan
from ml.registry.dataclasses import StorageKind
from ml.registry.dataclasses import ValidationResult
from ml.registry.dataclasses import ValidationRule
from ml.registry.dataclasses import ValidationRuleType

# Feature Registry Types
from ml.registry.feature_registry import FeatureInfo
from ml.registry.feature_registry import FeatureManifest
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.feature_registry import FeatureRole
from ml.registry.feature_registry import FeatureStage
from ml.registry.feature_registry import compute_schema_hash
from ml.registry.ab_testing_manager import ABTestingManager
from ml.registry.canary_deployment_mgr import CanaryDeploymentManager
from ml.registry.model_deployment_mgr import ModelDeploymentManager
from ml.registry.model_persistence import ModelPersistence
from ml.registry.model_quality_validator import ModelQualityValidator
from ml.registry.model_registry import ModelRegistry

# =============================================================================
# PERSISTENCE & CONFIGURATION
# =============================================================================
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.registry.persistence import PersistenceManager

# =============================================================================
# REGISTRY PROTOCOLS & BASE CLASSES
# =============================================================================
from ml.registry.protocols import RegistryProtocol
from ml.registry.protocols import TypedRegistryProtocol

# Statistical Utilities
from ml.registry.statistics import calculate_sample_size
from ml.registry.statistics import welch_t_test
from ml.registry.strategy_registry import MarketRegime

# Strategy Registry Types
from ml.registry.strategy_registry import StrategyInfo
from ml.registry.strategy_registry import StrategyManifest
from ml.registry.strategy_registry import StrategyRegistry
from ml.registry.strategy_registry import StrategyType
from ml.registry.summaries import ModelSummary
from ml.registry.summaries import build_model_summaries
from ml.registry.utils import assert_features_compatible

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
from ml.registry.utils import build_feature_schema
from ml.registry.utils import build_student_manifest


# =============================================================================
# PUBLIC API DEFINITION
# =============================================================================

__all__ = [
    "ABTestingManager",
    "AbstractRegistry",
    "BackendType",
    "CanaryConfig",
    "CanaryDeployment",
    "CanaryDeploymentManager",
    "DataContract",
    "DataRegistry",
    "DataRequirements",
    "DatasetLineageRecord",
    "DatasetManifest",
    "DatasetType",
    "DeploymentStatus",
    "DummyRegistry",
    "FeatureInfo",
    "FeatureManifest",
    "FeatureRegistry",
    "FeatureRole",
    "FeatureStage",
    "MarketRegime",
    "ModelDeploymentManager",
    "ModelInfo",
    "ModelManifest",
    "ModelPersistence",
    "ModelQualityValidator",
    "ModelRegistry",
    "ModelRole",
    "ModelSummary",
    "PersistenceConfig",
    "PersistenceManager",
    "QualityFlag",
    "QualityGate",
    "RegistryProtocol",
    "RolloutPlan",
    "StorageKind",
    "StrategyInfo",
    "StrategyManifest",
    "StrategyRegistry",
    "StrategyType",
    "TypedRegistryProtocol",
    "ValidationResult",
    "ValidationRule",
    "ValidationRuleType",
    "Watermark",
    "assert_features_compatible",
    "build_feature_schema",
    "build_model_summaries",
    "build_student_manifest",
    "calculate_sample_size",
    "compute_schema_hash",
    "welch_t_test",
]
