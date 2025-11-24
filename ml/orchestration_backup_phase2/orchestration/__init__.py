#!/usr/bin/env python3

"""
ML Pipeline Orchestration Module.

This module provides cold-path orchestration components for managing ML pipelines,
including scheduling, configuration management, and model promotion workflows.
All components are designed for batch/offline operations and must never be used
in hot-path actor code.

Universal ML Architecture Patterns Compliance:
- Pattern 1: N/A (orchestration components don't inherit from BaseMLInferenceActor)
- Pattern 2: Uses Protocol-first interface design for extensible components
- Pattern 3: Strictly cold-path only - no hot-path operations
- Pattern 4: Progressive fallback chains for external dependencies
- Pattern 5: Uses centralized metrics bootstrap (ml.common.metrics_bootstrap)

Notes
-----
- All orchestration components are cold-path only and should never be imported
  by actors, strategies, or other hot-path code
- Configuration is declarative via dataclasses with frozen=True for immutability
- Components use centralized event emission and metrics collection
- Progressive fallback strategies ensure resilience for external dependencies

"""

from __future__ import annotations

# Core orchestrator classes
from ml.orchestration.binding_resolver import BindingResolver
from ml.orchestration.binding_resolver import BindingResolverProtocol
from ml.orchestration.config_loader import IngestionStageConfig
from ml.orchestration.config_loader import OrchestratorRunConfig
from ml.orchestration.config_loader import Stage
from ml.orchestration.config_loader import TrainingStageConfig
from ml.orchestration.config_loader import load_orchestrator_config
from ml.orchestration.config_loader import load_orchestrator_run_config
from ml.orchestration.config_loader import to_pipeline_args
from ml.orchestration.config_resolver import ConfigResolver
from ml.orchestration.config_resolver import ConfigResolverProtocol
from ml.orchestration.config_types import AutoFillUniverseConfig
from ml.orchestration.config_types import DatasetBuildConfig
from ml.orchestration.config_types import HPOConfig
from ml.orchestration.config_types import IntegrationConfig
from ml.orchestration.config_types import OrchestratorConfig
from ml.orchestration.config_types import PreIngestionOptions
from ml.orchestration.config_types import PromotionsConfig
from ml.orchestration.config_types import StudentDistillConfig
from ml.orchestration.config_types import TeacherTrainConfig
from ml.orchestration.dataset_builder import BuildArtifacts
from ml.orchestration.dataset_builder import DatasetBuilder
from ml.orchestration.dataset_builder import DatasetBuilderProtocol
from ml.orchestration.discovery_client import DiscoveryClient
from ml.orchestration.discovery_client import DiscoveryClientProtocol
from ml.orchestration.ingestion_coordinator import IngestionCoordinator
from ml.orchestration.ingestion_coordinator import IngestionCoordinatorProtocol
from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator

# Promotion helpers
from ml.orchestration.promotions import register_and_promote_model
from ml.orchestration.promotions import register_or_refresh_features

# Scheduling utilities
from ml.orchestration.scheduler import compute_next_run
from ml.orchestration.scheduler import run_forever

# Configuration loading utilities
from . import config_loader


__all__ = [
    "AutoFillUniverseConfig",
    "BindingResolver",
    "BindingResolverProtocol",
    "BuildArtifacts",
    "ConfigResolver",
    "ConfigResolverProtocol",
    "DatasetBuildConfig",
    "DatasetBuilder",
    "DatasetBuilderProtocol",
    "DiscoveryClient",
    "DiscoveryClientProtocol",
    "HPOConfig",
    "IngestionCoordinator",
    "IngestionCoordinatorProtocol",
    "IngestionStageConfig",
    "IntegrationConfig",
    "MLPipelineOrchestrator",
    "OrchestratorConfig",
    "OrchestratorRunConfig",
    "PreIngestionOptions",
    "PromotionsConfig",
    "Stage",
    "StudentDistillConfig",
    "TeacherTrainConfig",
    "TrainingStageConfig",
    "compute_next_run",
    "config_loader",
    "load_orchestrator_config",
    "load_orchestrator_run_config",
    "register_and_promote_model",
    "register_or_refresh_features",
    "run_forever",
    "to_pipeline_args",
]
