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

# Configuration loading utilities
from ml.orchestration.config_loader import load_orchestrator_config
from ml.orchestration.config_loader import to_pipeline_args

# Core orchestrator classes
from ml.orchestration.pipeline_orchestrator import DatasetBuildConfig
from ml.orchestration.pipeline_orchestrator import HPOConfig
from ml.orchestration.pipeline_orchestrator import IntegrationConfig
from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator
from ml.orchestration.pipeline_orchestrator import OrchestratorConfig
from ml.orchestration.pipeline_orchestrator import PreIngestionOptions
from ml.orchestration.pipeline_orchestrator import PromotionsConfig
from ml.orchestration.pipeline_orchestrator import StudentDistillConfig
from ml.orchestration.pipeline_orchestrator import TeacherTrainConfig

# Promotion helpers
from ml.orchestration.promotions import register_and_promote_model
from ml.orchestration.promotions import register_or_refresh_features

# Scheduling utilities
from ml.orchestration.scheduler import compute_next_run
from ml.orchestration.scheduler import run_forever


__all__ = [
    "DatasetBuildConfig",
    "HPOConfig",
    "IntegrationConfig",
    "MLPipelineOrchestrator",
    "OrchestratorConfig",
    "PreIngestionOptions",
    "PromotionsConfig",
    "StudentDistillConfig",
    "TeacherTrainConfig",
    "compute_next_run",
    "load_orchestrator_config",
    "register_and_promote_model",
    "register_or_refresh_features",
    "run_forever",
    "to_pipeline_args",
]
