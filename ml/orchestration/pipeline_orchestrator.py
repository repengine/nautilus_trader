#!/usr/bin/env python3
"""
Canonical import path for the component-based ML pipeline orchestrator.

This module re-exports the facade implementation to preserve backwards
compatibility while keeping a single runtime path.
"""

from __future__ import annotations

from ml.orchestration.pipeline_orchestrator_cli import _dataset_only_config
from ml.orchestration.pipeline_orchestrator_cli import _run_ingestion_stage
from ml.orchestration.pipeline_orchestrator_cli import main
from ml.orchestration.pipeline_orchestrator_facade import MLPipelineOrchestratorFacade


MLPipelineOrchestrator = MLPipelineOrchestratorFacade

__all__ = [
    "MLPipelineOrchestrator",
    "MLPipelineOrchestratorFacade",
    "_dataset_only_config",
    "_run_ingestion_stage",
    "main",
]
