#!/usr/bin/env python3
"""
Canonical import path for the component-based ML pipeline orchestrator.

This module re-exports the facade implementation to preserve backwards
compatibility while keeping a single runtime path.
"""

from __future__ import annotations

from ml.data.ingest.orchestrator import IngestionOrchestrator
from ml.data.ingest.service import DatabentoIngestionService
from ml.data.ingest.subscription import SubscriptionPolicy as CoveragePolicy
from ml.orchestration import pipeline_orchestrator_cli as _cli
from ml.orchestration.config_loader import IngestionStageConfig
from ml.orchestration.config_types import AutoFillUniverseConfig
from ml.orchestration.config_types import DatasetBuildConfig
from ml.orchestration.pipeline_orchestrator_facade import MLPipelineOrchestratorFacade
from ml.tasks.ingest import populate_l2_efficient


MLPipelineOrchestrator = MLPipelineOrchestratorFacade

_apply_default_market_inputs = _cli._apply_default_market_inputs
_AutoFillMetrics = _cli._AutoFillMetrics
_build_auto_fill_config_from_args = _cli._build_auto_fill_config_from_args
_dataset_only_config = _cli._dataset_only_config
_parse_market_inputs_json = _cli._parse_market_inputs_json
_resolve_write_mode_tokens = _cli._resolve_write_mode_tokens
_extract_config_args = _cli._extract_config_args
_execute_stage = _cli._execute_stage
_IngestionPlanItem = _cli._IngestionPlanItem
_get_allowed_databento_datasets = _cli._get_allowed_databento_datasets
load_market_feed_descriptors = _cli.load_market_feed_descriptors
main = _cli.main
parse_args = _cli.parse_args

_CLI_BUILD_INGESTION_PLAN = _cli._build_ingestion_plan
_CLI_RUN_INGESTION_STAGE = _cli._run_ingestion_stage


def _build_ingestion_plan(
    *,
    ds_cfg: DatasetBuildConfig | None,
    ingestion_cfg: IngestionStageConfig,
) -> tuple[_cli._IngestionPlanItem, ...]:
    """
    Wrapper for ingestion-plan construction that honors local monkeypatches.
    """
    original_allowed = _cli._get_allowed_databento_datasets
    _cli._get_allowed_databento_datasets = _get_allowed_databento_datasets
    try:
        return _CLI_BUILD_INGESTION_PLAN(ds_cfg=ds_cfg, ingestion_cfg=ingestion_cfg)
    finally:
        _cli._get_allowed_databento_datasets = original_allowed


def _run_ingestion_stage(
    *,
    orch: MLPipelineOrchestratorFacade,
    ds_cfg: DatasetBuildConfig | None,
    auto_fill_cfg: AutoFillUniverseConfig,
    ingestion_cfg: IngestionStageConfig,
    ingestor: object | None,
    ingestion_service: DatabentoIngestionService | None,
) -> int:
    """
    Wrapper for ingestion stage execution that honors local monkeypatches.
    """
    original_build = _cli._build_ingestion_plan
    _cli._build_ingestion_plan = _build_ingestion_plan
    try:
        return _CLI_RUN_INGESTION_STAGE(
            orch=orch,
            ds_cfg=ds_cfg,
            auto_fill_cfg=auto_fill_cfg,
            ingestion_cfg=ingestion_cfg,
            ingestor=ingestor,
            ingestion_service=ingestion_service,
        )
    finally:
        _cli._build_ingestion_plan = original_build

__all__ = [
    "CoveragePolicy",
    "IngestionOrchestrator",
    "MLPipelineOrchestrator",
    "MLPipelineOrchestratorFacade",
    "_AutoFillMetrics",
    "_IngestionPlanItem",
    "_apply_default_market_inputs",
    "_build_auto_fill_config_from_args",
    "_build_ingestion_plan",
    "_dataset_only_config",
    "_execute_stage",
    "_extract_config_args",
    "_get_allowed_databento_datasets",
    "_parse_market_inputs_json",
    "_resolve_write_mode_tokens",
    "_run_ingestion_stage",
    "load_market_feed_descriptors",
    "main",
    "parse_args",
    "populate_l2_efficient",
]
