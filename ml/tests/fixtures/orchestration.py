#!/usr/bin/env python3
"""
Orchestration fixtures for ML pipeline tests.

This module provides mock fixtures for testing the decomposed MLPipelineOrchestrator
components including StageController, IngestionCoordinator, DatasetBuilder, etc.

Usage:
    def test_stage_controller(mock_stage_controller, sample_orchestrator_config):
        result = mock_stage_controller.run_pipeline(sample_orchestrator_config)
        assert result == 0
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest


if TYPE_CHECKING:
    from ml.data import DatasetMetadata
    from ml.orchestration.config_types import (
        DatasetBuildConfig,
        OrchestratorConfig,
    )


@pytest.fixture
def mock_stage_controller() -> Mock:
    """
    Mock StageController for unit tests.

    The StageController orchestrates pipeline stages in order and handles
    checkpointing. This mock provides default success behavior.

    Returns:
        Mock with StageController interface
    """
    controller = Mock()
    controller.run_pipeline.return_value = 0
    controller.run_training_only.return_value = 0
    controller.save_checkpoint.return_value = None
    controller.load_checkpoint.return_value = []
    return controller


@pytest.fixture
def mock_ingestion_coordinator() -> Mock:
    """
    Mock IngestionCoordinator for unit tests.

    The IngestionCoordinator handles data ingestion operations including
    pre-ingestion, backfill, and coverage queries.

    Returns:
        Mock with IngestionCoordinator interface
    """
    coordinator = Mock()
    coordinator.run_pre_ingestion.return_value = None
    # BackfillWindowList-like return
    backfill_result = Mock()
    backfill_result.rows_written = 100
    backfill_result.window_count = 1
    coordinator.backfill.return_value = backfill_result
    coordinator.backfill_binding.return_value = {}
    coordinator.backfill_coverage.return_value = []
    return coordinator


@pytest.fixture
def mock_dataset_builder() -> Mock:
    """
    Mock DatasetBuilder for unit tests.

    The DatasetBuilder handles dataset construction operations.

    Returns:
        Mock with DatasetBuilder interface
    """
    builder = Mock()
    builder.build_dataset.return_value = 0
    builder.prepare_dataset_config.return_value = Mock()
    return builder


@pytest.fixture
def mock_training_coordinator() -> Mock:
    """
    Mock TrainingCoordinator for unit tests.

    The TrainingCoordinator handles HPO, teacher training, and student
    distillation operations.

    Returns:
        Mock with TrainingCoordinator interface
    """
    coordinator = Mock()
    coordinator.run_hpo.return_value = 0
    coordinator.train_teacher.return_value = 0
    coordinator.distill_student.return_value = 0
    return coordinator


@pytest.fixture
def mock_registry_synchronizer() -> Mock:
    """
    Mock RegistrySynchronizer for unit tests.

    The RegistrySynchronizer handles registry operations during pipeline
    execution including feature and model synchronization.

    Returns:
        Mock with RegistrySynchronizer interface
    """
    synchronizer = Mock()
    synchronizer.sync_features.return_value = None
    synchronizer.sync_model.return_value = None
    return synchronizer


@pytest.fixture
def mock_runtime_attacher() -> Mock:
    """
    Mock RuntimeAttacher for unit tests.

    The RuntimeAttacher handles runtime attachment operations after
    training completes.

    Returns:
        Mock with RuntimeAttacher interface
    """
    attacher = Mock()
    attacher.attach_runtime.return_value = None
    return attacher


@pytest.fixture
def mock_config_resolver() -> Mock:
    """
    Mock ConfigResolver for unit tests.

    The ConfigResolver handles configuration parsing and validation.

    Returns:
        Mock with ConfigResolver interface
    """
    resolver = Mock()
    resolver.resolve.return_value = Mock()
    resolver.parse_symbols.return_value = ["SPY"]
    return resolver


@pytest.fixture
def mock_discovery_client() -> Mock:
    """
    Mock DiscoveryClient for unit tests.

    The DiscoveryClient handles dataset and service discovery operations.

    Returns:
        Mock with DiscoveryClient interface
    """
    client = Mock()
    client.discover_datasets.return_value = []
    client.discover_symbol_dataset.return_value = None
    return client


@pytest.fixture
def sample_orchestrator_config(tmp_path: Path) -> OrchestratorConfig:
    """
    Sample OrchestratorConfig for testing.

    Creates a minimal valid configuration with temporary directories
    for data and output.

    Args:
        tmp_path: Pytest temporary path fixture

    Returns:
        OrchestratorConfig with test values
    """
    from ml.orchestration.config_types import (
        DatasetBuildConfig,
        HPOConfig,
        OrchestratorConfig,
        TeacherTrainConfig,
    )
    from ml.tests.utils.targets import build_default_target_semantics_payload

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    return OrchestratorConfig(
        dataset=DatasetBuildConfig(
            data_dir=str(data_dir),
            symbols="SPY",
            out_dir=str(out_dir),
            dataset_id="test_dataset",
            target_semantics=build_default_target_semantics_payload(),
        ),
        hpo=HPOConfig(enabled=False),
        teacher=TeacherTrainConfig(enabled=True),
    )


@pytest.fixture
def sample_dataset_config(tmp_path: Path) -> DatasetBuildConfig:
    """
    Sample DatasetBuildConfig for testing.

    Creates a minimal valid dataset configuration.

    Args:
        tmp_path: Pytest temporary path fixture

    Returns:
        DatasetBuildConfig with test values
    """
    from ml.orchestration.config_types import DatasetBuildConfig
    from ml.tests.utils.targets import build_default_target_semantics_payload

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    return DatasetBuildConfig(
        data_dir=str(data_dir),
        symbols="SPY",
        out_dir=str(out_dir),
        dataset_id="test_dataset",
        target_semantics=build_default_target_semantics_payload(),
    )


@pytest.fixture
def sample_checkpoint_file(tmp_path: Path) -> Path:
    """
    Sample checkpoint file path for testing.

    Returns a path to a checkpoint file that doesn't exist yet,
    allowing tests to create it as needed.

    Args:
        tmp_path: Pytest temporary path fixture

    Returns:
        Path to checkpoint file (not created)
    """
    return tmp_path / "checkpoint.json"


@pytest.fixture
def existing_checkpoint(sample_checkpoint_file: Path) -> Path:
    """
    Create an existing checkpoint file with sample data.

    Creates a checkpoint file with some completed stages for
    testing checkpoint resume functionality.

    Args:
        sample_checkpoint_file: Path fixture for checkpoint file

    Returns:
        Path to created checkpoint file
    """
    import json
    import time

    checkpoint_data = {
        "pipeline_id": "test_pipeline_001",
        "stage": "DATASET",
        "timestamp": time.time_ns(),
        "state": {},
        "completed_stages": ["PRE_INGEST", "DATASET"],
        "progress": 0.5,
    }

    sample_checkpoint_file.write_text(json.dumps(checkpoint_data), encoding="utf-8")
    return sample_checkpoint_file


@pytest.fixture
def multi_symbol_config(tmp_path: Path) -> OrchestratorConfig:
    """
    Multi-symbol OrchestratorConfig for testing.

    Creates a configuration with multiple symbols to test
    multi-symbol processing isolation.

    Args:
        tmp_path: Pytest temporary path fixture

    Returns:
        OrchestratorConfig with multiple symbols
    """
    from ml.orchestration.config_types import (
        DatasetBuildConfig,
        HPOConfig,
        OrchestratorConfig,
        TeacherTrainConfig,
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    return OrchestratorConfig(
        dataset=DatasetBuildConfig(
            data_dir=str(data_dir),
            symbols="AAPL,GOOGL,MSFT",
            out_dir=str(out_dir),
            dataset_id="multi_symbol_test",
            target_semantics=build_default_target_semantics_payload(),
        ),
        hpo=HPOConfig(enabled=False),
        teacher=TeacherTrainConfig(enabled=True),
    )


@pytest.fixture
def sample_dataset_metadata() -> DatasetMetadata:
    """
    Sample DatasetMetadata for testing registry synchronizer and validation.

    Creates a minimal valid dataset metadata instance matching the default
    sample_dataset_config fixture values.

    Returns:
        DatasetMetadata with test values matching sample_dataset_config
    """
    from ml.data import DatasetMetadata
    from ml.data.vintage import VintagePolicy

    return DatasetMetadata(
        dataset_id="test_dataset",
        vintage_policy=VintagePolicy.REAL_TIME,
        vintage_cutoff=None,
        build_ts="2024-01-01T00:00:00Z",
        ts_event_start=None,
        ts_event_end=None,
        overall_window=None,
        train_window=None,
        validation_window=None,
        test_window=None,
        macro_observation_counts={},
        market_bindings=None,
    )


__all__ = [
    "existing_checkpoint",
    "mock_config_resolver",
    "mock_dataset_builder",
    "mock_discovery_client",
    "mock_ingestion_coordinator",
    "mock_registry_synchronizer",
    "mock_runtime_attacher",
    "mock_stage_controller",
    "mock_training_coordinator",
    "multi_symbol_config",
    "sample_checkpoint_file",
    "sample_dataset_config",
    "sample_dataset_metadata",
    "sample_orchestrator_config",
]
