#!/usr/bin/env python3

"""
Fixtures for orchestration component tests.

Provides mock objects and sample configurations for testing
StageController and related orchestration components.

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest


if TYPE_CHECKING:
    from ml.orchestration.config_types import OrchestratorConfig


@pytest.fixture
def mock_ingestion_coordinator() -> Mock:
    """Create a mock IngestionCoordinator."""
    mock = Mock()
    mock.run_pre_ingestion.return_value = 0
    mock.backfill.return_value = 0
    mock.backfill_binding.return_value = 0
    return mock


@pytest.fixture
def mock_dataset_builder() -> Mock:
    """Create a mock DatasetBuilder."""
    mock = Mock()
    mock.build_dataset.return_value = 0
    mock.validate_dataset.return_value = True
    return mock


@pytest.fixture
def mock_training_coordinator() -> Mock:
    """Create a mock TrainingCoordinator."""
    mock = Mock()
    mock.train_teacher.return_value = 0
    mock.distill_student.return_value = 0
    mock.run_hpo.return_value = 0
    return mock


@pytest.fixture
def mock_registry_synchronizer() -> Mock:
    """Create a mock RegistrySynchronizer."""
    mock = Mock()
    mock.synchronize_dataset_manifest.return_value = None
    mock.capture_cli_build_artifacts.return_value = None
    return mock


@pytest.fixture
def sample_orchestrator_config(tmp_path: Path) -> OrchestratorConfig:
    """Create a sample OrchestratorConfig for testing."""
    from ml.orchestration.config_types import DatasetBuildConfig
    from ml.orchestration.config_types import HPOConfig
    from ml.orchestration.config_types import OrchestratorConfig
    from ml.orchestration.config_types import TeacherTrainConfig

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "test_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    return OrchestratorConfig(
        dataset=DatasetBuildConfig(
            data_dir=str(data_dir),
            dataset_id="test_dataset",
            symbols="AAPL",
            out_dir=str(out_dir),
        ),
        hpo=HPOConfig(enabled=False),
        teacher=TeacherTrainConfig(enabled=True),
    )


@pytest.fixture
def multi_symbol_config(tmp_path: Path) -> OrchestratorConfig:
    """Create a multi-symbol OrchestratorConfig for testing."""
    from ml.orchestration.config_types import DatasetBuildConfig
    from ml.orchestration.config_types import HPOConfig
    from ml.orchestration.config_types import OrchestratorConfig
    from ml.orchestration.config_types import TeacherTrainConfig

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "test_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    return OrchestratorConfig(
        dataset=DatasetBuildConfig(
            data_dir=str(data_dir),
            dataset_id="test_dataset",
            symbols="AAPL,GOOGL,MSFT",
            out_dir=str(out_dir),
        ),
        hpo=HPOConfig(enabled=False),
        teacher=TeacherTrainConfig(enabled=True),
    )


@pytest.fixture
def sample_checkpoint_file(tmp_path: Path) -> Path:
    """Create a path for checkpoint file."""
    return tmp_path / "checkpoint.json"
