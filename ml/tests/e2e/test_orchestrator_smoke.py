#!/usr/bin/env python3
"""
System validation smoke tests for MLPipelineOrchestrator.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ml.orchestration import MLPipelineOrchestrator
from ml.orchestration.config_types import DatasetBuildConfig


pytestmark = [
    pytest.mark.usefixtures("isolated_prometheus_registry"),
]


@pytest.fixture
def mock_coverage() -> MagicMock:
    """Mock coverage provider."""
    coverage = MagicMock()
    coverage.read_bucket_coverage.return_value = set()
    return coverage


@pytest.fixture
def mock_writer() -> MagicMock:
    """Mock market data writer."""
    writer = MagicMock()
    writer.write_bars.return_value = 0
    return writer


@pytest.fixture
def mock_build_main() -> MagicMock:
    """Mock build CLI that simulates successful dataset build."""
    return MagicMock(return_value=0)


@pytest.fixture
def mock_teacher_main() -> MagicMock:
    """Mock teacher CLI that simulates successful training."""
    return MagicMock(return_value=0)


@pytest.fixture
def orchestrator(
    mock_coverage: MagicMock,
    mock_writer: MagicMock,
    mock_build_main: MagicMock,
    mock_teacher_main: MagicMock,
) -> MLPipelineOrchestrator:
    """Create orchestrator with minimal dependencies."""
    return MLPipelineOrchestrator(
        coverage=mock_coverage,
        writer=mock_writer,
        build_main=mock_build_main,
        teacher_main=mock_teacher_main,
    )


def test_components_initialized(orchestrator: MLPipelineOrchestrator) -> None:
    """Core components should be initialized."""
    assert orchestrator._stage_controller is not None
    assert orchestrator._dataset_builder is not None
    assert orchestrator._training_coordinator is not None
    assert orchestrator._ingestion_coordinator is not None


def test_health_status_reports_component_mode(orchestrator: MLPipelineOrchestrator) -> None:
    """Health status reports component-based implementation."""
    health = orchestrator.get_health_status()
    assert health["implementation"] == "component-based"


def test_build_dataset_delegates(orchestrator: MLPipelineOrchestrator, tmp_path: Path) -> None:
    """build_dataset delegates to DatasetBuilder."""
    cfg = DatasetBuildConfig(
        data_dir=str(tmp_path / "data"),
        symbols="SPY",
        out_dir=str(tmp_path / "output"),
        dataset_id="smoke_test",
    )
    (tmp_path / "data").mkdir(parents=True)

    orchestrator._dataset_builder.build_dataset = MagicMock(return_value=0)

    result = orchestrator.build_dataset(cfg)
    assert result == 0
    orchestrator._dataset_builder.build_dataset.assert_called_once_with(cfg)
