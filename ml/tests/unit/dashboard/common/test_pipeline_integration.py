"""
Unit tests for PipelineIntegrationComponent.

Tests extracted pipeline integration methods from DashboardService.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from ml.config.events import EventStatus
from ml.dashboard.common.pipeline_integration import PipelineIntegrationComponent
from ml.dashboard.config import DashboardConfig


@pytest.fixture
def dashboard_config() -> DashboardConfig:
    """Create a basic dashboard configuration for testing."""
    return DashboardConfig(
        db_connection="postgresql://test:test@localhost:5432/test_db",
        actor_port=9001,
        strategy_port=9002,
        pipeline_port=9003,
    )


@pytest.fixture
def component(dashboard_config: DashboardConfig) -> PipelineIntegrationComponent:
    """Create a PipelineIntegrationComponent instance."""
    return PipelineIntegrationComponent(dashboard_config)


# trigger_pipeline tests


def test_trigger_pipeline_success(component: PipelineIntegrationComponent) -> None:
    """Test triggering a pipeline successfully."""
    mock_service = Mock()
    mock_result = Mock(
        success=True,
        job_id="build_dataset_abc123",
        pipeline_type="build_dataset",
        status="QUEUED",
        message="Pipeline queued successfully",
        error=None,
    )

    with (
        patch.object(component, "_get_pipeline_service", return_value=mock_service),
        patch("ml.dashboard.common.pipeline_integration.asyncio.run", return_value=mock_result),
    ):
        result = component.trigger_pipeline("build_dataset", {"symbols": "AAPL"})

    assert result["success"] is True
    assert result["job_id"] == "build_dataset_abc123"
    assert result["pipeline_type"] == "build_dataset"
    assert result["status"] == "QUEUED"
    assert result["message"] == "Pipeline queued successfully"
    assert result["error"] is None


def test_trigger_pipeline_service_unavailable(component: PipelineIntegrationComponent) -> None:
    """Test triggering a pipeline when service is unavailable."""
    with patch.object(component, "_get_pipeline_service", return_value=None):
        result = component.trigger_pipeline("build_dataset", {"symbols": "AAPL"})

    assert result["success"] is False
    assert result["status"] == "UNAVAILABLE"
    assert result["pipeline_type"] == "build_dataset"
    assert result["error"] == "pipeline_service_unavailable"


def test_trigger_pipeline_internal_error(component: PipelineIntegrationComponent) -> None:
    """Test triggering a pipeline with internal error."""
    mock_service = Mock()
    mock_service.trigger_pipeline.side_effect = RuntimeError("Internal error")

    with (
        patch.object(component, "_get_pipeline_service", return_value=mock_service),
        patch("ml.dashboard.common.pipeline_integration.asyncio.run", side_effect=RuntimeError("Internal error")),
    ):
        result = component.trigger_pipeline("build_dataset", {"symbols": "AAPL"})

    assert result["success"] is False
    assert result["status"] == "ERROR"
    assert result["error"] == "internal_error"


# trigger_orchestrator_task tests


def test_trigger_orchestrator_task_backfill(component: PipelineIntegrationComponent) -> None:
    """Test triggering orchestrator backfill task."""
    with (
        patch("ml.core.integration.MLIntegrationManager"),
        patch("ml.orchestration.pipeline_orchestrator.MLPipelineOrchestrator"),
        patch("ml.stores.providers.SqlCoverageProvider"),
        patch("ml.stores.writers.DataStoreMarketDataWriter"),
    ):
        result = component.trigger_orchestrator_task("backfill", {"symbols": "AAPL"})

    assert result["ok"] is True
    assert result["result"]["status"] == "started"
    assert result["result"]["task"] == "backfill"


def test_trigger_orchestrator_task_build_dataset(component: PipelineIntegrationComponent) -> None:
    """Test triggering orchestrator build_dataset task."""
    with (
        patch("ml.core.integration.MLIntegrationManager"),
        patch("ml.orchestration.pipeline_orchestrator.MLPipelineOrchestrator"),
        patch("ml.stores.providers.SqlCoverageProvider"),
        patch("ml.stores.writers.DataStoreMarketDataWriter"),
    ):
        result = component.trigger_orchestrator_task("build_dataset", {"symbols": "AAPL"})

    assert result["ok"] is True
    assert result["result"]["status"] == "started"
    assert result["result"]["task"] == "build_dataset"


def test_trigger_orchestrator_task_unknown(component: PipelineIntegrationComponent) -> None:
    """Test triggering orchestrator with unknown task."""
    with (
        patch("ml.core.integration.MLIntegrationManager"),
        patch("ml.orchestration.pipeline_orchestrator.MLPipelineOrchestrator"),
        patch("ml.stores.providers.SqlCoverageProvider"),
        patch("ml.stores.writers.DataStoreMarketDataWriter"),
    ):
        result = component.trigger_orchestrator_task("unknown_task", {})

    assert result["ok"] is False
    assert "error" in result["result"]
    assert "Unknown task" in result["result"]["error"]


def test_trigger_orchestrator_task_error(component: PipelineIntegrationComponent) -> None:
    """Test triggering orchestrator task with initialization error."""
    with patch(
        "ml.core.integration.MLIntegrationManager",
        side_effect=RuntimeError("Init failed"),
    ):
        result = component.trigger_orchestrator_task("backfill", {})

    assert result["ok"] is False
    assert "error" in result["result"]


# list_pipeline_jobs tests


def test_list_pipeline_jobs_success(component: PipelineIntegrationComponent) -> None:
    """Test listing pipeline jobs successfully."""
    mock_service = Mock()
    mock_jobs = [
        Mock(
            job_id="job1",
            pipeline_type="build_dataset",
            status="COMPLETED",
            progress=1.0,
            current_stage="completed",
            eta_seconds=0,
            message="Done",
            error=None,
            started_at=1000.0,
            finished_at=1100.0,
            started_at_iso="2024-01-01T00:00:00Z",
            finished_at_iso="2024-01-01T00:01:40Z",
        ),
        Mock(
            job_id="job2",
            pipeline_type="train_model",
            status="RUNNING",
            progress=0.5,
            current_stage="training",
            eta_seconds=120,
            message="Training model",
            error=None,
            started_at=1200.0,
            finished_at=None,
            started_at_iso="2024-01-01T00:02:00Z",
            finished_at_iso=None,
        ),
    ]

    with (
        patch.object(component, "_get_pipeline_service", return_value=mock_service),
        patch("ml.dashboard.common.pipeline_integration.asyncio.run", return_value=mock_jobs),
    ):
        result = component.list_pipeline_jobs()

    assert result["status"] == EventStatus.SUCCESS.value
    assert len(result["jobs"]) == 2
    assert result["jobs"][0]["job_id"] == "job1"
    assert result["jobs"][1]["job_id"] == "job2"


def test_list_pipeline_jobs_empty(component: PipelineIntegrationComponent) -> None:
    """Test listing pipeline jobs when empty."""
    mock_service = Mock()

    with (
        patch.object(component, "_get_pipeline_service", return_value=mock_service),
        patch("ml.dashboard.common.pipeline_integration.asyncio.run", return_value=[]),
    ):
        result = component.list_pipeline_jobs()

    assert result["status"] == EventStatus.SUCCESS.value
    assert result["jobs"] == []


def test_list_pipeline_jobs_unavailable(component: PipelineIntegrationComponent) -> None:
    """Test listing pipeline jobs when service is unavailable."""
    with patch.object(component, "_get_pipeline_service", return_value=None):
        result = component.list_pipeline_jobs()

    assert result["status"] == "unavailable"
    assert result["jobs"] == []
    assert result["error"] == "pipeline_service_unavailable"


def test_list_pipeline_jobs_error(component: PipelineIntegrationComponent) -> None:
    """Test listing pipeline jobs with internal error."""
    mock_service = Mock()

    with (
        patch.object(component, "_get_pipeline_service", return_value=mock_service),
        patch("ml.dashboard.common.pipeline_integration.asyncio.run", side_effect=RuntimeError("Internal error")),
    ):
        result = component.list_pipeline_jobs()

    assert result["status"] == "error"
    assert result["jobs"] == []
    assert result["error"] == "internal_error"


# get_pipeline_job tests


def test_get_pipeline_job_success(component: PipelineIntegrationComponent) -> None:
    """Test getting a pipeline job successfully."""
    mock_service = Mock()
    mock_progress = Mock(
        job_id="job1",
        status="RUNNING",
        progress=0.5,
        current_stage="training",
        eta_seconds=120,
        message="Training model",
        error=None,
        started_at=1000.0,
        finished_at=None,
        started_at_iso="2024-01-01T00:00:00Z",
        finished_at_iso=None,
    )

    with (
        patch.object(component, "_get_pipeline_service", return_value=mock_service),
        patch("ml.dashboard.common.pipeline_integration.asyncio.run", return_value=mock_progress),
    ):
        result = component.get_pipeline_job("job1")

    assert result["status"] == EventStatus.SUCCESS.value
    assert result["job"]["job_id"] == "job1"
    assert result["job"]["status"] == "RUNNING"
    assert result["job"]["progress"] == 0.5


def test_get_pipeline_job_not_found(component: PipelineIntegrationComponent) -> None:
    """Test getting a pipeline job that doesn't exist."""
    mock_service = Mock()
    mock_progress = Mock(status="UNKNOWN")

    with (
        patch.object(component, "_get_pipeline_service", return_value=mock_service),
        patch("ml.dashboard.common.pipeline_integration.asyncio.run", return_value=mock_progress),
    ):
        result = component.get_pipeline_job("nonexistent")

    assert result["status"] == "not_found"
    assert result["error"] == "job_not_found"


def test_get_pipeline_job_unavailable(component: PipelineIntegrationComponent) -> None:
    """Test getting a pipeline job when service is unavailable."""
    with patch.object(component, "_get_pipeline_service", return_value=None):
        result = component.get_pipeline_job("job1")

    assert result["status"] == "unavailable"
    assert result["error"] == "pipeline_service_unavailable"


def test_get_pipeline_job_error(component: PipelineIntegrationComponent) -> None:
    """Test getting a pipeline job with internal error."""
    mock_service = Mock()

    with (
        patch.object(component, "_get_pipeline_service", return_value=mock_service),
        patch("ml.dashboard.common.pipeline_integration.asyncio.run", side_effect=RuntimeError("Internal error")),
    ):
        result = component.get_pipeline_job("job1")

    assert result["status"] == "error"
    assert result["error"] == "internal_error"


# purge_pipeline_job tests


def test_purge_pipeline_job_success(component: PipelineIntegrationComponent) -> None:
    """Test purging a pipeline job successfully."""
    mock_service = Mock()
    mock_result = Mock(
        success=True,
        job_id="job1",
        status="PURGED",
        message="Job purged",
        error=None,
    )

    with (
        patch.object(component, "_get_pipeline_service", return_value=mock_service),
        patch("ml.dashboard.common.pipeline_integration.asyncio.run", return_value=mock_result),
    ):
        result = component.purge_pipeline_job("job1")

    assert result["status"] == "purged"
    assert result["result"]["success"] is True
    assert result["result"]["job_id"] == "job1"


def test_purge_pipeline_job_not_found(component: PipelineIntegrationComponent) -> None:
    """Test purging a pipeline job that doesn't exist."""
    mock_service = Mock()
    mock_result = Mock(
        success=False,
        job_id="nonexistent",
        status="NOT_FOUND",
        message="Job not found",
        error=None,
    )

    with (
        patch.object(component, "_get_pipeline_service", return_value=mock_service),
        patch("ml.dashboard.common.pipeline_integration.asyncio.run", return_value=mock_result),
    ):
        result = component.purge_pipeline_job("nonexistent")

    assert result["status"] == "not_found"
    assert result["result"]["success"] is False


def test_purge_pipeline_job_failed(component: PipelineIntegrationComponent) -> None:
    """Test purging a pipeline job that fails."""
    mock_service = Mock()
    mock_result = Mock(
        success=False,
        job_id="job1",
        status="FAILED",
        message="Purge failed",
        error="store_error",
    )

    with (
        patch.object(component, "_get_pipeline_service", return_value=mock_service),
        patch("ml.dashboard.common.pipeline_integration.asyncio.run", return_value=mock_result),
    ):
        result = component.purge_pipeline_job("job1")

    assert result["status"] == "failed"
    assert result["result"]["success"] is False


def test_purge_pipeline_job_unavailable(component: PipelineIntegrationComponent) -> None:
    """Test purging a pipeline job when service is unavailable."""
    with patch.object(component, "_get_pipeline_service", return_value=None):
        result = component.purge_pipeline_job("job1")

    assert result["status"] == "unavailable"
    assert result["error"] == "pipeline_service_unavailable"


# build_dataset_pipeline tests


def test_build_dataset_pipeline_success(component: PipelineIntegrationComponent) -> None:
    """Test triggering build_dataset pipeline."""
    with patch.object(component, "trigger_pipeline", return_value={"success": True, "job_id": "job1"}):
        result = component.build_dataset_pipeline({"symbols": "AAPL"})

    assert result["success"] is True
    assert result["job_id"] == "job1"


def test_build_dataset_pipeline_failure(component: PipelineIntegrationComponent) -> None:
    """Test triggering build_dataset pipeline with failure."""
    with patch.object(component, "trigger_pipeline", return_value={"success": False, "error": "config_error"}):
        result = component.build_dataset_pipeline({"invalid": "config"})

    assert result["success"] is False
    assert result["error"] == "config_error"


# train_model_pipeline tests


def test_train_model_pipeline_success(component: PipelineIntegrationComponent) -> None:
    """Test triggering train_model pipeline."""
    with patch.object(component, "trigger_pipeline", return_value={"success": True, "job_id": "job2"}):
        result = component.train_model_pipeline({"model_type": "xgboost"})

    assert result["success"] is True
    assert result["job_id"] == "job2"


def test_train_model_pipeline_failure(component: PipelineIntegrationComponent) -> None:
    """Test triggering train_model pipeline with failure."""
    with patch.object(component, "trigger_pipeline", return_value={"success": False, "error": "unavailable"}):
        result = component.train_model_pipeline({"model_type": "invalid"})

    assert result["success"] is False
    assert result["error"] == "unavailable"


# run_hpo_pipeline tests


def test_run_hpo_pipeline_success(component: PipelineIntegrationComponent) -> None:
    """Test triggering run_hpo pipeline."""
    with patch.object(component, "trigger_pipeline", return_value={"success": True, "job_id": "job3"}):
        result = component.run_hpo_pipeline({"trials": 100})

    assert result["success"] is True
    assert result["job_id"] == "job3"


def test_run_hpo_pipeline_failure(component: PipelineIntegrationComponent) -> None:
    """Test triggering run_hpo pipeline with failure."""
    with patch.object(component, "trigger_pipeline", return_value={"success": False, "error": "invalid_config"}):
        result = component.run_hpo_pipeline({"trials": -1})

    assert result["success"] is False
    assert result["error"] == "invalid_config"


# get_pipeline_progress tests


def test_get_pipeline_progress_success(component: PipelineIntegrationComponent) -> None:
    """Test getting pipeline progress successfully."""
    mock_service = Mock()
    mock_progress = Mock(
        job_id="job1",
        status="RUNNING",
        progress=0.75,
        current_stage="validation",
        eta_seconds=60,
        message="Validating model",
        error=None,
        started_at=1000.0,
        finished_at=None,
        started_at_iso="2024-01-01T00:00:00Z",
        finished_at_iso=None,
    )

    with (
        patch.object(component, "_get_pipeline_service", return_value=mock_service),
        patch("ml.dashboard.common.pipeline_integration.asyncio.run", return_value=mock_progress),
    ):
        result = component.get_pipeline_progress("job1")

    assert result["status"] == "success"
    assert result["progress"]["job_id"] == "job1"
    assert result["progress"]["progress"] == 0.75
    assert result["progress"]["current_stage"] == "validation"


def test_get_pipeline_progress_not_found(component: PipelineIntegrationComponent) -> None:
    """Test getting progress for non-existent job."""
    mock_service = Mock()
    mock_progress = Mock(status="UNKNOWN")

    with (
        patch.object(component, "_get_pipeline_service", return_value=mock_service),
        patch("ml.dashboard.common.pipeline_integration.asyncio.run", return_value=mock_progress),
    ):
        result = component.get_pipeline_progress("nonexistent")

    assert result["status"] == "not_found"
    assert result["error"] == "job_not_found"


def test_get_pipeline_progress_unavailable(component: PipelineIntegrationComponent) -> None:
    """Test getting progress when service is unavailable."""
    with patch.object(component, "_get_pipeline_service", return_value=None):
        result = component.get_pipeline_progress("job1")

    assert result["status"] == "unavailable"
    assert result["error"] == "pipeline_service_unavailable"


def test_get_pipeline_progress_error(component: PipelineIntegrationComponent) -> None:
    """Test getting progress with internal error."""
    mock_service = Mock()

    with (
        patch.object(component, "_get_pipeline_service", return_value=mock_service),
        patch("ml.dashboard.common.pipeline_integration.asyncio.run", side_effect=RuntimeError("Internal error")),
    ):
        result = component.get_pipeline_progress("job1")

    assert result["status"] == "error"
    assert result["error"] == "internal_error"


# cancel_pipeline_job tests


def test_cancel_pipeline_job_success(component: PipelineIntegrationComponent) -> None:
    """Test cancelling a pipeline job successfully."""
    mock_service = Mock()
    mock_result = Mock(
        success=True,
        job_id="job1",
        status="CANCELLED",
        message="Cancellation acknowledged",
        error=None,
    )

    with (
        patch.object(component, "_get_pipeline_service", return_value=mock_service),
        patch("ml.dashboard.common.pipeline_integration.asyncio.run", return_value=mock_result),
    ):
        result = component.cancel_pipeline_job("job1")

    assert result["success"] is True
    assert result["job_id"] == "job1"
    assert result["status"] == "CANCELLED"


def test_cancel_pipeline_job_not_found(component: PipelineIntegrationComponent) -> None:
    """Test cancelling a non-existent job."""
    mock_service = Mock()
    mock_result = Mock(
        success=False,
        job_id="nonexistent",
        status="NOT_FOUND",
        message="Job not found",
        error="not_found",
    )

    with (
        patch.object(component, "_get_pipeline_service", return_value=mock_service),
        patch("ml.dashboard.common.pipeline_integration.asyncio.run", return_value=mock_result),
    ):
        result = component.cancel_pipeline_job("nonexistent")

    assert result["success"] is False
    assert result["status"] == "NOT_FOUND"


def test_cancel_pipeline_job_already_finished(component: PipelineIntegrationComponent) -> None:
    """Test cancelling a job that's already finished."""
    mock_service = Mock()
    mock_result = Mock(
        success=True,
        job_id="job1",
        status="COMPLETED",
        message="Job already finished",
        error=None,
    )

    with (
        patch.object(component, "_get_pipeline_service", return_value=mock_service),
        patch("ml.dashboard.common.pipeline_integration.asyncio.run", return_value=mock_result),
    ):
        result = component.cancel_pipeline_job("job1")

    assert result["success"] is True
    assert result["status"] == "COMPLETED"


def test_cancel_pipeline_job_unavailable(component: PipelineIntegrationComponent) -> None:
    """Test cancelling a job when service is unavailable."""
    with patch.object(component, "_get_pipeline_service", return_value=None):
        result = component.cancel_pipeline_job("job1")

    assert result["success"] is False
    assert result["status"] == "unavailable"
    assert result["error"] == "pipeline_service_unavailable"


def test_cancel_pipeline_job_error(component: PipelineIntegrationComponent) -> None:
    """Test cancelling a job with internal error."""
    mock_service = Mock()

    with (
        patch.object(component, "_get_pipeline_service", return_value=mock_service),
        patch("ml.dashboard.common.pipeline_integration.asyncio.run", side_effect=RuntimeError("Internal error")),
    ):
        result = component.cancel_pipeline_job("job1")

    assert result["success"] is False
    assert result["status"] == "error"
    assert result["error"] == "internal_error"


# get_integration_manager tests


def test_get_integration_manager_cached(component: PipelineIntegrationComponent) -> None:
    """Test getting cached integration manager."""
    mock_manager = Mock()
    component._pipeline_integration_manager = mock_manager

    result = component.get_integration_manager()

    assert result is mock_manager


def test_get_integration_manager_lazy_init(component: PipelineIntegrationComponent) -> None:
    """Test lazy initialization of integration manager."""
    mock_service = Mock()
    mock_manager = Mock()

    with (
        patch("ml.core.integration.MLIntegrationManager", return_value=mock_manager),
        patch("ml.dashboard.services.PipelineIntegrationService", return_value=mock_service),
    ):
        result = component.get_integration_manager()

    assert result is mock_manager
    assert component._pipeline_integration_manager is mock_manager


def test_get_integration_manager_none(component: PipelineIntegrationComponent) -> None:
    """Test getting integration manager when initialization fails."""
    with patch("ml.core.integration.MLIntegrationManager", side_effect=RuntimeError("Init failed")):
        result = component.get_integration_manager()

    assert result is None


# Helper method tests


def test_serialize_job_state() -> None:
    """Test serializing job state to dictionary."""
    mock_job = Mock(
        job_id="job1",
        pipeline_type="build_dataset",
        status="RUNNING",
        progress=0.5,
        current_stage="ingestion",
        eta_seconds=120,
        message="Ingesting data",
        error=None,
        started_at=1000.0,
        finished_at=None,
        started_at_iso="2024-01-01T00:00:00Z",
        finished_at_iso=None,
    )

    result = PipelineIntegrationComponent._serialize_job_state(mock_job)

    assert result["job_id"] == "job1"
    assert result["pipeline_type"] == "build_dataset"
    assert result["status"] == "RUNNING"
    assert result["progress"] == 0.5
    assert result["current_stage"] == "ingestion"
    assert result["eta_seconds"] == 120
    assert result["message"] == "Ingesting data"
    assert result["error"] is None
    assert result["started_at"] == 1000.0
    assert result["finished_at"] is None


def test_serialize_pipeline_progress() -> None:
    """Test serializing pipeline progress to dictionary."""
    mock_progress = Mock(
        job_id="job2",
        status="COMPLETED",
        progress=1.0,
        current_stage="completed",
        eta_seconds=0,
        message="Pipeline completed",
        error=None,
        started_at=1000.0,
        finished_at=1200.0,
        started_at_iso="2024-01-01T00:00:00Z",
        finished_at_iso="2024-01-01T00:03:20Z",
    )

    result = PipelineIntegrationComponent._serialize_pipeline_progress(mock_progress)

    assert result["job_id"] == "job2"
    assert result["status"] == "COMPLETED"
    assert result["progress"] == 1.0
    assert result["current_stage"] == "completed"
    assert result["eta_seconds"] == 0
    assert result["message"] == "Pipeline completed"
    assert result["error"] is None
    assert result["started_at"] == 1000.0
    assert result["finished_at"] == 1200.0


def test_run_pipeline() -> None:
    """Test running pipeline coroutine synchronously."""
    async def mock_coroutine() -> str:
        return "result"

    result = PipelineIntegrationComponent._run_pipeline(mock_coroutine())

    assert result == "result"
