from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ml.dashboard.config import DashboardConfig
from ml.dashboard.controllers import NoopServiceController
from ml.dashboard.service import DashboardService
from ml.dashboard.services import PipelineJobState
from ml.dashboard.services import PipelineProgress

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)


@pytest.mark.contracts
def test_dashboard_service_uses_integration_manager() -> None:
    config = DashboardConfig(db_connection="postgresql://test")
    service = DashboardService(config=config, controller=NoopServiceController())
    manager = MagicMock()

    with patch("ml.core.integration.MLIntegrationManager", return_value=manager) as manager_cls:
        assert service.get_integration_manager() is manager
        assert service.get_integration_manager() is manager
        manager_cls.assert_called_once_with(
            db_connection=config.db_connection,
            auto_start_postgres=False,
            auto_migrate=False,
            ensure_healthy=False,
        )


@pytest.mark.contracts
def test_dashboard_service_serializes_pipeline_state_schema() -> None:
    job = PipelineJobState(
        job_id="job-1",
        pipeline_type="training",
        status="RUNNING",
        progress=0.5,
        current_stage="feature_ingestion",
        eta_seconds=120,
        message="working",
        error=None,
        started_at=1.0,
        finished_at=None,
        started_at_iso="2025-01-01T00:00:00+00:00",
        finished_at_iso=None,
    )
    progress = PipelineProgress(
        job_id="job-2",
        status="RUNNING",
        progress=0.4,
        current_stage="training",
        eta_seconds=90,
        message="training",
        error=None,
        started_at=2.0,
        finished_at=None,
        started_at_iso="2025-01-01T00:00:01+00:00",
        finished_at_iso=None,
    )

    job_payload = DashboardService._serialize_job_state(job)
    progress_payload = DashboardService._serialize_pipeline_progress(progress)

    expected_keys = {
        "job_id",
        "pipeline_type",
        "status",
        "progress",
        "current_stage",
        "eta_seconds",
        "message",
        "error",
        "started_at",
        "finished_at",
        "started_at_iso",
        "finished_at_iso",
    }
    progress_keys = expected_keys - {"pipeline_type"}

    assert set(job_payload.keys()) == expected_keys
    assert set(progress_payload.keys()) == progress_keys
    assert job_payload["job_id"] == "job-1"
    assert progress_payload["job_id"] == "job-2"
