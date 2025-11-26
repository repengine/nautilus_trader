"""Tests for Pipeline Blueprint.

This module tests the pipeline blueprint routes:
- POST /api/pipeline/run - Trigger pipeline execution
- GET /api/pipeline/jobs - List pipeline jobs
- GET /api/pipeline/jobs/<job_id> - Get pipeline job details
- DELETE /api/pipeline/jobs/<job_id> - Purge pipeline job
- POST /api/pipeline/build-dataset - Build training dataset
- POST /api/pipeline/train-model - Train ML model
- POST /api/pipeline/run-hpo - Run hyperparameter optimization
- GET /api/pipeline/jobs/<job_id>/progress - Get job progress
- POST /api/pipeline/jobs/<job_id>/cancel - Cancel pipeline job

Tests (18):
1. test_pipeline_run_queued_202
2. test_pipeline_run_unavailable_503
3. test_pipeline_run_invalid_400
4. test_pipeline_run_unauthorized_401
5. test_pipeline_jobs_list_success
6. test_pipeline_job_detail_success
7. test_pipeline_job_detail_not_found
8. test_pipeline_job_purge_success
9. test_build_dataset_queued_202
10. test_train_model_queued_202
11. test_run_hpo_queued_202
12. test_pipeline_progress_success
13. test_pipeline_cancel_success
14. test_pipeline_cancel_not_found
15. test_pipeline_run_delegates_to_service
16. test_pipeline_run_accepts_legacy_mode
17. test_pipeline_jobs_list_delegates_to_service
18. test_pipeline_cancel_unavailable_503
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from flask import Flask
from flask.testing import FlaskClient

from flask import Blueprint

from ml.dashboard.blueprints.pipeline import register_pipeline_routes


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_service() -> MagicMock:
    """Provide a mock DashboardService with pipeline methods."""
    svc = MagicMock()

    # Default trigger_pipeline response (success/queued)
    svc.trigger_pipeline.return_value = {
        "success": True,
        "job_id": "job_123",
        "pipeline_type": "full",
        "status": "QUEUED",
        "message": "Pipeline queued successfully",
    }

    # Default list_pipeline_jobs response
    svc.list_pipeline_jobs.return_value = {
        "status": "success",
        "jobs": [
            {"job_id": "job_001", "status": "completed", "pipeline_type": "ingest"},
            {"job_id": "job_002", "status": "running", "pipeline_type": "train"},
        ],
    }

    # Default get_pipeline_job response
    svc.get_pipeline_job.return_value = {
        "status": "success",
        "job_id": "job_123",
        "pipeline_type": "full",
        "state": "running",
        "progress": 50,
    }

    # Default purge_pipeline_job response
    svc.purge_pipeline_job.return_value = {
        "status": "purged",
        "job_id": "job_123",
        "message": "Job purged successfully",
    }

    # Default build_dataset_pipeline response
    svc.build_dataset_pipeline.return_value = {
        "success": True,
        "job_id": "dataset_job_001",
        "status": "QUEUED",
        "message": "Dataset build queued",
    }

    # Default train_model_pipeline response
    svc.train_model_pipeline.return_value = {
        "success": True,
        "job_id": "train_job_001",
        "status": "QUEUED",
        "message": "Model training queued",
    }

    # Default run_hpo_pipeline response
    svc.run_hpo_pipeline.return_value = {
        "success": True,
        "job_id": "hpo_job_001",
        "status": "QUEUED",
        "message": "HPO queued",
    }

    # Default get_pipeline_progress response
    svc.get_pipeline_progress.return_value = {
        "status": "success",
        "job_id": "job_123",
        "progress": 75,
        "current_step": "training",
        "total_steps": 4,
    }

    # Default cancel_pipeline_job response
    svc.cancel_pipeline_job.return_value = {
        "success": True,
        "status": "success",
        "job_id": "job_123",
        "message": "Job cancelled",
    }

    return svc


@pytest.fixture
def app(mock_service: MagicMock) -> Flask:
    """Provide Flask test application with pipeline blueprint."""
    app = Flask(__name__)
    app.config["TESTING"] = True

    # Track auth state for require_token mock
    app.config["AUTH_ENABLED"] = True

    def require_token() -> bool:
        return app.config.get("AUTH_ENABLED", True)

    # Create a FRESH blueprint for each test to avoid Flask's
    # "already registered" error
    bp = Blueprint("pipeline", __name__, url_prefix="/api/pipeline")

    # Register routes with mock service
    register_pipeline_routes(bp, mock_service, require_token)
    app.register_blueprint(bp)

    return app


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    """Provide Flask test client."""
    return app.test_client()


# ============================================================================
# TEST 1: test_pipeline_run_queued_202
# ============================================================================


class TestPipelineRunQueued202:
    """Test POST /api/pipeline/run returns 202 when queued."""

    def test_pipeline_run_queued_202(
        self,
        client: FlaskClient,
        mock_service: MagicMock,
    ) -> None:
        """Test that successful pipeline trigger returns HTTP 202."""
        response = client.post(
            "/api/pipeline/run",
            json={"pipeline_type": "full", "config": {"param": "value"}},
        )

        assert response.status_code == 202
        data = response.get_json()
        assert data is not None
        assert data["success"] is True
        assert data["status"] == "QUEUED"
        assert data["job_id"] == "job_123"


# ============================================================================
# TEST 2: test_pipeline_run_unavailable_503
# ============================================================================


class TestPipelineRunUnavailable503:
    """Test POST /api/pipeline/run returns 503 when service unavailable."""

    def test_pipeline_run_unavailable_503(
        self,
        client: FlaskClient,
        mock_service: MagicMock,
    ) -> None:
        """Test that unavailable pipeline service returns HTTP 503."""
        mock_service.trigger_pipeline.return_value = {
            "success": False,
            "status": "UNAVAILABLE",
            "message": "Pipeline service not available",
        }

        response = client.post(
            "/api/pipeline/run",
            json={"pipeline_type": "full"},
        )

        assert response.status_code == 503
        data = response.get_json()
        assert data["status"] == "UNAVAILABLE"


# ============================================================================
# TEST 3: test_pipeline_run_invalid_400
# ============================================================================


class TestPipelineRunInvalid400:
    """Test POST /api/pipeline/run returns 400 for invalid request."""

    def test_pipeline_run_invalid_400(
        self,
        client: FlaskClient,
        mock_service: MagicMock,
    ) -> None:
        """Test that invalid pipeline request returns HTTP 400."""
        mock_service.trigger_pipeline.return_value = {
            "success": False,
            "status": "INVALID",
            "message": "Invalid pipeline configuration",
        }

        response = client.post(
            "/api/pipeline/run",
            json={"pipeline_type": "unknown"},
        )

        assert response.status_code == 400
        data = response.get_json()
        assert data["status"] == "INVALID"


# ============================================================================
# TEST 4: test_pipeline_run_unauthorized_401
# ============================================================================


class TestPipelineRunUnauthorized401:
    """Test POST /api/pipeline/run returns 401 when unauthorized."""

    def test_pipeline_run_unauthorized_401(
        self,
        app: Flask,
        client: FlaskClient,
    ) -> None:
        """Test that unauthorized request returns HTTP 401."""
        app.config["AUTH_ENABLED"] = False

        response = client.post(
            "/api/pipeline/run",
            json={"pipeline_type": "full"},
        )

        assert response.status_code == 401
        data = response.get_json()
        assert data["error"] == "unauthorized"


# ============================================================================
# TEST 5: test_pipeline_jobs_list_success
# ============================================================================


class TestPipelineJobsListSuccess:
    """Test GET /api/pipeline/jobs returns job list."""

    def test_pipeline_jobs_list_success(
        self,
        client: FlaskClient,
        mock_service: MagicMock,
    ) -> None:
        """Test that jobs list endpoint returns HTTP 200 with jobs."""
        response = client.get("/api/pipeline/jobs")

        assert response.status_code == 200
        data = response.get_json()
        assert data is not None
        assert data["status"] == "success"
        assert len(data["jobs"]) == 2


# ============================================================================
# TEST 6: test_pipeline_job_detail_success
# ============================================================================


class TestPipelineJobDetailSuccess:
    """Test GET /api/pipeline/jobs/<job_id> returns job details."""

    def test_pipeline_job_detail_success(
        self,
        client: FlaskClient,
        mock_service: MagicMock,
    ) -> None:
        """Test that job detail endpoint returns HTTP 200 with details."""
        response = client.get("/api/pipeline/jobs/job_123")

        assert response.status_code == 200
        data = response.get_json()
        assert data is not None
        assert data["status"] == "success"
        assert data["job_id"] == "job_123"
        mock_service.get_pipeline_job.assert_called_once_with("job_123")


# ============================================================================
# TEST 7: test_pipeline_job_detail_not_found
# ============================================================================


class TestPipelineJobDetailNotFound:
    """Test GET /api/pipeline/jobs/<job_id> returns 404 for unknown job."""

    def test_pipeline_job_detail_not_found(
        self,
        client: FlaskClient,
        mock_service: MagicMock,
    ) -> None:
        """Test that unknown job ID returns HTTP 404."""
        mock_service.get_pipeline_job.return_value = {
            "status": "not_found",
            "message": "Job not found",
        }

        response = client.get("/api/pipeline/jobs/nonexistent")

        assert response.status_code == 404
        data = response.get_json()
        assert data["status"] == "not_found"


# ============================================================================
# TEST 8: test_pipeline_job_purge_success
# ============================================================================


class TestPipelineJobPurgeSuccess:
    """Test DELETE /api/pipeline/jobs/<job_id> purges job."""

    def test_pipeline_job_purge_success(
        self,
        client: FlaskClient,
        mock_service: MagicMock,
    ) -> None:
        """Test that job purge returns HTTP 200 with purged status."""
        response = client.delete("/api/pipeline/jobs/job_123")

        assert response.status_code == 200
        data = response.get_json()
        assert data is not None
        assert data["status"] == "purged"
        mock_service.purge_pipeline_job.assert_called_once_with("job_123")


# ============================================================================
# TEST 9: test_build_dataset_queued_202
# ============================================================================


class TestBuildDatasetQueued202:
    """Test POST /api/pipeline/build-dataset returns 202 when queued."""

    def test_build_dataset_queued_202(
        self,
        client: FlaskClient,
        mock_service: MagicMock,
    ) -> None:
        """Test that dataset build returns HTTP 202 when queued."""
        response = client.post(
            "/api/pipeline/build-dataset",
            json={"symbols": ["SPY", "QQQ"], "start_date": "2024-01-01"},
        )

        assert response.status_code == 202
        data = response.get_json()
        assert data is not None
        assert data["success"] is True
        assert data["status"] == "QUEUED"
        assert data["job_id"] == "dataset_job_001"


# ============================================================================
# TEST 10: test_train_model_queued_202
# ============================================================================


class TestTrainModelQueued202:
    """Test POST /api/pipeline/train-model returns 202 when queued."""

    def test_train_model_queued_202(
        self,
        client: FlaskClient,
        mock_service: MagicMock,
    ) -> None:
        """Test that model training returns HTTP 202 when queued."""
        response = client.post(
            "/api/pipeline/train-model",
            json={"model_type": "xgboost", "dataset_id": "ds_001"},
        )

        assert response.status_code == 202
        data = response.get_json()
        assert data is not None
        assert data["success"] is True
        assert data["status"] == "QUEUED"
        assert data["job_id"] == "train_job_001"


# ============================================================================
# TEST 11: test_run_hpo_queued_202
# ============================================================================


class TestRunHpoQueued202:
    """Test POST /api/pipeline/run-hpo returns 202 when queued."""

    def test_run_hpo_queued_202(
        self,
        client: FlaskClient,
        mock_service: MagicMock,
    ) -> None:
        """Test that HPO run returns HTTP 202 when queued."""
        response = client.post(
            "/api/pipeline/run-hpo",
            json={"model_type": "xgboost", "n_trials": 100},
        )

        assert response.status_code == 202
        data = response.get_json()
        assert data is not None
        assert data["success"] is True
        assert data["status"] == "QUEUED"
        assert data["job_id"] == "hpo_job_001"


# ============================================================================
# TEST 12: test_pipeline_progress_success
# ============================================================================


class TestPipelineProgressSuccess:
    """Test GET /api/pipeline/jobs/<job_id>/progress returns progress."""

    def test_pipeline_progress_success(
        self,
        client: FlaskClient,
        mock_service: MagicMock,
    ) -> None:
        """Test that progress endpoint returns HTTP 200 with progress data."""
        response = client.get("/api/pipeline/jobs/job_123/progress")

        assert response.status_code == 200
        data = response.get_json()
        assert data is not None
        assert data["status"] == "success"
        assert data["progress"] == 75
        assert data["current_step"] == "training"
        mock_service.get_pipeline_progress.assert_called_once_with("job_123")


# ============================================================================
# TEST 13: test_pipeline_cancel_success
# ============================================================================


class TestPipelineCancelSuccess:
    """Test POST /api/pipeline/jobs/<job_id>/cancel cancels job."""

    def test_pipeline_cancel_success(
        self,
        client: FlaskClient,
        mock_service: MagicMock,
    ) -> None:
        """Test that cancel endpoint returns HTTP 200 on success."""
        response = client.post("/api/pipeline/jobs/job_123/cancel")

        assert response.status_code == 200
        data = response.get_json()
        assert data is not None
        assert data["success"] is True
        mock_service.cancel_pipeline_job.assert_called_once_with("job_123")


# ============================================================================
# TEST 14: test_pipeline_cancel_not_found
# ============================================================================


class TestPipelineCancelNotFound:
    """Test POST /api/pipeline/jobs/<job_id>/cancel returns 404 for unknown job."""

    def test_pipeline_cancel_not_found(
        self,
        client: FlaskClient,
        mock_service: MagicMock,
    ) -> None:
        """Test that cancel of unknown job returns HTTP 404."""
        mock_service.cancel_pipeline_job.return_value = {
            "success": False,
            "status": "NOT_FOUND",
            "message": "Job not found",
        }

        response = client.post("/api/pipeline/jobs/nonexistent/cancel")

        assert response.status_code == 404
        data = response.get_json()
        assert data["status"] == "NOT_FOUND"


# ============================================================================
# TEST 15: test_pipeline_run_delegates_to_service
# ============================================================================


class TestPipelineRunDelegatesToService:
    """Test that pipeline_run properly delegates to service."""

    def test_pipeline_run_delegates_to_service(
        self,
        client: FlaskClient,
        mock_service: MagicMock,
    ) -> None:
        """Test that pipeline_run passes correct args to service."""
        config = {"dataset": {"symbols": ["AAPL"]}, "output_dir": "/tmp"}

        response = client.post(
            "/api/pipeline/run",
            json={"pipeline_type": "ingest", "config": config},
        )

        assert response.status_code == 202
        mock_service.trigger_pipeline.assert_called_once()
        call_args = mock_service.trigger_pipeline.call_args
        assert call_args[0][0] == "ingest"
        assert call_args[0][1] == config


# ============================================================================
# TEST 16: test_pipeline_run_accepts_legacy_mode
# ============================================================================


class TestPipelineRunAcceptsLegacyMode:
    """Test that pipeline_run accepts legacy 'mode' parameter."""

    def test_pipeline_run_accepts_legacy_mode(
        self,
        client: FlaskClient,
        mock_service: MagicMock,
    ) -> None:
        """Test that 'mode' is accepted as alias for 'pipeline_type'."""
        response = client.post(
            "/api/pipeline/run",
            json={"mode": "dataset", "data_dir": "data/tier1"},
        )

        assert response.status_code == 202
        mock_service.trigger_pipeline.assert_called_once()
        call_args = mock_service.trigger_pipeline.call_args
        assert call_args[0][0] == "dataset"
        # Config should include non-mode fields
        assert call_args[0][1] == {"data_dir": "data/tier1"}


# ============================================================================
# TEST 17: test_pipeline_jobs_list_delegates_to_service
# ============================================================================


class TestPipelineJobsListDelegatesToService:
    """Test that pipeline_jobs_list delegates to service."""

    def test_pipeline_jobs_list_delegates_to_service(
        self,
        client: FlaskClient,
        mock_service: MagicMock,
    ) -> None:
        """Test that jobs list calls service method."""
        response = client.get("/api/pipeline/jobs")

        assert response.status_code == 200
        mock_service.list_pipeline_jobs.assert_called_once()


# ============================================================================
# TEST 18: test_pipeline_cancel_unavailable_503
# ============================================================================


class TestPipelineCancelUnavailable503:
    """Test POST /api/pipeline/jobs/<job_id>/cancel returns 503 when unavailable."""

    def test_pipeline_cancel_unavailable_503(
        self,
        client: FlaskClient,
        mock_service: MagicMock,
    ) -> None:
        """Test that cancel returns HTTP 503 when service unavailable."""
        mock_service.cancel_pipeline_job.return_value = {
            "success": False,
            "status": "UNAVAILABLE",
            "message": "Pipeline service not available",
        }

        response = client.post("/api/pipeline/jobs/job_123/cancel")

        assert response.status_code == 503
        data = response.get_json()
        assert data["status"] == "UNAVAILABLE"
