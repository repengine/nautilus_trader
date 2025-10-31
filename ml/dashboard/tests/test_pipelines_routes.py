"""
Tests for Pipeline Orchestration API Routes.

Comprehensive test coverage for pipeline triggering (build-dataset, train-model, run-
hpo), progress tracking, and job cancellation.

"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import pytest
from flask import Flask
from flask.testing import FlaskClient

from ml.dashboard.app import create_app
from ml.dashboard.config import DashboardConfig


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(scope="module")
def app() -> Iterator[Flask]:
    """
    Provide Flask test application with shared service instance.
    """
    from ml.dashboard.config import DashboardToken

    config = DashboardConfig(
        auth_tokens=(DashboardToken(value="test-token-123"),),
        db_connection="postgresql://test:test@localhost:5432/test",
    )
    app = create_app(config)
    yield app


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    """
    Provide Flask test client.
    """
    return app.test_client()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """
    Provide authentication headers.
    """
    return {"X-ML-DASHBOARD-TOKEN": "test-token-123"}


# ============================================================================
# BUILD DATASET ENDPOINT TESTS
# ============================================================================


class TestPipelinesBuildDatasetEndpoint:
    """
    Test /api/pipeline/build-dataset endpoint.
    """

    def test_build_dataset_success(
        self,
        client: FlaskClient,
        auth_headers: dict[str, str],
    ) -> None:
        """
        Test successful dataset building pipeline trigger.
        """
        with patch("ml.dashboard.service.DashboardService.build_dataset_pipeline") as mock_build:
            mock_build.return_value = {
                "success": True,
                "job_id": "build_dataset_abc123",
                "pipeline_type": "build_dataset",
                "status": "QUEUED",
                "message": "Pipeline build_dataset_abc123 queued successfully",
                "error": None,
            }

            response = client.post(
                "/api/pipeline/build-dataset",
                json={
                    "symbols": "SPY,QQQ,IWM",
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                },
                headers=auth_headers,
            )

            assert response.status_code == 202
            data = response.get_json()
            assert data["success"] is True
            assert data["job_id"] == "build_dataset_abc123"
            assert data["status"] == "QUEUED"
            mock_build.assert_called_once()

    def test_build_dataset_unavailable(
        self,
        client: FlaskClient,
        auth_headers: dict[str, str],
    ) -> None:
        """
        Test dataset building when pipeline service unavailable.
        """
        with patch("ml.dashboard.service.DashboardService.build_dataset_pipeline") as mock_build:
            mock_build.return_value = {
                "success": False,
                "status": "UNAVAILABLE",
                "error": "pipeline_service_unavailable",
            }

            response = client.post(
                "/api/pipeline/build-dataset",
                json={"symbols": "SPY"},
                headers=auth_headers,
            )

            assert response.status_code == 503
            data = response.get_json()
            assert data["success"] is False
            assert data["status"] == "UNAVAILABLE"

    def test_build_dataset_invalid_config(
        self,
        client: FlaskClient,
        auth_headers: dict[str, str],
    ) -> None:
        """
        Test dataset building with invalid configuration.
        """
        with patch("ml.dashboard.service.DashboardService.build_dataset_pipeline") as mock_build:
            mock_build.return_value = {
                "success": False,
                "status": "INVALID",
                "error": "Missing required field: symbols",
            }

            response = client.post(
                "/api/pipeline/build-dataset",
                json={},
                headers=auth_headers,
            )

            assert response.status_code == 400
            data = response.get_json()
            assert data["success"] is False
            assert data["status"] == "INVALID"

    def test_build_dataset_unauthorized(self, client: FlaskClient) -> None:
        """
        Test dataset building without authentication.
        """
        response = client.post(
            "/api/pipeline/build-dataset",
            json={"symbols": "SPY"},
        )

        assert response.status_code == 401
        data = response.get_json()
        assert data["error"] == "unauthorized"


# ============================================================================
# TRAIN MODEL ENDPOINT TESTS
# ============================================================================


class TestPipelinesTrainModelEndpoint:
    """
    Test /api/pipeline/train-model endpoint.
    """

    def test_train_model_success(
        self,
        client: FlaskClient,
        auth_headers: dict[str, str],
    ) -> None:
        """
        Test successful model training pipeline trigger.
        """
        with patch("ml.dashboard.service.DashboardService.train_model_pipeline") as mock_train:
            mock_train.return_value = {
                "success": True,
                "job_id": "train_model_xyz789",
                "pipeline_type": "train_model",
                "status": "QUEUED",
                "message": "Pipeline train_model_xyz789 queued successfully",
                "error": None,
            }

            response = client.post(
                "/api/pipeline/train-model",
                json={
                    "model_type": "Teacher",
                    "algorithm": "Transformer",
                    "dataset_id": "spy_features_v1",
                },
                headers=auth_headers,
            )

            assert response.status_code == 202
            data = response.get_json()
            assert data["success"] is True
            assert data["job_id"] == "train_model_xyz789"
            assert data["status"] == "QUEUED"
            mock_train.assert_called_once()

    def test_train_model_unavailable(
        self,
        client: FlaskClient,
        auth_headers: dict[str, str],
    ) -> None:
        """
        Test model training when pipeline service unavailable.
        """
        with patch("ml.dashboard.service.DashboardService.train_model_pipeline") as mock_train:
            mock_train.return_value = {
                "success": False,
                "status": "UNAVAILABLE",
                "error": "pipeline_service_unavailable",
            }

            response = client.post(
                "/api/pipeline/train-model",
                json={"model_type": "Teacher"},
                headers=auth_headers,
            )

            assert response.status_code == 503
            data = response.get_json()
            assert data["success"] is False
            assert data["status"] == "UNAVAILABLE"

    def test_train_model_unauthorized(self, client: FlaskClient) -> None:
        """
        Test model training without authentication.
        """
        response = client.post(
            "/api/pipeline/train-model",
            json={"model_type": "Teacher"},
        )

        assert response.status_code == 401
        data = response.get_json()
        assert data["error"] == "unauthorized"


# ============================================================================
# RUN HPO ENDPOINT TESTS
# ============================================================================


class TestPipelinesRunHpoEndpoint:
    """
    Test /api/pipeline/run-hpo endpoint.
    """

    def test_run_hpo_success(
        self,
        client: FlaskClient,
        auth_headers: dict[str, str],
    ) -> None:
        """
        Test successful HPO pipeline trigger.
        """
        with patch("ml.dashboard.service.DashboardService.run_hpo_pipeline") as mock_hpo:
            mock_hpo.return_value = {
                "success": True,
                "job_id": "run_hpo_def456",
                "pipeline_type": "run_hpo",
                "status": "QUEUED",
                "message": "Pipeline run_hpo_def456 queued successfully",
                "error": None,
            }

            response = client.post(
                "/api/pipeline/run-hpo",
                json={
                    "search_method": "Optuna",
                    "trials": 100,
                    "model_config": {"max_depth": [3, 5, 7]},
                },
                headers=auth_headers,
            )

            assert response.status_code == 202
            data = response.get_json()
            assert data["success"] is True
            assert data["job_id"] == "run_hpo_def456"
            assert data["status"] == "QUEUED"
            mock_hpo.assert_called_once()

    def test_run_hpo_unavailable(
        self,
        client: FlaskClient,
        auth_headers: dict[str, str],
    ) -> None:
        """
        Test HPO when pipeline service unavailable.
        """
        with patch("ml.dashboard.service.DashboardService.run_hpo_pipeline") as mock_hpo:
            mock_hpo.return_value = {
                "success": False,
                "status": "UNAVAILABLE",
                "error": "pipeline_service_unavailable",
            }

            response = client.post(
                "/api/pipeline/run-hpo",
                json={"search_method": "Optuna"},
                headers=auth_headers,
            )

            assert response.status_code == 503
            data = response.get_json()
            assert data["success"] is False
            assert data["status"] == "UNAVAILABLE"

    def test_run_hpo_unauthorized(self, client: FlaskClient) -> None:
        """
        Test HPO without authentication.
        """
        response = client.post(
            "/api/pipeline/run-hpo",
            json={"search_method": "Optuna"},
        )

        assert response.status_code == 401
        data = response.get_json()
        assert data["error"] == "unauthorized"


# ============================================================================
# PROGRESS ENDPOINT TESTS
# ============================================================================


class TestPipelinesProgressEndpoint:
    """
    Test /api/pipeline/jobs/<job_id>/progress endpoint.
    """

    def test_get_progress_success(
        self,
        client: FlaskClient,
        auth_headers: dict[str, str],
    ) -> None:
        """
        Test successful progress retrieval.
        """
        with patch("ml.dashboard.service.DashboardService.get_pipeline_progress") as mock_progress:
            mock_progress.return_value = {
                "status": "success",
                "progress": {
                    "job_id": "train_model_xyz789",
                    "status": "RUNNING",
                    "progress": 0.45,
                    "current_stage": "feature_ingestion",
                    "eta_seconds": 180,
                    "message": "Ingesting features",
                    "error": None,
                    "started_at": 1000.0,
                    "finished_at": None,
                    "started_at_iso": "2025-01-01T00:00:00+00:00",
                    "finished_at_iso": None,
                },
            }

            response = client.get(
                "/api/pipeline/jobs/train_model_xyz789/progress",
                headers=auth_headers,
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert data["progress"]["job_id"] == "train_model_xyz789"
            assert data["progress"]["progress"] == 0.45
            assert data["progress"]["current_stage"] == "feature_ingestion"
            mock_progress.assert_called_once_with("train_model_xyz789")

    def test_get_progress_not_found(
        self,
        client: FlaskClient,
        auth_headers: dict[str, str],
    ) -> None:
        """
        Test progress retrieval for non-existent job.
        """
        with patch("ml.dashboard.service.DashboardService.get_pipeline_progress") as mock_progress:
            mock_progress.return_value = {
                "status": "not_found",
                "error": "job_not_found",
            }

            response = client.get(
                "/api/pipeline/jobs/nonexistent_job/progress",
                headers=auth_headers,
            )

            assert response.status_code == 404
            data = response.get_json()
            assert data["status"] == "not_found"
            assert data["error"] == "job_not_found"

    def test_get_progress_unavailable(
        self,
        client: FlaskClient,
        auth_headers: dict[str, str],
    ) -> None:
        """
        Test progress retrieval when service unavailable.
        """
        with patch("ml.dashboard.service.DashboardService.get_pipeline_progress") as mock_progress:
            mock_progress.return_value = {
                "status": "unavailable",
                "error": "pipeline_service_unavailable",
            }

            response = client.get(
                "/api/pipeline/jobs/some_job/progress",
                headers=auth_headers,
            )

            assert response.status_code == 503
            data = response.get_json()
            assert data["status"] == "unavailable"

    def test_get_progress_unauthorized(self, client: FlaskClient) -> None:
        """
        Test progress retrieval without authentication.
        """
        response = client.get("/api/pipeline/jobs/some_job/progress")

        assert response.status_code == 401
        data = response.get_json()
        assert data["error"] == "unauthorized"


# ============================================================================
# CANCEL ENDPOINT TESTS
# ============================================================================


class TestPipelinesCancelEndpoint:
    """
    Test /api/pipeline/jobs/<job_id>/cancel endpoint.
    """

    def test_cancel_job_success(
        self,
        client: FlaskClient,
        auth_headers: dict[str, str],
    ) -> None:
        """
        Test successful job cancellation.
        """
        with patch("ml.dashboard.service.DashboardService.cancel_pipeline_job") as mock_cancel:
            mock_cancel.return_value = {
                "success": True,
                "job_id": "train_model_xyz789",
                "status": "CANCELLED",
                "message": "Cancellation request acknowledged",
                "error": None,
            }

            response = client.post(
                "/api/pipeline/jobs/train_model_xyz789/cancel",
                headers=auth_headers,
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data["success"] is True
            assert data["job_id"] == "train_model_xyz789"
            assert data["status"] == "CANCELLED"
            mock_cancel.assert_called_once_with("train_model_xyz789")

    def test_cancel_job_not_found(
        self,
        client: FlaskClient,
        auth_headers: dict[str, str],
    ) -> None:
        """
        Test cancellation of non-existent job.
        """
        with patch("ml.dashboard.service.DashboardService.cancel_pipeline_job") as mock_cancel:
            mock_cancel.return_value = {
                "success": False,
                "job_id": "nonexistent_job",
                "status": "NOT_FOUND",
                "message": None,
                "error": "Job not found",
            }

            response = client.post(
                "/api/pipeline/jobs/nonexistent_job/cancel",
                headers=auth_headers,
            )

            assert response.status_code == 404
            data = response.get_json()
            assert data["success"] is False
            assert data["status"] == "NOT_FOUND"

    def test_cancel_job_unavailable(
        self,
        client: FlaskClient,
        auth_headers: dict[str, str],
    ) -> None:
        """
        Test cancellation when service unavailable.
        """
        with patch("ml.dashboard.service.DashboardService.cancel_pipeline_job") as mock_cancel:
            mock_cancel.return_value = {
                "success": False,
                "status": "unavailable",
                "error": "pipeline_service_unavailable",
            }

            response = client.post(
                "/api/pipeline/jobs/some_job/cancel",
                headers=auth_headers,
            )

            assert response.status_code == 503
            data = response.get_json()
            assert data["success"] is False
            assert data["status"] == "unavailable"

    def test_cancel_job_unauthorized(self, client: FlaskClient) -> None:
        """
        Test job cancellation without authentication.
        """
        response = client.post("/api/pipeline/jobs/some_job/cancel")

        assert response.status_code == 401
        data = response.get_json()
        assert data["error"] == "unauthorized"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestPipelinesIntegration:
    """
    Integration tests for pipeline routes.
    """

    def test_full_pipeline_workflow(
        self,
        client: FlaskClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test complete workflow: trigger -> progress -> cancel."""
        with (
            patch("ml.dashboard.service.DashboardService.build_dataset_pipeline") as mock_build,
            patch("ml.dashboard.service.DashboardService.get_pipeline_progress") as mock_progress,
            patch("ml.dashboard.service.DashboardService.cancel_pipeline_job") as mock_cancel,
        ):

            # Step 1: Trigger pipeline
            mock_build.return_value = {
                "success": True,
                "job_id": "workflow_test_123",
                "pipeline_type": "build_dataset",
                "status": "QUEUED",
                "message": "Pipeline queued",
                "error": None,
            }

            trigger_response = client.post(
                "/api/pipeline/build-dataset",
                json={"symbols": "SPY"},
                headers=auth_headers,
            )

            assert trigger_response.status_code == 202
            trigger_data = trigger_response.get_json()
            job_id = trigger_data["job_id"]
            assert job_id == "workflow_test_123"

            # Step 2: Check progress
            mock_progress.return_value = {
                "status": "success",
                "progress": {
                    "job_id": job_id,
                    "status": "RUNNING",
                    "progress": 0.3,
                    "current_stage": "ingestion",
                    "eta_seconds": 120,
                    "message": "Running pipeline",
                    "error": None,
                },
            }

            progress_response = client.get(
                f"/api/pipeline/jobs/{job_id}/progress",
                headers=auth_headers,
            )

            assert progress_response.status_code == 200
            progress_data = progress_response.get_json()
            assert progress_data["progress"]["status"] == "RUNNING"

            # Step 3: Cancel job
            mock_cancel.return_value = {
                "success": True,
                "job_id": job_id,
                "status": "CANCELLED",
                "message": "Cancelled",
                "error": None,
            }

            cancel_response = client.post(
                f"/api/pipeline/jobs/{job_id}/cancel",
                headers=auth_headers,
            )

            assert cancel_response.status_code == 200
            cancel_data = cancel_response.get_json()
            assert cancel_data["status"] == "CANCELLED"


__all__ = [
    "TestPipelinesBuildDatasetEndpoint",
    "TestPipelinesCancelEndpoint",
    "TestPipelinesIntegration",
    "TestPipelinesProgressEndpoint",
    "TestPipelinesRunHpoEndpoint",
    "TestPipelinesTrainModelEndpoint",
]
