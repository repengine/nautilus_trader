"""Tests for Metrics Blueprint.

This module tests the metrics blueprint routes:
- GET /api/metrics/snapshot - Get real-time metrics snapshot with KPIs
- GET /api/metrics/portfolio - Get portfolio summary with positions
- GET /api/metrics/ingestion - Get data ingestion rates
- GET /api/metrics/experiments - Get active experiments status

Tests (10):
1. test_metrics_snapshot_returns_200
2. test_metrics_snapshot_returns_kpi_data
3. test_metrics_portfolio_returns_200
4. test_metrics_portfolio_returns_portfolio_data
5. test_metrics_ingestion_returns_200
6. test_metrics_ingestion_returns_ingestion_data
7. test_metrics_experiments_returns_200
8. test_metrics_experiments_returns_experiments_list
9. test_metrics_snapshot_handles_no_integration
10. test_metrics_snapshot_delegates_to_service
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from flask import Blueprint, Flask
from flask.testing import FlaskClient

from ml.dashboard.blueprints.metrics import register_metrics_routes


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_service() -> MagicMock:
    """Provide a mock DashboardService."""
    svc = MagicMock()
    svc._pipeline_integration_manager = MagicMock()
    return svc


@pytest.fixture
def mock_metrics_snapshot() -> MagicMock:
    """Provide a mock StoreMetricsSnapshot."""
    snapshot = MagicMock()
    snapshot.to_dict.return_value = {
        "daily_pnl": 1234.56,
        "sharpe_ratio": 1.5,
        "win_rate": 0.65,
        "max_drawdown": 0.05,
        "active_models": 3,
        "ingestion_rate": {
            "bars_per_sec": 100.5,
            "quotes_per_sec": 500.2,
            "l2_updates_per_sec": 1000.0,
            "data_quality": 0.99,
        },
        "portfolio": {
            "total_value": 100000.0,
            "cash": 50000.0,
            "margin_used": 25000.0,
            "positions": 5,
        },
    }
    return snapshot


@pytest.fixture
def mock_portfolio_snapshot() -> MagicMock:
    """Provide a mock PortfolioSnapshot."""
    snapshot = MagicMock()
    snapshot.to_dict.return_value = {
        "total_value": 100000.0,
        "cash": 50000.0,
        "margin_used": 25000.0,
        "positions": 5,
    }
    return snapshot


@pytest.fixture
def mock_ingestion_snapshot() -> MagicMock:
    """Provide a mock IngestionRateSnapshot."""
    snapshot = MagicMock()
    snapshot.to_dict.return_value = {
        "bars_per_sec": 100.5,
        "quotes_per_sec": 500.2,
        "l2_updates_per_sec": 1000.0,
        "data_quality": 0.99,
    }
    return snapshot


@pytest.fixture
def mock_store_service(
    mock_metrics_snapshot: MagicMock,
    mock_portfolio_snapshot: MagicMock,
    mock_ingestion_snapshot: MagicMock,
) -> MagicMock:
    """Provide a mock StoreIntegrationService."""
    service = MagicMock()

    # Create async mocks for async methods
    async def mock_get_metrics_snapshot() -> MagicMock:
        return mock_metrics_snapshot

    async def mock_get_portfolio_snapshot() -> MagicMock:
        return mock_portfolio_snapshot

    async def mock_get_ingestion_snapshot() -> MagicMock:
        return mock_ingestion_snapshot

    async def mock_get_experiments_snapshot() -> list[dict[str, Any]]:
        return [
            {
                "type": "model_training",
                "experiment_id": "exp_001",
                "status": "running",
                "created_at": "2025-01-15T12:00:00Z",
                "metrics": {},
            },
            {
                "type": "feature_selection",
                "experiment_id": "exp_002",
                "status": "active",
                "created_at": "2025-01-15T11:00:00Z",
                "metrics": {},
            },
        ]

    service.get_metrics_snapshot = mock_get_metrics_snapshot
    service.get_portfolio_snapshot = mock_get_portfolio_snapshot
    service.get_ingestion_snapshot = mock_get_ingestion_snapshot
    service.get_experiments_snapshot = mock_get_experiments_snapshot

    return service


@pytest.fixture
def app(mock_service: MagicMock) -> Flask:
    """Provide Flask test application with metrics blueprint."""
    app = Flask(__name__)
    app.config["TESTING"] = True

    # Create a FRESH blueprint for each test
    bp = Blueprint("metrics", __name__, url_prefix="/api/metrics")

    # Register routes with mock service
    register_metrics_routes(bp, mock_service)
    app.register_blueprint(bp)

    return app


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    """Provide Flask test client."""
    return app.test_client()


# ============================================================================
# TEST 1: test_metrics_snapshot_returns_200
# ============================================================================


class TestMetricsSnapshotReturns200:
    """Test GET /api/metrics/snapshot returns 200."""

    def test_metrics_snapshot_returns_200(
        self,
        client: FlaskClient,
        mock_store_service: MagicMock,
    ) -> None:
        """Test that metrics snapshot endpoint returns HTTP 200."""
        with patch(
            "ml.dashboard.services.metrics_service.StoreIntegrationService",
            return_value=mock_store_service,
        ):
            response = client.get("/api/metrics/snapshot")

            assert response.status_code == 200


# ============================================================================
# TEST 2: test_metrics_snapshot_returns_kpi_data
# ============================================================================


class TestMetricsSnapshotReturnsKpiData:
    """Test GET /api/metrics/snapshot returns KPI data."""

    def test_metrics_snapshot_returns_kpi_data(
        self,
        client: FlaskClient,
        mock_store_service: MagicMock,
    ) -> None:
        """Test that metrics snapshot returns all KPI fields."""
        with patch(
            "ml.dashboard.services.metrics_service.StoreIntegrationService",
            return_value=mock_store_service,
        ):
            response = client.get("/api/metrics/snapshot")

            assert response.status_code == 200
            data = response.get_json()
            assert data is not None
            assert "daily_pnl" in data
            assert "sharpe_ratio" in data
            assert "win_rate" in data
            assert "max_drawdown" in data
            assert "active_models" in data
            assert "ingestion_rate" in data
            assert "portfolio" in data


# ============================================================================
# TEST 3: test_metrics_portfolio_returns_200
# ============================================================================


class TestMetricsPortfolioReturns200:
    """Test GET /api/metrics/portfolio returns 200."""

    def test_metrics_portfolio_returns_200(
        self,
        client: FlaskClient,
        mock_store_service: MagicMock,
    ) -> None:
        """Test that portfolio endpoint returns HTTP 200."""
        with patch(
            "ml.dashboard.services.metrics_service.StoreIntegrationService",
            return_value=mock_store_service,
        ):
            response = client.get("/api/metrics/portfolio")

            assert response.status_code == 200


# ============================================================================
# TEST 4: test_metrics_portfolio_returns_portfolio_data
# ============================================================================


class TestMetricsPortfolioReturnsPortfolioData:
    """Test GET /api/metrics/portfolio returns portfolio data."""

    def test_metrics_portfolio_returns_portfolio_data(
        self,
        client: FlaskClient,
        mock_store_service: MagicMock,
    ) -> None:
        """Test that portfolio returns all portfolio fields."""
        with patch(
            "ml.dashboard.services.metrics_service.StoreIntegrationService",
            return_value=mock_store_service,
        ):
            response = client.get("/api/metrics/portfolio")

            assert response.status_code == 200
            data = response.get_json()
            assert data is not None
            assert "total_value" in data
            assert "cash" in data
            assert "margin_used" in data
            assert "positions" in data
            assert data["total_value"] == 100000.0
            assert data["positions"] == 5


# ============================================================================
# TEST 5: test_metrics_ingestion_returns_200
# ============================================================================


class TestMetricsIngestionReturns200:
    """Test GET /api/metrics/ingestion returns 200."""

    def test_metrics_ingestion_returns_200(
        self,
        client: FlaskClient,
        mock_store_service: MagicMock,
    ) -> None:
        """Test that ingestion endpoint returns HTTP 200."""
        with patch(
            "ml.dashboard.services.metrics_service.StoreIntegrationService",
            return_value=mock_store_service,
        ):
            response = client.get("/api/metrics/ingestion")

            assert response.status_code == 200


# ============================================================================
# TEST 6: test_metrics_ingestion_returns_ingestion_data
# ============================================================================


class TestMetricsIngestionReturnsIngestionData:
    """Test GET /api/metrics/ingestion returns ingestion data."""

    def test_metrics_ingestion_returns_ingestion_data(
        self,
        client: FlaskClient,
        mock_store_service: MagicMock,
    ) -> None:
        """Test that ingestion returns all ingestion rate fields."""
        with patch(
            "ml.dashboard.services.metrics_service.StoreIntegrationService",
            return_value=mock_store_service,
        ):
            response = client.get("/api/metrics/ingestion")

            assert response.status_code == 200
            data = response.get_json()
            assert data is not None
            assert "bars_per_sec" in data
            assert "quotes_per_sec" in data
            assert "l2_updates_per_sec" in data
            assert "data_quality" in data
            assert data["bars_per_sec"] == 100.5
            assert data["data_quality"] == 0.99


# ============================================================================
# TEST 7: test_metrics_experiments_returns_200
# ============================================================================


class TestMetricsExperimentsReturns200:
    """Test GET /api/metrics/experiments returns 200."""

    def test_metrics_experiments_returns_200(
        self,
        client: FlaskClient,
        mock_store_service: MagicMock,
    ) -> None:
        """Test that experiments endpoint returns HTTP 200."""
        with patch(
            "ml.dashboard.services.metrics_service.StoreIntegrationService",
            return_value=mock_store_service,
        ):
            response = client.get("/api/metrics/experiments")

            assert response.status_code == 200


# ============================================================================
# TEST 8: test_metrics_experiments_returns_experiments_list
# ============================================================================


class TestMetricsExperimentsReturnsExperimentsList:
    """Test GET /api/metrics/experiments returns experiments list."""

    def test_metrics_experiments_returns_experiments_list(
        self,
        client: FlaskClient,
        mock_store_service: MagicMock,
    ) -> None:
        """Test that experiments returns list of experiments."""
        with patch(
            "ml.dashboard.services.metrics_service.StoreIntegrationService",
            return_value=mock_store_service,
        ):
            response = client.get("/api/metrics/experiments")

            assert response.status_code == 200
            data = response.get_json()
            assert data is not None
            assert "experiments" in data
            assert isinstance(data["experiments"], list)
            assert len(data["experiments"]) == 2
            assert data["experiments"][0]["type"] == "model_training"
            assert data["experiments"][1]["type"] == "feature_selection"


# ============================================================================
# TEST 9: test_metrics_snapshot_handles_no_integration
# ============================================================================


class TestMetricsSnapshotHandlesNoIntegration:
    """Test GET /api/metrics/snapshot handles missing integration."""

    def test_metrics_snapshot_handles_no_integration(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that snapshot returns defaults when no integration available."""
        # Create a mock that returns empty snapshot
        mock_snapshot = MagicMock()
        mock_snapshot.to_dict.return_value = {
            "daily_pnl": 0.0,
            "sharpe_ratio": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "active_models": 0,
            "ingestion_rate": {
                "bars_per_sec": 0.0,
                "quotes_per_sec": 0.0,
                "l2_updates_per_sec": 0.0,
                "data_quality": 0.0,
            },
            "portfolio": {
                "total_value": 0.0,
                "cash": 0.0,
                "margin_used": 0.0,
                "positions": 0,
            },
        }

        async def mock_get_metrics_snapshot() -> MagicMock:
            return mock_snapshot

        mock_service = MagicMock()
        mock_service.get_metrics_snapshot = mock_get_metrics_snapshot

        with patch(
            "ml.dashboard.services.metrics_service.StoreIntegrationService",
            return_value=mock_service,
        ):
            response = client.get("/api/metrics/snapshot")

            assert response.status_code == 200
            data = response.get_json()
            assert data is not None
            assert data["daily_pnl"] == 0.0
            assert data["active_models"] == 0


# ============================================================================
# TEST 10: test_metrics_snapshot_delegates_to_service
# ============================================================================


class TestMetricsSnapshotDelegatesToService:
    """Test that metrics_snapshot delegates to StoreIntegrationService."""

    def test_metrics_snapshot_delegates_to_service(
        self,
        client: FlaskClient,
        mock_service: MagicMock,
        mock_store_service: MagicMock,
    ) -> None:
        """Test that metrics snapshot correctly delegates to service."""
        with patch(
            "ml.dashboard.services.metrics_service.StoreIntegrationService",
            return_value=mock_store_service,
        ) as mock_cls:
            response = client.get("/api/metrics/snapshot")

            assert response.status_code == 200
            # Verify service was instantiated with integration manager
            mock_cls.assert_called_once_with(mock_service._pipeline_integration_manager)


__all__ = [
    "TestMetricsExperimentsReturns200",
    "TestMetricsExperimentsReturnsExperimentsList",
    "TestMetricsIngestionReturns200",
    "TestMetricsIngestionReturnsIngestionData",
    "TestMetricsPortfolioReturns200",
    "TestMetricsPortfolioReturnsPortfolioData",
    "TestMetricsSnapshotDelegatesToService",
    "TestMetricsSnapshotHandlesNoIntegration",
    "TestMetricsSnapshotReturns200",
    "TestMetricsSnapshotReturnsKpiData",
]
