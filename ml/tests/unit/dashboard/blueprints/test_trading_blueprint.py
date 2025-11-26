"""Tests for Trading Blueprint.

This module tests the trading blueprint routes:
- POST /api/trading/toggle
- POST /api/trading/emergency
- GET /api/trading/health
- GET /api/trading/market-data

Tests:
1. test_trading_toggle_enable_success
2. test_trading_toggle_disable_success
3. test_trading_toggle_unauthorized_401
4. test_trading_toggle_safety_checks_failed
5. test_trading_emergency_success
6. test_trading_emergency_unauthorized_401
7. test_trading_emergency_failure_500
8. test_trading_health_returns_200
9. test_trading_health_with_metrics
10. test_trading_market_data_returns_200
11. test_trading_market_data_with_strategies
12. test_trading_toggle_no_payload
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from flask import Blueprint, Flask
from flask.testing import FlaskClient

from ml.dashboard.blueprints.trading import register_trading_routes

# Module path for patching services (imported inside route functions)
TRADING_SERVICE_PATH = "ml.dashboard.services.trading_service"
# Path for asdict used in the blueprint module
ASDICT_PATH = "ml.dashboard.blueprints.trading.asdict"


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_service() -> MagicMock:
    """Provide a mock DashboardService."""
    svc = MagicMock()
    svc._pipeline_integration_manager = None
    return svc


@pytest.fixture
def app(mock_service: MagicMock) -> Flask:
    """Provide Flask test application with trading blueprint."""
    app = Flask(__name__)
    app.config["TESTING"] = True

    # Track auth state for require_token mock
    app.config["AUTH_ENABLED"] = True

    def require_token() -> bool:
        return app.config.get("AUTH_ENABLED", True)

    # Create a FRESH blueprint for each test to avoid Flask's
    # "already registered" error
    trading_bp = Blueprint("trading", __name__, url_prefix="/api/trading")

    # Register routes with mock service
    register_trading_routes(trading_bp, mock_service, require_token)
    app.register_blueprint(trading_bp)

    return app


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    """Provide Flask test client."""
    return app.test_client()


# ============================================================================
# TEST: trading_toggle_enable_success
# ============================================================================


class TestTradingToggleEnableSuccess:
    """Test POST /api/trading/toggle enable success."""

    def test_trading_toggle_enable_success(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that POST /api/trading/toggle with enable=True returns 200."""
        result_dict = {
            "success": True,
            "live_trading_enabled": True,
            "timestamp": "2025-01-15T12:00:00Z",
            "safety_checks_passed": True,
            "mode": "LIVE",
            "controller_state": None,
            "error": None,
        }

        with (
            patch(f"{TRADING_SERVICE_PATH}.TradingIntegrationService") as mock_svc_class,
            patch(ASDICT_PATH, return_value=result_dict),
        ):
            mock_instance = MagicMock()
            mock_svc_class.return_value = mock_instance

            mock_result = MagicMock()
            mock_result.success = True

            async def mock_toggle(*args: Any, **kwargs: Any) -> MagicMock:
                return mock_result

            mock_instance.toggle_live_trading = mock_toggle

            response = client.post(
                "/api/trading/toggle",
                json={"enable": True, "safety_checks": {"risk_check": True}},
            )

        assert response.status_code == 200
        data = response.get_json()
        assert data is not None
        assert data["success"] is True
        assert data["live_trading_enabled"] is True


# ============================================================================
# TEST: trading_toggle_disable_success
# ============================================================================


class TestTradingToggleDisableSuccess:
    """Test POST /api/trading/toggle disable success."""

    def test_trading_toggle_disable_success(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that POST /api/trading/toggle with enable=False returns 200."""
        result_dict = {
            "success": True,
            "live_trading_enabled": False,
            "timestamp": "2025-01-15T12:00:00Z",
            "safety_checks_passed": True,
            "mode": "STOPPED",
            "controller_state": None,
            "error": None,
        }

        with (
            patch(f"{TRADING_SERVICE_PATH}.TradingIntegrationService") as mock_svc_class,
            patch(ASDICT_PATH, return_value=result_dict),
        ):
            mock_instance = MagicMock()
            mock_svc_class.return_value = mock_instance

            mock_result = MagicMock()
            mock_result.success = True

            async def mock_toggle(*args: Any, **kwargs: Any) -> MagicMock:
                return mock_result

            mock_instance.toggle_live_trading = mock_toggle

            response = client.post(
                "/api/trading/toggle",
                json={"enable": False},
            )

        assert response.status_code == 200
        data = response.get_json()
        assert data is not None
        assert data["success"] is True
        assert data["live_trading_enabled"] is False


# ============================================================================
# TEST: trading_toggle_unauthorized_401
# ============================================================================


class TestTradingToggleUnauthorized401:
    """Test POST /api/trading/toggle unauthorized."""

    def test_trading_toggle_unauthorized_401(
        self,
        app: Flask,
        client: FlaskClient,
    ) -> None:
        """Test that POST /api/trading/toggle without auth returns 401."""
        app.config["AUTH_ENABLED"] = False

        response = client.post(
            "/api/trading/toggle",
            json={"enable": True},
        )

        assert response.status_code == 401
        data = response.get_json()
        assert data is not None
        assert data["error"] == "unauthorized"


# ============================================================================
# TEST: trading_toggle_safety_checks_failed
# ============================================================================


class TestTradingToggleSafetyChecksFailed:
    """Test POST /api/trading/toggle with failed safety checks."""

    def test_trading_toggle_safety_checks_failed(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that POST /api/trading/toggle returns 400 when safety checks fail."""
        result_dict = {
            "success": False,
            "live_trading_enabled": False,
            "timestamp": "2025-01-15T12:00:00Z",
            "safety_checks_passed": False,
            "mode": "STOPPED",
            "controller_state": None,
            "error": "User safety checks failed: risk_check",
        }

        with (
            patch(f"{TRADING_SERVICE_PATH}.TradingIntegrationService") as mock_svc_class,
            patch(ASDICT_PATH, return_value=result_dict),
        ):
            mock_instance = MagicMock()
            mock_svc_class.return_value = mock_instance

            mock_result = MagicMock()
            mock_result.success = False

            async def mock_toggle(*args: Any, **kwargs: Any) -> MagicMock:
                return mock_result

            mock_instance.toggle_live_trading = mock_toggle

            response = client.post(
                "/api/trading/toggle",
                json={"enable": True, "safety_checks": {"risk_check": False}},
            )

        assert response.status_code == 400
        data = response.get_json()
        assert data is not None
        assert data["success"] is False
        assert data["safety_checks_passed"] is False


# ============================================================================
# TEST: trading_emergency_success
# ============================================================================


class TestTradingEmergencySuccess:
    """Test POST /api/trading/emergency success."""

    def test_trading_emergency_success(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that POST /api/trading/emergency returns 200 on success."""
        result_dict = {
            "success": True,
            "timestamp": "2025-01-15T12:00:00Z",
            "actions_taken": {
                "orders_cancelled": 5,
                "positions_closed": 3,
                "actors_stopped": 2,
                "data_feeds_stopped": True,
                "risk_manager_notified": True,
            },
            "message": "Emergency stop executed successfully",
            "error": None,
        }

        with (
            patch(f"{TRADING_SERVICE_PATH}.TradingIntegrationService") as mock_svc_class,
            patch(ASDICT_PATH, return_value=result_dict),
        ):
            mock_instance = MagicMock()
            mock_svc_class.return_value = mock_instance

            mock_result = MagicMock()
            mock_result.success = True

            async def mock_emergency(*args: Any, **kwargs: Any) -> MagicMock:
                return mock_result

            mock_instance.emergency_stop = mock_emergency

            response = client.post("/api/trading/emergency")

        assert response.status_code == 200
        data = response.get_json()
        assert data is not None
        assert data["success"] is True
        assert data["message"] == "Emergency stop executed successfully"


# ============================================================================
# TEST: trading_emergency_unauthorized_401
# ============================================================================


class TestTradingEmergencyUnauthorized401:
    """Test POST /api/trading/emergency unauthorized."""

    def test_trading_emergency_unauthorized_401(
        self,
        app: Flask,
        client: FlaskClient,
    ) -> None:
        """Test that POST /api/trading/emergency without auth returns 401."""
        app.config["AUTH_ENABLED"] = False

        response = client.post("/api/trading/emergency")

        assert response.status_code == 401
        data = response.get_json()
        assert data is not None
        assert data["error"] == "unauthorized"


# ============================================================================
# TEST: trading_emergency_failure_500
# ============================================================================


class TestTradingEmergencyFailure500:
    """Test POST /api/trading/emergency failure."""

    def test_trading_emergency_failure_500(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that POST /api/trading/emergency returns 500 on failure."""
        result_dict = {
            "success": False,
            "timestamp": "2025-01-15T12:00:00Z",
            "actions_taken": {
                "orders_cancelled": 0,
                "positions_closed": 0,
                "actors_stopped": 0,
                "data_feeds_stopped": False,
                "risk_manager_notified": False,
            },
            "message": "Emergency stop failed",
            "error": "Controller connection lost",
        }

        with (
            patch(f"{TRADING_SERVICE_PATH}.TradingIntegrationService") as mock_svc_class,
            patch(ASDICT_PATH, return_value=result_dict),
        ):
            mock_instance = MagicMock()
            mock_svc_class.return_value = mock_instance

            mock_result = MagicMock()
            mock_result.success = False

            async def mock_emergency(*args: Any, **kwargs: Any) -> MagicMock:
                return mock_result

            mock_instance.emergency_stop = mock_emergency

            response = client.post("/api/trading/emergency")

        assert response.status_code == 500
        data = response.get_json()
        assert data is not None
        assert data["success"] is False
        assert data["error"] == "Controller connection lost"


# ============================================================================
# TEST: trading_health_returns_200
# ============================================================================


class TestTradingHealthReturns200:
    """Test GET /api/trading/health returns 200."""

    def test_trading_health_returns_200(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that GET /api/trading/health returns 200 with health data."""
        health_data = {
            "healthy": True,
            "trading_enabled": False,
            "market_data": "standby",
            "risk_manager": "idle",
            "mode": "STOPPED",
            "last_transition": None,
            "total_positions": 0,
            "total_exposure": 0.0,
            "unrealized_pnl": 0.0,
        }

        with patch(
            f"{TRADING_SERVICE_PATH}.TradingIntegrationService"
        ) as mock_svc_class:
            mock_instance = MagicMock()
            mock_svc_class.return_value = mock_instance

            async def mock_health(*args: Any, **kwargs: Any) -> dict[str, Any]:
                return health_data

            mock_instance.health_check = mock_health

            response = client.get("/api/trading/health")

        assert response.status_code == 200
        data = response.get_json()
        assert data is not None
        assert data["healthy"] is True
        assert data["mode"] == "STOPPED"


# ============================================================================
# TEST: trading_health_with_metrics
# ============================================================================


class TestTradingHealthWithMetrics:
    """Test GET /api/trading/health with trading metrics."""

    def test_trading_health_with_metrics(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that GET /api/trading/health includes position metrics."""
        health_data = {
            "healthy": True,
            "trading_enabled": True,
            "market_data": "connected",
            "risk_manager": "active",
            "mode": "LIVE",
            "last_transition": "2025-01-15T12:00:00Z",
            "total_positions": 5,
            "total_exposure": 100000.0,
            "unrealized_pnl": 1500.0,
        }

        with patch(
            f"{TRADING_SERVICE_PATH}.TradingIntegrationService"
        ) as mock_svc_class:
            mock_instance = MagicMock()
            mock_svc_class.return_value = mock_instance

            async def mock_health(*args: Any, **kwargs: Any) -> dict[str, Any]:
                return health_data

            mock_instance.health_check = mock_health

            response = client.get("/api/trading/health")

        assert response.status_code == 200
        data = response.get_json()
        assert data is not None
        assert data["trading_enabled"] is True
        assert data["total_positions"] == 5
        assert data["total_exposure"] == 100000.0
        assert data["unrealized_pnl"] == 1500.0


# ============================================================================
# TEST: trading_market_data_returns_200
# ============================================================================


class TestTradingMarketDataReturns200:
    """Test GET /api/trading/market-data returns 200."""

    def test_trading_market_data_returns_200(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that GET /api/trading/market-data returns 200 with metrics."""
        metrics_dict = {
            "generated_at": "2025-01-15T12:00:00Z",
            "total_positions": 3,
            "total_exposure": 50000.0,
            "unrealized_pnl": 750.0,
            "realized_pnl": 1200.0,
            "strategies": [],
        }

        with (
            patch(f"{TRADING_SERVICE_PATH}.TradingIntegrationService") as mock_svc_class,
            patch(ASDICT_PATH, return_value=metrics_dict),
        ):
            mock_instance = MagicMock()
            mock_svc_class.return_value = mock_instance

            mock_metrics = MagicMock()

            async def mock_get_metrics(*args: Any, **kwargs: Any) -> MagicMock:
                return mock_metrics

            mock_instance.get_trading_metrics = mock_get_metrics

            response = client.get("/api/trading/market-data")

        assert response.status_code == 200
        data = response.get_json()
        assert data is not None
        assert data["total_positions"] == 3
        assert data["total_exposure"] == 50000.0


# ============================================================================
# TEST: trading_market_data_with_strategies
# ============================================================================


class TestTradingMarketDataWithStrategies:
    """Test GET /api/trading/market-data with strategy breakdown."""

    def test_trading_market_data_with_strategies(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that GET /api/trading/market-data includes strategy exposure."""
        metrics_dict = {
            "generated_at": "2025-01-15T12:00:00Z",
            "total_positions": 2,
            "total_exposure": 25000.0,
            "unrealized_pnl": 500.0,
            "realized_pnl": 800.0,
            "strategies": [
                {
                    "strategy_id": "strategy_1",
                    "positions": 2,
                    "exposure": 25000.0,
                    "unrealized_pnl": 500.0,
                    "realized_pnl": 800.0,
                }
            ],
        }

        with (
            patch(f"{TRADING_SERVICE_PATH}.TradingIntegrationService") as mock_svc_class,
            patch(ASDICT_PATH, return_value=metrics_dict),
        ):
            mock_instance = MagicMock()
            mock_svc_class.return_value = mock_instance

            mock_metrics = MagicMock()

            async def mock_get_metrics(*args: Any, **kwargs: Any) -> MagicMock:
                return mock_metrics

            mock_instance.get_trading_metrics = mock_get_metrics

            response = client.get("/api/trading/market-data")

        assert response.status_code == 200
        data = response.get_json()
        assert data is not None
        assert data["total_positions"] == 2
        assert "strategies" in data
        assert len(data["strategies"]) == 1


# ============================================================================
# TEST: trading_toggle_no_payload
# ============================================================================


class TestTradingToggleNoPayload:
    """Test POST /api/trading/toggle with no payload defaults."""

    def test_trading_toggle_no_payload(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that POST /api/trading/toggle with empty payload uses defaults."""
        result_dict = {
            "success": True,
            "live_trading_enabled": False,
            "timestamp": "2025-01-15T12:00:00Z",
            "safety_checks_passed": True,
            "mode": "STOPPED",
            "controller_state": None,
            "error": None,
        }

        with (
            patch(f"{TRADING_SERVICE_PATH}.TradingIntegrationService") as mock_svc_class,
            patch(ASDICT_PATH, return_value=result_dict),
        ):
            mock_instance = MagicMock()
            mock_svc_class.return_value = mock_instance

            mock_result = MagicMock()
            mock_result.success = True

            async def mock_toggle(*args: Any, **kwargs: Any) -> MagicMock:
                return mock_result

            mock_instance.toggle_live_trading = mock_toggle

            response = client.post(
                "/api/trading/toggle",
                json={},
            )

        assert response.status_code == 200
        data = response.get_json()
        assert data is not None
        assert data["live_trading_enabled"] is False


__all__ = [
    "TestTradingEmergencyFailure500",
    "TestTradingEmergencySuccess",
    "TestTradingEmergencyUnauthorized401",
    "TestTradingHealthReturns200",
    "TestTradingHealthWithMetrics",
    "TestTradingMarketDataReturns200",
    "TestTradingMarketDataWithStrategies",
    "TestTradingToggleDisableSuccess",
    "TestTradingToggleEnableSuccess",
    "TestTradingToggleNoPayload",
    "TestTradingToggleSafetyChecksFailed",
    "TestTradingToggleUnauthorized401",
]
