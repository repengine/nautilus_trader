"""Unit tests for Strategies Blueprint.

This module tests all strategy-related API routes extracted to the strategies blueprint.
Tests cover 10 routes:
    - GET /api/strategies - List strategies
    - POST /api/strategies - Create strategy (501)
    - POST /api/strategies/optimize - Optimize strategy (501)
    - POST /api/strategies/validate - Validate strategy code
    - POST /api/strategies/backtest - Run backtest
    - GET /api/strategies/backtest/<job_id>/status - Get backtest status
    - GET /api/strategies/backtest/<job_id>/results - Get backtest results
    - POST /api/strategies/deploy - Deploy strategy
    - GET /api/strategies/<strategy_id>/performance - Get performance
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from flask import Blueprint, Flask
from flask.testing import FlaskClient

from ml.dashboard.blueprints.strategies import register_strategies_routes


@dataclass
class MockCodeValidationResult:
    """Mock result for strategy code validation."""

    valid: bool
    errors: Sequence[str] = field(default_factory=list)
    warnings: Sequence[str] = field(default_factory=list)
    security_risk: bool = False
    syntax_error: bool = False
    signature_error: bool = False
    allowed_imports: Sequence[str] = field(default_factory=list)


@dataclass
class MockBacktestResult:
    """Mock result for backtest."""

    job_id: str
    status: str
    performance_metrics: Mapping[str, float] = field(default_factory=dict)
    trades: Sequence[Mapping[str, Any]] = field(default_factory=list)
    equity_curve: Sequence[Mapping[str, Any]] = field(default_factory=list)
    error: str | None = None
    execution_time_seconds: float = 0.0


@dataclass
class MockDeploymentResult:
    """Mock result for deployment."""

    deployment_id: str
    status: str
    environment: str
    message: str
    monitoring_url: str | None = None
    error: str | None = None


class MockDashboardService:
    """Mock DashboardService for testing blueprint routes."""

    def __init__(self) -> None:
        self.integration_manager = MagicMock()

    def get_integration_manager(self) -> MagicMock:
        return self.integration_manager


@pytest.fixture
def mock_service() -> MockDashboardService:
    """Create a mock dashboard service instance."""
    return MockDashboardService()


@pytest.fixture
def app(mock_service: MockDashboardService) -> Flask:
    """Create a test Flask app with the strategies blueprint."""
    app = Flask(__name__)
    app.config["TESTING"] = True

    # Store token state on app for tests to modify
    app.config["_token_required"] = False
    app.config["_token_valid"] = True

    def _require_token() -> bool:
        if not app.config["_token_required"]:
            return True
        return app.config["_token_valid"]

    # Create a fresh blueprint for each test to avoid the "already registered" error
    bp = Blueprint("strategies", __name__, url_prefix="/api/strategies")
    register_strategies_routes(bp, mock_service, _require_token)  # type: ignore[arg-type]
    app.register_blueprint(bp)
    return app


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    """Create a test client."""
    return app.test_client()


# =============================================================================
# GET /api/strategies
# =============================================================================


def test_strategies_list_success(client: FlaskClient) -> None:
    """Test GET /api/strategies returns 200 with strategies list."""
    mock_result = {
        "strategies": [
            {"strategy_id": "strategy_1", "name": "Momentum Strategy"},
            {"strategy_id": "strategy_2", "name": "Mean Reversion"},
        ],
        "count": 2,
    }

    with patch("ml.dashboard.services.strategy_service.StrategyService") as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.list_strategies.return_value = mock_result
        mock_svc_cls.return_value = mock_svc

        resp = client.get("/api/strategies")

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["count"] == 2
    assert len(data["strategies"]) == 2


def test_strategies_list_empty(client: FlaskClient) -> None:
    """Test GET /api/strategies returns empty list when no strategies."""
    mock_result = {"strategies": [], "count": 0}

    with patch("ml.dashboard.services.strategy_service.StrategyService") as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.list_strategies.return_value = mock_result
        mock_svc_cls.return_value = mock_svc

        resp = client.get("/api/strategies")

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["count"] == 0
    assert data["strategies"] == []


# =============================================================================
# POST /api/strategies (Not Implemented)
# =============================================================================


def test_strategies_create_returns_501(client: FlaskClient) -> None:
    """Test POST /api/strategies returns 501 Not Implemented."""
    resp = client.post(
        "/api/strategies",
        json={"name": "New Strategy"},
    )

    assert resp.status_code == 501
    data = resp.get_json()
    assert data["error"] == "not_implemented"
    assert "strategy registry CLI" in data["message"]


# =============================================================================
# POST /api/strategies/optimize (Not Implemented)
# =============================================================================


def test_strategies_optimize_returns_501(client: FlaskClient) -> None:
    """Test POST /api/strategies/optimize returns 501 Not Implemented."""
    resp = client.post(
        "/api/strategies/optimize",
        json={"strategy_id": "test"},
    )

    assert resp.status_code == 501
    data = resp.get_json()
    assert data["error"] == "not_implemented"
    assert "dedicated pipeline jobs" in data["message"]


# =============================================================================
# POST /api/strategies/validate
# =============================================================================


def test_strategies_validate_valid_code(client: FlaskClient) -> None:
    """Test POST /api/strategies/validate returns valid result."""
    mock_result = MockCodeValidationResult(
        valid=True,
        warnings=["Consider adding stop-loss logic"],
        allowed_imports=["ml.strategies.base", "numpy"],
    )

    with patch("ml.dashboard.services.strategy_service.StrategyService") as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.validate_strategy_code.return_value = mock_result
        mock_svc_cls.return_value = mock_svc

        resp = client.post(
            "/api/strategies/validate",
            json={
                "code": "class MyStrategy(MLTradingStrategy): pass",
                "strategy_name": "MyStrategy",
                "base_strategy": "MLTradingStrategy",
            },
        )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["valid"] is True
    assert "ml.strategies.base" in data["allowed_imports"]


def test_strategies_validate_invalid_syntax(client: FlaskClient) -> None:
    """Test POST /api/strategies/validate returns syntax error."""
    mock_result = MockCodeValidationResult(
        valid=False,
        errors=["Syntax error at line 1: invalid syntax"],
        syntax_error=True,
    )

    with patch("ml.dashboard.services.strategy_service.StrategyService") as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.validate_strategy_code.return_value = mock_result
        mock_svc_cls.return_value = mock_svc

        resp = client.post(
            "/api/strategies/validate",
            json={"code": "class MyStrategy("},
        )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["valid"] is False
    assert data["syntax_error"] is True


def test_strategies_validate_security_risk(client: FlaskClient) -> None:
    """Test POST /api/strategies/validate returns security risk."""
    mock_result = MockCodeValidationResult(
        valid=False,
        errors=["SECURITY: Dangerous import: os"],
        security_risk=True,
    )

    with patch("ml.dashboard.services.strategy_service.StrategyService") as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.validate_strategy_code.return_value = mock_result
        mock_svc_cls.return_value = mock_svc

        resp = client.post(
            "/api/strategies/validate",
            json={"code": "import os\nos.system('rm -rf /')"},
        )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["valid"] is False
    assert data["security_risk"] is True


def test_strategies_validate_empty_code(client: FlaskClient) -> None:
    """Test POST /api/strategies/validate returns 400 when code is empty."""
    resp = client.post(
        "/api/strategies/validate",
        json={"code": ""},
    )

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"] == "empty_code"
    assert data["valid"] is False


def test_strategies_validate_whitespace_code(client: FlaskClient) -> None:
    """Test POST /api/strategies/validate returns 400 when code is whitespace."""
    resp = client.post(
        "/api/strategies/validate",
        json={"code": "   \n\t  "},
    )

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"] == "empty_code"


def test_strategies_validate_unauthorized(client: FlaskClient, app: Flask) -> None:
    """Test POST /api/strategies/validate returns 401 when unauthorized."""
    app.config["_token_required"] = True
    app.config["_token_valid"] = False

    resp = client.post(
        "/api/strategies/validate",
        json={"code": "class MyStrategy(MLTradingStrategy): pass"},
    )

    assert resp.status_code == 401
    assert resp.get_json()["error"] == "unauthorized"


# =============================================================================
# POST /api/strategies/backtest
# =============================================================================


def test_strategies_backtest_queued(client: FlaskClient) -> None:
    """Test POST /api/strategies/backtest returns 202 when queued."""
    mock_result = MockBacktestResult(
        job_id="job-12345",
        status="queued",
    )

    with patch("ml.dashboard.services.strategy_service.StrategyService") as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.submit_backtest.return_value = mock_result
        mock_svc_cls.return_value = mock_svc

        resp = client.post(
            "/api/strategies/backtest",
            json={
                "code": "class MyStrategy(MLTradingStrategy): pass",
                "strategy_name": "MyStrategy",
                "start_date": "2024-01-01",
                "end_date": "2024-06-30",
                "initial_balance": 50000.0,
            },
        )

    assert resp.status_code == 202
    data = resp.get_json()
    assert data["job_id"] == "job-12345"
    assert data["status"] == "queued"


def test_strategies_backtest_validation_failed(client: FlaskClient) -> None:
    """Test POST /api/strategies/backtest returns 400 when validation fails."""
    mock_result = MockBacktestResult(
        job_id="",
        status="failed",
        error="Code validation failed",
    )

    with patch("ml.dashboard.services.strategy_service.StrategyService") as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.submit_backtest.return_value = mock_result
        mock_svc_cls.return_value = mock_svc

        resp = client.post(
            "/api/strategies/backtest",
            json={
                "code": "invalid code",
                "strategy_name": "MyStrategy",
            },
        )

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["status"] == "failed"
    assert "validation failed" in data["error"]


def test_strategies_backtest_empty_code(client: FlaskClient) -> None:
    """Test POST /api/strategies/backtest returns 400 when code is empty."""
    resp = client.post(
        "/api/strategies/backtest",
        json={
            "code": "",
            "strategy_name": "MyStrategy",
        },
    )

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"] == "empty_code"


def test_strategies_backtest_empty_strategy_name(client: FlaskClient) -> None:
    """Test POST /api/strategies/backtest returns 400 when strategy_name is empty."""
    resp = client.post(
        "/api/strategies/backtest",
        json={
            "code": "class MyStrategy: pass",
            "strategy_name": "",
        },
    )

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"] == "empty_strategy_name"


def test_strategies_backtest_unauthorized(client: FlaskClient, app: Flask) -> None:
    """Test POST /api/strategies/backtest returns 401 when unauthorized."""
    app.config["_token_required"] = True
    app.config["_token_valid"] = False

    resp = client.post(
        "/api/strategies/backtest",
        json={
            "code": "class MyStrategy: pass",
            "strategy_name": "MyStrategy",
        },
    )

    assert resp.status_code == 401
    assert resp.get_json()["error"] == "unauthorized"


# =============================================================================
# GET /api/strategies/backtest/<job_id>/status
# =============================================================================


def test_strategies_backtest_status_found(client: FlaskClient) -> None:
    """Test GET /api/strategies/backtest/<job_id>/status returns status."""
    mock_result = MockBacktestResult(
        job_id="job-12345",
        status="running",
    )

    with patch("ml.dashboard.services.strategy_service.StrategyService") as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.get_backtest_status.return_value = mock_result
        mock_svc_cls.return_value = mock_svc

        resp = client.get("/api/strategies/backtest/job-12345/status")

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["job_id"] == "job-12345"
    assert data["status"] == "running"


def test_strategies_backtest_status_not_found(client: FlaskClient) -> None:
    """Test GET /api/strategies/backtest/<job_id>/status returns 404."""
    with patch("ml.dashboard.services.strategy_service.StrategyService") as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.get_backtest_status.return_value = None
        mock_svc_cls.return_value = mock_svc

        resp = client.get("/api/strategies/backtest/unknown-job/status")

    assert resp.status_code == 404
    data = resp.get_json()
    assert data["error"] == "not_found"
    assert data["job_id"] == "unknown-job"


# =============================================================================
# GET /api/strategies/backtest/<job_id>/results
# =============================================================================


def test_strategies_backtest_results_completed(client: FlaskClient) -> None:
    """Test GET /api/strategies/backtest/<job_id>/results returns results."""
    mock_result = MockBacktestResult(
        job_id="job-12345",
        status="completed",
        performance_metrics={"sharpe_ratio": 1.5, "total_pnl": 5000.0},
        trades=[{"id": 1, "pnl": 100.0}],
        equity_curve=[{"timestamp": 1704067200, "equity": 100100.0}],
        execution_time_seconds=15.5,
    )

    with patch("ml.dashboard.services.strategy_service.StrategyService") as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.get_backtest_status.return_value = mock_result
        mock_svc_cls.return_value = mock_svc

        resp = client.get("/api/strategies/backtest/job-12345/results")

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["job_id"] == "job-12345"
    assert data["status"] == "completed"
    assert data["performance_metrics"]["sharpe_ratio"] == 1.5
    assert len(data["trades"]) == 1
    assert data["execution_time_seconds"] == 15.5


def test_strategies_backtest_results_not_completed(client: FlaskClient) -> None:
    """Test GET /api/strategies/backtest/<job_id>/results returns 400 if not completed."""
    mock_result = MockBacktestResult(
        job_id="job-12345",
        status="running",
    )

    with patch("ml.dashboard.services.strategy_service.StrategyService") as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.get_backtest_status.return_value = mock_result
        mock_svc_cls.return_value = mock_svc

        resp = client.get("/api/strategies/backtest/job-12345/results")

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"] == "not_completed"
    assert data["status"] == "running"


def test_strategies_backtest_results_not_found(client: FlaskClient) -> None:
    """Test GET /api/strategies/backtest/<job_id>/results returns 404."""
    with patch("ml.dashboard.services.strategy_service.StrategyService") as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.get_backtest_status.return_value = None
        mock_svc_cls.return_value = mock_svc

        resp = client.get("/api/strategies/backtest/unknown-job/results")

    assert resp.status_code == 404
    data = resp.get_json()
    assert data["error"] == "not_found"


# =============================================================================
# POST /api/strategies/deploy
# =============================================================================


def test_strategies_deploy_success(client: FlaskClient) -> None:
    """Test POST /api/strategies/deploy returns 201 on success."""
    mock_result = MockDeploymentResult(
        deployment_id="deploy-12345",
        status="deployed",
        environment="staging",
        message="Strategy deployed to staging environment",
        monitoring_url="/api/strategies/deploy-12345/performance",
    )

    with patch("ml.dashboard.services.strategy_service.StrategyService") as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.deploy_strategy.return_value = mock_result
        mock_svc_cls.return_value = mock_svc

        resp = client.post(
            "/api/strategies/deploy",
            json={
                "code": "class MyStrategy(MLTradingStrategy): pass",
                "strategy_name": "MyStrategy",
                "environment": "staging",
            },
        )

    assert resp.status_code == 201
    data = resp.get_json()
    assert data["deployment_id"] == "deploy-12345"
    assert data["status"] == "deployed"
    assert data["environment"] == "staging"


def test_strategies_deploy_pending_approval(client: FlaskClient) -> None:
    """Test POST /api/strategies/deploy returns 201 for pending_approval."""
    mock_result = MockDeploymentResult(
        deployment_id="deploy-12345",
        status="pending_approval",
        environment="live",
        message="Strategy deployment requires manual approval",
    )

    with patch("ml.dashboard.services.strategy_service.StrategyService") as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.deploy_strategy.return_value = mock_result
        mock_svc_cls.return_value = mock_svc

        resp = client.post(
            "/api/strategies/deploy",
            json={
                "code": "class MyStrategy: pass",
                "strategy_name": "MyStrategy",
                "environment": "live",
            },
        )

    assert resp.status_code == 201
    data = resp.get_json()
    assert data["status"] == "pending_approval"


def test_strategies_deploy_failed(client: FlaskClient) -> None:
    """Test POST /api/strategies/deploy returns 400 on failure."""
    mock_result = MockDeploymentResult(
        deployment_id="",
        status="failed",
        environment="staging",
        message="Code validation failed",
        error="Validation errors: ['Invalid code']",
    )

    with patch("ml.dashboard.services.strategy_service.StrategyService") as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.deploy_strategy.return_value = mock_result
        mock_svc_cls.return_value = mock_svc

        resp = client.post(
            "/api/strategies/deploy",
            json={
                "code": "invalid",
                "strategy_name": "MyStrategy",
            },
        )

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["status"] == "failed"


def test_strategies_deploy_empty_code(client: FlaskClient) -> None:
    """Test POST /api/strategies/deploy returns 400 when code is empty."""
    resp = client.post(
        "/api/strategies/deploy",
        json={
            "code": "",
            "strategy_name": "MyStrategy",
        },
    )

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"] == "empty_code"


def test_strategies_deploy_empty_strategy_name(client: FlaskClient) -> None:
    """Test POST /api/strategies/deploy returns 400 when strategy_name is empty."""
    resp = client.post(
        "/api/strategies/deploy",
        json={
            "code": "class MyStrategy: pass",
            "strategy_name": "",
        },
    )

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"] == "empty_strategy_name"


def test_strategies_deploy_unauthorized(client: FlaskClient, app: Flask) -> None:
    """Test POST /api/strategies/deploy returns 401 when unauthorized."""
    app.config["_token_required"] = True
    app.config["_token_valid"] = False

    resp = client.post(
        "/api/strategies/deploy",
        json={
            "code": "class MyStrategy: pass",
            "strategy_name": "MyStrategy",
        },
    )

    assert resp.status_code == 401
    assert resp.get_json()["error"] == "unauthorized"


# =============================================================================
# GET /api/strategies/<strategy_id>/performance
# =============================================================================


def test_strategies_performance_success(client: FlaskClient) -> None:
    """Test GET /api/strategies/<strategy_id>/performance returns metrics."""
    mock_result = {
        "strategy_id": "strategy-123",
        "status": "running",
        "metrics": {
            "total_pnl": 1250.50,
            "sharpe_ratio": 1.8,
            "max_drawdown": 0.05,
            "win_rate": 0.65,
            "total_trades": 42,
        },
        "recent_trades": [],
    }

    with patch("ml.dashboard.services.strategy_service.StrategyService") as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.get_strategy_performance.return_value = mock_result
        mock_svc_cls.return_value = mock_svc

        resp = client.get("/api/strategies/strategy-123/performance")

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["strategy_id"] == "strategy-123"
    assert data["metrics"]["total_pnl"] == 1250.50
    assert data["metrics"]["sharpe_ratio"] == 1.8


def test_strategies_performance_not_found(client: FlaskClient) -> None:
    """Test GET /api/strategies/<strategy_id>/performance returns error dict."""
    mock_result = {
        "strategy_id": "unknown",
        "error": "Strategy not found",
    }

    with patch("ml.dashboard.services.strategy_service.StrategyService") as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.get_strategy_performance.return_value = mock_result
        mock_svc_cls.return_value = mock_svc

        resp = client.get("/api/strategies/unknown/performance")

    assert resp.status_code == 200  # Service returns 200 with error in response
    data = resp.get_json()
    assert data["strategy_id"] == "unknown"
    assert "error" in data
