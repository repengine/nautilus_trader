"""Unit tests for Features Blueprint.

This module tests all feature engineering-related API routes extracted to the features blueprint.
Tests cover 4 routes:
    - POST /api/features/designer/generate
    - POST /api/features/validate-code
    - POST /api/features/analyze
    - GET /api/features/manifests
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flask import Blueprint, Flask
from flask.testing import FlaskClient

from ml.dashboard.blueprints.features import register_features_routes


@dataclass
class MockFeatureGenerationResult:
    """Mock result for feature generation."""

    success: bool
    feature_set_id: str
    feature_count: int = 0
    feature_names: Sequence[str] = field(default_factory=list)
    manifest: dict[str, Any] | None = None
    error: str | None = None
    validation_errors: Sequence[str] = field(default_factory=list)


@dataclass
class MockCodeValidationResult:
    """Mock result for code validation."""

    valid: bool
    errors: Sequence[str] = field(default_factory=list)
    warnings: Sequence[str] = field(default_factory=list)
    security_risk: bool = False
    syntax_error: bool = False
    signature_error: bool = False


@dataclass
class MockFeatureAnalysisResult:
    """Mock result for feature analysis."""

    success: bool
    total_features: int = 0
    feature_names: Sequence[str] = field(default_factory=list)
    avg_correlation: float | None = None
    max_correlation: float | None = None
    feature_importance_method: str | None = None
    top_features: Sequence[dict[str, Any]] = field(default_factory=list)
    data_quality: dict[str, Any] = field(default_factory=dict)
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
    """Create a test Flask app with the features blueprint."""
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
    bp = Blueprint("features", __name__, url_prefix="/api/features")
    register_features_routes(bp, mock_service, _require_token)  # type: ignore[arg-type]
    app.register_blueprint(bp)
    return app


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    """Create a test client."""
    return app.test_client()


# =============================================================================
# POST /api/features/designer/generate
# =============================================================================


def test_features_designer_generate_success(client: FlaskClient) -> None:
    """Test POST /api/features/designer/generate returns 200 on success."""
    mock_result = MockFeatureGenerationResult(
        success=True,
        feature_set_id="test_features",
        feature_count=5,
        feature_names=["return_10", "return_20", "momentum_10", "rsi", "atr"],
        manifest={"feature_set_id": "test_features", "version": "1.0.0"},
    )

    with patch(
        "ml.dashboard.services.features_service.FeatureEngineeringService"
    ) as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.generate_features = AsyncMock(return_value=mock_result)
        mock_svc_cls.return_value = mock_svc

        resp = client.post(
            "/api/features/designer/generate",
            json={
                "feature_set_name": "test_features",
                "price_features": True,
                "volume_features": False,
                "technical_indicators": ["rsi", "atr"],
                "lookback_periods": "10,20,50",
            },
        )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert data["feature_set_id"] == "test_features"
    assert data["feature_count"] == 5


def test_features_designer_generate_validation_error(client: FlaskClient) -> None:
    """Test POST /api/features/designer/generate returns 400 on validation failure."""
    mock_result = MockFeatureGenerationResult(
        success=False,
        feature_set_id="test_features",
        error="Code validation failed",
        validation_errors=["Invalid syntax"],
    )

    with patch(
        "ml.dashboard.services.features_service.FeatureEngineeringService"
    ) as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.generate_features = AsyncMock(return_value=mock_result)
        mock_svc_cls.return_value = mock_svc

        resp = client.post(
            "/api/features/designer/generate",
            json={
                "feature_set_name": "test_features",
                "custom_code": "invalid code",
            },
        )

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False
    assert data["error"] == "Code validation failed"


def test_features_designer_generate_missing_name(client: FlaskClient) -> None:
    """Test POST /api/features/designer/generate returns 400 when name missing."""
    resp = client.post(
        "/api/features/designer/generate",
        json={"price_features": True},
    )

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False
    assert "feature_set_name required" in data["error"]


def test_features_designer_generate_empty_name(client: FlaskClient) -> None:
    """Test POST /api/features/designer/generate returns 400 when name is empty."""
    resp = client.post(
        "/api/features/designer/generate",
        json={"feature_set_name": "   "},
    )

    assert resp.status_code == 400
    data = resp.get_json()
    assert "feature_set_name required" in data["error"]


def test_features_designer_generate_unauthorized(
    client: FlaskClient, app: Flask
) -> None:
    """Test POST /api/features/designer/generate returns 401 when unauthorized."""
    app.config["_token_required"] = True
    app.config["_token_valid"] = False

    resp = client.post(
        "/api/features/designer/generate",
        json={"feature_set_name": "test"},
    )

    assert resp.status_code == 401
    assert resp.get_json()["error"] == "unauthorized"


# =============================================================================
# POST /api/features/validate-code
# =============================================================================


def test_features_validate_code_valid(client: FlaskClient) -> None:
    """Test POST /api/features/validate-code returns valid result."""
    mock_result = MockCodeValidationResult(
        valid=True,
        warnings=["Consider using vectorized operations"],
    )

    with patch(
        "ml.dashboard.services.features_service.FeatureEngineeringService"
    ) as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.validate_code = AsyncMock(return_value=mock_result)
        mock_svc_cls.return_value = mock_svc

        resp = client.post(
            "/api/features/validate-code",
            json={
                "code": "def compute_custom_features(self, data):\n    return data",
                "test_execution": False,
            },
        )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["valid"] is True
    assert "Consider using vectorized operations" in data["warnings"]


def test_features_validate_code_invalid_syntax(client: FlaskClient) -> None:
    """Test POST /api/features/validate-code returns syntax error."""
    mock_result = MockCodeValidationResult(
        valid=False,
        errors=["Syntax error at line 1: invalid syntax"],
        syntax_error=True,
    )

    with patch(
        "ml.dashboard.services.features_service.FeatureEngineeringService"
    ) as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.validate_code = AsyncMock(return_value=mock_result)
        mock_svc_cls.return_value = mock_svc

        resp = client.post(
            "/api/features/validate-code",
            json={"code": "def invalid("},
        )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["valid"] is False
    assert data["syntax_error"] is True


def test_features_validate_code_security_risk(client: FlaskClient) -> None:
    """Test POST /api/features/validate-code returns security risk."""
    mock_result = MockCodeValidationResult(
        valid=False,
        errors=["Dangerous import not allowed: os"],
        security_risk=True,
    )

    with patch(
        "ml.dashboard.services.features_service.FeatureEngineeringService"
    ) as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.validate_code = AsyncMock(return_value=mock_result)
        mock_svc_cls.return_value = mock_svc

        resp = client.post(
            "/api/features/validate-code",
            json={"code": "import os\nos.system('rm -rf /')"},
        )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["valid"] is False
    assert data["security_risk"] is True


def test_features_validate_code_no_code(client: FlaskClient) -> None:
    """Test POST /api/features/validate-code returns 400 when no code provided."""
    resp = client.post(
        "/api/features/validate-code",
        json={},
    )

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["valid"] is False
    assert "No code provided" in data["errors"]


def test_features_validate_code_empty_code(client: FlaskClient) -> None:
    """Test POST /api/features/validate-code returns 400 when code is empty."""
    resp = client.post(
        "/api/features/validate-code",
        json={"code": "   "},
    )

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["valid"] is False


def test_features_validate_code_does_not_require_auth(client: FlaskClient) -> None:
    """Test POST /api/features/validate-code does not require authentication."""
    # This endpoint is public for code validation
    mock_result = MockCodeValidationResult(valid=True)

    with patch(
        "ml.dashboard.services.features_service.FeatureEngineeringService"
    ) as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.validate_code = AsyncMock(return_value=mock_result)
        mock_svc_cls.return_value = mock_svc

        # Even with invalid token, should work
        resp = client.post(
            "/api/features/validate-code",
            json={"code": "def compute_custom_features(self, data): return data"},
        )

    assert resp.status_code == 200


# =============================================================================
# POST /api/features/analyze
# =============================================================================


def test_features_analyze_success(client: FlaskClient) -> None:
    """Test POST /api/features/analyze returns 200 on success."""
    mock_result = MockFeatureAnalysisResult(
        success=True,
        total_features=10,
        feature_names=["return_10", "momentum_5", "rsi"],
        avg_correlation=0.35,
        max_correlation=0.85,
        feature_importance_method="shap",
        top_features=[{"name": "return_10", "importance": 0.25}],
        data_quality={"completeness": 0.99},
    )

    with patch(
        "ml.dashboard.services.features_service.FeatureEngineeringService"
    ) as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.analyze_features = AsyncMock(return_value=mock_result)
        mock_svc_cls.return_value = mock_svc

        resp = client.post(
            "/api/features/analyze",
            json={
                "feature_set_id": "test_features",
                "method": "shap",
                "limit": 1000,
            },
        )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert data["total_features"] == 10
    assert data["feature_importance_method"] == "shap"


def test_features_analyze_failure(client: FlaskClient) -> None:
    """Test POST /api/features/analyze returns 400 on failure."""
    mock_result = MockFeatureAnalysisResult(
        success=False,
        error="Feature set not found: unknown_features",
    )

    with patch(
        "ml.dashboard.services.features_service.FeatureEngineeringService"
    ) as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.analyze_features = AsyncMock(return_value=mock_result)
        mock_svc_cls.return_value = mock_svc

        resp = client.post(
            "/api/features/analyze",
            json={"feature_set_id": "unknown_features"},
        )

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False
    assert "Feature set not found" in data["error"]


def test_features_analyze_missing_feature_set_id(client: FlaskClient) -> None:
    """Test POST /api/features/analyze returns 400 when feature_set_id missing."""
    resp = client.post(
        "/api/features/analyze",
        json={"method": "shap"},
    )

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False
    assert "feature_set_id required" in data["error"]


def test_features_analyze_empty_feature_set_id(client: FlaskClient) -> None:
    """Test POST /api/features/analyze returns 400 when feature_set_id is empty."""
    resp = client.post(
        "/api/features/analyze",
        json={"feature_set_id": "   "},
    )

    assert resp.status_code == 400
    data = resp.get_json()
    assert "feature_set_id required" in data["error"]


def test_features_analyze_unauthorized(client: FlaskClient, app: Flask) -> None:
    """Test POST /api/features/analyze returns 401 when unauthorized."""
    app.config["_token_required"] = True
    app.config["_token_valid"] = False

    resp = client.post(
        "/api/features/analyze",
        json={"feature_set_id": "test_features"},
    )

    assert resp.status_code == 401
    assert resp.get_json()["error"] == "unauthorized"


# =============================================================================
# GET /api/features/manifests
# =============================================================================


def test_features_manifests_success(client: FlaskClient) -> None:
    """Test GET /api/features/manifests returns 200 with manifests."""
    mock_result = {
        "success": True,
        "count": 2,
        "manifests": [
            {"feature_set_id": "fs1", "version": "1.0.0"},
            {"feature_set_id": "fs2", "version": "2.0.0"},
        ],
    }

    with patch(
        "ml.dashboard.services.features_service.FeatureEngineeringService"
    ) as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.list_manifests = AsyncMock(return_value=mock_result)
        mock_svc_cls.return_value = mock_svc

        resp = client.get("/api/features/manifests")

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert data["count"] == 2
    assert len(data["manifests"]) == 2


def test_features_manifests_empty(client: FlaskClient) -> None:
    """Test GET /api/features/manifests returns empty list when no manifests."""
    mock_result = {
        "success": True,
        "count": 0,
        "manifests": [],
    }

    with patch(
        "ml.dashboard.services.features_service.FeatureEngineeringService"
    ) as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.list_manifests = AsyncMock(return_value=mock_result)
        mock_svc_cls.return_value = mock_svc

        resp = client.get("/api/features/manifests")

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert data["count"] == 0
    assert data["manifests"] == []


def test_features_manifests_error(client: FlaskClient) -> None:
    """Test GET /api/features/manifests returns 500 on error."""
    mock_result = {
        "success": False,
        "error": "Feature registry not available",
        "manifests": [],
    }

    with patch(
        "ml.dashboard.services.features_service.FeatureEngineeringService"
    ) as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.list_manifests = AsyncMock(return_value=mock_result)
        mock_svc_cls.return_value = mock_svc

        resp = client.get("/api/features/manifests")

    assert resp.status_code == 500
    data = resp.get_json()
    assert data["success"] is False
    assert "Feature registry not available" in data["error"]


def test_features_manifests_unauthorized(client: FlaskClient, app: Flask) -> None:
    """Test GET /api/features/manifests returns 401 when unauthorized."""
    app.config["_token_required"] = True
    app.config["_token_valid"] = False

    resp = client.get("/api/features/manifests")

    assert resp.status_code == 401
    assert resp.get_json()["error"] == "unauthorized"
