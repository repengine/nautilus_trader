"""Unit tests for Registry Blueprint.

This module tests all registry-related API routes extracted to the registry blueprint.
Tests cover all 16 routes including GET endpoints for read operations and POST endpoints
for authenticated write operations.
"""

from __future__ import annotations

from typing import Any

import pytest
from flask import Blueprint, Flask
from flask.testing import FlaskClient

from ml.dashboard.blueprints.registry import register_registry_routes
from ml.dashboard.service import DashboardService


class MockDashboardService:
    """Mock DashboardService for testing blueprint routes."""

    def __init__(self) -> None:
        self.list_models_result: list[dict[str, Any]] = []
        self.get_model_performance_history_result: list[dict[str, Any]] = []
        self.list_deployments_result: dict[str, list[str]] = {}
        self.list_features_result: list[dict[str, Any]] = []
        self.list_strategies_result: list[dict[str, Any]] = []
        self.list_datasets_result: list[dict[str, Any]] = []
        self.get_strategy_details_result: dict[str, Any] | None = None
        self.check_strategy_compatibility_result: dict[str, Any] = {}
        self.get_feature_lineage_result: list[dict[str, Any]] = []
        self.list_watermarks_result: list[dict[str, Any]] = []
        self.list_dataset_lineage_result: list[dict[str, Any]] = []
        self.promote_feature_result: dict[str, Any] = {}
        self.deprecate_feature_result: dict[str, Any] = {}
        self.deploy_model_result: dict[str, Any] = {}
        self.hot_reload_model_result: dict[str, Any] = {}
        self.rollback_deployment_result: dict[str, Any] = {}

        # Track calls for verification
        self.get_model_performance_history_calls: list[tuple[str, int | None]] = []
        self.list_features_calls: list[dict[str, str | None]] = []
        self.check_strategy_compatibility_calls: list[tuple[str, list[str]]] = []
        self.get_feature_lineage_calls: list[str] = []
        self.list_watermarks_calls: list[dict[str, Any]] = []
        self.list_dataset_lineage_calls: list[dict[str, Any]] = []
        self.promote_feature_calls: list[dict[str, Any]] = []
        self.deprecate_feature_calls: list[dict[str, Any]] = []
        self.deploy_model_calls: list[tuple[str, str]] = []
        self.hot_reload_model_calls: list[dict[str, str]] = []
        self.rollback_deployment_calls: list[dict[str, str]] = []

    def list_models(self) -> list[dict[str, Any]]:
        return self.list_models_result

    def get_model_performance_history(
        self, model_id: str, *, limit: int | None = None
    ) -> list[dict[str, Any]]:
        self.get_model_performance_history_calls.append((model_id, limit))
        return self.get_model_performance_history_result

    def list_deployments(self) -> dict[str, list[str]]:
        return self.list_deployments_result

    def list_features(
        self, *, role: str | None = None, stage: str | None = None
    ) -> list[dict[str, Any]]:
        self.list_features_calls.append({"role": role, "stage": stage})
        return self.list_features_result

    def list_strategies(self) -> list[dict[str, Any]]:
        return self.list_strategies_result

    def list_datasets(self) -> list[dict[str, Any]]:
        return self.list_datasets_result

    def get_strategy_details(self, strategy_id: str) -> dict[str, Any] | None:
        return self.get_strategy_details_result

    def check_strategy_compatibility(
        self, strategy_id: str, active: list[str]
    ) -> dict[str, Any]:
        self.check_strategy_compatibility_calls.append((strategy_id, active))
        return self.check_strategy_compatibility_result

    def get_feature_lineage(self, feature_set_id: str) -> list[dict[str, Any]]:
        self.get_feature_lineage_calls.append(feature_set_id)
        return self.get_feature_lineage_result

    def list_watermarks(
        self,
        *,
        dataset_id: str,
        instrument: str | None = None,
        source: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        self.list_watermarks_calls.append({
            "dataset_id": dataset_id,
            "instrument": instrument,
            "source": source,
            "limit": limit,
        })
        return self.list_watermarks_result

    def list_dataset_lineage(
        self,
        *,
        child: str | None = None,
        parent: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        self.list_dataset_lineage_calls.append({
            "child": child,
            "parent": parent,
            "limit": limit,
        })
        return self.list_dataset_lineage_result

    def promote_feature(
        self,
        feature_set_id: str,
        *,
        stage: str | None = None,
        gates: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.promote_feature_calls.append({
            "feature_set_id": feature_set_id,
            "stage": stage,
            "gates": gates,
        })
        return self.promote_feature_result

    def deprecate_feature(
        self,
        feature_set_id: str,
        *,
        reason: str | None = None,
    ) -> dict[str, Any]:
        self.deprecate_feature_calls.append({
            "feature_set_id": feature_set_id,
            "reason": reason,
        })
        return self.deprecate_feature_result

    def deploy_model(self, model_id: str, target: str) -> dict[str, Any]:
        self.deploy_model_calls.append((model_id, target))
        return self.deploy_model_result

    def hot_reload_model(self, target: str, new_model_id: str) -> dict[str, Any]:
        self.hot_reload_model_calls.append({
            "target": target,
            "new_model_id": new_model_id,
        })
        return self.hot_reload_model_result

    def rollback_deployment(self, target: str, to_model_id: str) -> dict[str, Any]:
        self.rollback_deployment_calls.append({
            "target": target,
            "to_model_id": to_model_id,
        })
        return self.rollback_deployment_result


@pytest.fixture
def mock_service() -> MockDashboardService:
    """Create a mock dashboard service instance."""
    return MockDashboardService()


@pytest.fixture
def app(mock_service: MockDashboardService) -> Flask:
    """Create a test Flask app with the registry blueprint."""
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
    bp = Blueprint("registry", __name__, url_prefix="/api/registry")
    register_registry_routes(bp, mock_service, _require_token)  # type: ignore[arg-type]
    app.register_blueprint(bp)
    return app


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    """Create a test client."""
    return app.test_client()


# =============================================================================
# GET /api/registry/models
# =============================================================================


def test_registry_models_returns_200(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test GET /api/registry/models returns 200 with model list."""
    mock_service.list_models_result = [
        {
            "model_id": "model_1",
            "role": "primary",
            "version": "1.0.0",
            "deployment_status": "active",
            "deployed_to": ["ml_signal_actor"],
            "architecture": "xgboost",
            "feature_schema_hash": "abc123",
        }
    ]

    resp = client.get("/api/registry/models")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data) == 1
    assert data[0]["model_id"] == "model_1"


def test_registry_models_returns_empty_list(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test GET /api/registry/models returns empty list when no models."""
    mock_service.list_models_result = []

    resp = client.get("/api/registry/models")
    assert resp.status_code == 200
    assert resp.get_json() == []


# =============================================================================
# GET /api/registry/models/<model_id>/history
# =============================================================================


def test_registry_model_history_success(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test GET /api/registry/models/<model_id>/history returns history."""
    mock_service.get_model_performance_history_result = [
        {"timestamp": "2024-01-01T00:00:00Z", "accuracy": 0.95},
        {"timestamp": "2024-01-02T00:00:00Z", "accuracy": 0.96},
    ]

    resp = client.get("/api/registry/models/model_1/history")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data) == 2
    assert mock_service.get_model_performance_history_calls == [("model_1", None)]


def test_registry_model_history_with_limit(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test GET /api/registry/models/<model_id>/history with limit parameter."""
    mock_service.get_model_performance_history_result = [
        {"timestamp": "2024-01-01T00:00:00Z", "accuracy": 0.95},
    ]

    resp = client.get("/api/registry/models/model_1/history?limit=1")
    assert resp.status_code == 200
    assert mock_service.get_model_performance_history_calls == [("model_1", 1)]


def test_registry_model_history_invalid_limit(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test GET /api/registry/models/<model_id>/history with invalid limit returns 400."""
    resp = client.get("/api/registry/models/model_1/history?limit=invalid")
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"] == "invalid_limit"


def test_registry_model_history_negative_limit(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test GET /api/registry/models/<model_id>/history with negative limit returns 400."""
    resp = client.get("/api/registry/models/model_1/history?limit=-5")
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"] == "invalid_limit"


# =============================================================================
# GET /api/registry/deployments
# =============================================================================


def test_registry_deployments_returns_200(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test GET /api/registry/deployments returns 200 with deployments."""
    mock_service.list_deployments_result = {
        "ml_signal_actor": ["model_1", "model_2"],
        "ml_strategy": ["model_3"],
    }

    resp = client.get("/api/registry/deployments")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "ml_signal_actor" in data
    assert len(data["ml_signal_actor"]) == 2


def test_registry_deployments_returns_empty_dict(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test GET /api/registry/deployments returns empty dict when no deployments."""
    mock_service.list_deployments_result = {}

    resp = client.get("/api/registry/deployments")
    assert resp.status_code == 200
    assert resp.get_json() == {}


# =============================================================================
# GET /api/registry/features
# =============================================================================


def test_registry_features_returns_200(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test GET /api/registry/features returns 200 with features."""
    mock_service.list_features_result = [
        {
            "feature_set_id": "fs_1",
            "role": "primary",
            "stage": "PROD",
            "schema_hash": "xyz789",
            "version": "1.0.0",
        }
    ]

    resp = client.get("/api/registry/features")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data) == 1
    assert data[0]["feature_set_id"] == "fs_1"
    assert mock_service.list_features_calls == [{"role": None, "stage": None}]


def test_registry_features_with_role_filter(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test GET /api/registry/features with role filter."""
    mock_service.list_features_result = []

    resp = client.get("/api/registry/features?role=primary")
    assert resp.status_code == 200
    assert mock_service.list_features_calls == [{"role": "primary", "stage": None}]


def test_registry_features_with_stage_filter(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test GET /api/registry/features with stage filter."""
    mock_service.list_features_result = []

    resp = client.get("/api/registry/features?stage=PROD")
    assert resp.status_code == 200
    assert mock_service.list_features_calls == [{"role": None, "stage": "PROD"}]


def test_registry_features_with_both_filters(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test GET /api/registry/features with both role and stage filters."""
    mock_service.list_features_result = []

    resp = client.get("/api/registry/features?role=secondary&stage=DEV")
    assert resp.status_code == 200
    assert mock_service.list_features_calls == [{"role": "secondary", "stage": "DEV"}]


# =============================================================================
# GET /api/registry/strategies
# =============================================================================


def test_registry_strategies_returns_200(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test GET /api/registry/strategies returns 200 with strategies."""
    mock_service.list_strategies_result = [
        {
            "strategy_id": "strategy_1",
            "type": "momentum",
            "version": "1.0.0",
            "required_models": ["model_1"],
        }
    ]

    resp = client.get("/api/registry/strategies")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data) == 1
    assert data[0]["strategy_id"] == "strategy_1"


# =============================================================================
# GET /api/registry/datasets
# =============================================================================


def test_registry_datasets_returns_200(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test GET /api/registry/datasets returns 200 with datasets."""
    mock_service.list_datasets_result = [
        {
            "dataset_id": "dataset_1",
            "dataset_type": "features",
            "location": "/data/features",
            "version": "1.0.0",
        }
    ]

    resp = client.get("/api/registry/datasets")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data) == 1
    assert data[0]["dataset_id"] == "dataset_1"


# =============================================================================
# GET /api/registry/strategies/<strategy_id>
# =============================================================================


def test_strategy_details_success(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test GET /api/registry/strategies/<strategy_id> returns details."""
    mock_service.get_strategy_details_result = {
        "strategy_id": "strategy_1",
        "type": "momentum",
        "version": "1.0.0",
        "required_models": ["model_1"],
        "required_features": ["feature_1"],
        "suitable_regimes": ["trending"],
        "instrument_types": ["equity"],
    }

    resp = client.get("/api/registry/strategies/strategy_1")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["strategy_id"] == "strategy_1"


def test_strategy_details_not_found_returns_empty(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test GET /api/registry/strategies/<strategy_id> returns empty when not found."""
    mock_service.get_strategy_details_result = None

    resp = client.get("/api/registry/strategies/unknown")
    assert resp.status_code == 200
    assert resp.get_json() == {}


# =============================================================================
# GET /api/registry/strategies/<strategy_id>/compatibility
# =============================================================================


def test_strategy_compatibility_success(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test GET /api/registry/strategies/<strategy_id>/compatibility returns result."""
    mock_service.check_strategy_compatibility_result = {
        "strategy_id": "strategy_1",
        "compatible": True,
    }

    resp = client.get(
        "/api/registry/strategies/strategy_1/compatibility?active=strategy_2,strategy_3"
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["compatible"] is True
    assert mock_service.check_strategy_compatibility_calls == [
        ("strategy_1", ["strategy_2", "strategy_3"])
    ]


def test_strategy_compatibility_empty_active(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test GET /api/registry/strategies/<strategy_id>/compatibility with empty active."""
    mock_service.check_strategy_compatibility_result = {
        "strategy_id": "strategy_1",
        "compatible": True,
    }

    resp = client.get("/api/registry/strategies/strategy_1/compatibility")
    assert resp.status_code == 200
    assert mock_service.check_strategy_compatibility_calls == [("strategy_1", [])]


def test_strategy_compatibility_active_with_whitespace(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test compatibility endpoint strips whitespace from active strategies."""
    mock_service.check_strategy_compatibility_result = {
        "strategy_id": "strategy_1",
        "compatible": True,
    }

    resp = client.get(
        "/api/registry/strategies/strategy_1/compatibility?active= strat_a , strat_b "
    )
    assert resp.status_code == 200
    assert mock_service.check_strategy_compatibility_calls == [
        ("strategy_1", ["strat_a", "strat_b"])
    ]


# =============================================================================
# GET /api/registry/features/<feature_set_id>/lineage
# =============================================================================


def test_feature_lineage_success(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test GET /api/registry/features/<feature_set_id>/lineage returns lineage."""
    mock_service.get_feature_lineage_result = [
        {
            "feature_set_id": "fs_1",
            "role": "primary",
            "stage": "PROD",
            "version": "1.0.0",
            "schema_hash": "abc123",
        }
    ]

    resp = client.get("/api/registry/features/fs_1/lineage")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data) == 1
    assert mock_service.get_feature_lineage_calls == ["fs_1"]


def test_feature_lineage_empty_result(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test feature lineage returns empty list when no lineage."""
    mock_service.get_feature_lineage_result = []

    resp = client.get("/api/registry/features/unknown/lineage")
    assert resp.status_code == 200
    assert resp.get_json() == []


# =============================================================================
# GET /api/registry/datasets/watermarks
# =============================================================================


def test_dataset_watermarks_no_dataset_id(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test GET /api/registry/datasets/watermarks without dataset_id returns empty."""
    resp = client.get("/api/registry/datasets/watermarks")
    assert resp.status_code == 200
    assert resp.get_json() == []
    assert len(mock_service.list_watermarks_calls) == 0


def test_dataset_watermarks_with_dataset_id(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test GET /api/registry/datasets/watermarks with dataset_id."""
    mock_service.list_watermarks_result = [
        {
            "dataset_id": "features",
            "instrument_id": "SPY",
            "source": "databento",
            "last_success_ns": 1234567890,
            "last_attempt_ns": 1234567890,
            "last_count": 1000,
            "completeness_pct": 99.5,
            "updated_at": "2024-01-01T00:00:00Z",
        }
    ]

    resp = client.get("/api/registry/datasets/watermarks?dataset_id=features")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data) == 1
    assert mock_service.list_watermarks_calls == [
        {"dataset_id": "features", "instrument": None, "source": None, "limit": 100}
    ]


def test_dataset_watermarks_with_all_filters(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test watermarks endpoint with all filter parameters."""
    mock_service.list_watermarks_result = []

    resp = client.get(
        "/api/registry/datasets/watermarks?dataset_id=ds&instrument=SPY&source=db&limit=50"
    )
    assert resp.status_code == 200
    assert mock_service.list_watermarks_calls == [
        {"dataset_id": "ds", "instrument": "SPY", "source": "db", "limit": 50}
    ]


def test_dataset_watermarks_invalid_limit_uses_default(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test watermarks endpoint uses default limit when invalid."""
    mock_service.list_watermarks_result = []

    resp = client.get(
        "/api/registry/datasets/watermarks?dataset_id=ds&limit=invalid"
    )
    assert resp.status_code == 200
    assert mock_service.list_watermarks_calls == [
        {"dataset_id": "ds", "instrument": None, "source": None, "limit": 100}
    ]


# =============================================================================
# GET /api/registry/datasets/lineage
# =============================================================================


def test_dataset_lineage_success(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test GET /api/registry/datasets/lineage returns lineage records."""
    mock_service.list_dataset_lineage_result = [
        {
            "transform_id": "t1",
            "child_dataset_id": "child_ds",
            "parent_dataset_id": "parent_ds",
            "ts_range": "2024-01-01:2024-02-01",
            "parameters": {},
            "created_at": "2024-01-01T00:00:00Z",
        }
    ]

    resp = client.get("/api/registry/datasets/lineage")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data) == 1
    assert mock_service.list_dataset_lineage_calls == [
        {"child": None, "parent": None, "limit": 100}
    ]


def test_dataset_lineage_with_child_filter(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test dataset lineage with child filter."""
    mock_service.list_dataset_lineage_result = []

    resp = client.get("/api/registry/datasets/lineage?child=child_ds")
    assert resp.status_code == 200
    assert mock_service.list_dataset_lineage_calls == [
        {"child": "child_ds", "parent": None, "limit": 100}
    ]


def test_dataset_lineage_with_all_params(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test dataset lineage with all parameters."""
    mock_service.list_dataset_lineage_result = []

    resp = client.get(
        "/api/registry/datasets/lineage?child=c&parent=p&limit=25"
    )
    assert resp.status_code == 200
    assert mock_service.list_dataset_lineage_calls == [
        {"child": "c", "parent": "p", "limit": 25}
    ]


def test_dataset_lineage_invalid_limit_uses_default(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test dataset lineage uses default limit when invalid."""
    mock_service.list_dataset_lineage_result = []

    resp = client.get("/api/registry/datasets/lineage?limit=bad")
    assert resp.status_code == 200
    assert mock_service.list_dataset_lineage_calls == [
        {"child": None, "parent": None, "limit": 100}
    ]


# =============================================================================
# POST /api/registry/features/<feature_set_id>:promote
# =============================================================================


def test_feature_promote_success(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test POST /api/registry/features/<feature_set_id>:promote success."""
    mock_service.promote_feature_result = {
        "ok": True,
        "feature_set_id": "fs_1",
        "stage": "PROD",
    }

    resp = client.post(
        "/api/registry/features/fs_1:promote",
        json={"stage": "PROD"},
    )
    assert resp.status_code == 202
    data = resp.get_json()
    assert data["ok"] is True
    assert data["stage"] == "PROD"
    assert mock_service.promote_feature_calls == [
        {"feature_set_id": "fs_1", "stage": "PROD", "gates": None}
    ]


def test_feature_promote_with_gates(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test feature promote with quality gates."""
    mock_service.promote_feature_result = {
        "ok": True,
        "feature_set_id": "fs_1",
        "stage": "PROD",
    }
    gates = [{"name": "accuracy", "threshold": 0.9}]

    resp = client.post(
        "/api/registry/features/fs_1:promote",
        json={"stage": "PROD", "gates": gates},
    )
    assert resp.status_code == 202
    assert mock_service.promote_feature_calls == [
        {"feature_set_id": "fs_1", "stage": "PROD", "gates": gates}
    ]


def test_feature_promote_invalid_gates_returns_400(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test feature promote with invalid gates returns 400."""
    resp = client.post(
        "/api/registry/features/fs_1:promote",
        json={"stage": "PROD", "gates": "invalid"},
    )
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "invalid_gates"


def test_feature_promote_unauthorized(
    client: FlaskClient, app: Flask, mock_service: MockDashboardService
) -> None:
    """Test feature promote returns 401 when unauthorized."""
    app.config["_token_required"] = True
    app.config["_token_valid"] = False

    resp = client.post(
        "/api/registry/features/fs_1:promote",
        json={"stage": "PROD"},
    )
    assert resp.status_code == 401
    assert resp.get_json()["error"] == "unauthorized"


def test_feature_promote_failure_returns_200(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test feature promote failure returns 200 (not 202)."""
    mock_service.promote_feature_result = {
        "ok": False,
        "feature_set_id": "fs_1",
        "stage": "PROD",
    }

    resp = client.post(
        "/api/registry/features/fs_1:promote",
        json={"stage": "PROD"},
    )
    assert resp.status_code == 200
    assert resp.get_json()["ok"] is False


# =============================================================================
# POST /api/registry/features/<feature_set_id>:deprecate
# =============================================================================


def test_feature_deprecate_success(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test POST /api/registry/features/<feature_set_id>:deprecate success."""
    mock_service.deprecate_feature_result = {
        "ok": True,
        "feature_set_id": "fs_1",
    }

    resp = client.post(
        "/api/registry/features/fs_1:deprecate",
        json={"reason": "outdated"},
    )
    assert resp.status_code == 202
    data = resp.get_json()
    assert data["ok"] is True
    assert mock_service.deprecate_feature_calls == [
        {"feature_set_id": "fs_1", "reason": "outdated"}
    ]


def test_feature_deprecate_without_reason(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test feature deprecate without reason."""
    mock_service.deprecate_feature_result = {
        "ok": True,
        "feature_set_id": "fs_1",
    }

    resp = client.post(
        "/api/registry/features/fs_1:deprecate",
        json={},
    )
    assert resp.status_code == 202
    assert mock_service.deprecate_feature_calls == [
        {"feature_set_id": "fs_1", "reason": None}
    ]


def test_feature_deprecate_unauthorized(
    client: FlaskClient, app: Flask, mock_service: MockDashboardService
) -> None:
    """Test feature deprecate returns 401 when unauthorized."""
    app.config["_token_required"] = True
    app.config["_token_valid"] = False

    resp = client.post(
        "/api/registry/features/fs_1:deprecate",
        json={},
    )
    assert resp.status_code == 401


# =============================================================================
# POST /api/registry/models/<model_id>:deploy
# =============================================================================


def test_registry_deploy_success(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test POST /api/registry/models/<model_id>:deploy success."""
    mock_service.deploy_model_result = {
        "ok": True,
        "model_id": "model_1",
        "target": "ml_signal_actor",
    }

    resp = client.post(
        "/api/registry/models/model_1:deploy",
        json={"target": "ml_signal_actor"},
    )
    assert resp.status_code == 202
    data = resp.get_json()
    assert data["ok"] is True
    assert mock_service.deploy_model_calls == [("model_1", "ml_signal_actor")]


def test_registry_deploy_default_target(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test deploy uses default target when not specified."""
    mock_service.deploy_model_result = {
        "ok": True,
        "model_id": "model_1",
        "target": "ml_signal_actor",
    }

    resp = client.post(
        "/api/registry/models/model_1:deploy",
        json={},
    )
    assert resp.status_code == 202
    assert mock_service.deploy_model_calls == [("model_1", "ml_signal_actor")]


def test_registry_deploy_invalid_target(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test deploy returns 400 for empty target."""
    resp = client.post(
        "/api/registry/models/model_1:deploy",
        json={"target": ""},
    )
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "invalid_target"


def test_registry_deploy_whitespace_target(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test deploy returns 400 for whitespace-only target."""
    resp = client.post(
        "/api/registry/models/model_1:deploy",
        json={"target": "   "},
    )
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "invalid_target"


def test_registry_deploy_unauthorized(
    client: FlaskClient, app: Flask, mock_service: MockDashboardService
) -> None:
    """Test deploy returns 401 when unauthorized."""
    app.config["_token_required"] = True
    app.config["_token_valid"] = False

    resp = client.post(
        "/api/registry/models/model_1:deploy",
        json={"target": "ml_signal_actor"},
    )
    assert resp.status_code == 401


# =============================================================================
# POST /api/registry/models/<model_id>:hot_reload
# =============================================================================


def test_hot_reload_success(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test POST /api/registry/models/<model_id>:hot_reload success."""
    mock_service.hot_reload_model_result = {
        "ok": True,
        "target": "ml_signal_actor",
        "model_id": "model_1",
    }

    resp = client.post(
        "/api/registry/models/model_1:hot_reload",
        json={"target": "ml_signal_actor"},
    )
    assert resp.status_code == 202
    data = resp.get_json()
    assert data["ok"] is True
    assert mock_service.hot_reload_model_calls == [
        {"target": "ml_signal_actor", "new_model_id": "model_1"}
    ]


def test_hot_reload_default_target(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test hot reload uses default target."""
    mock_service.hot_reload_model_result = {
        "ok": True,
        "target": "ml_signal_actor",
        "model_id": "model_1",
    }

    resp = client.post(
        "/api/registry/models/model_1:hot_reload",
        json={},
    )
    assert resp.status_code == 202
    assert mock_service.hot_reload_model_calls == [
        {"target": "ml_signal_actor", "new_model_id": "model_1"}
    ]


def test_hot_reload_invalid_target(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test hot reload returns 400 for empty target."""
    resp = client.post(
        "/api/registry/models/model_1:hot_reload",
        json={"target": ""},
    )
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "invalid_target"


def test_hot_reload_unauthorized(
    client: FlaskClient, app: Flask, mock_service: MockDashboardService
) -> None:
    """Test hot reload returns 401 when unauthorized."""
    app.config["_token_required"] = True
    app.config["_token_valid"] = False

    resp = client.post(
        "/api/registry/models/model_1:hot_reload",
        json={},
    )
    assert resp.status_code == 401


# =============================================================================
# POST /api/registry/deployments:rollback
# =============================================================================


def test_rollback_success(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test POST /api/registry/deployments:rollback success."""
    mock_service.rollback_deployment_result = {
        "ok": True,
        "target": "ml_signal_actor",
        "model_id": "model_old",
    }

    resp = client.post(
        "/api/registry/deployments:rollback",
        json={"target": "ml_signal_actor", "to_model_id": "model_old"},
    )
    assert resp.status_code == 202
    data = resp.get_json()
    assert data["ok"] is True
    assert mock_service.rollback_deployment_calls == [
        {"target": "ml_signal_actor", "to_model_id": "model_old"}
    ]


def test_rollback_missing_target(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test rollback returns 400 when target missing."""
    resp = client.post(
        "/api/registry/deployments:rollback",
        json={"to_model_id": "model_old"},
    )
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "invalid_params"


def test_rollback_missing_model_id(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test rollback returns 400 when to_model_id missing."""
    resp = client.post(
        "/api/registry/deployments:rollback",
        json={"target": "ml_signal_actor"},
    )
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "invalid_params"


def test_rollback_invalid_params(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test rollback returns 400 when both params empty."""
    resp = client.post(
        "/api/registry/deployments:rollback",
        json={"target": "", "to_model_id": ""},
    )
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "invalid_params"


def test_rollback_unauthorized(
    client: FlaskClient, app: Flask, mock_service: MockDashboardService
) -> None:
    """Test rollback returns 401 when unauthorized."""
    app.config["_token_required"] = True
    app.config["_token_valid"] = False

    resp = client.post(
        "/api/registry/deployments:rollback",
        json={"target": "ml_signal_actor", "to_model_id": "model_old"},
    )
    assert resp.status_code == 401


def test_rollback_failure_returns_200(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Test rollback failure returns 200 (not 202)."""
    mock_service.rollback_deployment_result = {
        "ok": False,
        "target": "ml_signal_actor",
        "model_id": "model_old",
    }

    resp = client.post(
        "/api/registry/deployments:rollback",
        json={"target": "ml_signal_actor", "to_model_id": "model_old"},
    )
    assert resp.status_code == 200
    assert resp.get_json()["ok"] is False


# =============================================================================
# Service delegation tests
# =============================================================================


def test_all_endpoints_delegate_to_service(
    client: FlaskClient, mock_service: MockDashboardService
) -> None:
    """Verify all endpoints properly delegate to the service."""
    # Setup all mock results
    mock_service.list_models_result = [{"model_id": "m1"}]
    mock_service.get_model_performance_history_result = [{"ts": 1}]
    mock_service.list_deployments_result = {"target": ["m1"]}
    mock_service.list_features_result = [{"feature_set_id": "f1"}]
    mock_service.list_strategies_result = [{"strategy_id": "s1"}]
    mock_service.list_datasets_result = [{"dataset_id": "d1"}]
    mock_service.get_strategy_details_result = {"strategy_id": "s1"}
    mock_service.check_strategy_compatibility_result = {"compatible": True}
    mock_service.get_feature_lineage_result = [{"feature_set_id": "f1"}]
    mock_service.list_watermarks_result = [{"dataset_id": "ds"}]
    mock_service.list_dataset_lineage_result = [{"transform_id": "t1"}]
    mock_service.promote_feature_result = {"ok": True}
    mock_service.deprecate_feature_result = {"ok": True}
    mock_service.deploy_model_result = {"ok": True}
    mock_service.hot_reload_model_result = {"ok": True}
    mock_service.rollback_deployment_result = {"ok": True}

    # Test all GET endpoints
    assert client.get("/api/registry/models").status_code == 200
    assert client.get("/api/registry/models/m1/history").status_code == 200
    assert client.get("/api/registry/deployments").status_code == 200
    assert client.get("/api/registry/features").status_code == 200
    assert client.get("/api/registry/strategies").status_code == 200
    assert client.get("/api/registry/datasets").status_code == 200
    assert client.get("/api/registry/strategies/s1").status_code == 200
    assert client.get("/api/registry/strategies/s1/compatibility").status_code == 200
    assert client.get("/api/registry/features/f1/lineage").status_code == 200
    assert client.get("/api/registry/datasets/watermarks?dataset_id=ds").status_code == 200
    assert client.get("/api/registry/datasets/lineage").status_code == 200

    # Test all POST endpoints
    assert client.post("/api/registry/features/f1:promote", json={}).status_code == 202
    assert client.post("/api/registry/features/f1:deprecate", json={}).status_code == 202
    assert client.post("/api/registry/models/m1:deploy", json={}).status_code == 202
    assert client.post("/api/registry/models/m1:hot_reload", json={}).status_code == 202
    assert client.post(
        "/api/registry/deployments:rollback",
        json={"target": "t", "to_model_id": "m"},
    ).status_code == 202
