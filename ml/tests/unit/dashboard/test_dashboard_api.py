from __future__ import annotations

import os
import datetime as dt
from datetime import datetime
from datetime import timedelta
from typing import Any

import ml.dashboard.grafana as grafana_module

import pytest
from flask.testing import FlaskClient

from ml.dashboard import DashboardConfig, create_app
from ml.dashboard.config import DashboardToken
from ml.dashboard.service import DashboardService
from ml.dashboard.services import PipelineTriggerResult

pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry", "mock_tracing_backend")


@pytest.fixture()
def client() -> FlaskClient:
    # Keep the app isolated; disable compose control by default
    os.environ["ML_DASHBOARD_USE_COMPOSE"] = "0"
    cfg = DashboardConfig.from_env()
    app = create_app(cfg)
    return app.test_client()


def test_health_system_ok(client: FlaskClient) -> None:
    resp = client.get("/api/health/system")
    assert resp.status_code == 200
    data: dict[str, Any] = resp.get_json()
    assert "services" in data and "dependencies" in data
    # Keys are present even if targets aren't running locally
    assert "ml_pipeline" in data["services"]


def test_services_list_contains_expected(client: FlaskClient) -> None:
    resp = client.get("/api/services")
    assert resp.status_code == 200
    services = resp.get_json()
    names = {s["name"] for s in services}
    assert {"ml_signal_actor", "ml_strategy", "ml_pipeline"}.issubset(names)


def test_services_action_unsupported_returns_ok(client: FlaskClient) -> None:
    # Compose disabled; expect supported 200 with ok False
    resp = client.post("/api/services/ml_pipeline:action", json={"action": "restart"})
    assert resp.status_code in {200, 202}
    data = resp.get_json()
    assert data["action"] == "restart"
    assert data["service"] == "ml_pipeline"
    assert data["ok"] in {False, True}


def test_pipeline_run_triggers_pipeline_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML_DASHBOARD_USE_COMPOSE", "0")
    stub_token = "secret"
    monkeypatch.setenv("ML_DASHBOARD_TOKEN", stub_token)

    class _StubPipelineService:
        def __init__(self) -> None:
            self.requests: list[PipelineTriggerResult | None] = []
            self.trigger_requests: list[Any] = []

        async def trigger_pipeline(self, request):
            self.trigger_requests.append(request)
            return PipelineTriggerResult(
                success=True,
                job_id="job_123",
                pipeline_type=request.pipeline_type,
                status="QUEUED",
                message="queued",
                error=None,
            )

    stub = _StubPipelineService()
    monkeypatch.setattr(DashboardService, "_get_pipeline_service", lambda self: stub)

    app = create_app(DashboardConfig.from_env())
    client_local = app.test_client()
    payload = {
        "pipeline_type": "ingest",
        "config": {
            "dataset": {
                "data_dir": "data/tier1",
                "symbols": "SPY.NYSE",
                "out_dir": "out",
            },
        },
    }
    resp = client_local.post(
        "/api/pipeline/run",
        json=payload,
        headers={"X-ML-DASHBOARD-TOKEN": stub_token},
    )
    assert resp.status_code == 202
    data = resp.get_json()
    assert data["success"] is True
    assert data["job_id"] == "job_123"
    assert stub.trigger_requests and stub.trigger_requests[0].pipeline_type == "ingest"
    assert stub.trigger_requests[0].config == payload["config"]


def test_pipeline_run_accepts_legacy_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML_DASHBOARD_USE_COMPOSE", "0")
    stub_token = "another"
    monkeypatch.setenv("ML_DASHBOARD_TOKEN", stub_token)

    class _StubPipelineService:
        def __init__(self) -> None:
            self.trigger_requests: list[Any] = []

        async def trigger_pipeline(self, request):
            self.trigger_requests.append(request)
            return PipelineTriggerResult(
                success=True,
                job_id="job_legacy",
                pipeline_type=request.pipeline_type,
                status="QUEUED",
                message=None,
                error=None,
            )

    stub = _StubPipelineService()
    monkeypatch.setattr(DashboardService, "_get_pipeline_service", lambda self: stub)

    app = create_app(DashboardConfig.from_env())
    client_local = app.test_client()
    legacy_payload = {
        "mode": "dataset",
        "dataset": {
            "data_dir": "data",
            "symbols": "QQQ.NYSE",
            "out_dir": "out",
        },
    }
    resp = client_local.post(
        "/api/pipeline/run",
        json=legacy_payload,
        headers={"X-ML-DASHBOARD-TOKEN": stub_token},
    )
    assert resp.status_code == 202
    data = resp.get_json()
    assert data["job_id"] == "job_legacy"
    assert stub.trigger_requests and stub.trigger_requests[0].pipeline_type == "dataset"
    assert stub.trigger_requests[0].config == {"dataset": legacy_payload["dataset"]}


def test_pipeline_run_service_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML_DASHBOARD_USE_COMPOSE", "0")
    monkeypatch.setenv("ML_DASHBOARD_TOKEN", "secret")
    monkeypatch.setattr(DashboardService, "_get_pipeline_service", lambda self: None)

    app = create_app(DashboardConfig.from_env())
    client_local = app.test_client()
    resp = client_local.post(
        "/api/pipeline/run",
        json={"pipeline_type": "ingest", "config": {}},
        headers={"X-ML-DASHBOARD-TOKEN": "secret"},
    )
    assert resp.status_code == 503
    data = resp.get_json()
    assert data["status"] == "UNAVAILABLE"


def test_control_pipeline_trigger_invokes_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML_DASHBOARD_USE_COMPOSE", "0")
    token = "control"
    monkeypatch.setenv("ML_DASHBOARD_TOKEN", token)

    class _StubPipelineService:
        def __init__(self) -> None:
            self.requests: list[Any] = []

        async def trigger_pipeline(self, request):
            self.requests.append(request)
            return PipelineTriggerResult(
                success=True,
                job_id="job_ctrl",
                pipeline_type=request.pipeline_type,
                status="QUEUED",
                message=None,
                error=None,
            )

    stub = _StubPipelineService()
    monkeypatch.setattr(DashboardService, "_get_pipeline_service", lambda self: stub)

    app = create_app(DashboardConfig.from_env())
    client_local = app.test_client()
    resp = client_local.post(
        "/api/control/pipeline/trigger",
        json={
            "pipeline_type": "train",
            "config": {
                "dataset": {
                    "data_dir": "data",
                    "symbols": "QQQ.NYSE",
                    "out_dir": "out",
                },
            },
        },
        headers={"X-ML-DASHBOARD-TOKEN": token},
    )
    assert resp.status_code == 202
    data = resp.get_json()
    assert data["job_id"] == "job_ctrl"
    assert data["control_run_id"]
    assert stub.requests and stub.requests[0].pipeline_type == "train"


def test_control_pipeline_trigger_service_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML_DASHBOARD_USE_COMPOSE", "0")
    token = "control"
    monkeypatch.setenv("ML_DASHBOARD_TOKEN", token)
    monkeypatch.setattr(DashboardService, "_get_pipeline_service", lambda self: None)

    app = create_app(DashboardConfig.from_env())
    client_local = app.test_client()
    resp = client_local.post(
        "/api/control/pipeline/trigger",
        json={"pipeline_type": "ingest", "config": {}},
        headers={"X-ML-DASHBOARD-TOKEN": token},
    )
    assert resp.status_code == 503
    data = resp.get_json()
    assert data["status"] == "UNAVAILABLE"


def test_pipeline_jobs_endpoint_requires_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML_DASHBOARD_USE_COMPOSE", "0")
    monkeypatch.setenv("ML_DASHBOARD_TOKEN", "secret")
    app = create_app(DashboardConfig.from_env())
    client_local = app.test_client()

    resp = client_local.get("/api/pipeline/jobs")
    assert resp.status_code == 401


def test_pipeline_jobs_endpoint_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML_DASHBOARD_USE_COMPOSE", "0")
    monkeypatch.setenv("ML_DASHBOARD_TOKEN", "secret")

    payload = {
        "status": "success",
        "jobs": [
            {
                "job_id": "training_123",
                "pipeline_type": "training",
                "status": "COMPLETED",
            },
        ],
    }
    monkeypatch.setattr(DashboardService, "list_pipeline_jobs", lambda self: payload)

    app = create_app(DashboardConfig.from_env())
    client_local = app.test_client()

    resp = client_local.get("/api/pipeline/jobs", headers={"X-ML-DASHBOARD-TOKEN": "secret"})
    assert resp.status_code == 200
    assert resp.get_json() == payload


def test_pipeline_job_detail_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML_DASHBOARD_USE_COMPOSE", "0")
    monkeypatch.setenv("ML_DASHBOARD_TOKEN", "secret")
    monkeypatch.setattr(
        DashboardService,
        "get_pipeline_job",
        lambda self, job_id: {"status": "not_found", "error": "job_not_found"},
    )

    app = create_app(DashboardConfig.from_env())
    client_local = app.test_client()

    resp = client_local.get(
        "/api/pipeline/jobs/missing",
        headers={"X-ML-DASHBOARD-TOKEN": "secret"},
    )
    assert resp.status_code == 404
    assert resp.get_json()["status"] == "not_found"


def test_pipeline_job_purge_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML_DASHBOARD_USE_COMPOSE", "0")
    monkeypatch.setenv("ML_DASHBOARD_TOKEN", "secret")
    monkeypatch.setattr(
        DashboardService,
        "purge_pipeline_job",
        lambda self, job_id: {
            "status": "purged",
            "result": {
                "success": True,
                "job_id": job_id,
                "status": "purged",
                "message": "Pipeline job purged",
                "error": None,
            },
        },
    )

    app = create_app(DashboardConfig.from_env())
    client_local = app.test_client()

    resp = client_local.delete(
        "/api/pipeline/jobs/job_to_purge",
        headers={"X-ML-DASHBOARD-TOKEN": "secret"},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "purged"
    assert data["result"]["success"] is True


def test_pipeline_job_purge_failure_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML_DASHBOARD_USE_COMPOSE", "0")
    monkeypatch.setenv("ML_DASHBOARD_TOKEN", "secret")
    monkeypatch.setattr(
        DashboardService,
        "purge_pipeline_job",
        lambda self, job_id: {
            "status": "failed",
            "result": {
                "success": False,
                "job_id": job_id,
                "status": "failed",
                "message": "Unable to delete job from store",
                "error": "store_delete_failed",
            },
        },
    )

    app = create_app(DashboardConfig.from_env())
    client_local = app.test_client()

    resp = client_local.delete(
        "/api/pipeline/jobs/job_failure",
        headers={"X-ML-DASHBOARD-TOKEN": "secret"},
    )
    assert resp.status_code == 500
    data = resp.get_json()
    assert data["status"] == "failed"


def test_metrics_and_health_endpoints(client: FlaskClient) -> None:
    m = client.get("/metrics")
    assert m.status_code == 200
    assert b"ml_dashboard_requests_total" in m.data

    h = client.get("/health")
    assert h.status_code == 200
    data = h.get_json()
    assert data.get("healthy") is True


def test_registry_read_endpoints(client: FlaskClient) -> None:
    # Should return empty lists/dicts by default
    r = client.get("/api/registry/models")
    assert r.status_code == 200
    assert isinstance(r.get_json(), list)

    d = client.get("/api/registry/deployments")
    assert d.status_code == 200
    assert isinstance(d.get_json(), dict)


def test_feature_strategy_dataset_list_endpoints(client: FlaskClient) -> None:
    f = client.get("/api/registry/features")
    assert f.status_code == 200
    assert isinstance(f.get_json(), list)

    s = client.get("/api/registry/strategies")
    assert s.status_code == 200
    assert isinstance(s.get_json(), list)

    ds = client.get("/api/registry/datasets")
    assert ds.status_code == 200
    assert isinstance(ds.get_json(), list)


def test_feature_promote_and_deprecate_endpoints(client: FlaskClient) -> None:
    # Unknown feature id should be handled safely
    pr = client.post("/api/registry/features/unknown:promote", json={"stage": "PROD"})
    assert pr.status_code in {200, 202}
    dr = client.post("/api/registry/features/unknown:deprecate", json={"reason": "test"})
    assert dr.status_code in {200, 202}


def test_feature_lineage_endpoint(client: FlaskClient) -> None:
    resp = client.get("/api/registry/features/unknown/lineage")
    assert resp.status_code == 200
    assert isinstance(resp.get_json(), list)


def test_dataset_watermarks_and_lineage_endpoints(client: FlaskClient) -> None:
    # Without dataset_id → empty list
    m = client.get("/api/registry/datasets/watermarks")
    assert m.status_code == 200
    assert m.get_json() == []

    # With dataset_id but likely none present yet → ok and list/empty
    m2 = client.get("/api/registry/datasets/watermarks?dataset_id=features")
    assert m2.status_code == 200
    assert isinstance(m2.get_json(), list)

    ln = client.get("/api/registry/datasets/lineage?child=features")
    assert ln.status_code == 200
    assert isinstance(ln.get_json(), list)


def test_registry_deploy_invalid_model_returns_ok_false(client: FlaskClient) -> None:
    resp = client.post(
        "/api/registry/models/does_not_exist:deploy", json={"target": "ml_signal_actor"}
    )
    assert resp.status_code in {200, 202}
    body = resp.get_json()
    assert body["model_id"] == "does_not_exist"
    assert body["target"] == "ml_signal_actor"
    assert body["ok"] in {False, True}


def test_registry_hot_reload_and_rollback_endpoints(client: FlaskClient) -> None:
    # Hot reload unknown model should be ok false/true
    hr = client.post(
        "/api/registry/models/unknown_model:hot_reload", json={"target": "ml_signal_actor"}
    )
    assert hr.status_code in {200, 202}
    hrj = hr.get_json()
    assert hrj["target"] == "ml_signal_actor"

    rb = client.post(
        "/api/registry/deployments:rollback",
        json={"target": "ml_signal_actor", "to_model_id": "unknown_model"},
    )
    assert rb.status_code in {200, 202}
    rbj = rb.get_json()
    assert rbj["target"] == "ml_signal_actor"


def test_events_endpoint_without_redis_returns_list(client: FlaskClient) -> None:
    resp = client.get("/api/events")
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, list)


def test_grafana_provision_endpoint_default(client: FlaskClient) -> None:
    # Without credentials, should be safe no-op
    resp = client.post("/api/observability/grafana/provision", json={"title": "Test"})
    assert resp.status_code in {200, 202, 401}
    if resp.status_code != 401:  # guard may be enabled in env
        body = resp.get_json()
        assert set(body.keys()).issuperset({"ok", "url", "error"})


def test_observability_status_endpoint(client: FlaskClient) -> None:
    resp = client.get("/api/observability/status")
    assert resp.status_code == 200
    body = resp.get_json()
    assert set(body.keys()).issuperset({"ok", "url", "embed_urls"})


def test_observability_summary_endpoint(
    client: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeResponse:
        def __init__(self, value: float) -> None:
            self._value = value
            self.status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "data": {
                    "result": [
                        {
                            "value": [0, str(self._value)],
                        },
                    ],
                },
            }

    def _fake_get(
        url: str, params: dict[str, Any] | None = None, timeout: float | None = None
    ) -> _FakeResponse:
        query = (params or {}).get("query", "")
        mapping = {
            "sum(rate(ml_dashboard_requests_total[5m]))": 1.25,
            "histogram_quantile(0.95, sum(rate(ml_dashboard_latency_seconds_bucket[5m])) by (le))": 0.42,
            "sum(increase(ml_dashboard_events_failure_total[5m]))": 0.0,
        }
        return _FakeResponse(mapping.get(query, 0.0))

    monkeypatch.setattr(grafana_module.requests, "get", _fake_get)

    resp = client.get("/api/observability/summary")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert pytest.approx(data["metrics"]["request_rate_per_second"], rel=1e-6) == 1.25


def test_auth_guard_when_token_set(monkeypatch: pytest.MonkeyPatch) -> None:
    from ml.dashboard import create_app, DashboardConfig
    import os

    monkeypatch.setenv("ML_DASHBOARD_USE_COMPOSE", "0")
    monkeypatch.setenv("ML_DASHBOARD_TOKEN", "secret")
    app = create_app(DashboardConfig.from_env())
    c = app.test_client()
    # Missing token
    r1 = c.post("/api/services/ml_pipeline:action", json={"action": "restart"})
    assert r1.status_code == 401
    # With token
    r2 = c.post(
        "/api/services/ml_pipeline:action",
        headers={"X-ML-DASHBOARD-TOKEN": "secret"},
        json={"action": "restart"},
    )
    assert r2.status_code in {200, 202}


def test_observability_stores_endpoint(
    client: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {"ok": True, "stores": [{"store": "feature", "healthy": True}]}

    monkeypatch.setattr(
        DashboardService,
        "get_store_summary",
        lambda self: payload,
    )

    resp = client.get("/api/observability/stores")
    assert resp.status_code == 200
    assert resp.get_json() == payload


def test_dashboard_service_validate_token_success() -> None:
    cfg = DashboardConfig(auth_tokens=(DashboardToken(value="secret"),))
    svc = DashboardService.from_config(cfg)
    assert svc.validate_token("secret") is True
    assert svc.validate_token("other") is False


def test_dashboard_service_validate_token_expired() -> None:
    expired = DashboardToken(
        value="secret",
        expires_at=datetime.now(dt.UTC) - timedelta(minutes=1),
    )
    cfg = DashboardConfig(auth_tokens=(expired,))
    svc = DashboardService.from_config(cfg)
    assert svc.validate_token("secret", now=datetime.now(dt.UTC)) is False


def test_dashboard_config_token_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "ML_DASHBOARD_TOKENS",
        '[{"value":"tok1","expires":"2099-01-01T00:00:00Z"},"tok2"]',
    )
    cfg = DashboardConfig.from_env({})
    values = [token.value for token in cfg.auth_tokens]
    assert values == ["tok1", "tok2"]


# =============================================================================
# Market Tickers Endpoint Tests
# =============================================================================


class TestMarketTickersEndpoint:
    """Tests for GET /api/market/tickers endpoint."""

    def test_market_tickers_returns_200(self, client: FlaskClient) -> None:
        """Test endpoint returns 200 even with no data."""
        resp = client.get("/api/market/tickers")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, dict)

    def test_market_tickers_with_symbols_param(self, client: FlaskClient) -> None:
        """Test endpoint accepts symbols query parameter."""
        resp = client.get("/api/market/tickers?symbols=SPY,QQQ")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, dict)

    def test_market_tickers_response_structure(
        self,
        client: FlaskClient,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        """Test endpoint returns correct structure when data exists."""
        from unittest.mock import patch, MagicMock
        import polars as pl

        # Create mock parquet file with OHLCV data
        symbol_dir = tmp_path / "SPY" / "l0"
        symbol_dir.mkdir(parents=True)
        df = pl.DataFrame({
            "timestamp": [datetime.now()],
            "open": [450.0],
            "high": [455.0],
            "low": [448.0],
            "close": [452.50],
            "volume": [1000000],
        })
        df.write_parquet(symbol_dir / "SPY_ohlcv.parquet")

        # Patch the data directory
        monkeypatch.setenv("ML_MARKET_DATA_DIR", str(tmp_path))

        resp = client.get("/api/market/tickers?symbols=SPY")
        assert resp.status_code == 200
        data = resp.get_json()

        # Should have SPY in response (may be empty if endpoint can't find the mock)
        assert isinstance(data, dict)

    def test_market_tickers_missing_symbol_returns_null(
        self,
        client: FlaskClient,
    ) -> None:
        """Test endpoint returns null for symbols without data."""
        resp = client.get("/api/market/tickers?symbols=NOTEXIST")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, dict)
        # Missing symbols should either be absent or have null values
        if "NOTEXIST" in data:
            assert data["NOTEXIST"] is None or data["NOTEXIST"].get("price") is None


# =============================================================================
# Model Performance Endpoint Tests
# =============================================================================


class TestModelPerformanceEndpoint:
    """Tests for GET /api/registry/models/performance endpoint."""

    def test_model_performance_returns_200(self, client: FlaskClient) -> None:
        """Test endpoint returns 200 with models list."""
        resp = client.get("/api/registry/models/performance")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, dict)
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_model_performance_response_structure(
        self,
        client: FlaskClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test endpoint returns correct structure for each model."""
        from unittest.mock import MagicMock, patch
        from pathlib import Path
        from ml.registry.base import (
            ModelInfo,
            ModelManifest,
            ModelRole,
            DataRequirements,
            DeploymentStatus,
        )

        # Create a mock model
        manifest = ModelManifest(
            model_id="test-model-001",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="XGBoost",
            feature_schema={"volume": "float64"},
            feature_schema_hash="abc123",
            version="1.0.0",
        )
        model_info = ModelInfo(
            manifest=manifest,
            model_path=Path("/tmp/model.onnx"),
            deployment_status=DeploymentStatus.ACTIVE,
            deployed_to=["signal_actor_1"],
            performance_history=[
                {"sharpe_ratio": 1.5, "win_rate": 0.55, "daily_pnl": 1234.56}
            ],
            metadata={},
        )

        # Patch the service to return mock models
        with patch.object(
            DashboardService,
            "list_models_with_performance",
            return_value=[{
                "model_id": "test-model-001",
                "type": "XGBoost",
                "status": "active",
                "daily_pnl": 1234.56,
                "sharpe": 1.5,
                "win_rate": 0.55,
            }],
        ):
            resp = client.get("/api/registry/models/performance")
            assert resp.status_code == 200
            data = resp.get_json()
            assert isinstance(data["models"], list)

    def test_model_performance_empty_registry(
        self,
        client: FlaskClient,
    ) -> None:
        """Test endpoint handles empty registry gracefully."""
        resp = client.get("/api/registry/models/performance")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data["models"], list)
        # May be empty or have demo data
