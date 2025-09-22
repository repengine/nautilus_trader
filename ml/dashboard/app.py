"""
Flask app factory for the Dashboard API.
"""

from __future__ import annotations

from typing import Any, cast

from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request

from ml.common.logging_config import bind_log_context
from ml.common.logging_config import configure_logging
from ml.common.metrics_export import CONTENT_TYPE_LATEST
from ml.common.metrics_export import generate_latest
from ml.dashboard.config import DashboardConfig
from ml.dashboard.service import DashboardService


def create_app(config: DashboardConfig | None = None) -> Flask:
    """
    Create a Flask application exposing the dashboard API.
    """
    app = Flask(__name__)
    # Light, idempotent logging configuration for API usage
    try:
        configure_logging()
        bind_log_context(component="ml.dashboard.api")
    except Exception:  # pragma: no cover - defensive
        ...
    cfg = config or DashboardConfig.from_env()
    svc = DashboardService.from_config(cfg)
    if cfg.events_poll_interval_seconds > 0.0:
        svc.start_event_polling(cfg.events_poll_interval_seconds)

    def _require_token() -> bool:
        """Return True if dashboard token requirement is satisfied or disabled."""
        import os

        required = os.getenv("ML_DASHBOARD_TOKEN")
        if not required:
            return True
        provided = request.headers.get("X-ML-DASHBOARD-TOKEN")
        return bool(provided and provided == required)

    @app.get("/api/health/system")
    def health_system() -> tuple[Any, int]:
        data = svc.get_system_health()
        return jsonify(data), 200

    @app.get("/api/services")
    def services_list() -> tuple[Any, int]:
        data = svc.list_services()
        return jsonify(data), 200

    @app.post("/api/services/<name>:action")
    def services_action(name: str) -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        action = str(payload.get("action", "")).strip().lower()
        if action not in {"start", "stop", "restart"}:
            return jsonify({"error": "invalid_action"}), 400
        res = svc.control_service(name, action)
        # 202 Accepted for async/semi-async operations
        return jsonify(res), 202 if res.get("ok") else 200

    @app.post("/api/pipeline/run")
    def pipeline_run() -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        mode = str(payload.get("mode", "")).strip().lower()
        if mode not in {"daily", "backfill", "realtime"}:
            return jsonify({"error": "invalid_mode"}), 400
        res = svc.trigger_pipeline(mode, payload)
        return jsonify(res), 202 if res.get("ok") else 200

    @app.get("/api/registry/models")
    def registry_models() -> tuple[Any, int]:
        return jsonify(svc.list_models()), 200

    @app.get("/api/registry/deployments")
    def registry_deployments() -> tuple[Any, int]:
        return jsonify(svc.list_deployments()), 200

    @app.get("/api/registry/features")
    def registry_features() -> tuple[Any, int]:
        role = request.args.get("role")
        stage = request.args.get("stage")
        return jsonify(svc.list_features(role=role or None, stage=stage or None)), 200

    @app.get("/api/registry/strategies")
    def registry_strategies() -> tuple[Any, int]:
        return jsonify(svc.list_strategies()), 200

    @app.get("/api/registry/datasets")
    def registry_datasets() -> tuple[Any, int]:
        return jsonify(svc.list_datasets()), 200

    @app.get("/api/registry/strategies/<strategy_id>")
    def strategy_details(strategy_id: str) -> tuple[Any, int]:
        data = svc.get_strategy_details(strategy_id)
        if data is None:
            return jsonify({}), 200
        return jsonify(data), 200

    @app.get("/api/registry/strategies/<strategy_id>/compatibility")
    def strategy_compatibility(strategy_id: str) -> tuple[Any, int]:
        active_raw = request.args.get("active", "")
        active = [s.strip() for s in active_raw.split(",") if s.strip()]
        return jsonify(svc.check_strategy_compatibility(strategy_id, active)), 200

    @app.get("/api/registry/features/<feature_set_id>/lineage")
    def feature_lineage(feature_set_id: str) -> tuple[Any, int]:
        return jsonify(svc.get_feature_lineage(feature_set_id)), 200

    @app.get("/api/registry/datasets/watermarks")
    def dataset_watermarks() -> tuple[Any, int]:
        ds = request.args.get("dataset_id")
        if not ds:
            return jsonify([]), 200
        instr = request.args.get("instrument")
        source = request.args.get("source")
        try:
            limit = int(str(request.args.get("limit", "100")))
        except Exception:
            limit = 100
        return (
            jsonify(svc.list_watermarks(dataset_id=ds, instrument=instr or None, source=source or None, limit=limit)),
            200,
        )

    @app.get("/api/registry/datasets/lineage")
    def dataset_lineage() -> tuple[Any, int]:
        child = request.args.get("child") or None
        parent = request.args.get("parent") or None
        try:
            limit = int(str(request.args.get("limit", "100")))
        except Exception:
            limit = 100
        return jsonify(svc.list_dataset_lineage(child=child, parent=parent, limit=limit)), 200

    @app.post("/api/registry/features/<feature_set_id>:promote")
    def registry_feature_promote(feature_set_id: str) -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        stage = str(payload.get("stage") or "") or None
        gates = payload.get("gates")
        if gates is not None and not isinstance(gates, list):
            return jsonify({"error": "invalid_gates"}), 400
        res = svc.promote_feature(feature_set_id, stage=stage, gates=cast(list[dict[str, Any]] | None, gates))
        return jsonify(res), 202 if res.get("ok") else 200

    @app.post("/api/registry/features/<feature_set_id>:deprecate")
    def registry_feature_deprecate(feature_set_id: str) -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        reason = str(payload.get("reason") or "") or None
        res = svc.deprecate_feature(feature_set_id, reason=reason)
        return jsonify(res), 202 if res.get("ok") else 200

    @app.post("/api/registry/models/<model_id>:deploy")
    def registry_deploy(model_id: str) -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        target = str(payload.get("target", "ml_signal_actor")).strip()
        if not target:
            return jsonify({"error": "invalid_target"}), 400
        res = svc.deploy_model(model_id, target)
        return jsonify(res), 202 if res.get("ok") else 200

    @app.post("/api/registry/models/<model_id>:hot_reload")
    def registry_hot_reload(model_id: str) -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        target = str(payload.get("target", "ml_signal_actor")).strip()
        if not target:
            return jsonify({"error": "invalid_target"}), 400
        res = svc.hot_reload_model(target=target, new_model_id=model_id)
        return jsonify(res), 202 if res.get("ok") else 200

    @app.post("/api/registry/deployments:rollback")
    def registry_rollback() -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        target = str(payload.get("target", "")).strip()
        to_model_id = str(payload.get("to_model_id", "")).strip()
        if not target or not to_model_id:
            return jsonify({"error": "invalid_params"}), 400
        res = svc.rollback_deployment(target=target, to_model_id=to_model_id)
        return jsonify(res), 202 if res.get("ok") else 200

    @app.get("/api/events")
    def events_list() -> tuple[Any, int]:
        args = request.args
        try:
            limit = int(str(args.get("limit", "100")))
        except Exception:
            limit = 100
        stage = str(args.get("stage", "")) or None
        source = str(args.get("source", "")) or None
        instrument = str(args.get("instrument", "")) or None
        data = svc.list_events(limit=limit, stage=stage, source=source, instrument_substr=instrument)
        return jsonify(data), 200

    @app.post("/api/observability/grafana/provision")
    def grafana_provision() -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        title = str(payload.get("title") or "") or None
        res = svc.provision_grafana_dashboard(title=title)
        return jsonify(res), 202 if res.get("ok") else 200

    @app.get("/health")
    def health() -> tuple[Any, int]:  # pragma: no cover - simple readiness
        return jsonify({"healthy": True}), 200

    @app.get("/metrics")
    def metrics() -> tuple[bytes, int, dict[str, str]]:  # pragma: no cover - passthrough
        return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

    @app.get("/")
    def index() -> tuple[str, int]:
        # Minimal visual UI for local/dev usage
        return render_template("index.html"), 200

    return app


__all__ = ["create_app"]
