"""
Flask app factory for the Dashboard API.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
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


def _status_to_http(status: str, success_code: int = 200) -> int:
    normalized = status.lower()
    if normalized in {"success", "purged"}:
        return success_code
    if normalized == "not_found":
        return 404
    if normalized == "unavailable":
        return 503
    if normalized in {"failed", "error"}:
        return 500
    return 500


def create_app(config: DashboardConfig | None = None) -> Flask:
    """
    Create a Flask application exposing the dashboard API.
    """
    # Configure Flask with static file serving
    static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    app = Flask(__name__, static_folder=static_folder, static_url_path="/static")
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

    @app.teardown_appcontext
    def _shutdown(_: object | None) -> None:  # pragma: no cover - teardown path
        svc.stop_event_polling()

    def _require_token() -> bool:
        """
        Return True when dashboard authentication (if enabled) is satisfied.
        """
        provided = request.headers.get("X-ML-DASHBOARD-TOKEN")
        if not provided:
            auth_header = request.headers.get("Authorization") or ""
            if auth_header.lower().startswith("bearer "):
                provided = auth_header[7:].strip() or None
        return svc.validate_token(provided)

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
        pipeline_type_raw = payload.get("pipeline_type") or payload.get("mode") or "full"
        pipeline_type = str(pipeline_type_raw).strip() or "full"

        config_payload = payload.get("config")
        if isinstance(config_payload, Mapping):
            config_dict: Mapping[str, Any] = dict(config_payload)
        else:
            config_dict = {
                key: value
                for key, value in payload.items()
                if key not in {"pipeline_type", "mode", "config"}
            }

        res = svc.trigger_pipeline(pipeline_type, config_dict)
        status_token = str(res.get("status", "ERROR")).upper()
        if status_token == "QUEUED" and res.get("success"):
            status_code = 202
        elif status_token == "UNAVAILABLE":
            status_code = 503
        elif status_token == "INVALID":
            status_code = 400
        elif res.get("success"):
            status_code = 202
        else:
            status_code = 500
        return jsonify(res), status_code

    @app.get("/api/pipeline/jobs")
    def pipeline_jobs_list() -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        res = svc.list_pipeline_jobs()
        status = res.get("status", "error")
        code = _status_to_http(status)
        return jsonify(res), code

    @app.get("/api/pipeline/jobs/<job_id>")
    def pipeline_job_detail(job_id: str) -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        res = svc.get_pipeline_job(job_id)
        status = res.get("status", "error")
        code = _status_to_http(status)
        return jsonify(res), code

    @app.delete("/api/pipeline/jobs/<job_id>")
    def pipeline_job_purge(job_id: str) -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        res = svc.purge_pipeline_job(job_id)
        status = res.get("status", "error")
        code = _status_to_http(status)
        return jsonify(res), code

    @app.post("/api/orchestrator/<task>")
    def orchestrator_task(task: str) -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        res = svc.trigger_orchestrator_task(task, payload)
        return jsonify(res), 202 if res.get("ok") else 200

    @app.get("/api/registry/models")
    def registry_models() -> tuple[Any, int]:
        return jsonify(svc.list_models()), 200

    @app.get("/api/registry/models/<model_id>/history")
    def registry_model_history(model_id: str) -> tuple[Any, int]:
        limit_raw = request.args.get("limit")
        limit_value: int | None = None
        if limit_raw:
            try:
                limit_value = int(limit_raw)
                if limit_value < 0:
                    raise ValueError
            except Exception:
                return jsonify({"error": "invalid_limit"}), 400
        data = svc.get_model_performance_history(model_id, limit=limit_value)
        return jsonify(data), 200

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
            jsonify(
                svc.list_watermarks(
                    dataset_id=ds, instrument=instr or None, source=source or None, limit=limit
                )
            ),
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
        res = svc.promote_feature(
            feature_set_id, stage=stage, gates=cast(list[dict[str, Any]] | None, gates)
        )
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
        data = svc.list_events(
            limit=limit, stage=stage, source=source, instrument_substr=instrument
        )
        return jsonify(data), 200

    # ===== Control Panel Endpoints =====

    @app.post("/api/control/actors/start")
    def control_start_actor() -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        actor_id = str(payload.get("actor_id", "")).strip()
        actor_type = str(payload.get("actor_type", "signal")).strip()
        config = payload.get("config", {})
        if not actor_id:
            return jsonify({"error": "invalid_actor_id"}), 400

        from ml.dashboard.control_simple import SimpleControlPanel

        control_panel = SimpleControlPanel.from_env()
        result = control_panel.start_actor(actor_id, actor_type, config)
        return jsonify(result), 202 if result.get("success") else 400

    @app.post("/api/control/actors/stop")
    def control_stop_actor() -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        actor_id = str(payload.get("actor_id", "")).strip()
        if not actor_id:
            return jsonify({"error": "invalid_actor_id"}), 400

        from ml.dashboard.control_simple import SimpleControlPanel

        control_panel = SimpleControlPanel.from_env()
        result = control_panel.stop_actor(actor_id)
        return jsonify(result), 200

    @app.post("/api/control/pipeline/trigger")
    def control_trigger_pipeline() -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        pipeline_type_raw = payload.get("pipeline_type") or payload.get("mode") or "full"
        pipeline_type = str(pipeline_type_raw).strip() or "full"

        config_payload = payload.get("config")
        if isinstance(config_payload, Mapping):
            config_dict: Mapping[str, Any] = dict(config_payload)
        else:
            config_dict = {
                key: value
                for key, value in payload.items()
                if key not in {"pipeline_type", "mode", "config"}
            }

        pipeline_result = svc.trigger_pipeline(pipeline_type, config_dict)

        from ml.dashboard.control_simple import SimpleControlPanel

        control_panel = SimpleControlPanel.from_env()
        control_state = control_panel.trigger_pipeline(
            pipeline_type,
            config_dict,
            job_id=str(pipeline_result.get("job_id")) if pipeline_result.get("job_id") else None,
            status=str(pipeline_result.get("status", "QUEUED")).lower(),
        )

        response_payload = {
            **pipeline_result,
            "control_run_id": control_state["run_id"],
            "control_status": control_state["status"],
        }

        status_token = str(pipeline_result.get("status", "ERROR")).upper()
        if status_token == "QUEUED" and pipeline_result.get("success"):
            status_code = 202
        elif status_token == "UNAVAILABLE":
            status_code = 503
        elif status_token == "INVALID":
            status_code = 400
        elif pipeline_result.get("success"):
            status_code = 202
        else:
            status_code = 500

        return jsonify(response_payload), status_code

    @app.post("/api/control/ingestion/start")
    def control_start_ingestion() -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        symbols = payload.get("symbols", [])
        source = str(payload.get("source", "databento")).strip()
        if not symbols:
            return jsonify({"error": "no_symbols"}), 400

        from ml.dashboard.control_simple import SimpleControlPanel

        control_panel = SimpleControlPanel.from_env()
        result = control_panel.start_ingestion(symbols, source)
        return jsonify(result), 202 if result.get("success") else 400

    @app.post("/api/control/ingestion/backfill")
    def control_backfill() -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        symbols = payload.get("symbols", [])
        start_date = payload.get("start_date")
        end_date = payload.get("end_date")
        if not symbols or not start_date or not end_date:
            return jsonify({"error": "missing_params"}), 400

        from datetime import datetime

        from ml.dashboard.control_panel import DashboardControlPanel

        control_panel = DashboardControlPanel.from_env()
        import asyncio

        result = asyncio.run(
            control_panel.trigger_backfill(
                symbols,
                datetime.fromisoformat(start_date),
                datetime.fromisoformat(end_date),
            ),
        )
        return jsonify(result), 202 if result.get("success") else 400

    @app.post("/api/control/emergency/stop")
    def control_emergency_stop() -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401

        from ml.dashboard.control_simple import SimpleControlPanel

        control_panel = SimpleControlPanel.from_env()
        result = control_panel.emergency_stop_all()
        return jsonify(result), 200

    @app.get("/api/control/status")
    def control_system_status() -> tuple[Any, int]:
        from ml.dashboard.control_simple import SimpleControlPanel

        control_panel = SimpleControlPanel.from_env()
        status = control_panel.get_system_status()
        return jsonify(status), 200

    @app.post("/api/observability/grafana/provision")
    def grafana_provision() -> tuple[Any, int]:
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        title = str(payload.get("title") or "") or None
        res = svc.provision_grafana_dashboard(title=title)
        return jsonify(res), 202 if res.get("ok") else 200

    @app.get("/api/observability/status")
    def observability_status() -> tuple[Any, int]:
        data = svc.get_grafana_status()
        return jsonify(data), 200

    @app.get("/api/observability/summary")
    def observability_summary() -> tuple[Any, int]:
        data = svc.get_prometheus_summary()
        return jsonify(data), 200

    @app.get("/api/observability/stores")
    def observability_stores() -> tuple[Any, int]:
        data = svc.get_store_summary()
        return jsonify(data), 200

    @app.get("/health")
    def health() -> tuple[Any, int]:  # pragma: no cover - simple readiness
        return jsonify({"healthy": True}), 200

    @app.get("/metrics")
    def metrics() -> tuple[bytes, int, dict[str, str]]:  # pragma: no cover - passthrough
        return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

    @app.get("/")
    def index() -> tuple[str, int]:
        # Check for UI preference via query param or cookie
        ui_type = request.args.get("ui") or request.cookies.get("ui_preference") or "standard"

        # Map UI types to template files
        template_map = {
            "unified": "index_unified.html",  # Unified control + advanced
            "control": "index_control.html",  # Control center only
            "enhanced": "index_enhanced.html",
            "advanced": "index_advanced.html",
            "standard": "index.html",
        }

        # Check if requested template exists, fallback to standard if not
        template = template_map.get(ui_type, "index.html")
        template_path = os.path.join(app.root_path, "templates", template)
        if not os.path.exists(template_path):
            template = "index.html"

        # Minimal visual UI for local/dev usage
        return (
            render_template(
                template,
                grafana_embed_enabled=cfg.grafana_embed_enabled,
                grafana_embed_urls=cfg.grafana_embed_urls(),
                grafana_dashboard_url=cfg.grafana_dashboard_url(),
                grafana_theme=cfg.grafana_embed_theme,
            ),
            200,
        )

    return app


__all__ = ["create_app"]
