"""
Dashboard app facade with blueprint-based route registration.
"""

from __future__ import annotations

import logging
import os
from typing import Any, cast

from flask import Blueprint
from flask import Flask
from flask import jsonify
from flask import request

from ml.common.logging_config import bind_log_context
from ml.common.logging_config import configure_logging
from ml.common.metrics_export import CONTENT_TYPE_LATEST
from ml.common.metrics_export import generate_latest
from ml.dashboard.blueprints import actors_bp as actors_bp_template
from ml.dashboard.blueprints import control_bp as control_bp_template
from ml.dashboard.blueprints import features_bp as features_bp_template
from ml.dashboard.blueprints import metrics_bp as metrics_bp_template
from ml.dashboard.blueprints import pipeline_bp as pipeline_bp_template
from ml.dashboard.blueprints import register_actors_routes
from ml.dashboard.blueprints import register_control_routes
from ml.dashboard.blueprints import register_features_routes
from ml.dashboard.blueprints import register_metrics_routes
from ml.dashboard.blueprints import register_pipeline_routes
from ml.dashboard.blueprints import register_registry_routes
from ml.dashboard.blueprints import register_strategies_routes
from ml.dashboard.blueprints import register_trading_routes
from ml.dashboard.blueprints import registry_bp as registry_bp_template
from ml.dashboard.blueprints import strategies_bp as strategies_bp_template
from ml.dashboard.blueprints import trading_bp as trading_bp_template
from ml.dashboard.config import DashboardConfig
from ml.dashboard.service import DashboardService


logger = logging.getLogger(__name__)


def _clone_blueprint(template_bp: Blueprint) -> Blueprint:
    """Create a per-app blueprint instance based on the template."""
    return Blueprint(
        template_bp.name,
        __name__,
        url_prefix=template_bp.url_prefix,
    )


def create_app_facade(config: DashboardConfig | None = None) -> Flask:
    """
    Create a Flask application exposing dashboard APIs via blueprints.

    Args:
        config: Optional dashboard configuration override.

    Returns:
        Flask application instance.
    """
    static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    app = Flask(__name__, static_folder=static_folder, static_url_path="/static")

    try:
        configure_logging()
        bind_log_context(component="ml.dashboard.api")
    except Exception:  # pragma: no cover - defensive
        logger.debug("dashboard_logging_config_failed", exc_info=True)

    cfg = config or DashboardConfig.from_env()
    svc = DashboardService.from_config(cfg)

    if cfg.events_poll_interval_seconds > 0.0:
        svc.start_event_polling(cfg.events_poll_interval_seconds)

    @app.teardown_appcontext
    def _shutdown(_: object | None) -> None:  # pragma: no cover - teardown path
        svc.stop_event_polling()

    def _require_token() -> bool:
        """Return True when dashboard authentication (if enabled) is satisfied."""
        provided = request.headers.get("X-ML-DASHBOARD-TOKEN")
        if not provided:
            auth_header = request.headers.get("Authorization") or ""
            if auth_header.lower().startswith("bearer "):
                provided = auth_header[7:].strip() or None
        return svc.validate_token(provided)

    # Register blueprint routes
    health_bp = Blueprint("health", __name__, url_prefix="/api")
    pipeline_bp = _clone_blueprint(pipeline_bp_template)
    registry_bp = _clone_blueprint(registry_bp_template)
    control_bp = _clone_blueprint(control_bp_template)
    metrics_bp = _clone_blueprint(metrics_bp_template)
    trading_bp = _clone_blueprint(trading_bp_template)
    actors_bp = _clone_blueprint(actors_bp_template)
    features_bp = _clone_blueprint(features_bp_template)
    strategies_bp = _clone_blueprint(strategies_bp_template)

    @health_bp.get("/health/system")
    def health_system() -> tuple[Any, int]:
        """Get system health status."""
        data = svc.get_system_health()
        return jsonify(data), 200

    @health_bp.get("/services")
    def services_list() -> tuple[Any, int]:
        """List registered services in a stable payload."""
        data = svc.list_services()
        return jsonify({"services": data}), 200

    @health_bp.post("/services/<name>:action")
    def services_action(name: str) -> tuple[Any, int]:
        """Control a service (start, stop, restart)."""
        if not _require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        action = str(payload.get("action", "")).strip().lower()
        if action not in {"start", "stop", "restart"}:
            return jsonify({"error": "invalid_action"}), 400
        res = svc.control_service(name, action)
        return jsonify(res), 202 if res.get("ok") else 200
    register_pipeline_routes(pipeline_bp, svc, _require_token)
    register_registry_routes(registry_bp, svc, _require_token)
    register_control_routes(control_bp, svc, _require_token)
    register_metrics_routes(metrics_bp, svc)
    register_trading_routes(trading_bp, svc, _require_token)
    register_actors_routes(actors_bp, svc, _require_token)
    register_features_routes(features_bp, svc, _require_token)
    register_strategies_routes(strategies_bp, svc, _require_token)

    app.register_blueprint(health_bp)
    app.register_blueprint(pipeline_bp)
    app.register_blueprint(registry_bp)
    app.register_blueprint(control_bp)
    app.register_blueprint(metrics_bp)
    app.register_blueprint(trading_bp)
    app.register_blueprint(actors_bp)
    app.register_blueprint(features_bp)
    app.register_blueprint(strategies_bp)

    # Non-blueprint routes retained for parity
    @app.get("/api/events")
    def events_list() -> tuple[Any, int]:
        """List recent dashboard events."""
        args = request.args
        try:
            limit = int(str(args.get("limit", "100")))
        except Exception:
            limit = 100
        stage = str(args.get("stage", "")) or None
        source = str(args.get("source", "")) or None
        instrument = str(args.get("instrument", "")) or None
        data = svc.list_events(
            limit=limit,
            stage=stage,
            source=source,
            instrument_substr=instrument,
        )
        return jsonify(data), 200

    @app.get("/api/openapi.json")
    def api_openapi_json() -> tuple[Any, int]:
        """Get OpenAPI specification for the dashboard."""
        from ml.dashboard.services.api_explorer_service import APIExplorerService

        api_explorer = APIExplorerService(svc._pipeline_integration_manager, app)
        spec = api_explorer.get_openapi_spec()
        return jsonify(spec), 200

    @app.get("/api/docs")
    def api_docs() -> tuple[str, int]:
        """Get interactive API documentation HTML."""
        from ml.dashboard.services.api_explorer_service import APIExplorerService

        api_explorer = APIExplorerService(svc._pipeline_integration_manager, app)
        html = api_explorer.get_swagger_ui_html()
        return html, 200

    @app.get("/api/observability/status")
    def observability_status() -> tuple[Any, int]:
        """Return observability status payload."""
        data = svc.get_grafana_status()
        return jsonify(data), 200

    @app.get("/health")
    def health() -> tuple[Any, int]:  # pragma: no cover - simple readiness
        """Simple readiness endpoint."""
        return jsonify({"healthy": True}), 200

    @app.get("/metrics")
    def metrics() -> tuple[bytes, int, dict[str, str]]:  # pragma: no cover - passthrough
        """Prometheus metrics endpoint."""
        return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

    @app.get("/")
    def index() -> tuple[str, int]:
        """Return a minimal landing response."""
        return "ML Dashboard API", 200

    return app


__all__ = [
    "create_app_facade",
]
