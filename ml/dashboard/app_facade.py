"""
Dashboard App Facade using decomposed blueprints.

This module provides a Flask app factory that uses the decomposed blueprints
instead of the monolithic route definitions in app.py. It enables gradual
migration from the legacy monolithic app to a modular blueprint-based architecture.

Feature Flag:
    ML_USE_LEGACY_DASHBOARD_APP=1 -> Use legacy monolithic app.py
    ML_USE_LEGACY_DASHBOARD_APP=0 -> Use new app_facade.py (default)

Architecture:
    - health_bp: System health and services
    - pipeline_bp: Pipeline operations (run, jobs, build-dataset, train-model, hpo)
    - registry_bp: Model/feature/strategy/dataset registry operations
    - control_bp: Actor/pipeline/ingestion control
    - metrics_bp: Real-time metrics and snapshots
    - trading_bp: Trading toggle, emergency stop, health
    - actors_bp: Actor lifecycle management
    - features_bp: Feature engineering operations
    - strategies_bp: Strategy validation, backtest, deployment

Routes NOT in blueprints (kept in main app):
    - /api/orchestrator/<task> - Orchestrator tasks
    - /api/events - Event listing
    - /api/openapi.json - OpenAPI spec
    - /api/docs - API documentation
    - /api/explorer/test - API testing
    - /api/terminal/* - Terminal operations
    - /api/settings - Dashboard settings
    - /api/observability/* - Grafana provisioning and monitoring
    - /health - Basic health check
    - /metrics - Prometheus metrics
    - / - Dashboard index
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, cast

from flask import Blueprint
from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request

from ml.common.logging_config import bind_log_context
from ml.common.logging_config import configure_logging
from ml.common.metrics_export import CONTENT_TYPE_LATEST
from ml.common.metrics_export import generate_latest
from ml.dashboard.blueprints import register_actors_routes
from ml.dashboard.blueprints import register_control_routes
from ml.dashboard.blueprints import register_features_routes
from ml.dashboard.blueprints import register_health_routes
from ml.dashboard.blueprints import register_metrics_routes
from ml.dashboard.blueprints import register_pipeline_routes
from ml.dashboard.blueprints import register_registry_routes
from ml.dashboard.blueprints import register_strategies_routes
from ml.dashboard.blueprints import register_trading_routes
from ml.dashboard.config import DashboardConfig
from ml.dashboard.service import DashboardService


if TYPE_CHECKING:

    pass


def use_legacy_dashboard_app() -> bool:
    """
    Check if legacy app should be used.

    Returns:
        True if ML_USE_LEGACY_DASHBOARD_APP=1, False otherwise.

    Example:
        >>> import os
        >>> os.environ["ML_USE_LEGACY_DASHBOARD_APP"] = "1"
        >>> use_legacy_dashboard_app()
        True
        >>> os.environ["ML_USE_LEGACY_DASHBOARD_APP"] = "0"
        >>> use_legacy_dashboard_app()
        False
    """
    return os.getenv("ML_USE_LEGACY_DASHBOARD_APP", "0") == "1"


def create_app_facade(config: DashboardConfig | None = None) -> Flask:
    """
    Create Flask app using decomposed blueprints.

    This factory function creates a Flask application with routes organized
    into modular blueprints. Routes not yet extracted to blueprints remain
    in the main app.

    Args:
        config: Dashboard configuration. If None, loads from environment.

    Returns:
        Configured Flask application instance.

    Example:
        >>> from ml.dashboard.config import DashboardConfig
        >>> cfg = DashboardConfig(db_connection="")
        >>> app = create_app_facade(cfg)
        >>> isinstance(app, Flask)
        True
    """
    # Configure Flask with static file serving
    static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    app = Flask(__name__, static_folder=static_folder, static_url_path="/static")

    # Light, idempotent logging configuration for API usage
    try:
        configure_logging()
        bind_log_context(component="ml.dashboard.api.facade")
    except Exception:  # pragma: no cover - defensive
        ...

    cfg = config or DashboardConfig.from_env()
    svc = DashboardService.from_config(cfg)

    if cfg.events_poll_interval_seconds > 0.0:
        svc.start_event_polling(cfg.events_poll_interval_seconds)

    @app.teardown_appcontext
    def _shutdown(_: object | None) -> None:  # pragma: no cover - teardown path
        svc.stop_event_polling()

    # -----------------
    # Define require_token helper
    # -----------------
    def require_token() -> bool:
        """
        Validate authentication token from request headers.

        Checks X-ML-DASHBOARD-TOKEN header first, then falls back to
        Authorization: Bearer <token> header.

        Returns:
            True if authentication is valid, False otherwise.
        """
        provided = request.headers.get("X-ML-DASHBOARD-TOKEN")
        if not provided:
            auth_header = request.headers.get("Authorization") or ""
            if auth_header.lower().startswith("bearer "):
                provided = auth_header[7:].strip() or None
        return svc.validate_token(provided)

    # -----------------
    # Create fresh blueprints (to allow multiple app instances in tests)
    # -----------------
    health_bp = Blueprint("health", __name__, url_prefix="/api")
    pipeline_bp = Blueprint("pipeline", __name__, url_prefix="/api/pipeline")
    registry_bp = Blueprint("registry", __name__, url_prefix="/api/registry")
    control_bp = Blueprint("control", __name__, url_prefix="/api/control")
    metrics_bp = Blueprint("metrics", __name__, url_prefix="/api/metrics")
    trading_bp = Blueprint("trading", __name__, url_prefix="/api/trading")
    actors_bp = Blueprint("actors", __name__, url_prefix="/api/actors")
    features_bp = Blueprint("features", __name__, url_prefix="/api/features")
    strategies_bp = Blueprint("strategies", __name__, url_prefix="/api/strategies")

    # -----------------
    # Register all blueprints
    # -----------------

    # Health and services (GET /api/health/system, GET /api/services, POST /api/services/<name>:action)
    register_health_routes(health_bp, svc, require_token)
    app.register_blueprint(health_bp)

    # Pipeline operations (POST /api/pipeline/run, GET /api/pipeline/jobs, etc.)
    register_pipeline_routes(pipeline_bp, svc, require_token)
    app.register_blueprint(pipeline_bp)

    # Registry operations (GET /api/registry/models, etc.)
    register_registry_routes(registry_bp, svc, require_token)
    app.register_blueprint(registry_bp)

    # Control operations (POST /api/control/actors/start, etc.)
    register_control_routes(control_bp, svc, require_token)
    app.register_blueprint(control_bp)

    # Metrics (GET /api/metrics/snapshot, etc.)
    register_metrics_routes(metrics_bp, svc)
    app.register_blueprint(metrics_bp)

    # Trading operations (POST /api/trading/toggle, etc.)
    register_trading_routes(trading_bp, svc, require_token)
    app.register_blueprint(trading_bp)

    # Actor lifecycle (POST /api/actors/deploy, etc.)
    register_actors_routes(actors_bp, svc, require_token)
    app.register_blueprint(actors_bp)

    # Feature engineering (POST /api/features/designer/generate, etc.)
    register_features_routes(features_bp, svc, require_token)
    app.register_blueprint(features_bp)

    # Strategy operations (GET /api/strategies, POST /api/strategies/backtest, etc.)
    register_strategies_routes(strategies_bp, svc, require_token)
    app.register_blueprint(strategies_bp)

    # -----------------
    # Routes not yet extracted to blueprints
    # -----------------

    @app.post("/api/orchestrator/<task>")
    def orchestrator_task(task: str) -> tuple[Any, int]:
        """Trigger orchestrator task."""
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        res = svc.trigger_orchestrator_task(task, payload)
        return jsonify(res), 202 if res.get("ok") else 200

    @app.get("/api/events")
    def events_list() -> tuple[Any, int]:
        """List recent events."""
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

    # API Explorer Routes
    @app.get("/api/openapi.json")
    def api_openapi_json() -> tuple[Any, int]:
        """Get OpenAPI specification."""
        from ml.dashboard.services.api_explorer_service import APIExplorerService

        api_explorer = APIExplorerService(svc._pipeline_integration_manager, app)
        spec = api_explorer.get_openapi_spec()
        return jsonify(spec), 200

    @app.get("/api/docs")
    def api_docs() -> tuple[str, int]:
        """Get interactive API documentation."""
        from ml.dashboard.services.api_explorer_service import APIExplorerService

        api_explorer = APIExplorerService(svc._pipeline_integration_manager, app)
        html = api_explorer.get_swagger_ui_html()
        return html, 200

    @app.post("/api/explorer/test")
    def api_test_endpoint() -> tuple[Any, int]:
        """Test any dashboard endpoint."""
        from ml.dashboard.services.api_explorer_service import APIExplorerService

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        method = str(payload.get("method", "GET")).strip().upper()
        endpoint = str(payload.get("endpoint", "")).strip()
        headers = payload.get("headers")
        body = payload.get("body")

        if not endpoint:
            return jsonify({"error": "missing_endpoint"}), 400

        api_explorer = APIExplorerService(svc._pipeline_integration_manager, app)
        result = api_explorer.test_endpoint(
            method=method, endpoint=endpoint, headers=headers, body=body
        )

        return jsonify(result), 200 if result.get("success") else 400

    # Terminal Routes
    @app.post("/api/terminal/execute")
    def terminal_execute() -> tuple[Any, int]:
        """Execute terminal command."""
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401

        from pathlib import Path

        from ml.dashboard.services.terminal_service import TerminalService

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        command = str(payload.get("command", "")).strip()
        user_id = str(payload.get("user_id", "")) or None

        if not command:
            return jsonify({"error": "empty_command"}), 400

        history_file = Path("ml_data/terminal_history.json")
        service = TerminalService(svc._pipeline_integration_manager, history_file=history_file)

        result = service.execute_command(command, user_id=user_id)

        return (
            jsonify(
                {
                    "command": result.command,
                    "output": result.output,
                    "exit_code": result.exit_code,
                    "duration_seconds": result.duration_seconds,
                    "timestamp": result.timestamp,
                    "command_type": result.command_type,
                    "success": result.success,
                    "error": result.error,
                }
            ),
            200 if result.success else 400,
        )

    @app.get("/api/terminal/history")
    def terminal_history() -> tuple[Any, int]:
        """Get command history."""
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401

        from pathlib import Path

        from ml.dashboard.services.terminal_service import TerminalService

        limit_raw = request.args.get("limit")
        limit_value: int | None = None
        if limit_raw:
            try:
                limit_value = int(limit_raw)
                if limit_value < 0:
                    raise ValueError
            except Exception:
                return jsonify({"error": "invalid_limit"}), 400

        history_file = Path("ml_data/terminal_history.json")
        service = TerminalService(svc._pipeline_integration_manager, history_file=history_file)

        history = service.get_command_history(limit=limit_value)

        return jsonify({"history": history, "total": len(history)}), 200

    @app.get("/api/settings")
    def terminal_settings_get() -> tuple[Any, int]:
        """Get dashboard settings."""
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401

        from pathlib import Path

        from ml.dashboard.services.terminal_service import TerminalService

        section = request.args.get("section") or None

        config_file = Path("ml_data/dashboard_settings.json")
        service = TerminalService(svc._pipeline_integration_manager, config_file=config_file)

        settings = service.get_settings(section=section)

        return jsonify({"settings": settings, "section": section}), 200

    @app.post("/api/settings")
    def terminal_settings_update() -> tuple[Any, int]:
        """Update dashboard settings."""
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401

        from pathlib import Path

        from ml.dashboard.services.terminal_service import TerminalService

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        section = str(payload.get("section", "")).strip()
        updates = payload.get("updates", {})
        validate = bool(payload.get("validate", True))

        if not section:
            return jsonify({"error": "missing_section"}), 400

        if not isinstance(updates, dict):
            return jsonify({"error": "invalid_updates"}), 400

        config_file = Path("ml_data/dashboard_settings.json")
        service = TerminalService(svc._pipeline_integration_manager, config_file=config_file)

        result = service.update_settings(section, updates, validate=validate)

        if result.get("success"):
            return jsonify(result), 200
        else:
            return jsonify(result), 400

    # Observability Routes
    @app.post("/api/observability/grafana/provision")
    def grafana_provision() -> tuple[Any, int]:
        """Provision Grafana dashboard."""
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        title = str(payload.get("title") or "") or None
        res = svc.provision_grafana_dashboard(title=title)
        return jsonify(res), 202 if res.get("ok") else 200

    @app.get("/api/observability/status")
    def observability_status() -> tuple[Any, int]:
        """Get observability status."""
        data = svc.get_grafana_status()
        return jsonify(data), 200

    @app.get("/api/observability/summary")
    def observability_summary() -> tuple[Any, int]:
        """Get Prometheus summary."""
        data = svc.get_prometheus_summary()
        return jsonify(data), 200

    @app.get("/api/observability/stores")
    def observability_stores() -> tuple[Any, int]:
        """Get store summary."""
        data = svc.get_store_summary()
        return jsonify(data), 200

    # Core routes
    @app.get("/health")
    def health() -> tuple[Any, int]:  # pragma: no cover - simple readiness
        """Basic health check endpoint."""
        return jsonify({"healthy": True}), 200

    @app.get("/metrics")
    def metrics() -> tuple[bytes, int, dict[str, str]]:  # pragma: no cover - passthrough
        """Prometheus metrics endpoint."""
        return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

    @app.get("/")
    def index() -> tuple[str, int]:
        """Dashboard index page."""
        ui_type = request.args.get("ui") or request.cookies.get("ui_preference") or "standard"

        template_map = {
            "unified": "index_unified.html",
            "control": "index_control.html",
            "enhanced": "index_enhanced.html",
            "advanced": "index_advanced.html",
            "standard": "index.html",
        }

        template = template_map.get(ui_type, "index.html")
        template_path = os.path.join(app.root_path, "templates", template)
        if not os.path.exists(template_path):
            template = "index.html"

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


def create_app(config: DashboardConfig | None = None) -> Flask:
    """
    Factory that selects legacy or facade based on feature flag.

    This is the main entry point for creating a Flask application.
    It checks the ML_USE_LEGACY_DASHBOARD_APP environment variable
    to determine which implementation to use.

    Args:
        config: Dashboard configuration. If None, loads from environment.

    Returns:
        Configured Flask application instance.

    Example:
        >>> import os
        >>> os.environ["ML_USE_LEGACY_DASHBOARD_APP"] = "0"
        >>> app = create_app()
        >>> isinstance(app, Flask)
        True
    """
    if use_legacy_dashboard_app():
        from ml.dashboard.app import create_app as legacy_create_app
        return legacy_create_app(config)
    return create_app_facade(config)


__all__ = [
    "create_app",
    "create_app_facade",
    "use_legacy_dashboard_app",
]
