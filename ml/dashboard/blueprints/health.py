"""
Health and Services Blueprint for Dashboard API.

This blueprint handles health check and service control endpoints:
- GET /api/health/system - Get system health status
- GET /api/services - List registered services
- POST /api/services/<name>:action - Control a service (start/stop/restart)

Example:
    >>> from ml.dashboard.blueprints.health import health_bp, register_health_routes
    >>> register_health_routes(health_bp, dashboard_service, require_token_fn)
    >>> app.register_blueprint(health_bp)
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from flask import Blueprint
from flask import jsonify
from flask import request


if TYPE_CHECKING:
    from flask import Response

    from ml.dashboard.service import DashboardService


health_bp = Blueprint("health", __name__, url_prefix="/api")


def register_health_routes(
    bp: Blueprint,
    svc: DashboardService,
    require_token: Callable[[], bool],
) -> None:
    """
    Register health and services routes with the blueprint.

    Args:
        bp: The Flask Blueprint to register routes on.
        svc: The DashboardService instance providing business logic.
        require_token: Callable that returns True if authentication is valid.

    Example:
        >>> register_health_routes(health_bp, dashboard_service, require_token_fn)
    """

    @bp.get("/health/system")
    def health_system() -> tuple[Response, int]:
        """
        Get system health status.

        Returns:
            JSON response with system health data and HTTP 200.
        """
        data = svc.get_system_health()
        return jsonify(data), 200

    @bp.get("/services")
    def services_list() -> tuple[Response, int]:
        """
        List all registered services.

        Returns:
            JSON response with list of services and HTTP 200.
        """
        data = svc.list_services()
        return jsonify(data), 200

    @bp.post("/services/<name>:action")
    def services_action(name: str) -> tuple[Response, int]:
        """
        Control a service (start, stop, restart).

        Args:
            name: The service name to control.

        Returns:
            JSON response with action result.
            HTTP 401 if unauthorized.
            HTTP 400 if action is invalid.
            HTTP 202 if action is accepted (async operation).
            HTTP 200 otherwise.
        """
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        action = str(payload.get("action", "")).strip().lower()
        if action not in {"start", "stop", "restart"}:
            return jsonify({"error": "invalid_action"}), 400
        res = svc.control_service(name, action)
        # 202 Accepted for async/semi-async operations
        return jsonify(res), 202 if res.get("ok") else 200


__all__ = ["health_bp", "register_health_routes"]
