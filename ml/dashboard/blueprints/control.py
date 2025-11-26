"""
Control Blueprint for Dashboard API.

This blueprint handles control panel endpoints for managing actors, pipelines,
ingestion, and emergency operations:
- POST /api/control/actors/start - Start an actor
- POST /api/control/actors/stop - Stop an actor
- POST /api/control/pipeline/trigger - Trigger pipeline execution
- POST /api/control/ingestion/start - Start data ingestion
- POST /api/control/ingestion/backfill - Trigger historical backfill
- POST /api/control/emergency/stop - Emergency stop all components
- GET /api/control/status - Get system control status

Example:
    >>> from ml.dashboard.blueprints.control import control_bp, register_control_routes
    >>> register_control_routes(control_bp, dashboard_service, require_token_fn)
    >>> app.register_blueprint(control_bp)
"""
from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from flask import Blueprint
from flask import jsonify
from flask import request


if TYPE_CHECKING:
    from flask import Response

    from ml.dashboard.service import DashboardService


control_bp = Blueprint("control", __name__, url_prefix="/api/control")


def _pipeline_response_code(res: dict[str, Any]) -> int:
    """
    Determine HTTP response code for pipeline trigger responses.

    Args:
        res: Response dictionary from pipeline service.

    Returns:
        HTTP status code based on status and success fields.

    Example:
        >>> _pipeline_response_code({"success": True, "status": "QUEUED"})
        202
        >>> _pipeline_response_code({"success": False, "status": "UNAVAILABLE"})
        503
    """
    status_token = str(res.get("status", "ERROR")).upper()
    if status_token == "QUEUED" and res.get("success"):
        return 202
    elif status_token == "UNAVAILABLE":
        return 503
    elif status_token == "INVALID":
        return 400
    elif res.get("success"):
        return 202
    else:
        return 500


def register_control_routes(
    bp: Blueprint,
    svc: DashboardService,
    require_token: Callable[[], bool],
) -> None:
    """
    Register control panel routes with the blueprint.

    This function registers the 7 control-related routes extracted from app.py.
    It delegates all business logic to the DashboardService and SimpleControlPanel
    while handling HTTP concerns (authentication, request parsing, response formatting).

    Args:
        bp: The Flask Blueprint to register routes on.
        svc: The DashboardService instance providing business logic.
        require_token: Callable that returns True if authentication is valid.

    Example:
        >>> from flask import Blueprint
        >>> from ml.dashboard.service import DashboardService
        >>> bp = Blueprint("control", __name__)
        >>> svc = DashboardService.from_config(config)
        >>> register_control_routes(bp, svc, lambda: True)
    """

    @bp.post("/actors/start")
    def control_start_actor() -> tuple[Response, int]:
        """
        Start an actor with specified configuration.

        Request body:
            actor_id: Unique identifier for the actor (required)
            actor_type: Type of actor to start (default: "signal")
            config: Actor configuration dictionary (optional)

        Returns:
            JSON response with actor start result.
            HTTP 401 if unauthorized.
            HTTP 400 if actor_id is missing or start failed.
            HTTP 202 if actor started successfully.
        """
        if not require_token():
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

    @bp.post("/actors/stop")
    def control_stop_actor() -> tuple[Response, int]:
        """
        Stop a running actor.

        Request body:
            actor_id: Unique identifier for the actor to stop (required)

        Returns:
            JSON response with actor stop result.
            HTTP 401 if unauthorized.
            HTTP 400 if actor_id is missing.
            HTTP 200 on success.
        """
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        actor_id = str(payload.get("actor_id", "")).strip()

        if not actor_id:
            return jsonify({"error": "invalid_actor_id"}), 400

        from ml.dashboard.control_simple import SimpleControlPanel

        control_panel = SimpleControlPanel.from_env()
        result = control_panel.stop_actor(actor_id)
        return jsonify(result), 200

    @bp.post("/pipeline/trigger")
    def control_trigger_pipeline() -> tuple[Response, int]:
        """
        Trigger pipeline execution with control tracking.

        Request body:
            pipeline_type or mode: Pipeline type to execute (default: "full")
            config: Pipeline configuration dictionary (optional)

        Returns:
            JSON response with pipeline trigger result including control metadata.
            HTTP 401 if unauthorized.
            HTTP 202 for queued, 400 for invalid, 503 for unavailable.
        """
        if not require_token():
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

        # Trigger pipeline via dashboard service
        pipeline_result = svc.trigger_pipeline(pipeline_type, config_dict)

        # Record in control panel state
        from ml.dashboard.control_simple import SimpleControlPanel

        control_panel = SimpleControlPanel.from_env()
        control_state = control_panel.trigger_pipeline(
            pipeline_type,
            config_dict,
            job_id=str(pipeline_result.get("job_id")) if pipeline_result.get("job_id") else None,
            status=str(pipeline_result.get("status", "QUEUED")).lower(),
        )

        # Merge results
        response_payload = {
            **pipeline_result,
            "control_run_id": control_state["run_id"],
            "control_status": control_state["status"],
        }

        status_code = _pipeline_response_code(pipeline_result)
        return jsonify(response_payload), status_code

    @bp.post("/ingestion/start")
    def control_start_ingestion() -> tuple[Response, int]:
        """
        Start data ingestion for specified symbols.

        Request body:
            symbols: List of symbols to ingest (required)
            source: Data source (default: "databento")

        Returns:
            JSON response with ingestion start result.
            HTTP 401 if unauthorized.
            HTTP 400 if symbols is empty or ingestion failed.
            HTTP 202 if ingestion started successfully.
        """
        if not require_token():
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

    @bp.post("/ingestion/backfill")
    def control_backfill() -> tuple[Response, int]:
        """
        Trigger historical data backfill.

        Request body:
            symbols: List of symbols to backfill (required)
            start_date: Start date in ISO format (required)
            end_date: End date in ISO format (required)

        Returns:
            JSON response with backfill result.
            HTTP 401 if unauthorized.
            HTTP 400 if required parameters are missing or backfill failed.
            HTTP 202 if backfill started successfully.
        """
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        symbols = payload.get("symbols", [])
        start_date = payload.get("start_date")
        end_date = payload.get("end_date")

        if not symbols or not start_date or not end_date:
            return jsonify({"error": "missing_params"}), 400

        import asyncio
        from datetime import datetime

        from ml.dashboard.control_panel import DashboardControlPanel

        control_panel = DashboardControlPanel.from_env()
        result = asyncio.run(
            control_panel.trigger_backfill(
                symbols,
                datetime.fromisoformat(start_date),
                datetime.fromisoformat(end_date),
            ),
        )
        return jsonify(result), 202 if result.get("success") else 400

    @bp.post("/emergency/stop")
    def control_emergency_stop() -> tuple[Response, int]:
        """
        Emergency stop all actors, pipelines, and ingestion tasks.

        Returns:
            JSON response with stopped components list.
            HTTP 401 if unauthorized.
            HTTP 200 on success.
        """
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401

        from ml.dashboard.control_simple import SimpleControlPanel

        control_panel = SimpleControlPanel.from_env()
        result = control_panel.emergency_stop_all()
        return jsonify(result), 200

    @bp.get("/status")
    def control_system_status() -> tuple[Response, int]:
        """
        Get system control status including actors, pipelines, and store health.

        Returns:
            JSON response with comprehensive system status.
            HTTP 200 on success.
        """
        from ml.dashboard.control_simple import SimpleControlPanel

        control_panel = SimpleControlPanel.from_env()
        status = control_panel.get_system_status()
        return jsonify(status), 200


__all__ = ["control_bp", "register_control_routes"]
