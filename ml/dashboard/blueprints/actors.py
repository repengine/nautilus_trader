"""
Actors Blueprint for Dashboard API.

This blueprint handles ML actor lifecycle management endpoints:
- POST /api/actors/deploy - Deploy new ML actor
- POST /api/actors/hot-reload - Hot reload actor model
- POST /api/actors/pause - Pause actor
- POST /api/actors/resume - Resume actor
- POST /api/actors/stop - Stop and dispose actor
- GET /api/actors/health - Get all actors health

Example:
    >>> from ml.dashboard.blueprints.actors import actors_bp, register_actors_routes
    >>> register_actors_routes(actors_bp, dashboard_service, require_token_fn)
    >>> app.register_blueprint(actors_bp)
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from flask import Blueprint
from flask import jsonify
from flask import request


if TYPE_CHECKING:
    from flask import Response

    from ml.dashboard.service import DashboardService


actors_bp = Blueprint("actors", __name__, url_prefix="/api/actors")


def register_actors_routes(
    bp: Blueprint,
    svc: DashboardService,
    require_token: Callable[[], bool],
) -> None:
    """
    Register actor lifecycle routes with the blueprint.

    Args:
        bp: The Flask Blueprint to register routes on.
        svc: The DashboardService instance providing business logic.
        require_token: Callable that returns True if authentication is valid.

    Example:
        >>> register_actors_routes(actors_bp, dashboard_service, require_token_fn)
    """

    @bp.post("/deploy")
    def actors_deploy() -> tuple[Response, int]:
        """
        Deploy new ML actor.

        Request Body (JSON):
            actor_type: str - Type of actor to deploy (default "MLSignalActor").
            config: dict - Actor configuration mapping.
            run_id: Optional str - Run identifier for tracking.

        Returns:
            JSON response with deployment result containing:
            - success: bool
            - actor_id: str
            - status: str
            - message: Optional str
            - error: Optional str

        Status Codes:
            202: Deployment accepted
            400: Deployment failed (invalid config or type)
            401: Unauthorized
        """
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401

        from ml.dashboard.services.actors_service import ActorDeploymentRequest
        from ml.dashboard.services.actors_service import ActorIntegrationService

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        actor_service = ActorIntegrationService(svc._pipeline_integration_manager)

        actor_type = payload.get("actor_type", "MLSignalActor")
        config = payload.get("config", {})
        run_id = payload.get("run_id")

        deploy_request = ActorDeploymentRequest(
            actor_type=actor_type,
            config=config,
            run_id=run_id,
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(actor_service.deploy_actor(deploy_request))
        finally:
            loop.close()

        response_data = {
            "success": result.success,
            "actor_id": result.actor_id,
            "status": result.status,
            "message": result.message,
            "error": result.error,
        }
        return jsonify(response_data), 202 if result.success else 400

    @bp.post("/hot-reload")
    def actors_hot_reload() -> tuple[Response, int]:
        """
        Hot reload actor model.

        Request Body (JSON):
            actor_id: str - ID of the actor to reload (required).
            model_id: str - New model ID to load (required).

        Returns:
            JSON response with hot reload result containing:
            - success: bool
            - actor_id: str
            - new_model_id: str
            - status: str
            - error: Optional str

        Status Codes:
            202: Hot reload accepted
            400: Hot reload failed or missing parameters
            401: Unauthorized
        """
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401

        from ml.dashboard.services.actors_service import ActorIntegrationService

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        actor_service = ActorIntegrationService(svc._pipeline_integration_manager)

        actor_id = payload.get("actor_id")
        new_model_id = payload.get("model_id")

        if not actor_id or not new_model_id:
            return jsonify({"error": "actor_id and model_id are required"}), 400

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                actor_service.hot_reload_model(actor_id, new_model_id)
            )
        finally:
            loop.close()

        response_data = {
            "success": result.success,
            "actor_id": result.actor_id,
            "new_model_id": result.new_model_id,
            "status": result.status,
            "error": result.error,
        }
        return jsonify(response_data), 202 if result.success else 400

    @bp.post("/pause")
    def actors_pause() -> tuple[Response, int]:
        """
        Pause actor.

        Request Body (JSON):
            actor_id: str - ID of the actor to pause (required).
            reason: Optional str - Reason for pausing.
            metadata: Optional dict - Additional metadata.

        Returns:
            JSON response with pause result containing:
            - success: bool
            - actor_id: str
            - status: str
            - message: Optional str
            - error: Optional str

        Status Codes:
            202: Pause accepted
            400: Pause failed or missing actor_id
            401: Unauthorized
        """
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401

        from ml.dashboard.services.actors_service import ActorIntegrationService
        from ml.dashboard.services.actors_service import ActorPauseRequest

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        actor_service = ActorIntegrationService(svc._pipeline_integration_manager)

        actor_id = payload.get("actor_id")
        if not actor_id:
            return jsonify({"error": "actor_id is required"}), 400

        pause_request = ActorPauseRequest(
            actor_id=actor_id,
            reason=payload.get("reason"),
            metadata=payload.get("metadata"),
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(actor_service.pause_actor(pause_request))
        finally:
            loop.close()

        response_data = {
            "success": result.success,
            "actor_id": result.actor_id,
            "status": result.status,
            "message": result.message,
            "error": result.error,
        }
        return jsonify(response_data), 202 if result.success else 400

    @bp.post("/resume")
    def actors_resume() -> tuple[Response, int]:
        """
        Resume actor.

        Request Body (JSON):
            actor_id: str - ID of the actor to resume (required).
            metadata: Optional dict - Additional metadata.

        Returns:
            JSON response with resume result containing:
            - success: bool
            - actor_id: str
            - status: str
            - message: Optional str
            - error: Optional str

        Status Codes:
            202: Resume accepted
            400: Resume failed or missing actor_id
            401: Unauthorized
        """
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401

        from ml.dashboard.services.actors_service import ActorIntegrationService
        from ml.dashboard.services.actors_service import ActorResumeRequest

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        actor_service = ActorIntegrationService(svc._pipeline_integration_manager)

        actor_id = payload.get("actor_id")
        if not actor_id:
            return jsonify({"error": "actor_id is required"}), 400

        resume_request = ActorResumeRequest(
            actor_id=actor_id,
            metadata=payload.get("metadata"),
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(actor_service.resume_actor(resume_request))
        finally:
            loop.close()

        response_data = {
            "success": result.success,
            "actor_id": result.actor_id,
            "status": result.status,
            "message": result.message,
            "error": result.error,
        }
        return jsonify(response_data), 202 if result.success else 400

    @bp.post("/stop")
    def actors_stop() -> tuple[Response, int]:
        """
        Stop and dispose actor.

        Request Body (JSON):
            actor_id: str - ID of the actor to stop (required).
            force: bool - Whether to force stop (default False).
            reason: Optional str - Reason for stopping.
            metadata: Optional dict - Additional metadata.

        Returns:
            JSON response with stop result containing:
            - success: bool
            - actor_id: str
            - status: str
            - message: Optional str
            - error: Optional str

        Status Codes:
            202: Stop accepted
            400: Stop failed or missing actor_id
            401: Unauthorized
        """
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401

        from ml.dashboard.services.actors_service import ActorIntegrationService
        from ml.dashboard.services.actors_service import ActorStopRequest

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        actor_service = ActorIntegrationService(svc._pipeline_integration_manager)

        actor_id = payload.get("actor_id")
        if not actor_id:
            return jsonify({"error": "actor_id is required"}), 400

        stop_request = ActorStopRequest(
            actor_id=actor_id,
            force=payload.get("force", False),
            reason=payload.get("reason"),
            metadata=payload.get("metadata"),
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(actor_service.stop_actor(stop_request))
        finally:
            loop.close()

        response_data = {
            "success": result.success,
            "actor_id": result.actor_id,
            "status": result.status,
            "message": result.message,
            "error": result.error,
        }
        return jsonify(response_data), 202 if result.success else 400

    @bp.get("/health")
    def actors_health() -> tuple[Response, int]:
        """
        Get all actors health.

        Returns:
            JSON response with actors health snapshot containing:
            - total_actors: int
            - healthy_actors: int
            - unhealthy_actors: int
            - paused_actors: int
            - actors: dict mapping actor_id to health info

        Status Codes:
            200: Health check successful
        """
        from ml.dashboard.services.actors_service import ActorIntegrationService

        actor_service = ActorIntegrationService(svc._pipeline_integration_manager)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            snapshot = loop.run_until_complete(actor_service.get_actor_health())
        finally:
            loop.close()

        response_data = {
            "total_actors": snapshot.total_actors,
            "healthy_actors": snapshot.healthy_actors,
            "unhealthy_actors": snapshot.unhealthy_actors,
            "paused_actors": snapshot.paused_actors,
            "actors": snapshot.actors,
        }
        return jsonify(response_data), 200


__all__ = ["actors_bp", "register_actors_routes"]
