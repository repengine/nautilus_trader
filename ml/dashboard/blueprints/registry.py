"""
Registry Blueprint for Dashboard API.

This module provides all registry-related API routes for the ML Dashboard,
extracted from the monolithic app.py for better modularity and testability.

Routes:
    GET /api/registry/models - List all registered models
    GET /api/registry/models/<model_id>/history - Get model performance history
    GET /api/registry/deployments - List active deployments
    GET /api/registry/features - List features (with optional filters)
    GET /api/registry/strategies - List strategies
    GET /api/registry/datasets - List datasets
    GET /api/registry/strategies/<strategy_id> - Get strategy details
    GET /api/registry/strategies/<strategy_id>/compatibility - Check strategy compatibility
    GET /api/registry/features/<feature_set_id>/lineage - Get feature lineage
    GET /api/registry/datasets/watermarks - Get dataset watermarks
    GET /api/registry/datasets/lineage - Get dataset lineage
    POST /api/registry/features/<feature_set_id>:promote - Promote feature
    POST /api/registry/features/<feature_set_id>:deprecate - Deprecate feature
    POST /api/registry/models/<model_id>:deploy - Deploy model
    POST /api/registry/models/<model_id>:hot_reload - Hot reload model
    POST /api/registry/deployments:rollback - Rollback deployment

"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from flask import Blueprint
from flask import Response
from flask import jsonify
from flask import request


if TYPE_CHECKING:
    from ml.dashboard.service import DashboardService

registry_bp = Blueprint("registry", __name__, url_prefix="/api/registry")


def register_registry_routes(
    bp: Blueprint,
    svc: DashboardService,
    require_token: Callable[[], bool],
) -> None:
    """
    Register all registry routes on the given blueprint.

    Args:
        bp: The Flask Blueprint to register routes on.
        svc: The DashboardService instance providing registry operations.
        require_token: A callable that returns True if the request is authorized.

    Example:
        >>> from flask import Flask
        >>> from ml.dashboard.blueprints.registry import registry_bp, register_registry_routes
        >>> app = Flask(__name__)
        >>> svc = DashboardService.from_config(config)
        >>> register_registry_routes(registry_bp, svc, lambda: True)
        >>> app.register_blueprint(registry_bp)

    """
    # -------------------------------------------------------------------------
    # GET routes (read-only)
    # -------------------------------------------------------------------------

    @bp.get("/models")
    def registry_models() -> tuple[Response, int]:
        """
        List all registered models.

        Returns:
            JSON array of model objects with fields:
            - model_id: str
            - role: str
            - version: str
            - deployment_status: str
            - deployed_to: list[str]
            - architecture: str
            - feature_schema_hash: str

        """
        return jsonify(svc.list_models()), 200

    @bp.get("/models/<model_id>/history")
    def registry_model_history(model_id: str) -> tuple[Response, int]:
        """
        Get performance history for a specific model.

        Args:
            model_id: The model identifier.

        Query Parameters:
            limit: Optional integer to limit the number of history entries.

        Returns:
            JSON array of performance history entries.

        """
        limit_raw = request.args.get("limit")
        limit_value: int | None = None
        if limit_raw:
            try:
                limit_value = int(limit_raw)
                if limit_value < 0:
                    raise ValueError("limit must be non-negative")
            except Exception:
                return jsonify({"error": "invalid_limit"}), 400
        data = svc.get_model_performance_history(model_id, limit=limit_value)
        return jsonify(data), 200

    @bp.get("/deployments")
    def registry_deployments() -> tuple[Response, int]:
        """
        List active deployments.

        Returns:
            JSON object mapping target names to lists of deployed model IDs.

        """
        return jsonify(svc.list_deployments()), 200

    @bp.get("/features")
    def registry_features() -> tuple[Response, int]:
        """
        List features with optional filtering.

        Query Parameters:
            role: Optional filter by feature role.
            stage: Optional filter by feature stage.

        Returns:
            JSON array of feature objects with fields:
            - feature_set_id: str
            - role: str
            - stage: str
            - schema_hash: str
            - version: str

        """
        role = request.args.get("role")
        stage = request.args.get("stage")
        return jsonify(svc.list_features(role=role or None, stage=stage or None)), 200

    @bp.get("/strategies")
    def registry_strategies() -> tuple[Response, int]:
        """
        List all registered strategies.

        Returns:
            JSON array of strategy objects with fields:
            - strategy_id: str
            - type: str
            - version: str
            - required_models: list[str]

        """
        return jsonify(svc.list_strategies()), 200

    @bp.get("/datasets")
    def registry_datasets() -> tuple[Response, int]:
        """
        List all registered datasets.

        Returns:
            JSON array of dataset objects with fields:
            - dataset_id: str
            - dataset_type: str
            - location: str
            - version: str

        """
        return jsonify(svc.list_datasets()), 200

    @bp.get("/strategies/<strategy_id>")
    def strategy_details(strategy_id: str) -> tuple[Response, int]:
        """
        Get detailed information for a specific strategy.

        Args:
            strategy_id: The strategy identifier.

        Returns:
            JSON object with strategy details or empty object if not found.

        """
        data = svc.get_strategy_details(strategy_id)
        if data is None:
            return jsonify({}), 200
        return jsonify(data), 200

    @bp.get("/strategies/<strategy_id>/compatibility")
    def strategy_compatibility(strategy_id: str) -> tuple[Response, int]:
        """
        Check strategy compatibility against active strategies.

        Args:
            strategy_id: The strategy identifier to check.

        Query Parameters:
            active: Comma-separated list of currently active strategy IDs.

        Returns:
            JSON object with:
            - strategy_id: str
            - compatible: bool

        """
        active_raw = request.args.get("active", "")
        active = [s.strip() for s in active_raw.split(",") if s.strip()]
        return jsonify(svc.check_strategy_compatibility(strategy_id, active)), 200

    @bp.get("/features/<feature_set_id>/lineage")
    def feature_lineage(feature_set_id: str) -> tuple[Response, int]:
        """
        Get lineage information for a feature set.

        Args:
            feature_set_id: The feature set identifier.

        Returns:
            JSON array of lineage entries with fields:
            - feature_set_id: str
            - role: str
            - stage: str
            - version: str
            - schema_hash: str

        """
        return jsonify(svc.get_feature_lineage(feature_set_id)), 200

    @bp.get("/datasets/watermarks")
    def dataset_watermarks() -> tuple[Response, int]:
        """
        Get watermarks for a dataset.

        Query Parameters:
            dataset_id: Required dataset identifier.
            instrument: Optional instrument filter.
            source: Optional source filter.
            limit: Optional limit (default 100).

        Returns:
            JSON array of watermark entries or empty array if dataset_id missing.

        """
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
                    dataset_id=ds,
                    instrument=instr or None,
                    source=source or None,
                    limit=limit,
                )
            ),
            200,
        )

    @bp.get("/datasets/lineage")
    def dataset_lineage() -> tuple[Response, int]:
        """
        Get lineage information for datasets.

        Query Parameters:
            child: Optional child dataset filter.
            parent: Optional parent dataset filter.
            limit: Optional limit (default 100).

        Returns:
            JSON array of lineage records.

        """
        child = request.args.get("child") or None
        parent = request.args.get("parent") or None
        try:
            limit = int(str(request.args.get("limit", "100")))
        except Exception:
            limit = 100
        return jsonify(svc.list_dataset_lineage(child=child, parent=parent, limit=limit)), 200

    # -------------------------------------------------------------------------
    # POST routes (require authentication)
    # -------------------------------------------------------------------------

    @bp.post("/features/<feature_set_id>:promote")
    def registry_feature_promote(feature_set_id: str) -> tuple[Response, int]:
        """
        Promote a feature set to a new stage.

        Args:
            feature_set_id: The feature set identifier.

        Request Body (JSON):
            stage: Optional target stage (default "PROD").
            gates: Optional list of quality gate configurations.

        Returns:
            JSON object with:
            - ok: bool
            - feature_set_id: str
            - stage: str

        Status Codes:
            202: Successfully accepted (if ok=True)
            200: Operation completed but not successful
            401: Unauthorized

        """
        if not require_token():
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

    @bp.post("/features/<feature_set_id>:deprecate")
    def registry_feature_deprecate(feature_set_id: str) -> tuple[Response, int]:
        """
        Deprecate a feature set.

        Args:
            feature_set_id: The feature set identifier.

        Request Body (JSON):
            reason: Optional deprecation reason.

        Returns:
            JSON object with:
            - ok: bool
            - feature_set_id: str

        Status Codes:
            202: Successfully accepted (if ok=True)
            200: Operation completed but not successful
            401: Unauthorized

        """
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        reason = str(payload.get("reason") or "") or None
        res = svc.deprecate_feature(feature_set_id, reason=reason)
        return jsonify(res), 202 if res.get("ok") else 200

    @bp.post("/models/<model_id>:deploy")
    def registry_deploy(model_id: str) -> tuple[Response, int]:
        """
        Deploy a model to a target.

        Args:
            model_id: The model identifier to deploy.

        Request Body (JSON):
            target: Target deployment name (default "ml_signal_actor").

        Returns:
            JSON object with:
            - ok: bool
            - model_id: str
            - target: str

        Status Codes:
            202: Successfully accepted (if ok=True)
            200: Operation completed but not successful
            400: Invalid target
            401: Unauthorized

        """
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        target = str(payload.get("target", "ml_signal_actor")).strip()
        if not target:
            return jsonify({"error": "invalid_target"}), 400
        res = svc.deploy_model(model_id, target)
        return jsonify(res), 202 if res.get("ok") else 200

    @bp.post("/models/<model_id>:hot_reload")
    def registry_hot_reload(model_id: str) -> tuple[Response, int]:
        """
        Hot reload a model to a target deployment.

        Args:
            model_id: The new model identifier to load.

        Request Body (JSON):
            target: Target deployment name (default "ml_signal_actor").

        Returns:
            JSON object with:
            - ok: bool
            - target: str
            - model_id: str

        Status Codes:
            202: Successfully accepted (if ok=True)
            200: Operation completed but not successful
            400: Invalid target
            401: Unauthorized

        """
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        target = str(payload.get("target", "ml_signal_actor")).strip()
        if not target:
            return jsonify({"error": "invalid_target"}), 400
        res = svc.hot_reload_model(target=target, new_model_id=model_id)
        return jsonify(res), 202 if res.get("ok") else 200

    @bp.post("/deployments:rollback")
    def registry_rollback() -> tuple[Response, int]:
        """
        Rollback a deployment to a previous model.

        Request Body (JSON):
            target: Target deployment name (required).
            to_model_id: Model ID to roll back to (required).

        Returns:
            JSON object with:
            - ok: bool
            - target: str
            - model_id: str

        Status Codes:
            202: Successfully accepted (if ok=True)
            200: Operation completed but not successful
            400: Invalid parameters
            401: Unauthorized

        """
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        target = str(payload.get("target", "")).strip()
        to_model_id = str(payload.get("to_model_id", "")).strip()
        if not target or not to_model_id:
            return jsonify({"error": "invalid_params"}), 400
        res = svc.rollback_deployment(target=target, to_model_id=to_model_id)
        return jsonify(res), 202 if res.get("ok") else 200


__all__ = ["register_registry_routes", "registry_bp"]
