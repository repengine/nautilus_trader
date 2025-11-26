"""
Features Blueprint for Dashboard API.

This module provides all feature engineering-related API routes for the ML Dashboard,
extracted from the monolithic app.py for better modularity and testability.

Routes:
    POST /api/features/designer/generate - Generate feature set from designer UI
    POST /api/features/validate-code - Validate custom feature code
    POST /api/features/analyze - Analyze feature importance and correlations
    GET /api/features/manifests - List all feature manifests

"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from flask import Blueprint
from flask import Response
from flask import jsonify
from flask import request


if TYPE_CHECKING:
    from ml.dashboard.service import DashboardService

features_bp = Blueprint("features", __name__, url_prefix="/api/features")


def register_features_routes(
    bp: Blueprint,
    svc: DashboardService,
    require_token: Callable[[], bool],
) -> None:
    """
    Register all features routes on the given blueprint.

    Args:
        bp: The Flask Blueprint to register routes on.
        svc: The DashboardService instance providing integration manager access.
        require_token: A callable that returns True if the request is authorized.

    Example:
        >>> from flask import Flask
        >>> from ml.dashboard.blueprints.features import features_bp, register_features_routes
        >>> app = Flask(__name__)
        >>> svc = DashboardService.from_config(config)
        >>> register_features_routes(features_bp, svc, lambda: True)
        >>> app.register_blueprint(features_bp)

    """
    # -------------------------------------------------------------------------
    # POST /api/features/designer/generate
    # -------------------------------------------------------------------------

    @bp.post("/designer/generate")
    def features_designer_generate() -> tuple[Response, int]:
        """
        Generate feature set from designer UI configuration.

        Requires authentication.

        Request Body (JSON):
            feature_set_name: str - Name for the feature set (required)
            price_features: bool - Include price-based features (default False)
            volume_features: bool - Include volume-based features (default False)
            microstructure: bool - Include microstructure features (default False)
            order_flow: bool - Include order flow features (default False)
            technical_indicators: list[str] - Technical indicators to include
            lookback_periods: str - Comma-separated lookback periods
            custom_code: str | None - Optional custom feature code

        Returns:
            JSON object with:
            - success: bool
            - feature_set_id: str
            - feature_count: int
            - feature_names: list[str]
            - manifest: dict | None
            - error: str | None
            - validation_errors: list[str]

        Status Codes:
            200: Success
            400: Validation failed or missing feature_set_name
            401: Unauthorized

        """
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})

        # Import service lazily to avoid circular imports
        from ml.dashboard.services.features_service import FeatureEngineeringService
        from ml.dashboard.services.features_service import FeatureGenerationRequest

        # Build request
        req = FeatureGenerationRequest(
            feature_set_name=str(payload.get("feature_set_name", "")).strip(),
            price_features=bool(payload.get("price_features", False)),
            volume_features=bool(payload.get("volume_features", False)),
            microstructure=bool(payload.get("microstructure", False)),
            order_flow=bool(payload.get("order_flow", False)),
            technical_indicators=payload.get("technical_indicators", []),
            lookback_periods=str(payload.get("lookback_periods", "10,20,50,100,200")),
            custom_code=payload.get("custom_code"),
        )

        # Validate feature set name
        if not req.feature_set_name:
            return jsonify({"success": False, "error": "feature_set_name required"}), 400

        # Execute generation
        integration_manager = svc.get_integration_manager()
        service = FeatureEngineeringService(integration_manager)
        result = asyncio.run(service.generate_features(req))

        # Convert result to dict
        result_dict = {
            "success": result.success,
            "feature_set_id": result.feature_set_id,
            "feature_count": result.feature_count,
            "feature_names": list(result.feature_names),
            "manifest": result.manifest,
            "error": result.error,
            "validation_errors": list(result.validation_errors),
        }

        return jsonify(result_dict), 200 if result.success else 400

    # -------------------------------------------------------------------------
    # POST /api/features/validate-code
    # -------------------------------------------------------------------------

    @bp.post("/validate-code")
    def features_validate_code() -> tuple[Response, int]:
        """
        Validate custom feature code with security analysis.

        Does not require authentication (code validation is safe).

        Request Body (JSON):
            code: str - Python code to validate (required)
            test_execution: bool - Whether to test execution (default False)

        Returns:
            JSON object with:
            - valid: bool
            - errors: list[str]
            - warnings: list[str]
            - security_risk: bool
            - syntax_error: bool
            - signature_error: bool

        Status Codes:
            200: Validation completed (check 'valid' field for result)
            400: No code provided

        """
        payload = cast(dict[str, Any], request.get_json(silent=True) or {})

        # Import service lazily
        from ml.dashboard.services.features_service import CodeValidationRequest
        from ml.dashboard.services.features_service import FeatureEngineeringService

        # Build request
        req = CodeValidationRequest(
            code=str(payload.get("code", "")).strip(),
            test_execution=bool(payload.get("test_execution", False)),
        )

        if not req.code:
            return jsonify({"valid": False, "errors": ["No code provided"]}), 400

        # Execute validation
        integration_manager = svc.get_integration_manager()
        service = FeatureEngineeringService(integration_manager)
        result = asyncio.run(service.validate_code(req))

        # Convert result to dict
        result_dict = {
            "valid": result.valid,
            "errors": list(result.errors),
            "warnings": list(result.warnings),
            "security_risk": result.security_risk,
            "syntax_error": result.syntax_error,
            "signature_error": result.signature_error,
        }

        return jsonify(result_dict), 200

    # -------------------------------------------------------------------------
    # POST /api/features/analyze
    # -------------------------------------------------------------------------

    @bp.post("/analyze")
    def features_analyze() -> tuple[Response, int]:
        """
        Analyze feature importance and correlations.

        Requires authentication.

        Request Body (JSON):
            feature_set_id: str - Feature set to analyze (required)
            method: str - Analysis method ('shap', 'permutation', etc.) (default 'shap')
            limit: int - Sample limit for analysis (default 1000)

        Returns:
            JSON object with:
            - success: bool
            - total_features: int
            - feature_names: list[str]
            - avg_correlation: float | None
            - max_correlation: float | None
            - feature_importance_method: str | None
            - top_features: list[dict]
            - data_quality: dict
            - error: str | None

        Status Codes:
            200: Success
            400: Missing feature_set_id or analysis failed
            401: Unauthorized

        """
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})

        # Import service lazily
        from ml.dashboard.services.features_service import FeatureAnalysisRequest
        from ml.dashboard.services.features_service import FeatureEngineeringService

        # Build request
        req = FeatureAnalysisRequest(
            feature_set_id=str(payload.get("feature_set_id", "")).strip(),
            method=str(payload.get("method", "shap")),
            limit=int(payload.get("limit", 1000)),
        )

        if not req.feature_set_id:
            return (
                jsonify({"success": False, "error": "feature_set_id required"}),
                400,
            )

        # Execute analysis
        integration_manager = svc.get_integration_manager()
        service = FeatureEngineeringService(integration_manager)
        result = asyncio.run(service.analyze_features(req))

        # Convert result to dict
        result_dict = {
            "success": result.success,
            "total_features": result.total_features,
            "feature_names": list(result.feature_names),
            "avg_correlation": result.avg_correlation,
            "max_correlation": result.max_correlation,
            "feature_importance_method": result.feature_importance_method,
            "top_features": list(result.top_features),
            "data_quality": result.data_quality,
            "error": result.error,
        }

        return jsonify(result_dict), 200 if result.success else 400

    # -------------------------------------------------------------------------
    # GET /api/features/manifests
    # -------------------------------------------------------------------------

    @bp.get("/manifests")
    def features_manifests() -> tuple[Response, int]:
        """
        List all feature manifests in the registry.

        Requires authentication.

        Returns:
            JSON object with:
            - success: bool
            - count: int
            - manifests: list[dict]
            - error: str | None (if failed)

        Status Codes:
            200: Success
            401: Unauthorized
            500: Internal error

        """
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401

        # Import service lazily
        from ml.dashboard.services.features_service import FeatureEngineeringService

        # Execute list
        integration_manager = svc.get_integration_manager()
        service = FeatureEngineeringService(integration_manager)
        result = asyncio.run(service.list_manifests())

        return jsonify(result), 200 if result.get("success") else 500


__all__ = ["features_bp", "register_features_routes"]
