"""
Strategies Blueprint for Dashboard API.

This module provides all strategy-related API routes for the ML Dashboard,
extracted from the monolithic app.py for better modularity and testability.

Routes:
    GET /api/strategies - List all strategies
    POST /api/strategies - Create strategy (501 Not Implemented)
    POST /api/strategies/optimize - Optimize strategy (501 Not Implemented)
    POST /api/strategies/validate - Validate strategy code
    POST /api/strategies/backtest - Run strategy backtest
    GET /api/strategies/backtest/<job_id>/status - Get backtest status
    GET /api/strategies/backtest/<job_id>/results - Get backtest results
    POST /api/strategies/deploy - Deploy strategy
    GET /api/strategies/<strategy_id>/performance - Get strategy performance

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

strategies_bp = Blueprint("strategies", __name__, url_prefix="/api/strategies")


def register_strategies_routes(
    bp: Blueprint,
    svc: DashboardService,
    require_token: Callable[[], bool],
) -> None:
    """
    Register all strategies routes on the given blueprint.

    Args:
        bp: The Flask Blueprint to register routes on.
        svc: The DashboardService instance providing integration manager access.
        require_token: A callable that returns True if the request is authorized.

    Example:
        >>> from flask import Flask
        >>> from ml.dashboard.blueprints.strategies import strategies_bp, register_strategies_routes
        >>> app = Flask(__name__)
        >>> svc = DashboardService.from_config(config)
        >>> register_strategies_routes(strategies_bp, svc, lambda: True)
        >>> app.register_blueprint(strategies_bp)

    """
    # -------------------------------------------------------------------------
    # GET /api/strategies
    # -------------------------------------------------------------------------

    @bp.get("")
    def strategies_list() -> tuple[Response, int]:
        """
        List all registered strategies.

        Returns:
            JSON object with:
            - strategies: list[dict]
            - count: int

        Status Codes:
            200: Success

        """
        from ml.dashboard.services.strategy_service import StrategyService

        integration_manager = svc.get_integration_manager()
        strategy_svc = StrategyService(integration_manager)
        result = strategy_svc.list_strategies()

        return jsonify(result), 200

    # -------------------------------------------------------------------------
    # POST /api/strategies (Not Implemented)
    # -------------------------------------------------------------------------

    @bp.post("")
    def strategies_create() -> tuple[Response, int]:
        """
        Placeholder endpoint for strategy registry integration.

        Returns:
            JSON object with error message.

        Status Codes:
            501: Not Implemented

        """
        return (
            jsonify({
                "error": "not_implemented",
                "message": "Strategy creation is managed by the strategy registry CLI.",
            }),
            501,
        )

    # -------------------------------------------------------------------------
    # POST /api/strategies/optimize (Not Implemented)
    # -------------------------------------------------------------------------

    @bp.post("/optimize")
    def strategies_optimize() -> tuple[Response, int]:
        """
        Placeholder endpoint for future strategy optimization workflows.

        Returns:
            JSON object with error message.

        Status Codes:
            501: Not Implemented

        """
        return (
            jsonify({
                "error": "not_implemented",
                "message": "Strategy optimization is handled by dedicated pipeline jobs.",
            }),
            501,
        )

    # -------------------------------------------------------------------------
    # POST /api/strategies/validate
    # -------------------------------------------------------------------------

    @bp.post("/validate")
    def strategies_validate() -> tuple[Response, int]:
        """
        Validate strategy code with security checks.

        Requires authentication.

        Request Body (JSON):
            code: str - Python code to validate (required)
            strategy_name: str | None - Optional strategy name
            base_strategy: str - Base strategy class (default 'MLTradingStrategy')

        Returns:
            JSON object with:
            - valid: bool
            - errors: list[str]
            - warnings: list[str]
            - security_risk: bool
            - syntax_error: bool
            - signature_error: bool
            - allowed_imports: list[str]

        Status Codes:
            200: Validation completed
            400: Empty code provided
            401: Unauthorized

        """
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        code = str(payload.get("code", ""))
        strategy_name = payload.get("strategy_name")
        base_strategy = str(payload.get("base_strategy", "MLTradingStrategy"))

        if not code.strip():
            return jsonify({"error": "empty_code", "valid": False}), 400

        # Import strategy service
        from ml.dashboard.services.strategy_service import CodeValidationRequest
        from ml.dashboard.services.strategy_service import StrategyService

        integration_manager = svc.get_integration_manager()
        strategy_svc = StrategyService(integration_manager)

        # Validate code
        validation_request = CodeValidationRequest(
            code=code,
            strategy_name=strategy_name,
            base_strategy=base_strategy,
        )
        result = strategy_svc.validate_strategy_code(validation_request)

        response = {
            "valid": result.valid,
            "errors": list(result.errors),
            "warnings": list(result.warnings),
            "security_risk": result.security_risk,
            "syntax_error": result.syntax_error,
            "signature_error": result.signature_error,
            "allowed_imports": list(result.allowed_imports),
        }

        return jsonify(response), 200

    # -------------------------------------------------------------------------
    # POST /api/strategies/backtest
    # -------------------------------------------------------------------------

    @bp.post("/backtest")
    def strategies_backtest() -> tuple[Response, int]:
        """
        Run strategy backtest.

        Requires authentication.

        Request Body (JSON):
            code: str - Strategy code (required)
            strategy_name: str - Strategy name (required)
            start_date: str - Backtest start date (default '2024-01-01')
            end_date: str - Backtest end date (default '2024-12-31')
            initial_balance: float - Initial balance (default 100000.0)
            instruments: list[str] - Instruments to trade (default ['EURUSD.SIM'])
            risk_params: dict - Risk parameters (default {})

        Returns:
            JSON object with:
            - job_id: str
            - status: str ('queued', 'running', 'completed', 'failed')
            - error: str | None

        Status Codes:
            202: Backtest queued successfully
            400: Invalid request (empty code/name or validation failed)
            401: Unauthorized

        """
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        code = str(payload.get("code", ""))
        strategy_name = str(payload.get("strategy_name", ""))

        if not code.strip():
            return jsonify({"error": "empty_code"}), 400

        if not strategy_name.strip():
            return jsonify({"error": "empty_strategy_name"}), 400

        # Import strategy service
        from ml.dashboard.services.strategy_service import BacktestRequest
        from ml.dashboard.services.strategy_service import StrategyService

        integration_manager = svc.get_integration_manager()
        strategy_svc = StrategyService(integration_manager)

        # Create backtest request
        backtest_request = BacktestRequest(
            strategy_code=code,
            strategy_name=strategy_name,
            start_date=str(payload.get("start_date", "2024-01-01")),
            end_date=str(payload.get("end_date", "2024-12-31")),
            initial_balance=float(payload.get("initial_balance", 100000.0)),
            instruments=list(payload.get("instruments", ["EURUSD.SIM"])),
            risk_params=dict(payload.get("risk_params", {})),
        )

        result = strategy_svc.submit_backtest(backtest_request)

        response = {
            "job_id": result.job_id,
            "status": result.status,
            "error": result.error,
        }

        # 202 Accepted for queued jobs, 400 for validation failures
        status_code = 202 if result.status == "queued" else 400

        return jsonify(response), status_code

    # -------------------------------------------------------------------------
    # GET /api/strategies/backtest/<job_id>/status
    # -------------------------------------------------------------------------

    @bp.get("/backtest/<job_id>/status")
    def strategies_backtest_status(job_id: str) -> tuple[Response, int]:
        """
        Get backtest status by job ID.

        Args:
            job_id: The backtest job identifier.

        Returns:
            JSON object with:
            - job_id: str
            - status: str
            - error: str | None

        Status Codes:
            200: Success
            404: Job not found

        """
        from ml.dashboard.services.strategy_service import StrategyService

        integration_manager = svc.get_integration_manager()
        strategy_svc = StrategyService(integration_manager)
        result = strategy_svc.get_backtest_status(job_id)

        if result is None:
            return jsonify({"error": "not_found", "job_id": job_id}), 404

        response = {
            "job_id": result.job_id,
            "status": result.status,
            "error": result.error,
        }

        return jsonify(response), 200

    # -------------------------------------------------------------------------
    # GET /api/strategies/backtest/<job_id>/results
    # -------------------------------------------------------------------------

    @bp.get("/backtest/<job_id>/results")
    def strategies_backtest_results(job_id: str) -> tuple[Response, int]:
        """
        Get backtest results by job ID.

        Args:
            job_id: The backtest job identifier.

        Returns:
            JSON object with:
            - job_id: str
            - status: str
            - performance_metrics: dict
            - trades: list[dict]
            - equity_curve: list[dict]
            - execution_time_seconds: float

        Status Codes:
            200: Success (backtest completed)
            400: Backtest not completed
            404: Job not found

        """
        from ml.dashboard.services.strategy_service import StrategyService

        integration_manager = svc.get_integration_manager()
        strategy_svc = StrategyService(integration_manager)
        result = strategy_svc.get_backtest_status(job_id)

        if result is None:
            return jsonify({"error": "not_found", "job_id": job_id}), 404

        if result.status != "completed":
            return jsonify({"error": "not_completed", "status": result.status}), 400

        response = {
            "job_id": result.job_id,
            "status": result.status,
            "performance_metrics": dict(result.performance_metrics),
            "trades": list(result.trades),
            "equity_curve": list(result.equity_curve),
            "execution_time_seconds": result.execution_time_seconds,
        }

        return jsonify(response), 200

    # -------------------------------------------------------------------------
    # POST /api/strategies/deploy
    # -------------------------------------------------------------------------

    @bp.post("/deploy")
    def strategies_deploy() -> tuple[Response, int]:
        """
        Deploy strategy live.

        Requires authentication.

        Request Body (JSON):
            code: str - Strategy code (required)
            strategy_name: str - Strategy name (required)
            environment: str - Deployment environment (default 'staging')
            risk_params: dict - Risk parameters (default {})
            instruments: list[str] - Instruments (default ['EURUSD.SIM'])

        Returns:
            JSON object with:
            - deployment_id: str
            - status: str ('deployed', 'pending_approval', 'failed')
            - environment: str
            - message: str
            - monitoring_url: str | None
            - error: str | None

        Status Codes:
            201: Deployed successfully
            400: Deployment failed
            401: Unauthorized

        """
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        code = str(payload.get("code", ""))
        strategy_name = str(payload.get("strategy_name", ""))
        environment = str(payload.get("environment", "staging"))

        if not code.strip():
            return jsonify({"error": "empty_code"}), 400

        if not strategy_name.strip():
            return jsonify({"error": "empty_strategy_name"}), 400

        # Import strategy service
        from ml.dashboard.services.strategy_service import DeploymentRequest
        from ml.dashboard.services.strategy_service import StrategyService

        integration_manager = svc.get_integration_manager()
        strategy_svc = StrategyService(integration_manager)

        # Create deployment request
        deployment_request = DeploymentRequest(
            strategy_name=strategy_name,
            strategy_code=code,
            environment=environment,
            risk_params=dict(payload.get("risk_params", {})),
            instruments=list(payload.get("instruments", ["EURUSD.SIM"])),
        )

        result = strategy_svc.deploy_strategy(deployment_request)

        response = {
            "deployment_id": result.deployment_id,
            "status": result.status,
            "environment": result.environment,
            "message": result.message,
            "monitoring_url": result.monitoring_url,
            "error": result.error,
        }

        # 201 Created for successful deployments, 400 for failures
        status_code = 201 if result.status in ("deployed", "pending_approval") else 400

        return jsonify(response), status_code

    # -------------------------------------------------------------------------
    # GET /api/strategies/<strategy_id>/performance
    # -------------------------------------------------------------------------

    @bp.get("/<strategy_id>/performance")
    def strategies_performance(strategy_id: str) -> tuple[Response, int]:
        """
        Get strategy performance.

        Args:
            strategy_id: The strategy identifier.

        Returns:
            JSON object with performance metrics.

        Status Codes:
            200: Success

        """
        from ml.dashboard.services.strategy_service import StrategyService

        integration_manager = svc.get_integration_manager()
        strategy_svc = StrategyService(integration_manager)
        result = strategy_svc.get_strategy_performance(strategy_id)

        return jsonify(result), 200


__all__ = ["register_strategies_routes", "strategies_bp"]
