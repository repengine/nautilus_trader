"""
Trading Blueprint for Dashboard API.

This blueprint handles trading control endpoints:
- POST /api/trading/toggle - Toggle live trading mode
- POST /api/trading/emergency - Emergency stop all trading
- GET /api/trading/health - Get trading system health
- GET /api/trading/market-data - Get live market data stream

Example:
    >>> from ml.dashboard.blueprints.trading import trading_bp, register_trading_routes
    >>> register_trading_routes(trading_bp, dashboard_service, require_token_fn)
    >>> app.register_blueprint(trading_bp)
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, cast

from flask import Blueprint
from flask import jsonify
from flask import request


if TYPE_CHECKING:
    from flask import Response

    from ml.dashboard.service import DashboardService


trading_bp = Blueprint("trading", __name__, url_prefix="/api/trading")


def register_trading_routes(
    bp: Blueprint,
    svc: DashboardService,
    require_token: Callable[[], bool],
) -> None:
    """
    Register trading control routes with the blueprint.

    Args:
        bp: The Flask Blueprint to register routes on.
        svc: The DashboardService instance providing business logic.
        require_token: Callable that returns True if authentication is valid.

    Example:
        >>> register_trading_routes(trading_bp, dashboard_service, require_token_fn)
    """

    @bp.post("/toggle")
    def trading_toggle() -> tuple[Response, int]:
        """
        Toggle live trading mode.

        Request Body (JSON):
            enable: bool - Whether to enable or disable live trading.
            safety_checks: Optional mapping of safety check names to bool values.

        Returns:
            JSON response with toggle result containing:
            - success: bool
            - live_trading_enabled: bool
            - timestamp: str
            - safety_checks_passed: bool
            - mode: str
            - error: Optional str

        Status Codes:
            200: Toggle successful
            400: Toggle failed (safety checks or other error)
            401: Unauthorized
        """
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401

        from ml.dashboard.services.trading_service import TradingIntegrationService
        from ml.dashboard.services.trading_service import TradingToggleRequest

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        enable = bool(payload.get("enable", False))
        safety_checks = payload.get("safety_checks")

        toggle_request = TradingToggleRequest(
            enable=enable,
            safety_checks=safety_checks,
        )

        trading_service = TradingIntegrationService(svc._pipeline_integration_manager)
        result = asyncio.run(trading_service.toggle_live_trading(toggle_request))

        return jsonify(asdict(result)), 200 if result.success else 400

    @bp.post("/emergency")
    def trading_emergency() -> tuple[Response, int]:
        """
        Emergency stop all trading.

        Immediately stops all trading activities, cancels orders,
        and closes positions.

        Returns:
            JSON response with emergency stop result containing:
            - success: bool
            - timestamp: str
            - actions_taken: dict with counts of cancelled orders,
              closed positions, stopped actors
            - message: str
            - error: Optional str

        Status Codes:
            200: Emergency stop successful
            401: Unauthorized
            500: Emergency stop failed
        """
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401

        from ml.dashboard.services.trading_service import TradingIntegrationService

        trading_service = TradingIntegrationService(svc._pipeline_integration_manager)
        result = asyncio.run(trading_service.emergency_stop())

        return jsonify(asdict(result)), 200 if result.success else 500

    @bp.get("/health")
    def trading_health() -> tuple[Response, int]:
        """
        Get trading system health.

        Returns:
            JSON response with trading health data containing:
            - healthy: bool
            - trading_enabled: bool
            - market_data: str (connection status)
            - risk_manager: str (status)
            - mode: str (STOPPED, PAPER, LIVE)
            - last_transition: Optional str (ISO timestamp)
            - total_positions: Optional int
            - total_exposure: Optional float
            - unrealized_pnl: Optional float

        Status Codes:
            200: Health check successful
        """
        from ml.dashboard.services.trading_service import TradingIntegrationService

        trading_service = TradingIntegrationService(svc._pipeline_integration_manager)
        health_data = asyncio.run(trading_service.health_check())

        return jsonify(health_data), 200

    @bp.get("/market-data")
    def trading_market_data() -> tuple[Response, int]:
        """
        Get live market data stream.

        Returns aggregated trading metrics for dashboard display.

        Returns:
            JSON response with market data containing:
            - generated_at: str (ISO timestamp)
            - total_positions: int
            - total_exposure: float
            - unrealized_pnl: float
            - realized_pnl: float
            - strategies: list of strategy exposure objects

        Status Codes:
            200: Market data retrieved successfully
        """
        from ml.dashboard.services.trading_service import TradingIntegrationService

        trading_service = TradingIntegrationService(svc._pipeline_integration_manager)
        metrics = asyncio.run(trading_service.get_trading_metrics())

        return jsonify(asdict(metrics)), 200


__all__ = ["register_trading_routes", "trading_bp"]
