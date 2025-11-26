"""
Metrics Blueprint for Dashboard API.

This blueprint handles metrics and monitoring endpoints:
- GET /api/metrics/snapshot - Get real-time metrics snapshot with KPIs
- GET /api/metrics/portfolio - Get portfolio summary with positions
- GET /api/metrics/ingestion - Get data ingestion rates
- GET /api/metrics/experiments - Get active experiments status

Example:
    >>> from ml.dashboard.blueprints.metrics import metrics_bp, register_metrics_routes
    >>> register_metrics_routes(metrics_bp, dashboard_service)
    >>> app.register_blueprint(metrics_bp)
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from flask import Blueprint
from flask import jsonify


if TYPE_CHECKING:
    from flask import Response

    from ml.dashboard.service import DashboardService


metrics_bp = Blueprint("metrics", __name__, url_prefix="/api/metrics")


def register_metrics_routes(
    bp: Blueprint,
    svc: DashboardService,
) -> None:
    """
    Register metrics routes with the blueprint.

    This function registers the 4 metrics-related routes extracted from app.py.
    It delegates all business logic to the StoreIntegrationService while handling
    HTTP concerns (response formatting).

    Note: These endpoints do not require authentication as they provide read-only
    metrics data for dashboard display.

    Args:
        bp: The Flask Blueprint to register routes on.
        svc: The DashboardService instance providing access to integration manager.

    Example:
        >>> from flask import Blueprint
        >>> from ml.dashboard.service import DashboardService
        >>> bp = Blueprint("metrics", __name__)
        >>> svc = DashboardService.from_config(config)
        >>> register_metrics_routes(bp, svc)
    """

    @bp.get("/snapshot")
    def metrics_snapshot() -> tuple[Response, int]:
        """
        Get real-time metrics snapshot with KPIs.

        Returns comprehensive metrics including daily PnL, Sharpe ratio,
        win rate, max drawdown, active models, ingestion rates, and portfolio
        summary.

        Returns:
            JSON response with metrics snapshot data and HTTP 200.

        Example response:
            {
                "daily_pnl": 1234.56,
                "sharpe_ratio": 1.5,
                "win_rate": 0.65,
                "max_drawdown": 0.05,
                "active_models": 3,
                "ingestion_rate": {...},
                "portfolio": {...}
            }
        """
        from ml.dashboard.services.metrics_service import StoreIntegrationService

        service = StoreIntegrationService(svc._pipeline_integration_manager)
        snapshot = asyncio.run(service.get_metrics_snapshot())

        return jsonify(snapshot.to_dict()), 200

    @bp.get("/portfolio")
    def metrics_portfolio() -> tuple[Response, int]:
        """
        Get portfolio summary with positions.

        Returns current portfolio state including total value, cash,
        margin used, and position count.

        Returns:
            JSON response with portfolio snapshot data and HTTP 200.

        Example response:
            {
                "total_value": 100000.0,
                "cash": 50000.0,
                "margin_used": 25000.0,
                "positions": 5
            }
        """
        from ml.dashboard.services.metrics_service import StoreIntegrationService

        service = StoreIntegrationService(svc._pipeline_integration_manager)
        portfolio = asyncio.run(service.get_portfolio_snapshot())

        return jsonify(portfolio.to_dict()), 200

    @bp.get("/ingestion")
    def metrics_ingestion() -> tuple[Response, int]:
        """
        Get data ingestion rates.

        Returns real-time ingestion metrics including bars per second,
        quotes per second, L2 updates per second, and data quality score.

        Returns:
            JSON response with ingestion rate data and HTTP 200.

        Example response:
            {
                "bars_per_sec": 100.5,
                "quotes_per_sec": 500.2,
                "l2_updates_per_sec": 1000.0,
                "data_quality": 0.99
            }
        """
        from ml.dashboard.services.metrics_service import StoreIntegrationService

        service = StoreIntegrationService(svc._pipeline_integration_manager)
        ingestion = asyncio.run(service.get_ingestion_snapshot())

        return jsonify(ingestion.to_dict()), 200

    @bp.get("/experiments")
    def metrics_experiments() -> tuple[Response, int]:
        """
        Get active experiments status.

        Returns list of active ML experiments including HPO runs,
        feature selection experiments, and architecture search trials.

        Returns:
            JSON response with experiments list and HTTP 200.

        Example response:
            {
                "experiments": [
                    {
                        "type": "model_training",
                        "experiment_id": "exp_001",
                        "status": "running",
                        "created_at": "2025-01-15T12:00:00Z",
                        "metrics": {}
                    }
                ]
            }
        """
        from ml.dashboard.services.metrics_service import StoreIntegrationService

        service = StoreIntegrationService(svc._pipeline_integration_manager)
        experiments = asyncio.run(service.get_experiments_snapshot())

        return jsonify({"experiments": experiments}), 200


__all__ = ["metrics_bp", "register_metrics_routes"]
