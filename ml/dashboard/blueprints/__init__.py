"""
Dashboard Blueprints for modular Flask route organization.

This package contains Flask blueprints for organizing the Dashboard API routes
into logical, modular components. Each blueprint handles a specific domain
of the API (health, registry, pipelines, control, metrics, trading, actors,
features, strategies, etc.).

Usage:
    from ml.dashboard.blueprints.health import health_bp, register_health_routes
    from ml.dashboard.blueprints.pipeline import pipeline_bp, register_pipeline_routes
    from ml.dashboard.blueprints.registry import registry_bp, register_registry_routes
    from ml.dashboard.blueprints.control import control_bp, register_control_routes
    from ml.dashboard.blueprints.metrics import metrics_bp, register_metrics_routes
    from ml.dashboard.blueprints.trading import trading_bp, register_trading_routes
    from ml.dashboard.blueprints.actors import actors_bp, register_actors_routes
    from ml.dashboard.blueprints.features import features_bp, register_features_routes
    from ml.dashboard.blueprints.strategies import strategies_bp, register_strategies_routes

    # In app factory:
    register_health_routes(health_bp, dashboard_service, require_token_fn)
    register_pipeline_routes(pipeline_bp, dashboard_service, require_token_fn)
    register_registry_routes(registry_bp, dashboard_service, require_token_fn)
    register_control_routes(control_bp, dashboard_service, require_token_fn)
    register_metrics_routes(metrics_bp, dashboard_service)
    register_trading_routes(trading_bp, dashboard_service, require_token_fn)
    register_actors_routes(actors_bp, dashboard_service, require_token_fn)
    register_features_routes(features_bp, dashboard_service, require_token_fn)
    register_strategies_routes(strategies_bp, dashboard_service, require_token_fn)
    app.register_blueprint(health_bp)
    app.register_blueprint(pipeline_bp)
    app.register_blueprint(registry_bp)
    app.register_blueprint(control_bp)
    app.register_blueprint(metrics_bp)
    app.register_blueprint(trading_bp)
    app.register_blueprint(actors_bp)
    app.register_blueprint(features_bp)
    app.register_blueprint(strategies_bp)
"""
from __future__ import annotations

from ml.dashboard.blueprints.actors import actors_bp
from ml.dashboard.blueprints.actors import register_actors_routes
from ml.dashboard.blueprints.control import control_bp
from ml.dashboard.blueprints.control import register_control_routes
from ml.dashboard.blueprints.features import features_bp
from ml.dashboard.blueprints.features import register_features_routes
from ml.dashboard.blueprints.health import health_bp
from ml.dashboard.blueprints.health import register_health_routes
from ml.dashboard.blueprints.metrics import metrics_bp
from ml.dashboard.blueprints.metrics import register_metrics_routes
from ml.dashboard.blueprints.pipeline import pipeline_bp
from ml.dashboard.blueprints.pipeline import register_pipeline_routes
from ml.dashboard.blueprints.registry import register_registry_routes
from ml.dashboard.blueprints.registry import registry_bp
from ml.dashboard.blueprints.strategies import register_strategies_routes
from ml.dashboard.blueprints.strategies import strategies_bp
from ml.dashboard.blueprints.trading import register_trading_routes
from ml.dashboard.blueprints.trading import trading_bp


__all__ = [
    "actors_bp",
    "control_bp",
    "features_bp",
    "health_bp",
    "metrics_bp",
    "pipeline_bp",
    "register_actors_routes",
    "register_control_routes",
    "register_features_routes",
    "register_health_routes",
    "register_metrics_routes",
    "register_pipeline_routes",
    "register_registry_routes",
    "register_strategies_routes",
    "register_trading_routes",
    "registry_bp",
    "strategies_bp",
    "trading_bp",
]
