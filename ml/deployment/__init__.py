# ruff: noqa: RUF022
"""
ML Deployment Module.

This module provides deployment utilities, validators, and entrypoints for the ML system.
It follows Universal ML Architecture Patterns and enforces cold path deployment practices.

The deployment module includes:
- Database migration utilities for ML schema management
- Health check systems for service validation
- Container entrypoints for ML actors, strategies, and pipelines
- Dry run systems for testing and validation
- CI/CD integration utilities

All deployment components operate in the cold path and are designed for:
- Container orchestration and Docker deployment
- Progressive fallback chains for service resilience
- Comprehensive health monitoring and validation
- Safe database migrations with SQL parsing
- Environment-driven configuration management

Universal ML Architecture Pattern Compliance:
- Pattern 1: Uses mandatory 4-store + 4-registry integration in entrypoints
- Pattern 2: Protocol-first interface design for health checks
- Pattern 3: Strict cold path operations (no hot path concerns)
- Pattern 4: Progressive fallback chains in all service entrypoints
- Pattern 5: Centralized metrics bootstrap in all components

Security Considerations:
- No pickle/joblib usage in deployment components
- Safe SQL migration parsing with proper splitters
- Environment variable validation and sanitization
- Graceful error handling with structured logging

Performance Characteristics:
- Cold path operations only (no latency constraints)
- Optimized for deployment reliability over speed
- Batch processing capabilities for data operations
- Async/await patterns for I/O intensive operations

Usage Examples:
    Database Migrations:
        from ml.deployment.migrations import apply_migrations_via_compose
        apply_migrations_via_compose(compose_file=Path("docker-compose.yml"))

    Health Checks:
        from ml.deployment.check_health import check_service_health
        healthy, status = check_service_health("postgres", check_postgres)

    Container Entrypoints:
        # For ML Signal Actor
        from ml.deployment.entrypoint_actor import MLSignalActorNode
        node = MLSignalActorNode()
        node.setup()
        await node.run()

        # For ML Trading Strategy
        from ml.deployment.entrypoint_strategy import MLStrategyNode
        strategy_node = MLStrategyNode()
        strategy_node.setup()
        await strategy_node.run()

    Dry Run Testing:
        from ml.deployment.run_local_dry_run import LocalDryRunSystem
        system = LocalDryRunSystem()
        if system.check_prerequisites():
            await system.setup_and_run()

Module Dependencies:
- Core: nautilus_trader.core, nautilus_trader.model
- ML: ml.actors, ml.strategies, ml.stores, ml.config
- Infrastructure: docker, postgresql, asyncio
- Monitoring: prometheus metrics, health checks
- Data: databento adapters for market data

Author: Nautilus Trader ML Team
Version: Compatible with Universal ML Architecture Patterns v1.0
"""

from __future__ import annotations

# Health Check and Validation Systems
from ml.deployment.check_health import check_docker_compose
from ml.deployment.check_health import check_grafana
from ml.deployment.check_health import check_ml_pipeline
from ml.deployment.check_health import check_postgres
from ml.deployment.check_health import check_prometheus
from ml.deployment.check_health import check_redis
from ml.deployment.check_health import check_service_health

# CI/CD Integration Utilities
from ml.deployment.ci_migration_smoke import check_views
from ml.deployment.ci_migration_smoke import wait_for_postgres

# Container Entrypoints for ML Components
from ml.deployment.entrypoint_actor import MLSignalActorNode
from ml.deployment.entrypoint_pipeline import PipelineRunner
from ml.deployment.entrypoint_strategy import MLStrategyNode

# Database and Migration Utilities
from ml.deployment.migrations import apply_migrations_via_compose
from ml.deployment.migrations import list_migration_files

# Dry Run and Testing Systems
from ml.deployment.run_backtest_dry_run import run_backtest_dry_run
from ml.deployment.run_local_dry_run import LocalDryRunSystem


__all__ = [
    "LocalDryRunSystem",
    "MLSignalActorNode",
    "MLStrategyNode",
    "PipelineRunner",
    "apply_migrations_via_compose",
    "check_docker_compose",
    "check_grafana",
    "check_ml_pipeline",
    "check_postgres",
    "check_prometheus",
    "check_redis",
    "check_service_health",
    "check_views",
    "list_migration_files",
    "run_backtest_dry_run",
    "wait_for_postgres",
]
# ruff: noqa: RUF022
