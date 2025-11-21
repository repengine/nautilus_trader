"""
Tests for store integration service metrics aggregation.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from ml.dashboard.services.metrics_service import StoreIntegrationService

if TYPE_CHECKING:
    from ml.tests.fixtures.database_fixtures import TestDatabase


pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)

@dataclass(slots=True)
class _StubIntegrationManager:
    """
    Minimal integration manager providing a database connection.
    """

    db_connection: str

    data_store: object | None = None
    model_store: object | None = None
    strategy_store: object | None = None
    feature_store: object | None = None
    strategy_registry: object | None = None


@pytest.fixture
def integration_manager(
    store_integration_metrics_database: TestDatabase,
    patch_engine_manager,
) -> Iterator[_StubIntegrationManager]:
    """
    Provide an integration manager wired to the shared EngineManager harness.
    """

    with patch_engine_manager(engine=store_integration_metrics_database.engine):
        yield _StubIntegrationManager(
            db_connection=store_integration_metrics_database.connection_string,
        )


@pytest.mark.database
@pytest.mark.serial
@pytest.mark.asyncio
async def test_store_metrics_snapshot_aggregates_real_data(
    integration_manager: _StubIntegrationManager,
) -> None:
    service = StoreIntegrationService(integration_manager)

    snapshot = await service.get_metrics_snapshot()

    # Portfolio-driven metrics
    assert snapshot.daily_pnl == pytest.approx(55.0)
    assert snapshot.portfolio.total_value == pytest.approx(25000.0)
    assert snapshot.portfolio.margin_used == pytest.approx(13000.0)
    assert snapshot.portfolio.cash == pytest.approx(12000.0)
    assert snapshot.portfolio.positions == 2

    # Performance metrics derived from strategy/model activity
    assert snapshot.active_models == 2
    assert snapshot.max_drawdown == pytest.approx(0.15)
    assert snapshot.win_rate == pytest.approx(2 / 3)
    strengths = [0.6, -0.3, 0.4]
    expected_sharpe = statistics.mean(strengths) / (statistics.pstdev(strengths) or 1.0)
    assert snapshot.sharpe_ratio == pytest.approx(expected_sharpe)

    # Ingestion metrics over the five minute window
    assert snapshot.ingestion_rate.bars_per_sec == pytest.approx(600 / 300.0)
    assert snapshot.ingestion_rate.quotes_per_sec == pytest.approx(300 / 300.0)
    assert snapshot.ingestion_rate.l2_updates_per_sec == pytest.approx(150 / 300.0)
    assert snapshot.ingestion_rate.data_quality == pytest.approx(900 / 1050.0)
