"""Unit tests for metrics_service module."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from ml.dashboard.services.metrics_service import IngestionRateSnapshot
from ml.dashboard.services.metrics_service import PerformanceMetricsAggregate
from ml.dashboard.services.metrics_service import PortfolioSnapshot
from ml.dashboard.services.metrics_service import StoreIntegrationService
from ml.dashboard.services.metrics_service import StoreMetricsSnapshot

pytestmark = pytest.mark.usefixtures("mock_tracing_backend")


@pytest.fixture
def mock_integration_manager() -> Mock:
    """Mock integration manager for testing."""
    mock = Mock()
    mock.db_connection = "postgresql://test:test@localhost:5432/test"
    mock.model_registry = None
    mock.feature_registry = None
    mock.feature_store = None
    mock.model_store = None
    mock.strategy_store = None
    mock.data_store = None
    return mock


@pytest.fixture
def service_with_integration(mock_integration_manager: Mock) -> StoreIntegrationService:
    """Service with mocked integration manager."""
    return StoreIntegrationService(mock_integration_manager)


@pytest.fixture
def service_no_integration() -> StoreIntegrationService:
    """Service without integration manager."""
    return StoreIntegrationService(None)


@pytest.fixture(autouse=True)
def _isolated_prom_registry(isolated_prometheus_registry: Any) -> None:
    """Ensure Prometheus collectors are isolated per test."""
    # Fixture execution is sufficient; yielded harness handles cleanup.
    del isolated_prometheus_registry


class TestStoreIntegrationService:
    """Tests for StoreIntegrationService."""

    def test_get_service_name(self, service_no_integration: StoreIntegrationService) -> None:
        """Test get_service_name returns correct name."""
        assert service_no_integration.get_service_name() == "store_integration"

    @pytest.mark.asyncio
    async def test_health_check_no_integration(
        self,
        service_no_integration: StoreIntegrationService,
    ) -> None:
        """Test health check when no integration manager."""
        health = await service_no_integration.health_check()
        assert health["healthy"] is False
        assert "No integration manager" in health["reason"]

    @pytest.mark.asyncio
    async def test_health_check_with_integration(
        self,
        service_with_integration: StoreIntegrationService,
    ) -> None:
        """Test health check with integration manager."""
        health = await service_with_integration.health_check()
        assert isinstance(health, dict)
        # Expect all stores to be unavailable in test
        for store_name in ("data_store", "model_store", "feature_store", "strategy_store"):
            assert store_name in health
            assert health[store_name]["healthy"] is False


class TestMetricsSnapshot:
    """Tests for get_metrics_snapshot method."""

    @pytest.mark.asyncio
    async def test_get_metrics_snapshot_no_integration(
        self,
        service_no_integration: StoreIntegrationService,
    ) -> None:
        """Test metrics snapshot when no integration manager."""
        snapshot = await service_no_integration.get_metrics_snapshot()
        assert isinstance(snapshot, StoreMetricsSnapshot)
        assert snapshot.daily_pnl == 0.0
        assert snapshot.sharpe_ratio == 0.0
        assert snapshot.win_rate == 0.0
        assert snapshot.max_drawdown == 0.0
        assert snapshot.active_models == 0

    @pytest.mark.asyncio
    async def test_get_metrics_snapshot_with_mock_engine(
        self,
        service_with_integration: StoreIntegrationService,
    ) -> None:
        """Test metrics snapshot with mocked database engine."""
        with patch(
            "ml.dashboard.services.metrics_service.EngineManager.get_engine"
        ) as mock_get_engine:
            # Mock the engine and connection
            mock_engine = MagicMock()
            mock_conn = MagicMock()
            mock_engine.connect.return_value.__enter__.return_value = mock_conn

            # Mock query results
            mock_signal_row = MagicMock()
            mock_signal_row._mapping = {
                "strategy_count": 5,
                "avg_strength": 0.75,
                "std_strength": 0.1,
                "positive_ratio": 0.65,
            }

            mock_model_row = MagicMock()
            mock_model_row._mapping = {"model_count": 3}

            mock_risk_row = MagicMock()
            mock_risk_row._mapping = {"max_drawdown": 0.05}

            # Setup execute to return different results for different queries
            mock_conn.execute.side_effect = [
                MagicMock(one_or_none=lambda: mock_signal_row),
                MagicMock(one_or_none=lambda: mock_model_row),
                MagicMock(one_or_none=lambda: mock_risk_row),
                MagicMock(fetchall=list),  # Ingestion query
                MagicMock(one_or_none=lambda: None),  # Portfolio query
            ]

            mock_get_engine.return_value = mock_engine

            snapshot = await service_with_integration.get_metrics_snapshot()

            assert isinstance(snapshot, StoreMetricsSnapshot)
            assert snapshot.sharpe_ratio > 0  # 0.75 / 0.1 = 7.5
            assert 0 <= snapshot.win_rate <= 1
            assert snapshot.max_drawdown == 0.05
            assert snapshot.active_models >= 3

    def test_metrics_snapshot_to_dict(self) -> None:
        """Test StoreMetricsSnapshot.to_dict conversion."""
        snapshot = StoreMetricsSnapshot(
            daily_pnl=100.0,
            sharpe_ratio=2.5,
            win_rate=0.65,
            max_drawdown=0.15,
            active_models=5,
        )

        result = snapshot.to_dict()
        assert result["daily_pnl"] == 100.0
        assert result["sharpe_ratio"] == 2.5
        assert result["win_rate"] == 0.65
        assert result["max_drawdown"] == 0.15
        assert result["active_models"] == 5
        assert "ingestion_rate" in result
        assert "portfolio" in result


class TestPortfolioSnapshot:
    """Tests for get_portfolio_snapshot method."""

    @pytest.mark.asyncio
    async def test_get_portfolio_snapshot_no_integration(
        self,
        service_no_integration: StoreIntegrationService,
    ) -> None:
        """Test portfolio snapshot when no integration manager."""
        portfolio = await service_no_integration.get_portfolio_snapshot()
        assert isinstance(portfolio, PortfolioSnapshot)
        assert portfolio.total_value == 0.0
        assert portfolio.cash == 0.0
        assert portfolio.margin_used == 0.0
        assert portfolio.positions == 0

    @pytest.mark.asyncio
    async def test_get_portfolio_snapshot_with_mock_engine(
        self,
        service_with_integration: StoreIntegrationService,
    ) -> None:
        """Test portfolio snapshot with mocked database."""
        with patch(
            "ml.dashboard.services.metrics_service.EngineManager.get_engine"
        ) as mock_get_engine:
            mock_engine = MagicMock()
            mock_conn = MagicMock()
            mock_engine.connect.return_value.__enter__.return_value = mock_conn

            # Mock portfolio query result
            mock_row = MagicMock()
            mock_row._mapping = {
                "total_value": 100000.0,
                "exposure": 25000.0,
                "unrealized_pnl": 500.0,
                "realized_pnl": 1500.0,
                "position_count": 10,
            }

            mock_conn.execute.return_value.one_or_none.return_value = mock_row
            mock_get_engine.return_value = mock_engine

            portfolio = await service_with_integration.get_portfolio_snapshot()

            assert isinstance(portfolio, PortfolioSnapshot)
            assert portfolio.total_value == 100000.0
            assert portfolio.margin_used == 25000.0
            assert portfolio.positions == 10
            assert portfolio.cash == 75000.0  # total - exposure

    def test_portfolio_snapshot_to_dict(self) -> None:
        """Test PortfolioSnapshot.to_dict conversion."""
        portfolio = PortfolioSnapshot(
            total_value=50000.0,
            cash=30000.0,
            margin_used=20000.0,
            positions=5,
        )

        result = portfolio.to_dict()
        assert result["total_value"] == 50000.0
        assert result["cash"] == 30000.0
        assert result["margin_used"] == 20000.0
        assert result["positions"] == 5


class TestIngestionSnapshot:
    """Tests for get_ingestion_snapshot method."""

    @pytest.mark.asyncio
    async def test_get_ingestion_snapshot_no_integration(
        self,
        service_no_integration: StoreIntegrationService,
    ) -> None:
        """Test ingestion snapshot when no integration manager."""
        ingestion = await service_no_integration.get_ingestion_snapshot()
        assert isinstance(ingestion, IngestionRateSnapshot)
        assert ingestion.bars_per_sec == 0.0
        assert ingestion.quotes_per_sec == 0.0
        assert ingestion.l2_updates_per_sec == 0.0
        assert ingestion.data_quality == 0.0

    @pytest.mark.asyncio
    async def test_get_ingestion_snapshot_with_mock_engine(
        self,
        service_with_integration: StoreIntegrationService,
    ) -> None:
        """Test ingestion snapshot with mocked database."""
        with patch(
            "ml.dashboard.services.metrics_service.EngineManager.get_engine"
        ) as mock_get_engine:
            mock_engine = MagicMock()
            mock_conn = MagicMock()
            mock_engine.connect.return_value.__enter__.return_value = mock_conn

            # Mock ingestion query results
            mock_bar_row = MagicMock()
            mock_bar_row._mapping = {
                "dataset_id": "bars_1min",
                "total_count": 3000,
                "success_count": 2950,
            }

            mock_quote_row = MagicMock()
            mock_quote_row._mapping = {
                "dataset_id": "quotes_tick",
                "total_count": 15000,
                "success_count": 14800,
            }

            mock_l2_row = MagicMock()
            mock_l2_row._mapping = {
                "dataset_id": "l2_updates",
                "total_count": 9000,
                "success_count": 8900,
            }

            mock_rows = [mock_bar_row, mock_quote_row, mock_l2_row]

            mock_conn.execute.return_value.fetchall.return_value = mock_rows
            mock_get_engine.return_value = mock_engine

            ingestion = await service_with_integration.get_ingestion_snapshot()

            assert isinstance(ingestion, IngestionRateSnapshot)
            # Should be total_count / window_seconds (300)
            assert ingestion.bars_per_sec == pytest.approx(3000.0 / 300.0, rel=0.01)
            assert ingestion.quotes_per_sec == pytest.approx(15000.0 / 300.0, rel=0.01)
            assert ingestion.l2_updates_per_sec == pytest.approx(9000.0 / 300.0, rel=0.01)
            # Quality = success / total = 26650 / 27000
            assert 0.95 <= ingestion.data_quality <= 1.0

    def test_ingestion_snapshot_to_dict(self) -> None:
        """Test IngestionRateSnapshot.to_dict conversion."""
        ingestion = IngestionRateSnapshot(
            bars_per_sec=10.5,
            quotes_per_sec=50.2,
            l2_updates_per_sec=30.8,
            data_quality=0.98,
        )

        result = ingestion.to_dict()
        assert result["bars_per_sec"] == 10.5
        assert result["quotes_per_sec"] == 50.2
        assert result["l2_updates_per_sec"] == 30.8
        assert result["data_quality"] == 0.98


class TestExperimentsSnapshot:
    """Tests for get_experiments_snapshot method."""

    @pytest.mark.asyncio
    async def test_get_experiments_snapshot_no_integration(
        self,
        service_no_integration: StoreIntegrationService,
    ) -> None:
        """Test experiments snapshot when no integration manager."""
        experiments = await service_no_integration.get_experiments_snapshot()
        assert isinstance(experiments, list)
        assert len(experiments) == 0

    @pytest.mark.asyncio
    async def test_get_experiments_snapshot_with_registries(
        self,
        service_with_integration: StoreIntegrationService,
    ) -> None:
        """Test experiments snapshot with mocked registries."""
        # Mock model registry
        mock_model_registry = MagicMock()
        mock_model_registry.list_experiments.return_value = [
            {
                "experiment_id": "exp_model_001",
                "status": "running",
                "created_at": "2025-01-01T00:00:00Z",
                "metrics": {"loss": 0.25, "accuracy": 0.85},
            },
        ]

        # Mock feature registry
        mock_feature_registry = MagicMock()
        mock_feature_registry.list_experiments.return_value = [
            {
                "experiment_id": "exp_feature_001",
                "status": "active",
                "created_at": "2025-01-02T00:00:00Z",
                "metrics": {"importance_score": 0.75},
            },
        ]

        service_with_integration._integration.model_registry = mock_model_registry
        service_with_integration._integration.feature_registry = mock_feature_registry

        experiments = await service_with_integration.get_experiments_snapshot()

        assert isinstance(experiments, list)
        assert len(experiments) == 2

        # Check model experiment
        model_exp = next(e for e in experiments if e["type"] == "model_training")
        assert model_exp["experiment_id"] == "exp_model_001"
        assert model_exp["status"] == "running"
        assert "metrics" in model_exp

        # Check feature experiment
        feature_exp = next(e for e in experiments if e["type"] == "feature_selection")
        assert feature_exp["experiment_id"] == "exp_feature_001"
        assert feature_exp["status"] == "active"

    @pytest.mark.asyncio
    async def test_get_experiments_snapshot_no_list_experiments_method(
        self,
        service_with_integration: StoreIntegrationService,
    ) -> None:
        """Test experiments snapshot when registries don't have list_experiments."""
        # Mock registries without list_experiments
        mock_model_registry = MagicMock()
        del mock_model_registry.list_experiments

        mock_feature_registry = MagicMock()
        del mock_feature_registry.list_experiments

        service_with_integration._integration.model_registry = mock_model_registry
        service_with_integration._integration.feature_registry = mock_feature_registry

        experiments = await service_with_integration.get_experiments_snapshot()

        assert isinstance(experiments, list)
        assert len(experiments) == 0


class TestHelperMethods:
    """Tests for internal helper methods."""

    def test_calculate_sharpe_ratio_valid(self) -> None:
        """Test Sharpe ratio calculation with valid inputs."""
        service = StoreIntegrationService(None)
        sharpe = service._calculate_sharpe_ratio(0.1, 0.02)
        assert sharpe == pytest.approx(5.0, rel=0.01)

    def test_calculate_sharpe_ratio_zero_std(self) -> None:
        """Test Sharpe ratio calculation with zero standard deviation."""
        service = StoreIntegrationService(None)
        sharpe = service._calculate_sharpe_ratio(0.1, 0.0)
        assert sharpe == 0.0

    def test_calculate_sharpe_ratio_negative_std(self) -> None:
        """Test Sharpe ratio calculation with negative standard deviation."""
        service = StoreIntegrationService(None)
        sharpe = service._calculate_sharpe_ratio(0.1, -0.02)
        assert sharpe == 0.0

    def test_categorize_dataset_bars(self) -> None:
        """Test dataset categorization for bars."""
        service = StoreIntegrationService(None)
        assert service._categorize_dataset("bars_1min") == "bars"
        assert service._categorize_dataset("BARS_5MIN") == "bars"
        assert service._categorize_dataset("historical_bar_data") == "bars"

    def test_categorize_dataset_quotes(self) -> None:
        """Test dataset categorization for quotes."""
        service = StoreIntegrationService(None)
        assert service._categorize_dataset("quotes_tick") == "quotes"
        assert service._categorize_dataset("NBBO_data") == "quotes"
        assert service._categorize_dataset("quote_feed") == "quotes"

    def test_categorize_dataset_l2(self) -> None:
        """Test dataset categorization for L2 updates."""
        service = StoreIntegrationService(None)
        assert service._categorize_dataset("l2_updates") == "l2"
        assert service._categorize_dataset("book_depth_10") == "l2"
        assert service._categorize_dataset("depth_snapshot") == "l2"

    def test_categorize_dataset_unknown(self) -> None:
        """Test dataset categorization for unknown types."""
        service = StoreIntegrationService(None)
        assert service._categorize_dataset("unknown_dataset") is None
        assert service._categorize_dataset("trades_data") is None


class TestPerformanceMetricsAggregate:
    """Tests for PerformanceMetricsAggregate dataclass."""

    def test_default_values(self) -> None:
        """Test default values for performance metrics."""
        aggregate = PerformanceMetricsAggregate()
        assert aggregate.sharpe_ratio == 0.0
        assert aggregate.win_rate == 0.0
        assert aggregate.max_drawdown == 0.0
        assert aggregate.active_models == 0
        assert aggregate.active_strategies == 0

    def test_custom_values(self) -> None:
        """Test custom values for performance metrics."""
        aggregate = PerformanceMetricsAggregate(
            sharpe_ratio=2.5,
            win_rate=0.65,
            max_drawdown=0.15,
            active_models=5,
            active_strategies=3,
        )
        assert aggregate.sharpe_ratio == 2.5
        assert aggregate.win_rate == 0.65
        assert aggregate.max_drawdown == 0.15
        assert aggregate.active_models == 5
        assert aggregate.active_strategies == 3


class TestStoreHealthIntegration:
    """Tests for store health summary integration."""

    @pytest.mark.asyncio
    async def test_get_store_health_summary(
        self,
        service_with_integration: StoreIntegrationService,
    ) -> None:
        """Test get_store_health_summary method."""
        with patch(
            "ml.dashboard.store_health.summarize_all_stores"
        ) as mock_summarize:
            from ml.dashboard.store_health import StoreHealthSummary
            from ml.dashboard.store_health import StoreItemSummary

            # Mock store health summaries
            mock_summaries = [
                StoreHealthSummary(
                    store="feature",
                    healthy=True,
                    connectivity_ok=True,
                    write_ok=True,
                    buffer_backlog=0,
                    latest_event_ns=time.time_ns(),
                    latest_event_iso="2025-01-01T00:00:00Z",
                    age_seconds=10.0,
                    items=(),
                    fallback_active=False,
                    error=None,
                ),
                StoreHealthSummary(
                    store="model",
                    healthy=True,
                    connectivity_ok=True,
                    write_ok=True,
                    buffer_backlog=5,
                    latest_event_ns=time.time_ns(),
                    latest_event_iso="2025-01-01T00:00:00Z",
                    age_seconds=5.0,
                    items=(),
                    fallback_active=False,
                    error=None,
                ),
                StoreHealthSummary(
                    store="strategy",
                    healthy=False,
                    connectivity_ok=False,
                    write_ok=None,
                    buffer_backlog=None,
                    latest_event_ns=None,
                    latest_event_iso=None,
                    age_seconds=None,
                    items=(),
                    fallback_active=True,
                    error="connection_failed",
                ),
                StoreHealthSummary(
                    store="data",
                    healthy=True,
                    connectivity_ok=None,
                    write_ok=None,
                    buffer_backlog=None,
                    latest_event_ns=time.time_ns(),
                    latest_event_iso="2025-01-01T00:00:00Z",
                    age_seconds=2.0,
                    items=(
                        StoreItemSummary(
                            key="dataset_1",
                            latest_event_ns=time.time_ns(),
                            latest_event_iso="2025-01-01T00:00:00Z",
                            age_seconds=1.0,
                        ),
                    ),
                    fallback_active=False,
                    error=None,
                ),
            ]

            mock_summarize.return_value = tuple(mock_summaries)

            summary = await service_with_integration.get_store_health_summary(top_dataset_limit=5)

            assert summary.stores is not None
            assert len(summary.stores) == 4
            assert summary.generated_at is not None

            # Check that stores are correctly converted
            feature_store = next(s for s in summary.stores if s.store == "feature")
            assert feature_store.healthy is True
            assert feature_store.fallback_active is False

            strategy_store = next(s for s in summary.stores if s.store == "strategy")
            assert strategy_store.healthy is False
            assert strategy_store.fallback_active is True
            assert strategy_store.error == "connection_failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
