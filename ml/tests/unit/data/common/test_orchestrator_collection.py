"""
Unit tests for OrchestratorCollectionComponent.

Tests extracted orchestrator-based collection logic from DataScheduler:
- collect_via_orchestrator() with various scenarios
- SQL provider/writer initialization
- Domain loader protocol implementation
- Market binding resolution and backfill

Test count: 10
Coverage target: 95%

"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from ml.data.common.orchestrator_collection import OrchestratorCollectionComponent
from ml.data.common.orchestrator_collection import OrchestratorCollectionProtocol


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def component() -> OrchestratorCollectionComponent:
    """Create an OrchestratorCollectionComponent instance for testing."""
    return OrchestratorCollectionComponent()


@pytest.fixture
def minimal_config() -> Any:
    """Create minimal SchedulerConfig for testing."""
    from ml.config.scheduler_config import DatabentoConfig
    from ml.config.scheduler_config import SchedulerConfig

    return SchedulerConfig(
        symbols=["SPY.XNAS", "QQQ.XNAS"],
        retention_days=30,
        feature_store_enabled=False,
        databento=DatabentoConfig(
            dataset="GLBX.MDP3",
            schema="ohlcv-1m",
            api_key=None,
        ),
    )


@pytest.fixture
def config_with_api_key() -> Any:
    """Create SchedulerConfig with Databento API key."""
    from ml.config.scheduler_config import DatabentoConfig
    from ml.config.scheduler_config import SchedulerConfig

    return SchedulerConfig(
        symbols=["SPY.XNAS"],
        databento=DatabentoConfig(
            dataset="GLBX.MDP3",
            schema="ohlcv-1m",
            api_key="test_api_key_123",
            price_precision=6,
        ),
    )


@pytest.fixture
def config_with_market_bindings() -> Any:
    """Create SchedulerConfig with market_inputs."""
    from ml.config.market_data import MarketDatasetInput
    from ml.config.scheduler_config import DatabentoConfig
    from ml.config.scheduler_config import SchedulerConfig

    return SchedulerConfig(
        symbols=["SPY.XNAS"],
        databento=DatabentoConfig(
            dataset="GLBX.MDP3",
            schema="ohlcv-1m",
            api_key="test_api_key_123",
        ),
        market_dataset_id="EQUS.MINI",
        market_inputs=(
            MarketDatasetInput(dataset_id="EQUS.MINI", schema_override="ohlcv-1m"),
        ),
    )


@pytest.fixture
def mock_registry() -> MagicMock:
    """Create mock DataRegistry."""
    registry = MagicMock()
    registry.emit_event = MagicMock()
    return registry


@pytest.fixture
def mock_catalog() -> MagicMock:
    """Create mock ParquetDataCatalog."""
    catalog = MagicMock()
    catalog.write_data = MagicMock()
    return catalog


# -----------------------------------------------------------------------------
# Basic Collection Tests (4 tests)
# -----------------------------------------------------------------------------


class TestCollectViaOrchestrator:
    """Tests for collect_via_orchestrator method."""

    def test_collect_via_orchestrator_basic(
        self,
        component: OrchestratorCollectionComponent,
        config_with_api_key: Any,
        mock_registry: MagicMock,
        mock_catalog: MagicMock,
    ) -> None:
        """Test basic collection via orchestrator succeeds."""
        connection = "postgresql://user:pass@host:5432/db"

        with (
            patch.object(
                component, "create_sql_coverage_provider"
            ) as mock_coverage,
            patch.object(component, "create_sql_writer") as mock_writer,
            patch.object(component, "_create_ingestor") as mock_ingestor,
            patch.object(component, "_create_orchestrator") as mock_orch,
        ):
            mock_orchestrator = MagicMock()
            mock_orch.return_value = mock_orchestrator
            mock_orchestrator.backfill_gaps = MagicMock()

            component.collect_via_orchestrator(
                config=config_with_api_key,
                connection=connection,
                registry=mock_registry,
                catalog=mock_catalog,
                dual_write=False,
            )

            mock_coverage.assert_called_once_with(connection, "market_data")
            mock_writer.assert_called_once_with(connection, "market_data")
            mock_ingestor.assert_called_once_with("test_api_key_123")
            mock_orchestrator.backfill_gaps.assert_called()

    def test_collect_via_orchestrator_no_api_key(
        self,
        component: OrchestratorCollectionComponent,
        minimal_config: Any,
        mock_registry: MagicMock,
        mock_catalog: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test collect_via_orchestrator raises ValueError when no API key."""
        # Ensure no env var is set
        monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
        connection = "postgresql://user:pass@host:5432/db"

        with pytest.raises(ValueError, match="DATABENTO_API_KEY required"):
            component.collect_via_orchestrator(
                config=minimal_config,
                connection=connection,
                registry=mock_registry,
                catalog=mock_catalog,
                dual_write=False,
            )

    def test_collect_via_orchestrator_no_db_connection(
        self,
        component: OrchestratorCollectionComponent,
        config_with_api_key: Any,
        mock_registry: MagicMock,
        mock_catalog: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test collect_via_orchestrator raises ValueError when no DB connection."""
        # Ensure no env vars are set
        monkeypatch.delenv("DB_CONNECTION", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("NAUTILUS_DB_CONNECTION", raising=False)

        with pytest.raises(ValueError, match="DB connection required"):
            component.collect_via_orchestrator(
                config=config_with_api_key,
                connection=None,
                registry=mock_registry,
                catalog=mock_catalog,
                dual_write=False,
            )

    def test_collect_via_orchestrator_no_registry(
        self,
        component: OrchestratorCollectionComponent,
        config_with_api_key: Any,
        mock_catalog: MagicMock,
    ) -> None:
        """Test collect_via_orchestrator raises RuntimeError when no registry."""
        connection = "postgresql://user:pass@host:5432/db"

        with pytest.raises(RuntimeError, match="DataRegistry not initialized"):
            component.collect_via_orchestrator(
                config=config_with_api_key,
                connection=connection,
                registry=None,
                catalog=mock_catalog,
                dual_write=False,
            )


# -----------------------------------------------------------------------------
# Dual-Write Tests (1 test)
# -----------------------------------------------------------------------------


class TestDualWrite:
    """Tests for dual-write functionality."""

    def test_collect_via_orchestrator_with_dual_write(
        self,
        component: OrchestratorCollectionComponent,
        config_with_api_key: Any,
        mock_registry: MagicMock,
        mock_catalog: MagicMock,
    ) -> None:
        """Test collection with dual_write creates raw_writer and domain_loader."""
        connection = "postgresql://user:pass@host:5432/db"

        with (
            patch.object(
                component, "create_sql_coverage_provider"
            ) as mock_coverage,
            patch.object(component, "create_sql_writer") as mock_writer,
            patch.object(component, "_create_ingestor") as mock_ingestor,
            patch.object(component, "create_raw_writer") as mock_raw_writer,
            patch.object(
                component, "_create_domain_loader"
            ) as mock_domain_loader,
            patch.object(component, "_create_orchestrator") as mock_orch,
        ):
            mock_orchestrator = MagicMock()
            mock_orch.return_value = mock_orchestrator
            mock_orchestrator.backfill_gaps = MagicMock()

            component.collect_via_orchestrator(
                config=config_with_api_key,
                connection=connection,
                registry=mock_registry,
                catalog=mock_catalog,
                dual_write=True,
            )

            # Verify dual-write components were created
            mock_raw_writer.assert_called_once_with(mock_catalog)
            mock_domain_loader.assert_called_once_with(
                "test_api_key_123",
                config_with_api_key,
            )


# -----------------------------------------------------------------------------
# Market Binding Tests (2 tests)
# -----------------------------------------------------------------------------


class TestMarketBindings:
    """Tests for market binding resolution and backfill."""

    def test_collect_via_orchestrator_with_market_bindings(
        self,
        component: OrchestratorCollectionComponent,
        config_with_market_bindings: Any,
        mock_registry: MagicMock,
        mock_catalog: MagicMock,
    ) -> None:
        """Test collection resolves and uses market bindings."""
        connection = "postgresql://user:pass@host:5432/db"

        # Create mock binding
        mock_binding = MagicMock()
        mock_binding.binding_id = "test_binding_1"

        with (
            patch.object(
                component, "create_sql_coverage_provider"
            ),
            patch.object(component, "create_sql_writer"),
            patch.object(component, "_create_ingestor"),
            patch.object(component, "_create_orchestrator") as mock_orch,
            patch.object(
                component, "_resolve_bindings"
            ) as mock_resolve,
        ):
            mock_orchestrator = MagicMock()
            mock_orch.return_value = mock_orchestrator
            mock_resolve.return_value = (mock_binding,)

            component.collect_via_orchestrator(
                config=config_with_market_bindings,
                connection=connection,
                registry=mock_registry,
                catalog=mock_catalog,
                dual_write=False,
            )

            # Verify backfill_binding was called instead of backfill_gaps
            mock_orchestrator.backfill_binding.assert_called_once()
            mock_orchestrator.backfill_gaps.assert_not_called()

    def test_collect_via_orchestrator_backfill_gaps(
        self,
        component: OrchestratorCollectionComponent,
        config_with_api_key: Any,
        mock_registry: MagicMock,
        mock_catalog: MagicMock,
    ) -> None:
        """Test collection uses backfill_gaps when no market bindings."""
        connection = "postgresql://user:pass@host:5432/db"

        with (
            patch.object(
                component, "create_sql_coverage_provider"
            ),
            patch.object(component, "create_sql_writer"),
            patch.object(component, "_create_ingestor"),
            patch.object(component, "_create_orchestrator") as mock_orch,
        ):
            mock_orchestrator = MagicMock()
            mock_orch.return_value = mock_orchestrator

            component.collect_via_orchestrator(
                config=config_with_api_key,
                connection=connection,
                registry=mock_registry,
                catalog=mock_catalog,
                dual_write=False,
            )

            # Verify backfill_gaps was called
            mock_orchestrator.backfill_gaps.assert_called()
            mock_orchestrator.backfill_binding.assert_not_called()


# -----------------------------------------------------------------------------
# Domain Loader Tests (1 test)
# -----------------------------------------------------------------------------


class TestDomainLoader:
    """Tests for domain loader protocol implementation."""

    def test_domain_loader_protocol_load(
        self,
        component: OrchestratorCollectionComponent,
        config_with_api_key: Any,
    ) -> None:
        """Test _create_domain_loader returns a working protocol implementation."""
        # Create the domain loader
        domain_loader = component._create_domain_loader(
            "test_api_key",
            config_with_api_key,
        )

        # Verify it has the load method
        assert hasattr(domain_loader, "load")
        assert callable(domain_loader.load)

        # Verify it stores the key and precision
        assert domain_loader._key == "test_api_key"
        assert domain_loader._price_precision == 6


# -----------------------------------------------------------------------------
# SQL Provider/Writer Initialization Tests (2 tests)
# -----------------------------------------------------------------------------


class TestSqlProviderInit:
    """Tests for SQL provider and writer initialization."""

    def test_orchestrator_sql_providers_init(
        self,
        component: OrchestratorCollectionComponent,
    ) -> None:
        """Test create_sql_coverage_provider creates SqlCoverageProvider."""
        with patch(
            "ml.stores.providers.SqlCoverageProvider"
        ) as mock_class:
            mock_provider = MagicMock()
            mock_class.return_value = mock_provider

            result = component.create_sql_coverage_provider(
                "postgresql://user:pass@host:5432/db",
                "market_data",
            )

            assert result is mock_provider
            mock_class.assert_called_once_with(
                connection_string="postgresql://user:pass@host:5432/db",
                table_name="market_data",
            )

    def test_orchestrator_raw_writer_init(
        self,
        component: OrchestratorCollectionComponent,
        mock_catalog: MagicMock,
    ) -> None:
        """Test create_raw_writer creates ParquetCatalogRawWriter."""
        with patch(
            "ml.stores.io_raw.ParquetCatalogRawWriter"
        ) as mock_class:
            mock_writer = MagicMock()
            mock_class.return_value = mock_writer

            result = component.create_raw_writer(mock_catalog)

            assert result is mock_writer
            mock_class.assert_called_once_with(mock_catalog)


# -----------------------------------------------------------------------------
# Protocol Compliance Test
# -----------------------------------------------------------------------------


class TestProtocolCompliance:
    """Tests for protocol compliance."""

    def test_component_satisfies_protocol(
        self,
        component: OrchestratorCollectionComponent,
    ) -> None:
        """Test OrchestratorCollectionComponent satisfies OrchestratorCollectionProtocol."""
        # Protocol compliance is structural in Python - verify methods exist
        assert hasattr(component, "collect_via_orchestrator")
        assert hasattr(component, "create_sql_coverage_provider")
        assert hasattr(component, "create_sql_writer")
        assert hasattr(component, "create_raw_writer")
        assert callable(component.collect_via_orchestrator)
        assert callable(component.create_sql_coverage_provider)
        assert callable(component.create_sql_writer)
        assert callable(component.create_raw_writer)
