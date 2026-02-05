"""
Unit tests for MLIntegrationManagerFacade.

This module tests the decomposed facade implementation of MLIntegrationManager,
verifying that it correctly delegates to the 7 component classes while preserving
the exact public API of the legacy implementation.

Test Design Reference: reports/tests/phase_3_6_test_design_report.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from ml.tests.utils.db import build_postgres_url


# =============================================================================
# Fixtures
# =============================================================================

TEST_DB_CONNECTION = build_postgres_url()
ALT_DB_CONNECTION = build_postgres_url(port="5433")


@pytest.fixture
def mock_db_candidates() -> tuple[str, ...]:
    """Provide mock database connection candidates."""
    return (
        TEST_DB_CONNECTION,
        ALT_DB_CONNECTION,
    )


@pytest.fixture
def mock_stores_bundle() -> dict[str, MagicMock]:
    """Provide mock stores with flush and health methods."""
    stores = {
        "feature_store": MagicMock(),
        "model_store": MagicMock(),
        "strategy_store": MagicMock(),
        "data_store": MagicMock(),
    }
    for store in stores.values():
        store.flush = MagicMock(return_value=None)
        store.get_statistics = MagicMock(return_value={})
    stores["data_store"].registry = MagicMock()
    return stores


@pytest.fixture
def mock_registries_bundle() -> dict[str, MagicMock]:
    """Provide mock registries with list methods."""
    registries = {
        "feature_registry": MagicMock(),
        "model_registry": MagicMock(),
        "strategy_registry": MagicMock(),
        "data_registry": MagicMock(),
    }
    registries["feature_registry"].list_features = MagicMock(return_value=[])
    registries["model_registry"].list_models = MagicMock(return_value=[])
    registries["strategy_registry"].list_strategies = MagicMock(return_value=[])
    registries["data_registry"].list_datasets = MagicMock(return_value=[])
    return registries


@dataclass
class MockConfig:
    """Mock configuration for testing."""

    db_connection: str | None = TEST_DB_CONNECTION
    use_dummy_stores: bool = False
    allow_dummy_fallback: bool = True


# =============================================================================
# Helper to create facade with mocked components
# =============================================================================


def create_mocked_facade(
    monkeypatch: pytest.MonkeyPatch,
    mock_stores: dict[str, MagicMock] | None = None,
    mock_registries: dict[str, MagicMock] | None = None,
    postgres_running: bool = False,
    ensure_healthy: bool = False,
) -> Any:
    """
    Create a facade with mocked components for testing.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.
    mock_stores : dict[str, MagicMock] | None
        Optional pre-configured mock stores.
    mock_registries : dict[str, MagicMock] | None
        Optional pre-configured mock registries.
    postgres_running : bool
        Whether to simulate PostgreSQL being available.
    ensure_healthy : bool
        Whether to run health checks on init.

    Returns
    -------
    MLIntegrationManagerFacade
        The mocked facade instance.

    """
    # Mock collect_postgres_candidates
    monkeypatch.setattr(
        "ml.core.integration_facade.collect_postgres_candidates",
        lambda *args, **kwargs: MagicMock(
            urls=(TEST_DB_CONNECTION,),
        ),
    )

    # Mock DatabaseLifecycleComponent.is_postgres_running
    monkeypatch.setattr(
        "ml.core.common.database_lifecycle.DatabaseLifecycleComponent.is_postgres_running",
        lambda self: postgres_running,
    )

    # Mock store initialization
    if mock_stores:
        monkeypatch.setattr(
            "ml.core.common.store_initialization.StoreInitializationComponent.init_stores",
            lambda self: None,
        )
        # Set stores directly
        def set_mock_stores(self: Any) -> None:
            self.feature_store = mock_stores.get("feature_store")
            self.model_store = mock_stores.get("model_store")
            self.strategy_store = mock_stores.get("strategy_store")
            self.data_store = mock_stores.get("data_store")

        monkeypatch.setattr(
            "ml.core.common.store_initialization.StoreInitializationComponent.init_stores",
            set_mock_stores,
        )
    else:
        # Use dummy stores
        monkeypatch.setattr(
            "ml.core.common.store_initialization.StoreInitializationComponent.enable_file_fallback",
            lambda self: False,
        )

    # Mock registry initialization
    if mock_registries:

        def set_mock_registries(self: Any) -> None:
            self.feature_registry = mock_registries.get("feature_registry")
            self.model_registry = mock_registries.get("model_registry")
            self.strategy_registry = mock_registries.get("strategy_registry")
            self.data_registry = mock_registries.get("data_registry")
            self.persistence_config = MagicMock()

        monkeypatch.setattr(
            "ml.core.common.registry_initialization.RegistryInitializationComponent.init_registries",
            set_mock_registries,
        )
        monkeypatch.setattr(
            "ml.core.common.registry_initialization.RegistryInitializationComponent.create_data_store",
            lambda self: MagicMock(),
        )
        monkeypatch.setattr(
            "ml.core.common.registry_initialization.RegistryInitializationComponent.inject_data_registry_into_stores",
            lambda self, fs, ms: None,
        )

    # Mock partition manager
    monkeypatch.setattr(
        "ml.core.integration_facade.MLIntegrationManagerFacade._init_partition_manager",
        lambda self: None,
    )

    from ml.core.integration_facade import MLIntegrationManagerFacade

    return MLIntegrationManagerFacade(
        db_connection=TEST_DB_CONNECTION,
        auto_start_postgres=False,
        auto_migrate=False,
        ensure_healthy=ensure_healthy,
        strict_protocol_validation=False,
    )


# =============================================================================
# Happy Path Tests
# =============================================================================


class TestFacadeInit:
    """Tests for facade initialization."""

    def test_facade_init_delegates_to_components(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_stores_bundle: dict[str, MagicMock],
        mock_registries_bundle: dict[str, MagicMock],
    ) -> None:
        """
        Verify facade initialization delegates to components correctly.

        The facade should initialize all 7 components and wire their
        attributes to the facade's public interface.

        """
        facade = create_mocked_facade(
            monkeypatch,
            mock_stores=mock_stores_bundle,
            mock_registries=mock_registries_bundle,
            postgres_running=False,
            ensure_healthy=False,
        )

        # Verify component references exist
        assert hasattr(facade, "_db_lifecycle")
        assert hasattr(facade, "_store_init")
        assert hasattr(facade, "_registry_init")
        assert hasattr(facade, "_health_monitoring")
        assert hasattr(facade, "_observability")
        assert hasattr(facade, "_actor_factory")
        assert hasattr(facade, "_event_ingestion")

        # Verify stores are wired
        assert facade.feature_store is not None
        assert facade.model_store is not None
        assert facade.strategy_store is not None

        # Verify registries are wired
        assert facade.feature_registry is not None
        assert facade.model_registry is not None
        assert facade.strategy_registry is not None
        assert facade.data_registry is not None

    def test_facade_accepts_legacy_config_parameter(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Verify facade accepts config with db_connection attribute.

        This ensures backward compatibility with legacy config patterns.

        """
        test_db_conn = build_postgres_url(
            user="test",
            password="test",
            database="db",
        )

        # Mock dependencies - return the config's db_connection
        monkeypatch.setattr(
            "ml.core.integration_facade.collect_postgres_candidates",
            lambda *args, **kwargs: MagicMock(urls=(test_db_conn,)),
        )
        monkeypatch.setattr(
            "ml.core.common.database_lifecycle.DatabaseLifecycleComponent.is_postgres_running",
            lambda self: False,
        )
        monkeypatch.setattr(
            "ml.core.common.store_initialization.StoreInitializationComponent.enable_file_fallback",
            lambda self: False,
        )
        monkeypatch.setattr(
            "ml.core.integration_facade.MLIntegrationManagerFacade._init_partition_manager",
            lambda self: None,
        )

        config = MockConfig(db_connection=test_db_conn)

        from ml.core.integration_facade import MLIntegrationManagerFacade

        facade = MLIntegrationManagerFacade(
            config=config,
            auto_start_postgres=False,
            auto_migrate=False,
            ensure_healthy=False,
            strict_protocol_validation=False,
        )

        assert facade.db_connection == test_db_conn


class TestFacadeHealthChecks:
    """Tests for facade health checking methods."""

    def test_facade_check_health_aggregates_component_health(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_stores_bundle: dict[str, MagicMock],
        mock_registries_bundle: dict[str, MagicMock],
    ) -> None:
        """
        Verify facade health aggregation matches legacy behavior.

        The check_health method should return a dict with health status
        for each component.

        """
        facade = create_mocked_facade(
            monkeypatch,
            mock_stores=mock_stores_bundle,
            mock_registries=mock_registries_bundle,
            postgres_running=False,
            ensure_healthy=False,
        )

        # Mock the health monitoring component's check_health
        mock_health = {
            "postgres": False,
            "feature_store": True,
            "model_store": True,
            "strategy_store": True,
            "feature_registry": True,
            "model_registry": True,
            "strategy_registry": True,
            "data_registry": True,
            "data_store": True,
            "partitions": False,
        }
        facade._health_monitoring.check_health = MagicMock(return_value=mock_health)

        health = facade.check_health()

        assert isinstance(health, dict)
        assert "postgres" in health
        assert "feature_store" in health
        assert "model_store" in health
        assert "strategy_store" in health
        assert "feature_registry" in health
        assert "model_registry" in health
        assert "strategy_registry" in health
        assert "data_registry" in health
        assert "data_store" in health
        assert "partitions" in health


class TestFacadeShutdown:
    """Tests for facade shutdown behavior."""

    def test_facade_shutdown_delegates_to_actor_factory(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_stores_bundle: dict[str, MagicMock],
        mock_registries_bundle: dict[str, MagicMock],
    ) -> None:
        """
        Verify shutdown delegates to ActorFactoryComponent.

        All stores should be flushed during shutdown.

        """
        facade = create_mocked_facade(
            monkeypatch,
            mock_stores=mock_stores_bundle,
            mock_registries=mock_registries_bundle,
            postgres_running=False,
            ensure_healthy=False,
        )

        # Track shutdown calls
        shutdown_called = False
        original_shutdown = facade._actor_factory.shutdown

        def mock_shutdown() -> None:
            nonlocal shutdown_called
            shutdown_called = True
            original_shutdown()

        facade._actor_factory.shutdown = mock_shutdown

        facade.shutdown()

        assert shutdown_called


class TestFacadeConfigStubs:
    """Tests for no-op configuration stub methods."""

    def test_facade_config_stub_methods_return_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_stores_bundle: dict[str, MagicMock],
        mock_registries_bundle: dict[str, MagicMock],
    ) -> None:
        """
        Verify all config stub methods return None.

        These are TDD convenience hooks maintained for backward compatibility.

        """
        facade = create_mocked_facade(
            monkeypatch,
            mock_stores=mock_stores_bundle,
            mock_registries=mock_registries_bundle,
            postgres_running=False,
            ensure_healthy=False,
        )

        # All stub methods should return None
        assert facade.configure_message_bus() is None
        assert facade.configure_event_emission() is None
        assert facade.configure_event_system() is None
        assert facade.configure_domain_bookkeeping(MagicMock()) is None
        assert facade.start_end_to_end_tracking() is None
        assert facade.start_health_checks() is None
        assert facade.emit_cross_domain_event({}) is None


class TestFacadeObservability:
    """Tests for facade observability methods."""

    def test_facade_initialize_observability_pipeline(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_stores_bundle: dict[str, MagicMock],
        mock_registries_bundle: dict[str, MagicMock],
    ) -> None:
        """
        Verify observability pipeline initialization.

        """
        facade = create_mocked_facade(
            monkeypatch,
            mock_stores=mock_stores_bundle,
            mock_registries=mock_registries_bundle,
            postgres_running=False,
            ensure_healthy=False,
        )

        # Mock the observability component
        mock_service = MagicMock()
        facade._observability.observability_service = None
        facade._observability.initialize_observability_pipeline = MagicMock()

        def set_service() -> None:
            facade._observability.observability_service = mock_service

        facade._observability.initialize_observability_pipeline.side_effect = set_service

        facade.initialize_observability_pipeline()

        facade._observability.initialize_observability_pipeline.assert_called_once()

    def test_facade_collect_observability_dataframes_returns_dict(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_stores_bundle: dict[str, MagicMock],
        mock_registries_bundle: dict[str, MagicMock],
    ) -> None:
        """
        Verify collect_observability_dataframes returns expected dict.

        """
        facade = create_mocked_facade(
            monkeypatch,
            mock_stores=mock_stores_bundle,
            mock_registries=mock_registries_bundle,
            postgres_running=False,
            ensure_healthy=False,
        )

        expected_keys = {"latency", "metrics", "correlation", "health"}
        mock_result = dict.fromkeys(expected_keys)
        facade._observability.collect_observability_dataframes = MagicMock(
            return_value=mock_result
        )

        result = facade.collect_observability_dataframes()

        assert isinstance(result, dict)
        assert set(result.keys()) == expected_keys


class TestFacadeEventIngestion:
    """Tests for facade event ingestion methods."""

    def test_facade_ingest_events_delegates_to_component(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_stores_bundle: dict[str, MagicMock],
        mock_registries_bundle: dict[str, MagicMock],
    ) -> None:
        """
        Verify ingest_events delegates to EventIngestionComponent.

        """
        facade = create_mocked_facade(
            monkeypatch,
            mock_stores=mock_stores_bundle,
            mock_registries=mock_registries_bundle,
            postgres_running=False,
            ensure_healthy=False,
        )

        mock_config = MagicMock()
        expected_path = Path("/test/events.parquet")
        facade._event_ingestion.ingest_events = MagicMock(return_value=expected_path)

        result = facade.ingest_events(mock_config)

        facade._event_ingestion.ingest_events.assert_called_once_with(mock_config)
        assert result == expected_path


class TestFacadeActorFactory:
    """Tests for facade actor factory methods."""

    def test_facade_create_integrated_actor_delegates_to_component(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_stores_bundle: dict[str, MagicMock],
        mock_registries_bundle: dict[str, MagicMock],
    ) -> None:
        """
        Verify create_integrated_actor delegates to ActorFactoryComponent.

        """
        facade = create_mocked_facade(
            monkeypatch,
            mock_stores=mock_stores_bundle,
            mock_registries=mock_registries_bundle,
            postgres_running=False,
            ensure_healthy=False,
        )

        mock_actor_class = MagicMock
        mock_config = MagicMock()
        expected_actor = MagicMock()
        facade._actor_factory.create_integrated_actor = MagicMock(
            return_value=expected_actor
        )

        result = facade.create_integrated_actor(mock_actor_class, mock_config)

        facade._actor_factory.create_integrated_actor.assert_called_once_with(
            mock_actor_class, mock_config
        )
        assert result == expected_actor

    def test_facade_set_message_publisher_delegates_to_component(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_stores_bundle: dict[str, MagicMock],
        mock_registries_bundle: dict[str, MagicMock],
    ) -> None:
        """
        Verify set_message_publisher delegates to ActorFactoryComponent.

        """
        facade = create_mocked_facade(
            monkeypatch,
            mock_stores=mock_stores_bundle,
            mock_registries=mock_registries_bundle,
            postgres_running=False,
            ensure_healthy=False,
        )

        mock_publisher = MagicMock()
        facade._actor_factory.set_message_publisher = MagicMock()

        facade.set_message_publisher(mock_publisher)

        facade._actor_factory.set_message_publisher.assert_called_once_with(
            mock_publisher
        )


class TestFacadeEmitCascade:
    """Tests for facade emit_cascade method."""

    def test_facade_emit_cascade_preserves_correlation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_stores_bundle: dict[str, MagicMock],
        mock_registries_bundle: dict[str, MagicMock],
    ) -> None:
        """
        Verify emit_cascade preserves correlation_id.

        """
        facade = create_mocked_facade(
            monkeypatch,
            mock_stores=mock_stores_bundle,
            mock_registries=mock_registries_bundle,
            postgres_running=False,
            ensure_healthy=False,
        )

        source_event = {
            "domain": "features",
            "event_type": "feature_computed",
            "correlation_id": "test-correlation-123",
            "instrument_id": "BTC.USD",
            "ts_event": 1000000000,
            "event_id": "evt_001",
            "payload": {"feature_name": "price_sma_20"},
        }

        expected_result = {
            "domain": "model",
            "correlation_id": "test-correlation-123",
        }
        facade._actor_factory.emit_cascade = MagicMock(return_value=expected_result)

        result = facade.emit_cascade(source_event, "model", delay_ns=100)

        assert result["correlation_id"] == source_event["correlation_id"]
        assert result["domain"] == "model"


class TestFacadeLegacyInternalMethods:
    """Tests for legacy internal methods exposed for backward compatibility."""

    def test_facade_is_postgres_running_delegates_to_db_lifecycle(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_stores_bundle: dict[str, MagicMock],
        mock_registries_bundle: dict[str, MagicMock],
    ) -> None:
        """
        Verify _is_postgres_running delegates to DatabaseLifecycleComponent.

        """
        facade = create_mocked_facade(
            monkeypatch,
            mock_stores=mock_stores_bundle,
            mock_registries=mock_registries_bundle,
            postgres_running=False,
            ensure_healthy=False,
        )

        facade._db_lifecycle.is_postgres_running = MagicMock(return_value=True)

        result = facade._is_postgres_running()

        assert result is True
        facade._db_lifecycle.is_postgres_running.assert_called_once()

    def test_facade_properties_expose_internal_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_stores_bundle: dict[str, MagicMock],
        mock_registries_bundle: dict[str, MagicMock],
    ) -> None:
        """
        Verify legacy properties expose internal component state.

        """
        facade = create_mocked_facade(
            monkeypatch,
            mock_stores=mock_stores_bundle,
            mock_registries=mock_registries_bundle,
            postgres_running=False,
            ensure_healthy=False,
        )

        # These should not raise
        _ = facade._json_fallback
        _ = facade._file_fallback
        _ = facade._file_store_path
        _ = facade._connection_candidates
        _ = facade._allow_dummy


class TestModuleLevelFunctions:
    """Tests for module-level functions."""

    def test_get_integration_manager_returns_singleton(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Verify get_integration_manager returns singleton instance.

        """
        from ml.core import integration_facade

        # Reset singleton
        integration_facade._integration_manager = None

        # Mock facade creation
        mock_facade = MagicMock()
        monkeypatch.setattr(
            integration_facade,
            "MLIntegrationManagerFacade",
            MagicMock(return_value=mock_facade),
        )

        result1 = integration_facade.get_integration_manager()
        result2 = integration_facade.get_integration_manager()

        assert result1 is result2

        # Cleanup
        integration_facade._integration_manager = None

    def test_reset_integration_manager_clears_singleton(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Verify reset_integration_manager clears the singleton.

        """
        from ml.core import integration_facade

        # Set up a mock singleton
        mock_facade = MagicMock()
        integration_facade._integration_manager = mock_facade

        integration_facade.reset_integration_manager()

        assert integration_facade._integration_manager is None
        mock_facade.shutdown.assert_called_once()


# =============================================================================
# Error Condition Tests
# =============================================================================


class TestFacadeErrorConditions:
    """Tests for error handling in the facade."""

    def test_facade_init_raises_when_no_candidates(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Verify facade raises ValueError when no connection candidates.

        """
        monkeypatch.setattr(
            "ml.core.integration_facade.collect_postgres_candidates",
            lambda *args, **kwargs: MagicMock(urls=()),
        )

        from ml.core.integration_facade import MLIntegrationManagerFacade

        with pytest.raises(ValueError, match="No PostgreSQL connection candidates"):
            MLIntegrationManagerFacade(
                auto_start_postgres=False,
                auto_migrate=False,
                ensure_healthy=False,
            )

    def test_facade_ensure_healthy_raises_when_unhealthy(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_stores_bundle: dict[str, MagicMock],
        mock_registries_bundle: dict[str, MagicMock],
    ) -> None:
        """
        Verify ensure_healthy raises RuntimeError when components are unhealthy.

        """
        facade = create_mocked_facade(
            monkeypatch,
            mock_stores=mock_stores_bundle,
            mock_registries=mock_registries_bundle,
            postgres_running=False,
            ensure_healthy=False,
        )

        # Mock health monitoring to raise
        def mock_ensure_healthy() -> None:
            raise RuntimeError("Unhealthy components: ['postgres']")

        facade._health_monitoring.ensure_healthy = mock_ensure_healthy

        with pytest.raises(RuntimeError, match="Unhealthy components"):
            facade.ensure_healthy()
