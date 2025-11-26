"""
Parity tests for MLIntegrationManager: Legacy vs Facade.

This module ensures the decomposed MLIntegrationManagerFacade behaves identically
to the legacy MLIntegrationManager implementation. These tests are critical for
validating the god-class decomposition (Phase 3.6).

Test Design Reference: reports/tests/phase_3_6_test_design_report.md

Parity Test Strategy:
1. Create both legacy and facade instances with identical configuration
2. Execute the same operations on both
3. Assert identical results and behavior

Feature Flag:
- ML_USE_LEGACY_INTEGRATION_MANAGER=1 uses legacy implementation
- ML_USE_LEGACY_INTEGRATION_MANAGER=0 (default) uses facade implementation
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Fixtures
# =============================================================================


@dataclass
class MockConfig:
    """Mock configuration for testing."""

    db_connection: str | None = "postgresql://postgres:postgres@localhost:5432/nautilus"
    use_dummy_stores: bool = True
    allow_dummy_fallback: bool = True


@pytest.fixture
def mock_postgres_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock PostgreSQL as unavailable for testing fallback paths."""
    # Mock for both legacy and facade
    monkeypatch.setattr(
        "ml.core.integration.MLIntegrationManager._is_postgres_running",
        lambda self: False,
    )
    monkeypatch.setattr(
        "ml.core.integration.MLIntegrationManager._enable_file_fallback",
        lambda self: False,
    )

    # For facade components
    monkeypatch.setattr(
        "ml.core.common.database_lifecycle.DatabaseLifecycleComponent.is_postgres_running",
        lambda self: False,
    )
    monkeypatch.setattr(
        "ml.core.common.store_initialization.StoreInitializationComponent.enable_file_fallback",
        lambda self: False,
    )


@pytest.fixture
def mock_db_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock database connection candidates for both implementations."""
    mock_candidates = MagicMock(
        urls=("postgresql://postgres:postgres@localhost:5432/nautilus",)
    )
    monkeypatch.setattr(
        "ml.core.integration.collect_postgres_candidates",
        lambda *args, **kwargs: mock_candidates,
    )
    monkeypatch.setattr(
        "ml.core.integration_facade.collect_postgres_candidates",
        lambda *args, **kwargs: mock_candidates,
    )


def create_legacy_manager(
    monkeypatch: pytest.MonkeyPatch,
    ensure_healthy: bool = False,
) -> Any:
    """Create a legacy MLIntegrationManager with mocked dependencies."""
    # Mock collect_postgres_candidates
    mock_candidates = MagicMock(
        urls=("postgresql://postgres:postgres@localhost:5432/nautilus",)
    )
    monkeypatch.setattr(
        "ml.core.integration.collect_postgres_candidates",
        lambda *args, **kwargs: mock_candidates,
    )
    # Mock PostgreSQL unavailable
    monkeypatch.setattr(
        "ml.core.integration.MLIntegrationManager._is_postgres_running",
        lambda self: False,
    )
    monkeypatch.setattr(
        "ml.core.integration.MLIntegrationManager._enable_file_fallback",
        lambda self: False,
    )
    monkeypatch.setattr(
        "ml.core.integration.MLIntegrationManager._init_partition_manager",
        lambda self: None,
    )
    monkeypatch.setattr(
        "ml.core.integration.MLIntegrationManager._maybe_run_backfill_on_start",
        lambda self: None,
    )

    from ml.core.integration import MLIntegrationManager

    return MLIntegrationManager(
        db_connection="postgresql://postgres:postgres@localhost:5432/nautilus",
        auto_start_postgres=False,
        auto_migrate=False,
        ensure_healthy=ensure_healthy,
        strict_protocol_validation=False,
    )


def create_facade_manager(
    monkeypatch: pytest.MonkeyPatch,
    ensure_healthy: bool = False,
) -> Any:
    """Create an MLIntegrationManagerFacade with mocked dependencies."""
    # Mock collect_postgres_candidates
    mock_candidates = MagicMock(
        urls=("postgresql://postgres:postgres@localhost:5432/nautilus",)
    )
    monkeypatch.setattr(
        "ml.core.integration_facade.collect_postgres_candidates",
        lambda *args, **kwargs: mock_candidates,
    )
    # Mock PostgreSQL unavailable
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
    monkeypatch.setattr(
        "ml.core.integration_facade.MLIntegrationManagerFacade._maybe_run_backfill_on_start",
        lambda self: None,
    )

    from ml.core.integration_facade import MLIntegrationManagerFacade

    return MLIntegrationManagerFacade(
        db_connection="postgresql://postgres:postgres@localhost:5432/nautilus",
        auto_start_postgres=False,
        auto_migrate=False,
        ensure_healthy=ensure_healthy,
        strict_protocol_validation=False,
    )


# =============================================================================
# Parity Tests
# =============================================================================


class TestCheckHealthParity:
    """Test check_health() parity between legacy and facade."""

    def test_check_health_parity_legacy_vs_facade(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Verify check_health returns identical structure in both modes.

        Both implementations should return a dict with the same keys.

        """
        legacy = create_legacy_manager(monkeypatch)
        facade = create_facade_manager(monkeypatch)

        legacy_health = legacy.check_health()
        facade_health = facade.check_health()

        # Verify same keys
        assert set(legacy_health.keys()) == set(facade_health.keys())

        # Verify expected keys present
        expected_keys = {
            "postgres",
            "feature_store",
            "model_store",
            "strategy_store",
            "feature_registry",
            "model_registry",
            "strategy_registry",
            "data_registry",
            "data_store",
            "partitions",
        }
        assert expected_keys.issubset(set(legacy_health.keys()))
        assert expected_keys.issubset(set(facade_health.keys()))


class TestAggregateHealthParity:
    """Test aggregate_health() parity between legacy and facade."""

    def test_aggregate_health_parity_legacy_vs_facade(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Verify aggregate_health returns identical structure in both modes.

        Both implementations should return a dict with:
        - components: per-component health
        - domains: domain-level health
        - system: overall system health

        """
        legacy = create_legacy_manager(monkeypatch)
        facade = create_facade_manager(monkeypatch)

        legacy_summary = legacy.aggregate_health()
        facade_summary = facade.aggregate_health()

        # Verify same top-level keys
        assert set(legacy_summary.keys()) == set(facade_summary.keys())
        assert "components" in legacy_summary
        assert "domains" in legacy_summary
        assert "system" in legacy_summary

        # Verify domains structure
        assert set(legacy_summary["domains"].keys()) == set(facade_summary["domains"].keys())
        expected_domains = {"data", "features", "model", "strategy"}
        assert expected_domains == set(legacy_summary["domains"].keys())

        # Verify system structure
        assert "healthy" in legacy_summary["system"]
        assert "unhealthy" in legacy_summary["system"]
        assert "healthy" in facade_summary["system"]
        assert "unhealthy" in facade_summary["system"]


class TestInitStoresParity:
    """Test store initialization parity between legacy and facade."""

    def test_init_stores_parity_legacy_vs_facade(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Verify store initialization creates same types in both modes.

        In fallback mode (PostgreSQL unavailable), both implementations
        should create DummyStore instances.

        """
        legacy = create_legacy_manager(monkeypatch)
        facade = create_facade_manager(monkeypatch)

        # Both should be using dummy stores in fallback mode
        assert legacy.feature_store is not None
        assert facade.feature_store is not None
        assert legacy.model_store is not None
        assert facade.model_store is not None
        assert legacy.strategy_store is not None
        assert facade.strategy_store is not None

        # Type names should match (both DummyStore in fallback)
        assert type(legacy.feature_store).__name__ == type(facade.feature_store).__name__
        assert type(legacy.model_store).__name__ == type(facade.model_store).__name__
        assert type(legacy.strategy_store).__name__ == type(facade.strategy_store).__name__


class TestInitRegistriesParity:
    """Test registry initialization parity between legacy and facade."""

    def test_init_registries_parity_legacy_vs_facade(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Verify registry initialization creates same types in both modes.

        """
        legacy = create_legacy_manager(monkeypatch)
        facade = create_facade_manager(monkeypatch)

        # Both should have all 4 registries
        assert legacy.feature_registry is not None
        assert facade.feature_registry is not None
        assert legacy.model_registry is not None
        assert facade.model_registry is not None
        assert legacy.strategy_registry is not None
        assert facade.strategy_registry is not None
        assert legacy.data_registry is not None
        assert facade.data_registry is not None

        # Type names should match
        assert type(legacy.feature_registry).__name__ == type(facade.feature_registry).__name__
        assert type(legacy.model_registry).__name__ == type(facade.model_registry).__name__
        assert type(legacy.strategy_registry).__name__ == type(facade.strategy_registry).__name__
        assert type(legacy.data_registry).__name__ == type(facade.data_registry).__name__


class TestShutdownParity:
    """Test shutdown() parity between legacy and facade."""

    def test_shutdown_parity_legacy_vs_facade(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Verify shutdown behavior is identical in both modes.

        Both implementations should flush all stores on shutdown.

        """
        legacy = create_legacy_manager(monkeypatch)
        facade = create_facade_manager(monkeypatch)

        # Mock flush methods to track calls
        legacy_flushes = []
        facade_flushes = []

        for store_name in ["feature_store", "model_store", "strategy_store"]:
            legacy_store = getattr(legacy, store_name, None)
            facade_store = getattr(facade, store_name, None)

            if legacy_store and hasattr(legacy_store, "flush"):
                original_flush = legacy_store.flush

                def make_tracker(name: str, tracked: list[str]) -> Any:
                    def tracker() -> None:
                        tracked.append(name)

                    return tracker

                legacy_store.flush = make_tracker(store_name, legacy_flushes)

            if facade_store and hasattr(facade_store, "flush"):
                facade_store.flush = make_tracker(store_name, facade_flushes)

        legacy.shutdown()
        facade.shutdown()

        # Both should attempt to flush the same stores
        assert set(legacy_flushes) == set(facade_flushes)


class TestFeatureFlagSwitching:
    """Test feature flag toggles between implementations."""

    def test_feature_flag_switches_implementation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Verify ML_USE_LEGACY_INTEGRATION_MANAGER toggles implementation.

        """
        from ml.core.integration_facade import _use_legacy_integration_manager

        # Default should be False (use facade)
        monkeypatch.delenv("ML_USE_LEGACY_INTEGRATION_MANAGER", raising=False)
        assert _use_legacy_integration_manager() is False

        # Set to 1 should return True (use legacy)
        monkeypatch.setenv("ML_USE_LEGACY_INTEGRATION_MANAGER", "1")
        assert _use_legacy_integration_manager() is True

        # Set to 0 should return False (use facade)
        monkeypatch.setenv("ML_USE_LEGACY_INTEGRATION_MANAGER", "0")
        assert _use_legacy_integration_manager() is False


class TestConfigStubsParity:
    """Test configuration stub methods parity."""

    def test_config_stubs_parity_legacy_vs_facade(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Verify all config stub methods return None in both implementations.

        """
        legacy = create_legacy_manager(monkeypatch)
        facade = create_facade_manager(monkeypatch)

        # All stubs should return None
        assert legacy.configure_message_bus() is None
        assert facade.configure_message_bus() is None

        assert legacy.configure_event_emission() is None
        assert facade.configure_event_emission() is None

        assert legacy.configure_event_system() is None
        assert facade.configure_event_system() is None

        assert legacy.configure_domain_bookkeeping(MagicMock()) is None
        assert facade.configure_domain_bookkeeping(MagicMock()) is None

        assert legacy.start_end_to_end_tracking() is None
        assert facade.start_end_to_end_tracking() is None

        assert legacy.start_health_checks() is None
        assert facade.start_health_checks() is None

        assert legacy.emit_cross_domain_event({}) is None
        assert facade.emit_cross_domain_event({}) is None


class TestEmitCascadeParity:
    """Test emit_cascade parity between legacy and facade."""

    def test_emit_cascade_parity_legacy_vs_facade(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Verify emit_cascade produces identical results in both implementations.

        """
        legacy = create_legacy_manager(monkeypatch)
        facade = create_facade_manager(monkeypatch)

        source_event = {
            "domain": "features",
            "event_type": "feature_computed",
            "correlation_id": "test-correlation-123",
            "instrument_id": "BTC.USD",
            "ts_event": 1000000000,
            "event_id": "evt_001",
            "payload": {"feature_name": "sma_20"},
        }

        legacy_result = legacy.emit_cascade(source_event, "model", delay_ns=100)
        facade_result = facade.emit_cascade(source_event, "model", delay_ns=100)

        # Correlation should be preserved in both
        assert legacy_result["correlation_id"] == source_event["correlation_id"]
        assert facade_result["correlation_id"] == source_event["correlation_id"]

        # Domain should be updated in both
        assert legacy_result["domain"] == "model"
        assert facade_result["domain"] == "model"


class TestObservabilityParity:
    """Test observability methods parity."""

    def test_collect_observability_dataframes_parity(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Verify collect_observability_dataframes returns same structure.

        """
        legacy = create_legacy_manager(monkeypatch)
        facade = create_facade_manager(monkeypatch)

        # Without initializing observability, both should return dict with None values
        legacy_dfs = legacy.collect_observability_dataframes()
        facade_dfs = facade.collect_observability_dataframes()

        assert set(legacy_dfs.keys()) == set(facade_dfs.keys())
        expected_keys = {"latency", "metrics", "correlation", "health"}
        assert expected_keys == set(legacy_dfs.keys())

    def test_get_observability_async_status_parity(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Verify get_observability_async_status returns same structure.

        """
        legacy = create_legacy_manager(monkeypatch)
        facade = create_facade_manager(monkeypatch)

        legacy_status = legacy.get_observability_async_status()
        facade_status = facade.get_observability_async_status()

        # Both should have same keys
        assert set(legacy_status.keys()) == set(facade_status.keys())
        assert "running" in legacy_status
        assert "queue_size" in legacy_status

        # When not running, both should report False/0
        assert legacy_status["running"] is False
        assert facade_status["running"] is False
        assert legacy_status["queue_size"] == 0
        assert facade_status["queue_size"] == 0


class TestAttributeParity:
    """Test public attribute parity."""

    def test_public_attributes_exist_in_both(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Verify both implementations expose the same public attributes.

        """
        legacy = create_legacy_manager(monkeypatch)
        facade = create_facade_manager(monkeypatch)

        # All public attributes from legacy should exist in facade
        public_attrs = [
            "feature_store",
            "model_store",
            "strategy_store",
            "data_store",
            "feature_registry",
            "model_registry",
            "strategy_registry",
            "data_registry",
            "partition_manager",
            "db_connection",
            "auto_start_postgres",
            "auto_migrate",
        ]

        for attr in public_attrs:
            assert hasattr(legacy, attr), f"Legacy missing attribute: {attr}"
            assert hasattr(facade, attr), f"Facade missing attribute: {attr}"


class TestMethodSignatureParity:
    """Test method signature parity."""

    def test_public_methods_exist_in_both(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Verify both implementations expose the same public methods.

        """
        legacy = create_legacy_manager(monkeypatch)
        facade = create_facade_manager(monkeypatch)

        public_methods = [
            "ingest_events",
            "ensure_healthy",
            "aggregate_health",
            "check_health",
            "create_integrated_actor",
            "shutdown",
            "configure_message_bus",
            "configure_event_emission",
            "configure_event_system",
            "configure_domain_bookkeeping",
            "initialize_observability_pipeline",
            "start_end_to_end_tracking",
            "start_health_checks",
            "collect_observability_dataframes",
            "flush_observability_to_path",
            "flush_observability_to_db",
            "start_observability_flush",
            "stop_observability_flush",
            "start_observability_from_config",
            "stop_observability_async",
            "get_observability_async_status",
            "start_observability_from_env",
            "emit_cross_domain_event",
            "emit_cascade",
            "set_message_publisher",
        ]

        for method in public_methods:
            assert hasattr(legacy, method) and callable(
                getattr(legacy, method)
            ), f"Legacy missing method: {method}"
            assert hasattr(facade, method) and callable(
                getattr(facade, method)
            ), f"Facade missing method: {method}"
