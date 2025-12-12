"""
Unit tests for ActorFactoryComponent.

This module tests the actor factory component extracted from MLIntegrationManager
(Phase 3.6.6). Tests cover:

- Happy path: actor creation, shutdown, message publisher attachment
- Error conditions: frozen config, flush exceptions
- Edge cases: no data store, config stub methods

"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from ml.core.common.actor_factory import ActorFactoryComponent
from ml.tests.utils.db import build_postgres_url


TEST_DB_CONNECTION = build_postgres_url()


if TYPE_CHECKING:
    pass


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_feature_store() -> MagicMock:
    """Provide a mock feature store with flush method."""
    store = MagicMock()
    store.flush = MagicMock(return_value=None)
    return store


@pytest.fixture
def mock_model_store() -> MagicMock:
    """Provide a mock model store with flush method."""
    store = MagicMock()
    store.flush = MagicMock(return_value=None)
    return store


@pytest.fixture
def mock_strategy_store() -> MagicMock:
    """Provide a mock strategy store with flush method."""
    store = MagicMock()
    store.flush = MagicMock(return_value=None)
    return store


@pytest.fixture
def mock_data_store() -> MagicMock:
    """Provide a mock data store with flush and publisher attributes."""
    store = MagicMock()
    store.flush = MagicMock(return_value=None)
    store.publisher = None
    return store


@pytest.fixture
def actor_factory_component(
    mock_feature_store: MagicMock,
    mock_model_store: MagicMock,
    mock_strategy_store: MagicMock,
    mock_data_store: MagicMock,
) -> ActorFactoryComponent:
    """Provide a fully configured ActorFactoryComponent."""
    return ActorFactoryComponent(
        db_connection=TEST_DB_CONNECTION,
        feature_store=mock_feature_store,
        model_store=mock_model_store,
        strategy_store=mock_strategy_store,
        data_store=mock_data_store,
    )


# =============================================================================
# Happy Path Tests
# =============================================================================


class TestHappyPath:
    """Tests for successful operation paths."""

    def test_create_integrated_actor_attaches_db_connection(
        self,
        actor_factory_component: ActorFactoryComponent,
    ) -> None:
        """Verify db_connection attached to config.

        Input: Actor class, config without db_connection.
        Expected Behavior: Actor created with db_connection in config.
        """

        # Simple config class without db_connection
        @dataclass
        class SimpleConfig:
            name: str = "test"

        config = SimpleConfig()

        # Simple actor class that stores config
        class SimpleActor:
            def __init__(self, config: object) -> None:
                self.config = config

        actor = actor_factory_component.create_integrated_actor(SimpleActor, config)

        assert hasattr(actor, "config")
        assert actor.config.db_connection == actor_factory_component.db_connection

    def test_shutdown_flushes_all_stores(
        self,
        actor_factory_component: ActorFactoryComponent,
        mock_feature_store: MagicMock,
        mock_model_store: MagicMock,
        mock_strategy_store: MagicMock,
        mock_data_store: MagicMock,
    ) -> None:
        """Verify graceful shutdown flushes all stores.

        Input: Component with all stores.
        Expected Behavior: All store flush() methods called.
        """
        actor_factory_component.shutdown()

        mock_feature_store.flush.assert_called_once()
        mock_model_store.flush.assert_called_once()
        mock_strategy_store.flush.assert_called_once()
        mock_data_store.flush.assert_called_once()

    def test_set_message_publisher_attaches_to_data_store(
        self,
        actor_factory_component: ActorFactoryComponent,
        mock_data_store: MagicMock,
    ) -> None:
        """Verify publisher attachment to data store.

        Input: DataStore with publisher attribute.
        Expected Behavior: Publisher assigned.
        """
        mock_publisher = MagicMock()

        actor_factory_component.set_message_publisher(mock_publisher)

        assert mock_data_store.publisher is mock_publisher

    def test_emit_cascade_preserves_correlation(
        self,
        actor_factory_component: ActorFactoryComponent,
    ) -> None:
        """Verify cascade event preserves correlation_id.

        Input: Source event with correlation_id.
        Expected Behavior: Target event has same correlation_id and updated domain.
        """
        source_event: dict[str, Any] = {
            "domain": "features",
            "event_type": "feature_computed",
            "correlation_id": "abc123",
            "instrument_id": "BTC.USD",
            "ts_event": 1000000000,
            "event_id": "evt_001",
            "payload": {"feature_name": "sma_20"},
        }

        result = actor_factory_component.emit_cascade(
            source_event, "model", delay_ns=100
        )

        assert result["correlation_id"] == "abc123"
        assert result["domain"] == "model"
        assert result["ts_event"] == 1000000100  # original + delay
        assert result["source_event_id"] == "evt_001"

    def test_emit_cascade_without_delay(
        self,
        actor_factory_component: ActorFactoryComponent,
    ) -> None:
        """Verify cascade event without delay preserves timestamp.

        Input: Source event without delay.
        Expected Behavior: Timestamp unchanged.
        """
        source_event: dict[str, Any] = {
            "domain": "features",
            "event_type": "feature_computed",
            "correlation_id": "xyz789",
            "instrument_id": "ETH.USD",
            "ts_event": 2000000000,
            "event_id": "evt_002",
        }

        result = actor_factory_component.emit_cascade(source_event, "strategy")

        assert result["ts_event"] == 2000000000
        assert result["domain"] == "strategy"


# =============================================================================
# Error Condition Tests
# =============================================================================


class TestErrorConditions:
    """Tests for error handling paths."""

    def test_create_integrated_actor_handles_frozen_config(
        self,
        actor_factory_component: ActorFactoryComponent,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify handling of immutable (frozen) configs.

        Input: Frozen dataclass config.
        Expected Behavior: Logs exception, continues without crashing.
        """

        # Frozen config class - cannot add attributes
        @dataclass(frozen=True)
        class FrozenConfig:
            name: str = "test"

        config = FrozenConfig()

        # Simple actor class
        class SimpleActor:
            def __init__(self, config: object) -> None:
                self.config = config

        # Should not raise, but should log exception
        import logging

        with caplog.at_level(logging.ERROR):
            actor = actor_factory_component.create_integrated_actor(SimpleActor, config)

        # Actor should still be created
        assert actor is not None
        assert isinstance(actor, SimpleActor)
        # Should have logged the exception
        assert any(
            "Failed to attach db_connection" in record.getMessage()
            for record in caplog.records
        )

    def test_shutdown_handles_flush_exceptions(
        self,
        mock_feature_store: MagicMock,
        mock_model_store: MagicMock,
        mock_strategy_store: MagicMock,
        mock_data_store: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify shutdown resilience when store flush raises.

        Input: Store flush() raises exception.
        Expected Behavior: Other stores still flushed, no exception propagates.
        """
        # Make feature_store.flush raise
        mock_feature_store.flush.side_effect = RuntimeError("Flush failed!")

        component = ActorFactoryComponent(
            db_connection=TEST_DB_CONNECTION,
            feature_store=mock_feature_store,
            model_store=mock_model_store,
            strategy_store=mock_strategy_store,
            data_store=mock_data_store,
        )

        import logging

        with caplog.at_level(logging.ERROR):
            # Should not raise
            component.shutdown()

        # All stores should have been attempted
        mock_feature_store.flush.assert_called_once()
        mock_model_store.flush.assert_called_once()
        mock_strategy_store.flush.assert_called_once()
        mock_data_store.flush.assert_called_once()

        # Should have logged the exception
        assert any(
            "Failed to flush feature_store" in record.getMessage()
            for record in caplog.records
        )

    def test_set_message_publisher_handles_assignment_error(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify graceful handling of publisher assignment failure.

        Input: data_store.publisher raises on assignment.
        Expected Behavior: Debug logged, no exception propagates.
        """

        class ReadOnlyStore:
            """Store with read-only publisher attribute."""

            @property
            def publisher(self) -> None:
                return None

            @publisher.setter
            def publisher(self, value: object) -> None:
                raise AttributeError("Cannot set publisher")

        component = ActorFactoryComponent(
            data_store=ReadOnlyStore(),
        )

        import logging

        # Must capture at DEBUG level for the specific logger
        with caplog.at_level(
            logging.DEBUG, logger="ml.core.common.actor_factory"
        ):
            # Should not raise
            component.set_message_publisher(MagicMock())

        # Should have logged the debug message (check all records including DEBUG level)
        assert any(
            "Failed to attach publisher" in record.getMessage()
            for record in caplog.records
            if record.name == "ml.core.common.actor_factory"
        )


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_set_message_publisher_noop_when_no_data_store(self) -> None:
        """Verify safe no-op when data_store is None.

        Input: data_store = None.
        Expected Behavior: No exception raised.
        """
        component = ActorFactoryComponent(
            db_connection=TEST_DB_CONNECTION,
            data_store=None,
        )

        # Should not raise
        component.set_message_publisher(MagicMock())

    def test_set_message_publisher_noop_when_no_publisher_attr(self) -> None:
        """Verify safe no-op when data_store has no publisher attribute.

        Input: data_store without publisher attribute.
        Expected Behavior: No exception raised.
        """

        class StoreWithoutPublisher:
            pass

        component = ActorFactoryComponent(
            data_store=StoreWithoutPublisher(),
        )

        # Should not raise
        component.set_message_publisher(MagicMock())

    def test_configure_stubs_return_none(
        self,
        actor_factory_component: ActorFactoryComponent,
    ) -> None:
        """Verify all no-op configuration stubs return None.

        Input: Stub method calls.
        Expected Behavior: All return None.
        """
        assert (
            actor_factory_component.configure_message_bus(
                backend="redis",
                topic_prefix="ml.",
                retention_hours=24,
                max_size_mb=100,
            )
            is None
        )

        assert (
            actor_factory_component.configure_event_emission(
                batching_enabled=True,
                batch_size=100,
                flush_interval_ms=1000,
                correlation_strategy="uuid",
            )
            is None
        )

        assert actor_factory_component.configure_event_system(key="value") is None

        assert actor_factory_component.configure_domain_bookkeeping({"some": "config"}) is None

        assert actor_factory_component.emit_cross_domain_event({"event": "data"}) is None

    def test_shutdown_with_none_stores(self) -> None:
        """Verify shutdown handles None stores gracefully.

        Input: All stores are None.
        Expected Behavior: No exception raised.
        """
        component = ActorFactoryComponent(
            db_connection=TEST_DB_CONNECTION,
            feature_store=None,
            model_store=None,
            strategy_store=None,
            data_store=None,
        )

        # Should not raise
        component.shutdown()

    def test_shutdown_with_stores_without_flush(self) -> None:
        """Verify shutdown handles stores without flush method.

        Input: Stores without flush attribute.
        Expected Behavior: No exception raised.
        """

        class StoreWithoutFlush:
            pass

        component = ActorFactoryComponent(
            db_connection=TEST_DB_CONNECTION,
            feature_store=StoreWithoutFlush(),
            model_store=StoreWithoutFlush(),
            strategy_store=StoreWithoutFlush(),
            data_store=StoreWithoutFlush(),
        )

        # Should not raise
        component.shutdown()

    def test_create_integrated_actor_with_existing_db_connection(
        self,
        actor_factory_component: ActorFactoryComponent,
    ) -> None:
        """Verify actor creation when config already has db_connection.

        Input: Config with existing db_connection.
        Expected Behavior: Actor created, no exception.
        """

        @dataclass
        class ConfigWithConnection:
            db_connection: str = "other_connection"
            name: str = "test"

        config = ConfigWithConnection()

        class SimpleActor:
            def __init__(self, config: object) -> None:
                self.config = config

        actor = actor_factory_component.create_integrated_actor(SimpleActor, config)

        # Actor should be created
        assert actor is not None
        # Original connection should be preserved (hasattr returns True)
        assert hasattr(actor.config, "db_connection")

    def test_emit_cascade_with_missing_fields(
        self,
        actor_factory_component: ActorFactoryComponent,
    ) -> None:
        """Verify cascade handles minimal source event.

        Input: Source event with only required fields.
        Expected Behavior: Returns event with defaults for missing fields.

        Note: The underlying emit_cascade function uses "cascade" as default for
        event_type only when the field is completely missing. However, our
        component extracts values with empty string defaults, which get passed
        through (empty string is not None/missing).
        """
        source_event: dict[str, Any] = {
            "correlation_id": "minimal123",
            "ts_event": 500000000,
        }

        result = actor_factory_component.emit_cascade(source_event, "target_domain")

        assert result["correlation_id"] == "minimal123"
        assert result["domain"] == "target_domain"
        assert result["ts_event"] == 500000000
        # The component extracts event_type with default "", which then goes through
        # to the underlying cascade function - empty string passes through
        assert result["event_type"] == ""
        assert result["instrument_id"] == ""
        assert result["source_event_id"] == "unknown"
        assert result["payload"] == {}

    def test_emit_cascade_uses_source_event_id_fallback(
        self,
        actor_factory_component: ActorFactoryComponent,
    ) -> None:
        """Verify cascade uses source_event_id when event_id missing.

        Input: Source event with source_event_id but no event_id.
        Expected Behavior: Uses source_event_id.
        """
        source_event: dict[str, Any] = {
            "correlation_id": "test123",
            "ts_event": 100,
            "source_event_id": "parent_evt_001",
        }

        result = actor_factory_component.emit_cascade(source_event, "domain")

        assert result["source_event_id"] == "parent_evt_001"

    def test_component_default_values(self) -> None:
        """Verify component initializes with correct defaults."""
        component = ActorFactoryComponent()

        assert component.db_connection is None
        assert component.feature_store is None
        assert component.model_store is None
        assert component.strategy_store is None
        assert component.data_store is None


# =============================================================================
# Parity Tests (verify behavior matches legacy MLIntegrationManager)
# =============================================================================


class TestParityWithLegacy:
    """Tests to verify behavior matches the legacy MLIntegrationManager."""

    def test_shutdown_logs_completion_message(
        self,
        actor_factory_component: ActorFactoryComponent,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify shutdown logs the same completion message as legacy.

        Expected: "ML integration manager shutdown complete" logged.
        """
        import logging

        with caplog.at_level(logging.INFO):
            actor_factory_component.shutdown()

        assert any(
            "ML integration manager shutdown complete" in record.getMessage()
            for record in caplog.records
        )

    def test_create_integrated_actor_logs_exception_on_frozen_config(
        self,
        actor_factory_component: ActorFactoryComponent,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify exception logging matches legacy behavior.

        Legacy uses: logging.exception("Failed to attach db_connection to config")
        """

        @dataclass(frozen=True)
        class FrozenConfig:
            name: str = "frozen"

        class Actor:
            def __init__(self, config: object) -> None:
                self.config = config

        import logging

        with caplog.at_level(logging.ERROR):
            actor_factory_component.create_integrated_actor(Actor, FrozenConfig())

        # Check the exact message pattern from legacy
        assert any(
            "Failed to attach db_connection to config" in record.getMessage()
            for record in caplog.records
        )
