#!/usr/bin/env python3

"""
Unit tests for FeatureHealthComponent (Phase 3.7.6).

Tests health check, feature clearing, flush operations, and connection
management utilities extracted from FeatureStore.

Coverage target: 95%

"""

from __future__ import annotations

from typing import Any, Self
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from ml.stores.common.feature_health import (
    FeatureHealthComponent,
    FeatureHealthConfig,
    FeatureHealthProtocol,
)


# =========================================================================
# Mock Classes and Helpers
# =========================================================================


class MockColumn:
    """Mock SQLAlchemy column for testing."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, other: object) -> MagicMock:  # type: ignore[override]
        """Return mock equality clause."""
        clause = MagicMock()
        clause._comparison_value = other
        clause._column_name = self.name
        return clause


class MockTable:
    """Mock SQLAlchemy Table for testing."""

    def __init__(self, name: str = "ml_feature_values") -> None:
        self.name = name
        self.c = MagicMock()
        self.c.instrument_id = MockColumn("instrument_id")
        self.c.feature_version = MockColumn("feature_version")
        self._delete_stmt: MagicMock | None = None

    def delete(self) -> MagicMock:
        """Return mock delete statement."""
        self._delete_stmt = MagicMock()
        self._delete_stmt.where = MagicMock(return_value=self._delete_stmt)
        return self._delete_stmt


class MockConnection:
    """Mock SQLAlchemy connection for testing."""

    def __init__(self, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.executed_statements: list[Any] = []

    def execute(self, statement: Any) -> MagicMock:
        """Execute mock statement."""
        if self.should_fail:
            msg = "Connection failed"
            raise ConnectionError(msg)
        self.executed_statements.append(statement)
        return MagicMock()

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Exit context manager."""


class MockEngine:
    """Mock SQLAlchemy engine for testing."""

    def __init__(
        self,
        should_fail_connect: bool = False,
        should_fail_execute: bool = False,
    ) -> None:
        self.should_fail_connect = should_fail_connect
        self.should_fail_execute = should_fail_execute
        self._connection: MockConnection | None = None

    def connect(self) -> MockConnection:
        """Return mock connection."""
        if self.should_fail_connect:
            msg = "Failed to connect to database"
            raise ConnectionError(msg)
        self._connection = MockConnection(should_fail=self.should_fail_execute)
        return self._connection

    def begin(self) -> MockConnection:
        """Return mock connection with transaction."""
        if self.should_fail_connect:
            msg = "Failed to connect to database"
            raise ConnectionError(msg)
        self._connection = MockConnection(should_fail=self.should_fail_execute)
        return self._connection


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def mock_engine() -> MockEngine:
    """Create a mock SQLAlchemy engine."""
    return MockEngine()


@pytest.fixture
def mock_table() -> MockTable:
    """Create a mock feature values table."""
    return MockTable()


@pytest.fixture
def feature_health_component(
    mock_engine: MockEngine,
    mock_table: MockTable,
) -> FeatureHealthComponent:
    """Create a FeatureHealthComponent for testing."""
    return FeatureHealthComponent(
        engine=mock_engine,  # type: ignore[arg-type]
        table=mock_table,  # type: ignore[arg-type]
    )


@pytest.fixture
def feature_health_component_with_config(
    mock_engine: MockEngine,
    mock_table: MockTable,
) -> FeatureHealthComponent:
    """Create a FeatureHealthComponent with custom config."""
    config = FeatureHealthConfig(
        health_check_timeout_seconds=10.0,
        emit_metrics=False,
    )
    return FeatureHealthComponent(
        engine=mock_engine,  # type: ignore[arg-type]
        table=mock_table,  # type: ignore[arg-type]
        config=config,
    )


# =========================================================================
# Protocol Conformance Tests
# =========================================================================


class TestProtocolConformance:
    """Test that FeatureHealthComponent conforms to FeatureHealthProtocol."""

    def test_implements_protocol(
        self,
        feature_health_component: FeatureHealthComponent,
    ) -> None:
        """Test that component implements the protocol."""
        assert isinstance(feature_health_component, FeatureHealthProtocol)

    def test_has_is_healthy_method(
        self,
        feature_health_component: FeatureHealthComponent,
    ) -> None:
        """Test that component has is_healthy method."""
        assert hasattr(feature_health_component, "is_healthy")
        assert callable(feature_health_component.is_healthy)

    def test_has_clear_features_method(
        self,
        feature_health_component: FeatureHealthComponent,
    ) -> None:
        """Test that component has clear_features method."""
        assert hasattr(feature_health_component, "clear_features")
        assert callable(feature_health_component.clear_features)

    def test_has_flush_method(
        self,
        feature_health_component: FeatureHealthComponent,
    ) -> None:
        """Test that component has flush method."""
        assert hasattr(feature_health_component, "flush")
        assert callable(feature_health_component.flush)


# =========================================================================
# Configuration Tests
# =========================================================================


class TestFeatureHealthConfig:
    """Tests for FeatureHealthConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = FeatureHealthConfig()
        assert config.health_check_timeout_seconds == 5.0
        assert config.emit_metrics is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = FeatureHealthConfig(
            health_check_timeout_seconds=15.0,
            emit_metrics=False,
        )
        assert config.health_check_timeout_seconds == 15.0
        assert config.emit_metrics is False

    def test_config_is_frozen(self) -> None:
        """Test that config is immutable."""
        config = FeatureHealthConfig()
        with pytest.raises(AttributeError):
            config.health_check_timeout_seconds = 10.0  # type: ignore[misc]

    def test_invalid_timeout_raises(self) -> None:
        """Test that non-positive timeout raises ValueError."""
        with pytest.raises(ValueError, match="health_check_timeout_seconds must be positive"):
            FeatureHealthConfig(health_check_timeout_seconds=0.0)

        with pytest.raises(ValueError, match="health_check_timeout_seconds must be positive"):
            FeatureHealthConfig(health_check_timeout_seconds=-1.0)


# =========================================================================
# is_healthy Tests
# =========================================================================


class TestIsHealthy:
    """Tests for is_healthy method."""

    def test_is_healthy_returns_true_when_db_accessible(
        self,
        feature_health_component: FeatureHealthComponent,
    ) -> None:
        """Test is_healthy returns True when database is accessible."""
        result = feature_health_component.is_healthy()
        assert result is True

    def test_is_healthy_returns_false_on_connection_error(
        self,
        mock_table: MockTable,
    ) -> None:
        """Test is_healthy returns False when connection fails."""
        failing_engine = MockEngine(should_fail_connect=True)
        component = FeatureHealthComponent(
            engine=failing_engine,  # type: ignore[arg-type]
            table=mock_table,  # type: ignore[arg-type]
        )

        result = component.is_healthy()
        assert result is False

    def test_is_healthy_returns_false_on_query_error(
        self,
        mock_table: MockTable,
    ) -> None:
        """Test is_healthy returns False when query fails."""
        failing_engine = MockEngine(should_fail_execute=True)
        component = FeatureHealthComponent(
            engine=failing_engine,  # type: ignore[arg-type]
            table=mock_table,  # type: ignore[arg-type]
        )

        result = component.is_healthy()
        assert result is False

    def test_is_healthy_logs_warning_on_failure(
        self,
        mock_table: MockTable,
    ) -> None:
        """Test is_healthy logs warning when health check fails."""
        failing_engine = MockEngine(should_fail_connect=True)
        component = FeatureHealthComponent(
            engine=failing_engine,  # type: ignore[arg-type]
            table=mock_table,  # type: ignore[arg-type]
        )

        with patch("ml.stores.common.feature_health.logger") as mock_logger:
            result = component.is_healthy()
            assert result is False
            mock_logger.warning.assert_called_once()
            assert "health check failed" in str(mock_logger.warning.call_args)

    def test_is_healthy_emits_metrics_on_success(
        self,
        mock_engine: MockEngine,
        mock_table: MockTable,
    ) -> None:
        """Test is_healthy emits metrics when health check succeeds."""
        config = FeatureHealthConfig(emit_metrics=True)
        component = FeatureHealthComponent(
            engine=mock_engine,  # type: ignore[arg-type]
            table=mock_table,  # type: ignore[arg-type]
            config=config,
        )

        with patch("ml.stores.common.feature_health.HAS_PROMETHEUS", True):
            with patch(
                "ml.stores.common.feature_health.health_check_counter"
            ) as mock_counter:
                result = component.is_healthy()
                assert result is True
                mock_counter.labels.assert_called_once_with(status="healthy")
                mock_counter.labels().inc.assert_called_once()

    def test_is_healthy_emits_metrics_on_failure(
        self,
        mock_table: MockTable,
    ) -> None:
        """Test is_healthy emits metrics when health check fails."""
        failing_engine = MockEngine(should_fail_connect=True)
        config = FeatureHealthConfig(emit_metrics=True)
        component = FeatureHealthComponent(
            engine=failing_engine,  # type: ignore[arg-type]
            table=mock_table,  # type: ignore[arg-type]
            config=config,
        )

        with patch("ml.stores.common.feature_health.HAS_PROMETHEUS", True):
            with patch(
                "ml.stores.common.feature_health.health_check_counter"
            ) as mock_counter:
                result = component.is_healthy()
                assert result is False
                mock_counter.labels.assert_called_once_with(status="unhealthy")
                mock_counter.labels().inc.assert_called_once()

    def test_is_healthy_does_not_emit_metrics_when_disabled(
        self,
        feature_health_component_with_config: FeatureHealthComponent,
    ) -> None:
        """Test is_healthy does not emit metrics when emit_metrics is False."""
        with patch("ml.stores.common.feature_health.HAS_PROMETHEUS", True):
            with patch(
                "ml.stores.common.feature_health.health_check_counter"
            ) as mock_counter:
                result = feature_health_component_with_config.is_healthy()
                assert result is True
                mock_counter.labels.assert_not_called()


# =========================================================================
# clear_features Tests
# =========================================================================


class TestClearFeatures:
    """Tests for clear_features method."""

    def test_clear_features_deletes_by_instrument(
        self,
        mock_engine: MockEngine,
        mock_table: MockTable,
    ) -> None:
        """Test clear_features deletes by instrument ID."""
        component = FeatureHealthComponent(
            engine=mock_engine,  # type: ignore[arg-type]
            table=mock_table,  # type: ignore[arg-type]
        )

        component.clear_features(instrument_id="SPY.DATABENTO")

        # Verify delete was called
        assert mock_table._delete_stmt is not None
        mock_table._delete_stmt.where.assert_called_once()

    def test_clear_features_deletes_by_version(
        self,
        mock_engine: MockEngine,
        mock_table: MockTable,
    ) -> None:
        """Test clear_features deletes by feature version."""
        component = FeatureHealthComponent(
            engine=mock_engine,  # type: ignore[arg-type]
            table=mock_table,  # type: ignore[arg-type]
        )

        component.clear_features(feature_version="v1.2.0")

        # Verify delete was called
        assert mock_table._delete_stmt is not None
        mock_table._delete_stmt.where.assert_called_once()

    def test_clear_features_deletes_all_when_no_filter(
        self,
        mock_engine: MockEngine,
        mock_table: MockTable,
    ) -> None:
        """Test clear_features deletes all when no filters provided."""
        component = FeatureHealthComponent(
            engine=mock_engine,  # type: ignore[arg-type]
            table=mock_table,  # type: ignore[arg-type]
        )

        component.clear_features()

        # Verify delete was called without where clause
        assert mock_table._delete_stmt is not None
        mock_table._delete_stmt.where.assert_not_called()

    def test_clear_features_deletes_by_instrument_and_version(
        self,
        mock_engine: MockEngine,
        mock_table: MockTable,
    ) -> None:
        """Test clear_features deletes by both instrument and version."""
        component = FeatureHealthComponent(
            engine=mock_engine,  # type: ignore[arg-type]
            table=mock_table,  # type: ignore[arg-type]
        )

        component.clear_features(
            instrument_id="SPY.DATABENTO",
            feature_version="v1.2.0",
        )

        # Verify delete was called with both where clauses
        assert mock_table._delete_stmt is not None
        assert mock_table._delete_stmt.where.call_count == 2

    def test_clear_features_emits_metrics_for_instrument_scope(
        self,
        mock_engine: MockEngine,
        mock_table: MockTable,
    ) -> None:
        """Test clear_features emits metrics with instrument scope."""
        config = FeatureHealthConfig(emit_metrics=True)
        component = FeatureHealthComponent(
            engine=mock_engine,  # type: ignore[arg-type]
            table=mock_table,  # type: ignore[arg-type]
            config=config,
        )

        with patch("ml.stores.common.feature_health.HAS_PROMETHEUS", True):
            with patch(
                "ml.stores.common.feature_health.clear_features_counter"
            ) as mock_counter:
                component.clear_features(instrument_id="SPY.DATABENTO")
                mock_counter.labels.assert_called_once_with(scope="instrument")
                mock_counter.labels().inc.assert_called_once()

    def test_clear_features_emits_metrics_for_version_scope(
        self,
        mock_engine: MockEngine,
        mock_table: MockTable,
    ) -> None:
        """Test clear_features emits metrics with version scope."""
        config = FeatureHealthConfig(emit_metrics=True)
        component = FeatureHealthComponent(
            engine=mock_engine,  # type: ignore[arg-type]
            table=mock_table,  # type: ignore[arg-type]
            config=config,
        )

        with patch("ml.stores.common.feature_health.HAS_PROMETHEUS", True):
            with patch(
                "ml.stores.common.feature_health.clear_features_counter"
            ) as mock_counter:
                component.clear_features(feature_version="v1.2.0")
                mock_counter.labels.assert_called_once_with(scope="version")
                mock_counter.labels().inc.assert_called_once()

    def test_clear_features_emits_metrics_for_all_scope(
        self,
        mock_engine: MockEngine,
        mock_table: MockTable,
    ) -> None:
        """Test clear_features emits metrics with all scope."""
        config = FeatureHealthConfig(emit_metrics=True)
        component = FeatureHealthComponent(
            engine=mock_engine,  # type: ignore[arg-type]
            table=mock_table,  # type: ignore[arg-type]
            config=config,
        )

        with patch("ml.stores.common.feature_health.HAS_PROMETHEUS", True):
            with patch(
                "ml.stores.common.feature_health.clear_features_counter"
            ) as mock_counter:
                component.clear_features()
                mock_counter.labels.assert_called_once_with(scope="all")
                mock_counter.labels().inc.assert_called_once()

    def test_clear_features_emits_metrics_for_combined_scope(
        self,
        mock_engine: MockEngine,
        mock_table: MockTable,
    ) -> None:
        """Test clear_features emits metrics with combined scope."""
        config = FeatureHealthConfig(emit_metrics=True)
        component = FeatureHealthComponent(
            engine=mock_engine,  # type: ignore[arg-type]
            table=mock_table,  # type: ignore[arg-type]
            config=config,
        )

        with patch("ml.stores.common.feature_health.HAS_PROMETHEUS", True):
            with patch(
                "ml.stores.common.feature_health.clear_features_counter"
            ) as mock_counter:
                component.clear_features(
                    instrument_id="SPY.DATABENTO",
                    feature_version="v1.2.0",
                )
                mock_counter.labels.assert_called_once_with(scope="instrument_and_version")
                mock_counter.labels().inc.assert_called_once()

    def test_clear_features_logs_debug_message(
        self,
        feature_health_component: FeatureHealthComponent,
    ) -> None:
        """Test clear_features logs debug message."""
        with patch("ml.stores.common.feature_health.logger") as mock_logger:
            feature_health_component.clear_features(instrument_id="SPY.DATABENTO")
            mock_logger.debug.assert_called_once()
            assert "Cleared features" in str(mock_logger.debug.call_args)


# =========================================================================
# flush Tests
# =========================================================================


class TestFlush:
    """Tests for flush method."""

    def test_flush_is_noop_for_sync_writes(
        self,
        feature_health_component: FeatureHealthComponent,
    ) -> None:
        """Test flush is a no-op for synchronous writes."""
        # Should not raise any exceptions
        feature_health_component.flush()

    def test_flush_logs_debug_message(
        self,
        feature_health_component: FeatureHealthComponent,
    ) -> None:
        """Test flush logs debug message."""
        with patch("ml.stores.common.feature_health.logger") as mock_logger:
            feature_health_component.flush()
            mock_logger.debug.assert_called_once()
            assert "no-op" in str(mock_logger.debug.call_args)

    def test_flush_can_be_called_multiple_times(
        self,
        feature_health_component: FeatureHealthComponent,
    ) -> None:
        """Test flush can be called multiple times safely."""
        feature_health_component.flush()
        feature_health_component.flush()
        feature_health_component.flush()
        # Should not raise any exceptions


# =========================================================================
# Property Tests
# =========================================================================


class TestProperties:
    """Tests for component properties."""

    def test_engine_property(
        self,
        mock_engine: MockEngine,
        mock_table: MockTable,
    ) -> None:
        """Test engine property returns the engine."""
        component = FeatureHealthComponent(
            engine=mock_engine,  # type: ignore[arg-type]
            table=mock_table,  # type: ignore[arg-type]
        )
        assert component.engine is mock_engine

    def test_table_property(
        self,
        mock_engine: MockEngine,
        mock_table: MockTable,
    ) -> None:
        """Test table property returns the table."""
        component = FeatureHealthComponent(
            engine=mock_engine,  # type: ignore[arg-type]
            table=mock_table,  # type: ignore[arg-type]
        )
        assert component.table is mock_table


# =========================================================================
# _get_connection Tests
# =========================================================================


class TestGetConnection:
    """Tests for _get_connection method."""

    def test_get_connection_returns_connection(
        self,
        feature_health_component: FeatureHealthComponent,
    ) -> None:
        """Test _get_connection returns a connection context manager."""
        connection = feature_health_component._get_connection()
        assert connection is not None
        assert isinstance(connection, MockConnection)


# =========================================================================
# Integration-like Tests
# =========================================================================


class TestComponentIntegration:
    """Tests for component integration scenarios."""

    def test_health_check_after_clear_features(
        self,
        feature_health_component: FeatureHealthComponent,
    ) -> None:
        """Test health check works after clearing features."""
        feature_health_component.clear_features(instrument_id="SPY.DATABENTO")
        result = feature_health_component.is_healthy()
        assert result is True

    def test_flush_after_clear_features(
        self,
        feature_health_component: FeatureHealthComponent,
    ) -> None:
        """Test flush works after clearing features."""
        feature_health_component.clear_features()
        feature_health_component.flush()
        # Should not raise

    def test_component_with_default_config(
        self,
        mock_engine: MockEngine,
        mock_table: MockTable,
    ) -> None:
        """Test component works with default config."""
        component = FeatureHealthComponent(
            engine=mock_engine,  # type: ignore[arg-type]
            table=mock_table,  # type: ignore[arg-type]
        )
        assert component._config is not None
        assert component._config.health_check_timeout_seconds == 5.0
        assert component._config.emit_metrics is True

    def test_component_with_custom_config(
        self,
        mock_engine: MockEngine,
        mock_table: MockTable,
    ) -> None:
        """Test component works with custom config."""
        config = FeatureHealthConfig(
            health_check_timeout_seconds=20.0,
            emit_metrics=False,
        )
        component = FeatureHealthComponent(
            engine=mock_engine,  # type: ignore[arg-type]
            table=mock_table,  # type: ignore[arg-type]
            config=config,
        )
        assert component._config is config
        assert component._config.health_check_timeout_seconds == 20.0
        assert component._config.emit_metrics is False
