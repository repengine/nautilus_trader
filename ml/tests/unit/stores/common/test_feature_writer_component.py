#!/usr/bin/env python3

"""
Unit tests for FeatureWriterComponent (Phase 3.7.1).

Tests feature writing operations including explicit args mode, batch mode,
backward compatibility, circuit breaker integration, and message bus publishing.

Coverage target: 95%

"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ml.stores.common.feature_writer import (
    FeatureWriterComponent,
    FeatureWriterConfig,
    FeatureWriterProtocol,
    MessagePublisherProtocol,
)


# Patch target for sqlalchemy insert used inside _execute_write
POSTGRES_INSERT_PATCH_TARGET = "sqlalchemy.dialects.postgresql.insert"


# =========================================================================
# Mock Classes
# =========================================================================


class MockCircuitBreaker:
    """Mock circuit breaker for testing."""

    def __init__(self, *, can_execute: bool = True) -> None:
        self._can_execute = can_execute
        self.success_count = 0
        self.failure_count = 0

    def can_execute(self) -> bool:
        return self._can_execute

    def record_success(self) -> None:
        self.success_count += 1

    def record_failure(self) -> None:
        self.failure_count += 1


class MockPublisher:
    """Mock message publisher for testing."""

    def __init__(self) -> None:
        self.published: list[tuple[str, dict[str, Any]]] = []

    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        self.published.append((topic, payload))


class MockFeatureData:
    """Mock FeatureData for testing."""

    def __init__(
        self,
        feature_set_id: str = "test_fs",
        instrument_id: str = "SPY.DATABENTO",
        values: dict[str, float] | None = None,
        ts_event: int = 1700000000000000000,
        ts_init: int | None = None,
    ) -> None:
        self.feature_set_id = feature_set_id
        self.instrument_id = instrument_id
        self.values = values or {"close_return": 0.01}
        self._ts_event = ts_event
        self._ts_init = ts_init if ts_init is not None else ts_event

    @property
    def feature_values(self) -> dict[str, float]:
        return dict(self.values)

    @property
    def ts_event(self) -> int:
        return self._ts_event

    @property
    def ts_init(self) -> int:
        return self._ts_init


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def mock_engine() -> MagicMock:
    """Create a mock SQLAlchemy engine."""
    engine = MagicMock()
    conn_mock = MagicMock()
    engine.begin.return_value.__enter__ = MagicMock(return_value=conn_mock)
    engine.begin.return_value.__exit__ = MagicMock(return_value=False)
    return engine


@pytest.fixture
def mock_table() -> MagicMock:
    """Create a mock SQLAlchemy table."""
    table = MagicMock()
    table.c = MagicMock()
    return table


@pytest.fixture
def mock_circuit_breaker() -> MockCircuitBreaker:
    """Create a mock circuit breaker that allows execution."""
    return MockCircuitBreaker(can_execute=True)


@pytest.fixture
def mock_circuit_breaker_open() -> MockCircuitBreaker:
    """Create a mock circuit breaker that blocks execution."""
    return MockCircuitBreaker(can_execute=False)


@pytest.fixture
def mock_publisher() -> MockPublisher:
    """Create a mock message publisher."""
    return MockPublisher()


@pytest.fixture
def feature_writer(mock_engine: MagicMock, mock_table: MagicMock) -> FeatureWriterComponent:
    """Create a FeatureWriterComponent for testing."""
    return FeatureWriterComponent(
        engine=mock_engine,
        table=mock_table,
        get_feature_set_id=lambda: "default_fs",
    )


@pytest.fixture
def feature_writer_with_publishing(
    mock_engine: MagicMock,
    mock_table: MagicMock,
    mock_publisher: MockPublisher,
) -> FeatureWriterComponent:
    """Create a FeatureWriterComponent with publishing enabled."""
    config = FeatureWriterConfig(
        enable_publishing=True,
        publish_mode="batch",
    )
    return FeatureWriterComponent(
        engine=mock_engine,
        table=mock_table,
        get_feature_set_id=lambda: "default_fs",
        publisher=mock_publisher,
        config=config,
    )


@pytest.fixture
def feature_writer_with_row_publishing(
    mock_engine: MagicMock,
    mock_table: MagicMock,
    mock_publisher: MockPublisher,
) -> FeatureWriterComponent:
    """Create a FeatureWriterComponent with per-row publishing enabled."""
    config = FeatureWriterConfig(
        enable_publishing=True,
        publish_mode="row",
    )
    return FeatureWriterComponent(
        engine=mock_engine,
        table=mock_table,
        get_feature_set_id=lambda: "default_fs",
        publisher=mock_publisher,
        config=config,
    )


# =========================================================================
# Protocol Compliance Tests
# =========================================================================


class TestFeatureWriterProtocol:
    """Test protocol compliance."""

    def test_component_satisfies_protocol(
        self,
        feature_writer: FeatureWriterComponent,
    ) -> None:
        """Verify FeatureWriterComponent satisfies FeatureWriterProtocol."""
        assert isinstance(feature_writer, FeatureWriterProtocol)

    def test_mock_publisher_satisfies_protocol(
        self,
        mock_publisher: MockPublisher,
    ) -> None:
        """Verify MockPublisher satisfies MessagePublisherProtocol."""
        assert isinstance(mock_publisher, MessagePublisherProtocol)


# =========================================================================
# Configuration Tests
# =========================================================================


class TestFeatureWriterConfig:
    """Test configuration validation."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = FeatureWriterConfig()
        assert config.enable_publishing is False
        assert config.publish_mode == "batch"
        assert config.topic_scheme == "domain_op"
        assert config.topic_prefix == "events.ml"

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = FeatureWriterConfig(
            enable_publishing=True,
            publish_mode="both",
            topic_scheme="stage_first",
            topic_prefix="custom.prefix",
        )
        assert config.enable_publishing is True
        assert config.publish_mode == "both"
        assert config.topic_scheme == "stage_first"
        assert config.topic_prefix == "custom.prefix"

    def test_invalid_publish_mode_raises(self) -> None:
        """Test that invalid publish_mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid publish_mode"):
            FeatureWriterConfig(publish_mode="invalid")  # type: ignore[arg-type]


# =========================================================================
# Happy Path Tests
# =========================================================================


class TestWriteFeaturesExplicitArgs:
    """Test write_features with explicit arguments."""

    def test_write_features_explicit_args_stores_correctly(
        self,
        feature_writer: FeatureWriterComponent,
        mock_engine: MagicMock,
    ) -> None:
        """Verify explicit argument mode stores features correctly."""
        with patch(POSTGRES_INSERT_PATCH_TARGET) as mock_insert:
            mock_stmt = MagicMock()
            mock_insert.return_value.values.return_value = mock_stmt
            mock_stmt.on_conflict_do_update.return_value = mock_stmt

            feature_writer.write_features(
                feature_set_id="test_fs_001",
                instrument_id="SPY.DATABENTO",
                features={"close_return": 0.01, "volume_ratio": 1.5},
                ts_event=1700000000000000000,
                ts_init=1700000000000000000,
            )

            # Verify insert was called with correct values
            mock_insert.return_value.values.assert_called_once()
            call_args = mock_insert.return_value.values.call_args
            row = call_args[0][0]
            assert row["feature_set_id"] == "test_fs_001"
            assert row["instrument_id"] == "SPY.DATABENTO"
            assert row["values"]["close_return"] == 0.01
            assert row["values"]["volume_ratio"] == 1.5
            assert row["is_live"] is False
            assert row["source"] == "computed"

    def test_write_features_uses_ts_event_for_ts_init_when_not_provided(
        self,
        feature_writer: FeatureWriterComponent,
    ) -> None:
        """Verify ts_event is used for ts_init when ts_init is not provided."""
        with patch(POSTGRES_INSERT_PATCH_TARGET) as mock_insert:
            mock_stmt = MagicMock()
            mock_insert.return_value.values.return_value = mock_stmt
            mock_stmt.on_conflict_do_update.return_value = mock_stmt

            ts_event = 1700000000000000000
            feature_writer.write_features(
                feature_set_id="test_fs",
                instrument_id="SPY.DATABENTO",
                features={"close_return": 0.01},
                ts_event=ts_event,
            )

            call_args = mock_insert.return_value.values.call_args
            row = call_args[0][0]
            assert row["ts_init"] == ts_event


class TestWriteFeaturesBatchMode:
    """Test write_features with batch mode."""

    def test_write_features_batch_mode_processes_all_items(
        self,
        feature_writer: FeatureWriterComponent,
    ) -> None:
        """Verify batch mode processes all items in the batch."""
        batch = [
            MockFeatureData(
                feature_set_id="fs_001",
                instrument_id="SPY.DATABENTO",
                values={"close_return": 0.01},
                ts_event=1700000000000000000,
            ),
            MockFeatureData(
                feature_set_id="fs_002",
                instrument_id="AAPL.DATABENTO",
                values={"close_return": 0.02},
                ts_event=1700000001000000000,
            ),
        ]

        execute_write_calls: list[dict[str, Any]] = []
        original_execute_write = feature_writer._execute_write

        def mock_execute_write(row: dict[str, Any]) -> None:
            execute_write_calls.append(row)

        feature_writer._execute_write = mock_execute_write  # type: ignore[method-assign]

        feature_writer.write_features(data=batch)

        assert len(execute_write_calls) == 2
        assert execute_write_calls[0]["feature_set_id"] == "fs_001"
        assert execute_write_calls[1]["feature_set_id"] == "fs_002"

    def test_write_features_backward_compat_feature_data_list(
        self,
        feature_writer: FeatureWriterComponent,
    ) -> None:
        """Verify backward compatibility with write_features([FeatureData])."""
        batch = [
            MockFeatureData(
                feature_set_id="fs_001",
                instrument_id="SPY.DATABENTO",
            ),
        ]

        execute_write_calls: list[dict[str, Any]] = []

        def mock_execute_write(row: dict[str, Any]) -> None:
            execute_write_calls.append(row)

        feature_writer._execute_write = mock_execute_write  # type: ignore[method-assign]

        # Call with list as first positional arg (backward compat)
        feature_writer.write_features(batch)  # type: ignore[arg-type]

        assert len(execute_write_calls) == 1
        assert execute_write_calls[0]["feature_set_id"] == "fs_001"


class TestExecuteWrite:
    """Test _execute_write method."""

    def test_execute_write_upserts_on_conflict(
        self,
        feature_writer: FeatureWriterComponent,
        mock_engine: MagicMock,
    ) -> None:
        """Verify upsert behavior with ON CONFLICT."""
        with patch(POSTGRES_INSERT_PATCH_TARGET) as mock_insert:
            mock_stmt = MagicMock()
            mock_insert.return_value.values.return_value = mock_stmt
            mock_stmt.on_conflict_do_update.return_value = mock_stmt

            row = {
                "feature_set_id": "fs_001",
                "instrument_id": "SPY.DATABENTO",
                "ts_event": 1700000000000000000,
                "ts_init": 1700000000000000000,
                "values": {"close_return": 0.01},
                "is_live": False,
                "source": "computed",
            }
            feature_writer._execute_write(row)

            # Verify on_conflict_do_update was called
            mock_stmt.on_conflict_do_update.assert_called_once()
            call_kwargs = mock_stmt.on_conflict_do_update.call_args[1]
            assert "index_elements" in call_kwargs
            assert call_kwargs["index_elements"] == [
                "feature_set_id",
                "instrument_id",
                "ts_event",
            ]

    def test_execute_write_normalizes_timestamps(
        self,
        feature_writer: FeatureWriterComponent,
    ) -> None:
        """Verify timestamp normalization is applied."""
        with patch(POSTGRES_INSERT_PATCH_TARGET) as mock_insert:
            mock_stmt = MagicMock()
            mock_insert.return_value.values.return_value = mock_stmt
            mock_stmt.on_conflict_do_update.return_value = mock_stmt

            # Use millisecond timestamps (should be normalized to ns)
            row = {
                "feature_set_id": "fs_001",
                "instrument_id": "SPY.DATABENTO",
                "ts_event": 1700000000000,  # milliseconds
                "ts_init": 1700000000000,  # milliseconds
                "values": {"close_return": 0.01},
                "is_live": False,
                "source": "computed",
            }
            feature_writer._execute_write(row)

            # Verify timestamps were normalized
            call_args = mock_insert.return_value.values.call_args
            written_row = call_args[0][0]
            # Should be normalized to nanoseconds
            assert written_row["ts_event"] >= 1700000000000000000

    def test_execute_write_respects_circuit_breaker(
        self,
        mock_engine: MagicMock,
        mock_table: MagicMock,
        mock_circuit_breaker_open: MockCircuitBreaker,
    ) -> None:
        """Verify write is skipped when circuit breaker is open."""
        writer = FeatureWriterComponent(
            engine=mock_engine,
            table=mock_table,
            get_feature_set_id=lambda: "default_fs",
            circuit_breaker=mock_circuit_breaker_open,
        )

        with patch(POSTGRES_INSERT_PATCH_TARGET) as mock_insert:
            row = {
                "feature_set_id": "fs_001",
                "instrument_id": "SPY.DATABENTO",
                "ts_event": 1700000000000000000,
                "ts_init": 1700000000000000000,
                "values": {"close_return": 0.01},
                "is_live": False,
                "source": "computed",
            }
            writer._execute_write(row)

            # Verify no database insert occurred
            mock_engine.begin.assert_not_called()

    def test_execute_write_records_circuit_breaker_success(
        self,
        mock_engine: MagicMock,
        mock_table: MagicMock,
        mock_circuit_breaker: MockCircuitBreaker,
    ) -> None:
        """Verify circuit breaker success is recorded on successful write."""
        writer = FeatureWriterComponent(
            engine=mock_engine,
            table=mock_table,
            get_feature_set_id=lambda: "default_fs",
            circuit_breaker=mock_circuit_breaker,
        )

        with patch(POSTGRES_INSERT_PATCH_TARGET) as mock_insert:
            mock_stmt = MagicMock()
            mock_insert.return_value.values.return_value = mock_stmt
            mock_stmt.on_conflict_do_update.return_value = mock_stmt

            row = {
                "feature_set_id": "fs_001",
                "instrument_id": "SPY.DATABENTO",
                "ts_event": 1700000000000000000,
                "ts_init": 1700000000000000000,
                "values": {"close_return": 0.01},
                "is_live": False,
                "source": "computed",
            }
            writer._execute_write(row)

            assert mock_circuit_breaker.success_count == 1
            assert mock_circuit_breaker.failure_count == 0

    def test_execute_write_records_circuit_breaker_failure(
        self,
        mock_engine: MagicMock,
        mock_table: MagicMock,
        mock_circuit_breaker: MockCircuitBreaker,
    ) -> None:
        """Verify circuit breaker failure is recorded on database error."""
        mock_engine.begin.return_value.__enter__.side_effect = Exception("DB error")

        writer = FeatureWriterComponent(
            engine=mock_engine,
            table=mock_table,
            get_feature_set_id=lambda: "default_fs",
            circuit_breaker=mock_circuit_breaker,
        )

        with patch(POSTGRES_INSERT_PATCH_TARGET) as mock_insert:
            mock_stmt = MagicMock()
            mock_insert.return_value.values.return_value = mock_stmt
            mock_stmt.on_conflict_do_update.return_value = mock_stmt

            row = {
                "feature_set_id": "fs_001",
                "instrument_id": "SPY.DATABENTO",
                "ts_event": 1700000000000000000,
                "ts_init": 1700000000000000000,
                "values": {"close_return": 0.01},
                "is_live": False,
                "source": "computed",
            }

            with pytest.raises(Exception, match="DB error"):
                writer._execute_write(row)

            assert mock_circuit_breaker.failure_count == 1
            assert mock_circuit_breaker.success_count == 0

    def test_execute_write_publishes_per_row_event(
        self,
        feature_writer_with_row_publishing: FeatureWriterComponent,
        mock_publisher: MockPublisher,
    ) -> None:
        """Verify per-row event is published when publish_mode is 'row'."""
        with patch(POSTGRES_INSERT_PATCH_TARGET) as mock_insert:
            mock_stmt = MagicMock()
            mock_insert.return_value.values.return_value = mock_stmt
            mock_stmt.on_conflict_do_update.return_value = mock_stmt

            row = {
                "feature_set_id": "fs_001",
                "instrument_id": "SPY.DATABENTO",
                "ts_event": 1700000000000000000,
                "ts_init": 1700000000000000000,
                "values": {"close_return": 0.01},
                "is_live": False,
                "source": "computed",
            }
            feature_writer_with_row_publishing._execute_write(row)

            # Verify event was published
            assert len(mock_publisher.published) == 1
            topic, payload = mock_publisher.published[0]
            assert "features" in topic
            assert payload["instrument_id"] == "SPY.DATABENTO"
            assert payload["count"] == 1
            assert payload["status"] == "success"


class TestStoreFeatures:
    """Test store_features alias method."""

    def test_store_features_alias_delegates_correctly(
        self,
        feature_writer: FeatureWriterComponent,
    ) -> None:
        """Verify store_features delegates to write_features."""
        execute_write_calls: list[dict[str, Any]] = []

        def mock_execute_write(row: dict[str, Any]) -> None:
            execute_write_calls.append(row)

        feature_writer._execute_write = mock_execute_write  # type: ignore[method-assign]

        feature_writer.store_features(
            instrument_id="SPY.DATABENTO",
            ts_event=1700000000000000000,
            features={"close_return": 0.01},
        )

        assert len(execute_write_calls) == 1
        # Should use default feature_set_id from get_feature_set_id callback
        assert execute_write_calls[0]["feature_set_id"] == "default_fs"
        assert execute_write_calls[0]["instrument_id"] == "SPY.DATABENTO"

    def test_store_features_delegates_with_full_signature(
        self,
        feature_writer: FeatureWriterComponent,
    ) -> None:
        """Verify store_features delegates when full signature is provided."""
        execute_write_calls: list[dict[str, Any]] = []

        def mock_execute_write(row: dict[str, Any]) -> None:
            execute_write_calls.append(row)

        feature_writer._execute_write = mock_execute_write  # type: ignore[method-assign]

        feature_writer.store_features(
            feature_set_id="explicit_fs",
            instrument_id="SPY.DATABENTO",
            ts_event=1700000000000000000,
            features={"close_return": 0.01},
        )

        assert len(execute_write_calls) == 1
        # Should use explicit feature_set_id
        assert execute_write_calls[0]["feature_set_id"] == "explicit_fs"


class TestWriteBatch:
    """Test write_batch method."""

    def test_write_batch_clears_buffer_after_success(
        self,
        feature_writer: FeatureWriterComponent,
    ) -> None:
        """Verify buffer is cleared after successful write_batch."""
        batch = [
            MockFeatureData(feature_set_id="fs_001"),
            MockFeatureData(feature_set_id="fs_002"),
        ]

        with patch(POSTGRES_INSERT_PATCH_TARGET) as mock_insert:
            mock_stmt = MagicMock()
            mock_insert.return_value.values.return_value = mock_stmt
            mock_stmt.on_conflict_do_update.return_value = mock_stmt

            # Before write_batch, buffer should be empty
            assert len(feature_writer._write_buffer) == 0

            feature_writer.write_batch(batch)

            # After write_batch, buffer should be cleared
            assert len(feature_writer._write_buffer) == 0

    def test_write_batch_publishes_summary_event(
        self,
        feature_writer_with_publishing: FeatureWriterComponent,
        mock_publisher: MockPublisher,
    ) -> None:
        """Verify batch summary event is published."""
        batch = [
            MockFeatureData(
                feature_set_id="fs_001",
                instrument_id="SPY.DATABENTO",
                ts_event=1700000000000000000,
            ),
            MockFeatureData(
                feature_set_id="fs_002",
                instrument_id="SPY.DATABENTO",
                ts_event=1700000001000000000,
            ),
        ]

        with patch(POSTGRES_INSERT_PATCH_TARGET) as mock_insert:
            mock_stmt = MagicMock()
            mock_insert.return_value.values.return_value = mock_stmt
            mock_stmt.on_conflict_do_update.return_value = mock_stmt

            feature_writer_with_publishing.write_batch(batch)

            # Verify batch summary event was published
            assert len(mock_publisher.published) == 1
            topic, payload = mock_publisher.published[0]
            assert "features" in topic
            assert payload["count"] == 2
            assert payload["ts_min"] == 1700000000000000000
            assert payload["ts_max"] == 1700000001000000000
            assert payload["status"] == "success"

    def test_write_batch_empty_list_no_op(
        self,
        feature_writer: FeatureWriterComponent,
    ) -> None:
        """Verify empty batch is a no-op."""
        with patch(POSTGRES_INSERT_PATCH_TARGET) as mock_insert:
            feature_writer.write_batch([])

            # No insert should have been called
            mock_insert.assert_not_called()


# =========================================================================
# Error Condition Tests
# =========================================================================


class TestWriteFeaturesErrors:
    """Test error handling in write_features."""

    def test_write_features_raises_on_missing_args(
        self,
        feature_writer: FeatureWriterComponent,
    ) -> None:
        """Verify TypeError raised when required args missing."""
        with pytest.raises(
            TypeError,
            match="requires explicit arguments or a FeatureData batch",
        ):
            feature_writer.write_features()

    def test_write_features_raises_on_partial_args(
        self,
        feature_writer: FeatureWriterComponent,
    ) -> None:
        """Verify TypeError raised when only some required args provided."""
        with pytest.raises(
            TypeError,
            match="requires explicit arguments or a FeatureData batch",
        ):
            feature_writer.write_features(
                feature_set_id="fs_001",
                instrument_id="SPY.DATABENTO",
                # Missing features and ts_event
            )

    def test_write_features_raises_on_unsupported_data_type(
        self,
        feature_writer: FeatureWriterComponent,
    ) -> None:
        """Verify TypeError raised when data type is unsupported."""
        with pytest.raises(
            TypeError,
            match="Unsupported data type for write_features",
        ):
            feature_writer.write_features(data="invalid_data_type")


# =========================================================================
# Edge Case Tests
# =========================================================================


class TestEdgeCases:
    """Test edge cases."""

    def test_write_features_single_feature_data_object(
        self,
        feature_writer: FeatureWriterComponent,
    ) -> None:
        """Verify single FeatureData object is processed as batch of 1."""
        feature_data = MockFeatureData(
            feature_set_id="fs_001",
            instrument_id="SPY.DATABENTO",
        )

        execute_write_calls: list[dict[str, Any]] = []

        def mock_execute_write(row: dict[str, Any]) -> None:
            execute_write_calls.append(row)

        feature_writer._execute_write = mock_execute_write  # type: ignore[method-assign]

        feature_writer.write_features(data=feature_data)

        assert len(execute_write_calls) == 1
        assert execute_write_calls[0]["feature_set_id"] == "fs_001"

    def test_write_features_empty_features_dict(
        self,
        feature_writer: FeatureWriterComponent,
    ) -> None:
        """Verify empty features dict is handled."""
        with patch(POSTGRES_INSERT_PATCH_TARGET) as mock_insert:
            mock_stmt = MagicMock()
            mock_insert.return_value.values.return_value = mock_stmt
            mock_stmt.on_conflict_do_update.return_value = mock_stmt

            feature_writer.write_features(
                feature_set_id="fs_001",
                instrument_id="SPY.DATABENTO",
                features={},
                ts_event=1700000000000000000,
            )

            call_args = mock_insert.return_value.values.call_args
            row = call_args[0][0]
            assert row["values"] == {}

    def test_publish_disabled_no_events(
        self,
        feature_writer: FeatureWriterComponent,
    ) -> None:
        """Verify no events published when publishing is disabled."""
        # feature_writer has publishing disabled by default
        publisher = MockPublisher()
        feature_writer.publisher = publisher

        with patch(POSTGRES_INSERT_PATCH_TARGET) as mock_insert:
            mock_stmt = MagicMock()
            mock_insert.return_value.values.return_value = mock_stmt
            mock_stmt.on_conflict_do_update.return_value = mock_stmt

            feature_writer.write_features(
                feature_set_id="fs_001",
                instrument_id="SPY.DATABENTO",
                features={"close_return": 0.01},
                ts_event=1700000000000000000,
            )

            # No events should be published
            assert len(publisher.published) == 0

    def test_publish_bus_false_suppresses_publishing(
        self,
        feature_writer_with_publishing: FeatureWriterComponent,
        mock_publisher: MockPublisher,
    ) -> None:
        """Verify publish_bus=False suppresses publishing."""
        with patch(POSTGRES_INSERT_PATCH_TARGET) as mock_insert:
            mock_stmt = MagicMock()
            mock_insert.return_value.values.return_value = mock_stmt
            mock_stmt.on_conflict_do_update.return_value = mock_stmt

            feature_writer_with_publishing.write_features(
                feature_set_id="fs_001",
                instrument_id="SPY.DATABENTO",
                features={"close_return": 0.01},
                ts_event=1700000000000000000,
                publish_bus=False,
            )

            # No events should be published
            assert len(mock_publisher.published) == 0

    def test_buffer_alias_exists(
        self,
        feature_writer: FeatureWriterComponent,
    ) -> None:
        """Verify _buffer alias exists for backward compatibility."""
        assert hasattr(feature_writer, "_buffer")
        assert feature_writer._buffer is feature_writer._write_buffer

    def test_feature_data_with_exception_when_accessing_values(
        self,
        feature_writer: FeatureWriterComponent,
    ) -> None:
        """Verify handling when accessing feature_values during batch processing fails."""

        class FeatureDataWithBrokenValues:
            """FeatureData-like object where values access fails during iteration."""

            feature_set_id = "fs_001"
            instrument_id = "SPY.DATABENTO"
            ts_event = 1700000000000000000
            ts_init = 1700000000000000000
            _call_count = 0

            @property
            def feature_values(self) -> dict[str, float]:
                # First call succeeds (for hasattr check), second call fails
                self._call_count += 1
                if self._call_count > 1:
                    raise RuntimeError("Cannot access values")
                return {"close_return": 0.01}

        execute_write_calls: list[dict[str, Any]] = []

        def mock_execute_write(row: dict[str, Any]) -> None:
            execute_write_calls.append(row)

        feature_writer._execute_write = mock_execute_write  # type: ignore[method-assign]

        # The object has feature_values attribute but iteration access will fail
        # This tests the try/except in _write_batch_internal
        fd = FeatureDataWithBrokenValues()
        feature_writer.write_features(data=fd)

        assert len(execute_write_calls) == 1
        # Should have written successfully with the first successful access
        assert execute_write_calls[0]["feature_set_id"] == "fs_001"


# =========================================================================
# Observability Tests
# =========================================================================


class TestObservability:
    """Test observability and audit logging."""

    def test_audit_logging_sampled(
        self,
        feature_writer: FeatureWriterComponent,
    ) -> None:
        """Verify audit logging is sampled based on ML_AUDIT env var."""
        with (
            patch(POSTGRES_INSERT_PATCH_TARGET) as mock_insert,
            patch.dict("os.environ", {"ML_AUDIT": "1"}),
            patch.object(feature_writer, "_audit_log_sampled") as mock_audit,
        ):
            mock_stmt = MagicMock()
            mock_insert.return_value.values.return_value = mock_stmt
            mock_stmt.on_conflict_do_update.return_value = mock_stmt

            row = {
                "feature_set_id": "fs_001",
                "instrument_id": "SPY.DATABENTO",
                "ts_event": 1700000000000000000,
                "ts_init": 1700000000000000000,
                "values": {"close_return": 0.01},
                "is_live": False,
                "source": "computed",
            }
            feature_writer._execute_write(row)

            # Audit log should be called
            mock_audit.assert_called_once_with(row)

    def test_observability_stage_boundary_recorded(
        self,
        feature_writer: FeatureWriterComponent,
    ) -> None:
        """Verify observability stage boundary is recorded."""
        with (
            patch(POSTGRES_INSERT_PATCH_TARGET) as mock_insert,
            patch.object(
                feature_writer, "_record_observability_stage_boundary"
            ) as mock_obs,
        ):
            mock_stmt = MagicMock()
            mock_insert.return_value.values.return_value = mock_stmt
            mock_stmt.on_conflict_do_update.return_value = mock_stmt

            row = {
                "feature_set_id": "fs_001",
                "instrument_id": "SPY.DATABENTO",
                "ts_event": 1700000000000000000,
                "ts_init": 1700000000000000000,
                "values": {"close_return": 0.01},
                "is_live": False,
                "source": "computed",
            }
            feature_writer._execute_write(row)

            # Observability should be recorded
            mock_obs.assert_called_once()
            call_kwargs = mock_obs.call_args[1]
            assert call_kwargs["stage"] == "feature_storage"
            assert call_kwargs["instrument_id"] == "SPY.DATABENTO"
            assert call_kwargs["row_count"] == 1
