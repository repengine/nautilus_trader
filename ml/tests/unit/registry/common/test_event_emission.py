#!/usr/bin/env python3
"""
Unit tests for EventEmissionComponent.

Tests cover event emission, storage, trimming, and persistence
for the DataRegistry event handling extracted from the legacy DataRegistry.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from ml.config.events import EventStatus, Source, Stage
from ml.registry.persistence import BackendType, PersistenceConfig


if TYPE_CHECKING:
    from ml.registry.common.event_emission import EventEmissionComponent


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def json_persistence_config(tmp_path: Path) -> PersistenceConfig:
    """Create JSON backend persistence config for testing."""
    return PersistenceConfig(
        backend=BackendType.JSON,
        json_path=tmp_path / "registry",
    )


# =============================================================================
# Basic Emission Tests
# =============================================================================


class TestEmitEvent:
    """Tests for emit_event method."""

    def test_emit_event_success(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify successful event emission."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.event_emission import EventEmissionComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = EventEmissionComponent(persistence=persistence)

        component.emit_event(
            dataset_id="test_dataset",
            instrument_id="EUR/USD",
            stage=Stage.CATALOG_WRITTEN,
            source=Source.HISTORICAL,
            run_id="run_1",
            ts_min=1000000000000000000,
            ts_max=2000000000000000000,
            count=100,
            status=EventStatus.SUCCESS,
        )

        assert len(persistence._events) >= 1
        event = persistence._events[-1]
        assert event["dataset_id"] == "test_dataset"
        assert event["instrument_id"] == "EUR/USD"
        assert event["count"] == 100
        assert "created_at" in event

    def test_emit_event_normalizes_enums(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify enum values converted to strings."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.event_emission import EventEmissionComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = EventEmissionComponent(persistence=persistence)

        component.emit_event(
            dataset_id="test_dataset",
            instrument_id="EUR/USD",
            stage=Stage.CATALOG_WRITTEN,
            source=Source.HISTORICAL,
            run_id="run_1",
            ts_min=1000000000000000000,
            ts_max=2000000000000000000,
            count=100,
            status=EventStatus.SUCCESS,
        )

        event = persistence._events[-1]
        assert event["stage"] == "CATALOG_WRITTEN"
        assert event["source"] == "historical"
        assert event["status"] == "success"

    def test_emit_event_with_metadata(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify metadata correctly stored."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.event_emission import EventEmissionComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = EventEmissionComponent(persistence=persistence)

        metadata = {"correlation_id": "corr_123", "custom_field": "value"}

        component.emit_event(
            dataset_id="test_dataset",
            instrument_id="EUR/USD",
            stage=Stage.CATALOG_WRITTEN,
            source=Source.HISTORICAL,
            run_id="run_1",
            ts_min=1000000000000000000,
            ts_max=2000000000000000000,
            count=100,
            status=EventStatus.SUCCESS,
            metadata=metadata,
        )

        event = persistence._events[-1]
        assert event["metadata"] == metadata

    def test_emit_event_with_error(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify error field stored correctly."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.event_emission import EventEmissionComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = EventEmissionComponent(persistence=persistence)

        error_msg = "Connection timeout"

        component.emit_event(
            dataset_id="test_dataset",
            instrument_id="EUR/USD",
            stage=Stage.CATALOG_WRITTEN,
            source=Source.HISTORICAL,
            run_id="run_1",
            ts_min=1000000000000000000,
            ts_max=2000000000000000000,
            count=0,
            status=EventStatus.FAILED,
            error=error_msg,
        )

        event = persistence._events[-1]
        assert event["error"] == error_msg


# =============================================================================
# Event Trimming Tests
# =============================================================================


class TestEventTrimming:
    """Tests for event list trimming."""

    def test_emit_event_trims_old_events_json(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify JSON backend trims events at 10000."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.event_emission import EventEmissionComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = EventEmissionComponent(persistence=persistence)

        # Pre-populate with 9999 events (bypass emit to avoid save overhead)
        for i in range(9999):
            persistence._events.append({
                "dataset_id": f"dataset_{i}",
                "instrument_id": "EUR/USD",
                "stage": Stage.CATALOG_WRITTEN.value,
                "source": Source.HISTORICAL.value,
                "run_id": f"run_{i}",
                "ts_min": i,
                "ts_max": i + 1,
                "count": 1,
                "status": EventStatus.SUCCESS.value,
            })

        # Emit 2 more events to trigger trimming (9999 + 2 = 10001 > 10000)
        component.emit_event(
            dataset_id="dataset_9999",
            instrument_id="EUR/USD",
            stage=Stage.CATALOG_WRITTEN,
            source=Source.HISTORICAL,
            run_id="run_9999",
            ts_min=9999,
            ts_max=10000,
            count=1,
            status=EventStatus.SUCCESS,
        )
        component.emit_event(
            dataset_id="dataset_10000",
            instrument_id="EUR/USD",
            stage=Stage.CATALOG_WRITTEN,
            source=Source.HISTORICAL,
            run_id="run_10000",
            ts_min=10000,
            ts_max=10001,
            count=1,
            status=EventStatus.SUCCESS,
        )

        assert len(persistence._events) == 10000
        # Oldest events should be removed (events 0 was trimmed)
        assert persistence._events[0]["dataset_id"] == "dataset_1"


# =============================================================================
# Persistence Tests
# =============================================================================


class TestEventPersistence:
    """Tests for event persistence behavior."""

    def test_emit_event_saves_immediately_json(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify JSON backend saves immediately."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.event_emission import EventEmissionComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = EventEmissionComponent(persistence=persistence)

        component.emit_event(
            dataset_id="test_dataset",
            instrument_id="EUR/USD",
            stage=Stage.CATALOG_WRITTEN,
            source=Source.HISTORICAL,
            run_id="run_1",
            ts_min=1000000000000000000,
            ts_max=2000000000000000000,
            count=100,
            status=EventStatus.SUCCESS,
        )

        # File should contain event
        registry_file = tmp_path / "registry" / "data_registry.json"
        import json
        data = json.loads(registry_file.read_text())
        assert len(data.get("events", [])) >= 1


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in event emission."""

    def test_emit_event_zero_counts_allowed(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify zero count events allowed."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.event_emission import EventEmissionComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = EventEmissionComponent(persistence=persistence)

        component.emit_event(
            dataset_id="test_dataset",
            instrument_id="EUR/USD",
            stage=Stage.CATALOG_WRITTEN,
            source=Source.HISTORICAL,
            run_id="run_1",
            ts_min=1000000000000000000,
            ts_max=2000000000000000000,
            count=0,
            status=EventStatus.SUCCESS,
        )

        event = persistence._events[-1]
        assert event["count"] == 0

    def test_emit_event_with_correlation_id(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify correlation_id in metadata."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.event_emission import EventEmissionComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = EventEmissionComponent(persistence=persistence)

        component.emit_event(
            dataset_id="test_dataset",
            instrument_id="EUR/USD",
            stage=Stage.CATALOG_WRITTEN,
            source=Source.HISTORICAL,
            run_id="run_1",
            ts_min=1000000000000000000,
            ts_max=2000000000000000000,
            count=100,
            status=EventStatus.SUCCESS,
            metadata={"correlation_id": "test_corr_123"},
        )

        event = persistence._events[-1]
        assert event["metadata"]["correlation_id"] == "test_corr_123"


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread-safe event emission."""

    def test_emit_event_thread_safe(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify thread-safe event emission."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.event_emission import EventEmissionComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = EventEmissionComponent(persistence=persistence)

        errors: list[Exception] = []
        events_emitted = []

        def emit_operation(thread_id: int) -> None:
            try:
                for i in range(10):
                    component.emit_event(
                        dataset_id=f"dataset_t{thread_id}_{i}",
                        instrument_id="EUR/USD",
                        stage=Stage.CATALOG_WRITTEN,
                        source=Source.HISTORICAL,
                        run_id=f"run_t{thread_id}_{i}",
                        ts_min=i,
                        ts_max=i + 1,
                        count=1,
                        status=EventStatus.SUCCESS,
                    )
                    events_emitted.append(f"t{thread_id}_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=emit_operation, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety violated: {errors}"
        assert len(persistence._events) == 50  # 5 threads * 10 events each


# =============================================================================
# PostgreSQL Backend Tests (Mocked)
# =============================================================================


class TestPostgresBackend:
    """Tests for PostgreSQL backend event emission."""

    def test_emit_event_postgres_session_failure_raises(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify RuntimeError when session unavailable."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.event_emission import EventEmissionComponent

        # Use JSON backend but test the error path for PostgreSQL code
        config = PersistenceConfig(
            backend=BackendType.JSON,
            json_path=tmp_path / "registry",
        )

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=config,
        )

        # Override backend type to test PostgreSQL code path
        object.__setattr__(persistence.persistence.config, "backend", BackendType.POSTGRES)

        with patch.object(persistence.persistence, "get_session", return_value=None):
            component = EventEmissionComponent(persistence=persistence)

            with pytest.raises(RuntimeError, match="Failed to get database session"):
                component.emit_event(
                    dataset_id="test_dataset",
                    instrument_id="EUR/USD",
                    stage=Stage.CATALOG_WRITTEN,
                    source=Source.HISTORICAL,
                    run_id="run_1",
                    ts_min=1000000000000000000,
                    ts_max=2000000000000000000,
                    count=100,
                    status=EventStatus.SUCCESS,
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
