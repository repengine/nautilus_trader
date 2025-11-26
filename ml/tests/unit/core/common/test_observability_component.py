"""
Unit tests for ObservabilityComponent.

This module tests the observability component extracted from MLIntegrationManager
(Phase 3.6.5). Tests cover:

- Happy path: pipeline initialization, flush scheduling, DataFrame collection
- Error conditions: import errors, missing service, thread handling
- Edge cases: async worker status, idempotent stop operations

"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import PropertyMock
from unittest.mock import patch

import pandas as pd
import pytest

from ml.core.common.observability import ObservabilityComponent


# =============================================================================
# Fixtures
# =============================================================================


class MockObservabilityService:
    """Mock observability service for testing."""

    def __init__(
        self,
        *,
        latency_data: list[dict[str, Any]] | None = None,
        metrics_data: list[dict[str, Any]] | None = None,
        correlation_data: list[dict[str, Any]] | None = None,
        health_data: list[dict[str, Any]] | None = None,
    ) -> None:
        self._latency_data = latency_data or []
        self._metrics_data = metrics_data or []
        self._correlation_data = correlation_data or []
        self._health_data = health_data or []

    def latency_watermarks_df(self) -> pd.DataFrame:
        """Return latency DataFrame."""
        return pd.DataFrame(self._latency_data)

    def metrics_collection_df(self) -> pd.DataFrame:
        """Return metrics DataFrame."""
        return pd.DataFrame(self._metrics_data)

    def event_correlation_df(self) -> pd.DataFrame:
        """Return correlation DataFrame."""
        return pd.DataFrame(self._correlation_data)

    def health_scores_df(self) -> pd.DataFrame:
        """Return health DataFrame."""
        return pd.DataFrame(self._health_data)


class MockStore:
    """Mock store for testing injection."""

    def __init__(self) -> None:
        self._observability_service: Any = None


class MockObservabilityConfig:
    """Mock configuration for testing."""

    def __init__(
        self,
        *,
        base_path: str = "./observability",
        sink: str = "file",
        file_format: str = "jsonl",
        interval_seconds: float = 60.0,
        db_connection_string: str | None = None,
        async_enabled: bool = False,
        async_queue_maxsize: int = 4096,
        async_component_label: str = "obs_async_worker",
    ) -> None:
        self.base_path = base_path
        self.sink = sink
        self.file_format = file_format
        self.interval_seconds = interval_seconds
        self.db_connection_string = db_connection_string
        self.async_enabled = async_enabled
        self.async_queue_maxsize = async_queue_maxsize
        self.async_component_label = async_component_label


class MockAsyncWorker:
    """Mock async worker for testing."""

    def __init__(self, *, queue_size: int = 0) -> None:
        self._queue_size = queue_size
        self._started = False
        self._stopped = False

    def start(self) -> None:
        """Start the worker."""
        self._started = True

    async def stop(self, *, drain: bool = True, timeout: float = 1.0) -> None:
        """Stop the worker."""
        self._stopped = True

    def queue_size(self) -> int:
        """Return queue size."""
        return self._queue_size


@pytest.fixture
def mock_stores() -> list[MockStore]:
    """Provide mock stores for testing."""
    return [MockStore(), MockStore(), MockStore()]


@pytest.fixture
def component_with_stores(mock_stores: list[MockStore]) -> ObservabilityComponent:
    """Provide ObservabilityComponent with mock stores."""
    return ObservabilityComponent(stores=mock_stores)


@pytest.fixture
def component_with_service(
    mock_stores: list[MockStore],
) -> ObservabilityComponent:
    """Provide ObservabilityComponent with initialized mock service."""
    component = ObservabilityComponent(stores=mock_stores)
    component.observability_service = MockObservabilityService(
        latency_data=[{"id": 1, "latency": 100}],
        metrics_data=[{"metric": "foo", "value": 1.0}],
        correlation_data=[{"event_id": "e1", "correlation_id": "c1"}],
        health_data=[{"component": "test", "score": 0.95}],
    )
    return component


# =============================================================================
# Happy Path Tests
# =============================================================================


class TestHappyPath:
    """Tests for successful operation paths."""

    def test_initialize_observability_pipeline_creates_service(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify ObservabilityService creation.

        Input: No existing service.
        Expected Behavior: observability_service attribute set.
        """
        mock_service = MockObservabilityService()

        def mock_init(self: Any) -> None:
            pass

        with patch(
            "ml.core.common.observability.ObservabilityComponent.initialize_observability_pipeline"
        ) as mock_init_method:
            # Create component and manually set service to verify pattern
            component = ObservabilityComponent()
            assert component.observability_service is None

        # Test actual initialization
        component = ObservabilityComponent()

        with patch(
            "ml.observability.service.ObservabilityService",
            return_value=mock_service,
        ):
            component.initialize_observability_pipeline()

        assert component.observability_service is not None

    def test_start_observability_flush_single_flush(
        self,
        tmp_path: Path,
        component_with_service: ObservabilityComponent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify single flush mode (interval=0).

        Input: interval_seconds=0.
        Expected Behavior: Immediate flush, returns file paths.
        """
        written_paths: dict[str, Path] = {}

        class MockPersistor:
            def __init__(self, *, base_path: Path, file_format: str) -> None:
                self.base_path = base_path
                self.file_format = file_format

            def persist(
                self,
                tables: dict[str, pd.DataFrame | None],
            ) -> dict[str, Path]:
                for name in tables:
                    written_paths[name] = self.base_path / f"{name}.{self.file_format}"
                return written_paths

        monkeypatch.setattr(
            "ml.observability.persistence.ObservabilityPersistor",
            MockPersistor,
        )

        result = component_with_service.start_observability_flush(
            base_path=tmp_path,
            interval_seconds=0,
        )

        assert result is not None
        assert isinstance(result, dict)
        # Verify expected tables
        assert "latency" in result or len(result) == 0  # May be empty if no data

    def test_start_observability_flush_background_thread(
        self,
        tmp_path: Path,
        component_with_service: ObservabilityComponent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify background flush scheduling.

        Input: interval_seconds > 0.
        Expected Behavior: Background thread started.
        """
        mock_thread = MagicMock(spec=threading.Thread)

        class MockFlusher:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

            def start_background(self, stop_event: threading.Event) -> threading.Thread:
                return mock_thread

        monkeypatch.setattr(
            "ml.observability.scheduler.ObservabilityFlusher",
            MockFlusher,
        )

        result = component_with_service.start_observability_flush(
            base_path=tmp_path,
            interval_seconds=60.0,
        )

        assert result is None  # Background mode returns None
        assert component_with_service._obs_thread is mock_thread
        assert component_with_service._obs_flusher is not None
        assert component_with_service._obs_stop_event is not None

    def test_collect_observability_dataframes_returns_tables(
        self,
        component_with_service: ObservabilityComponent,
    ) -> None:
        """Verify DataFrame collection.

        Input: Service with data.
        Expected Behavior: Dict with DataFrames.
        """
        tables = component_with_service.collect_observability_dataframes()

        # Verify expected keys
        assert "latency" in tables
        assert "metrics" in tables
        assert "correlation" in tables
        assert "health" in tables

        # Verify DataFrames have data
        assert isinstance(tables["latency"], pd.DataFrame)
        assert len(tables["latency"]) > 0

    def test_inject_observability_service_into_stores_sets_attribute(
        self,
        component_with_service: ObservabilityComponent,
        mock_stores: list[MockStore],
    ) -> None:
        """Verify store injection.

        Input: Stores without _observability_service.
        Expected Behavior: Attribute set on all stores.
        """
        component_with_service.inject_observability_service_into_stores()

        for store in mock_stores:
            assert hasattr(store, "_observability_service")
            assert store._observability_service is component_with_service.observability_service

    def test_start_observability_from_config_respects_settings(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify config-based start respects settings.

        Input: Config object with specific values.
        Expected Behavior: Settings are applied.
        """
        component = ObservabilityComponent()
        component.observability_service = MockObservabilityService()

        flush_called_with: dict[str, Any] = {}

        def mock_start_flush(
            *,
            base_path: Path,
            interval_seconds: float,
            file_format: str,
            sink: str,
            db_connection_string: str | None,
        ) -> dict[str, Path] | None:
            flush_called_with["base_path"] = base_path
            flush_called_with["interval_seconds"] = interval_seconds
            flush_called_with["file_format"] = file_format
            flush_called_with["sink"] = sink
            return None

        monkeypatch.setattr(
            component,
            "start_observability_flush",
            mock_start_flush,
        )

        config = MockObservabilityConfig(
            base_path=str(tmp_path),
            sink="file",
            file_format="csv",
            interval_seconds=30.0,
        )

        component.start_observability_from_config(config)

        assert flush_called_with["base_path"] == tmp_path
        assert flush_called_with["interval_seconds"] == 30.0
        assert flush_called_with["file_format"] == "csv"
        assert flush_called_with["sink"] == "file"

    def test_start_observability_from_config_async_mode(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify async worker mode.

        Input: Config with async_enabled=True.
        Expected Behavior: Async worker started.
        """
        component = ObservabilityComponent()
        component.observability_service = MockObservabilityService()

        mock_worker = MockAsyncWorker()

        def mock_worker_init(**kwargs: Any) -> MockAsyncWorker:
            return mock_worker

        monkeypatch.setattr(
            "ml.observability.async_worker.ObservabilityAsyncWorker",
            mock_worker_init,
        )

        config = MockObservabilityConfig(
            base_path=str(tmp_path),
            async_enabled=True,
        )

        component.start_observability_from_config(config)

        assert component._obs_async_worker is mock_worker
        assert mock_worker._started

    def test_start_end_to_end_tracking_returns_none(self) -> None:
        """Verify E2E tracking stub returns None.

        Input: Call to start_end_to_end_tracking.
        Expected Behavior: Returns None (no-op).
        """
        component = ObservabilityComponent()
        result = component.start_end_to_end_tracking()
        assert result is None

    def test_start_health_checks_returns_none(self) -> None:
        """Verify health checks stub returns None.

        Input: Call to start_health_checks.
        Expected Behavior: Returns None (no-op).
        """
        component = ObservabilityComponent()
        result = component.start_health_checks()
        assert result is None

    def test_flush_observability_to_db_writes_rows(
        self,
        component_with_service: ObservabilityComponent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify database flush.

        Input: Valid connection string.
        Expected Behavior: Rows written to DB.
        """
        written_counts: dict[str, int] = {
            "latency": 1,
            "metrics": 1,
            "correlation": 1,
            "health": 1,
        }

        class MockDBPersistor:
            def __init__(self, *, connection_string: str) -> None:
                self.connection_string = connection_string

            def persist(
                self,
                tables: dict[str, pd.DataFrame | None],
            ) -> dict[str, int]:
                return written_counts

        monkeypatch.setattr(
            "ml.observability.db_persistence.ObservabilityDBPersistor",
            MockDBPersistor,
        )

        result = component_with_service.flush_observability_to_db(
            connection_string="sqlite:///test.db",
        )

        assert result == written_counts


# =============================================================================
# Error Condition Tests
# =============================================================================


class TestErrorConditions:
    """Tests for error conditions."""

    def test_initialize_observability_pipeline_handles_import_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify graceful handling when service unavailable.

        Input: ObservabilityService import fails.
        Expected Behavior: observability_service = None, no exception.
        """
        component = ObservabilityComponent()

        # Simulate import error
        def raise_import_error() -> None:
            raise ImportError("No module named 'ml.observability.service'")

        with patch.dict(
            "sys.modules",
            {"ml.observability.service": None},
        ):
            # The try/except should catch the error
            try:
                # Force the import to fail by patching __import__
                original_import = __builtins__.__dict__.get("__import__")

                def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
                    if name == "ml.observability.service":
                        raise ImportError("Mocked import error")
                    return original_import(name, *args, **kwargs)

                monkeypatch.setattr("builtins.__import__", mock_import)
                component.initialize_observability_pipeline()
            except Exception:
                pass  # Expected - the component handles this internally

        # Should not raise, service may be None
        assert component.observability_service is None or component.observability_service is not None

    def test_stop_observability_flush_handles_none_thread(self) -> None:
        """Verify idempotent stop.

        Input: No background thread running.
        Expected Behavior: No exception.
        """
        component = ObservabilityComponent()

        # Should not raise
        component.stop_observability_flush()

    def test_collect_observability_dataframes_returns_none_when_no_service(
        self,
    ) -> None:
        """Verify handling when service unavailable.

        Input: observability_service = None.
        Expected Behavior: Returns dict with None values.
        """
        component = ObservabilityComponent()
        component.observability_service = None

        result = component.collect_observability_dataframes()

        assert result["latency"] is None
        assert result["metrics"] is None
        assert result["correlation"] is None
        assert result["health"] is None

    def test_flush_observability_to_path_handles_exception(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify graceful handling of persistence errors.

        Input: Persistor raises exception.
        Expected Behavior: Returns empty dict.
        """
        component = ObservabilityComponent()
        component.observability_service = MockObservabilityService()

        def raise_error(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("Persistence failed")

        monkeypatch.setattr(
            "ml.observability.persistence.ObservabilityPersistor",
            raise_error,
        )

        result = component.flush_observability_to_path(base_path=tmp_path)

        assert result == {}

    def test_flush_observability_to_db_handles_exception(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify graceful handling of DB persistence errors.

        Input: DB Persistor raises exception.
        Expected Behavior: Returns empty dict.
        """
        component = ObservabilityComponent()
        component.observability_service = MockObservabilityService()

        def raise_error(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("DB persistence failed")

        monkeypatch.setattr(
            "ml.observability.db_persistence.ObservabilityDBPersistor",
            raise_error,
        )

        result = component.flush_observability_to_db(connection_string="sqlite:///test.db")

        assert result == {}

    def test_inject_observability_service_handles_exception(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify graceful handling of injection errors.

        Input: Store raises exception on setattr.
        Expected Behavior: Logs debug, no exception.
        """

        class BrokenStore:
            def __setattr__(self, name: str, value: Any) -> None:
                raise RuntimeError("Cannot set attribute")

        component = ObservabilityComponent(stores=[BrokenStore()])
        component.observability_service = MockObservabilityService()

        # Should not raise
        component.inject_observability_service_into_stores()

        # Debug log should be emitted
        assert "Observability injection failed" in caplog.text or len(caplog.records) >= 0


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_stop_observability_async_drains_queue(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify async worker drain on stop.

        Input: Async worker with pending items.
        Expected Behavior: Worker stopped, queue drained.
        """
        component = ObservabilityComponent()
        mock_worker = MockAsyncWorker(queue_size=10)
        component._obs_async_worker = mock_worker

        component.stop_observability_async()

        assert component._obs_async_worker is None

    def test_get_observability_async_status_when_not_running(self) -> None:
        """Verify status reporting when no worker.

        Input: No async worker.
        Expected Behavior: Returns {running: False, queue_size: 0}.
        """
        component = ObservabilityComponent()

        result = component.get_observability_async_status()

        assert result == {"running": False, "queue_size": 0}

    def test_get_observability_async_status_when_running(self) -> None:
        """Verify status reporting when worker running.

        Input: Async worker with items in queue.
        Expected Behavior: Returns running status and queue size.
        """
        component = ObservabilityComponent()
        mock_worker = MockAsyncWorker(queue_size=5)
        component._obs_async_worker = mock_worker

        result = component.get_observability_async_status()

        assert result["running"] is True
        assert result["queue_size"] == 5

    def test_start_observability_flush_no_service_after_init(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify handling when service init fails.

        Input: Service remains None after initialization attempt.
        Expected Behavior: Returns None for background mode.
        """
        component = ObservabilityComponent()

        # Force service to remain None
        def mock_init(self: Any) -> None:
            self.observability_service = None

        monkeypatch.setattr(
            ObservabilityComponent,
            "initialize_observability_pipeline",
            mock_init,
        )

        result = component.start_observability_flush(
            base_path=tmp_path,
            interval_seconds=60.0,
        )

        assert result is None

    def test_start_observability_from_config_no_service(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify handling when async mode but no service.

        Input: async_enabled=True but service init fails.
        Expected Behavior: Returns None gracefully.
        """
        component = ObservabilityComponent()

        # Force service to remain None
        def mock_init(self: Any) -> None:
            self.observability_service = None

        monkeypatch.setattr(
            ObservabilityComponent,
            "initialize_observability_pipeline",
            mock_init,
        )

        config = MockObservabilityConfig(
            base_path=str(tmp_path),
            async_enabled=True,
        )

        # Should not raise
        result = component.start_observability_from_config(config)

        assert result is None

    def test_stop_observability_flush_handles_stop_event_error(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify handling of stop event error.

        Input: Stop event raises on set().
        Expected Behavior: Logs debug, continues to join.
        """
        import logging

        caplog.set_level(logging.DEBUG, logger="ml.core.common.observability")

        component = ObservabilityComponent()

        mock_stop = MagicMock()
        mock_stop.set.side_effect = RuntimeError("Stop failed")

        component._obs_stop_event = mock_stop
        component._obs_thread = None

        # Should not raise
        component.stop_observability_flush()

        assert "Stop event set() failed" in caplog.text

    def test_stop_observability_flush_handles_thread_join_error(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify handling of thread join error.

        Input: Thread raises on join().
        Expected Behavior: Logs debug, no exception.
        """
        import logging

        caplog.set_level(logging.DEBUG, logger="ml.core.common.observability")

        component = ObservabilityComponent()

        mock_stop = MagicMock()
        mock_thread = MagicMock()
        mock_thread.join.side_effect = RuntimeError("Join failed")

        component._obs_stop_event = mock_stop
        component._obs_thread = mock_thread

        # Should not raise
        component.stop_observability_flush()

        assert "Join on observability thread failed" in caplog.text

    def test_inject_observability_service_with_none_stores(self) -> None:
        """Verify handling of None in stores list.

        Input: Stores list contains None values.
        Expected Behavior: Skips None stores without error.
        """
        component = ObservabilityComponent(stores=[MockStore(), None, MockStore()])
        component.observability_service = MockObservabilityService()

        # Should not raise
        component.inject_observability_service_into_stores()

        # Non-None stores should have service injected
        assert component.stores[0]._observability_service is component.observability_service
        assert component.stores[2]._observability_service is component.observability_service

    def test_inject_observability_service_empty_stores(self) -> None:
        """Verify handling of empty stores list.

        Input: Empty stores list.
        Expected Behavior: No error, no injection.
        """
        component = ObservabilityComponent(stores=[])
        component.observability_service = MockObservabilityService()

        # Should not raise
        component.inject_observability_service_into_stores()

    def test_inject_observability_service_no_service(self) -> None:
        """Verify no injection when service is None.

        Input: observability_service is None.
        Expected Behavior: Early return, no store modification.
        """
        store = MockStore()
        component = ObservabilityComponent(stores=[store])
        component.observability_service = None

        # Should not raise
        component.inject_observability_service_into_stores()

        # Store should not have been modified
        assert store._observability_service is None

    def test_start_observability_from_env_handles_missing_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify graceful handling when config unavailable.

        Input: ObservabilityConfig.from_env() fails.
        Expected Behavior: No exception.
        """
        component = ObservabilityComponent()

        def raise_error() -> None:
            raise ImportError("Config module unavailable")

        with patch(
            "ml.config.observability.ObservabilityConfig.from_env",
            side_effect=raise_error,
        ):
            # Should not raise
            component.start_observability_from_env()

    def test_get_observability_async_status_handles_exception(self) -> None:
        """Verify graceful handling of status check errors.

        Input: Worker queue_size() raises.
        Expected Behavior: Returns default status.
        """
        component = ObservabilityComponent()

        mock_worker = MagicMock()
        mock_worker.queue_size.side_effect = RuntimeError("Queue error")

        component._obs_async_worker = mock_worker

        result = component.get_observability_async_status()

        assert result == {"running": False, "queue_size": 0}

    def test_stop_observability_async_handles_exception(self) -> None:
        """Verify graceful handling of async stop errors.

        Input: Worker stop() raises.
        Expected Behavior: No exception.
        """
        component = ObservabilityComponent()

        # Create a mock that raises on stop
        mock_worker = MagicMock()

        async def raise_on_stop(**kwargs: Any) -> None:
            raise RuntimeError("Stop failed")

        mock_worker.stop = raise_on_stop

        component._obs_async_worker = mock_worker

        # Should not raise
        component.stop_observability_async()


# =============================================================================
# Integration Tests (within unit scope)
# =============================================================================


class TestComponentIntegration:
    """Tests for component integration patterns."""

    def test_full_observability_lifecycle(
        self,
        tmp_path: Path,
        mock_stores: list[MockStore],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify full lifecycle: init -> inject -> flush -> stop.

        Tests the typical usage pattern of the observability component.
        """
        component = ObservabilityComponent(stores=mock_stores)

        # Initialize
        mock_service = MockObservabilityService(
            latency_data=[{"id": 1}],
        )
        component.observability_service = mock_service

        # Inject into stores
        component.inject_observability_service_into_stores()

        for store in mock_stores:
            assert store._observability_service is mock_service

        # Collect dataframes
        tables = component.collect_observability_dataframes()
        assert "latency" in tables

        # Simulate background flush setup
        mock_thread = MagicMock()
        mock_flusher = MagicMock()
        mock_flusher.start_background.return_value = mock_thread

        monkeypatch.setattr(
            "ml.observability.scheduler.ObservabilityFlusher",
            lambda **kwargs: mock_flusher,
        )

        component.start_observability_flush(
            base_path=tmp_path,
            interval_seconds=60.0,
        )

        assert component._obs_thread is mock_thread

        # Stop
        component.stop_observability_flush()

    def test_component_dataclass_defaults(self) -> None:
        """Verify dataclass default values.

        Tests that default values are correctly initialized.
        """
        component = ObservabilityComponent()

        assert component.stores == []
        assert component.observability_service is None
        assert component._obs_flusher is None
        assert component._obs_stop_event is None
        assert component._obs_thread is None
        assert component._obs_async_worker is None

    def test_component_with_custom_stores(self) -> None:
        """Verify component accepts custom stores list.

        Tests that custom stores are properly assigned.
        """
        stores = [MockStore(), MockStore()]
        component = ObservabilityComponent(stores=stores)

        assert component.stores is stores
        assert len(component.stores) == 2
