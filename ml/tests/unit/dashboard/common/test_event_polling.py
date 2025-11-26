"""Unit tests for EventPollingComponent.

Tests cover list_events filtering, background polling lifecycle, cache behavior,
and thread safety.

"""

from __future__ import annotations

import json
import time
from threading import Thread
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from ml.dashboard.common.event_polling import (
    EventPollingComponent,
    _EventCache,
)


class TestEventCache:
    """Test _EventCache TTL and thread-safe behavior."""

    def test_snapshot_empty_cache_returns_not_fresh(self) -> None:
        """Verify empty cache returns not fresh."""
        cache = _EventCache(ttl_seconds=10.0, max_entries=100)
        events, is_fresh = cache.snapshot()
        assert events == []
        assert not is_fresh

    def test_snapshot_fresh_cache_returns_fresh(self) -> None:
        """Verify recently updated cache is fresh."""
        cache = _EventCache(ttl_seconds=10.0, max_entries=100)
        test_events = [{"id": "1", "topic": "test"}]
        cache.update(test_events)
        events, is_fresh = cache.snapshot()
        assert events == test_events
        assert is_fresh

    def test_snapshot_expired_cache_returns_stale(self) -> None:
        """Verify expired cache returns not fresh."""
        # update() calls clock once (for expires_at), snapshot() calls clock once (for now)
        mock_clock = Mock(side_effect=[0.0, 11.0])
        cache = _EventCache(ttl_seconds=10.0, max_entries=100, _clock=mock_clock)
        test_events = [{"id": "1", "topic": "test"}]
        cache.update(test_events)  # Sets expires_at = 0.0 + 10.0 = 10.0
        events, is_fresh = cache.snapshot()  # Now is 11.0, expires_at is 10.0
        assert events == test_events
        assert not is_fresh  # Expired: 11.0 >= 10.0

    def test_update_trims_to_max_entries(self) -> None:
        """Verify update trims events to max_entries."""
        cache = _EventCache(ttl_seconds=10.0, max_entries=3)
        test_events = [
            {"id": "1", "topic": "test1"},
            {"id": "2", "topic": "test2"},
            {"id": "3", "topic": "test3"},
            {"id": "4", "topic": "test4"},
            {"id": "5", "topic": "test5"},
        ]
        cache.update(test_events)
        events, _ = cache.snapshot()
        assert len(events) == 3
        assert events == test_events[:3]

    def test_stale_snapshot_returns_events_regardless_of_ttl(self) -> None:
        """Verify stale_snapshot ignores TTL."""
        # Clock calls: update (2 calls for set), stale_snapshot (1 call - but it doesn't check time!)
        mock_clock = Mock(side_effect=[0.0, 20.0])  # update calls
        cache = _EventCache(ttl_seconds=10.0, max_entries=100, _clock=mock_clock)
        test_events = [{"id": "1", "topic": "test"}]
        cache.update(test_events)
        events = cache.stale_snapshot()  # stale_snapshot doesn't call clock
        assert events == test_events

    def test_cache_thread_safety(self) -> None:
        """Verify cache operations are thread-safe."""
        cache = _EventCache(ttl_seconds=10.0, max_entries=100)
        results: list[int] = []

        def reader() -> None:
            for _ in range(100):
                events, _ = cache.snapshot()
                results.append(len(events))

        def writer() -> None:
            for i in range(100):
                cache.update([{"id": str(i), "topic": f"test{i}"}])

        threads = [
            Thread(target=reader),
            Thread(target=reader),
            Thread(target=writer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without deadlock or exceptions
        assert len(results) == 200  # 2 readers x 100 iterations


class TestEventPollingComponentInit:
    """Test EventPollingComponent initialization."""

    def test_init_creates_cache_with_correct_params(self) -> None:
        """Verify initialization creates cache with correct parameters."""
        component = EventPollingComponent(ttl_seconds=30.0, max_entries=500)
        assert component._event_cache.ttl_seconds == 30.0
        assert component._event_cache.max_entries == 500
        assert component._event_poll_thread is None
        assert component._event_poll_stop is None


class TestEventPollingComponentListEvents:
    """Test list_events method with various filters."""

    @patch("ml.dashboard.common.event_polling.MessageBusConfig.from_env")
    def test_list_events_when_bus_disabled_returns_empty(
        self,
        mock_config: Mock,
    ) -> None:
        """Verify list_events returns empty when message bus disabled."""
        mock_config.return_value = Mock(enabled=False)
        component = EventPollingComponent(ttl_seconds=30.0, max_entries=500)
        events = component.list_events()
        assert events == []

    @patch("redis.Redis")
    @patch("ml.dashboard.common.event_polling.MessageBusConfig.from_env")
    def test_list_events_polls_and_caches_on_cache_miss(
        self,
        mock_config: Mock,
        mock_redis_class: Mock,
    ) -> None:
        """Verify list_events polls and caches on cache miss."""
        mock_config.return_value = Mock(
            enabled=True,
            backend="redis",
            redis_url="redis://localhost:6379",
            redis_stream="ml_events",
        )
        mock_client = MagicMock()
        mock_redis_class.from_url.return_value = mock_client
        mock_client.xrevrange.return_value = [
            ("1-0", {"topic": "ml.training", "payload": '{"source": "orchestrator"}'}),
            ("2-0", {"topic": "ml.ingestion", "payload": '{"source": "actor"}'}),
        ]

        component = EventPollingComponent(ttl_seconds=30.0, max_entries=500)
        events = component.list_events(limit=10)

        assert len(events) == 2
        assert events[0]["id"] == "1-0"
        assert events[0]["topic"] == "ml.training"
        assert events[0]["payload"]["source"] == "orchestrator"

    @patch("redis.Redis")
    @patch("ml.dashboard.common.event_polling.MessageBusConfig.from_env")
    def test_list_events_uses_cache_when_fresh(
        self,
        mock_config: Mock,
        mock_redis_class: Mock,
    ) -> None:
        """Verify list_events uses cache when fresh."""
        mock_config.return_value = Mock(
            enabled=True,
            backend="redis",
            redis_url="redis://localhost:6379",
            redis_stream="ml_events",
        )
        mock_client = MagicMock()
        mock_redis_class.from_url.return_value = mock_client
        mock_client.xrevrange.return_value = [
            ("1-0", {"topic": "ml.training", "payload": '{"source": "orchestrator"}'}),
        ]

        component = EventPollingComponent(ttl_seconds=30.0, max_entries=500)
        # First call populates cache
        events1 = component.list_events(limit=10)
        # Second call uses cache
        events2 = component.list_events(limit=10)

        # Redis should only be called once
        assert mock_client.xrevrange.call_count == 1
        assert events1 == events2

    @patch("redis.Redis")
    @patch("ml.dashboard.common.event_polling.MessageBusConfig.from_env")
    def test_list_events_filter_by_stage(
        self,
        mock_config: Mock,
        mock_redis_class: Mock,
    ) -> None:
        """Verify list_events filters by stage in topic."""
        mock_config.return_value = Mock(
            enabled=True,
            backend="redis",
            redis_url="redis://localhost:6379",
            redis_stream="ml_events",
        )
        mock_client = MagicMock()
        mock_redis_class.from_url.return_value = mock_client
        mock_client.xrevrange.return_value = [
            ("1-0", {"topic": "ml.training.started", "payload": "{}"}),
            ("2-0", {"topic": "ml.ingestion.completed", "payload": "{}"}),
            ("3-0", {"topic": "ml.training.completed", "payload": "{}"}),
        ]

        component = EventPollingComponent(ttl_seconds=30.0, max_entries=500)
        events = component.list_events(stage="training")

        assert len(events) == 2
        assert all("training" in e["topic"] for e in events)

    @patch("redis.Redis")
    @patch("ml.dashboard.common.event_polling.MessageBusConfig.from_env")
    def test_list_events_filter_by_source(
        self,
        mock_config: Mock,
        mock_redis_class: Mock,
    ) -> None:
        """Verify list_events filters by source in payload."""
        mock_config.return_value = Mock(
            enabled=True,
            backend="redis",
            redis_url="redis://localhost:6379",
            redis_stream="ml_events",
        )
        mock_client = MagicMock()
        mock_redis_class.from_url.return_value = mock_client
        mock_client.xrevrange.return_value = [
            ("1-0", {"topic": "ml.training", "payload": '{"source": "orchestrator"}'}),
            ("2-0", {"topic": "ml.training", "payload": '{"source": "actor"}'}),
            ("3-0", {"topic": "ml.training", "payload": '{"source": "orchestrator"}'}),
        ]

        component = EventPollingComponent(ttl_seconds=30.0, max_entries=500)
        events = component.list_events(source="orchestrator")

        assert len(events) == 2
        assert all(e["payload"]["source"] == "orchestrator" for e in events)

    @patch("redis.Redis")
    @patch("ml.dashboard.common.event_polling.MessageBusConfig.from_env")
    def test_list_events_filter_by_instrument_in_payload(
        self,
        mock_config: Mock,
        mock_redis_class: Mock,
    ) -> None:
        """Verify list_events filters by instrument in payload params."""
        mock_config.return_value = Mock(
            enabled=True,
            backend="redis",
            redis_url="redis://localhost:6379",
            redis_stream="ml_events",
        )
        mock_client = MagicMock()
        mock_redis_class.from_url.return_value = mock_client
        mock_client.xrevrange.return_value = [
            (
                "1-0",
                {
                    "topic": "ml.training",
                    "payload": '{"params": {"instrument": "EURUSD.SIM"}}',
                },
            ),
            (
                "2-0",
                {
                    "topic": "ml.training",
                    "payload": '{"params": {"instrument": "GBPUSD.SIM"}}',
                },
            ),
            (
                "3-0",
                {
                    "topic": "ml.training",
                    "payload": '{"params": {"instrument": "EURUSD.SIM"}}',
                },
            ),
        ]

        component = EventPollingComponent(ttl_seconds=30.0, max_entries=500)
        events = component.list_events(instrument_substr="EURUSD")

        assert len(events) == 2
        assert all("EURUSD" in e["payload"]["params"]["instrument"] for e in events)

    @patch("redis.Redis")
    @patch("ml.dashboard.common.event_polling.MessageBusConfig.from_env")
    def test_list_events_filter_by_instrument_in_topic(
        self,
        mock_config: Mock,
        mock_redis_class: Mock,
    ) -> None:
        """Verify list_events filters by instrument in topic when not in payload."""
        mock_config.return_value = Mock(
            enabled=True,
            backend="redis",
            redis_url="redis://localhost:6379",
            redis_stream="ml_events",
        )
        mock_client = MagicMock()
        mock_redis_class.from_url.return_value = mock_client
        mock_client.xrevrange.return_value = [
            ("1-0", {"topic": "ml.training.EURUSD", "payload": "{}"}),
            ("2-0", {"topic": "ml.training.GBPUSD", "payload": "{}"}),
            ("3-0", {"topic": "ml.training.EURUSD", "payload": "{}"}),
        ]

        component = EventPollingComponent(ttl_seconds=30.0, max_entries=500)
        events = component.list_events(instrument_substr="EURUSD")

        assert len(events) == 2
        assert all("EURUSD" in e["topic"] for e in events)

    @patch("redis.Redis")
    @patch("ml.dashboard.common.event_polling.MessageBusConfig.from_env")
    def test_list_events_respects_limit(
        self,
        mock_config: Mock,
        mock_redis_class: Mock,
    ) -> None:
        """Verify list_events respects limit parameter."""
        mock_config.return_value = Mock(
            enabled=True,
            backend="redis",
            redis_url="redis://localhost:6379",
            redis_stream="ml_events",
        )
        mock_client = MagicMock()
        mock_redis_class.from_url.return_value = mock_client
        mock_client.xrevrange.return_value = [
            (f"{i}-0", {"topic": f"ml.event{i}", "payload": "{}"}) for i in range(10)
        ]

        component = EventPollingComponent(ttl_seconds=30.0, max_entries=500)
        events = component.list_events(limit=5)

        assert len(events) == 5

    @patch("redis.Redis")
    @patch("ml.dashboard.common.event_polling.MessageBusConfig.from_env")
    def test_list_events_handles_invalid_json_payload(
        self,
        mock_config: Mock,
        mock_redis_class: Mock,
    ) -> None:
        """Verify list_events handles invalid JSON in payload."""
        mock_config.return_value = Mock(
            enabled=True,
            backend="redis",
            redis_url="redis://localhost:6379",
            redis_stream="ml_events",
        )
        mock_client = MagicMock()
        mock_redis_class.from_url.return_value = mock_client
        mock_client.xrevrange.return_value = [
            ("1-0", {"topic": "ml.training", "payload": "invalid json"}),
        ]

        component = EventPollingComponent(ttl_seconds=30.0, max_entries=500)
        events = component.list_events()

        assert len(events) == 1
        assert events[0]["payload"] == {"raw": "invalid json"}

    @patch("redis.Redis")
    @patch("ml.dashboard.common.event_polling.MessageBusConfig.from_env")
    def test_list_events_on_redis_error_returns_cached(
        self,
        mock_config: Mock,
        mock_redis_class: Mock,
    ) -> None:
        """Verify list_events returns cached events on Redis error."""
        mock_config.return_value = Mock(
            enabled=True,
            backend="redis",
            redis_url="redis://localhost:6379",
            redis_stream="ml_events",
        )
        mock_client = MagicMock()
        mock_redis_class.from_url.return_value = mock_client

        # First call succeeds
        mock_client.xrevrange.return_value = [
            ("1-0", {"topic": "ml.training", "payload": '{"source": "orchestrator"}'}),
        ]
        component = EventPollingComponent(ttl_seconds=30.0, max_entries=500)
        events1 = component.list_events()
        assert len(events1) == 1

        # Second call after cache expires should fail but return stale cache
        component._event_cache._expires_at = 0.0  # Force cache expiry
        mock_client.xrevrange.side_effect = RuntimeError("Redis down")
        events2 = component.list_events()
        assert len(events2) == 1  # Returns stale cached events


class TestEventPollingComponentBackgroundPolling:
    """Test start_event_polling and stop_event_polling lifecycle."""

    @patch("redis.Redis")
    @patch("ml.dashboard.common.event_polling.MessageBusConfig.from_env")
    def test_start_event_polling_creates_thread(
        self,
        mock_config: Mock,
        mock_redis_class: Mock,
    ) -> None:
        """Verify start_event_polling creates background thread."""
        mock_config.return_value = Mock(
            enabled=True,
            backend="redis",
            redis_url="redis://localhost:6379",
            redis_stream="ml_events",
        )
        mock_client = MagicMock()
        mock_redis_class.from_url.return_value = mock_client
        mock_client.xrevrange.return_value = []

        component = EventPollingComponent(ttl_seconds=30.0, max_entries=500)
        component.start_event_polling(interval_seconds=0.1)

        assert component._event_poll_thread is not None
        assert component._event_poll_thread.is_alive()
        assert component._event_poll_stop is not None

        component.stop_event_polling()

    def test_start_event_polling_with_zero_interval_is_noop(self) -> None:
        """Verify start_event_polling with zero interval is no-op."""
        component = EventPollingComponent(ttl_seconds=30.0, max_entries=500)
        component.start_event_polling(interval_seconds=0.0)
        assert component._event_poll_thread is None

    def test_start_event_polling_when_already_running_is_noop(self) -> None:
        """Verify start_event_polling when already running is no-op."""
        component = EventPollingComponent(ttl_seconds=30.0, max_entries=500)
        component._event_poll_thread = Mock(is_alive=Mock(return_value=True))
        component.start_event_polling(interval_seconds=1.0)
        # Should not replace existing thread
        assert component._event_poll_thread is not None

    @patch("redis.Redis")
    @patch("ml.dashboard.common.event_polling.MessageBusConfig.from_env")
    def test_background_polling_updates_cache(
        self,
        mock_config: Mock,
        mock_redis_class: Mock,
    ) -> None:
        """Verify background polling updates cache periodically."""
        mock_config.return_value = Mock(
            enabled=True,
            backend="redis",
            redis_url="redis://localhost:6379",
            redis_stream="ml_events",
        )
        mock_client = MagicMock()
        mock_redis_class.from_url.return_value = mock_client
        mock_client.xrevrange.return_value = [
            ("1-0", {"topic": "ml.training", "payload": '{"source": "orchestrator"}'}),
        ]

        component = EventPollingComponent(ttl_seconds=30.0, max_entries=500)
        component.start_event_polling(interval_seconds=0.1)

        # Wait for at least one poll
        time.sleep(0.3)

        events, is_fresh = component._event_cache.snapshot()
        assert len(events) > 0
        assert is_fresh

        component.stop_event_polling()

    def test_stop_event_polling_stops_thread(self) -> None:
        """Verify stop_event_polling stops background thread."""
        component = EventPollingComponent(ttl_seconds=30.0, max_entries=500)
        mock_thread = Mock()
        mock_thread.is_alive.return_value = True
        mock_stop = Mock()
        component._event_poll_thread = mock_thread
        component._event_poll_stop = mock_stop

        component.stop_event_polling()

        mock_stop.set.assert_called_once()
        mock_thread.join.assert_called_once_with(timeout=1.0)
        assert component._event_poll_thread is None
        assert component._event_poll_stop is None

    def test_stop_event_polling_when_not_running_is_noop(self) -> None:
        """Verify stop_event_polling when not running is no-op."""
        component = EventPollingComponent(ttl_seconds=30.0, max_entries=500)
        component.stop_event_polling()  # Should not raise
        assert component._event_poll_thread is None

    @patch("ml.dashboard.common.event_polling.MessageBusConfig.from_env")
    def test_background_polling_handles_disabled_bus(
        self,
        mock_config: Mock,
    ) -> None:
        """Verify background polling handles disabled bus gracefully."""
        mock_config.return_value = Mock(enabled=False)

        component = EventPollingComponent(ttl_seconds=30.0, max_entries=500)
        component.start_event_polling(interval_seconds=0.1)

        # Wait for poll attempts
        time.sleep(0.3)

        # Should not crash, cache remains empty
        events, _ = component._event_cache.snapshot()
        assert events == []

        component.stop_event_polling()


class TestEventPollingProtocol:
    """Test EventPollingProtocol compliance."""

    def test_component_implements_protocol(self) -> None:
        """Verify EventPollingComponent implements EventPollingProtocol."""
        from ml.dashboard.common.event_polling import EventPollingProtocol

        component = EventPollingComponent(ttl_seconds=30.0, max_entries=500)
        # Should satisfy protocol
        assert isinstance(component, EventPollingProtocol)
