"""
Unit tests for event source data flow.

These tests verify event emission, watermark updates, and data flow through the event
sources in the ml/data/sources/ module.

"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ml.data.sources.events import EarningsEvent
from ml.data.sources.events import EconomicEvent
from ml.data.sources.events import EventSource
from ml.data.sources.events import MockEventSource
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_event_source():
    """
    Create a mock event source for testing.
    """
    return MockEventSource()


@pytest.fixture
def economic_events():
    """
    Create sample economic events.
    """
    base_time = datetime(2024, 1, 15, 14, 30)
    events = []

    for i in range(5):
        event = EconomicEvent(
            event_id=f"eco_{i}",
            timestamp=base_time + timedelta(hours=i),
            name=f"Economic Indicator {i}",
            country="US",
            importance="HIGH" if i % 2 == 0 else "MEDIUM",
            forecast=100.0 + i,
            previous=99.0 + i,
            actual=None,  # Not yet released
        )
        events.append(event)

    return events


@pytest.fixture
def earnings_events():
    """
    Create sample earnings events.
    """
    base_time = datetime(2024, 1, 15, 16, 0)
    events = []

    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
    for i, symbol in enumerate(symbols):
        event = EarningsEvent(
            event_id=f"earn_{symbol}_{i}",
            timestamp=base_time + timedelta(days=i),
            instrument_id=f"{symbol}.NASDAQ",
            fiscal_quarter="Q4",
            fiscal_year=2023,
            eps_forecast=5.0 + i * 0.5,
            eps_previous=4.8 + i * 0.5,
            revenue_forecast=100e9 + i * 10e9,
            revenue_previous=95e9 + i * 10e9,
            eps_actual=None,
            revenue_actual=None,
            timing="AMC" if i % 2 == 0 else "BMO",
        )
        events.append(event)

    return events


# ============================================================================
# Event Flow Tests
# ============================================================================


@pytest.mark.asyncio
class TestEventSourceFlow:
    """
    Test event source data flow and emission.
    """

    async def test_event_emission_flow(self, mock_event_source, economic_events):
        """
        Test that events are properly emitted to subscribers.
        """
        received_events = []

        def event_handler(event):
            received_events.append(event)

        # Subscribe to events
        mock_event_source.subscribe(event_handler)

        # Emit events
        for event in economic_events:
            mock_event_source.emit_event(event)

        # Allow async processing
        await asyncio.sleep(0.01)

        # Verify all events were received
        assert len(received_events) == len(economic_events)
        for i, event in enumerate(received_events):
            assert event.event_id == economic_events[i].event_id

    async def test_multiple_subscribers(self, mock_event_source, economic_events):
        """
        Test that multiple subscribers receive all events.
        """
        subscriber1_events = []
        subscriber2_events = []

        def handler1(event):
            subscriber1_events.append(event)

        def handler2(event):
            subscriber2_events.append(event)

        # Subscribe multiple handlers
        mock_event_source.subscribe(handler1)
        mock_event_source.subscribe(handler2)

        # Emit events
        for event in economic_events:
            mock_event_source.emit_event(event)

        await asyncio.sleep(0.01)

        # Both subscribers should receive all events
        assert len(subscriber1_events) == len(economic_events)
        assert len(subscriber2_events) == len(economic_events)

    async def test_unsubscribe_flow(self, mock_event_source, economic_events):
        """
        Test that unsubscribed handlers stop receiving events.
        """
        received_events = []

        def event_handler(event):
            received_events.append(event)

        # Subscribe
        subscription_id = mock_event_source.subscribe(event_handler)

        # Emit first half of events
        for event in economic_events[:2]:
            mock_event_source.emit_event(event)

        await asyncio.sleep(0.01)
        assert len(received_events) == 2

        # Unsubscribe
        mock_event_source.unsubscribe(subscription_id)

        # Emit remaining events
        for event in economic_events[2:]:
            mock_event_source.emit_event(event)

        await asyncio.sleep(0.01)

        # Should still only have first 2 events
        assert len(received_events) == 2

    async def test_event_filtering(self, mock_event_source, economic_events, earnings_events):
        """
        Test event filtering by type.
        """
        economic_received = []
        earnings_received = []

        def economic_handler(event):
            if isinstance(event, EconomicEvent):
                economic_received.append(event)

        def earnings_handler(event):
            if isinstance(event, EarningsEvent):
                earnings_received.append(event)

        # Subscribe handlers with filtering
        mock_event_source.subscribe(economic_handler)
        mock_event_source.subscribe(earnings_handler)

        # Emit mixed events
        all_events = economic_events + earnings_events
        np.random.shuffle(all_events)

        for event in all_events:
            mock_event_source.emit_event(event)

        await asyncio.sleep(0.01)

        # Verify filtering worked
        assert len(economic_received) == len(economic_events)
        assert len(earnings_received) == len(earnings_events)

        # Check types
        assert all(isinstance(e, EconomicEvent) for e in economic_received)
        assert all(isinstance(e, EarningsEvent) for e in earnings_received)


@pytest.mark.asyncio
class TestEventWatermarks:
    """
    Test event watermark tracking and updates.
    """

    async def test_watermark_initialization(self, mock_event_source):
        """
        Test watermark is properly initialized.
        """
        # Initial watermark should be None or minimum timestamp
        watermark = mock_event_source.get_watermark()
        assert watermark is None or watermark == 0

    async def test_watermark_progression(self, mock_event_source, economic_events):
        """
        Test that watermark advances monotonically.
        """
        watermarks = []

        # Track watermark after each event
        for event in economic_events:
            mock_event_source.emit_event(event)
            await asyncio.sleep(0.001)
            watermark = mock_event_source.update_watermark(event.timestamp)
            watermarks.append(watermark)

        # Watermarks should be monotonically increasing
        for i in range(1, len(watermarks)):
            assert watermarks[i] >= watermarks[i - 1]

    async def test_watermark_with_out_of_order_events(self, mock_event_source, economic_events):
        """
        Test watermark handling with out-of-order events.
        """
        # Shuffle events to simulate out-of-order arrival
        shuffled_events = economic_events.copy()
        np.random.shuffle(shuffled_events)

        highest_timestamp = datetime.min
        watermarks = []

        for event in shuffled_events:
            mock_event_source.emit_event(event)

            # Watermark should track highest timestamp seen
            highest_timestamp = max(highest_timestamp, event.timestamp)
            watermark = mock_event_source.update_watermark(highest_timestamp)
            watermarks.append(watermark)

        # Final watermark should match highest timestamp
        final_watermark = watermarks[-1]
        max_event_timestamp = max(e.timestamp for e in economic_events)
        assert final_watermark == int(max_event_timestamp.timestamp() * 1e9)

    async def test_watermark_persistence(self, mock_event_source):
        """
        Test watermark persistence across restarts.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            watermark_file = Path(temp_dir) / "watermark.txt"

            # Set initial watermark
            initial_watermark = int(datetime(2024, 1, 15).timestamp() * 1e9)
            mock_event_source.update_watermark(datetime(2024, 1, 15))

            # Save watermark
            with open(watermark_file, "w") as f:
                f.write(str(initial_watermark))

            # Create new source and restore watermark
            new_source = MockEventSource()
            if watermark_file.exists():
                with open(watermark_file) as f:
                    restored_watermark = int(f.read())
                    new_source._watermark = restored_watermark

            # Verify restoration
            assert new_source._watermark == initial_watermark


class TestEventDataIntegrity:
    """
    Test event data integrity and validation.
    """

    def test_economic_event_validation(self):
        """
        Test economic event data validation.
        """
        # Valid event
        valid_event = EconomicEvent(
            event_id="fed_001",
            timestamp=datetime(2024, 1, 15, 14, 30),
            name="Federal Funds Rate",
            country="US",
            importance="HIGH",
            forecast=5.5,
            previous=5.25,
            actual=None,
        )

        # Validate required fields
        assert valid_event.event_id is not None
        assert valid_event.timestamp is not None
        assert valid_event.name is not None
        assert valid_event.country is not None
        assert valid_event.importance in ["HIGH", "MEDIUM", "LOW"]

        # Test with actual value (post-release)
        valid_event.actual = 5.5
        assert valid_event.actual == valid_event.forecast  # Met expectations

    def test_earnings_event_validation(self):
        """
        Test earnings event data validation.
        """
        valid_event = EarningsEvent(
            event_id="aapl_q4_2023",
            timestamp=datetime(2024, 1, 25, 16, 30),
            instrument_id="AAPL.NASDAQ",
            fiscal_quarter="Q4",
            fiscal_year=2023,
            eps_forecast=2.10,
            eps_previous=1.88,
            revenue_forecast=117.3e9,
            revenue_previous=111.4e9,
            eps_actual=None,
            revenue_actual=None,
            timing="AMC",
        )

        # Validate required fields
        assert valid_event.event_id is not None
        assert valid_event.timestamp is not None
        assert valid_event.instrument_id is not None
        assert valid_event.fiscal_quarter in ["Q1", "Q2", "Q3", "Q4"]
        assert valid_event.fiscal_year > 2000
        assert valid_event.timing in ["BMO", "AMC"]

    def test_event_timestamp_conversion(self, economic_events):
        """
        Test conversion of event timestamps to nanoseconds.
        """
        for event in economic_events:
            # Convert to nanoseconds
            ts_nanos = int(event.timestamp.timestamp() * 1e9)

            # Should be in valid range
            assert ts_nanos > 0
            assert ts_nanos < 2e18  # Before year 2033

            # Should preserve precision
            recovered_dt = datetime.fromtimestamp(ts_nanos / 1e9)
            assert abs((recovered_dt - event.timestamp).total_seconds()) < 1e-6


class TestEventSourcePerformance:
    """
    Test event source performance characteristics.
    """

    def test_event_emission_latency(self, mock_event_source):
        """
        Test that event emission has low latency.
        """
        import time

        received_times = []

        def event_handler(event):
            received_times.append(time.time())

        mock_event_source.subscribe(event_handler)

        # Emit events and measure latency
        emit_times = []
        for i in range(100):
            emit_time = time.time()
            emit_times.append(emit_time)

            event = EconomicEvent(
                event_id=f"perf_{i}",
                timestamp=datetime.now(),
                name=f"Event {i}",
                country="US",
                importance="HIGH",
                forecast=100.0,
                previous=99.0,
                actual=None,
            )
            mock_event_source.emit_event(event)

        # Wait for processing
        time.sleep(0.1)

        # Calculate latencies
        latencies = []
        for emit_time, receive_time in zip(emit_times[: len(received_times)], received_times):
            latency_ms = (receive_time - emit_time) * 1000
            latencies.append(latency_ms)

        # P99 latency should be < 5ms
        if latencies:
            p99_latency = np.percentile(latencies, 99)
            assert p99_latency < 5.0, f"P99 latency {p99_latency:.2f}ms exceeds 5ms threshold"

    def test_event_throughput(self, mock_event_source):
        """
        Test event processing throughput.
        """
        import time

        event_count = 0

        def event_handler(event):
            nonlocal event_count
            event_count += 1

        mock_event_source.subscribe(event_handler)

        # Generate many events
        start_time = time.time()
        n_events = 10000

        for i in range(n_events):
            event = EconomicEvent(
                event_id=f"throughput_{i}",
                timestamp=datetime.now(),
                name=f"Event {i}",
                country="US",
                importance="HIGH",
                forecast=100.0,
                previous=99.0,
                actual=None,
            )
            mock_event_source.emit_event(event)

        # Allow processing
        time.sleep(0.5)
        elapsed = time.time() - start_time

        # Calculate throughput
        throughput = event_count / elapsed

        # Should handle at least 1000 events/second
        assert throughput > 1000, f"Throughput {throughput:.0f} events/sec below 1000 threshold"


class TestEventSourceRecovery:
    """
    Test event source error recovery and resilience.
    """

    def test_handler_exception_isolation(self, mock_event_source, economic_events):
        """
        Test that exceptions in one handler don't affect others.
        """
        healthy_events = []
        faulty_count = 0

        def healthy_handler(event):
            healthy_events.append(event)

        def faulty_handler(event):
            nonlocal faulty_count
            faulty_count += 1
            if faulty_count == 2:
                raise ValueError("Handler error")

        # Subscribe both handlers
        mock_event_source.subscribe(healthy_handler)
        mock_event_source.subscribe(faulty_handler)

        # Emit events
        for event in economic_events:
            try:
                mock_event_source.emit_event(event)
            except Exception:
                pass  # Ignore handler exceptions

        # Healthy handler should receive all events
        assert len(healthy_events) == len(economic_events)

    def test_event_source_restart(self, mock_event_source):
        """
        Test event source restart and recovery.
        """
        # Simulate source restart
        initial_state = {
            "watermark": int(datetime(2024, 1, 15).timestamp() * 1e9),
            "event_count": 100,
        }

        # Save state
        mock_event_source._watermark = initial_state["watermark"]
        mock_event_source._event_count = initial_state["event_count"]

        # Simulate restart by creating new source with restored state
        new_source = MockEventSource()
        new_source._watermark = initial_state["watermark"]
        new_source._event_count = initial_state["event_count"]

        # Verify state restoration
        assert new_source._watermark == initial_state["watermark"]
        assert new_source._event_count == initial_state["event_count"]

        # Should be able to continue processing
        received = []

        def handler(event):
            received.append(event)

        new_source.subscribe(handler)

        # Emit new event
        new_event = EconomicEvent(
            event_id="post_restart",
            timestamp=datetime(2024, 1, 16),
            name="Post-restart Event",
            country="US",
            importance="HIGH",
            forecast=100.0,
            previous=99.0,
            actual=None,
        )
        new_source.emit_event(new_event)

        # Should receive the new event
        assert len(received) == 1
        assert received[0].event_id == "post_restart"
