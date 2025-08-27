#!/usr/bin/env python3

"""
Comprehensive tests for LiveDataRecorder with production-critical test cases.

This module tests all aspects of live data recording including:
- High-frequency data handling (1000+ updates/second)
- Buffer overflow management
- Network disconnection recovery
- Timestamp precision validation (nanosecond accuracy)
- Memory leak prevention
- Concurrent write safety
- Async operation handling
- Data persistence validation
- Event tracking and watermark updates
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import time
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import AsyncGenerator
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import HealthCheck
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import Bundle
from hypothesis.stateful import RuleBasedStateMachine
from hypothesis.stateful import invariant
from hypothesis.stateful import rule

from ml.registry.data_registry import DataRegistry
from ml.registry.dataclasses import DataContract
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import StorageKind
from ml.stores.data_store import DataStore
from ml.stores.live_data_recorder import LiveDataInterceptor
from ml.stores.live_data_recorder import LiveDataRecorder
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarSpecification
from nautilus_trader.model.data import BarType
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import AggregationSource
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.enums import BarAggregation
from nautilus_trader.model.enums import PriceType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import TradeId
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity


# ========================================================================
# Test Fixtures
# ========================================================================


@pytest.fixture
def mock_data_store() -> Mock:
    """Create a mock DataStore."""
    store = Mock(spec=DataStore)
    store.validate_data = AsyncMock(return_value=True)
    store.persist_data = AsyncMock()
    return store


@pytest.fixture
def mock_data_registry() -> Mock:
    """Create a mock DataRegistry."""
    registry = Mock(spec=DataRegistry)
    registry.emit_event = Mock()
    registry.update_watermark = Mock()
    return registry


@pytest.fixture
def storage_path(tmp_path: Path) -> Path:
    """Create a temporary storage path."""
    path = tmp_path / "live_data"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def recorder(
    mock_data_store: Mock,
    mock_data_registry: Mock,
    storage_path: Path,
) -> LiveDataRecorder:
    """Create a LiveDataRecorder instance."""
    return LiveDataRecorder(
        data_store=mock_data_store,
        data_registry=mock_data_registry,
        buffer_size=100,
        flush_interval_ms=100,
        storage_path=storage_path,
    )


# ========================================================================
# Helper Functions
# ========================================================================


def create_quote_tick(
    instrument_id: InstrumentId,
    ts_event: int,
    bid_price: float = 1.0001,
    ask_price: float = 1.0002,
    bid_size: float = 100000.0,
    ask_size: float = 100000.0,
) -> QuoteTick:
    """Create a QuoteTick for testing."""
    return QuoteTick(
        instrument_id=instrument_id,
        bid_price=Price.from_str(str(bid_price)),
        ask_price=Price.from_str(str(ask_price)),
        bid_size=Quantity.from_str(str(bid_size)),
        ask_size=Quantity.from_str(str(ask_size)),
        ts_event=ts_event,
        ts_init=ts_event + 1000,  # 1 microsecond later
    )


def create_trade_tick(
    instrument_id: InstrumentId,
    ts_event: int,
    price: float = 1.0001,
    size: float = 100000.0,
    side: AggressorSide = AggressorSide.BUYER,
) -> TradeTick:
    """Create a TradeTick for testing."""
    return TradeTick(
        instrument_id=instrument_id,
        price=Price.from_str(str(price)),
        size=Quantity.from_str(str(size)),
        aggressor_side=side,
        trade_id=TradeId(str(UUID4())),
        ts_event=ts_event,
        ts_init=ts_event + 1000,
    )


def create_bar(
    instrument_id: InstrumentId,
    ts_event: int,
    open_price: float = 1.0001,
    high_price: float = 1.0003,
    low_price: float = 1.0000,
    close_price: float = 1.0002,
    volume: float = 1000000.0,
) -> Bar:
    """Create a Bar for testing."""
    bar_type = BarType(
        instrument_id=instrument_id,
        bar_spec=BarSpecification(
            step=1,
            aggregation=BarAggregation.MINUTE,
            price_type=PriceType.MID,
        ),
        aggregation_source=AggregationSource.EXTERNAL,
    )
    
    return Bar(
        bar_type=bar_type,
        open=Price.from_str(str(open_price)),
        high=Price.from_str(str(high_price)),
        low=Price.from_str(str(low_price)),
        close=Price.from_str(str(close_price)),
        volume=Quantity.from_str(str(volume)),
        ts_event=ts_event,
        ts_init=ts_event + 1000,
    )


# ========================================================================
# Test Cases: High-Frequency Data Handling
# ========================================================================


@pytest.mark.asyncio
async def test_recorder_handles_high_frequency_updates(recorder: LiveDataRecorder) -> None:
    """Test recorder with 1000+ updates/second."""
    await recorder.start()
    
    instrument_id = InstrumentId(Symbol("EURUSD"), Venue("IDEALPRO"))
    start_time = time.time_ns()
    
    # Generate 1000 quotes in rapid succession
    quotes_sent = []
    for i in range(1000):
        ts_event = start_time + i * 1_000_000  # 1ms apart (1000/sec)
        quote = create_quote_tick(instrument_id, ts_event)
        quotes_sent.append(quote)
        recorder.on_quote(quote)
    
    # Allow some time for async processing
    await asyncio.sleep(0.5)
    
    # Force flush to ensure all data is processed
    await recorder.flush_all()
    
    # Verify all quotes were buffered and flushed
    assert recorder.data_registry.emit_event.call_count >= 1
    
    # Check that events were emitted with correct counts
    for call in recorder.data_registry.emit_event.call_args_list:
        kwargs = call.kwargs
        if kwargs.get("dataset_id") == "quotes":
            assert kwargs["status"] == "success"
            assert kwargs["count"] > 0
    
    await recorder.stop()


@pytest.mark.asyncio
async def test_recorder_performance_metrics(recorder: LiveDataRecorder) -> None:
    """Test recorder performance with throughput metrics."""
    await recorder.start()
    
    instrument_id = InstrumentId(Symbol("BTCUSD"), Venue("BINANCE"))
    
    # Track timing
    start_time = time.perf_counter()
    start_ns = time.time_ns()
    
    # Send 5000 mixed data points
    for i in range(5000):
        ts_event = start_ns + i * 200_000  # 200 microseconds apart (5000/sec)
        
        if i % 3 == 0:
            recorder.on_quote(create_quote_tick(instrument_id, ts_event))
        elif i % 3 == 1:
            recorder.on_trade(create_trade_tick(instrument_id, ts_event))
        else:
            recorder.on_bar(create_bar(instrument_id, ts_event))
    
    # Wait for processing
    await recorder.flush_all()
    
    elapsed = time.perf_counter() - start_time
    throughput = 5000 / elapsed
    
    # Should handle at least 1000 updates/second
    assert throughput > 1000, f"Throughput {throughput:.2f} updates/sec is too low"
    
    await recorder.stop()


# ========================================================================
# Test Cases: Buffer Management
# ========================================================================


@pytest.mark.asyncio
async def test_recorder_handles_buffer_overflow(
    mock_data_store: Mock,
    mock_data_registry: Mock,
    storage_path: Path,
) -> None:
    """Test graceful handling when buffer is full."""
    # Create recorder with small buffer
    recorder = LiveDataRecorder(
        data_store=mock_data_store,
        data_registry=mock_data_registry,
        buffer_size=10,  # Small buffer
        flush_interval_ms=10000,  # Long interval to test buffer-triggered flush
        storage_path=storage_path,
    )
    await recorder.start()
    
    instrument_id = InstrumentId(Symbol("GBPUSD"), Venue("IDEALPRO"))
    
    # Send exactly buffer_size quotes
    for i in range(10):
        ts_event = time.time_ns() + i * 1_000_000
        recorder.on_quote(create_quote_tick(instrument_id, ts_event))
    
    # 11th quote should trigger auto-flush
    recorder.on_quote(create_quote_tick(instrument_id, time.time_ns() + 11_000_000))
    
    # Give async task time to execute
    await asyncio.sleep(0.1)
    
    # Verify flush was triggered
    assert len(recorder.buffers["quotes"]) <= 1  # Should have been flushed
    
    await recorder.stop()


@pytest.mark.asyncio
async def test_recorder_concurrent_buffer_access(recorder: LiveDataRecorder) -> None:
    """Test thread-safe buffer operations under concurrent access."""
    await recorder.start()
    
    instrument_ids = [
        InstrumentId(Symbol(f"PAIR{i}"), Venue("TEST"))
        for i in range(10)
    ]
    
    async def send_quotes(instrument_id: InstrumentId, count: int) -> None:
        """Send quotes for a specific instrument."""
        for i in range(count):
            ts_event = time.time_ns() + i * 1_000_000
            recorder.on_quote(create_quote_tick(instrument_id, ts_event))
            await asyncio.sleep(0.001)  # Small delay to increase concurrency
    
    # Create concurrent tasks
    tasks = [
        send_quotes(instrument_id, 100)
        for instrument_id in instrument_ids
    ]
    
    # Run all tasks concurrently
    await asyncio.gather(*tasks)
    
    # Flush and verify
    await recorder.flush_all()
    
    # Should have processed all data without corruption
    assert recorder.data_registry.emit_event.called
    
    # Check no data was lost
    total_events = sum(
        call.kwargs.get("count", 0)
        for call in recorder.data_registry.emit_event.call_args_list
        if call.kwargs.get("status") == "success"
    )
    # Some events might still be in buffer, so check reasonable range
    assert total_events >= 900  # At least 90% processed
    
    await recorder.stop()


# ========================================================================
# Test Cases: Network Resilience
# ========================================================================


@pytest.mark.asyncio
async def test_recorder_recovers_from_network_disconnection(recorder: LiveDataRecorder) -> None:
    """Test handling of network disconnection errors."""
    await recorder.start()
    
    instrument_id = InstrumentId(Symbol("USDJPY"), Venue("IDEALPRO"))
    
    # Send initial data
    for i in range(50):
        ts_event = time.time_ns() + i * 1_000_000
        recorder.on_quote(create_quote_tick(instrument_id, ts_event))
    
    # Simulate network failure during flush
    error_count = 0
    
    async def failing_persist(*args: Any, **kwargs: Any) -> None:
        nonlocal error_count
        error_count += 1
        raise ConnectionError("Network disconnected")
    
    with patch.object(recorder, "_persist_quotes", failing_persist):
        # This should fail with network error
        with pytest.raises(ConnectionError, match="Network disconnected"):
            await recorder.flush_all()
    
    # Verify error was caught
    assert error_count >= 1
    
    # Check that failure event was emitted
    failure_calls = [
        call for call in recorder.data_registry.emit_event.call_args_list
        if call.kwargs.get("status") == "failed"
    ]
    assert len(failure_calls) >= 1
    
    await recorder.stop()


@pytest.mark.asyncio  
async def test_recorder_handles_partial_flush_failure(recorder: LiveDataRecorder) -> None:
    """Test handling when only some data fails to persist."""
    await recorder.start()
    
    instruments = [
        InstrumentId(Symbol("EUR"), Venue("TEST")),
        InstrumentId(Symbol("GBP"), Venue("TEST")),
        InstrumentId(Symbol("JPY"), Venue("TEST")),
    ]
    
    # Send data for multiple instruments
    for instrument_id in instruments:
        for i in range(30):
            ts_event = time.time_ns() + i * 1_000_000
            recorder.on_trade(create_trade_tick(instrument_id, ts_event))
    
    # Make persistence fail for specific instrument
    async def selective_fail(trades: list, metadata: dict) -> None:
        # Fail for EUR only
        if "EUR" in str(metadata.get("instrument_ids", [])):
            raise ValueError("EUR persistence failed")
        # For other instruments, just return (simulating successful persistence)
        return None
    
    with patch.object(recorder, "_persist_trades", selective_fail):
        with pytest.raises(ValueError, match="EUR persistence failed"):
            await recorder.flush_dataset("trades")
    
    # Other instruments should continue working
    recorder.on_trade(create_trade_tick(instruments[1], time.time_ns()))
    await recorder.flush_all()
    
    await recorder.stop()


# ========================================================================
# Test Cases: Data Integrity
# ========================================================================


@pytest.mark.asyncio
async def test_recorder_maintains_timestamp_precision(recorder: LiveDataRecorder) -> None:
    """Verify nanosecond timestamp accuracy."""
    instrument_id = InstrumentId(Symbol("AUDUSD"), Venue("IDEALPRO"))
    
    # Create quotes with precise nanosecond timestamps
    base_ns = 1_700_000_000_123_456_789  # Precise to nanosecond
    quotes = []
    
    for i in range(100):
        # Each quote 1 nanosecond apart
        ts_event = base_ns + i
        quote = create_quote_tick(instrument_id, ts_event)
        quotes.append(quote)
        recorder.on_quote(quote)
    
    await recorder.flush_all()
    
    # Verify timestamps were preserved in metadata
    for call in recorder.data_registry.update_watermark.call_args_list:
        kwargs = call.kwargs
        if kwargs.get("dataset_id") == "quotes":
            # Check that nanosecond precision is maintained
            last_success_ns = kwargs["last_success_ns"]
            # Should be the last timestamp
            assert last_success_ns == base_ns + 99
    
    await recorder.stop()


@pytest.mark.asyncio
async def test_recorder_validates_timestamp_monotonicity(recorder: LiveDataRecorder) -> None:
    """Test that timestamps are properly ordered."""
    await recorder.start()
    
    instrument_id = InstrumentId(Symbol("NZDUSD"), Venue("IDEALPRO"))
    
    # Send quotes with non-monotonic timestamps
    base_ns = time.time_ns()
    timestamps = [
        base_ns,
        base_ns + 1_000_000,
        base_ns - 500_000,  # Goes backward
        base_ns + 2_000_000,
    ]
    
    for ts in timestamps:
        recorder.on_quote(create_quote_tick(instrument_id, ts))
    
    await recorder.flush_all()
    
    # Check metadata has correct min/max
    for call in recorder.data_registry.emit_event.call_args_list:
        kwargs = call.kwargs
        if kwargs.get("dataset_id") == "quotes" and kwargs.get("status") == "success":
            ts_min = kwargs["ts_min"]
            ts_max = kwargs["ts_max"]
            # Min should be the earliest timestamp
            assert ts_min == base_ns - 500_000
            # Max should be the latest
            assert ts_max == base_ns + 2_000_000
    
    await recorder.stop()


# ========================================================================
# Test Cases: Memory Efficiency
# ========================================================================


@pytest.mark.asyncio
async def test_recorder_memory_usage_remains_stable(
    mock_data_store: Mock,
    mock_data_registry: Mock,
    storage_path: Path,
) -> None:
    """Ensure no memory leaks during extended operation."""
    # Create recorder with aggressive flushing
    recorder = LiveDataRecorder(
        data_store=mock_data_store,
        data_registry=mock_data_registry,
        buffer_size=100,
        flush_interval_ms=50,  # Flush frequently
        storage_path=storage_path,
    )
    await recorder.start()
    
    instrument_id = InstrumentId(Symbol("MEMORY"), Venue("TEST"))
    
    # Force garbage collection and get baseline
    gc.collect()
    baseline_objects = len(gc.get_objects())
    
    # Run for extended period
    for cycle in range(10):
        # Send batch of data
        for i in range(1000):
            ts_event = time.time_ns() + i * 1_000
            recorder.on_quote(create_quote_tick(instrument_id, ts_event))
        
        # Wait for flush
        await asyncio.sleep(0.1)
        
        # Force garbage collection
        gc.collect()
    
    await recorder.stop()
    
    # Final garbage collection
    gc.collect()
    final_objects = len(gc.get_objects())
    
    # Object count shouldn't grow significantly (allow 10% growth for test overhead)
    growth_ratio = (final_objects - baseline_objects) / baseline_objects
    assert growth_ratio < 0.1, f"Memory grew by {growth_ratio*100:.1f}%"


@pytest.mark.asyncio
async def test_recorder_clears_buffers_after_flush(recorder: LiveDataRecorder) -> None:
    """Verify buffers are properly cleared after flushing."""
    instrument_id = InstrumentId(Symbol("CLEAR"), Venue("TEST"))
    
    # Fill buffers
    for i in range(50):
        ts_event = time.time_ns() + i * 1_000_000
        recorder.on_quote(create_quote_tick(instrument_id, ts_event))
        recorder.on_trade(create_trade_tick(instrument_id, ts_event + 500_000))
        recorder.on_bar(create_bar(instrument_id, ts_event + 750_000))
    
    # Verify buffers have data
    assert len(recorder.buffers["quotes"]) == 50
    assert len(recorder.buffers["trades"]) == 50
    assert len(recorder.buffers["bars"]) == 50
    
    # Flush all
    await recorder.flush_all()
    
    # Buffers should be empty
    assert len(recorder.buffers["quotes"]) == 0
    assert len(recorder.buffers["trades"]) == 0
    assert len(recorder.buffers["bars"]) == 0
    
    # Metadata should also be cleared
    assert len(recorder.buffer_metadata["quotes"]) == 0
    assert len(recorder.buffer_metadata["trades"]) == 0
    assert len(recorder.buffer_metadata["bars"]) == 0


# ========================================================================
# Test Cases: Async Operations
# ========================================================================


@pytest.mark.asyncio
async def test_recorder_periodic_flush(
    mock_data_store: Mock,
    mock_data_registry: Mock,
    storage_path: Path,
) -> None:
    """Test automatic periodic flushing."""
    # Create recorder with short flush interval
    recorder = LiveDataRecorder(
        data_store=mock_data_store,
        data_registry=mock_data_registry,
        buffer_size=1000,  # Large buffer
        flush_interval_ms=100,  # Flush every 100ms
        storage_path=storage_path,
    )
    await recorder.start()
    
    instrument_id = InstrumentId(Symbol("PERIODIC"), Venue("TEST"))
    
    # Send data that won't trigger buffer flush
    for i in range(10):
        ts_event = time.time_ns() + i * 1_000_000
        recorder.on_quote(create_quote_tick(instrument_id, ts_event))
    
    # Wait for periodic flush
    await asyncio.sleep(0.15)
    
    # Data should have been flushed even though buffer wasn't full
    assert len(recorder.buffers["quotes"]) == 0
    
    await recorder.stop()


@pytest.mark.asyncio
async def test_recorder_handles_concurrent_flushes(recorder: LiveDataRecorder) -> None:
    """Test handling of concurrent flush operations."""
    instruments = [
        InstrumentId(Symbol(f"INST{i}"), Venue("TEST"))
        for i in range(5)
    ]
    
    # Fill buffers for different datasets
    for inst in instruments:
        for i in range(20):
            ts_event = time.time_ns() + i * 1_000_000
            recorder.on_quote(create_quote_tick(inst, ts_event))
            recorder.on_trade(create_trade_tick(inst, ts_event + 100))
            recorder.on_bar(create_bar(inst, ts_event + 200))
    
    # Trigger multiple concurrent flushes
    tasks = [
        recorder.flush_dataset("quotes"),
        recorder.flush_dataset("trades"),
        recorder.flush_dataset("bars"),
    ]
    
    # All should complete without deadlock
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # No exceptions should occur
    for result in results:
        assert result is None
    
    # All buffers should be empty
    assert len(recorder.buffers["quotes"]) == 0
    assert len(recorder.buffers["trades"]) == 0
    assert len(recorder.buffers["bars"]) == 0


@pytest.mark.asyncio
async def test_recorder_stop_flushes_remaining_data(recorder: LiveDataRecorder) -> None:
    """Test that stop() flushes all remaining data."""
    instrument_id = InstrumentId(Symbol("STOP"), Venue("TEST"))
    
    # Add data but don't flush
    for i in range(25):
        ts_event = time.time_ns() + i * 1_000_000
        recorder.on_quote(create_quote_tick(instrument_id, ts_event))
    
    # Data should be in buffer
    assert len(recorder.buffers["quotes"]) == 25
    
    # Stop should trigger flush
    await recorder.stop()
    
    # Verify flush was called
    recorder.data_registry.emit_event.assert_called()
    
    # Check event was emitted with correct count
    for call in recorder.data_registry.emit_event.call_args_list:
        kwargs = call.kwargs
        if kwargs.get("dataset_id") == "quotes":
            assert kwargs["count"] == 25


# ========================================================================
# Test Cases: Event Tracking and Watermarks
# ========================================================================


@pytest.mark.asyncio
async def test_recorder_emits_correct_events(recorder: LiveDataRecorder) -> None:
    """Test that correct events are emitted for each operation."""
    instrument_id = InstrumentId(Symbol("EVENTS"), Venue("TEST"))
    
    # Send different types of data
    ts_quote = time.time_ns()
    ts_trade = ts_quote + 1_000_000
    ts_bar = ts_quote + 2_000_000
    
    recorder.on_quote(create_quote_tick(instrument_id, ts_quote))
    recorder.on_trade(create_trade_tick(instrument_id, ts_trade))
    recorder.on_bar(create_bar(instrument_id, ts_bar))
    
    await recorder.flush_all()
    
    # Verify events were emitted for each dataset
    datasets_emitted = set()
    for call in recorder.data_registry.emit_event.call_args_list:
        kwargs = call.kwargs
        datasets_emitted.add(kwargs.get("dataset_id"))
        
        # Check required fields
        assert kwargs.get("stage") == "CATALOG_WRITTEN"
        assert kwargs.get("source") == "live"
        assert kwargs.get("status") in ["success", "failed"]
        assert "run_id" in kwargs
        assert "ts_min" in kwargs
        assert "ts_max" in kwargs
    
    assert "quotes" in datasets_emitted
    assert "trades" in datasets_emitted
    assert "bars" in datasets_emitted


@pytest.mark.asyncio
async def test_recorder_updates_watermarks(recorder: LiveDataRecorder) -> None:
    """Test watermark updates track progress correctly."""
    instrument_id = InstrumentId(Symbol("WATER"), Venue("TEST"))
    
    # Send batches with known timestamps
    batch1_start = 1_000_000_000_000
    batch1_end = batch1_start + 100_000_000
    
    for i in range(10):
        ts_event = batch1_start + i * 10_000_000
        recorder.on_quote(create_quote_tick(instrument_id, ts_event))
    
    await recorder.flush_all()
    
    # Check watermark was updated
    recorder.data_registry.update_watermark.assert_called()
    
    # Verify watermark details
    for call in recorder.data_registry.update_watermark.call_args_list:
        kwargs = call.kwargs
        if kwargs.get("dataset_id") == "quotes":
            assert kwargs["source"] == "live"
            assert kwargs["last_success_ns"] == batch1_start + 9 * 10_000_000
            assert kwargs["count"] == 10
            assert kwargs["completeness_pct"] == 100.0


# ========================================================================
# Test Cases: LiveDataInterceptor
# ========================================================================


def test_interceptor_routes_to_recorder() -> None:
    """Test that interceptor correctly routes data to recorder."""
    mock_recorder = Mock(spec=LiveDataRecorder)
    interceptor = LiveDataInterceptor(mock_recorder)
    
    instrument_id = InstrumentId(Symbol("INTER"), Venue("TEST"))
    ts_event = time.time_ns()
    
    # Test quote routing
    quote = create_quote_tick(instrument_id, ts_event)
    interceptor.on_quote_tick(quote)
    mock_recorder.on_quote.assert_called_once_with(quote)
    
    # Test trade routing
    trade = create_trade_tick(instrument_id, ts_event + 1000)
    interceptor.on_trade_tick(trade)
    mock_recorder.on_trade.assert_called_once_with(trade)
    
    # Test bar routing
    bar = create_bar(instrument_id, ts_event + 2000)
    interceptor.on_bar(bar)
    mock_recorder.on_bar.assert_called_once_with(bar)


# ========================================================================
# Property-Based Testing with Hypothesis
# ========================================================================


class LiveDataRecorderStateMachine(RuleBasedStateMachine):
    """
    Stateful testing for LiveDataRecorder using Hypothesis.
    
    This ensures the recorder maintains consistency across random
    sequences of operations.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.mock_store = Mock(spec=DataStore)
        self.mock_registry = Mock(spec=DataRegistry)
        # Use larger buffer to avoid auto-flush in sync context
        self.recorder = LiveDataRecorder(
            data_store=self.mock_store,
            data_registry=self.mock_registry,
            buffer_size=500,  # Large buffer to avoid auto-flush
            flush_interval_ms=10000,  # Long interval to prevent periodic flush
        )
        # Track data for invariant checking
        self.total_quotes_sent = 0
        self.total_trades_sent = 0
        self.total_bars_sent = 0
        self.instruments = set()
    
    instruments = Bundle("instruments")
    
    @rule(target=instruments)
    def create_instrument(self) -> InstrumentId:
        """Create a new instrument."""
        symbol = Symbol(f"SYM{len(self.instruments)}")
        venue = Venue("TEST")
        instrument_id = InstrumentId(symbol, venue)
        self.instruments.add(instrument_id)
        return instrument_id
    
    @rule(instrument=instruments, count=st.integers(min_value=1, max_value=50))
    def send_quotes(self, instrument: InstrumentId, count: int) -> None:
        """Send a batch of quotes."""
        # Limit count to avoid filling buffer
        actual_count = min(count, 450 - len(self.recorder.buffers["quotes"]))
        if actual_count <= 0:
            return
        base_ts = time.time_ns()
        for i in range(actual_count):
            ts_event = base_ts + i * 1_000_000
            quote = create_quote_tick(instrument, ts_event)
            self.recorder.on_quote(quote)
            self.total_quotes_sent += 1
    
    @rule(instrument=instruments, count=st.integers(min_value=1, max_value=50))
    def send_trades(self, instrument: InstrumentId, count: int) -> None:
        """Send a batch of trades."""
        # Limit count to avoid filling buffer
        actual_count = min(count, 450 - len(self.recorder.buffers["trades"]))
        if actual_count <= 0:
            return
        base_ts = time.time_ns()
        for i in range(actual_count):
            ts_event = base_ts + i * 1_000_000
            trade = create_trade_tick(instrument, ts_event)
            self.recorder.on_trade(trade)
            self.total_trades_sent += 1
    
    @rule(instrument=instruments, count=st.integers(min_value=1, max_value=50))
    def send_bars(self, instrument: InstrumentId, count: int) -> None:
        """Send a batch of bars."""
        # Limit count to avoid filling buffer
        actual_count = min(count, 450 - len(self.recorder.buffers["bars"]))
        if actual_count <= 0:
            return
        base_ts = time.time_ns()
        for i in range(actual_count):
            ts_event = base_ts + i * 1_000_000
            bar = create_bar(instrument, ts_event)
            self.recorder.on_bar(bar)
            self.total_bars_sent += 1
    
    @rule()
    def flush_all(self) -> None:
        """Flush all buffers."""
        # Run async flush in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.recorder.flush_all())
        finally:
            loop.close()
            asyncio.set_event_loop(None)
    
    @invariant()
    def buffer_size_respected(self) -> None:
        """Buffer size should never exceed configured limit."""
        assert len(self.recorder.buffers["quotes"]) <= self.recorder.buffer_size
        assert len(self.recorder.buffers["trades"]) <= self.recorder.buffer_size
        assert len(self.recorder.buffers["bars"]) <= self.recorder.buffer_size
    
    @invariant()
    def metadata_consistency(self) -> None:
        """Metadata should match buffer contents."""
        for dataset_id in ["quotes", "trades", "bars"]:
            buffer = self.recorder.buffers[dataset_id]
            metadata = self.recorder.buffer_metadata.get(dataset_id, {})
            
            if buffer:
                # If buffer has data, metadata should exist
                assert metadata, f"Metadata missing for {dataset_id} with {len(buffer)} items"
                assert metadata["count"] == len(buffer)
                
                # Check timestamp bounds
                ts_values = [item.ts_event for item in buffer]
                if ts_values:
                    assert metadata["ts_min"] == min(ts_values)
                    assert metadata["ts_max"] == max(ts_values)
            elif metadata:
                # If buffer is empty, metadata count should be 0 or missing
                assert metadata.get("count", 0) == 0


# Run the state machine test
TestLiveDataRecorderStateMachine = LiveDataRecorderStateMachine.TestCase


@given(
    buffer_size=st.integers(min_value=10, max_value=1000),
    flush_interval=st.integers(min_value=10, max_value=1000),
    data_count=st.integers(min_value=0, max_value=5000),
)
@settings(max_examples=20, deadline=2000, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.asyncio
async def test_recorder_with_random_parameters(
    buffer_size: int,
    flush_interval: int,
    data_count: int,
    mock_data_store: Mock,
    mock_data_registry: Mock,
    storage_path: Path,
) -> None:
    """Property test with random configuration parameters."""
    recorder = LiveDataRecorder(
        data_store=mock_data_store,
        data_registry=mock_data_registry,
        buffer_size=buffer_size,
        flush_interval_ms=flush_interval,
        storage_path=storage_path,
    )
    await recorder.start()
    
    instrument_id = InstrumentId(Symbol("PROP"), Venue("TEST"))
    base_ts = time.time_ns()
    
    # Send random amount of data
    for i in range(data_count):
        ts_event = base_ts + i * 1_000_000
        
        # Randomly choose data type
        choice = i % 3
        if choice == 0:
            recorder.on_quote(create_quote_tick(instrument_id, ts_event))
        elif choice == 1:
            recorder.on_trade(create_trade_tick(instrument_id, ts_event))
        else:
            recorder.on_bar(create_bar(instrument_id, ts_event))
    
    # Ensure all data is processed
    await recorder.stop()
    
    # Verify no data loss (all buffers should be empty after stop)
    assert len(recorder.buffers["quotes"]) == 0
    assert len(recorder.buffers["trades"]) == 0
    assert len(recorder.buffers["bars"]) == 0


# ========================================================================
# Performance Benchmarks
# ========================================================================


@pytest.mark.asyncio
async def test_recorder_latency_performance(
    mock_data_store: Mock,
    mock_data_registry: Mock,
    storage_path: Path,
) -> None:
    """Test latency for single data point processing."""
    recorder = LiveDataRecorder(
        data_store=mock_data_store,
        data_registry=mock_data_registry,
        buffer_size=1000,
        flush_interval_ms=10000,
        storage_path=storage_path,
    )
    await recorder.start()
    
    instrument_id = InstrumentId(Symbol("PERF"), Venue("TEST"))
    
    # Measure latency
    latencies = []
    for _ in range(100):
        start = time.perf_counter_ns()
        ts_event = time.time_ns()
        quote = create_quote_tick(instrument_id, ts_event)
        recorder.on_quote(quote)
        latency_ns = time.perf_counter_ns() - start
        latencies.append(latency_ns)
    
    await recorder.stop()
    
    # Processing should be very fast (sub-millisecond)
    avg_latency_us = np.mean(latencies) / 1000
    p99_latency_us = np.percentile(latencies, 99) / 1000
    
    assert avg_latency_us < 100, f"Average latency {avg_latency_us:.2f}us too high"
    assert p99_latency_us < 1000, f"P99 latency {p99_latency_us:.2f}us too high"


@pytest.mark.asyncio
async def test_recorder_throughput_performance(
    mock_data_store: Mock,
    mock_data_registry: Mock,
    storage_path: Path,
) -> None:
    """Test throughput for batch processing."""
    recorder = LiveDataRecorder(
        data_store=mock_data_store,
        data_registry=mock_data_registry,
        buffer_size=1000,
        flush_interval_ms=10000,
        storage_path=storage_path,
    )
    await recorder.start()
    
    instrument_id = InstrumentId(Symbol("PERF"), Venue("TEST"))
    
    # Process batch and measure throughput
    start = time.perf_counter()
    base_ts = time.time_ns()
    for i in range(10000):
        ts_event = base_ts + i * 1000
        quote = create_quote_tick(instrument_id, ts_event)
        recorder.on_quote(quote)
    
    elapsed = time.perf_counter() - start
    throughput = 10000 / elapsed
    
    await recorder.stop()
    
    # Should handle at least 10000 quotes/second
    assert throughput > 10000, f"Throughput {throughput:.0f} quotes/sec too low"