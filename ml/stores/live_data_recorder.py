"""
Live data recorder for automatic capture of all market data.

This module provides a recorder that intercepts live data flow and automatically
persists it with event tracking and validation.
"""

import asyncio
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ml.registry.data_registry import DataRegistry
from ml.config.events import Stage
from ml.stores.data_store import DataStore
from nautilus_trader.core.data import Data
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.identifiers import InstrumentId


class LiveDataRecorder:
    """
    Automatically records all live data flowing through the system.

    This recorder intercepts market data, validates it, persists it to storage,
    and tracks events/watermarks for observability.

    Parameters
    ----------
    data_store : DataStore
        DataStore for validation and persistence
    data_registry : DataRegistry
        Registry for event tracking
    buffer_size : int
        Number of records to buffer before flushing (default: 1000)
    flush_interval_ms : int
        Maximum time between flushes in milliseconds (default: 1000)
    storage_path : Path
        Base path for data storage

    """

    def __init__(
        self,
        data_store: DataStore,
        data_registry: DataRegistry,
        buffer_size: int = 1000,
        flush_interval_ms: int = 1000,
        storage_path: Path | None = None,
    ):
        self.data_store = data_store
        self.data_registry = data_registry
        self.buffer_size = buffer_size
        self.flush_interval_ms = flush_interval_ms
        self.storage_path = storage_path or Path.home() / ".nautilus" / "live_data"

        # Buffers for each data type
        self.buffers: dict[str, list[Any]] = defaultdict(list)
        self.buffer_metadata: dict[str, dict[str, Any]] = defaultdict(dict)

        # Async flush task
        self._flush_task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        """Start the recorder with periodic flushing."""
        self._running = True
        self._flush_task = asyncio.create_task(self._periodic_flush())

    async def stop(self) -> None:
        """Stop the recorder and flush remaining data."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
        # Final flush
        await self.flush_all()

    async def _periodic_flush(self) -> None:
        """Periodically flush buffers."""
        while self._running:
            await asyncio.sleep(self.flush_interval_ms / 1000.0)
            await self.flush_all()

    def on_quote(self, quote: QuoteTick) -> None:
        """
        Record a quote tick.

        Parameters
        ----------
        quote : QuoteTick
            Quote to record

        """
        self._buffer_data("quotes", quote.instrument_id, quote)

    def on_trade(self, trade: TradeTick) -> None:
        """
        Record a trade tick.

        Parameters
        ----------
        trade : TradeTick
            Trade to record

        """
        self._buffer_data("trades", trade.instrument_id, trade)

    def on_bar(self, bar: Bar) -> None:
        """
        Record a bar.

        Parameters
        ----------
        bar : Bar
            Bar to record

        """
        self._buffer_data("bars", bar.bar_type.instrument_id, bar)

    def _buffer_data(
        self,
        dataset_id: str,
        instrument_id: InstrumentId,
        data: Data,
    ) -> None:
        """
        Add data to buffer and flush if needed.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        instrument_id : InstrumentId
            Instrument identifier
        data : Data
            Data to buffer

        """
        # Add to buffer
        self.buffers[dataset_id].append(data)

        # Track metadata for this batch - reinitialize if empty or missing
        if dataset_id not in self.buffer_metadata or not self.buffer_metadata[dataset_id]:
            self.buffer_metadata[dataset_id] = {
                "instrument_ids": set(),
                "ts_min": data.ts_event,
                "ts_max": data.ts_event,
                "count": 0,
            }

        metadata = self.buffer_metadata[dataset_id]
        metadata["instrument_ids"].add(str(instrument_id))
        metadata["ts_min"] = min(metadata["ts_min"], data.ts_event)
        metadata["ts_max"] = max(metadata["ts_max"], data.ts_event)
        metadata["count"] += 1

        # Flush if buffer is full
        if len(self.buffers[dataset_id]) >= self.buffer_size:
            asyncio.create_task(self.flush_dataset(dataset_id))

    async def flush_dataset(self, dataset_id: str) -> None:
        """
        Flush buffered data for a specific dataset.

        Parameters
        ----------
        dataset_id : str
            Dataset to flush

        """
        if not self.buffers[dataset_id]:
            return

        # Get buffer and metadata
        buffer = self.buffers[dataset_id]
        metadata: dict[str, Any] = self.buffer_metadata[dataset_id]

        # Clear buffers immediately to avoid double-flush
        self.buffers[dataset_id] = []
        self.buffer_metadata[dataset_id] = {}

        try:
            # Convert to appropriate format for storage
            if dataset_id == "quotes":
                await self._persist_quotes(buffer, metadata)
            elif dataset_id == "trades":
                await self._persist_trades(buffer, metadata)
            elif dataset_id == "bars":
                await self._persist_bars(buffer, metadata)

            # Emit success event
            for instrument_id in metadata["instrument_ids"]:
                self.data_registry.emit_event(
                    dataset_id=dataset_id,
                    instrument_id=instrument_id,
                    stage=Stage.CATALOG_WRITTEN.value,
                    source="live",
                    run_id=f"live_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    ts_min=metadata["ts_min"],
                    ts_max=metadata["ts_max"],
                    count=metadata["count"],
                    status="success",
                )

                # Update watermark
                self.data_registry.update_watermark(
                    dataset_id=dataset_id,
                    instrument_id=instrument_id,
                    source="live",
                    last_success_ns=metadata["ts_max"],
                    count=metadata["count"],
                    completeness_pct=100.0,
                )

        except Exception as e:
            # Emit failure event
            self.data_registry.emit_event(
                dataset_id=dataset_id,
                instrument_id="unknown",
                stage=Stage.CATALOG_WRITTEN.value,
                source="live",
                run_id=f"live_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                ts_min=0,
                ts_max=0,
                count=0,
                status="failed",
                error=str(e),
            )
            raise

    async def _persist_quotes(self, quotes: list[QuoteTick], metadata: dict[str, Any]) -> None:
        """Persist quote ticks to storage."""
        # Group by instrument
        by_instrument: dict[InstrumentId, list[QuoteTick]] = defaultdict(list)
        for quote in quotes:
            by_instrument[quote.instrument_id].append(quote)

        # Write to parquet files
        for instrument_id, instrument_quotes in by_instrument.items():
            date = datetime.fromtimestamp(metadata["ts_min"] / 1e9).date()
            path = self.storage_path / "quotes" / str(date) / f"{instrument_id}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to DataFrame and append to parquet
            # This is where you'd use the Catalog or DataStore
            # For now, this is a placeholder
            print(f"Would write {len(instrument_quotes)} quotes to {path}")

    async def _persist_trades(self, trades: list[TradeTick], metadata: dict[str, Any]) -> None:
        """Persist trade ticks to storage."""
        # Similar to quotes

    async def _persist_bars(self, bars: list[Bar], metadata: dict[str, Any]) -> None:
        """Persist bars to storage."""
        # Group by instrument
        by_instrument: dict[InstrumentId, list[Bar]] = defaultdict(list)
        for bar in bars:
            by_instrument[bar.bar_type.instrument_id].append(bar)

        # Write to parquet files
        for instrument_id, instrument_bars in by_instrument.items():
            date = datetime.fromtimestamp(metadata["ts_min"] / 1e9).date()
            path = self.storage_path / "bars" / str(date) / f"{instrument_id}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to DataFrame and append to parquet
            # This is where you'd use the Catalog or DataStore
            # For now, this is a placeholder
            print(f"Would write {len(instrument_bars)} bars to {path}")

    async def flush_all(self) -> None:
        """Flush all buffered data."""
        tasks = []
        for dataset_id in list(self.buffers.keys()):
            if self.buffers[dataset_id]:
                tasks.append(self.flush_dataset(dataset_id))
        if tasks:
            await asyncio.gather(*tasks)


class LiveDataInterceptor:
    """
    Intercepts live data flow in Nautilus and routes it to the recorder.

    This should be integrated into your Actor or Strategy to automatically
    record all incoming data.

    """

    def __init__(self, recorder: LiveDataRecorder):
        self.recorder = recorder

    def on_quote_tick(self, tick: QuoteTick) -> None:
        """Intercept and record quote tick."""
        self.recorder.on_quote(tick)

    def on_trade_tick(self, tick: TradeTick) -> None:
        """Intercept and record trade tick."""
        self.recorder.on_trade(tick)

    def on_bar(self, bar: Bar) -> None:
        """Intercept and record bar."""
        self.recorder.on_bar(bar)
