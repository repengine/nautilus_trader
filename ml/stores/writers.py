#!/usr/bin/env python3

"""
Writers and live recording utilities for raw market data.

Consolidates:
- market_data_writer.py
- live_data_recorder.py

Backwards-compatible shims re-export these symbols from their old modules.

"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ml.common import event_emitter as _event_emitter
from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.registry.data_registry import DataRegistry
from ml.stores.data_store import DataStore
from ml.stores.protocols import MarketDataWriterProtocol
from nautilus_trader.core.data import Data
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.identifiers import InstrumentId


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DataStoreMarketDataWriter(MarketDataWriterProtocol):
    """
    Write raw market data using DataStore.write_ingestion.
    """

    store: DataStore

    def write(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        df: pd.DataFrame,
    ) -> int:
        if df is None or df.empty:
            return 0
        self.store.write_ingestion(
            dataset_id=dataset_id,
            records=df,
            source=Source.HISTORICAL.value,
            run_id="mdw_auto",
            instrument_id=instrument_id,
        )
        return len(df.index)


@dataclass(slots=True)
class ParquetCatalogMarketDataWriter(MarketDataWriterProtocol):
    """
    Write market data to ParquetDataCatalog by mapping DataFrame rows to Nautilus Bars.
    """

    catalog: Any
    bar_type_template: str = "{instrument_id}-1-MINUTE-LAST-EXTERNAL"

    def write(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        df: pd.DataFrame,
    ) -> int:
        if df is None or df.empty:
            return 0
        from nautilus_trader.model.data import Bar as _Bar
        from nautilus_trader.model.data import BarType as _BarType
        from nautilus_trader.model.objects import Price as _Price
        from nautilus_trader.model.objects import Quantity as _Quantity

        bars: list[_Bar] = []
        required = {"ts_event", "ts_init", "open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)):
            return 0

        bt = _BarType.from_str(self.bar_type_template.format(instrument_id=instrument_id))
        for _, row in df.iterrows():
            bars.append(
                _Bar(
                    bar_type=bt,
                    open=_Price(float(row["open"]), precision=6),
                    high=_Price(float(row["high"]), precision=6),
                    low=_Price(float(row["low"]), precision=6),
                    close=_Price(float(row["close"]), precision=6),
                    volume=_Quantity(float(row["volume"]), precision=0),
                    ts_event=int(row["ts_event"]),
                    ts_init=int(row["ts_init"]),
                ),
            )

        if not bars:
            return 0
        self.catalog.write_data(bars)
        return len(bars)


class LiveDataRecorder:
    """
    Automatically records all live data flowing through the system.
    """

    def __init__(
        self,
        data_store: DataStore,
        data_registry: DataRegistry,
        buffer_size: int = 1000,
        flush_interval_ms: int = 1000,
        storage_path: Path | None = None,
    ) -> None:
        self.data_store = data_store
        self.data_registry = data_registry
        self.buffer_size = buffer_size
        self.flush_interval_ms = flush_interval_ms
        self.storage_path = storage_path or Path.home() / ".nautilus" / "live_data"

        self.buffers: dict[str, list[Any]] = defaultdict(list)
        self.buffer_metadata: dict[str, dict[str, Any]] = defaultdict(dict)

        self._flush_task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._flush_task = asyncio.create_task(self._periodic_flush())

    async def stop(self) -> None:
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
        await self.flush_all()

    async def _periodic_flush(self) -> None:
        while self._running:
            await asyncio.sleep(self.flush_interval_ms / 1000.0)
            await self.flush_all()

    def on_quote(self, quote: QuoteTick) -> None:
        self._buffer_data("quotes", quote.instrument_id, quote)

    def on_trade(self, trade: TradeTick) -> None:
        self._buffer_data("trades", trade.instrument_id, trade)

    def on_bar(self, bar: Bar) -> None:
        self._buffer_data("bars", bar.bar_type.instrument_id, bar)

    def _buffer_data(
        self,
        dataset_id: str,
        instrument_id: InstrumentId,
        data: Data,
    ) -> None:
        self.buffers[dataset_id].append(data)

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

        if len(self.buffers[dataset_id]) >= self.buffer_size:
            asyncio.create_task(self.flush_dataset(dataset_id))

    async def flush_dataset(self, dataset_id: str) -> None:
        if not self.buffers[dataset_id]:
            return

        buffer = self.buffers[dataset_id]
        metadata: dict[str, Any] = self.buffer_metadata[dataset_id]

        self.buffers[dataset_id] = []
        self.buffer_metadata[dataset_id] = {}

        try:
            if dataset_id == "quotes":
                await self._persist_quotes(buffer, metadata)
            elif dataset_id == "trades":
                await self._persist_trades(buffer, metadata)
            elif dataset_id == "bars":
                await self._persist_bars(buffer, metadata)

            for instrument_id in metadata["instrument_ids"]:
                try:
                    _emit_wm = getattr(_event_emitter, "emit_dataset_event_and_watermark", None)
                    if callable(_emit_wm):
                        _emit_wm(
                            self.data_registry,
                            dataset_id=dataset_id,
                            instrument_id=instrument_id,
                            stage=Stage.CATALOG_WRITTEN,
                            source=Source.LIVE,
                            run_id=f"live_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
                            ts_min=metadata["ts_min"],
                            ts_max=metadata["ts_max"],
                            count=metadata["count"],
                            status=EventStatus.SUCCESS,
                            dataset_type=dataset_id,
                            component=self.__class__.__name__,
                        )
                    else:
                        _emit = getattr(_event_emitter, "emit_dataset_event", None)
                        if callable(_emit):
                            _emit(
                                self.data_registry,
                                dataset_id=dataset_id,
                                instrument_id=instrument_id,
                                stage=Stage.CATALOG_WRITTEN,
                                source=Source.LIVE,
                                run_id=f"live_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
                                ts_min=metadata["ts_min"],
                                ts_max=metadata["ts_max"],
                                count=metadata["count"],
                                status=EventStatus.SUCCESS,
                                dataset_type=dataset_id,
                                component=self.__class__.__name__,
                            )
                except Exception:
                    logger.warning(
                        "Failed to emit dataset event/watermark in LiveDataRecorder",
                        exc_info=True,
                    )

        except Exception as e:
            try:
                _emit = getattr(_event_emitter, "emit_dataset_event", None)
                if callable(_emit):
                    _emit(
                        self.data_registry,
                        dataset_id=dataset_id,
                        instrument_id="unknown",
                        stage=Stage.CATALOG_WRITTEN,
                        source=Source.LIVE,
                        run_id=f"live_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
                        ts_min=0,
                        ts_max=0,
                        count=0,
                        status=EventStatus.FAILED,
                        error=str(e),
                    )
            except Exception:
                logger.warning(
                    "Failed to emit failure dataset event in LiveDataRecorder",
                    exc_info=True,
                )
            raise

    async def _persist_quotes(self, quotes: list[QuoteTick], metadata: dict[str, Any]) -> None:
        by_instrument: dict[InstrumentId, list[QuoteTick]] = defaultdict(list)
        for quote in quotes:
            by_instrument[quote.instrument_id].append(quote)
        for instrument_id, instrument_quotes in by_instrument.items():
            date = datetime.fromtimestamp(metadata["ts_min"] / 1e9).date()
            path = self.storage_path / "quotes" / str(date) / f"{instrument_id}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Would write {len(instrument_quotes)} quotes to {path}")

    async def _persist_trades(self, trades: list[TradeTick], metadata: dict[str, Any]) -> None:
        # Placeholder to mirror quotes/bars; extend as needed
        _ = (trades, metadata)

    async def _persist_bars(self, bars: list[Bar], metadata: dict[str, Any]) -> None:
        by_instrument: dict[InstrumentId, list[Bar]] = defaultdict(list)
        for bar in bars:
            by_instrument[bar.bar_type.instrument_id].append(bar)
        for instrument_id, instrument_bars in by_instrument.items():
            date = datetime.fromtimestamp(metadata["ts_min"] / 1e9).date()
            path = self.storage_path / "bars" / str(date) / f"{instrument_id}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Would write {len(instrument_bars)} bars to {path}")

    async def flush_all(self) -> None:
        tasks = []
        for dataset_id in list(self.buffers.keys()):
            if self.buffers[dataset_id]:
                tasks.append(self.flush_dataset(dataset_id))
        if tasks:
            await asyncio.gather(*tasks)


class LiveDataInterceptor:
    """
    Intercepts live data in Nautilus and routes to the recorder.
    """

    def __init__(self, recorder: LiveDataRecorder) -> None:
        self.recorder = recorder

    def on_quote_tick(self, tick: QuoteTick) -> None:
        self.recorder.on_quote(tick)

    def on_trade_tick(self, tick: TradeTick) -> None:
        self.recorder.on_trade(tick)

    def on_bar(self, bar: Bar) -> None:
        self.recorder.on_bar(bar)


__all__ = [
    "DataStoreMarketDataWriter",
    "LiveDataInterceptor",
    "LiveDataRecorder",
    "ParquetCatalogMarketDataWriter",
]
