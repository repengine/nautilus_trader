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
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from nautilus_trader.core.data import Data
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.identifiers import InstrumentId

from ml.common import event_emitter as _event_emitter
from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.registry.data_registry import DataRegistry
from ml.schema import map_schema_to_dataset_type
from ml.stores.data_store import DataStore
from ml.stores.io_raw import ParquetCatalogRawWriter
from ml.stores.protocols import DataStoreFacadeProtocol
from ml.stores.protocols import MarketDataWriterProtocol


if TYPE_CHECKING:  # pragma: no cover - typing only
    from ml.registry.dataclasses import DatasetManifest


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DataStoreMarketDataWriter(MarketDataWriterProtocol):
    """
    Write raw market data using DataStore.write_ingestion.
    """

    store: DataStore | DataStoreFacadeProtocol

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


class CatalogWriteFacade:
    """
    Lightweight facade exposing a DataStore-like `write_ingestion` for catalog writers.

    Used in file-backed fallback when PostgreSQL is unavailable. The facade converts
    record lists to DataFrame if necessary and delegates to a MarketDataWriterProtocol
    implementation (e.g., ParquetCatalogMarketDataWriter) for persistence.
    """

    def __init__(self, writer: MarketDataWriterProtocol) -> None:
        self._writer = writer

    def write_ingestion(
        self,
        *,
        dataset_id: str,
        records: list[dict[str, Any]] | pd.DataFrame,
        source: str,
        run_id: str,
        instrument_id: str | None = None,
    ) -> object:
        _ = (source, run_id)  # Parameters kept for interface parity; not used here
        if isinstance(records, list):
            if len(records) == 0:
                # Nothing to write
                return object()
            frame = pd.DataFrame.from_records(records)
        else:
            frame = records
        if instrument_id is None:
            instrument_id = "UNKNOWN"
        # Default schema token for bars; writer may not require it strictly
        self._writer.write(
            dataset_id=dataset_id,
            schema="ohlcv",
            instrument_id=str(instrument_id),
            df=frame,
        )
        return object()


@dataclass(slots=True)
class ParquetCatalogMarketDataWriter(MarketDataWriterProtocol):
    """
    Write market data to ParquetDataCatalog by mapping DataFrame rows to Nautilus Bars.
    """

    catalog: Any
    manifest_resolver: Callable[[str], DatasetManifest | None] | None = None
    default_bar_type_template: str = "{instrument_id}-1-MINUTE-LAST-EXTERNAL"

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

        template = self._resolve_bar_type_template(dataset_id)
        resolved_instrument_id = self._resolve_instrument_token(
            instrument_id=instrument_id,
            df=df,
        )
        if resolved_instrument_id is None:
            logger.warning(
                "Skipping Parquet mirror write due to unresolved instrument identifier",
                extra={
                    "dataset_id": dataset_id,
                    "schema": schema,
                    "instrument_id": instrument_id,
                },
            )
            return 0

        try:
            bt = _BarType.from_str(template.format(instrument_id=resolved_instrument_id))
        except ValueError:
            logger.warning(
                "Skipping Parquet mirror write due to invalid bar type",
                exc_info=True,
                extra={
                    "dataset_id": dataset_id,
                    "schema": schema,
                    "instrument_id": instrument_id,
                    "resolved_instrument_id": resolved_instrument_id,
                },
            )
            return 0
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

    def _resolve_bar_type_template(self, dataset_id: str) -> str:
        if self.manifest_resolver is None:
            return self.default_bar_type_template
        try:
            manifest = self.manifest_resolver(dataset_id)
        except Exception:
            return self.default_bar_type_template
        if manifest is None:
            return self.default_bar_type_template
        metadata = getattr(manifest, "metadata", {})
        candidate = metadata.get("bar_type_template") if isinstance(metadata, dict) else None
        if isinstance(candidate, str) and "{instrument_id}" in candidate:
            return candidate
        return self.default_bar_type_template

    def _resolve_instrument_token(self, *, instrument_id: str, df: pd.DataFrame) -> str | None:
        candidates = self._candidate_instrument_tokens(
            instrument_id=instrument_id,
            df=df,
        )
        for token in candidates:
            try:
                InstrumentId.from_str(token)
            except Exception:
                continue
            return token
        return None

    def _candidate_instrument_tokens(
        self,
        *,
        instrument_id: str,
        df: pd.DataFrame,
    ) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()

        def _add(value: str | None) -> None:
            if value is None:
                return
            token = value.strip()
            if not token or token in seen:
                return
            seen.add(token)
            candidates.append(token)

        _add(instrument_id)

        symbol_like_columns = ("instrument_id", "instrument", "symbol")
        for column in symbol_like_columns:
            if column in df.columns:
                _add(self._first_non_null_str(df[column]))

        venues: list[str] = []
        venue_columns = ("publisher_id", "venue", "exchange", "primary_exchange")
        for column in venue_columns:
            if column in df.columns:
                venue_value = self._first_non_null_str(df[column])
                if venue_value:
                    venues.append(venue_value.replace(" ", ""))

        base_tokens = [token for token in candidates if "." not in token]
        for base in base_tokens:
            for venue in venues:
                _add(f"{base}.{venue}")

        return candidates

    @staticmethod
    def _first_non_null_str(series: pd.Series) -> str | None:
        if series.empty:
            return None
        idx = series.first_valid_index()
        if idx is None:
            return None
        value = series.loc[idx]
        if pd.isna(value):
            return None
        text = str(value).strip()
        return text or None


@dataclass(slots=True)
class ParquetCatalogRawMarketDataWriter(MarketDataWriterProtocol):
    """
    Write market data DataFrames into the Parquet catalog via ``ParquetCatalogRawWriter``.

    This wrapper resolves dataset types from schemas (bars/tbbo/trades/mbp) and reuses
    the raw writer conversions to persist the appropriate Nautilus domain objects.
    """

    catalog: Any
    replace_on_overlap: bool = False

    def __post_init__(self) -> None:
        self._raw_writer = ParquetCatalogRawWriter(
            self.catalog,
            replace_on_overlap=self.replace_on_overlap,
        )

    def write(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        df: pd.DataFrame,
    ) -> int:
        _ = (dataset_id, instrument_id)
        dataset_type = map_schema_to_dataset_type(schema)
        return int(
            self._raw_writer.write(
                dataset_type=dataset_type,
                data=df,
            ),
        )


@dataclass(slots=True)
class FanoutMarketDataWriter(MarketDataWriterProtocol):
    """
    Route market data writes to a primary writer with optional mirror writers.

    Examples
    --------
    >>> primary = DataStoreMarketDataWriter(store=data_store)
    >>> mirror = ParquetCatalogMarketDataWriter(catalog=catalog)
    >>> writer = FanoutMarketDataWriter(primary=primary, mirrors=(mirror,))
    >>> writer.write(
    ...     dataset_id="EQUS.MINI",
    ...     schema="ohlcv-1m",
    ...     instrument_id="SPY.NYSE",
    ...     df=pd.DataFrame(...),
    ... )
    42
    """

    primary: MarketDataWriterProtocol
    mirrors: tuple[MarketDataWriterProtocol, ...] = ()

    def write(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        df: pd.DataFrame,
    ) -> int:
        result = self.primary.write(
            dataset_id=dataset_id,
            schema=schema,
            instrument_id=instrument_id,
            df=df,
        )
        if not self.mirrors or df is None or df.empty:
            return result
        for mirror in self.mirrors:
            try:
                mirror.write(
                    dataset_id=dataset_id,
                    schema=schema,
                    instrument_id=instrument_id,
                    df=df,
                )
            except Exception:
                logger.warning(
                    "Mirror market data write failed",
                    exc_info=True,
                    extra={
                        "dataset_id": dataset_id,
                        "schema": schema,
                        "instrument_id": instrument_id,
                        "mirror": mirror.__class__.__name__,
                    },
                )
        return result


class LiveDataRecorder:
    """
    Automatically records all live data flowing through the system.
    """

    def __init__(
        self,
        data_store: DataStore | DataStoreFacadeProtocol,
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

        # Initialize module metrics via centralized bootstrap (no direct prometheus imports)
        try:
            from ml.common.metrics_manager import MetricsManager as _MM

            _mm = _MM.default()
            self._flush_total = _mm.counter(
                "ml_live_recorder_flush_total",
                "Total number of LiveDataRecorder flushes",
                ["dataset_id"],
            )
            self._records_total = _mm.counter(
                "ml_live_recorder_records_total",
                "Total number of records seen by LiveDataRecorder",
                ["dataset_id"],
            )
            self._flush_seconds = _mm.histogram(
                "ml_live_recorder_flush_seconds",
                "LiveDataRecorder flush duration (seconds)",
                ["dataset_id"],
            )
        except Exception:
            # No-op fallbacks if metrics backend is unavailable
            class _NoMetric:
                def labels(self, *_: object, **__: object) -> _NoMetric:  # pragma: no cover - trivial
                    return self

                def inc(self, *_: object, **__: object) -> None:  # pragma: no cover - trivial
                    return None

                def observe(self, *_: object, **__: object) -> None:  # pragma: no cover - trivial
                    return None

            self._flush_total = _NoMetric()
            self._records_total = _NoMetric()
            self._flush_seconds = _NoMetric()

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

        try:
            self._records_total.labels(dataset_id=dataset_id).inc()
        except Exception:
            logger.debug("Recorder metrics increment failed (ignored)", exc_info=True)

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
            import time as _time
            _t0 = _time.perf_counter()
            if dataset_id == "quotes":
                await self._persist_quotes(buffer, metadata)
            elif dataset_id == "trades":
                await self._persist_trades(buffer, metadata)
            elif dataset_id == "bars":
                await self._persist_bars(buffer, metadata)
            # Emit dataset event/watermark only if persistence path didn't handle emission
            if not metadata.get("emitted_by_store", False):
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

            # Record flush metrics (best-effort)
            try:
                self._flush_total.labels(dataset_id=dataset_id).inc()
                self._flush_seconds.labels(dataset_id=dataset_id).observe(_time.perf_counter() - _t0)
            except Exception:
                logger.debug("Recorder flush metrics observe failed (ignored)", exc_info=True)

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
        if not quotes:
            return
        by_instrument: dict[InstrumentId, list[QuoteTick]] = defaultdict(list)
        for q in quotes:
            by_instrument[q.instrument_id].append(q)

        run_id = f"live_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        for instrument_id, items in by_instrument.items():
            records: list[dict[str, Any]] = []
            def _price_like(obj: Any, *names: str) -> float | None:
                for name in names:
                    val = getattr(obj, name, None)
                    if val is not None:
                        try:
                            return float(val.as_double())
                        except Exception:
                            logger.debug("Failed to extract price-like value for %s", name, exc_info=True)
                            return None
                return None

            for q in items:
                bid_v = _price_like(q, "bid_price", "bid")
                ask_v = _price_like(q, "ask_price", "ask")
                bsz_v = _price_like(q, "bid_size")
                asz_v = _price_like(q, "ask_size")
                if bid_v is None or ask_v is None:
                    logger.debug("Skipping quote lacking bid/ask for %s", instrument_id)
                    continue
                records.append(
                    {
                        "instrument_id": str(instrument_id),
                        "ts_event": int(q.ts_event),
                        "ts_init": int(q.ts_init),
                        "bid": bid_v,
                        "ask": ask_v,
                        "bid_size": float(bsz_v or 0.0),
                        "ask_size": float(asz_v or 0.0),
                    },
                )
            if not records:
                continue
            self.data_store.write_ingestion(
                dataset_id="quotes",
                records=records,
                source=Source.LIVE.value,
                run_id=run_id,
                instrument_id=str(instrument_id),
            )
        # Mark as emitted only when using a real DataStore (facade does not emit events)
        try:
            if isinstance(self.data_store, DataStore):
                metadata["emitted_by_store"] = True
        except Exception:
            logger.debug("Recorder metadata mark failed (ignored)", exc_info=True)

    async def _persist_trades(self, trades: list[TradeTick], metadata: dict[str, Any]) -> None:
        if not trades:
            return
        by_instrument: dict[InstrumentId, list[TradeTick]] = defaultdict(list)
        for t in trades:
            by_instrument[t.instrument_id].append(t)

        run_id = f"live_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        for instrument_id, items in by_instrument.items():
            records: list[dict[str, Any]] = []
            for t in items:
                # Compose record; skip if required fields missing
                price_obj = getattr(t, "price", None)
                size_obj = getattr(t, "size", None)
                if price_obj is None or size_obj is None:
                    logger.debug("Skipping trade lacking price/size for %s", instrument_id)
                    continue
                rec: dict[str, Any] = {
                    "instrument_id": str(instrument_id),
                    "ts_event": int(t.ts_event),
                    "ts_init": int(t.ts_init),
                    "price": float(price_obj.as_double()),
                    "size": float(size_obj.as_double()),
                }
                trade_id = getattr(t, "trade_id", None)
                if trade_id is not None:
                    rec["trade_id"] = str(trade_id)
                aggr = getattr(t, "aggressor_side", None)
                if aggr is not None:
                    rec["aggressor_side"] = str(aggr)
                records.append(rec)
            if not records:
                continue
            self.data_store.write_ingestion(
                dataset_id="trades",
                records=records,
                source=Source.LIVE.value,
                run_id=run_id,
                instrument_id=str(instrument_id),
            )
        try:
            if isinstance(self.data_store, DataStore):
                metadata["emitted_by_store"] = True
        except Exception:
            logger.debug("Recorder metadata mark failed (ignored)", exc_info=True)

    async def _persist_bars(self, bars: list[Bar], metadata: dict[str, Any]) -> None:
        # Persist bars via DataStore.write_ingestion; DataStore emits events/watermarks.
        if not bars:
            return

        by_instrument: dict[InstrumentId, list[Bar]] = defaultdict(list)
        for bar in bars:
            by_instrument[bar.bar_type.instrument_id].append(bar)

        run_id = f"live_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        for instrument_id, instrument_bars in by_instrument.items():
            records: list[dict[str, Any]] = []
            for b in instrument_bars:
                # Use primitive conversions to avoid object allocations
                records.append(
                    {
                        "instrument_id": str(b.bar_type.instrument_id),
                        "ts_event": int(b.ts_event),
                        "ts_init": int(b.ts_init),
                        "open": float(b.open.as_double()),
                        "high": float(b.high.as_double()),
                        "low": float(b.low.as_double()),
                        "close": float(b.close.as_double()),
                        "volume": float(b.volume.as_double()),
                        "source_dataset": "LIVE",
                    },
                )

            self.data_store.write_ingestion(
                dataset_id="bars",
                records=records,
                source=Source.LIVE.value,
                run_id=run_id,
                instrument_id=str(instrument_id),
            )
        try:
            if isinstance(self.data_store, DataStore):
                # Mark emission handled to prevent duplicate events from recorder layer
                metadata["emitted_by_store"] = True
        except Exception:
            logger.debug("Recorder metadata mark failed (ignored)", exc_info=True)

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
    "CatalogWriteFacade",
    "DataStoreMarketDataWriter",
    "FanoutMarketDataWriter",
    "LiveDataInterceptor",
    "LiveDataRecorder",
    "ParquetCatalogMarketDataWriter",
    "ParquetCatalogRawMarketDataWriter",
]
