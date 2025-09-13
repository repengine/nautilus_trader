#!/usr/bin/env python3

"""
MarketDataWriterProtocol implementation backed by DataStore.

Bridges ingestion orchestrators that operate on DataFrames to the DataStore
facade, ensuring validation + eventing + watermark semantics are applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ml.config.events import Source
from ml.stores.data_store import DataStore
from ml.stores.protocols import MarketDataWriterProtocol


@dataclass(slots=True)
class DataStoreMarketDataWriter(MarketDataWriterProtocol):
    """
    Write raw market data using DataStore.write_ingestion.

    Note: DataStore event semantics are observed — SUCCESS + watermark is only
    emitted if a configured raw writer actually persists data. Otherwise PARTIAL
    or FAILED events are emitted without a watermark update.
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
        # Pass through to DataStore (source=historical for backfills)
        self.store.write_ingestion(
            dataset_id=dataset_id,
            records=df,  # DataFrameLike
            source=Source.HISTORICAL.value,
            run_id="mdw_auto",
            instrument_id=instrument_id,
        )
        return len(df.index)


__all__ = ["DataStoreMarketDataWriter"]


@dataclass(slots=True)
class ParquetCatalogMarketDataWriter(MarketDataWriterProtocol):
    """
    Write market data to ParquetDataCatalog by mapping DataFrame rows to Nautilus Bars.

    This writer is intended for orchestrators (cold path). It converts rows
    with columns `instrument_id, ts_event, ts_init, open, high, low, close, volume`
    into `Bar` domain objects and calls `catalog.write_data(...)`.

    The `bar_type_template` controls BarType resolution and should be a format
    string that includes `{instrument_id}` and timeframe, e.g.:
    "{instrument_id}-1-MINUTE-LAST-EXTERNAL".
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
        # Import Nautilus types lazily
        from nautilus_trader.model.data import Bar as _Bar
        from nautilus_trader.model.data import BarType as _BarType
        from nautilus_trader.model.objects import Price as _Price
        from nautilus_trader.model.objects import Quantity as _Quantity

        # Build bars
        bars: list[_Bar] = []
        # Ensure required columns present
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


__all__ += ["ParquetCatalogMarketDataWriter"]
