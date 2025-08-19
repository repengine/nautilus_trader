"""
Utilities for working directly with Nautilus ParquetDataCatalog.

This module provides helper functions to work with ParquetDataCatalog
for ML workflows, replacing the need for custom loaders.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from ml._imports import HAS_POLARS
from ml._imports import check_ml_dependencies
from ml._imports import pl
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


if TYPE_CHECKING:
    import polars as pl


def bars_to_dataframe(
    catalog: ParquetDataCatalog,
    instrument_ids: list[str],
    start: datetime | str | None = None,
    end: datetime | str | None = None,
) -> pl.DataFrame:
    """
    Load bars from catalog and convert to Polars DataFrame.

    Parameters
    ----------
    catalog : ParquetDataCatalog
        The Nautilus data catalog.
    instrument_ids : list[str]
        List of instrument IDs to load.
    start : datetime | str | None
        Start time for data range.
    end : datetime | str | None
        End time for data range.

    Returns
    -------
    pl.DataFrame
        DataFrame with OHLCV data and timestamps.

    """
    if not HAS_POLARS:
        check_ml_dependencies(["polars"])

    # Convert string instrument IDs to InstrumentId objects
    instrument_id_objs = [InstrumentId.from_str(id_str) for id_str in instrument_ids]

    # Load bars from catalog
    bars = catalog.bars(
        instrument_ids=instrument_id_objs,
        start=start,
        end=end,
    )

    if not bars:
        # Return empty DataFrame with expected schema
        return pl.DataFrame({
            "instrument_id": [],
            "timestamp": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        })

    # Convert to DataFrame
    data = []
    for bar in bars:
        data.append({
            "instrument_id": str(bar.bar_type.instrument_id),
            "timestamp": bar.ts_event,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
        })

    return pl.DataFrame(data)


def quotes_to_dataframe(
    catalog: ParquetDataCatalog,
    instrument_ids: list[str],
    start: datetime | str | None = None,
    end: datetime | str | None = None,
) -> pl.DataFrame:
    """
    Load quotes from catalog and convert to Polars DataFrame.

    Parameters
    ----------
    catalog : ParquetDataCatalog
        The Nautilus data catalog.
    instrument_ids : list[str]
        List of instrument IDs to load.
    start : datetime | str | None
        Start time for data range.
    end : datetime | str | None
        End time for data range.

    Returns
    -------
    pl.DataFrame
        DataFrame with quote data.

    """
    if not HAS_POLARS:
        check_ml_dependencies(["polars"])

    # Convert string instrument IDs to InstrumentId objects
    instrument_id_objs = [InstrumentId.from_str(id_str) for id_str in instrument_ids]

    # Load quotes from catalog
    quotes = catalog.quote_ticks(
        instrument_ids=instrument_id_objs,
        start=start,
        end=end,
    )

    if not quotes:
        return pl.DataFrame({
            "instrument_id": [],
            "timestamp": [],
            "bid": [],
            "ask": [],
            "bid_size": [],
            "ask_size": [],
        })

    # Convert to DataFrame
    data = []
    for quote in quotes:
        data.append({
            "instrument_id": str(quote.instrument_id),
            "timestamp": quote.ts_event,
            "bid": float(quote.bid_price),
            "ask": float(quote.ask_price),
            "bid_size": float(quote.bid_size),
            "ask_size": float(quote.ask_size),
        })

    return pl.DataFrame(data)


def trades_to_dataframe(
    catalog: ParquetDataCatalog,
    instrument_ids: list[str],
    start: datetime | str | None = None,
    end: datetime | str | None = None,
) -> pl.DataFrame:
    """
    Load trades from catalog and convert to Polars DataFrame.

    Parameters
    ----------
    catalog : ParquetDataCatalog
        The Nautilus data catalog.
    instrument_ids : list[str]
        List of instrument IDs to load.
    start : datetime | str | None
        Start time for data range.
    end : datetime | str | None
        End time for data range.

    Returns
    -------
    pl.DataFrame
        DataFrame with trade data.

    """
    if not HAS_POLARS:
        check_ml_dependencies(["polars"])

    # Convert string instrument IDs to InstrumentId objects
    instrument_id_objs = [InstrumentId.from_str(id_str) for id_str in instrument_ids]

    # Load trades from catalog
    trades = catalog.trade_ticks(
        instrument_ids=instrument_id_objs,
        start=start,
        end=end,
    )

    if not trades:
        return pl.DataFrame({
            "instrument_id": [],
            "timestamp": [],
            "price": [],
            "size": [],
            "aggressor_side": [],
        })

    # Convert to DataFrame
    data = []
    for trade in trades:
        data.append({
            "instrument_id": str(trade.instrument_id),
            "timestamp": trade.ts_event,
            "price": float(trade.price),
            "size": float(trade.size),
            "aggressor_side": str(trade.aggressor_side),
        })

    return pl.DataFrame(data)
