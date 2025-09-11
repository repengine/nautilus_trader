"""
Utilities for working directly with Nautilus ParquetDataCatalog.

This module provides helper functions to work with ParquetDataCatalog for ML workflows,
replacing the need for custom loaders.

"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Iterable

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
    return _load_and_build_df(
        catalog=catalog,
        instrument_ids=instrument_ids,
        start=start,
        end=end,
        loader=lambda cat, ids, s, e: cat.bars(instrument_ids=ids, start=s, end=e),
        row_builder=lambda bar: {
            "instrument_id": str(bar.bar_type.instrument_id),
            "timestamp": bar.ts_event,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
        },
        empty_columns=[
            "instrument_id",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ],
    )


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
    return _load_and_build_df(
        catalog=catalog,
        instrument_ids=instrument_ids,
        start=start,
        end=end,
        loader=lambda cat, ids, s, e: cat.quote_ticks(instrument_ids=ids, start=s, end=e),
        row_builder=lambda quote: {
            "instrument_id": str(quote.instrument_id),
            "timestamp": quote.ts_event,
            "bid": float(quote.bid_price),
            "ask": float(quote.ask_price),
            "bid_size": float(quote.bid_size),
            "ask_size": float(quote.ask_size),
        },
        empty_columns=["instrument_id", "timestamp", "bid", "ask", "bid_size", "ask_size"],
    )


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
    return _load_and_build_df(
        catalog=catalog,
        instrument_ids=instrument_ids,
        start=start,
        end=end,
        loader=lambda cat, ids, s, e: cat.trade_ticks(instrument_ids=ids, start=s, end=e),
        row_builder=lambda trade: {
            "instrument_id": str(trade.instrument_id),
            "timestamp": trade.ts_event,
            "price": float(trade.price),
            "size": float(trade.size),
            "aggressor_side": str(trade.aggressor_side),
        },
        empty_columns=["instrument_id", "timestamp", "price", "size", "aggressor_side"],
    )


def _load_and_build_df(
    *,
    catalog: ParquetDataCatalog,
    instrument_ids: list[str],
    start: datetime | str | None,
    end: datetime | str | None,
    loader: Callable[[ParquetDataCatalog, list[InstrumentId], datetime | str | None, datetime | str | None], Iterable[Any]],
    row_builder: Callable[[Any], dict[str, Any]],
    empty_columns: list[str],
) -> pl.DataFrame:
    """
    Shared loader + builder for catalog → Polars DataFrame transforms.
    """
    if not HAS_POLARS:
        check_ml_dependencies(["polars"])

    # Convert string instrument IDs to InstrumentId objects once
    instrument_id_objs = [InstrumentId.from_str(id_str) for id_str in instrument_ids]

    # Load items from catalog via provided loader
    items = list(loader(catalog, instrument_id_objs, start, end))

    if not items:
        # Return empty DataFrame with expected schema
        return pl.DataFrame({col: [] for col in empty_columns})

    # Build rows
    data = [row_builder(item) for item in items]
    return pl.DataFrame(data)
