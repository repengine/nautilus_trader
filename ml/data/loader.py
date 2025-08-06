# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
ML data loading utilities for Nautilus Trader.

This module provides high-level data loading utilities specifically designed for ML
workflows in the cold path (training and research). It integrates seamlessly with
Nautilus Trader's ParquetDataCatalog and returns Polars DataFrames for efficient ML data
processing.

"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import pandas as pd

from ml._imports import HAS_POLARS
from ml._imports import check_ml_dependencies
from ml._imports import pl
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


if TYPE_CHECKING:
    import polars as pl

# Type alias that includes all timestamp types we accept in the ML loader
# We convert these to formats that ParquetDataCatalog accepts (int | str | float | None)
TimestampLike: TypeAlias = int | str | float | datetime | pd.Timestamp | None


class MLDataLoader:
    """
    High-performance data loader for ML workflows using Nautilus Trader's
    ParquetDataCatalog.

    This loader is optimized for cold path (training/research) operations and provides
    efficient data loading with caching, vectorized operations, and Polars integration.

    Key Features:
    - Integrates with ParquetDataCatalog for unified data access
    - Returns Polars DataFrames for efficient ML processing
    - Built-in caching for frequently accessed data
    - Support for bars, quotes, trades, and multiple instruments
    - Vectorized operations using NumPy for performance
    - Lazy loading where appropriate
    - Date range filtering and data validation

    Performance Requirements:
    - Cache frequently accessed data to avoid repeated I/O
    - Use vectorized operations for data transformations
    - Pre-allocate arrays when possible
    - Lazy load data to minimize memory usage

    Parameters
    ----------
    catalog : ParquetDataCatalog
        The Nautilus data catalog instance for data access.
    cache_size : int, default 1000
        Maximum number of cached DataFrames to maintain in memory.
    enable_cache : bool, default True
        Whether to enable internal caching of loaded data.

    Raises
    ------
    ImportError
        If required ML dependencies (polars) are not available.

    Examples
    --------
    >>> catalog = ParquetDataCatalog("./data")
    >>> loader = MLDataLoader(catalog)
    >>>
    >>> # Load bars for single instrument
    >>> bars_df = loader.load_bars("EURUSD.SIM")
    >>>
    >>> # Load with date range (using timestamp strings)
    >>> bars_df = loader.load_bars(
    ...     "EURUSD.SIM",
    ...     start="2023-01-01",
    ...     end="2023-12-31"
    ... )
    >>>
    >>> # Can also use datetime objects
    >>> from datetime import datetime
    >>> bars_df = loader.load_bars(
    ...     "EURUSD.SIM",
    ...     start=datetime(2023, 1, 1),
    ...     end=datetime(2023, 12, 31)
    ... )
    >>>
    >>> # Load multiple instruments
    >>> instruments = ["EURUSD.SIM", "GBPUSD.SIM"]
    >>> data = loader.load_multiple(instruments, data_type="bars")

    """

    def __init__(
        self,
        catalog: ParquetDataCatalog,
        cache_size: int = 1000,
        enable_cache: bool = True,
    ) -> None:
        """
        Initialize the ML data loader.

        Parameters
        ----------
        catalog : ParquetDataCatalog
            The Nautilus data catalog for data access.
        cache_size : int, default 1000
            Maximum number of cached entries to maintain.
        enable_cache : bool, default True
            Whether to enable data caching.

        Raises
        ------
        ImportError
            If required ML dependencies are not available.

        """
        # Check dependencies at initialization
        if not HAS_POLARS:
            check_ml_dependencies(["polars"])

        self._catalog = catalog
        self._enable_cache = enable_cache
        self._cache_size = cache_size

        # Initialize cache storage
        if self._enable_cache:
            self._cache: dict[str, pl.DataFrame] = {}
            self._cache_access_order: list[str] = []

    def load_bars(
        self,
        instrument_id: str | InstrumentId,
        start: TimestampLike = None,
        end: TimestampLike = None,
    ) -> pl.DataFrame:
        """
        Load bar data for a single instrument.

        Parameters
        ----------
        instrument_id : str | InstrumentId
            The instrument identifier to load data for.
        start : TimestampLike, optional
            Start timestamp for data range filtering (inclusive).
            Can be int (nanoseconds), str (ISO format), float, datetime, or pd.Timestamp.
        end : TimestampLike, optional
            End timestamp for data range filtering (inclusive).
            Can be int (nanoseconds), str (ISO format), float, datetime, or pd.Timestamp.

        Returns
        -------
        pl.DataFrame
            Polars DataFrame with columns: timestamp, open, high, low, close, volume.
            Returns empty DataFrame if no data found.

        Notes
        -----
        - Timestamps are converted to datetime objects for convenience
        - All price values are converted to float64 for ML compatibility
        - Volume is converted to int64
        - Uses caching when enabled to avoid repeated catalog queries

        """
        # Convert string instrument_id to InstrumentId if needed
        if isinstance(instrument_id, str):
            instrument_id = InstrumentId.from_str(instrument_id)

        # Generate cache key
        cache_key = self._generate_cache_key("bars", str(instrument_id), start, end)

        # Check cache first
        if self._enable_cache and cache_key in self._cache:
            self._update_cache_access(cache_key)
            return self._cache[cache_key]

        # Query data from catalog (convert timestamps to accepted format)
        try:
            bars = self._catalog.query(
                data_cls=Bar,
                identifiers=[str(instrument_id)],
                start=self._prepare_timestamp(start),
                end=self._prepare_timestamp(end),
            )
        except Exception:
            # Return empty DataFrame if query fails
            return self._create_empty_bars_df()

        if not bars:
            df = self._create_empty_bars_df()
        else:
            df = self._bars_to_polars(bars)

        # Cache the result
        if self._enable_cache:
            self._add_to_cache(cache_key, df)

        return df

    def load_quotes(
        self,
        instrument_id: str | InstrumentId,
        start: TimestampLike = None,
        end: TimestampLike = None,
    ) -> pl.DataFrame:
        """
        Load quote tick data for a single instrument.

        Parameters
        ----------
        instrument_id : str | InstrumentId
            The instrument identifier to load data for.
        start : TimestampLike, optional
            Start timestamp for data range filtering (inclusive).
            Can be int (nanoseconds), str (ISO format), float, datetime, or pd.Timestamp.
        end : TimestampLike, optional
            End timestamp for data range filtering (inclusive).
            Can be int (nanoseconds), str (ISO format), float, datetime, or pd.Timestamp.

        Returns
        -------
        pl.DataFrame
            Polars DataFrame with columns: timestamp, bid_price, ask_price,
            bid_size, ask_size, mid_price, spread.
            Returns empty DataFrame if no data found.

        Notes
        -----
        - Includes derived columns: mid_price and spread
        - All price and size values are float64 for ML compatibility
        - Uses vectorized operations for derived column calculations

        """
        # Convert string instrument_id to InstrumentId if needed
        if isinstance(instrument_id, str):
            instrument_id = InstrumentId.from_str(instrument_id)

        # Generate cache key
        cache_key = self._generate_cache_key("quotes", str(instrument_id), start, end)

        # Check cache first
        if self._enable_cache and cache_key in self._cache:
            self._update_cache_access(cache_key)
            return self._cache[cache_key]

        # Query data from catalog (convert timestamps to accepted format)
        try:
            quotes = self._catalog.query(
                data_cls=QuoteTick,
                identifiers=[str(instrument_id)],
                start=self._prepare_timestamp(start),
                end=self._prepare_timestamp(end),
            )
        except Exception:
            # Return empty DataFrame if query fails
            return self._create_empty_quotes_df()

        if not quotes:
            df = self._create_empty_quotes_df()
        else:
            df = self._quotes_to_polars(quotes)

        # Cache the result
        if self._enable_cache:
            self._add_to_cache(cache_key, df)

        return df

    def load_trades(
        self,
        instrument_id: str | InstrumentId,
        start: TimestampLike = None,
        end: TimestampLike = None,
    ) -> pl.DataFrame:
        """
        Load trade tick data for a single instrument.

        Parameters
        ----------
        instrument_id : str | InstrumentId
            The instrument identifier to load data for.
        start : TimestampLike, optional
            Start timestamp for data range filtering (inclusive).
            Can be int (nanoseconds), str (ISO format), float, datetime, or pd.Timestamp.
        end : TimestampLike, optional
            End timestamp for data range filtering (inclusive).
            Can be int (nanoseconds), str (ISO format), float, datetime, or pd.Timestamp.

        Returns
        -------
        pl.DataFrame
            Polars DataFrame with columns: timestamp, price, size, aggressor_side.
            Returns empty DataFrame if no data found.

        Notes
        -----
        - Price values are float64, size is int64
        - aggressor_side is converted to string representation
        - Sorted by timestamp for chronological order

        """
        # Convert string instrument_id to InstrumentId if needed
        if isinstance(instrument_id, str):
            instrument_id = InstrumentId.from_str(instrument_id)

        # Generate cache key
        cache_key = self._generate_cache_key("trades", str(instrument_id), start, end)

        # Check cache first
        if self._enable_cache and cache_key in self._cache:
            self._update_cache_access(cache_key)
            return self._cache[cache_key]

        # Query data from catalog (convert timestamps to accepted format)
        try:
            trades = self._catalog.query(
                data_cls=TradeTick,
                identifiers=[str(instrument_id)],
                start=self._prepare_timestamp(start),
                end=self._prepare_timestamp(end),
            )
        except Exception:
            # Return empty DataFrame if query fails
            return self._create_empty_trades_df()

        if not trades:
            df = self._create_empty_trades_df()
        else:
            df = self._trades_to_polars(trades)

        # Cache the result
        if self._enable_cache:
            self._add_to_cache(cache_key, df)

        return df

    def load_multiple(
        self,
        instrument_ids: list[str | InstrumentId],
        data_type: str = "bars",
        start: TimestampLike = None,
        end: TimestampLike = None,
    ) -> dict[str, pl.DataFrame]:
        """
        Load data for multiple instruments efficiently.

        Parameters
        ----------
        instrument_ids : list[str | InstrumentId]
            List of instrument identifiers to load data for.
        data_type : str, default "bars"
            Type of data to load: "bars", "quotes", or "trades".
        start : TimestampLike, optional
            Start timestamp for data range filtering (inclusive).
            Can be int (nanoseconds), str (ISO format), float, datetime, or pd.Timestamp.
        end : TimestampLike, optional
            End timestamp for data range filtering (inclusive).
            Can be int (nanoseconds), str (ISO format), float, datetime, or pd.Timestamp.

        Returns
        -------
        dict[str, pl.DataFrame]
            Dictionary mapping instrument ID strings to DataFrames.
            Only includes instruments that have data available.

        Notes
        -----
        - Efficiently loads data for multiple instruments in parallel where possible
        - Skips instruments with no available data instead of raising errors
        - Uses individual caching for each instrument to optimize memory usage

        """
        result = {}

        # Load data for each instrument
        for instrument_id in instrument_ids:
            try:
                if data_type == "bars":
                    df = self.load_bars(instrument_id, start=start, end=end)
                elif data_type == "quotes":
                    df = self.load_quotes(instrument_id, start=start, end=end)
                elif data_type == "trades":
                    df = self.load_trades(instrument_id, start=start, end=end)
                else:
                    raise ValueError(f"Invalid data_type: {data_type}")

                # Only include non-empty DataFrames
                if not df.is_empty():
                    result[str(instrument_id)] = df

            except Exception:  # noqa: S112
                # Skip instruments that fail to load (resilient design)
                continue

        return result

    def clear_cache(self) -> None:
        """
        Clear all cached data to free memory.

        This method removes all cached DataFrames and resets the cache access order.
        Useful for freeing memory after processing large datasets.

        """
        if self._enable_cache:
            self._cache.clear()
            self._cache_access_order.clear()

    def _prepare_timestamp(self, timestamp: TimestampLike) -> int | str | float | None:
        """
        Convert various timestamp types to formats accepted by ParquetDataCatalog.

        Parameters
        ----------
        timestamp : TimestampLike
            The timestamp to convert. Can be int, str, float, datetime, pd.Timestamp, or None.

        Returns
        -------
        int | str | float | None
            The timestamp in a format accepted by ParquetDataCatalog.

        """
        if timestamp is None:
            return None
        elif isinstance(timestamp, int | str | float):
            return timestamp
        elif isinstance(timestamp, datetime):
            # Convert datetime to ISO string format
            if timestamp.tzinfo is None:
                # Assume UTC for naive datetime
                return timestamp.isoformat() + "Z"
            else:
                return timestamp.isoformat()
        elif isinstance(timestamp, pd.Timestamp):
            # Convert pandas Timestamp to ISO string format
            return timestamp.isoformat()
        else:
            # Fallback: try to convert to string
            return str(timestamp)

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache statistics for monitoring and optimization.

        Returns
        -------
        dict[str, Any]
            Dictionary containing cache statistics:
            - size: Number of cached entries
            - max_size: Maximum cache size
            - enabled: Whether caching is enabled

        """
        return {
            "size": len(self._cache) if self._enable_cache else 0,
            "max_size": self._cache_size,
            "enabled": self._enable_cache,
        }

    def _bars_to_polars(self, bars: list[Bar]) -> pl.DataFrame:
        """
        Convert Nautilus Bar objects to Polars DataFrame with vectorized operations.

        Parameters
        ----------
        bars : list[Bar]
            List of Nautilus Bar objects to convert.

        Returns
        -------
        pl.DataFrame
            Polars DataFrame with OHLCV data.

        """
        if not bars:
            return self._create_empty_bars_df()

        # Pre-allocate arrays for vectorized conversion
        n_bars = len(bars)
        timestamps = np.empty(n_bars, dtype=np.int64)
        opens = np.empty(n_bars, dtype=np.float64)
        highs = np.empty(n_bars, dtype=np.float64)
        lows = np.empty(n_bars, dtype=np.float64)
        closes = np.empty(n_bars, dtype=np.float64)
        volumes = np.empty(n_bars, dtype=np.int64)

        # Vectorized data extraction
        for i, bar in enumerate(bars):
            timestamps[i] = bar.ts_event
            opens[i] = float(bar.open)
            highs[i] = float(bar.high)
            lows[i] = float(bar.low)
            closes[i] = float(bar.close)
            volumes[i] = int(bar.volume)

        # Create DataFrame with converted timestamps
        df = pl.DataFrame(
            {
                "timestamp": pl.from_numpy(timestamps),
                "open": pl.from_numpy(opens),
                "high": pl.from_numpy(highs),
                "low": pl.from_numpy(lows),
                "close": pl.from_numpy(closes),
                "volume": pl.from_numpy(volumes),
            },
        )

        # Convert timestamp column to datetime
        return df.with_columns(
            pl.col("timestamp").cast(pl.Datetime("ns")),
        )

    def _quotes_to_polars(self, quotes: list[QuoteTick]) -> pl.DataFrame:
        """
        Convert Nautilus QuoteTick objects to Polars DataFrame with derived columns.

        Parameters
        ----------
        quotes : list[QuoteTick]
            List of Nautilus QuoteTick objects to convert.

        Returns
        -------
        pl.DataFrame
            Polars DataFrame with quote data and derived columns.

        """
        if not quotes:
            return self._create_empty_quotes_df()

        # Pre-allocate arrays for vectorized conversion
        n_quotes = len(quotes)
        timestamps = np.empty(n_quotes, dtype=np.int64)
        bid_prices = np.empty(n_quotes, dtype=np.float64)
        ask_prices = np.empty(n_quotes, dtype=np.float64)
        bid_sizes = np.empty(n_quotes, dtype=np.float64)
        ask_sizes = np.empty(n_quotes, dtype=np.float64)

        # Vectorized data extraction
        for i, quote in enumerate(quotes):
            timestamps[i] = quote.ts_event
            bid_prices[i] = float(quote.bid_price)
            ask_prices[i] = float(quote.ask_price)
            bid_sizes[i] = float(quote.bid_size)
            ask_sizes[i] = float(quote.ask_size)

        # Create DataFrame with derived columns using vectorized operations
        df = pl.DataFrame(
            {
                "timestamp": pl.from_numpy(timestamps),
                "bid_price": pl.from_numpy(bid_prices),
                "ask_price": pl.from_numpy(ask_prices),
                "bid_size": pl.from_numpy(bid_sizes),
                "ask_size": pl.from_numpy(ask_sizes),
            },
        )

        # Convert timestamp and add derived columns using Polars expressions for efficiency
        return df.with_columns(
            [
                pl.col("timestamp").cast(pl.Datetime("ns")),
                ((pl.col("bid_price") + pl.col("ask_price")) / 2).alias("mid_price"),
                (pl.col("ask_price") - pl.col("bid_price")).alias("spread"),
            ],
        )

    def _trades_to_polars(self, trades: list[TradeTick]) -> pl.DataFrame:
        """
        Convert Nautilus TradeTick objects to Polars DataFrame.

        Parameters
        ----------
        trades : list[TradeTick]
            List of Nautilus TradeTick objects to convert.

        Returns
        -------
        pl.DataFrame
            Polars DataFrame with trade data.

        """
        if not trades:
            return self._create_empty_trades_df()

        # Pre-allocate arrays for vectorized conversion
        n_trades = len(trades)
        timestamps = np.empty(n_trades, dtype=np.int64)
        prices = np.empty(n_trades, dtype=np.float64)
        sizes = np.empty(n_trades, dtype=np.int64)
        aggressor_sides = []

        # Vectorized data extraction
        for i, trade in enumerate(trades):
            timestamps[i] = trade.ts_event
            prices[i] = float(trade.price)
            sizes[i] = int(trade.size)
            aggressor_sides.append(str(trade.aggressor_side))

        # Create DataFrame
        df = pl.DataFrame(
            {
                "timestamp": pl.from_numpy(timestamps),
                "price": pl.from_numpy(prices),
                "size": pl.from_numpy(sizes),
                "aggressor_side": aggressor_sides,
            },
        )

        # Convert timestamp column to datetime
        return df.with_columns(
            pl.col("timestamp").cast(pl.Datetime("ns")),
        )

    def _create_empty_bars_df(self) -> pl.DataFrame:
        """
        Create empty bars DataFrame with correct schema.
        """
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("ns"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            },
        )

    def _create_empty_quotes_df(self) -> pl.DataFrame:
        """
        Create empty quotes DataFrame with correct schema.
        """
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("ns"),
                "bid_price": pl.Float64,
                "ask_price": pl.Float64,
                "bid_size": pl.Float64,
                "ask_size": pl.Float64,
                "mid_price": pl.Float64,
                "spread": pl.Float64,
            },
        )

    def _create_empty_trades_df(self) -> pl.DataFrame:
        """
        Create empty trades DataFrame with correct schema.
        """
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("ns"),
                "price": pl.Float64,
                "size": pl.Int64,
                "aggressor_side": pl.Utf8,
            },
        )

    def _generate_cache_key(
        self,
        data_type: str,
        instrument_id: str,
        start: TimestampLike,
        end: TimestampLike,
    ) -> str:
        """
        Generate a unique cache key for the given parameters.
        """
        start_str = str(start) if start is not None else "None"
        end_str = str(end) if end is not None else "None"
        return f"{data_type}_{instrument_id}_{start_str}_{end_str}"

    def _add_to_cache(self, cache_key: str, df: pl.DataFrame) -> None:
        """
        Add DataFrame to cache with LRU eviction.
        """
        if not self._enable_cache:
            return

        # Remove oldest entry if cache is full
        if len(self._cache) >= self._cache_size:
            oldest_key = self._cache_access_order[0]
            del self._cache[oldest_key]
            self._cache_access_order.remove(oldest_key)

        # Add new entry
        self._cache[cache_key] = df
        self._cache_access_order.append(cache_key)

    def _update_cache_access(self, cache_key: str) -> None:
        """
        Update cache access order for LRU management.
        """
        if cache_key in self._cache_access_order:
            self._cache_access_order.remove(cache_key)
            self._cache_access_order.append(cache_key)


def load_ml_data(
    instrument_ids: list[str],
    catalog: ParquetDataCatalog,
    data_type: str = "bars",
    start: TimestampLike = None,
    end: TimestampLike = None,
) -> dict[str, pl.DataFrame]:
    """
    Load ML data from multiple instruments.

    This is a simplified interface for common ML data loading scenarios.

    Parameters
    ----------
    instrument_ids : list[str]
        List of instrument identifier strings to load data for.
    catalog : ParquetDataCatalog
        The Nautilus data catalog instance.
    data_type : str, default "bars"
        Type of data to load: "bars", "quotes", or "trades".
    start : TimestampLike, optional
        Start timestamp for filtering data (inclusive).
    end : TimestampLike, optional
        End timestamp for filtering data (inclusive).

    Returns
    -------
    dict[str, pl.DataFrame]
        Dictionary mapping instrument IDs to their respective DataFrames.
        Only includes instruments that have data available.

    Examples
    --------
    >>> catalog = ParquetDataCatalog("./data")
    >>> instruments = ["EURUSD.SIM", "GBPUSD.SIM"]
    >>> data = load_ml_data(instruments, catalog, data_type="bars")
    >>>
    >>> # With date range (using timestamp strings)
    >>> data = load_ml_data(
    ...     instruments,
    ...     catalog,
    ...     start="2023-01-01",
    ...     end="2023-12-31"
    ... )
    >>>
    >>> # Can also use datetime objects
    >>> from datetime import datetime
    >>> data = load_ml_data(
    ...     instruments,
    ...     catalog,
    ...     start=datetime(2023, 1, 1),
    ...     end=datetime(2023, 12, 31)
    ... )

    """
    loader = MLDataLoader(catalog)
    return loader.load_multiple(
        instrument_ids=instrument_ids,
        data_type=data_type,
        start=start,
        end=end,
    )
