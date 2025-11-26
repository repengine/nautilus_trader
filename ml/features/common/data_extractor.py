"""
DataExtractor component - encapsulates data extraction and preparation from DataFrames.

Extracted from FeatureEngineer god class (Phase 2.1.5).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt


if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    DataFrameLike = pd.DataFrame | pl.DataFrame
else:
    DataFrameLike = Any


class DataExtractor:
    """
    Encapsulates data extraction operations from DataFrames.

    Provides methods to extract numerical arrays from both Pandas and Polars DataFrames,
    handling OHLCV data, L2 quote data, and trade tick data. Implements graceful fallback
    when optional columns are missing.

    This component bridges the gap between raw data storage formats (DataFrames) and the
    numerical arrays required by feature calculation algorithms.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> extractor = DataExtractor()
    >>>
    >>> # Extract OHLCV arrays
    >>> df = pd.DataFrame({
    ...     "open": [100.0, 101.0],
    ...     "high": [101.5, 102.0],
    ...     "low": [99.5, 100.5],
    ...     "close": [100.5, 101.5],
    ...     "volume": [10000.0, 12000.0],
    ... })
    >>> open_arr, high_arr, low_arr, close_arr, vol_arr = extractor.extract_price_arrays(df)
    >>> assert open_arr.shape == (2,)
    >>> assert all(arr.dtype == np.float64 for arr in (open_arr, high_arr, low_arr, close_arr, vol_arr))
    >>>
    >>> # Extract with missing columns (fallback to close)
    >>> df_minimal = pd.DataFrame({"close": [100.0, 101.0]})
    >>> open_arr, high_arr, low_arr, close_arr, vol_arr = extractor.extract_price_arrays(df_minimal)
    >>> assert np.allclose(open_arr, close_arr)  # Fallback to close
    >>> assert np.allclose(vol_arr, np.zeros(2))  # Fallback to zeros
    >>>
    >>> # Extract L2 quote data
    >>> l2_df = pd.DataFrame({
    ...     "bid_price": [99.9, 100.0],
    ...     "ask_price": [100.1, 100.2],
    ...     "bid_size": [1000.0, 1200.0],
    ...     "ask_size": [1100.0, 1300.0],
    ... })
    >>> bid, ask, bid_sz, ask_sz = extractor.extract_bid_ask_data(l2_df)
    >>> assert np.all(bid <= ask)  # Invariant holds
    """

    def __init__(self) -> None:
        """Initialize the DataExtractor."""
        # No state needed - all methods are stateless

    def extract_price_arrays(
        self,
        df: DataFrameLike,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """
        Extract OHLCV price arrays from DataFrame with fallback logic.

        Extracts open, high, low, close, and volume arrays from a DataFrame.
        When optional columns (open, high, low, volume) are missing, falls back
        to close prices for OHLC and zeros for volume.

        Parameters
        ----------
        df : DataFrameLike
            Pandas or Polars DataFrame containing price data.
            Required column: 'close'
            Optional columns: 'open', 'high', 'low', 'volume'

        Returns
        -------
        tuple[NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64]]
            Tuple of (open, high, low, close, volume) arrays, all with dtype np.float64.

        Raises
        ------
        ValueError
            If DataFrame is empty (0 rows).
        KeyError
            If 'close' column is missing (required).

        Examples
        --------
        >>> import pandas as pd
        >>> extractor = DataExtractor()
        >>> df = pd.DataFrame({
        ...     "open": [100.0, 101.0],
        ...     "high": [101.0, 102.0],
        ...     "low": [99.0, 100.0],
        ...     "close": [100.5, 101.5],
        ...     "volume": [10000.0, 12000.0],
        ... })
        >>> open_arr, high_arr, low_arr, close_arr, vol_arr = extractor.extract_price_arrays(df)
        >>> assert open_arr.shape == (2,)
        >>> assert np.allclose(close_arr, [100.5, 101.5])
        """
        # Check for empty DataFrame
        if len(df) == 0:
            raise ValueError("Cannot extract arrays from empty DataFrame")

        # Detect DataFrame type
        is_polars = self._detect_dataframe_type(df) == "polars"

        # Extract arrays based on DataFrame type
        if is_polars:
            # Polars DataFrame
            close_prices = self._ensure_float_array(df["close"].to_numpy())
            open_prices = self._ensure_float_array(
                df["open"].to_numpy() if "open" in df.columns else df["close"].to_numpy()
            )
            high_prices = self._ensure_float_array(
                df["high"].to_numpy() if "high" in df.columns else df["close"].to_numpy()
            )
            low_prices = self._ensure_float_array(
                df["low"].to_numpy() if "low" in df.columns else df["close"].to_numpy()
            )
            volume_source = df["volume"].to_numpy() if "volume" in df.columns else np.zeros(len(df))
            volumes = self._ensure_float_array(volume_source)
        else:
            # Pandas DataFrame
            close_prices = self._ensure_float_array(df["close"].to_numpy())
            open_prices = self._ensure_float_array(
                df["open"].to_numpy() if "open" in df.columns else df["close"].to_numpy()
            )
            high_prices = self._ensure_float_array(
                df["high"].to_numpy() if "high" in df.columns else df["close"].to_numpy()
            )
            low_prices = self._ensure_float_array(
                df["low"].to_numpy() if "low" in df.columns else df["close"].to_numpy()
            )
            volume_source = df["volume"].to_numpy() if "volume" in df.columns else np.zeros(len(df))
            volumes = self._ensure_float_array(volume_source)

        return open_prices, high_prices, low_prices, close_prices, volumes

    def extract_data_arrays(
        self,
        df: DataFrameLike,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64] | None,
        npt.NDArray[np.float64] | None,
    ]:
        """
        Extract close, high, and low arrays from DataFrame.

        Extracts close (required), high (optional), and low (optional) arrays.
        Returns None for high/low when columns are absent.

        Parameters
        ----------
        df : DataFrameLike
            Pandas or Polars DataFrame containing price data.
            Required column: 'close'
            Optional columns: 'high', 'low'

        Returns
        -------
        tuple[NDArray[float64], NDArray[float64] | None, NDArray[float64] | None]
            Tuple of (close, high, low) arrays.
            close is always an array, high and low may be None if columns missing.

        Raises
        ------
        ValueError
            If DataFrame is empty (0 rows).
        KeyError
            If 'close' column is missing (required).

        Examples
        --------
        >>> import pandas as pd
        >>> extractor = DataExtractor()
        >>>
        >>> # Full data
        >>> df = pd.DataFrame({
        ...     "close": [100.0, 101.0],
        ...     "high": [101.0, 102.0],
        ...     "low": [99.0, 100.0],
        ... })
        >>> close, high, low = extractor.extract_data_arrays(df)
        >>> assert high is not None and low is not None
        >>>
        >>> # Missing high/low
        >>> df_minimal = pd.DataFrame({"close": [100.0, 101.0]})
        >>> close, high, low = extractor.extract_data_arrays(df_minimal)
        >>> assert high is None and low is None
        """
        # Check for empty DataFrame
        if len(df) == 0:
            raise ValueError("Cannot extract arrays from empty DataFrame")

        # Detect DataFrame type
        is_polars = self._detect_dataframe_type(df) == "polars"

        if is_polars:
            # Polars DataFrame
            close_array = self._ensure_float_array(df["close"].to_numpy())
            high_array = (
                self._ensure_float_array(df["high"].to_numpy()) if "high" in df.columns else None
            )
            low_array = (
                self._ensure_float_array(df["low"].to_numpy()) if "low" in df.columns else None
            )
        else:
            # Pandas DataFrame
            close_array = self._ensure_float_array(df["close"].to_numpy())
            high_array = (
                self._ensure_float_array(df["high"].to_numpy()) if "high" in df.columns else None
            )
            low_array = (
                self._ensure_float_array(df["low"].to_numpy()) if "low" in df.columns else None
            )

        return close_array, high_array, low_array

    def extract_bid_ask_data(
        self,
        df: DataFrameLike,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """
        Extract bid/ask price and size arrays from L2 quote DataFrame.

        Extracts bid_price, ask_price, bid_size, and ask_size arrays.
        All four columns are required.

        Parameters
        ----------
        df : DataFrameLike
            Pandas or Polars DataFrame containing L2 quote data.
            Required columns: 'bid_price', 'ask_price', 'bid_size', 'ask_size'

        Returns
        -------
        tuple[NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64]]
            Tuple of (bid_prices, ask_prices, bid_sizes, ask_sizes), all dtype np.float64.

        Raises
        ------
        ValueError
            If DataFrame is empty (0 rows).
        KeyError
            If any required column is missing.

        Examples
        --------
        >>> import pandas as pd
        >>> extractor = DataExtractor()
        >>> df = pd.DataFrame({
        ...     "bid_price": [99.9, 100.0],
        ...     "ask_price": [100.1, 100.2],
        ...     "bid_size": [1000.0, 1200.0],
        ...     "ask_size": [1100.0, 1300.0],
        ... })
        >>> bid, ask, bid_sz, ask_sz = extractor.extract_bid_ask_data(df)
        >>> assert np.all(bid <= ask)  # Invariant: bid <= ask
        """
        # Check for empty DataFrame
        if len(df) == 0:
            raise ValueError("Cannot extract arrays from empty DataFrame")

        # Detect DataFrame type
        is_polars = self._detect_dataframe_type(df) == "polars"

        if is_polars:
            return (
                self._ensure_float_array(df["bid_price"].to_numpy()),
                self._ensure_float_array(df["ask_price"].to_numpy()),
                self._ensure_float_array(df["bid_size"].to_numpy()),
                self._ensure_float_array(df["ask_size"].to_numpy()),
            )

        # Pandas DataFrame
        return (
            self._ensure_float_array(df["bid_price"].to_numpy()),
            self._ensure_float_array(df["ask_price"].to_numpy()),
            self._ensure_float_array(df["bid_size"].to_numpy()),
            self._ensure_float_array(df["ask_size"].to_numpy()),
        )

    def extract_trade_data(
        self,
        df: DataFrameLike,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Extract trade price, volume, and side arrays from trade DataFrame.

        Extracts trade_price, trade_volume, and trade_side arrays.
        All three columns are required.

        Parameters
        ----------
        df : DataFrameLike
            Pandas or Polars DataFrame containing trade tick data.
            Required columns: 'trade_price', 'trade_volume', 'trade_side'

        Returns
        -------
        tuple[NDArray[float64], NDArray[float64], NDArray[float64]]
            Tuple of (trade_prices, trade_volumes, trade_sides), all dtype np.float64.
            trade_side values: 1.0 (buy), -1.0 (sell), 0.0 (neutral/auction)

        Raises
        ------
        ValueError
            If DataFrame is empty (0 rows).
        KeyError
            If any required column is missing.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> extractor = DataExtractor()
        >>> df = pd.DataFrame({
        ...     "trade_price": [100.0, 100.1],
        ...     "trade_volume": [500.0, 600.0],
        ...     "trade_side": [1.0, -1.0],  # buy, sell
        ... })
        >>> prices, volumes, sides = extractor.extract_trade_data(df)
        >>> assert np.all(np.isin(sides, [-1.0, 0.0, 1.0]))  # Valid side values
        """
        # Check for empty DataFrame
        if len(df) == 0:
            raise ValueError("Cannot extract arrays from empty DataFrame")

        # Detect DataFrame type
        is_polars = self._detect_dataframe_type(df) == "polars"

        if is_polars:
            return (
                self._ensure_float_array(df["trade_price"].to_numpy()),
                self._ensure_float_array(df["trade_volume"].to_numpy()),
                self._ensure_float_array(df["trade_side"].to_numpy()),
            )

        # Pandas DataFrame
        return (
            self._ensure_float_array(df["trade_price"].to_numpy()),
            self._ensure_float_array(df["trade_volume"].to_numpy()),
            self._ensure_float_array(df["trade_side"].to_numpy()),
        )

    @staticmethod
    def _ensure_float_array(array: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
        """
        Convert any array-like to np.float64 array.

        Handles integer arrays, float32, mixed types, NaN values, and Python lists.
        Preserves NaN values during conversion.

        Parameters
        ----------
        array : NDArray[Any]
            Input array or array-like object.

        Returns
        -------
        NDArray[float64]
            Array converted to np.float64 dtype.

        Examples
        --------
        >>> extractor = DataExtractor()
        >>>
        >>> # Integer array
        >>> arr = np.array([1, 2, 3], dtype=np.int32)
        >>> result = extractor._ensure_float_array(arr)
        >>> assert result.dtype == np.float64
        >>>
        >>> # Float32 array
        >>> arr = np.array([1.5, 2.5], dtype=np.float32)
        >>> result = extractor._ensure_float_array(arr)
        >>> assert result.dtype == np.float64
        >>>
        >>> # Preserve NaN
        >>> arr = np.array([1.0, np.nan, 3.0])
        >>> result = extractor._ensure_float_array(arr)
        >>> assert np.isnan(result[1])
        """
        return np.asarray(array, dtype=np.float64)

    @staticmethod
    def _detect_dataframe_type(df: DataFrameLike) -> Literal["polars", "pandas"]:
        """
        Detect whether a DataFrame is Polars or Pandas.

        Uses attribute checking to determine DataFrame type without requiring
        imports at module level.

        Parameters
        ----------
        df : DataFrameLike
            DataFrame to check.

        Returns
        -------
        Literal["polars", "pandas"]
            "polars" if Polars DataFrame, "pandas" otherwise.

        Examples
        --------
        >>> import pandas as pd
        >>> extractor = DataExtractor()
        >>> df_pd = pd.DataFrame({"a": [1, 2]})
        >>> assert extractor._detect_dataframe_type(df_pd) == "pandas"
        >>>
        >>> # With Polars (if available)
        >>> try:
        ...     import polars as pl
        ...     df_pl = pl.DataFrame({"a": [1, 2]})
        ...     assert extractor._detect_dataframe_type(df_pl) == "polars"
        ... except ImportError:
        ...     pass
        """
        # Polars DataFrames have a to_numpy() method, Pandas use .to_numpy() or .values
        # Check for Polars-specific attribute
        if hasattr(df, "to_numpy") and hasattr(df, "schema"):
            return "polars"
        return "pandas"
