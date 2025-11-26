"""
FeatureMetricsCollector component - calculates various data quality and market microstructure metrics.

Extracted from FeatureEngineer god class (Phase 2.1.3).
Provides pure calculation methods for column quality metrics, spread metrics, and trade metrics.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    import numpy.typing as npt
    import polars as pl


logger = logging.getLogger(__name__)


class FeatureMetricsCollector:
    """
    Component for calculating various metrics from feature and market data.

    Provides pure calculation methods for:
    - Column quality metrics (null rate, zero rate, outlier detection)
    - Market microstructure metrics (spreads, imbalances from L2 data)
    - Trade flow metrics (VWAP, trade flow imbalance, price impact)

    All methods are defensive and handle edge cases gracefully (empty data,
    invalid values, zero denominators).

    Parameters
    ----------
    logger : logging.Logger | None
        Optional logger instance. If None, uses module-level logger.

    Examples
    --------
    >>> import polars as pl
    >>> import numpy as np
    >>> collector = FeatureMetricsCollector()
    >>>
    >>> # Calculate column quality metrics
    >>> col_data = pl.Series("price", [100.0, 101.0, None, 103.0], dtype=pl.Float64)
    >>> metrics = collector._calculate_column_metrics(col_data, total_rows=4)
    >>> assert "null_rate" in metrics
    >>> assert metrics["null_rate"] == 0.25  # 1 out of 4
    >>>
    >>> # Calculate spread metrics
    >>> bid_prices = np.array([99.0, 99.5, 100.0])
    >>> ask_prices = np.array([100.0, 100.5, 101.0])
    >>> bid_sizes = np.array([100.0, 150.0, 200.0])
    >>> ask_sizes = np.array([80.0, 120.0, 180.0])
    >>> spreads, rel_spreads, imbalances, mid_prices = collector._calculate_spread_metrics(
    ...     bid_prices, ask_prices, bid_sizes, ask_sizes, start_idx=0, end_idx=2
    ... )
    >>> assert len(spreads) == 3
    >>> assert all(s == 1.0 for s in spreads)
    >>>
    >>> # Calculate trade metrics
    >>> trade_prices = np.array([100.0, 100.5, 99.5])
    >>> trade_volumes = np.array([10.0, 20.0, 15.0])
    >>> trade_sides = np.array([1.0, 1.0, -1.0])  # Buy, Buy, Sell
    >>> flow_imb, vwap, intensity, impact, had_trades = collector._calculate_trade_metrics(
    ...     trade_prices, trade_volumes, trade_sides, start_idx=0, end_idx=2
    ... )
    >>> assert had_trades is True
    >>> assert -1.0 <= flow_imb <= 1.0
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the FeatureMetricsCollector.

        Parameters
        ----------
        logger : logging.Logger | None
            Optional logger instance. If None, uses module-level logger.
        """
        self._logger = logger if logger is not None else globals()["logger"]

    def _calculate_column_metrics(
        self,
        col_data: pl.Series,
        total_rows: int,
    ) -> dict[str, float]:
        """
        Calculate quality metrics for a single numeric column.

        Computes various data quality metrics including null rate, zero rate,
        uniqueness ratio, infinity rate (for float columns), and outlier rate
        (using IQR method).

        Parameters
        ----------
        col_data : pl.Series
            Polars Series containing the column data to analyze.
        total_rows : int
            Total number of rows in the dataset (used for rate calculations).

        Returns
        -------
        dict[str, float]
            Dictionary containing quality metrics:
            - null_rate: Fraction of null values in [0, 1]
            - zero_rate: Fraction of zero values in [0, 1]
            - unique_ratio: Fraction of unique values in [0, 1]
            - inf_rate: Fraction of infinite values in [0, 1] (float columns only)
            - outlier_rate: Fraction of outliers in [0, 1] (IQR method)

        Notes
        -----
        - For non-float columns (e.g., integers), inf_rate and outlier_rate default to 0.0
        - Empty series (total_rows=0) returns all rates as 0.0
        - Outlier detection uses the IQR rule-of-thumb (1.5 * IQR)

        Examples
        --------
        >>> import polars as pl
        >>> collector = FeatureMetricsCollector()
        >>> col_data = pl.Series("values", [1.0, 2.0, None, 4.0, 0.0], dtype=pl.Float64)
        >>> metrics = collector._calculate_column_metrics(col_data, total_rows=5)
        >>> assert metrics["null_rate"] == 0.2  # 1/5 nulls
        >>> assert metrics["zero_rate"] == 0.2  # 1/5 zeros
        >>> assert 0.0 <= metrics["outlier_rate"] <= 1.0
        """
        # Basic metrics
        null_count = col_data.null_count()
        zero_count = (col_data == 0.0).sum()
        unique_count = col_data.n_unique()

        metrics = {
            "null_rate": float(null_count) / float(total_rows) if total_rows else 0.0,
            "zero_rate": float(zero_count) / float(total_rows) if total_rows else 0.0,
            "unique_ratio": float(unique_count) / float(total_rows) if total_rows else 0.0,
            "inf_rate": 0.0,
            "outlier_rate": 0.0,
        }

        # Additional metrics for numeric columns only
        import polars as _pl  # local import for type/attr checks

        if col_data.dtype in (_pl.Float32, _pl.Float64):
            inf_count = col_data.is_infinite().sum()
            metrics["inf_rate"] = float(inf_count) / float(total_rows) if total_rows else 0.0
            metrics["outlier_rate"] = self._calculate_outlier_rate(col_data, total_rows)

        return metrics

    def _calculate_outlier_rate(
        self,
        col_data: pl.Series,
        total_rows: int,
    ) -> float:
        """
        Calculate outlier rate using the IQR rule-of-thumb (1.5 * IQR).

        Uses Tukey's fences method for outlier detection:
        - Lower bound: Q1 - 1.5 * IQR
        - Upper bound: Q3 + 1.5 * IQR
        - Outliers: values < lower OR values > upper

        Parameters
        ----------
        col_data : pl.Series
            Polars Series containing numeric data to analyze.
        total_rows : int
            Total number of rows (used for rate calculation).

        Returns
        -------
        float
            Outlier rate in [0, 1] range. Returns 0.0 for edge cases (empty data,
            zero IQR, NaN quantiles, or any exceptions).

        Notes
        -----
        - Defensive behavior: returns 0.0 on any exception (logged at debug level)
        - Zero IQR (all values identical) → 0.0 outlier rate
        - NaN quantiles → 0.0 outlier rate
        - Empty series → 0.0 outlier rate

        Examples
        --------
        >>> import polars as pl
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> collector = FeatureMetricsCollector()
        >>>
        >>> # Normal distribution with outliers
        >>> normal_vals = np.random.normal(50, 10, 95).tolist()
        >>> outliers = [0.0, 0.0, 100.0, 100.0, 100.0]
        >>> col_data = pl.Series("values", normal_vals + outliers, dtype=pl.Float64)
        >>> outlier_rate = collector._calculate_outlier_rate(col_data, total_rows=100)
        >>> assert 0.04 <= outlier_rate <= 0.06  # ~5% outliers
        >>>
        >>> # All same value (zero IQR)
        >>> col_uniform = pl.Series("uniform", [5.0] * 10, dtype=pl.Float64)
        >>> assert collector._calculate_outlier_rate(col_uniform, 10) == 0.0
        """
        try:
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            if q1 is None or q3 is None:
                return 0.0
            if np.isnan(q1) or np.isnan(q3):
                return 0.0
            iqr = q3 - q1
            if iqr <= 0:
                return 0.0
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_count = ((col_data < lower) | (col_data > upper)).sum()
            return float(outlier_count) / float(total_rows) if total_rows else 0.0
        except Exception:
            self._logger.debug("Outlier ratio calculation failed", exc_info=True)
            return 0.0

    def _calculate_spread_metrics(
        self,
        bid_prices: npt.NDArray[np.float64],
        ask_prices: npt.NDArray[np.float64],
        bid_sizes: npt.NDArray[np.float64],
        ask_sizes: npt.NDArray[np.float64],
        start_idx: int,
        end_idx: int,
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        """
        Calculate spread and imbalance metrics for given window.

        Computes market microstructure metrics from Level 2 (L2) bid/ask data:
        - Spread: ask_price - bid_price
        - Relative spread: spread / mid_price
        - Size imbalance: (bid_size - ask_size) / (bid_size + ask_size)
        - Mid price: (bid_price + ask_price) / 2

        Parameters
        ----------
        bid_prices : npt.NDArray[np.float64]
            Array of bid prices.
        ask_prices : npt.NDArray[np.float64]
            Array of ask prices.
        bid_sizes : npt.NDArray[np.float64]
            Array of bid sizes (quantities).
        ask_sizes : npt.NDArray[np.float64]
            Array of ask sizes (quantities).
        start_idx : int
            Starting index of the window (inclusive).
        end_idx : int
            Ending index of the window (inclusive).

        Returns
        -------
        tuple[list[float], list[float], list[float], list[float]]
            Tuple containing:
            - spreads: List of spread values (ask - bid)
            - relative_spreads: List of relative spreads (spread / mid_price)
            - size_imbalances: List of size imbalances in [-1, 1]
            - mid_prices: List of mid prices ((bid + ask) / 2)

        Notes
        -----
        - Only valid ticks are included (bid > 0 and ask > bid)
        - Invalid ticks (bid >= ask, bid <= 0) are skipped silently
        - Zero total_size (bid_size + ask_size = 0) → size_imbalance = 0.0
        - Empty window (start_idx > end_idx) → all lists are empty
        - All lists have the same length (only valid ticks)

        Examples
        --------
        >>> import numpy as np
        >>> collector = FeatureMetricsCollector()
        >>> bid_prices = np.array([99.0, 99.5, 100.0])
        >>> ask_prices = np.array([100.0, 100.5, 101.0])
        >>> bid_sizes = np.array([100.0, 150.0, 200.0])
        >>> ask_sizes = np.array([80.0, 120.0, 180.0])
        >>> spreads, rel_spreads, imbalances, mid_prices = collector._calculate_spread_metrics(
        ...     bid_prices, ask_prices, bid_sizes, ask_sizes, start_idx=0, end_idx=2
        ... )
        >>> assert len(spreads) == 3
        >>> assert all(s == 1.0 for s in spreads)
        >>> assert all(m > 0 for m in mid_prices)
        """
        spreads = []
        relative_spreads = []
        size_imbalances = []
        mid_prices = []

        for i in range(start_idx, end_idx + 1):
            bid = float(bid_prices[i])
            ask = float(ask_prices[i])
            bid_sz = float(bid_sizes[i])
            ask_sz = float(ask_sizes[i])

            if bid > 0 and ask > bid:
                spread = ask - bid
                mid_price = (bid + ask) / 2.0

                spreads.append(spread)
                relative_spreads.append(spread / mid_price if mid_price > 0 else 0.0)
                mid_prices.append(mid_price)

                # Size imbalance: (bid_size - ask_size) / (bid_size + ask_size)
                total_size = bid_sz + ask_sz
                if total_size > 0:
                    size_imbalances.append((bid_sz - ask_sz) / total_size)
                else:
                    size_imbalances.append(0.0)

        return spreads, relative_spreads, size_imbalances, mid_prices

    def _calculate_trade_metrics(
        self,
        trade_prices: npt.NDArray[np.float64],
        trade_volumes: npt.NDArray[np.float64],
        trade_sides: npt.NDArray[np.float64],
        start_idx: int,
        end_idx: int,
    ) -> tuple[float, float, float, float, bool]:
        """
        Calculate trade metrics for given window.

        Computes trade flow and execution quality metrics:
        - Trade flow imbalance: (buy_volume - sell_volume) / total_volume
        - VWAP: Volume-weighted average price
        - Trade intensity: Normalized trade count (capped at 5.0)
        - Average price impact: Mean of price changes relative to previous price

        Parameters
        ----------
        trade_prices : npt.NDArray[np.float64]
            Array of trade prices.
        trade_volumes : npt.NDArray[np.float64]
            Array of trade volumes (quantities).
        trade_sides : npt.NDArray[np.float64]
            Array of trade sides (1.0 for buy, -1.0 for sell).
        start_idx : int
            Starting index of the window (inclusive).
        end_idx : int
            Ending index of the window (inclusive).

        Returns
        -------
        tuple[float, float, float, float, bool]
            Tuple containing:
            - trade_flow_imbalance: Imbalance in [-1, 1] (1=all buy, -1=all sell)
            - vwap: Volume-weighted average price
            - trade_intensity: Normalized trade count in [0, 5.0]
            - avg_price_impact: Average price impact (relative to previous price)
            - had_trades: True if any valid trades, False otherwise

        Notes
        -----
        - Only valid trades are included (volume > 0 and price > 0)
        - Buy trades: side > 0, Sell trades: side <= 0
        - Trade intensity formula: min(trade_count / 20.0, 5.0)
        - No trades (total_volume = 0) → flow_imbalance=0.0, vwap=0.0, intensity=1.0, impact=0.0
        - Price impact only calculated when previous price exists and is > 0
        - Empty window (start_idx > end_idx) → all metrics default to zero/baseline

        Examples
        --------
        >>> import numpy as np
        >>> collector = FeatureMetricsCollector()
        >>> trade_prices = np.array([100.0, 100.5, 99.5, 101.0, 100.0])
        >>> trade_volumes = np.array([10.0, 20.0, 15.0, 25.0, 30.0])
        >>> trade_sides = np.array([1.0, 1.0, -1.0, 1.0, -1.0])
        >>> flow_imb, vwap, intensity, impact, had_trades = collector._calculate_trade_metrics(
        ...     trade_prices, trade_volumes, trade_sides, start_idx=0, end_idx=4
        ... )
        >>> assert had_trades is True
        >>> assert -1.0 <= flow_imb <= 1.0
        >>> assert vwap > 0
        >>> assert impact >= 0.0
        """
        buy_volume = 0.0
        sell_volume = 0.0
        total_volume = 0.0
        vwap_numerator = 0.0
        trade_count = 0
        price_impacts = []

        prev_price = None

        for i in range(start_idx, end_idx + 1):
            price = float(trade_prices[i])
            volume = float(trade_volumes[i])
            side = float(trade_sides[i])

            if volume > 0 and price > 0:
                total_volume += volume
                vwap_numerator += price * volume
                trade_count += 1

                # Separate buy/sell volumes
                if side > 0:  # Buy
                    buy_volume += volume
                else:  # Sell
                    sell_volume += volume

                # Price impact calculation
                if prev_price is not None and prev_price > 0:
                    impact = abs(price - prev_price) / prev_price
                    price_impacts.append(impact)

                prev_price = price

        # Calculate derived metrics
        trade_flow_imbalance = (
            (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0.0
        )
        vwap = vwap_numerator / total_volume if total_volume > 0 else 0.0
        if trade_count <= 0:
            trade_intensity = 1.0
        else:
            trade_intensity = min(float(trade_count) / 20.0, 5.0)  # Normalize and cap
        avg_price_impact = float(np.mean(price_impacts)) if price_impacts else 0.0

        had_trades = trade_count > 0
        return trade_flow_imbalance, vwap, trade_intensity, avg_price_impact, had_trades
