"""
Indicator management for feature engineering.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from nautilus_trader.model.data import Bar

from ml.config.constants import IndicatorNames
from ml.config.constants import SystemConstants
from ml.features.config import FeatureConfig


from ml._imports import HAS_POLARS, pl

logger = logging.getLogger(__name__)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Perform safe division with zero check.

    Parameters
    ----------
    numerator : float
        The numerator value.
    denominator : float
        The denominator value.
    default : float, default 0.0
        Default value to return if denominator is zero.

    Returns
    -------
    float
        Result of division or default value if denominator is zero.

    """
    if denominator == 0 or denominator is None:
        return default
    return numerator / denominator


# sklearn is optional; import lazily where used and gate via HAS_SKLEARN

try:  # pragma: no cover - import compatibility shim
    # Repository layout exposes ATR at `indicators.atr` (compiled extension)
    from nautilus_trader.indicators.atr import AverageTrueRange
except Exception:
    try:
        # Older or alternative packaging layouts
        from nautilus_trader.indicators.volatility import AverageTrueRange
    except Exception:
        try:
            from nautilus_trader.indicators.average_true_range import AverageTrueRange
        except Exception:

            class _FallbackAverageTrueRange:
                """
                Minimal ATR fallback to keep services operational when indicators are
                unavailable.

                Implements the subset used by FeatureEngineer: update_raw/high-low-close,
                handle_bar(bar), value, initialized, and reset().

                """

                def __init__(self, period: int) -> None:
                    self._period = max(2, int(period))
                    self._values: list[float] = []
                    self._prev_close: float | None = None
                    self.value: float = 0.0
                    self.initialized: bool = False

                def update_raw(self, high: float, low: float, close: float) -> None:
                    tr = float(high) - float(low)
                    if self._prev_close is not None:
                        tr = max(
                            tr,
                            abs(float(high) - self._prev_close),
                            abs(float(low) - self._prev_close),
                        )
                    self._prev_close = float(close)
                    self._values.append(tr)
                    if len(self._values) > self._period:
                        self._values.pop(0)
                    if len(self._values) >= self._period:
                        self.value = sum(self._values) / float(len(self._values))
                        self.initialized = True

                def handle_bar(self, bar: Any) -> None:  # Use Any to avoid early import
                    self.update_raw(float(bar.high), float(bar.low), float(bar.close))

                def reset(self) -> None:
                    self._values.clear()
                    self._prev_close = None
                    self.value = 0.0
                    self.initialized = False

            # Alias fallback under expected name
            AverageTrueRange = _FallbackAverageTrueRange

# Import indicators from the repository layout (compiled Cython extensions)
# Many indicators are optional at import time; if unavailable, fall back to None
try:
    from nautilus_trader.indicators.average.ema import ExponentialMovingAverage
    from nautilus_trader.indicators.average.sma import SimpleMovingAverage
except Exception:
    ExponentialMovingAverage = None
    SimpleMovingAverage = None

try:
    from nautilus_trader.indicators.bollinger_bands import BollingerBands
except Exception:
    BollingerBands = None

try:
    from nautilus_trader.indicators.macd import MovingAverageConvergenceDivergence as MACD
except Exception:
    MACD = None

try:
    from nautilus_trader.indicators.rsi import RelativeStrengthIndex
except Exception:
    RelativeStrengthIndex = None


class IndicatorManager:
    """
    Manages Nautilus indicators for consistent feature calculation.

    This class maintains stateful indicators and provides methods to update them with
    market data and retrieve their values for feature computation. It ensures perfect
    consistency between batch and real-time calculations.

    """

    def __init__(self, config: FeatureConfig) -> None:
        """
        Initialize indicator manager.

        Parameters
        ----------
        config : FeatureConfig
            Configuration for indicators and features.

        """
        self.config = config
        self.indicators: dict[str, Any] = {}
        self._initialize_indicators()

        # Track price history for features that need it (with memory limits)
        self.price_history: dict[str, list[float]] = {
            "closes": [],
            "volumes": [],
            "highs": [],
            "lows": [],
        }

    def _initialize_indicators(self) -> None:
        """
        Initialize all Nautilus indicators based on configuration.
        """
        specs = self.config.get_indicator_specs()

        for name, spec in specs.items():
            if spec["type"] == "SMA":
                if SimpleMovingAverage is not None:
                    self.indicators[name] = SimpleMovingAverage(spec["period"])
                else:
                    # Use numpy fallback
                    self.indicators[name] = None
            elif spec["type"] == "EMA":
                if ExponentialMovingAverage is not None:
                    self.indicators[name] = ExponentialMovingAverage(spec["period"])
                else:
                    # Use numpy fallback
                    self.indicators[name] = None
            elif spec["type"] == "RSI":
                if RelativeStrengthIndex is not None:
                    self.indicators[name] = RelativeStrengthIndex(spec["period"])
                else:
                    # Use numpy fallback
                    self.indicators[name] = None
            elif spec["type"] == "BB":
                if BollingerBands is not None:
                    self.indicators[name] = BollingerBands(spec["period"], spec["std"])
                else:
                    # Use numpy fallback
                    self.indicators[name] = None
            elif spec["type"] == "ATR":
                self.indicators[name] = AverageTrueRange(spec["period"])
            elif spec["type"] == "MACD":
                if MACD is not None:
                    self.indicators[name] = MACD(spec["fast"], spec["slow"])
                else:
                    # Use numpy fallback
                    self.indicators[name] = None

    def update_from_bar(self, bar: Bar) -> None:
        """
        Update all indicators from a bar.

        Parameters
        ----------
        bar : Bar
            The bar to update indicators with.

        """
        # Update price history with memory management
        self.price_history["closes"].append(float(bar.close))
        self.price_history["volumes"].append(float(bar.volume))
        self.price_history["highs"].append(float(bar.high))
        self.price_history["lows"].append(float(bar.low))

        # Keep history limited to avoid memory issues
        max_history = SystemConstants.PRICE_HISTORY_MAXLEN
        for key in self.price_history:
            if len(self.price_history[key]) > max_history:
                self.price_history[key] = self.price_history[key][-max_history:]

        # Update indicators
        specs = self.config.get_indicator_specs()

        for name, indicator in self.indicators.items():
            spec = specs.get(name)
            if spec is None or indicator is None:
                continue

            # Handle different input types
            if spec.get("input") == "volume":
                indicator.update_raw(float(bar.volume))
            elif spec["type"] in ["ATR", "BB", "MACD"]:
                # These need full bar data
                indicator.handle_bar(bar)
            else:
                # Default to close price
                indicator.update_raw(float(bar.close))

    def update_from_values(self, *, close: float, high: float, low: float, volume: float) -> None:
        """
        Update all indicators from raw OHLCV values.

        This mirrors update_from_bar but avoids constructing Bar objects, and is used by
        the FeatureEngineer hot path convenience API.

        """
        # Update price history with memory management
        self.price_history["closes"].append(float(close))
        self.price_history["volumes"].append(float(volume))
        self.price_history["highs"].append(float(high))
        self.price_history["lows"].append(float(low))

        max_history = SystemConstants.PRICE_HISTORY_MAXLEN
        for key in self.price_history:
            if len(self.price_history[key]) > max_history:
                self.price_history[key] = self.price_history[key][-max_history:]

        # Update indicators based on spec
        specs = self.config.get_indicator_specs()
        for name, indicator in self.indicators.items():
            spec = specs.get(name)
            # Skip indicators which are not configured or which are using
            # numpy fallbacks (represented by None). This mirrors the
            # guard used in `update_from_bar` to keep parity across paths.
            if spec is None or indicator is None:
                continue
            if spec.get("input") == "volume":
                indicator.update_raw(float(volume))
            elif spec["type"] == "ATR":
                indicator.update_raw(float(high), float(low), float(close))
            elif spec["type"] == "BB":
                indicator.update_raw(float(high), float(low), float(close))
            elif spec["type"] == "MACD":
                indicator.update_raw(float(close))
            else:
                indicator.update_raw(float(close))

    def update_batch_vectorized(
        self,
        open_prices: npt.NDArray[np.float64],
        high_prices: npt.NDArray[np.float64],
        low_prices: npt.NDArray[np.float64],
        close_prices: npt.NDArray[np.float64],
        volumes: npt.NDArray[np.float64],
    ) -> list[dict[str, float]]:
        """
        Update indicators using vectorized operations for batch processing.

        This method processes all price data efficiently without creating Bar objects,
        while maintaining perfect feature parity with the online update_from_bar method.

        Parameters
        ----------
        open_prices : npt.NDArray[np.float64]
            Array of open prices.
        high_prices : npt.NDArray[np.float64]
            Array of high prices.
        low_prices : npt.NDArray[np.float64]
            Array of low prices.
        close_prices : npt.NDArray[np.float64]
            Array of close prices.
        volumes : npt.NDArray[np.float64]
            Array of volumes.

        Returns
        -------
        list[dict[str, float]]
            List of indicator value dictionaries for each timestamp.

        """
        # Update price history for continuity
        self.price_history["closes"].extend(close_prices)
        self.price_history["volumes"].extend(volumes)
        self.price_history["highs"].extend(high_prices)
        self.price_history["lows"].extend(low_prices)

        max_history = SystemConstants.PRICE_HISTORY_MAXLEN
        for key in self.price_history:
            if len(self.price_history[key]) > max_history:
                self.price_history[key] = self.price_history[key][-max_history:]

        if HAS_POLARS and pl is not None:
            try:
                return self._update_batch_polars(
                    open_prices, high_prices, low_prices, close_prices, volumes
                )
            except Exception as e:
                logger.warning(
                    "Polars vectorization failed, falling back to loop: %s",
                    e,
                    exc_info=True,
                )

        # Fallback to loop
        n_bars = len(close_prices)
        all_values = []

        # Process each bar but without creating Bar objects
        for idx in range(n_bars):
            # Update indicators directly with raw values
            specs = self.config.get_indicator_specs()

            for name, indicator in self.indicators.items():
                spec = specs.get(name)
                # Maintain parity with update_from_bar: skip unconfigured or
                # unavailable indicators (None indicates numpy fallback path).
                if spec is None or indicator is None:
                    continue

                # Handle different input types
                if spec.get("input") == "volume":
                    indicator.update_raw(float(volumes[idx]))
                elif spec["type"] == "ATR":
                    # ATR has update_raw(high, low, close)
                    indicator.update_raw(
                        float(high_prices[idx]),
                        float(low_prices[idx]),
                        float(close_prices[idx]),
                    )
                elif spec["type"] == "BB":
                    # Bollinger Bands has update_raw(high, low, close)
                    indicator.update_raw(
                        float(high_prices[idx]),
                        float(low_prices[idx]),
                        float(close_prices[idx]),
                    )
                elif spec["type"] == "MACD":
                    # MACD needs full bar data, but we're in batch mode so create a minimal bar-like structure
                    # For consistency with update_from_bar which uses handle_bar()
                    # We can't create full Bar objects here, but MACD only needs close price anyway
                    indicator.update_raw(float(close_prices[idx]))
                else:
                    # Default to close price (for SMA, EMA, RSI, etc.)
                    indicator.update_raw(float(close_prices[idx]))

            # Get indicator values for this timestamp (don't normalize here - normalization happens in feature calculation)
            values = self.get_values()
            all_values.append(values)

        return all_values

    def _update_batch_polars(
        self,
        open_prices: npt.NDArray[np.float64],
        high_prices: npt.NDArray[np.float64],
        low_prices: npt.NDArray[np.float64],
        close_prices: npt.NDArray[np.float64],
        volumes: npt.NDArray[np.float64],
    ) -> list[dict[str, float]]:
        """Internal Polars implementation for batch updates."""
        if pl is None:
            raise RuntimeError("Polars is not available")
        df = pl.DataFrame(
            {
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": volumes,
            }
        )

        specs = self.config.get_indicator_specs()
        exprs = []

        for name, spec in specs.items():
            if spec["type"] == "SMA":
                input_col = spec.get("input", "close")
                exprs.append(
                    pl.col(input_col)
                    .rolling_mean(spec["period"])
                    .fill_null(0.0)
                    .alias(name)
                )
            elif spec["type"] == "EMA":
                # Standard EMA
                exprs.append(
                    pl.col("close")
                    .ewm_mean(span=spec["period"], adjust=False)
                    .fill_null(0.0)
                    .alias(name)
                )
            elif spec["type"] == "RSI":
                # RSI with Wilder's smoothing
                period = spec["period"]
                delta = pl.col("close").diff()
                up = delta.clip(lower_bound=0.0)
                down = -delta.clip(upper_bound=0.0)
                
                # Wilder's smoothing: alpha = 1/period
                alpha = 1.0 / period
                avg_up = up.ewm_mean(alpha=alpha, adjust=False, min_periods=period)
                avg_down = down.ewm_mean(alpha=alpha, adjust=False, min_periods=period)
                
                rs = avg_up / avg_down
                rsi = 100.0 - (100.0 / (1.0 + rs))
                
                # Normalize to [-1, 1] for ML: (RSI/100 - 0.5) * 2
                rsi_norm = ((rsi / 100.0) - 0.5) * 2.0
                exprs.append(rsi_norm.fill_null(0.0).alias(name))

            elif spec["type"] == "BB":
                period = spec["period"]
                std_dev = spec["std"]
                mid = pl.col("close").rolling_mean(period)
                std = pl.col("close").rolling_std(period)
                upper = mid + (std * std_dev)
                lower = mid - (std * std_dev)
                
                exprs.append(upper.fill_null(0.0).alias(IndicatorNames.BB_UPPER))
                exprs.append(mid.fill_null(0.0).alias(IndicatorNames.BB_MIDDLE))
                exprs.append(lower.fill_null(0.0).alias(IndicatorNames.BB_LOWER))

            elif spec["type"] == "ATR":
                # ATR (Wilder's smoothing or SMA? Fallback used SMA)
                # Using SMA to match fallback behavior for now
                period = spec["period"]
                tr1 = pl.col("high") - pl.col("low")
                tr2 = (pl.col("high") - pl.col("close").shift(1)).abs()
                tr3 = (pl.col("low") - pl.col("close").shift(1)).abs()
                # Max of TRs
                tr = pl.max_horizontal(tr1, tr2, tr3)
                
                # Using SMA for ATR as per fallback implementation
                atr = tr.rolling_mean(period)
                exprs.append(atr.fill_null(0.0).alias(name))

            elif spec["type"] == "MACD":
                fast = spec["fast"]
                slow = spec["slow"]
                # Signal is not used in output currently (0.0)
                
                fast_ema = pl.col("close").ewm_mean(span=fast, adjust=False)
                slow_ema = pl.col("close").ewm_mean(span=slow, adjust=False)
                macd_line = fast_ema - slow_ema
                
                # Normalize by close price
                macd_norm = macd_line / pl.col("close")
                
                exprs.append(macd_norm.fill_null(0.0).alias(IndicatorNames.MACD_LINE))
                exprs.append(pl.lit(0.0).alias(IndicatorNames.MACD_SIGNAL))
                exprs.append(pl.lit(0.0).alias(IndicatorNames.MACD_DIFF))

        # Collect and convert to dicts
        # Note: Some values might be null (NaN) initially due to lookback
        # fill_null(0.0) was applied above
        return cast(list[dict[str, float]], df.lazy().with_columns(exprs).collect().to_dicts())

    def get_values(self, current_price: float | None = None) -> dict[str, float]:
        """
        Get current values from all indicators.

        Parameters
        ----------
        current_price : float, optional
            Current price for normalization.

        Returns
        -------
        dict[str, float]
            Dictionary of indicator names to their current values.

        """
        values = {}

        for name, indicator in self.indicators.items():
            if indicator is not None and indicator.initialized:
                if name == "bb":
                    # Bollinger Bands has multiple outputs
                    values[IndicatorNames.BB_UPPER] = indicator.upper
                    values[IndicatorNames.BB_MIDDLE] = indicator.middle
                    values[IndicatorNames.BB_LOWER] = indicator.lower
                elif name == "macd":
                    # MACD in Nautilus only provides the MACD line (difference between EMAs)
                    # Normalize by price if provided to match batch processing
                    macd_value = indicator.value
                    if current_price and current_price > 0:
                        macd_value = safe_divide(macd_value, current_price)
                    values[IndicatorNames.MACD_LINE] = macd_value
                    # For now, set signal and diff to 0 as Nautilus MACD doesn't compute them
                    values[IndicatorNames.MACD_SIGNAL] = 0.0
                    values[IndicatorNames.MACD_DIFF] = 0.0
                else:
                    # Apply same normalization as batch processing
                    if name == "rsi":
                        # Nautilus RSI returns values in [0, 1] range, not [0, 100]
                        # Normalize to [-1, 1] for ML: (RSI - 0.5) * 2
                        raw_rsi = indicator.value
                        if not 0.0 <= raw_rsi <= 1.0:
                            logger.warning(
                                "RSI value out of expected [0, 1] range; clamping for normalization",
                                extra={
                                    "indicator": "rsi",
                                    "raw_value": float(raw_rsi),
                                    "component": "IndicatorManager.get_current_values",
                                },
                            )
                            raw_rsi = max(0.0, min(1.0, float(raw_rsi)))
                        normalized_rsi = (raw_rsi - 0.5) * 2.0
                        if not -1.0 <= normalized_rsi <= 1.0:
                            logger.warning(
                                "Normalized RSI value out of expected [-1, 1] range; clamping result",
                                extra={
                                    "indicator": "rsi",
                                    "normalized_value": float(normalized_rsi),
                                    "component": "IndicatorManager.get_current_values",
                                },
                            )
                            normalized_rsi = max(-1.0, min(1.0, float(normalized_rsi)))
                        values[name] = normalized_rsi
                    else:
                        values[name] = indicator.value
            else:
                # Not initialized yet
                if name == "bb":
                    values[IndicatorNames.BB_UPPER] = 0.0
                    values[IndicatorNames.BB_MIDDLE] = 0.0
                    values[IndicatorNames.BB_LOWER] = 0.0
                elif name == "macd":
                    values[IndicatorNames.MACD_LINE] = 0.0
                    values[IndicatorNames.MACD_SIGNAL] = 0.0
                    values[IndicatorNames.MACD_DIFF] = 0.0
                else:
                    values[name] = 0.0

        return values

    def all_initialized(self) -> bool:
        """
        Check if all indicators are initialized.

        Returns
        -------
        bool
            True if all indicators are initialized, False otherwise.

        """
        # Treat missing indicators (fallback None) as initialized to avoid blocking.
        # Actual update loops guard against None entries.
        return all(
            (ind is None) or getattr(ind, "initialized", False) for ind in self.indicators.values()
        )

    def reset(self) -> None:
        """
        Reset all indicators and clear history.
        """
        for indicator in self.indicators.values():
            if indicator is not None:
                indicator.reset()

        # Clear price history
        for key in self.price_history:
            self.price_history[key].clear()
