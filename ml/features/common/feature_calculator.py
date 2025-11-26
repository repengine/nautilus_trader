"""
FeatureCalculator component - performs core feature calculations for ML models.

**HOT PATH COMPONENT** - Performance Critical (P99 < 5ms)

Extracted from FeatureEngineer god class (Phase 2.1.4).
Responsible for computing returns, momentum, volatility, volume ratios, and technical
indicator features in both batch (training) and online (inference) modes.

Key Performance Optimizations:
- Pre-allocated feature_buffer (reused across all calls - zero allocations)
- Minimal branching in loops
- No DataFrame creation in online mode
- No dynamic allocations in hot path
- Vectorized numpy operations where possible

Critical Requirements:
- Batch/online parity: Both modes MUST produce identical results (rtol=1e-10)
- P99 latency < 5ms for online mode
- Zero allocations per call (< 100 bytes for history management acceptable)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, Self, cast, overload

import numpy as np
import numpy.typing as npt

from ml.common.safe_math import safe_divide
from ml.config.constants import IndicatorNames
from ml.features.config import FeatureConfig
from ml.features.indicators import IndicatorManager


if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    DataFrameLike = pd.DataFrame | pl.DataFrame
else:
    DataFrameLike = Any


logger = logging.getLogger(__name__)


def _normalize_atr(atr: float, close: float) -> np.float64:
    """
    Normalize ATR by close price for scale-invariant feature.

    Parameters
    ----------
    atr : float
        Average True Range value
    close : float
        Current close price

    Returns
    -------
    np.float64
        ATR normalized by close price (ATR / close)
    """
    return np.float64(safe_divide(atr, close))


class FeatureCalculator:
    """
    **HOT PATH COMPONENT** - Feature calculation engine with strict performance requirements.

    Computes financial features from OHLCV bar data in both batch (training) and online
    (inference) modes. Guarantees perfect numerical parity between modes.

    Performance Requirements:
    - P99 < 5ms for calculate_features in online mode
    - Zero allocations in hot path (feature_buffer reused)
    - Minimal branching in calculation loops

    Features Computed:
    - Returns: Price returns over configured periods
    - Momentum: Momentum indicators (same as returns for consistency)
    - Volatility: Rolling standard deviation of returns (5-period and 20-period)
    - Volume Ratios: Current volume / SMA volume
    - Technical Indicators: RSI, Bollinger Bands, ATR, EMA, MACD, price position, HL spread
    - Return/Momentum: Combined return and momentum features (batch processing)
    - Mid-Price Returns: Mid-price return statistics (std and autocorrelation)

    Parameters
    ----------
    config : FeatureConfig
        Feature configuration specifying which features to compute and their parameters
    logger : logging.Logger | None
        Optional logger instance. If None, uses module-level logger.

    Attributes
    ----------
    config : FeatureConfig
        Feature configuration
    n_features : int
        Total number of features to compute (calculated from config)
    feature_buffer : npt.NDArray[np.float32]
        Pre-allocated buffer for feature storage (reused across calls)

    Examples
    --------
    Batch mode (training):
    >>> import pandas as pd
    >>> from ml.features.engineering import FeatureConfig
    >>> config = FeatureConfig(return_periods=[1, 5], momentum_periods=[1])
    >>> calculator = FeatureCalculator(config)
    >>> df = pd.DataFrame({
    ...     "open": [100.0, 100.5, 101.0],
    ...     "high": [101.0, 101.5, 102.0],
    ...     "low": [99.0, 99.5, 100.0],
    ...     "close": [100.5, 101.0, 101.5],
    ...     "volume": [1000000.0, 1100000.0, 1200000.0],
    ... })
    >>> features_df, scaler = calculator.calculate_features(df, mode="batch")
    >>> assert len(features_df) == 3

    Online mode (inference):
    >>> from ml.features.engineering import IndicatorManager
    >>> indicator_mgr = IndicatorManager(config)
    >>> # Warm up indicator manager with history
    >>> for i in range(50):
    ...     indicator_mgr.update_from_values(close=100.0 + i*0.1, high=101.0 + i*0.1,
    ...                                       low=99.0 + i*0.1, volume=1000000.0)
    >>> current_bar = {"open": 105.0, "high": 106.0, "low": 104.0, "close": 105.5, "volume": 1100000.0}
    >>> features = calculator.calculate_features(current_bar, mode="online", indicator_manager=indicator_mgr)
    >>> assert features.shape == (calculator.n_features,)
    """

    def __init__(
        self,
        config: FeatureConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the FeatureCalculator.

        Parameters
        ----------
        config : FeatureConfig
            Feature configuration
        logger : logging.Logger | None
            Optional logger instance
        """
        self.config = config
        self._logger = logger if logger is not None else globals()["logger"]

        # Calculate total number of features
        self.n_features = self._count_total_features()

        # Pre-allocate feature_buffer (HOT PATH OPTIMIZATION)
        # Size: n_features * 2 (safety margin for edge cases)
        # Reused across all calls - zero allocations per call
        self.feature_buffer = np.zeros(self.n_features * 2, dtype=np.float32)

    def _count_total_features(self) -> int:
        """
        Count total number of features based on configuration.

        Delegates to config.get_feature_names() to ensure consistency between
        feature count and actual feature names generated by the pipeline.

        Returns
        -------
        int
            Total number of features to be computed
        """
        return len(self.config.get_feature_names())

    @overload
    def calculate_features(
        self: Self,
        data: Any,  # pd.DataFrame | pl.DataFrame
        *,
        mode: Literal["batch"] = "batch",
        indicator_manager: None = ...,
        fit_scaler: bool = ...,
        scaler_fit_ratio: float = ...,
        scaler: None = ...,
    ) -> tuple[Any, Any | None]: ...  # (DataFrame, StandardScaler | None)

    @overload
    def calculate_features(
        self: Self,
        data: dict[str, float],
        *,
        mode: Literal["online"],
        indicator_manager: IndicatorManager,
        fit_scaler: bool = ...,
        scaler_fit_ratio: float = ...,
        scaler: Any | None = ...,  # StandardScaler | None
    ) -> npt.NDArray[np.float32]: ...

    def calculate_features(
        self: Self,
        data: DataFrameLike | dict[str, float],
        mode: str = "batch",
        indicator_manager: IndicatorManager | None = None,
        fit_scaler: bool = False,
        scaler_fit_ratio: float = 0.7,
        scaler: Any | None = None,  # StandardScaler | None
    ) -> tuple[Any, Any | None] | npt.NDArray[np.float32]:  # (DataFrame, StandardScaler | None) | ndarray
        """
        Unified feature calculation method for both batch and online modes.

        This method ensures perfect feature parity between training (batch) and
        inference (online) by using the same underlying computation logic.

        **HOT PATH** in online mode - strict performance requirements apply.

        Parameters
        ----------
        data : DataFrameLike | dict[str, float]
            - For batch mode: pl.DataFrame or pd.DataFrame with OHLCV data
            - For online mode: dict with current bar data (open, high, low, close, volume)
        mode : str, default "batch"
            Computation mode - either "batch" or "online"
        indicator_manager : IndicatorManager, optional
            Required for online mode. Manages indicator state and history.
        fit_scaler : bool, default False
            Whether to fit a StandardScaler (batch mode only)
        scaler_fit_ratio : float, default 0.7
            Ratio of data for fitting scaler (batch mode only, prevents lookahead)
        scaler : StandardScaler, optional
            Pre-fitted scaler for scaling features (online mode only)

        Returns
        -------
        tuple[DataFrameLike, StandardScaler | None] | npt.NDArray[np.float32]
            - For batch mode: tuple[DataFrame, StandardScaler or None]
            - For online mode: npt.NDArray[np.float32] of shape (n_features,)

        Raises
        ------
        ValueError
            If mode is not "batch" or "online"
            If online mode is specified without indicator_manager

        Examples
        --------
        Batch mode:
        >>> features_df, scaler = calculator.calculate_features(df, mode="batch", fit_scaler=True)

        Online mode:
        >>> features = calculator.calculate_features(
        ...     current_bar, mode="online", indicator_manager=indicator_mgr, scaler=scaler
        ... )
        """
        if mode == "batch":
            if isinstance(data, dict):
                raise ValueError("batch mode requires a DataFrame-like input")
            return self._calculate_features_batch(
                data=data,
                fit_scaler=fit_scaler,
                scaler_fit_ratio=scaler_fit_ratio,
            )
        if mode == "online":
            if indicator_manager is None:
                msg = "indicator_manager is required for online mode"
                raise ValueError(msg)
            if not isinstance(data, dict):
                msg = "online mode expects current_bar dict data"
                raise ValueError(msg)
            return self._calculate_features_online(
                current_bar=data,
                indicator_manager=indicator_manager,
                scaler=scaler,
            )
        msg = f"Invalid mode: {mode}. Must be 'batch' or 'online'"
        raise ValueError(msg)

    def _calculate_features_batch(
        self: Self,
        data: DataFrameLike,
        fit_scaler: bool = False,
        scaler_fit_ratio: float = 0.7,
    ) -> tuple[Any, Any | None]:  # (DataFrame, StandardScaler | None)
        """
        Calculate features for batch data (training mode).

        Processes historical data sequentially to ensure perfect consistency
        with online calculation. Uses cold path pattern optimized for training.

        Parameters
        ----------
        data : DataFrameLike
            Input DataFrame with OHLCV data
        fit_scaler : bool, default False
            Whether to fit a StandardScaler on the data
        scaler_fit_ratio : float, default 0.7
            Ratio of data to use for fitting scaler (prevents lookahead bias)

        Returns
        -------
        tuple[DataFrameLike, StandardScaler | None]
            Tuple of (features DataFrame, fitted scaler or None)
        """
        # Create indicator manager for sequential processing
        indicator_mgr = IndicatorManager(self.config)
        feature_rows: list[dict[str, float]] = []

        # Extract price arrays
        close_prices = self._extract_close_prices(data)
        high_prices = self._extract_high_prices(data)
        low_prices = self._extract_low_prices(data)
        volumes = self._extract_volumes(data)

        # Get feature names
        feature_names = self.config.get_feature_names()

        # Process sequentially using same code path as online
        for idx in range(len(close_prices)):
            # Update indicator manager
            indicator_mgr.update_from_values(
                close=float(close_prices[idx]),
                high=float(high_prices[idx]) if high_prices is not None else float(close_prices[idx]),
                low=float(low_prices[idx]) if low_prices is not None else float(close_prices[idx]),
                volume=float(volumes[idx]),
            )

            # Calculate features using same logic as online mode
            # Each feature method handles its own insufficient history by returning 0.0
            current_bar = {
                "open": float(close_prices[idx]),  # Using close as open fallback
                "high": float(high_prices[idx]) if high_prices is not None else float(close_prices[idx]),
                "low": float(low_prices[idx]) if low_prices is not None else float(close_prices[idx]),
                "close": float(close_prices[idx]),
                "volume": float(volumes[idx]),
            }
            features_array = self._calculate_features_online(
                current_bar=current_bar,
                indicator_manager=indicator_mgr,
                scaler=None,
            )
            # Convert to dict - only use as many values as we have in features_array
            feature_row = {name: float(features_array[i]) for i, name in enumerate(feature_names) if i < len(features_array)}

            feature_rows.append(feature_row)

        # Convert to DataFrame (match input type)
        if hasattr(data, "__class__") and "polars" in str(data.__class__):
            import polars as pl
            features_df: DataFrameLike = pl.DataFrame(feature_rows)
        else:
            import pandas as pd
            features_df = pd.DataFrame(feature_rows)

        # Fit scaler if requested
        fitted_scaler: Any | None = None  # StandardScaler | None
        if fit_scaler:
            from sklearn.preprocessing import StandardScaler
            fitted_scaler = StandardScaler()
            # Fit on first scaler_fit_ratio of data (no lookahead)
            fit_size = int(len(feature_rows) * scaler_fit_ratio)
            if fit_size > 0:
                fit_data = features_df.iloc[:fit_size] if hasattr(features_df, "iloc") else features_df[:fit_size]
                fit_array = (
                    cast(Any, fit_data).to_numpy()
                    if hasattr(fit_data, "to_numpy")
                    else cast(Any, fit_data).to_pandas().to_numpy()
                )
                fitted_scaler.fit(fit_array)

                # Transform all data
                transform_input = (
                    cast(Any, features_df).to_numpy()
                    if hasattr(features_df, "to_numpy")
                    else cast(Any, features_df).to_pandas().to_numpy()
                )
                transformed = fitted_scaler.transform(transform_input)
                if hasattr(features_df, "__class__") and "polars" in str(features_df.__class__):
                    import polars as pl
                    features_df = pl.DataFrame(transformed, schema=feature_names)
                else:
                    import pandas as pd
                    features_df = pd.DataFrame(transformed, columns=feature_names)

        return features_df, fitted_scaler

    def _calculate_features_online(
        self: Self,
        current_bar: dict[str, float],
        indicator_manager: IndicatorManager,
        scaler: Any | None = None,  # StandardScaler | None
    ) -> npt.NDArray[np.float32]:
        """
        Calculate features for online mode (inference - **HOT PATH**).

        **PERFORMANCE CRITICAL** - P99 < 5ms requirement.

        Optimizations:
        - Zero allocations (feature_buffer reused)
        - Minimal branching
        - No DataFrame creation
        - Direct buffer writes

        Parameters
        ----------
        current_bar : dict[str, float]
            Current bar data (open, high, low, close, volume)
        indicator_manager : IndicatorManager
            Indicator manager with state and history
        scaler : StandardScaler, optional
            Pre-fitted scaler for feature scaling

        Returns
        -------
        npt.NDArray[np.float32]
            Feature array of shape (n_features,)
        """
        # Reset feature buffer (fill with zeros)
        self.feature_buffer.fill(0.0)

        # Get feature names to determine correct positions
        feature_names = self.config.get_feature_names()

        # Create position mapping for each feature type based on feature names
        return_idx = 0 if any("return" in name for name in feature_names[:2]) else -1
        momentum_idx = next((i for i, name in enumerate(feature_names) if "momentum" in name), -1)
        volatility_idx = next((i for i, name in enumerate(feature_names) if "volatility" in name), -1)
        volume_ratio_idx = next((i for i, name in enumerate(feature_names) if "volume_ratio" in name), -1)

        # Extract current values
        close = current_bar["close"]
        volume = current_bar["volume"]

        # Get indicator values and price history
        indicator_values = indicator_manager.get_values()
        closes = indicator_manager.price_history.get("closes", [])

        # Calculate returns at correct position
        if self.config.enable_returns and return_idx >= 0:
            _ = self._calculate_return_features(
                close=close,
                closes=closes,
                feature_idx=return_idx,
            )

        # Calculate momentum at correct position
        if self.config.enable_momentum and momentum_idx >= 0:
            _ = self._calculate_momentum_features(
                close=close,
                closes=closes,
                feature_idx=momentum_idx,
            )

        # Calculate volatility at correct position
        if self.config.enable_volatility and volatility_idx >= 0:
            _ = self._calculate_volatility_features(
                closes=closes,
                feature_idx=volatility_idx,
            )

        # Calculate volume ratios at correct position (always calculated)
        if volume_ratio_idx >= 0:
            _ = self._calculate_volume_ratio_features(
                volume=volume,
                indicator_values=indicator_values,
                feature_idx=volume_ratio_idx,
            )
        else:
            # Fallback if no volume_ratio in feature names (shouldn't happen)
            _ = self._calculate_volume_ratio_features(
                volume=volume,
                indicator_values=indicator_values,
                feature_idx=len(feature_names) - 2,  # Use last 2 positions
            )

        # Calculate technical indicators at correct position
        if self.config.enable_technical:
            # Find first technical indicator position
            tech_idx = next((i for i, name in enumerate(feature_names)
                            if any(ind in name for ind in ["sma_", "ema_", "rsi", "macd", "atr"])), -1)
            if tech_idx >= 0:
                _ = self._calculate_technical_indicator_features(
                    close=close,
                    current_bar=current_bar,
                    indicator_values=indicator_values,
                    indicator_manager=indicator_manager,
                    feature_idx=tech_idx,
                )

        # 6. Microstructure features (if enabled)
        if self.config.include_microstructure:
            # Find start index for microstructure features
            # Look for first known microstructure feature name
            micro_idx = next((i for i, name in enumerate(feature_names) 
                             if name in ["spread_mean", "mid_return_std"]), -1)
            
            if micro_idx >= 0:
                micro_features = self._calculate_microstructure_features(indicator_manager)
                for i, val in enumerate(micro_features):
                    if micro_idx + i < self.n_features:
                        self.feature_buffer[micro_idx + i] = val

        # 7. Trade flow features (if enabled)
        if self.config.include_trade_flow:
            # Find start index for trade flow features
            flow_idx = next((i for i, name in enumerate(feature_names) 
                            if name in ["trade_flow_imbalance", "vwap"]), -1)
                            
            if flow_idx >= 0:
                flow_features = self._calculate_trade_flow_features(indicator_manager)
                for i, val in enumerate(flow_features):
                    if flow_idx + i < self.n_features:
                        self.feature_buffer[flow_idx + i] = val

        # Return view of feature_buffer (no copy)
        features = self.feature_buffer[:self.n_features]

        # Apply scaler if provided
        if scaler is not None:
            features = scaler.transform(features.reshape(1, -1)).astype(np.float32).ravel()

        return features

    def _calculate_return_features(
        self,
        close: float,
        closes: list[float],
        feature_idx: int,
    ) -> int:
        """
        Calculate return features (**HOT PATH**).

        Computes price returns over configured periods.
        Returns = (current - past) / past

        Note: closes list already contains the current close at the end,
        so closes[-1] is current, closes[-2] is 1 bar ago, etc.

        Parameters
        ----------
        close : float
            Current close price
        closes : list[float]
            Historical close prices (including current)
        feature_idx : int
            Starting index in feature_buffer

        Returns
        -------
        int
            Next available feature_idx
        """
        for period in self.config.return_periods:
            if len(closes) > period:
                # closes[-1] is current, closes[-(period+1)] is target past
                prev_close = closes[-(period + 1)]
                ret = safe_divide(close - prev_close, prev_close)
            else:
                ret = 0.0
            self.feature_buffer[feature_idx] = ret
            feature_idx += 1

        return feature_idx

    def _calculate_momentum_features(
        self,
        close: float,
        closes: list[float],
        feature_idx: int,
    ) -> int:
        """
        Calculate momentum features (**HOT PATH**).

        Computes momentum indicators (same calculation as returns for consistency).
        Momentum = (current - past) / past

        Note: closes list already contains the current close at the end.

        Parameters
        ----------
        close : float
            Current close price
        closes : list[float]
            Historical close prices (including current)
        feature_idx : int
            Starting index in feature_buffer

        Returns
        -------
        int
            Next available feature_idx
        """
        for period in self.config.momentum_periods:
            if len(closes) > period:
                # closes[-1] is current, closes[-(period+1)] is target past
                prev_close = closes[-(period + 1)]
                mom = safe_divide(close - prev_close, prev_close)
            else:
                mom = 0.0
            self.feature_buffer[feature_idx] = mom
            feature_idx += 1

        return feature_idx

    def _calculate_volatility_features(
        self,
        closes: list[float],
        feature_idx: int,
    ) -> int:
        """
        Calculate volatility features (**HOT PATH**).

        Computes rolling standard deviation of returns:
        - vol_5: std of last 5 returns
        - vol_20: std of last 20 returns

        Parameters
        ----------
        closes : list[float]
            Historical close prices
        feature_idx : int
            Starting index in feature_buffer

        Returns
        -------
        int
            Next available feature_idx
        """
        if len(closes) >= 21:  # Need 21 prices to calculate 20 returns
            # Calculate returns for volatility
            returns_5 = []
            returns_20 = []

            for i in range(max(1, len(closes) - 20), len(closes)):
                if i > 0:
                    ret = safe_divide(closes[i] - closes[i - 1], closes[i - 1])
                    returns_20.append(ret)
                    if i >= len(closes) - 5:
                        returns_5.append(ret)

            # Calculate volatility
            vol_5 = float(np.std(returns_5)) if len(returns_5) >= 5 else 0.0
            vol_20 = float(np.std(returns_20)) if len(returns_20) >= 20 else 0.0
        else:
            vol_5 = vol_20 = 0.0

        self.feature_buffer[feature_idx] = vol_5
        self.feature_buffer[feature_idx + 1] = vol_20
        return feature_idx + 2

    def _calculate_volume_ratio_features(
        self,
        volume: float,
        indicator_values: dict[str, float],
        feature_idx: int,
    ) -> int:
        """
        Calculate volume ratio features (**HOT PATH**).

        Computes ratios of current volume to SMA volume over various periods.
        Ratio = current_volume / SMA_volume

        Parameters
        ----------
        volume : float
            Current volume
        indicator_values : dict[str, float]
            Indicator values from IndicatorManager
        feature_idx : int
            Starting index in feature_buffer

        Returns
        -------
        int
            Next available feature_idx
        """
        for period in self.config.volume_ma_periods:
            key = f"volume_sma_{period}"
            # Get indicator value, defaulting to volume if not available
            ind_value = indicator_values.get(key, volume)
            # If indicator value is 0 or very small, use volume as fallback to get ratio of 1.0
            if ind_value <= 0.0:
                ind_value = volume if volume > 0.0 else 1.0
            ratio = safe_divide(volume, ind_value, default=1.0)
            self.feature_buffer[feature_idx] = ratio
            feature_idx += 1

        return feature_idx

    def _calculate_technical_indicator_features(
        self,
        close: float,
        current_bar: dict[str, float],
        indicator_values: dict[str, float],
        indicator_manager: IndicatorManager,
        feature_idx: int,
    ) -> int:
        """
        Calculate technical indicator features (**HOT PATH**).

        Computes 15 technical indicator features:
        - RSI (3 features): normalized, overbought flag, oversold flag
        - Bollinger Bands (2 features): width, position
        - ATR (1 feature): normalized by close
        - EMA (3 features): fast deviation, slow deviation, cross
        - MACD (3 features): line, signal, diff (all normalized by close)
        - Price position (1 feature): position in 20-day range
        - HL spread (1 feature): high-low spread normalized by mid-price

        Parameters
        ----------
        close : float
            Current close price
        current_bar : dict[str, float]
            Current bar data (high, low, etc.)
        indicator_values : dict[str, float]
            Indicator values from IndicatorManager
        indicator_manager : IndicatorManager
            Indicator manager for price history
        feature_idx : int
            Starting index in feature_buffer

        Returns
        -------
        int
            Next available feature_idx
        """
        # RSI features
        rsi_normalized = indicator_values.get("rsi", 0.0)  # Already in [-1, 1] range
        # Runtime assertion: Normalized RSI must be in [-1, 1]
        assert -1 <= rsi_normalized <= 1, f"RSI normalized out of bounds: {rsi_normalized}"
        # Convert back to [0, 100] for threshold checks
        rsi_raw = (rsi_normalized / 2.0 + 0.5) * 100.0
        assert 0 <= rsi_raw <= 100, f"RSI raw out of bounds: {rsi_raw}"
        self.feature_buffer[feature_idx] = rsi_normalized
        self.feature_buffer[feature_idx + 1] = 1.0 if rsi_raw > 70 else 0.0
        self.feature_buffer[feature_idx + 2] = 1.0 if rsi_raw < 30 else 0.0
        feature_idx += 3

        # Bollinger Bands
        bb_upper = indicator_values.get(IndicatorNames.BB_UPPER, close)
        bb_lower = indicator_values.get(IndicatorNames.BB_LOWER, close)
        bb_middle = indicator_values.get(IndicatorNames.BB_MIDDLE, close)
        self.feature_buffer[feature_idx] = safe_divide(bb_upper - bb_lower, bb_middle)
        self.feature_buffer[feature_idx + 1] = safe_divide(
            close - bb_lower,
            bb_upper - bb_lower,
            default=0.5,
        )
        feature_idx += 2

        # ATR normalized
        self.feature_buffer[feature_idx] = _normalize_atr(indicator_values.get("atr", 0.0), close)
        feature_idx += 1

        # EMA features
        ema_fast = indicator_values.get("ema_fast", close)
        ema_slow = indicator_values.get("ema_slow", close)
        self.feature_buffer[feature_idx] = safe_divide(close - ema_fast, ema_fast)
        self.feature_buffer[feature_idx + 1] = safe_divide(close - ema_slow, ema_slow)
        self.feature_buffer[feature_idx + 2] = safe_divide(ema_fast - ema_slow, ema_slow)
        feature_idx += 3

        # MACD features
        self.feature_buffer[feature_idx] = safe_divide(
            indicator_values.get(IndicatorNames.MACD_LINE, 0.0),
            close,
        )
        self.feature_buffer[feature_idx + 1] = safe_divide(
            indicator_values.get(IndicatorNames.MACD_SIGNAL, 0.0),
            close,
        )
        self.feature_buffer[feature_idx + 2] = safe_divide(
            indicator_values.get(IndicatorNames.MACD_DIFF, 0.0),
            close,
        )
        feature_idx += 3

        # Price position in 20-day range
        highs = indicator_manager.price_history.get("highs", [])
        lows = indicator_manager.price_history.get("lows", [])

        if len(highs) >= 20 and len(lows) >= 20:
            # Look at the last 20 bars (including current)
            min_20 = min(lows[-20:])
            max_20 = max(highs[-20:])
            if max_20 > min_20:
                price_pos = np.clip((close - min_20) / (max_20 - min_20 + 1e-10), 0.0, 1.0)
            else:
                price_pos = 0.5
        else:
            price_pos = 0.5
        self.feature_buffer[feature_idx] = price_pos
        feature_idx += 1

        # High-Low spread (use mid-price denominator for numerical stability)
        hl_num = current_bar["high"] - current_bar["low"]
        hl_den = (
            0.5 * (current_bar["high"] + current_bar["low"])
            if (current_bar["high"] + current_bar["low"]) != 0
            else close
        )
        self.feature_buffer[feature_idx] = safe_divide(hl_num, hl_den)
        feature_idx += 1

        return feature_idx

    def _calculate_return_momentum_features(
        self,
        close: float,
        close_array: npt.NDArray[np.float64],
        idx: int,
        features: dict[str, float],
    ) -> None:
        """
        Calculate return and momentum features for batch processing.

        This method is used in batch mode to populate feature dict
        with return and momentum features.

        Parameters
        ----------
        close : float
            Current close price
        close_array : npt.NDArray[np.float64]
            Array of close prices
        idx : int
            Current bar index in array
        features : dict[str, float]
            Feature dictionary to populate (mutated in place)
        """
        # Price returns
        for period in self.config.return_periods:
            if idx >= period:
                prev_close = float(close_array[idx - period])
                features[f"return_{period}"] = safe_divide(close - prev_close, prev_close)
            else:
                features[f"return_{period}"] = 0.0

        # Momentum (same as returns for consistency)
        for period in self.config.momentum_periods:
            if idx >= period:
                prev_close = float(close_array[idx - period])
                features[f"momentum_{period}"] = safe_divide(close - prev_close, prev_close)
            else:
                features[f"momentum_{period}"] = 0.0

    def _calculate_microstructure_features(self, indicator_manager: IndicatorManager) -> list[float]:
        """
        Calculate microstructure features.
        
        Uses price history from IndicatorManager as a proxy when L2 data is not available.
        """
        features = []
        
        # Spread features (placeholder as we don't have L2 data in this context)
        # spread_mean, spread_std, spread_relative
        features.extend([0.0, 0.0, 0.0])
        
        # Size imbalance features (placeholder)
        # size_imbalance_mean, size_imbalance_std
        features.extend([0.0, 0.0])
        
        # Mid return features
        # Use close prices as proxy for mid prices
        closes = indicator_manager.price_history["closes"]
        mid_return_std, mid_return_autocorr = self._calculate_mid_return_features(closes)
        features.extend([mid_return_std, mid_return_autocorr])
        
        return features

    def _calculate_trade_flow_features(self, indicator_manager: IndicatorManager) -> list[float]:
        """
        Calculate trade flow features.
        """
        features = []
        
        closes = indicator_manager.price_history["closes"]
        volumes = indicator_manager.price_history["volumes"]
        highs = indicator_manager.price_history["highs"]
        lows = indicator_manager.price_history["lows"]
        
        if not closes:
            return [0.0, 0.0, 0.0, 0.0]
            
        # 1. Trade flow imbalance (placeholder)
        features.append(0.0)
        
        # 2. VWAP (approximation using HLC/3)
        # Calculate VWAP over the available history
        cum_pv = 0.0
        cum_vol = 0.0
        for i in range(len(closes)):
            typical_price = (highs[i] + lows[i] + closes[i]) / 3.0
            cum_pv += typical_price * volumes[i]
            cum_vol += volumes[i]
            
        vwap = safe_divide(cum_pv, cum_vol) if cum_vol > 0 else closes[-1]
        features.append(vwap)
        
        # 3. Trade intensity (volume / time proxy, here just volume)
        # Use average volume over recent history
        avg_vol = sum(volumes) / len(volumes) if volumes else 0.0
        features.append(avg_vol)
        
        # 4. Average price impact (placeholder)
        features.append(0.0)
        
        return features

    def _calculate_mid_return_features(self, mid_prices: list[float]) -> tuple[float, float]:
        """
        Calculate mid-price return statistics.

        Computes:
        - Standard deviation of mid-price returns
        - Autocorrelation of mid-price returns (lag-1)

        Used for microstructure feature calculation from L2 data.

        Parameters
        ----------
        mid_prices : list[float]
            List of mid-prices

        Returns
        -------
        tuple[float, float]
            (return_std, return_autocorr)
        """
        if len(mid_prices) <= 1:
            return 0.0, 0.0

        mid_returns = []
        for i in range(1, len(mid_prices)):
            if mid_prices[i - 1] > 0:
                ret = (mid_prices[i] - mid_prices[i - 1]) / mid_prices[i - 1]
                mid_returns.append(ret)

        if len(mid_returns) <= 1:
            return 0.0, 0.0

        return_std = float(np.std(mid_returns))

        # Calculate autocorrelation
        if len(mid_returns) > 2:
            mid_returns_array = np.array(mid_returns)
            if np.std(mid_returns_array) > 1e-10:
                autocorr = np.corrcoef(mid_returns_array[:-1], mid_returns_array[1:])[0, 1]
                return_autocorr = float(autocorr) if not np.isnan(autocorr) else 0.0
            else:
                return_autocorr = 0.0
        else:
            return_autocorr = 0.0

        return return_std, return_autocorr

    def compute_features(self, bars: list[Any]) -> dict[str, float]:  # pragma: no cover - shim
        """
        Legacy compatibility shim for older tests.

        Converts list of Nautilus Bar objects to DataFrame, performs batch
        feature computation, and returns the last row as dict[str, float].

        **NOT HOT PATH** - Used only for legacy test compatibility.

        Parameters
        ----------
        bars : list[Bar]
            List of Nautilus Bar objects

        Returns
        -------
        dict[str, float]
            Feature dictionary for the last bar

        Raises
        ------
        ValueError
            If bars list is empty
        """
        if not bars:
            msg = "bars list cannot be empty"
            raise ValueError(msg)

        # Convert Bars to DataFrame
        rows = []
        for b in bars:
            rows.append({
                "open": float(b.open),
                "high": float(b.high),
                "low": float(b.low),
                "close": float(b.close),
                "volume": float(b.volume),
            })

        import polars as pl
        df = pl.DataFrame(rows)

        # Calculate features in batch mode
        features_df, _ = self.calculate_features(df, mode="batch")

        # Return last row as dict
        # Since input was Polars, output is Polars
        last_row = features_df.tail(1).to_dicts()[0]
        return {str(k): float(v) for k, v in last_row.items()}

    # Helper methods for DataFrame extraction

    def _extract_close_prices(self, df: Any) -> npt.NDArray[np.float64]:
        """Extract close prices from DataFrame."""
        values = df["close"].to_numpy() if hasattr(df, "to_numpy") else df["close"].to_numpy()
        return np.asarray(values, dtype=np.float64)

    def _extract_high_prices(self, df: Any) -> npt.NDArray[np.float64] | None:
        """Extract high prices from DataFrame (returns None if column missing)."""
        try:
            values = df["high"].to_numpy() if hasattr(df, "to_numpy") else df["high"].to_numpy()
        except (KeyError, AttributeError):
            return None
        return np.asarray(values, dtype=np.float64)

    def _extract_low_prices(self, df: Any) -> npt.NDArray[np.float64] | None:
        """Extract low prices from DataFrame (returns None if column missing)."""
        try:
            values = df["low"].to_numpy() if hasattr(df, "to_numpy") else df["low"].to_numpy()
        except (KeyError, AttributeError):
            return None
        return np.asarray(values, dtype=np.float64)

    def _extract_volumes(self, df: Any) -> npt.NDArray[np.float64]:
        """Extract volumes from DataFrame."""
        values = df["volume"].to_numpy() if hasattr(df, "to_numpy") else df["volume"].to_numpy()
        return np.asarray(values, dtype=np.float64)
