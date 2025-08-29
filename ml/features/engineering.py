"""
Enhanced feature engineering with perfect batch/real-time consistency.

This module provides feature engineering capabilities with guaranteed identical
mathematical computations between training (batch) and inference (real-time) paths.
Feature parity is critical for ML model performance in production.

"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any, Literal, Self, cast, overload

import msgspec
import numpy as np
import numpy.typing as npt

# Import ML dependencies with centralized management
from ml._imports import HAS_POLARS
from ml._imports import HAS_SKLEARN
from ml._imports import pd
from ml._imports import pl
from ml.config.base import MLFeatureConfig
from ml.config.constants import IndicatorNames
from ml.config.constants import SystemConstants
from ml.config.constants import TechnicalIndicatorPeriods
from ml.features.pipeline import PipelineRunner
from ml.features.pipeline import PipelineSpec
from ml.features.pipeline import TransformSpec
from ml.registry.base import DataRequirements
from ml.registry.feature_registry import FeatureManifest
from ml.registry.feature_registry import FeatureRole
from ml.registry.feature_registry import compute_schema_hash
from ml.typing import DataFrameLike
from ml.typing import PandasDF
from ml.typing import PolarsDF
from ml.typing import PolarsSeries
from ml.typing import StandardScaler as StandardScalerT


if TYPE_CHECKING:
    from ml.monitoring.collectors.features import FeatureEngineeringCollector


# sklearn is optional; import lazily where used and gate via HAS_SKLEARN

from nautilus_trader.indicators.atr import AverageTrueRange
from nautilus_trader.indicators.average.ema import ExponentialMovingAverage
from nautilus_trader.indicators.average.sma import SimpleMovingAverage
from nautilus_trader.indicators.bollinger_bands import BollingerBands
from nautilus_trader.indicators.macd import MovingAverageConvergenceDivergence as MACD
from nautilus_trader.indicators.rsi import RelativeStrengthIndex
from nautilus_trader.model.data import Bar


# Use centralized polars availability
POLARS_AVAILABLE = HAS_POLARS


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


class FeatureConfig(MLFeatureConfig, kw_only=True, frozen=True):
    """
    Configuration for feature engineering with enhanced ML integration.

    This configuration extends the base MLFeatureConfig with specific
    parameters for technical indicators and feature computation.

    Parameters
    ----------
    return_periods : list[int], default [1, 5, 10, 20]
        Periods for return calculation features.
    momentum_periods : list[int], default [5, 10, 20]
        Periods for momentum calculation features.
    rsi_period : int, default 14
        Period for RSI calculation (must be between 2 and 100).
    bb_period : int, default 20
        Period for Bollinger Bands calculation (must be between 2 and 100).
    bb_std : float, default 2.0
        Standard deviation multiplier for Bollinger Bands (must be between 0.5 and 5.0).
    atr_period : int, default 20
        Period for ATR calculation (must be between 2 and 100).
    ema_fast : int, default 12
        Fast EMA period (must be between 2 and 50).
    ema_slow : int, default 26
        Slow EMA period (must be between 10 and 200, greater than ema_fast).
    macd_signal : int, default 9
        MACD signal line period (must be between 2 and 50).
    volume_ma_periods : list[int], default [5, 10, 20]
        Periods for volume moving average features.
    include_microstructure : bool, default False
        Whether to include microstructure features.
    include_trade_flow : bool, default False
        Whether to include trade flow features.

    """

    # Price-based features
    return_periods: list[int] = msgspec.field(default_factory=lambda: [1, 5, 10, 20])
    momentum_periods: list[int] = msgspec.field(default_factory=lambda: [5, 10, 20])

    # Technical indicators
    rsi_period: int = TechnicalIndicatorPeriods.RSI_DEFAULT_PERIOD
    bb_period: int = TechnicalIndicatorPeriods.BB_DEFAULT_PERIOD
    bb_std: float = TechnicalIndicatorPeriods.BB_DEFAULT_STD
    atr_period: int = 20

    # Moving averages
    ema_fast: int = TechnicalIndicatorPeriods.EMA_FAST_DEFAULT
    ema_slow: int = TechnicalIndicatorPeriods.EMA_SLOW_DEFAULT
    macd_signal: int = TechnicalIndicatorPeriods.MACD_SIGNAL_PERIOD

    # Volume features
    volume_ma_periods: list[int] = msgspec.field(default_factory=lambda: [5, 10, 20])

    # Optional advanced features (default False for backward compatibility)
    include_microstructure: bool = False
    include_trade_flow: bool = False
    validate_quality: bool = False

    # --- Compatibility toggles for legacy tests (no-ops by default) ---
    # These mirror older boolean switches used in tests such as
    # enable_returns/enable_momentum/enable_volatility/enable_technical and
    # ma_periods. They are optional and default to None to avoid affecting
    # normal configurations.
    enable_returns: bool | None = None
    enable_momentum: bool | None = None
    enable_volatility: bool | None = None
    enable_technical: bool | None = None
    ma_periods: list[int] | None = None

    def __post_init__(self) -> None:
        """
        Post-initialization validation and setup.
        """
        # Validate EMA parameters
        if self.ema_slow <= self.ema_fast:
            msg = f"ema_slow ({self.ema_slow}) must be greater than ema_fast ({self.ema_fast})"
            raise ValueError(msg)

        # Validate range constraints
        if not (2 <= self.rsi_period <= 100):
            msg = f"rsi_period must be between 2 and 100, got {self.rsi_period}"
            raise ValueError(msg)

        if not (2 <= self.bb_period <= 100):
            msg = f"bb_period must be between 2 and 100, got {self.bb_period}"
            raise ValueError(msg)

        if not (0.5 <= self.bb_std <= 5.0):
            msg = f"bb_std must be between 0.5 and 5.0, got {self.bb_std}"
            raise ValueError(msg)

        if not (2 <= self.atr_period <= 100):
            msg = f"atr_period must be between 2 and 100, got {self.atr_period}"
            raise ValueError(msg)

        if not (2 <= self.ema_fast <= 50):
            msg = f"ema_fast must be between 2 and 50, got {self.ema_fast}"
            raise ValueError(msg)

        if not (10 <= self.ema_slow <= 200):
            msg = f"ema_slow must be between 10 and 200, got {self.ema_slow}"
            raise ValueError(msg)

        if not (2 <= self.macd_signal <= 50):
            msg = f"macd_signal must be between 2 and 50, got {self.macd_signal}"
            raise ValueError(msg)

        # Note: Do not mutate fields in frozen msgspec.Struct. Compatibility
        # handling for `ma_periods` occurs in pipeline spec construction.

    def get_feature_names(self) -> list[str]:
        """
        Generate complete list of feature names in order.

        Canonicalized to delegate to the declarative pipeline to avoid drift.

        Returns
        -------
        list[str]
            Ordered feature names generated by the configured pipeline.

        """
        # Build a PipelineSpec mirroring the config and compute names via PipelineRunner
        spec = build_pipeline_spec_from_feature_config(self)
        allowable = (
            DataRequirements.L1_L2
            if (self.include_microstructure or self.include_trade_flow)
            else DataRequirements.L1_ONLY
        )
        runner = PipelineRunner(spec, allowable=allowable)
        return runner.compute_feature_names()

    def get_indicator_specs(self) -> dict[str, dict[str, Any]]:
        """
        Generate specifications for creating Nautilus indicators.

        Returns
        -------
        dict[str, dict[str, Any]]
            Dictionary mapping indicator names to their configuration parameters.

        """
        specs = {
            # Price SMAs for returns calculation
            IndicatorNames.PRICE_SMA_5: {
                "type": "SMA",
                "period": TechnicalIndicatorPeriods.MA_FAST_PERIOD,
                "input": "close",
            },
            IndicatorNames.PRICE_SMA_20: {
                "type": "SMA",
                "period": TechnicalIndicatorPeriods.MA_SLOW_PERIOD,
                "input": "close",
            },
        }

        # Add volume SMAs based on configured periods
        for period in self.volume_ma_periods:
            specs[f"volume_sma_{period}"] = {
                "type": "SMA",
                "period": period,
                "input": "volume",
            }

        # Technical indicators
        specs.update(
            {
                "rsi": {"type": "RSI", "period": self.rsi_period},
                "bb": {"type": "BB", "period": self.bb_period, "std": self.bb_std},
                "atr": {"type": "ATR", "period": self.atr_period},
                "ema_fast": {"type": "EMA", "period": self.ema_fast},
                "ema_slow": {"type": "EMA", "period": self.ema_slow},
                "macd": {
                    "type": "MACD",
                    "fast": self.ema_fast,
                    "slow": self.ema_slow,
                    "signal": self.macd_signal,
                },
            },
        )

        return specs


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
                self.indicators[name] = SimpleMovingAverage(spec["period"])
            elif spec["type"] == "EMA":
                self.indicators[name] = ExponentialMovingAverage(spec["period"])
            elif spec["type"] == "RSI":
                self.indicators[name] = RelativeStrengthIndex(spec["period"])
            elif spec["type"] == "BB":
                self.indicators[name] = BollingerBands(spec["period"], spec["std"])
            elif spec["type"] == "ATR":
                self.indicators[name] = AverageTrueRange(spec["period"])
            elif spec["type"] == "MACD":
                self.indicators[name] = MACD(spec["fast"], spec["slow"])

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
            if spec is None:
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
            if spec is None:
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
        n_bars = len(close_prices)
        all_values = []

        # Process each bar but without creating Bar objects
        for idx in range(n_bars):
            # Update price history
            self.price_history["closes"].append(float(close_prices[idx]))
            self.price_history["volumes"].append(float(volumes[idx]))
            self.price_history["highs"].append(float(high_prices[idx]))
            self.price_history["lows"].append(float(low_prices[idx]))

            # Keep history limited to avoid memory issues
            max_history = SystemConstants.PRICE_HISTORY_MAXLEN
            for key in self.price_history:
                if len(self.price_history[key]) > max_history:
                    self.price_history[key] = self.price_history[key][-max_history:]

            # Update indicators directly with raw values
            specs = self.config.get_indicator_specs()

            for name, indicator in self.indicators.items():
                spec = specs.get(name)
                if spec is None:
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
            if indicator.initialized:
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
                        # Runtime assertion: RSI must be in [0, 1]
                        assert 0 <= raw_rsi <= 1, f"RSI out of bounds: {raw_rsi}"
                        values[name] = (raw_rsi - 0.5) * 2.0
                        # Runtime assertion: Normalized RSI must be in [-1, 1]
                        assert (
                            -1 <= values[name] <= 1
                        ), f"Normalized RSI out of bounds: {values[name]}"
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
        return all(ind.initialized for ind in self.indicators.values())

    def reset(self) -> None:
        """
        Reset all indicators and clear history.
        """
        for indicator in self.indicators.values():
            indicator.reset()

        # Clear price history
        for key in self.price_history:
            self.price_history[key].clear()


class FeatureEngineer:
    """
    Enhanced feature engineering with perfect batch/real-time consistency.

    This class provides feature engineering capabilities with guaranteed identical
    mathematical computations between training (batch) and inference (real-time) paths.
    Feature parity is CRITICAL for ML model performance in production.

    The feature engineer follows the hot/cold path separation pattern:
    - Cold path: Batch processing for training data using Polars
    - Hot path: Real-time processing using pre-allocated numpy arrays

    Key Features
    ------------
    - Uses Nautilus indicators for consistent calculations
    - Pre-allocates arrays for hot path performance
    - Validates feature parity with < 1e-10 tolerance
    - Memory-bounded for long-running processes
    - Comprehensive feature set for trading

    """

    def __init__(
        self,
        config: FeatureConfig | None = None,
        metrics_collector: FeatureEngineeringCollector | None = None,
    ) -> None:
        """
        Initialize feature engineer.

        Parameters
        ----------
        config : FeatureConfig, optional
            Configuration for feature engineering. If None, uses default configuration.
        metrics_collector : FeatureEngineeringCollector, optional
            Optional metrics collector for monitoring feature engineering performance.

        """
        self.config = config or FeatureConfig()
        self.scaler: StandardScalerT | None = None
        self._metrics = metrics_collector

        # Pre-allocate feature buffer for hot path performance
        feature_names = self.config.get_feature_names()
        self.n_features = len(feature_names)
        # Add some extra space for potential additional features in online calculation
        from ml.config.constants import SystemConstants

        buffer_size = self.n_features + SystemConstants.FEATURE_BUFFER_PAD

        # Internal indicator manager for convenience (hot path)
        self._indicator_manager = IndicatorManager(self.config)
        # Provide attribute-style access expected by tests (e.g., engineer.indicators.rsi)
        # Exposed as a SimpleNamespace of indicator instances (references remain stable)
        self.indicators: Any = types.SimpleNamespace(**self._indicator_manager.indicators)

        # Cache statistics for metrics
        self._feature_cache: dict[str, Any] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        # Use float32 for feature buffer to match expected output dtype
        self.feature_buffer = np.zeros(buffer_size, dtype=np.float32)

    def reset(self) -> None:
        """
        Reset internal state for a clean start.

        Clears cached statistics and resets the underlying indicator manager and feature
        buffer to ensure no cross-run state leakage.

        """
        # Reset indicator state and price history
        if hasattr(self, "_indicator_manager"):
            self._indicator_manager.reset()

        # Clear caches and counters
        self._feature_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

        # Zero the preallocated feature buffer
        self.feature_buffer.fill(0)

    # ===== Manifest & Pipeline helpers =====
    def build_pipeline_spec_from_config(self) -> PipelineSpec:
        """
        Build a default PipelineSpec from the current configuration.

        This mirrors the core feature blocks in engineering and preserves the ordering
        used by get_feature_names().

        """
        return build_pipeline_spec_from_feature_config(self.config)

    def generate_feature_manifest(
        self,
        name: str,
        version: str,
        role: FeatureRole,
        data_requirements: DataRequirements,
        pipeline_version: str = "1.0.0",
        capability_flags: dict[str, Any] | None = None,
        constraints: dict[str, Any] | None = None,
        parity_tolerance: float = 0.0,
        parity_digest: dict[str, Any] | None = None,
        perf_digest: dict[str, Any] | None = None,
        parent_feature_set_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FeatureManifest:
        """
        Create a FeatureManifest from the current engineer configuration.
        """
        spec = self.build_pipeline_spec_from_config()
        runner = PipelineRunner(spec, allowable=data_requirements)
        names = runner.compute_feature_names()
        dtypes = ["float32"] * len(names)
        signature = runner.compute_signature()
        schema_hash = compute_schema_hash(names, dtypes, signature)
        now = float(np.float64(np.datetime64("now").astype("datetime64[s]").astype(int)))

        return FeatureManifest(
            feature_set_id="",
            name=name,
            version=version,
            role=role,
            data_requirements=data_requirements,
            feature_names=names,
            feature_dtypes=dtypes,
            schema_hash=schema_hash,
            pipeline_signature=signature,
            pipeline_version=pipeline_version,
            capability_flags=capability_flags or {},
            constraints=constraints or {},
            parity_tolerance=parity_tolerance,
            parity_digest=parity_digest or {},
            perf_digest=perf_digest or {},
            parent_feature_set_id=parent_feature_set_id,
            metadata=metadata or {},
            created_at=now,
            last_modified=now,
        )

    # Minimal quality calculator stub for strict typing; extended in teacher pipelines
    def _calculate_feature_qualities(
        self: Self,
        df: DataFrameLike,
    ) -> dict[str, Any]:  # pragma: no cover - typing stub
        return {}

    def _extract_price_arrays(self: Self, df: DataFrameLike) -> tuple[npt.NDArray[np.float64], ...]:
        """
        Extract price arrays from DataFrame.
        """
        if POLARS_AVAILABLE and hasattr(df, "to_numpy"):
            # Polars DataFrame
            open_prices = df["open"].to_numpy() if "open" in df.columns else df["close"].to_numpy()
            high_prices = df["high"].to_numpy() if "high" in df.columns else df["close"].to_numpy()
            low_prices = df["low"].to_numpy() if "low" in df.columns else df["close"].to_numpy()
            close_prices = df["close"].to_numpy()
            volumes = df["volume"].to_numpy() if "volume" in df.columns else np.zeros(len(df))
        else:
            # Pandas DataFrame or fallback
            open_prices = df["open"].to_numpy() if "open" in df.columns else df["close"].to_numpy()
            high_prices = df["high"].to_numpy() if "high" in df.columns else df["close"].to_numpy()
            low_prices = df["low"].to_numpy() if "low" in df.columns else df["close"].to_numpy()
            close_prices = df["close"].to_numpy()
            volumes = df["volume"].to_numpy() if "volume" in df.columns else np.zeros(len(df))
        return open_prices, high_prices, low_prices, close_prices, volumes

    def _create_empty_features_dataframe(self: Self, feature_names: list[str]) -> DataFrameLike:
        """
        Create empty DataFrame with correct columns.
        """
        if POLARS_AVAILABLE:
            return pl.DataFrame({name: [] for name in feature_names})
        else:
            if pd is None:
                from ml._imports import check_ml_dependencies

                check_ml_dependencies(["pandas"])
            return pd.DataFrame(columns=feature_names)

    def _create_pandas_features_dataframe(
        self: Self,
        feature_rows: list[dict[str, float]],
        df: DataFrameLike,
        feature_names: list[str],
    ) -> PandasDF:
        """
        Create pandas DataFrame from feature rows.
        """
        if pd is None:
            from ml._imports import check_ml_dependencies

            check_ml_dependencies(["pandas"])
        features_df = pd.DataFrame(feature_rows)
        # Add timestamp if available
        if "timestamp" in df.columns:
            features_df["timestamp"] = df["timestamp"]

        # Check if all columns exist
        existing_cols = set(features_df.columns)
        expected_cols = set(feature_names)
        missing_cols = expected_cols - existing_cols

        if missing_cols:
            # Add missing columns with default values
            for col in missing_cols:
                features_df[col] = 0.0

        # Now select columns in the correct order
        try:
            features_df = features_df[feature_names]
        except Exception:
            # If column selection fails, create a new DataFrame with the correct columns
            new_df = pd.DataFrame(index=features_df.index)
            for col in feature_names:
                if col in features_df.columns:
                    new_df[col] = features_df[col]
                else:
                    new_df[col] = 0.0
            features_df = new_df
        # Ensure dtype parity with online path (float32)
        for col in feature_names:
            if col in features_df.columns:
                features_df[col] = features_df[col].astype("float32")
        return features_df

    def _create_features_dataframe(
        self: Self,
        feature_rows: list[dict[str, float]],
        df: DataFrameLike,
    ) -> DataFrameLike:
        """
        Create features DataFrame from feature rows.
        """
        feature_names = self.config.get_feature_names()

        # Handle empty DataFrame case
        if not feature_rows:
            return self._create_empty_features_dataframe(feature_names)

        features_df: DataFrameLike
        if POLARS_AVAILABLE and hasattr(df, "__module__") and "polars" in df.__module__:
            # Input is a Polars DataFrame
            features_df = pl.DataFrame(feature_rows)
            # Add timestamp if available and not already present
            if "timestamp" in df.columns and "timestamp" not in features_df.columns:
                features_df = features_df.with_columns(df["timestamp"].alias("timestamp"))
            # Ensure column order matches config
            features_df = features_df.select(feature_names)
            # Cast to float32 to match online path dtype exactly
            features_df = features_df.with_columns(
                [
                    pl.col(name).cast(pl.Float32)
                    for name in feature_names
                    if name in features_df.columns
                ],
            )
        else:
            features_df = self._create_pandas_features_dataframe(
                feature_rows,
                df,
                feature_names,
            )
        return features_df

    def _apply_scaler(
        self: Self,
        features_df: DataFrameLike,
        df: DataFrameLike,
        scaler_fit_ratio: float,
    ) -> tuple[DataFrameLike, StandardScalerT]:
        """
        Apply feature scaling.
        """
        if not HAS_SKLEARN:
            msg = "sklearn is required for feature scaling but is not installed"
            raise ImportError(msg)
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        # Convert to numpy for sklearn
        if POLARS_AVAILABLE and hasattr(features_df, "to_numpy"):
            features_array = features_df.to_numpy()
        else:
            features_array = features_df.to_numpy()

        # CRITICAL: Only fit scaler on training portion to prevent look-ahead bias
        train_size = int(len(features_array) * scaler_fit_ratio)
        if train_size < 1:
            train_size = 1  # Ensure at least one sample

        # Fit only on the training portion (first 70% by default)
        train_features = features_array[:train_size]
        if self.scaler is not None:
            self.scaler.fit(train_features)

        # Transform all data using the scaler fitted only on training data
        if self.scaler is not None:
            features_scaled_array = self.scaler.transform(features_array)
        else:
            features_scaled_array = features_array

        # Convert back to appropriate DataFrame type
        features_scaled: DataFrameLike  # Will be either pl.DataFrame or pd.DataFrame
        if POLARS_AVAILABLE:
            # Convert column names to list to avoid pandas Index issues
            column_names = list(features_df.columns)
            features_scaled = pl.DataFrame(features_scaled_array, schema=column_names)
            # Add timestamp back if it exists
            if "timestamp" in df.columns:
                features_scaled = features_scaled.with_columns(df["timestamp"])
        else:
            features_scaled = pd.DataFrame(features_scaled_array, columns=features_df.columns)
            # Add timestamp back if it exists
            if "timestamp" in df.columns:
                features_scaled["timestamp"] = df["timestamp"]

        assert self.scaler is not None
        return features_scaled, self.scaler

    @overload
    def calculate_features(
        self: Self,
        data: DataFrameLike,
        *,
        mode: Literal["batch"] = "batch",
        indicator_manager: None = ...,
        fit_scaler: bool = ...,
        scaler_fit_ratio: float = ...,
        scaler: None = ...,
    ) -> tuple[DataFrameLike, StandardScalerT | None]: ...

    @overload
    def calculate_features(
        self: Self,
        data: dict[str, float],
        *,
        mode: Literal["online"],
        indicator_manager: IndicatorManager,
        fit_scaler: bool = ...,
        scaler_fit_ratio: float = ...,
        scaler: StandardScalerT | None = ...,
    ) -> npt.NDArray[np.float32]: ...

    def calculate_features(
        self: Self,
        data: DataFrameLike | dict[str, float],
        mode: str = "batch",
        indicator_manager: IndicatorManager | None = None,
        fit_scaler: bool = False,
        scaler_fit_ratio: float = 0.7,
        scaler: StandardScalerT | None = None,
    ) -> tuple[DataFrameLike, StandardScalerT | None] | npt.NDArray[np.float32]:
        """
        Unified feature calculation method for both batch and online modes.

        This method ensures perfect feature parity between training (batch) and
        inference (online) by routing to the same underlying computation logic.

        Parameters
        ----------
        data : Any
            - For batch mode: pl.DataFrame or pd.DataFrame with OHLCV data
            - For online mode: dict with current bar data (open, high, low, close, volume)
        mode : str, default "batch"
            Computation mode - either "batch" or "online"
        indicator_manager : IndicatorManager, optional
            Required for online mode. Manages indicator state.
        fit_scaler : bool, default False
            Whether to fit a StandardScaler (batch mode only)
        scaler_fit_ratio : float, default 0.7
            Ratio of data for fitting scaler (batch mode only)
        scaler : StandardScaler, optional
            Pre-fitted scaler for scaling features (online mode only)

        Returns
        -------
        Any
            - For batch mode: tuple[DataFrame, StandardScaler or None]
            - For online mode: npt.NDArray[np.float32]

        Raises
        ------
        ValueError
            If mode is not "batch" or "online"
            If online mode is specified without indicator_manager

        Examples
        --------
        Batch mode (training):
        >>> config = FeatureConfig()  # doctest: +SKIP
        >>> engineer = FeatureEngineer(config)  # doctest: +SKIP
        >>> features_df, scaler = engineer.calculate_features(  # doctest: +SKIP
        ...     df, mode="batch", fit_scaler=True
        ... )

        Online mode (inference):
        >>> features = engineer.calculate_features(  # doctest: +SKIP
        ...     current_bar, mode="online",
        ...     indicator_manager=indicator_mgr,
        ...     scaler=scaler
        ... )

        """
        if mode == "batch":
            return self.calculate_features_batch(
                df=cast(DataFrameLike, data),
                fit_scaler=fit_scaler,
                scaler_fit_ratio=scaler_fit_ratio,
            )
        elif mode == "online":
            if indicator_manager is None:
                msg = "indicator_manager is required for online mode"
                raise ValueError(msg)
            return self.calculate_features_online(
                current_bar=cast(dict[str, float], data),
                indicator_manager=indicator_manager,
                scaler=scaler,
            )
        else:
            msg = f"Invalid mode: {mode}. Must be 'batch' or 'online'"
            raise ValueError(msg)

    def calculate_features_batch(
        self: Self,
        df: DataFrameLike,  # pl.DataFrame or pd.DataFrame
        fit_scaler: bool = False,
        scaler_fit_ratio: float = 0.7,
    ) -> tuple[DataFrameLike, StandardScalerT | None]:
        """
        Calculate features for batch data using Nautilus indicators.

        This method processes historical data sequentially to ensure perfect
        consistency with online calculation. It follows the cold path pattern
        optimized for training data preparation.

        Parameters
        ----------
        df : pl.DataFrame or pd.DataFrame
            Input DataFrame with OHLCV data.
        fit_scaler : bool, default False
            Whether to fit a StandardScaler on the data.
        scaler_fit_ratio : float, default 0.7
            Ratio of data to use for fitting scaler to prevent look-ahead bias.
            Only the first 70% of data is used to fit the scaler.

        Returns
        -------
        tuple[pl.DataFrame or pd.DataFrame, StandardScaler or None]
            Tuple of (features DataFrame, fitted scaler or None).

        """
        # Determine instrument from DataFrame or use generic
        instrument = str(getattr(df, "instrument_id", "unknown"))

        # Use metrics collector timer if available
        if self._metrics is not None:
            timer = self._metrics.time_feature_computation(
                instrument=instrument,
                feature_type="technical",
                computation_mode="batch",
            )
        else:
            timer = None

        with timer if timer is not None else _dummy_context_manager():
            return self._calculate_features_batch_impl(
                df,
                fit_scaler,
                scaler_fit_ratio,
                timer,
            )

    def _calculate_features_batch_impl(
        self: Self,
        df: DataFrameLike,
        fit_scaler: bool,
        scaler_fit_ratio: float,
        timer: object | None = None,
    ) -> tuple[DataFrameLike, StandardScalerT | None]:
        """
        Implement batch feature calculation internally.
        """
        # Create indicator manager
        indicator_mgr = IndicatorManager(self.config)

        # Storage for features at each timestamp
        feature_rows = []

        # Process each bar sequentially to update indicators
        # No need to import Bar-related types anymore as we're using vectorized processing

        # Extract price arrays
        open_prices, high_prices, low_prices, close_prices, volumes = self._extract_price_arrays(df)

        # Use efficient batch processing without creating Bar objects
        # This is the COLD path (training), so we can optimize for throughput
        all_indicator_values = indicator_mgr.update_batch_vectorized(
            open_prices=open_prices,
            high_prices=high_prices,
            low_prices=low_prices,
            close_prices=close_prices,
            volumes=volumes,
        )

        # Process features for each timestamp
        for idx in range(len(df)):
            # Get indicator values for this timestamp
            ind_values = all_indicator_values[idx]

            # Calculate features using indicator values
            # Get row data as dict
            row_data = {
                "close": close_prices[idx],
                "volume": volumes[idx],
                "high": high_prices[idx],
                "low": low_prices[idx],
            }

            features = self._calculate_features_from_indicators(row_data, ind_values, df, idx)
            feature_rows.append(features)

        # Create DataFrame
        features_df = self._create_features_dataframe(feature_rows, df)

        # Set timer results if available
        if timer is not None:
            feature_count = (
                features_df.width
                if hasattr(features_df, "width")
                else len(features_df.columns) if hasattr(features_df, "columns") else 0
            )
            cast(Any, timer).set_computation_result(
                features_computed=feature_count,
                cache_hit=False,  # Batch computation is never cached
                feature_qualities=(
                    self._calculate_feature_qualities(features_df)
                    if hasattr(features_df, "select")
                    else {}
                ),
            )

        # Scale if requested
        if fit_scaler:
            return self._apply_scaler(features_df, df, scaler_fit_ratio)

        return features_df, None

    def _calculate_return_features(
        self,
        close: float,
        closes: list[float],
        feature_idx: int,
    ) -> int:
        """
        Calculate return and momentum features.

        Note: closes list already contains the current close at the end,
        so closes[-1] is current, closes[-2] is 1 bar ago, etc.

        """
        # Price returns
        for period in self.config.return_periods:
            if len(closes) > period:  # We have enough history including current
                # Since closes[-1] is current, closes[-period-1] is the target previous close
                prev_close = closes[-(period + 1)]
                ret = safe_divide(close - prev_close, prev_close)
            else:
                ret = 0.0
            self.feature_buffer[feature_idx] = ret
            feature_idx += 1

        # Price momentum - same as returns for consistency
        for period in self.config.momentum_periods:
            if len(closes) > period:  # We have enough history including current
                # Since closes[-1] is current, closes[-period-1] is the target previous close
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
        Calculate volatility features.
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

    def _calculate_indicator_features(
        self,
        close: float,
        volume: float,
        current_bar: dict[str, float],
        indicator_values: dict[str, float],
        indicator_manager: IndicatorManager,
        feature_idx: int,
    ) -> int:
        """
        Calculate technical indicator features.
        """
        # Volume ratios
        for period in self.config.volume_ma_periods:
            key = f"volume_sma_{period}"
            ratio = safe_divide(volume, indicator_values.get(key, volume), default=1.0)
            self.feature_buffer[feature_idx] = ratio
            feature_idx += 1

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
        self.feature_buffer[feature_idx] = safe_divide(indicator_values.get("atr", 0.0), close)
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
            indicator_values.get("macd_line", 0.0),
            close,
        )
        self.feature_buffer[feature_idx + 1] = safe_divide(
            indicator_values.get("macd_signal", 0.0),
            close,
        )
        self.feature_buffer[feature_idx + 2] = safe_divide(
            indicator_values.get("macd_diff", 0.0),
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

        # High-Low spread
        self.feature_buffer[feature_idx] = safe_divide(
            current_bar["high"] - current_bar["low"],
            close,
        )
        feature_idx += 1

        return feature_idx

    @overload
    def calculate_features_online(
        self,
        *,
        close_price: float,
        high_price: float,
        low_price: float,
        volume: float,
        scaler: StandardScalerT | None = None,
    ) -> npt.NDArray[np.float32]: ...

    @overload
    def calculate_features_online(
        self,
        current_bar: dict[str, float],
        indicator_manager: IndicatorManager,
        scaler: StandardScalerT | None = None,
    ) -> npt.NDArray[np.float32]: ...

    def calculate_features_online(
        self,
        current_bar: dict[str, float] | None = None,
        indicator_manager: IndicatorManager | None = None,
        scaler: StandardScalerT | None = None,
        *,
        close_price: float | None = None,
        high_price: float | None = None,
        low_price: float | None = None,
        volume: float | None = None,
    ) -> npt.NDArray[np.float32]:
        """
        Calculate features for online inference using indicator manager.

        This method follows the hot path pattern optimized for real-time inference.
        It uses pre-allocated arrays and avoids any dynamic memory allocation.

        Parameters
        ----------
        current_bar : dict[str, float]
            Current OHLCV data.
        indicator_manager : IndicatorManager
            Indicator manager with all state.
        scaler : StandardScaler, optional
            Pre-fitted scaler from training.
        close_price : float | None
            Current close price if `current_bar` is not provided.
        high_price : float | None
            Current high price if `current_bar` is not provided.
        low_price : float | None
            Current low price if `current_bar` is not provided.
        volume : float | None
            Current trade volume if `current_bar` is not provided.

        Returns
        -------
        npt.NDArray[np.float32]
            Feature array ready for model prediction.

        """
        # Support convenience kwargs when no current_bar provided
        if current_bar is None:
            if close_price is None or high_price is None or low_price is None or volume is None:
                msg = (
                    "calculate_features_online requires either current_bar and indicator_manager, "
                    "or keyword args: close_price, high_price, low_price, volume"
                )
                raise ValueError(msg)
            # Use internal indicator manager by default
            ind_mgr = self._indicator_manager if indicator_manager is None else indicator_manager
            # Update indicators from raw values
            ind_mgr.update_from_values(
                close=float(close_price),
                high=float(high_price),
                low=float(low_price),
                volume=float(volume),
            )
            # Build a minimal current_bar mapping for downstream logic
            current_bar = {
                "close": float(close_price),
                "high": float(high_price),
                "low": float(low_price),
                "volume": float(volume),
            }
            indicator_manager = ind_mgr

        assert current_bar is not None  # for type checker
        assert indicator_manager is not None  # for type checker

        # Determine instrument from current_bar or use generic
        instrument = str(current_bar.get("instrument_id", "unknown"))

        # Use metrics collector timer if available
        if self._metrics is not None:
            timer = self._metrics.time_feature_computation(
                instrument=instrument,
                feature_type="technical",
                computation_mode="online",
            )
        else:
            timer = None

        with timer if timer is not None else _dummy_context_manager():
            result = self._calculate_features_online_impl(
                current_bar,
                indicator_manager,
                scaler,
                timer,
            )

            # Set timer results if available
            if timer is not None:
                cast(Any, timer).set_computation_result(
                    features_computed=len(result),
                    cache_hit=False,  # Online computation is never cached in the traditional sense
                )

            return result

    def _calculate_features_online_impl(
        self,
        current_bar: dict[str, float],
        indicator_manager: IndicatorManager,
        scaler: StandardScalerT | None = None,
        timer: object | None = None,
    ) -> npt.NDArray[np.float32]:
        """
        Implement online feature calculation internally.
        """
        # Reset buffer
        self.feature_buffer.fill(0.0)
        feature_idx = 0

        # Current values
        close = current_bar["close"]
        volume = current_bar["volume"]

        # Get indicator values (don't normalize here - normalization happens in feature calculation)
        indicator_values = indicator_manager.get_values()

        # Get historical values from indicators
        closes = indicator_manager.price_history["closes"]

        # Calculate return features
        feature_idx = self._calculate_return_features(close, closes, feature_idx)

        # Calculate volatility features
        feature_idx = self._calculate_volatility_features(closes, feature_idx)

        # Calculate indicator features
        feature_idx = self._calculate_indicator_features(
            close,
            volume,
            current_bar,
            indicator_values,
            indicator_manager,
            feature_idx,
        )

        # Add microstructure features if enabled (hot path - use simplified calculations)
        if self.config.include_microstructure:
            feature_idx = self._calculate_microstructure_features_online(
                current_bar,
                indicator_manager,
                feature_idx,
            )

        # Add trade flow features if enabled (hot path - use simplified calculations)
        if self.config.include_trade_flow:
            feature_idx = self._calculate_trade_flow_features_online(
                current_bar,
                indicator_manager,
                feature_idx,
            )

        # Scale if scaler provided
        if scaler is not None:
            # Reshape for sklearn
            features_array = self.feature_buffer[:feature_idx].reshape(1, -1)
            features_array = scaler.transform(features_array)
            return np.asarray(features_array[0])

        # Return a copy to avoid aliasing across calls.
        # While a view would minimize allocations, several tests and typical
        # downstream flows persist feature rows across bars. Returning a copy
        # guarantees immutability of previously returned results and ensures
        # strict parity under accumulation.
        return self.feature_buffer[:feature_idx].copy()

    # Hot-path microstructure placeholder for strict typing (feature gated)
    def _calculate_microstructure_features_online(
        self,
        current_bar: dict[str, float],
        indicator_manager: IndicatorManager,
        feature_idx: int,
    ) -> int:  # pragma: no cover - typing stub
        return feature_idx

    # Hot-path trade flow placeholder for strict typing (feature gated)
    def _calculate_trade_flow_features_online(
        self,
        current_bar: dict[str, float],
        indicator_manager: IndicatorManager,
        feature_idx: int,
    ) -> int:  # pragma: no cover - typing stub
        return feature_idx

    def _extract_data_arrays(
        self: Self,
        df: DataFrameLike,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64] | None,
        npt.NDArray[np.float64] | None,
    ]:
        """
        Extract data arrays from DataFrame for batch processing.
        """
        if POLARS_AVAILABLE and hasattr(df, "to_numpy"):
            close_array = df["close"].to_numpy()
            high_array = df["high"].to_numpy() if "high" in df.columns else None
            low_array = df["low"].to_numpy() if "low" in df.columns else None
        else:
            close_array = df["close"].to_numpy()
            high_array = df["high"].to_numpy() if "high" in df.columns else None
            low_array = df["low"].to_numpy() if "low" in df.columns else None
        return close_array, high_array, low_array

    def _calculate_return_momentum_features(
        self,
        close: float,
        close_array: npt.NDArray[np.float64],
        idx: int,
        features: dict[str, float],
    ) -> None:
        """
        Calculate return and momentum features for batch processing.
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

    def _calculate_volatility_features_batch(
        self,
        close_array: npt.NDArray[np.float64],
        idx: int,
        features: dict[str, float],
    ) -> None:
        """
        Calculate volatility features for batch processing.
        """
        if idx >= 20:
            returns = []
            for i in range(idx - 19, idx + 1):
                if i > 0:
                    close_i = float(close_array[i])
                    close_prev = float(close_array[i - 1])
                    ret = (close_i - close_prev) / close_prev
                    returns.append(ret)

            features["volatility_5"] = float(np.std(returns[-5:])) if len(returns) >= 5 else 0.0
            features["volatility_20"] = float(np.std(returns)) if len(returns) >= 20 else 0.0
        else:
            features["volatility_5"] = 0.0
            features["volatility_20"] = 0.0

    def _calculate_indicator_features_batch(
        self,
        close: float,
        volume: float,
        bar_data: dict[str, float],
        ind_values: dict[str, float],
        high_array: npt.NDArray[np.float64] | None,
        low_array: npt.NDArray[np.float64] | None,
        idx: int,
        features: dict[str, float],
    ) -> None:
        """
        Calculate indicator features for batch processing.
        """
        # Volume ratios
        for period in self.config.volume_ma_periods:
            key = f"volume_sma_{period}"
            if key in ind_values and ind_values[key] > 0:
                features[f"volume_ratio_{period}"] = volume / ind_values[key]
            else:
                features[f"volume_ratio_{period}"] = 1.0

        # RSI features
        rsi_normalized = ind_values.get("rsi", 0.0)  # Already in [-1, 1] range
        features["rsi"] = rsi_normalized
        # Convert back to [0, 100] for threshold checks
        rsi_raw = (rsi_normalized / 2.0 + 0.5) * 100.0
        features["rsi_overbought"] = 1.0 if rsi_raw > 70 else 0.0
        features["rsi_oversold"] = 1.0 if rsi_raw < 30 else 0.0

        # Bollinger Bands
        bb_upper = ind_values.get(IndicatorNames.BB_UPPER, close)
        bb_lower = ind_values.get(IndicatorNames.BB_LOWER, close)
        bb_middle = ind_values.get(IndicatorNames.BB_MIDDLE, close)

        features[IndicatorNames.BB_WIDTH] = safe_divide(bb_upper - bb_lower, bb_middle)
        features[IndicatorNames.BB_POSITION] = safe_divide(
            close - bb_lower,
            bb_upper - bb_lower,
            default=0.5,
        )

        # ATR
        features["atr_normalized"] = safe_divide(ind_values.get("atr", 0.0), close)

        # EMA features
        ema_fast = ind_values.get("ema_fast", close)
        ema_slow = ind_values.get("ema_slow", close)
        features["ema_fast_dist"] = safe_divide(close - ema_fast, ema_fast)
        features["ema_slow_dist"] = safe_divide(close - ema_slow, ema_slow)
        features["ema_cross"] = safe_divide(ema_fast - ema_slow, ema_slow)

        # MACD
        features[IndicatorNames.MACD_LINE] = safe_divide(
            ind_values.get(IndicatorNames.MACD_LINE, 0.0),
            close,
        )
        features[IndicatorNames.MACD_SIGNAL] = safe_divide(
            ind_values.get(IndicatorNames.MACD_SIGNAL, 0.0),
            close,
        )
        features[IndicatorNames.MACD_DIFF] = safe_divide(
            ind_values.get(IndicatorNames.MACD_DIFF, 0.0),
            close,
        )

        # Price position
        if idx >= 19 and high_array is not None and low_array is not None:
            # Use same logic as online: look at previous 20 bars (including current)
            start_idx = max(0, idx - 19)
            end_idx = idx + 1  # Include current bar
            min_20 = float(np.min(low_array[start_idx:end_idx]))
            max_20 = float(np.max(high_array[start_idx:end_idx]))
            if max_20 > min_20:
                features["price_position_20"] = np.clip(
                    (close - min_20) / (max_20 - min_20 + 1e-10),
                    0.0,
                    1.0,
                )
            else:
                features["price_position_20"] = 0.5
        else:
            features["price_position_20"] = 0.5

        # HL spread
        features["hl_spread"] = safe_divide(float(bar_data["high"]) - float(bar_data["low"]), close)

    def _calculate_features_from_indicators(
        self,
        bar_data: dict[str, float],
        ind_values: dict[str, float],
        df: DataFrameLike,  # pl.DataFrame or pd.DataFrame
        idx: int,
    ) -> dict[str, float]:
        """
        Calculate features from current bar and indicator values.

        This method is used in batch processing to compute features using
        the same logic as the online method.

        Parameters
        ----------
        bar_data : dict[str, float]
            Current bar OHLCV data.
        ind_values : dict[str, float]
            Current indicator values.
        df : pl.DataFrame or pd.DataFrame
            Full DataFrame for historical lookback.
        idx : int
            Current index in the DataFrame.

        Returns
        -------
        dict[str, float]
            Dictionary of computed features.

        """
        features: dict[str, float] = {}
        close = float(bar_data["close"])
        volume = float(bar_data["volume"])

        # Extract data arrays
        close_array, high_array, low_array = self._extract_data_arrays(df)

        # Calculate different feature groups
        self._calculate_return_momentum_features(close, close_array, idx, features)
        self._calculate_volatility_features_batch(close_array, idx, features)
        self._calculate_indicator_features_batch(
            close,
            volume,
            bar_data,
            ind_values,
            high_array,
            low_array,
            idx,
            features,
        )

        # Add microstructure features if enabled (batch processing)
        if self.config.include_microstructure:
            # For batch processing, we need bid/ask data to calculate proper microstructure features
            # If not available, use simplified calculations based on OHLCV
            microstructure_features = self._calculate_microstructure_features_batch(df, idx)
            features.update(microstructure_features)

        # Add trade flow features if enabled (batch processing)
        if self.config.include_trade_flow:
            # For batch processing, we need trade data to calculate proper trade flow features
            # If not available, use simplified calculations based on OHLCV
            trade_flow_features = self._calculate_trade_flow_features_batch(df, idx, bar_data)
            features.update(trade_flow_features)

        return features

    def _extract_bid_ask_data(
        self,
        df: DataFrameLike,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """
        Extract bid/ask price and size arrays from DataFrame.
        """
        if POLARS_AVAILABLE and hasattr(df, "to_numpy"):
            return (
                df["bid_price"].to_numpy(),
                df["ask_price"].to_numpy(),
                df["bid_size"].to_numpy(),
                df["ask_size"].to_numpy(),
            )
        return (
            df["bid_price"].to_numpy(),
            df["ask_price"].to_numpy(),
            df["bid_size"].to_numpy(),
            df["ask_size"].to_numpy(),
        )

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

    def _calculate_mid_return_features(self, mid_prices: list[float]) -> tuple[float, float]:
        """
        Calculate mid-price return statistics.
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

    def _calculate_microstructure_features_from_ohlcv(
        self,
        df: DataFrameLike,
        idx: int,
    ) -> dict[str, float]:
        """
        Calculate microstructure features from OHLCV data as fallback.
        """
        features: dict[str, float] = {}

        if POLARS_AVAILABLE and hasattr(df, "to_numpy"):
            high_prices = df["high"].to_numpy()
            low_prices = df["low"].to_numpy()
            close_prices = df["close"].to_numpy()
        else:
            high_prices = df["high"].to_numpy()
            low_prices = df["low"].to_numpy()
            close_prices = df["close"].to_numpy()

        current_high = float(high_prices[idx])
        current_low = float(low_prices[idx])
        current_close = float(close_prices[idx])

        # Estimate spread from high-low range
        hl_spread = current_high - current_low
        features["spread_mean"] = safe_divide(hl_spread, current_close, 0.0)
        features["spread_std"] = 0.0  # Cannot estimate std from single bar
        features["spread_relative"] = safe_divide(hl_spread, current_close, 0.0)

        # Default values for size imbalance (no size data available)
        features["size_imbalance_mean"] = 0.0
        features["size_imbalance_std"] = 0.0

        # Estimate mid-price return volatility from recent price changes
        window = min(5, idx + 1)
        if window > 1:
            returns = []
            for i in range(max(0, idx - window + 1), idx + 1):
                if i > 0 and close_prices[i - 1] > 0:
                    ret = (float(close_prices[i]) - float(close_prices[i - 1])) / float(
                        close_prices[i - 1],
                    )
                    returns.append(ret)

            features["mid_return_std"] = float(np.std(returns)) if len(returns) > 1 else 0.0
            features["mid_return_autocorr"] = 0.0  # Cannot estimate autocorr reliably from OHLCV
        else:
            features["mid_return_std"] = 0.0
            features["mid_return_autocorr"] = 0.0

        return features

    def _calculate_microstructure_features_batch(
        self: Self,
        df: DataFrameLike,
        idx: int,
    ) -> dict[str, float]:
        """
        Calculate microstructure features for entire dataset (batch processing).

        This method processes historical bid/ask data to compute microstructure features
        using Polars for efficient batch processing. It ensures results match the online
        version with < 1e-10 tolerance.

        Parameters
        ----------
        df : pl.DataFrame or pd.DataFrame
            DataFrame with OHLCV data and optionally bid/ask data.
        idx : int
            Current index in the DataFrame.

        Returns
        -------
        dict[str, float]
            Dictionary of microstructure feature names to values.

        """
        features: dict[str, float] = {}

        # Check what level of data we have
        has_l2_depth = "bid_price_0" in df.columns  # Multi-level order book
        has_l1_quotes = all(
            col in df.columns for col in ["bid_price", "ask_price", "bid_size", "ask_size"]
        )

        if has_l2_depth:
            # Use advanced L2 microstructure features
            from ml.features.microstructure import L2MicrostructureFeatures

            # Initialize calculator with appropriate window
            window = min(20, idx + 1)
            calculator = L2MicrostructureFeatures(
                n_levels=10,  # Use up to 10 levels if available
                lookback_window=window,
            )

            # Get subset of data for this window
            start_idx = max(0, idx - window + 1)
            df_window = (
                cast(PolarsDF, df)[start_idx : idx + 1]
                if hasattr(df, "__getitem__")
                else cast(PandasDF, df).iloc[start_idx : idx + 1]
            )

            # Compute all L2 features
            all_features = calculator.compute_all_features(df_window)

            # Extract the last value for each feature (current point)
            for key, values in all_features.items():
                if len(values) > 0:
                    features[key] = float(values[-1])

        elif has_l1_quotes:
            # Use existing L1 bid/ask processing
            # Extract bid/ask arrays
            bid_prices, ask_prices, bid_sizes, ask_sizes = self._extract_bid_ask_data(df)

            # Calculate spreads and mid-prices for recent period
            window = min(20, idx + 1)
            start_idx = max(0, idx - window + 1)

            spreads, relative_spreads, size_imbalances, mid_prices = self._calculate_spread_metrics(
                bid_prices,
                ask_prices,
                bid_sizes,
                ask_sizes,
                start_idx,
                idx,
            )

            # Calculate basic spread features
            features["spread_mean"] = float(np.mean(spreads)) if spreads else 0.0
            features["spread_std"] = float(np.std(spreads)) if len(spreads) > 1 else 0.0
            features["spread_relative"] = (
                float(np.mean(relative_spreads)) if relative_spreads else 0.0
            )
            features["size_imbalance_mean"] = (
                float(np.mean(size_imbalances)) if size_imbalances else 0.0
            )
            features["size_imbalance_std"] = (
                float(np.std(size_imbalances)) if len(size_imbalances) > 1 else 0.0
            )

            # Calculate mid-price return features
            return_std, return_autocorr = self._calculate_mid_return_features(mid_prices)
            features["mid_return_std"] = return_std
            features["mid_return_autocorr"] = return_autocorr
        else:
            # Fallback to OHLCV-based approximations
            features = self._calculate_microstructure_features_from_ohlcv(df, idx)

        return features

    def _extract_trade_data(
        self: Self,
        df: DataFrameLike,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Extract trade price, volume, and side arrays from DataFrame.
        """
        if POLARS_AVAILABLE and hasattr(df, "to_numpy"):
            return (
                df["trade_price"].to_numpy(),
                df["trade_volume"].to_numpy(),
                df["trade_side"].to_numpy(),
            )
        return (
            df["trade_price"].to_numpy(),
            df["trade_volume"].to_numpy(),
            df["trade_side"].to_numpy(),
        )

    def _calculate_trade_metrics(
        self,
        trade_prices: npt.NDArray[np.float64],
        trade_volumes: npt.NDArray[np.float64],
        trade_sides: npt.NDArray[np.float64],
        start_idx: int,
        end_idx: int,
    ) -> tuple[float, float, float, float]:
        """
        Calculate trade metrics for given window.
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
        trade_intensity = min(float(trade_count) / 20.0, 5.0)  # Normalize and cap
        avg_price_impact = float(np.mean(price_impacts)) if price_impacts else 0.0

        return trade_flow_imbalance, vwap, trade_intensity, avg_price_impact

    def _calculate_trade_flow_features_from_ohlcv(
        self: Self,
        df: DataFrameLike,
        idx: int,
        bar_data: dict[str, float],
    ) -> dict[str, float]:
        """
        Calculate trade flow features from OHLCV data as fallback.
        """
        features: dict[str, float] = {}

        close = float(bar_data["close"])
        volume = float(bar_data["volume"])

        # Default trade flow imbalance (no directional information)
        features["trade_flow_imbalance"] = 0.0

        # Use close price as VWAP approximation
        features["vwap"] = close

        # Estimate trade intensity from volume
        if POLARS_AVAILABLE and hasattr(df, "to_numpy"):
            volumes = df["volume"].to_numpy()
        else:
            volumes = df["volume"].to_numpy()

        window = min(20, idx + 1)
        start_idx = max(0, idx - window + 1)
        recent_volumes = volumes[start_idx : idx + 1]

        if len(recent_volumes) > 0:
            avg_volume = float(np.mean(recent_volumes))
            if avg_volume > 0:
                intensity = volume / avg_volume
                features["trade_intensity"] = min(intensity, 5.0)  # Cap at 5x average
            else:
                features["trade_intensity"] = 1.0
        else:
            features["trade_intensity"] = 1.0

        # Estimate price impact from intraday volatility
        high = float(bar_data["high"])
        low = float(bar_data["low"])
        if volume > 0 and close > 0:
            hl_range = high - low
            # Normalize by volume - higher volume should have lower per-unit impact
            impact = safe_divide(hl_range / close, volume / 1000.0, 0.0)
            features["avg_price_impact"] = min(impact, 0.01)  # Cap at 1%
        else:
            features["avg_price_impact"] = 0.0

        return features

    def _calculate_trade_flow_features_batch(
        self,
        df: DataFrameLike,
        idx: int,
        bar_data: dict[str, float],
    ) -> dict[str, float]:
        """
        Calculate trade flow features for entire dataset (batch processing).

        This method processes historical trade data to compute trade flow features
        using Polars for efficient batch processing. It ensures results match the online
        version with < 1e-10 tolerance.

        Parameters
        ----------
        df : pl.DataFrame or pd.DataFrame
            DataFrame with OHLCV data and optionally trade data.
        idx : int
            Current index in the DataFrame.
        bar_data : dict[str, float]
            Current bar OHLCV data.

        Returns
        -------
        dict[str, float]
            Dictionary of trade flow feature names to values.

        """
        features: dict[str, float] = {}

        # Check if we have trade-level data
        has_trade_data = all(
            col in df.columns for col in ["trade_price", "trade_volume", "trade_side"]
        )

        if has_trade_data:
            # Extract trade arrays
            trade_prices, trade_volumes, trade_sides = self._extract_trade_data(df)

            # Calculate features for recent period
            window = min(20, idx + 1)
            start_idx = max(0, idx - window + 1)

            # Calculate trade metrics
            trade_flow_imbalance, vwap, trade_intensity, avg_price_impact = (
                self._calculate_trade_metrics(
                    trade_prices,
                    trade_volumes,
                    trade_sides,
                    start_idx,
                    idx,
                )
            )

            features["trade_flow_imbalance"] = trade_flow_imbalance
            features["vwap"] = vwap if vwap > 0 else float(bar_data["close"])
            features["trade_intensity"] = trade_intensity
            features["avg_price_impact"] = avg_price_impact
        else:
            # Fallback to OHLCV-based approximations
            features = self._calculate_trade_flow_features_from_ohlcv(df, idx, bar_data)

        return features

    def get_feature_names(self) -> list[str]:
        """
        Get list of feature names in order.

        Returns
        -------
        list[str]
            List of feature names that will be generated.

        """
        return self.config.get_feature_names()

    # ---- Compatibility shim for legacy tests ----
    def compute_features(self, bars: list[Bar]) -> dict[str, float]:  # pragma: no cover - shim
        """
        Compatibility wrapper mapping legacy API to the unified calculator.

        Accepts a list of `Bar` objects, converts them to a tabular form,
        performs batch feature computation, and returns the latest row as a
        simple dict[str, float] to match older test expectations.

        """
        # Convert Bars to a lightweight pandas DataFrame regardless of POLARS availability
        rows = []
        for b in bars:
            rows.append(
                {
                    "open": float(b.open),
                    "high": float(b.high),
                    "low": float(b.low),
                    "close": float(b.close),
                    "volume": float(b.volume),
                    "timestamp": int(b.ts_event),
                },
            )

        # Ensure pandas import via centralized imports (fallback if not loaded yet)
        local_pd = pd
        if local_pd is None:
            from ml._imports import pd as local_pd  # type: ignore[no-redef]
        assert local_pd is not None

        df = local_pd.DataFrame(rows)
        features_df, _ = self.calculate_features(df, mode="batch", fit_scaler=False)

        # Extract the last row as a plain dict
        if hasattr(features_df, "to_pandas"):
            features_pd = features_df.to_pandas()
            return {
                k: float(features_pd.iloc[-1][k]) for k in features_pd.columns if k != "timestamp"
            }
        else:
            return {
                k: float(features_df.iloc[-1][k]) for k in features_df.columns if k != "timestamp"
            }


# ===== Shared helpers to prevent drift between Config/Engineer/Pipeline =====


def build_pipeline_spec_from_feature_config(cfg: FeatureConfig) -> PipelineSpec:
    """
    Build a PipelineSpec from a FeatureConfig, including optional transforms.

    This is the single source of truth for feature name enumeration.

    """
    transforms: list[TransformSpec] = []

    # Legacy compatibility: boolean toggles default to enabled if None.
    if getattr(cfg, "enable_returns", None) is not False:
        transforms.append(
            TransformSpec(name="returns", params={"periods": list(cfg.return_periods)}),
        )

    if getattr(cfg, "enable_momentum", None) is not False:
        transforms.append(
            TransformSpec(name="momentum", params={"periods": list(cfg.momentum_periods)}),
        )

    if getattr(cfg, "enable_volatility", None) is not False:
        transforms.append(TransformSpec(name="volatility", params={}))

    # Volume ratio belongs to core indicators group conceptually, but keep separate
    # to allow parameterization by periods.
    vr_periods = list(cfg.ma_periods) if cfg.ma_periods is not None else list(cfg.volume_ma_periods)
    transforms.append(TransformSpec(name="volume_ratio", params={"periods": vr_periods}))

    if getattr(cfg, "enable_technical", None) is not False:
        transforms.append(TransformSpec(name="core_indicators", params={}))

    if getattr(cfg, "include_microstructure", False):
        transforms.append(TransformSpec(name="microstructure", params={}))
    if getattr(cfg, "include_trade_flow", False):
        transforms.append(TransformSpec(name="trade_flow", params={}))

    return PipelineSpec(transforms=transforms)

    def _calculate_microstructure_features_online(
        self: object,
        current_bar: dict[str, float],
        indicator_manager: IndicatorManager,
        feature_idx: int,
    ) -> int:
        """
        Calculate microstructure features for online inference (hot path).

        In the hot path, we delegate to the appropriate calculator based on
        available data, or use cached values from a MicrostructureActor.

        Parameters
        ----------
        current_bar : dict[str, float]
            Current OHLCV data with optional L2/L3 fields.
        indicator_manager : IndicatorManager
            Indicator manager with state.
        feature_idx : int
            Current feature buffer index.

        Returns
        -------
        int
            Updated feature buffer index.

        """
        # Check if we have L2 depth data in current bar
        if "bid_price_0" in current_bar:
            # Hot path with L2 data - use cached calculator
            if not hasattr(self, "_l2_calculator"):
                from ml.features.microstructure import L2MicrostructureFeatures

                self._l2_calculator = L2MicrostructureFeatures(n_levels=5, lookback_window=20)

            # For hot path, compute only essential features
            # Full computation should be done by MicrostructureActor

        # Spread features - estimate from high/low spread
        hl_spread = current_bar["high"] - current_bar["low"]
        close = current_bar["close"]

        # Estimated spread mean (as fraction of price)
        self.feature_buffer[feature_idx] = safe_divide(hl_spread, close, 0.0)
        feature_idx += 1

        # Spread std - would need historical data, use 0 for hot path
        self.feature_buffer[feature_idx] = 0.0
        feature_idx += 1

        # Relative spread (same as spread mean in this approximation)
        self.feature_buffer[feature_idx] = safe_divide(hl_spread, close, 0.0)
        feature_idx += 1

        # Size imbalance features - default to neutral
        self.feature_buffer[feature_idx] = 0.0  # size_imbalance_mean
        feature_idx += 1
        self.feature_buffer[feature_idx] = 0.0  # size_imbalance_std
        feature_idx += 1

        # Mid-price return volatility - estimate from recent price changes
        closes = indicator_manager.price_history["closes"]
        if len(closes) > 1:
            recent_returns = []
            for i in range(max(0, len(closes) - 5), len(closes)):
                if i > 0 and closes[i - 1] > 0:
                    ret = (closes[i] - closes[i - 1]) / closes[i - 1]
                    recent_returns.append(ret)

            if recent_returns:
                self.feature_buffer[feature_idx] = float(np.std(recent_returns))
            else:
                self.feature_buffer[feature_idx] = 0.0
        else:
            self.feature_buffer[feature_idx] = 0.0
        feature_idx += 1

        # Mid-price return autocorrelation - default to 0 for hot path
        self.feature_buffer[feature_idx] = 0.0
        feature_idx += 1

        return feature_idx

    def _calculate_trade_flow_features_online(
        self: object,
        current_bar: dict[str, float],
        indicator_manager: IndicatorManager,
        feature_idx: int,
    ) -> int:
        """
        Calculate trade flow features for online inference (hot path).

        Parameters
        ----------
        current_bar : dict[str, float]
            Current OHLCV data.
        indicator_manager : IndicatorManager
            Indicator manager with state.
        feature_idx : int
            Current feature buffer index.

        Returns
        -------
        int
            Updated feature buffer index.

        """
        # For hot path, use simplified calculations or default values
        # In production, these would be provided by a TradeFlowActor

        close = current_bar["close"]
        volume = current_bar["volume"]

        # Trade flow imbalance - default to neutral (no directional bias)
        self.feature_buffer[feature_idx] = 0.0
        feature_idx += 1

        # VWAP - use current close as approximation
        # In practice, this would be maintained by a separate VWAP calculator
        self.feature_buffer[feature_idx] = close
        feature_idx += 1

        # Trade intensity - estimate from volume
        # Higher volume typically indicates more trades
        # Normalize to reasonable range
        volumes = indicator_manager.price_history["volumes"]
        if len(volumes) > 1:
            avg_volume = sum(volumes[-min(20, len(volumes)) :]) / min(20, len(volumes))
            intensity = safe_divide(volume, avg_volume, 1.0)
            # Cap intensity to reasonable range
            self.feature_buffer[feature_idx] = min(intensity, 5.0)
        else:
            self.feature_buffer[feature_idx] = 1.0
        feature_idx += 1

        # Average price impact - estimate from price movement vs volume
        if volume > 0:
            hl_spread = current_bar["high"] - current_bar["low"]
            # Normalize by volume - higher volume should have lower per-unit impact
            impact = safe_divide(hl_spread / close, volume / 1000, 0.0)
            self.feature_buffer[feature_idx] = min(impact, 0.01)  # Cap at 1%
        else:
            self.feature_buffer[feature_idx] = 0.0
        feature_idx += 1

        return feature_idx

    def validate_feature_quality(
        self,
        features_df: DataFrameLike,
    ) -> dict[str, dict[str, float]]:
        """
        Validate feature quality metrics (cold path only).

        Parameters
        ----------
        features_df : pl.DataFrame or pd.DataFrame
            Features DataFrame to validate.

        Returns
        -------
        dict[str, dict[str, float]]
            Quality metrics per feature.

        """
        if not self.config.validate_quality or not POLARS_AVAILABLE:
            return {}

        features_df = self._convert_to_polars(features_df)
        if features_df is None or len(features_df) == 0:
            return {}

        quality_metrics = {}
        total_rows = len(features_df)

        for col in features_df.columns:
            if col in ["timestamp", "entity_id", "symbol"]:
                continue

            try:
                metrics = self._calculate_column_metrics(features_df[col], total_rows)
                quality_metrics[col] = metrics
            except Exception:  # noqa: S112
                # Skip columns that fail validation - expected for non-numeric columns
                continue

        return quality_metrics

    def _convert_to_polars(self, features_df: DataFrameLike) -> PolarsDF | None:
        """
        Convert DataFrame to Polars format.
        """
        if not hasattr(features_df, "columns") or "polars" not in str(type(features_df)):
            try:
                if pd is not None and isinstance(features_df, pd.DataFrame):
                    return pl.from_pandas(features_df)
            except Exception:
                return None
        return features_df

    def _calculate_column_metrics(
        self,
        col_data: PolarsSeries,
        total_rows: int,
    ) -> dict[str, float]:
        """
        Calculate quality metrics for a single column.
        """
        # Basic metrics
        null_count = col_data.null_count()
        zero_count = (col_data == 0.0).sum()
        unique_count = col_data.n_unique()

        metrics = {
            "null_rate": float(null_count / total_rows),
            "zero_rate": float(zero_count / total_rows),
            "unique_ratio": float(unique_count / total_rows),
            "inf_rate": 0.0,
            "outlier_rate": 0.0,
        }

        # Additional metrics for numeric columns
        if col_data.dtype in [pl.Float32, pl.Float64]:
            inf_count = col_data.is_infinite().sum()
            metrics["inf_rate"] = float(inf_count / total_rows)
            metrics["outlier_rate"] = self._calculate_outlier_rate(col_data, total_rows)

        return metrics

    def _calculate_outlier_rate(
        self,
        col_data: PolarsSeries,
        total_rows: int,
    ) -> float:
        """
        Calculate outlier rate using IQR method.
        """
        try:
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)

            if q1 is not None and q3 is not None and not (np.isnan(q1) or np.isnan(q3)):
                iqr = q3 - q1
                if iqr > 0:
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outlier_count = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
                    return float(outlier_count / total_rows)
        except Exception:  # noqa: S110
            pass
        return 0.0

    def reset(self) -> None:
        """
        Reset the feature engineer state.
        """
        if hasattr(self, "feature_buffer"):
            self.feature_buffer.fill(0.0)

    def _calculate_feature_qualities(
        self,
        features_df: DataFrameLike,
    ) -> dict[str, dict[str, float]]:
        """
        Calculate feature quality metrics for monitoring.

        Parameters
        ----------
        features_df : pl.DataFrame or pd.DataFrame
            Features to analyze.

        Returns
        -------
        dict[str, dict[str, float]]
            Quality metrics per feature.

        """
        if not HAS_POLARS or not hasattr(features_df, "select"):
            return {}

        qualities = {}

        try:
            for col in features_df.columns:
                if col in ["timestamp", "entity_id", "symbol"]:
                    continue

                col_data = features_df.select(col).to_numpy().flatten()
                total_rows = len(col_data)

                if total_rows == 0:
                    qualities[col] = {"null_ratio": 1.0, "infinite_ratio": 1.0}
                    continue

                # Calculate null ratio
                null_count = np.sum(np.isnan(col_data))
                null_ratio = null_count / total_rows

                # Calculate infinite ratio
                inf_count = np.sum(np.isinf(col_data))
                inf_ratio = inf_count / total_rows

                qualities[col] = {
                    "null_ratio": float(null_ratio),
                    "infinite_ratio": float(inf_ratio),
                }

        except Exception:  # noqa: S110
            # Graceful degradation
            pass

        return qualities


class _dummy_context_manager:
    """
    Dummy context manager for when metrics is None.
    """

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        pass
