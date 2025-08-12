"""
Constants for ML components.

This module centralizes all constants used in machine learning components for better
maintainability and clarity.

"""

from enum import Enum
from typing import Final


# ============================================================================
# MODEL EXPORT FORMATS
# ============================================================================


class ExportFormats(Enum):
    """
    Supported model export formats.
    """

    ONNX = "onnx"
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    PICKLE = "pickle"
    JSON = "json"


# File extensions (not part of enum)
SUFFIX_ONNX = ".onnx"
SUFFIX_XGB = ".xgb"
SUFFIX_LGB = ".lgb"


class Versions:
    """
    Version constants for ML components.
    """

    ONNX_OPSET = 17
    LIGHTGBM_MIN = "3.0.0"
    XGBOOST_MIN = "1.6.0"
    DEFAULT_MANIFEST_VERSION = "1.0.0"
    DEFAULT_TRAINER_VERSION = "1.0.0"


class Providers:
    """
    ONNX Runtime providers.
    """

    CPU = "CPUExecutionProvider"
    CUDA = "CUDAExecutionProvider"


# ============================================================================
# TIME CONSTANTS
# ============================================================================


class TimeConstants:
    """
    Time-related constants for ML components.
    """

    # Nanosecond conversions
    NS_IN_SECOND: Final[int] = 1_000_000_000
    NS_IN_MINUTE: Final[int] = 60 * NS_IN_SECOND
    NS_IN_HOUR: Final[int] = 3600 * NS_IN_SECOND
    NS_IN_DAY: Final[int] = 86400 * NS_IN_SECOND

    # Trading calendar
    TRADING_DAYS_PER_YEAR: Final[int] = 252
    TRADING_HOURS_PER_DAY: Final[float] = 6.5
    TRADING_WEEKS_PER_YEAR: Final[int] = 52


# ============================================================================
# TECHNICAL INDICATOR CONSTANTS
# ============================================================================


class TechnicalIndicatorPeriods:
    """
    Standard periods for technical indicators.
    """

    # Moving averages
    MA_FAST_PERIOD: Final[int] = 5
    MA_MEDIUM_PERIOD: Final[int] = 10
    MA_SLOW_PERIOD: Final[int] = 20
    MA_LONG_PERIOD: Final[int] = 50

    # EMA periods
    EMA_FAST_DEFAULT: Final[int] = 12
    EMA_SLOW_DEFAULT: Final[int] = 26

    # RSI
    RSI_DEFAULT_PERIOD: Final[int] = 14
    RSI_OVERSOLD_LEVEL: Final[int] = 30
    RSI_OVERBOUGHT_LEVEL: Final[int] = 70

    # Bollinger Bands
    BB_DEFAULT_PERIOD: Final[int] = 20
    BB_DEFAULT_STD: Final[float] = 2.0

    # MACD
    MACD_FAST_PERIOD: Final[int] = 12
    MACD_SLOW_PERIOD: Final[int] = 26
    MACD_SIGNAL_PERIOD: Final[int] = 9

    # Volatility
    VOLATILITY_SHORT_PERIOD: Final[int] = 5
    VOLATILITY_MEDIUM_PERIOD: Final[int] = 20
    VOLATILITY_LONG_PERIOD: Final[int] = 60


# ============================================================================
# ML MODEL CONSTANTS
# ============================================================================


class MLConstants:
    """
    Machine learning related constants.
    """

    # Data windows
    DEFAULT_LOOKBACK_DAYS: Final[int] = 252  # 1 year
    MIN_LOOKBACK_DAYS: Final[int] = 20
    MAX_LOOKBACK_DAYS: Final[int] = 1260  # 5 years

    # Model thresholds
    DEFAULT_CONFIDENCE_THRESHOLD: Final[float] = 0.6
    MIN_CONFIDENCE_THRESHOLD: Final[float] = 0.5
    HIGH_CONFIDENCE_THRESHOLD: Final[float] = 0.8

    # Feature engineering
    MAX_LAG_FEATURES: Final[int] = 10
    DEFAULT_LAG_PERIODS: Final[list[int]] = [1, 2, 3, 5, 10, 20]

    # Performance monitoring
    MAX_INFERENCE_LATENCY_MS: Final[float] = 5.0
    PERFORMANCE_REGRESSION_THRESHOLD: Final[float] = 0.2  # 20% regression allowed

    # Feature parity validation
    FEATURE_PARITY_TOLERANCE: Final[float] = 1e-10


# ============================================================================
# SYSTEM CONSTANTS
# ============================================================================


class SystemConstants:
    """
    System-level constants.
    """

    # Queue and buffer sizes
    DEFAULT_QUEUE_MAXLEN: Final[int] = 100
    PRICE_HISTORY_MAXLEN: Final[int] = 252

    # Memory limits for hot path
    MAX_FEATURE_BUFFER_SIZE: Final[int] = 1000
    MAX_INDICATOR_HISTORY: Final[int] = 252


# ============================================================================
# FEATURE COLUMN NAMES
# ============================================================================


class FeatureColumns(str, Enum):
    """
    Standard feature column names.
    """

    # Price data
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"

    # Derived price features
    RETURNS = "returns"
    LOG_RETURNS = "log_returns"
    PRICE = "price"

    # Time features
    TIMESTAMP = "timestamp"
    DATE = "date"
    TIME = "time"

    # Target variables
    TARGET = "target"
    SIGNAL = "signal"
    PREDICTION = "prediction"


# ============================================================================
# INDICATOR NAMES
# ============================================================================


class IndicatorNames:
    """
    Standard indicator naming patterns.
    """

    # Moving averages
    SMA_PREFIX = "sma_"
    EMA_PREFIX = "ema_"

    # Price indicators
    PRICE_SMA_5 = "price_sma_5"
    PRICE_SMA_20 = "price_sma_20"
    PRICE_EMA_12 = "price_ema_12"
    PRICE_EMA_26 = "price_ema_26"

    # Volume indicators
    VOLUME_SMA_5 = "volume_sma_5"
    VOLUME_SMA_10 = "volume_sma_10"
    VOLUME_SMA_20 = "volume_sma_20"

    # Volatility indicators
    VOLATILITY_5 = "volatility_5"
    VOLATILITY_20 = "volatility_20"
    VOLATILITY_60 = "volatility_60"

    # Bollinger Bands
    BB_UPPER = "bb_upper"
    BB_MIDDLE = "bb_middle"
    BB_LOWER = "bb_lower"
    BB_WIDTH = "bb_width"
    BB_POSITION = "bb_position"

    # MACD
    MACD_LINE = "macd_line"
    MACD_SIGNAL = "macd_signal"
    MACD_DIFF = "macd_diff"

    # RSI
    RSI = "rsi"
    RSI_OVERBOUGHT = "rsi_overbought"
    RSI_OVERSOLD = "rsi_oversold"

    # Feature patterns
    LAG_PREFIX = "lag_"
    FEATURE_PREFIX = "feature_"

    @staticmethod
    def lag_feature(period: int) -> str:
        """
        Generate lag feature name.
        """
        return f"{IndicatorNames.LAG_PREFIX}{period}"

    @staticmethod
    def feature_index(index: int) -> str:
        """
        Generate indexed feature name.
        """
        return f"{IndicatorNames.FEATURE_PREFIX}{index}"
