"""
Advanced ML preprocessing utilities for Nautilus Trader.

This module provides cold-path preprocessing tools for financial time series data,
implementing techniques from "Advances in Financial Machine Learning" by López de Prado
and other academic literature. All preprocessing is designed for batch operations
and must not be used in hot paths (actors, on_* handlers).

Key Components
--------------

Stationarity and Transformations:
    - StationarityTransformer: Fractional differencing for stationarity preservation
    - DataNormalizer: Robust, rank, and Box-Cox normalization techniques
    - FeatureLagGenerator: Comprehensive lag feature creation with rolling statistics

Point-in-Time Data Management:
    - asof_join: Point-in-time correct joins to prevent lookahead bias
    - embargo_window: Event-based embargo windows for training data
    - validate_no_lookahead: Validation to ensure temporal consistency
    - create_lag_features: Time-aware lag feature creation

Cross-Validation:
    - PurgedCrossValidator: Purged walk-forward CV for financial time series

Market Microstructure:
    - MarketMicrostructureFeatures: Roll's spread, Kyle's lambda, Amihud illiquidity, VPIN

Universal ML Architecture Patterns Compliance
---------------------------------------------

This module follows the Universal ML Architecture Patterns:

Pattern 3 - Cold Path Operations:
    All preprocessing utilities are designed for cold path batch operations.
    Never use these in hot paths (actors, on_* handlers) as they involve:
    - Heavy DataFrame operations (Pandas/Polars)
    - Statistical computations with SciPy/StatsModels
    - Large memory allocations
    - File I/O operations

Examples
--------
Fractional Differencing for Stationarity:
    >>> from ml.preprocessing import StationarityTransformer
    >>> transformer = StationarityTransformer(method="fractional", d=0.5)
    >>> stationary_series = transformer.fit_transform(price_series, auto_d=True)

Point-in-Time Joins:
    >>> # Note: Due to circular imports, use direct import for joins
    >>> from ml.preprocessing.joins import asof_join
    >>> # Join market data with news events (no lookahead bias)
    >>> joined_df = asof_join(
    ...     market_data, news_events,
    ...     on="ts_event", by="instrument_id"
    ... )

Purged Cross-Validation:
    >>> from ml.preprocessing import PurgedCrossValidator
    >>> cv = PurgedCrossValidator(n_splits=5, purge_gap=10, embargo_pct=0.1)
    >>> for train_idx, test_idx in cv.split(X, y):
    ...     # Train and evaluate with proper temporal splits
    ...     pass

Market Microstructure Features:
    >>> from ml.preprocessing import MarketMicrostructureFeatures
    >>> features = MarketMicrostructureFeatures()
    >>> spread = features.roll_spread(prices)
    >>> lambda_param = features.kyle_lambda(prices, volumes)

Data Normalization:
    >>> from ml.preprocessing import DataNormalizer
    >>> normalizer = DataNormalizer(method="robust")  # Resistant to outliers
    >>> normalized_data = normalizer.fit_transform(raw_features)

Lag Feature Engineering:
    >>> from ml.preprocessing import FeatureLagGenerator
    >>> lag_gen = FeatureLagGenerator(lag_periods=[1, 5, 20])
    >>> lag_features = lag_gen.create_lagged_features(
    ...     price_series, include_rolling=True, include_ewm=True
    ... )

Embargo Windows:
    >>> # Note: Use direct import for joins utilities
    >>> from ml.preprocessing.joins import embargo_window
    >>> # Exclude data around earnings announcements
    >>> embargoed_df = embargo_window(
    ...     market_df, earnings_timestamps,
    ...     embargo_before_ns=3600_000_000_000,  # 1 hour before
    ...     embargo_after_ns=7200_000_000_000    # 2 hours after
    ... )

Notes
-----
- All functions work with both Pandas and Polars DataFrames where applicable
- Timestamp handling assumes nanosecond precision (Nautilus standard)
- Memory-efficient implementations using NumPy for heavy computations
- JIT compilation with Numba where available for performance
- Extensive validation to prevent lookahead bias and data leakage

Important: Circular Import Limitation
------------------------------------
Due to circular imports in the ML dependency chain (ml._imports ↔ ml.common.metrics_bootstrap),
the joins utilities (asof_join, embargo_window, etc.) should be imported directly:

    from ml.preprocessing.joins import asof_join, embargo_window

Stationarity utilities can be imported through this module without issues:

    from ml.preprocessing import StationarityTransformer, PurgedCrossValidator

See Also
--------
ml.config.preprocessing : Configuration classes for preprocessing pipelines
ml.features : Feature engineering and transformation pipelines
ml.common.validation : Data validation utilities

"""

from __future__ import annotations

# Import order: Standard library, third-party, local
from typing import TYPE_CHECKING, Any


# Lazy imports to avoid circular import issues in the ML ecosystem
def __getattr__(name: str) -> Any:
    """
    Lazy loading of preprocessing components to avoid circular imports.

    This approach prevents the circular import issue in the broader ML codebase while
    maintaining a clean public API for cold-path preprocessing operations.

    """
    # Stationarity transformations
    if name == "StationarityTransformer":
        from ml.preprocessing.stationarity import StationarityTransformer

        return StationarityTransformer
    elif name == "DataNormalizer":
        from ml.preprocessing.stationarity import DataNormalizer

        return DataNormalizer
    elif name == "FeatureLagGenerator":
        from ml.preprocessing.stationarity import FeatureLagGenerator

        return FeatureLagGenerator
    elif name == "MarketMicrostructureFeatures":
        from ml.preprocessing.stationarity import MarketMicrostructureFeatures

        return MarketMicrostructureFeatures
    elif name == "PurgedCrossValidator":
        from ml.preprocessing.stationarity import PurgedCrossValidator

        return PurgedCrossValidator

    # Point-in-time join utilities (with circular import protection)
    elif name == "asof_join":
        try:
            from ml.preprocessing.joins import asof_join

            return asof_join
        except ImportError as e:
            raise ImportError(
                f"Cannot import {name} due to circular import in ML dependencies. "
                f"Import directly: 'from ml.preprocessing.joins import {name}'. "
                f"Original error: {e}",
            ) from e
    elif name == "create_lag_features":
        try:
            from ml.preprocessing.joins import create_lag_features

            return create_lag_features
        except ImportError as e:
            raise ImportError(
                f"Cannot import {name} due to circular import in ML dependencies. "
                f"Import directly: 'from ml.preprocessing.joins import {name}'. "
                f"Original error: {e}",
            ) from e
    elif name == "embargo_window":
        try:
            from ml.preprocessing.joins import embargo_window

            return embargo_window
        except ImportError as e:
            raise ImportError(
                f"Cannot import {name} due to circular import in ML dependencies. "
                f"Import directly: 'from ml.preprocessing.joins import {name}'. "
                f"Original error: {e}",
            ) from e
    elif name == "validate_no_lookahead":
        try:
            from ml.preprocessing.joins import validate_no_lookahead

            return validate_no_lookahead
        except ImportError as e:
            raise ImportError(
                f"Cannot import {name} due to circular import in ML dependencies. "
                f"Import directly: 'from ml.preprocessing.joins import {name}'. "
                f"Original error: {e}",
            ) from e
    elif name == "EventIngestionConfig":
        from ml.preprocessing.event_ingestion import EventIngestionConfig

        return EventIngestionConfig
    elif name == "EventIngestionUtility":
        from ml.preprocessing.event_ingestion import EventIngestionUtility

        return EventIngestionUtility

    # If attribute not found, raise AttributeError
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


if TYPE_CHECKING:
    # Type imports for static analysis only
    pass

# Public API - sorted alphabetically
__all__ = [
    "DataNormalizer",
    "EventIngestionConfig",
    "EventIngestionUtility",
    "FeatureLagGenerator",
    "MarketMicrostructureFeatures",
    "PurgedCrossValidator",
    "StationarityTransformer",
    "asof_join",
    "create_lag_features",
    "embargo_window",
    "validate_no_lookahead",
]

# Version info
__version__ = "1.0.0"
__author__ = "Nautilus ML Team"

# Module metadata
__module_type__ = "cold_path"
__performance_budget__ = "unlimited"  # Cold path operations
__dependencies__ = ["numpy", "scipy", "statsmodels", "numba?", "polars?", "pandas?"]
