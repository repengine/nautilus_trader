"""
Static instrument feature helpers for TFT datasets and realtime inference.

This module centralizes static feature defaults (asset class, tick size, exchange)
to keep dataset builders and realtime calculators consistent without duplicating
logic across components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from ml._imports import pd as pd_runtime
from ml._imports import pl as pl_runtime


if TYPE_CHECKING:
    import pandas as _pd
    import polars as _pl
else:  # pragma: no cover - typing fallback
    _pd = Any
    _pl = Any


pl: Any = cast(Any, pl_runtime)
pd: Any = cast(Any, pd_runtime)


STATIC_FEATURE_MAP: dict[str, dict[str, str | float]] = {
    "SPY": {"asset_class": "ETF", "tick_size": 0.01, "exchange": "ARCA"},
    "QQQ": {"asset_class": "ETF", "tick_size": 0.01, "exchange": "NASDAQ"},
    "AAPL": {"asset_class": "STOCK", "tick_size": 0.01, "exchange": "NASDAQ"},
    "MSFT": {"asset_class": "STOCK", "tick_size": 0.01, "exchange": "NASDAQ"},
    "NVDA": {"asset_class": "STOCK", "tick_size": 0.01, "exchange": "NASDAQ"},
    "AMZN": {"asset_class": "STOCK", "tick_size": 0.01, "exchange": "NASDAQ"},
    "META": {"asset_class": "STOCK", "tick_size": 0.01, "exchange": "NASDAQ"},
    "GOOGL": {"asset_class": "STOCK", "tick_size": 0.01, "exchange": "NASDAQ"},
    "TSLA": {"asset_class": "STOCK", "tick_size": 0.01, "exchange": "NASDAQ"},
}

DEFAULT_STATIC_FEATURES: dict[str, str | float] = {
    "asset_class": "STOCK",
    "tick_size": 0.01,
    "exchange": "UNKNOWN",
}


def resolve_static_features(symbol: str) -> dict[str, str | float]:
    """
    Resolve static feature defaults for a symbol.

    Args:
        symbol: Base instrument symbol (no exchange suffix).

    Returns:
        Mapping of static feature values.
    """
    return STATIC_FEATURE_MAP.get(symbol, DEFAULT_STATIC_FEATURES)


def resolve_tick_size(instrument_id: str) -> float:
    """
    Resolve tick size for an instrument identifier.

    Args:
        instrument_id: Full instrument identifier (may include exchange suffix).

    Returns:
        Tick size value.
    """
    symbol = instrument_id.split(".")[0] if instrument_id else ""
    static = resolve_static_features(symbol)
    return float(static.get("tick_size", 0.01))


def add_static_features_polars(df: _pl.DataFrame) -> _pl.DataFrame:
    """
    Add static instrument features using Polars.

    Adds asset_class, tick_size, and exchange columns based on the
    instrument_id column. Uses default values for unknown symbols.

    Args:
        df: Polars DataFrame with instrument_id column.

    Returns:
        DataFrame with added static feature columns.

    Raises:
        ValueError: If instrument_id column is missing.
    """
    if pl is None:
        raise RuntimeError("Polars is required for static feature enrichment")
    if "instrument_id" not in df.columns:
        raise ValueError("Missing required 'instrument_id' column for static features")

    if df.is_empty():
        return df.with_columns(
            [
                pl.lit(None).cast(pl.Utf8).alias("asset_class"),
                pl.lit(None).cast(pl.Float64).alias("tick_size"),
                pl.lit(None).cast(pl.Utf8).alias("exchange"),
            ],
        )

    instruments = df["instrument_id"].unique().to_list()
    result = df
    for instrument in instruments:
        static = STATIC_FEATURE_MAP.get(instrument, DEFAULT_STATIC_FEATURES)
        result = result.with_columns(
            [
                pl.when(pl.col("instrument_id") == instrument)
                .then(pl.lit(static["asset_class"]))
                .otherwise(
                    pl.col("asset_class")
                    if "asset_class" in result.columns
                    else pl.lit("UNKNOWN"),
                )
                .alias("asset_class"),
                pl.when(pl.col("instrument_id") == instrument)
                .then(pl.lit(static["tick_size"]))
                .otherwise(
                    pl.col("tick_size") if "tick_size" in result.columns else pl.lit(0.01),
                )
                .alias("tick_size"),
                pl.when(pl.col("instrument_id") == instrument)
                .then(pl.lit(static["exchange"]))
                .otherwise(
                    pl.col("exchange") if "exchange" in result.columns else pl.lit("UNKNOWN"),
                )
                .alias("exchange"),
            ],
        )
    return result


def add_static_features_pandas(df: _pd.DataFrame) -> _pd.DataFrame:
    """
    Add static instrument features using Pandas.

    Adds asset_class, tick_size, and exchange columns based on the
    instrument_id column. Uses default values for unknown symbols.

    Args:
        df: Pandas DataFrame with instrument_id column.

    Returns:
        DataFrame with added static feature columns.

    Raises:
        ValueError: If instrument_id column is missing.
    """
    if pd is None:
        raise RuntimeError("Pandas is required for static feature enrichment")
    if "instrument_id" not in df.columns:
        raise ValueError("Missing required 'instrument_id' column for static features")

    if len(df) == 0:
        result = df.copy()
        result["asset_class"] = pd.Series([], dtype=str)
        result["tick_size"] = pd.Series([], dtype=float)
        result["exchange"] = pd.Series([], dtype=str)
        return result

    result = df.copy()

    def get_asset_class(value: str) -> str:
        return str(resolve_static_features(value).get("asset_class", "STOCK"))

    def get_tick_size(value: str) -> float:
        return float(resolve_static_features(value).get("tick_size", 0.01))

    def get_exchange(value: str) -> str:
        return str(resolve_static_features(value).get("exchange", "UNKNOWN"))

    result["asset_class"] = result["instrument_id"].map(get_asset_class)
    result["tick_size"] = result["instrument_id"].map(get_tick_size)
    result["exchange"] = result["instrument_id"].map(get_exchange)
    return result


__all__ = [
    "DEFAULT_STATIC_FEATURES",
    "STATIC_FEATURE_MAP",
    "add_static_features_pandas",
    "add_static_features_polars",
    "resolve_static_features",
    "resolve_tick_size",
]
