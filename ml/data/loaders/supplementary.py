"""
Supplementary data loaders used by synthetic population tasks.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

SUPPLEMENTARY_SYMBOLS: Mapping[str, tuple[str, ...]] = {
    "indices": ("^GSPC", "^DJI", "^IXIC", "^RUT"),
    "sectors": ("XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLB", "XLRE", "XLU", "XLC"),
    "factors": ("IWF", "IWD", "IWM", "IWB", "MTUM", "QUAL", "USMV"),
    "international": ("EWJ", "EWG", "EWU", "FXI", "EWZ", "EWA", "EWC", "INDA", "EEM", "EFA"),
    "commodities": ("GLD", "SLV", "USO", "UNG", "DBA", "DBB", "DBC"),
    "bonds": ("SHY", "IEF", "TLT", "TIP", "LQD", "HYG", "EMB", "AGG"),
    "currencies": ("UUP", "FXE", "FXY", "FXB", "FXC", "FXA", "FXF"),
    "volatility": ("VXX", "VIXY", "VXZ", "SVXY", "UVXY"),
}

DEFAULT_BASE_SYMBOLS: tuple[str, ...] = ("SPY", "QQQ", "TLT", "GLD", "XLK", "XLF")


@dataclass(slots=True, frozen=True)
class SpreadDefinition:
    """
    Definition of a spread ratio between two symbols.
    """

    long_symbol: str
    short_symbol: str
    name: str


DEFAULT_SPREADS: tuple[SpreadDefinition, ...] = (
    SpreadDefinition("TLT", "IEF", "yield_curve"),
    SpreadDefinition("HYG", "LQD", "credit_spread"),
    SpreadDefinition("XLK", "XLU", "tech_utilities"),
    SpreadDefinition("IWF", "IWD", "growth_value"),
    SpreadDefinition("EEM", "EFA", "em_dm"),
    SpreadDefinition("GLD", "TLT", "gold_bonds"),
    SpreadDefinition("FXY", "FXA", "safe_risk_fx"),
)


@dataclass(slots=True, frozen=True)
class SupplementaryOutputs:
    """
    Generated file artifacts and summary metadata.
    """

    ohlcv_path: Path
    correlations_path: Path | None
    spreads_path: Path | None
    metadata_path: Path
    record_count: int
    symbol_count: int
    start: datetime
    end: datetime


@dataclass(slots=True, frozen=True)
class SupplementaryDataConfig:
    """
    Configuration for synthetic supplementary data generation.
    """

    output_dir: Path
    base_symbols: tuple[str, ...] = DEFAULT_BASE_SYMBOLS
    spread_definitions: tuple[SpreadDefinition, ...] = DEFAULT_SPREADS
    synthetic_years: int = 2


@dataclass(slots=True, frozen=True)
class PopulateSupplementaryTaskConfig:
    """
    Arguments accepted by :func:`populate_supplementary_data`.
    """

    output_dir: Path
    base_symbols: tuple[str, ...] = DEFAULT_BASE_SYMBOLS
    synthetic_years: int = 2


@dataclass(slots=True, frozen=True)
class PopulateYahooDataTaskConfig:
    """
    Arguments accepted by :func:`populate_yahoo_data`.
    """

    output_dir: Path
    categories: Sequence[str] | None = None
    synthetic_years: int = 2


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0.0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0.0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def create_synthetic_supplementary_data(config: SupplementaryDataConfig) -> pd.DataFrame:
    LOGGER.info("Creating synthetic supplementary data (years=%s)", config.synthetic_years)

    end_date = datetime.now(tz=UTC)
    start_date = end_date - timedelta(days=365 * config.synthetic_years)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    all_symbols: list[str] = []
    for symbols in SUPPLEMENTARY_SYMBOLS.values():
        all_symbols.extend(symbols)

    frames: list[pd.DataFrame] = []
    volatility_symbols = SUPPLEMENTARY_SYMBOLS.get("volatility", ())
    bond_symbols = SUPPLEMENTARY_SYMBOLS.get("bonds", ())
    commodity_symbols = SUPPLEMENTARY_SYMBOLS.get("commodities", ())

    for symbol in all_symbols:
        base_price = 100.0
        volatility = 0.02
        if symbol in volatility_symbols:
            volatility = 0.10
        elif symbol in bond_symbols:
            volatility = 0.01
        elif symbol in commodity_symbols:
            volatility = 0.03

        rng = np.random.default_rng(seed=hash(symbol) & 0xFFFFFFFF)
        returns = rng.normal(0.0001, volatility, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        trend = np.linspace(0.0, 0.2, len(dates))
        prices = prices * (1 + trend)

        frame = pd.DataFrame(
            {
                "timestamp": dates,
                "symbol": symbol,
                "open": prices * (1 + rng.normal(0.0, 0.001, len(dates))),
                "high": prices * (1 + np.abs(rng.normal(0.0, 0.005, len(dates)))),
                "low": prices * (1 - np.abs(rng.normal(0.0, 0.005, len(dates)))),
                "close": prices,
                "volume": rng.lognormal(15, 1, len(dates)).astype(int) * 1000,
            },
        )
        frame["returns"] = frame["close"].pct_change()
        frame["log_returns"] = np.log(frame["close"] / frame["close"].shift(1))
        frame["volatility_20d"] = frame["returns"].rolling(20).std()
        frame["volume_ma_20d"] = frame["volume"].rolling(20).mean()
        frame["rsi_14"] = calculate_rsi(frame["close"], 14)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["symbol", "timestamp"])
    combined = combined.reset_index(drop=True)
    LOGGER.info(
        "Synthetic supplementary dataset created (symbols=%s, rows=%s)",
        len(all_symbols),
        len(combined),
    )
    return combined


def calculate_correlations(data: pd.DataFrame, base_symbols: Iterable[str]) -> pd.DataFrame:
    LOGGER.info("Calculating supplementary correlations")
    pivot = data.pivot_table(index="timestamp", columns="symbol", values="returns")
    results: list[pd.DataFrame] = []
    for base in base_symbols:
        if base not in pivot:
            continue
        for symbol in pivot.columns:
            if symbol == base:
                continue
            corr = pivot[base].rolling(60).corr(pivot[symbol])
            corr_df = pd.DataFrame(
                {
                    "timestamp": corr.index,
                    "base_symbol": base,
                    "corr_symbol": symbol,
                    "correlation_60d": corr.to_numpy(),
                },
            )
            results.append(corr_df)
    if results:
        merged = pd.concat(results, ignore_index=True)
        LOGGER.info("Correlation rows computed: %s", len(merged))
        return merged
    return pd.DataFrame()


def calculate_spreads(
    data: pd.DataFrame,
    spread_definitions: Sequence[SpreadDefinition] = DEFAULT_SPREADS,
) -> pd.DataFrame:
    LOGGER.info("Calculating supplementary spreads")
    pivot = data.pivot_table(index="timestamp", columns="symbol", values="close")
    results: list[pd.DataFrame] = []
    for definition in spread_definitions:
        long_sym = definition.long_symbol
        short_sym = definition.short_symbol
        if long_sym not in pivot or short_sym not in pivot:
            continue
        ratio = pivot[long_sym] / pivot[short_sym]
        spread_df = pd.DataFrame(
            {
                "timestamp": ratio.index,
                "spread_name": definition.name,
                f"{definition.name}_ratio": ratio,
            },
        )
        spread_df[f"{definition.name}_ma20"] = ratio.rolling(20).mean()
        rolling_mean = ratio.rolling(60).mean()
        rolling_std = ratio.rolling(60).std()
        spread_df[f"{definition.name}_zscore"] = (ratio - rolling_mean) / rolling_std
        results.append(spread_df)
    if results:
        merged = pd.concat(results, ignore_index=True)
        LOGGER.info("Spread rows computed: %s", len(merged))
        return merged
    return pd.DataFrame()


def write_supplementary_outputs(
    data: pd.DataFrame,
    correlations: pd.DataFrame,
    spreads: pd.DataFrame,
    config: SupplementaryDataConfig,
) -> SupplementaryOutputs:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ohlcv_path = output_dir / "supplementary_ohlcv.parquet"
    data.to_parquet(ohlcv_path)

    correlations_path: Path | None = None
    if not correlations.empty:
        correlations_path = output_dir / "correlations.parquet"
        correlations.to_parquet(correlations_path)

    spreads_path: Path | None = None
    if not spreads.empty:
        spreads_path = output_dir / "spreads.parquet"
        spreads.to_parquet(spreads_path)

    metadata = {
        "created": datetime.now(tz=UTC).isoformat(),
        "symbols": {category: list(symbols) for category, symbols in SUPPLEMENTARY_SYMBOLS.items()},
        "total_symbols": int(data["symbol"].nunique()),
        "date_range": {
            "start": data["timestamp"].min().isoformat(),
            "end": data["timestamp"].max().isoformat(),
        },
        "record_count": len(data),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return SupplementaryOutputs(
        ohlcv_path=ohlcv_path,
        correlations_path=correlations_path,
        spreads_path=spreads_path,
        metadata_path=metadata_path,
        record_count=len(data),
        symbol_count=int(data["symbol"].nunique()),
        start=pd.to_datetime(data["timestamp"].min()).to_pydatetime(),
        end=pd.to_datetime(data["timestamp"].max()).to_pydatetime(),
    )


def populate_supplementary_data(config: PopulateSupplementaryTaskConfig) -> SupplementaryOutputs:
    """
    Generate synthetic supplementary data and persist parquet outputs.
    """
    data_config = SupplementaryDataConfig(
        output_dir=config.output_dir,
        base_symbols=config.base_symbols,
        synthetic_years=config.synthetic_years,
    )
    data = create_synthetic_supplementary_data(data_config)
    if data.empty:
        raise ValueError("Supplementary data generation produced no rows")

    correlations = calculate_correlations(data, data_config.base_symbols)
    spreads = calculate_spreads(data, data_config.spread_definitions)
    return write_supplementary_outputs(data, correlations, spreads, data_config)


def _select_yahoo_symbols(categories: Sequence[str] | None) -> tuple[str, ...]:
    if not categories:
        symbols: list[str] = []
        for values in SUPPLEMENTARY_SYMBOLS.values():
            symbols.extend(values)
        return tuple(symbols)

    invalid_categories = [category for category in categories if category not in SUPPLEMENTARY_SYMBOLS]
    if invalid_categories:
        raise ValueError(f"Unknown Yahoo categories: {', '.join(invalid_categories)}")

    selected_symbols: list[str] = []
    for category in categories:
        selected_symbols.extend(SUPPLEMENTARY_SYMBOLS[category])
    return tuple(selected_symbols)


def populate_yahoo_data(config: PopulateYahooDataTaskConfig) -> SupplementaryOutputs:
    """
    Generate Yahoo-style supplementary data and persist parquet outputs.
    """
    symbols = _select_yahoo_symbols(config.categories)
    data_config = SupplementaryDataConfig(
        output_dir=config.output_dir,
        synthetic_years=config.synthetic_years,
    )
    data = create_synthetic_supplementary_data(data_config)
    data = data[data["symbol"].isin(symbols)].reset_index(drop=True)
    if data.empty:
        raise ValueError("No synthetic Yahoo data generated for requested categories")

    correlations = calculate_correlations(data, data_config.base_symbols)
    spreads = calculate_spreads(data, data_config.spread_definitions)
    return write_supplementary_outputs(data, correlations, spreads, data_config)


__all__ = [
    "DEFAULT_BASE_SYMBOLS",
    "DEFAULT_SPREADS",
    "SUPPLEMENTARY_SYMBOLS",
    "PopulateSupplementaryTaskConfig",
    "PopulateYahooDataTaskConfig",
    "SpreadDefinition",
    "SupplementaryDataConfig",
    "SupplementaryOutputs",
    "calculate_correlations",
    "calculate_spreads",
    "create_synthetic_supplementary_data",
    "populate_supplementary_data",
    "populate_yahoo_data",
    "write_supplementary_outputs",
]
