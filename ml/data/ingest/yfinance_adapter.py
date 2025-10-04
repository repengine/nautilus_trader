"""YFinance ingestion utilities for asset price and return data."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd
import polars as pl

from ml.ml_types import PolarsDF


LOGGER = logging.getLogger(__name__)


_YFINANCE_MODULE: Any | None = None


def _ensure_yfinance() -> Any:
    """Return the yfinance module, importing it lazily."""
    global _YFINANCE_MODULE
    if _YFINANCE_MODULE is not None:
        return _YFINANCE_MODULE

    try:
        module = importlib.import_module("yfinance")
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "yfinance is required for this ingestion helper. Install with `pip install yfinance`.",
        ) from exc

    _YFINANCE_MODULE = module
    return module


class PriceFetcherProtocol(Protocol):
    """Protocol describing the minimal interface for fetching price data."""

    def fetch(self, config: YFinanceIngestConfig) -> pd.DataFrame:
        """Return a wide-format price DataFrame for the supplied configuration."""


@dataclass(frozen=True)
class YFinanceIngestConfig:
    """Configuration for YFinance price ingestion."""

    symbols: tuple[str, ...]
    start: datetime
    end: datetime | None = None
    interval: str = "1d"
    auto_adjust: bool = False
    threads: bool | int | None = True
    progress: bool = False

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValueError("At least one symbol must be provided")
        if self.end is not None and self.end <= self.start:
            raise ValueError("End timestamp must be after start timestamp")


class YFinanceIngestError(RuntimeError):
    """Raised when the yfinance ingestion pipeline fails."""


class YFinancePriceFetcher:
    """Concrete price fetcher backed by the yfinance library."""

    def __init__(self) -> None:
        self._client = _ensure_yfinance()

    def fetch(self, config: YFinanceIngestConfig) -> pd.DataFrame:
        LOGGER.info(
            "Fetching %s symbols from yfinance between %s and %s",
            len(config.symbols),
            config.start,
            config.end,
        )
        try:
            data = self._client.download(
                tickers=list(config.symbols),
                start=config.start,
                end=config.end,
                interval=config.interval,
                auto_adjust=config.auto_adjust,
                progress=config.progress,
                threads=config.threads,
                group_by="column",
            )
        except Exception as exc:  # pragma: no cover - transport failure
            msg = f"yfinance download failed for symbols {config.symbols}: {exc}"
            raise YFinanceIngestError(msg) from exc
        data_df = cast(pd.DataFrame, data)
        if data_df.empty:
            msg = f"yfinance returned no data for symbols {config.symbols}"
            raise YFinanceIngestError(msg)
        return data_df


def fetch_asset_history(
    config: YFinanceIngestConfig,
    *,
    fetcher: PriceFetcherProtocol | None = None,
) -> PolarsDF:
    """Fetch and normalize asset price history into a tidy Polars DataFrame."""
    concrete_fetcher = fetcher or YFinancePriceFetcher()
    raw_prices = concrete_fetcher.fetch(config)
    tidy_pd = _normalize_prices(raw_prices, config.symbols)
    tidy_pd["timestamp"] = _localize_timestamp(tidy_pd["timestamp"].to_numpy())
    tidy_pd["symbol"] = tidy_pd["symbol"].astype(str)
    tidy_pd = tidy_pd.sort_values(["symbol", "timestamp"], kind="mergesort")

    tidy_pl = pl.from_pandas(
        tidy_pd,
        schema_overrides={"timestamp": pl.Datetime("us", "UTC")},
    )
    return tidy_pl


def compute_returns(
    prices: PolarsDF,
    *,
    price_column: str = "adj_close",
) -> PolarsDF:
    """Compute simple and log returns for each symbol."""
    if price_column not in prices.columns:
        msg = f"Price column '{price_column}' not present in DataFrame"
        raise YFinanceIngestError(msg)

    ordered = prices.sort(["symbol", "timestamp"])
    returns = ordered.with_columns(
        pl.col(price_column)
        .pct_change()
        .over("symbol")
        .alias("return"),
        (pl.col(price_column).log().diff().over("symbol")).alias("log_return"),
    )
    return returns


def ingest_asset_returns(
    config: YFinanceIngestConfig,
    output_path: str | None = None,
    *,
    fetcher: PriceFetcherProtocol | None = None,
) -> PolarsDF:
    """
    Fetch prices using yfinance and compute returns. Optionally persist to parquet.

    Parameters
    ----------
    config : YFinanceIngestConfig
        Configuration describing symbols, time range, and options.
    output_path : str | Path | None, optional
        If provided, the computed returns are written to this parquet file.
    fetcher : PriceFetcherProtocol | None, optional
        Custom fetcher implementation for testing or alternative data sources.

    Returns
    -------
    PolarsDF
        DataFrame containing price history and return columns.
    """
    prices = fetch_asset_history(config, fetcher=fetcher)
    returns = compute_returns(prices)

    if output_path is not None:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        returns.write_parquet(dest)

    return returns


def _normalize_prices(raw: pd.DataFrame, symbols: Iterable[str]) -> pd.DataFrame:
    """Convert a multi-index DataFrame into tidy long-form price data."""
    symbol_tuple = tuple(symbols)
    if isinstance(raw.columns, pd.MultiIndex):
        index_name = raw.index.name or "timestamp"
        wide = raw.copy()
        wide.index = wide.index.set_names(index_name)
        flattened_names = [f"{col[1]}__{col[0]}" for col in wide.columns]
        wide.columns = flattened_names
        wide = wide.reset_index()
        melted = wide.melt(id_vars=[index_name], var_name="symbol_field", value_name="value")
        symbol_field = melted["symbol_field"].str.split("__", n=1, expand=True)
        melted["symbol"] = symbol_field[0]
        melted["field"] = symbol_field[1]
        pivoted = (
            melted
            .pivot(index=[index_name, "symbol"], columns="field", values="value")
            .reset_index()
        )
        pivoted.columns = [str(col) for col in pivoted.columns]
        pivoted = pivoted.rename(columns={index_name: "timestamp"})
    else:
        if len(symbol_tuple) != 1:
            msg = (
                "Expected single symbol for single-level yfinance frame; got "
                f"{len(symbol_tuple)} symbols"
            )
            raise YFinanceIngestError(msg)
        single_symbol = symbol_tuple[0]
        rename_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
        normalized = raw.rename(columns=rename_map).copy()
        if "timestamp" in normalized.columns:
            frame = normalized.reset_index(drop=True)
            timestamp_series = frame.pop("timestamp")
            if isinstance(timestamp_series, pd.Series):
                timestamp_values = timestamp_series.reset_index(drop=True)
            else:
                timestamp_values = pd.Series(timestamp_series)
        else:
            frame = normalized.reset_index()
            index_column = frame.columns[0]
            timestamp_series = frame.pop(index_column)
            if isinstance(timestamp_series, pd.Series):
                timestamp_values = timestamp_series.reset_index(drop=True)
            elif isinstance(timestamp_series, pd.Index):
                timestamp_values = timestamp_series.to_series(index=False)
            else:
                timestamp_values = pd.Series(timestamp_series)
        frame.insert(0, "timestamp", timestamp_values)
        frame.insert(1, "symbol", single_symbol)
        pivoted = frame

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    pivoted = pivoted.rename(columns={old: new for old, new in rename_map.items() if old in pivoted.columns})

    required = {"timestamp", "symbol"}
    if not required.issubset(pivoted.columns):
        missing = required - set(pivoted.columns)
        msg = f"Normalized price frame missing columns: {sorted(missing)}"
        raise YFinanceIngestError(msg)

    for column in ("open", "high", "low", "close", "adj_close", "volume"):
        if column not in pivoted.columns:
            pivoted[column] = pd.NA

    ordered = pivoted[[
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]]
    ordered["volume"] = ordered["volume"].astype("float64")
    return ordered


def _localize_timestamp(values: Iterable[object]) -> pd.Series:
    array = np.asarray(list(values))
    timestamps = pd.to_datetime(array)
    if getattr(timestamps, "tz", None) is None:
        localized = timestamps.tz_localize(UTC)
    else:
        localized = timestamps.tz_convert(UTC)
    return pd.Series(localized)
