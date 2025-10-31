"""Data fetchers for the 3D sector risk model."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast

import pandas as pd
import polars as pl
import structlog

from ml._imports import fredapi as _fredapi
from ml.common.metrics_manager import MetricsManager
from ml.data.ingest.yfinance_adapter import PriceFetcherProtocol
from ml.data.ingest.yfinance_adapter import YFinanceIngestConfig
from ml.data.ingest.yfinance_adapter import compute_returns
from ml.data.ingest.yfinance_adapter import fetch_asset_history
from playground.risk_model.dataset import FactorDataRequest
from playground.risk_model.dataset import FactorReturnFetcher
from playground.risk_model.dataset import SectorDataRequest
from playground.risk_model.dataset import SectorReturnFetcher


LOGGER = structlog.get_logger(__name__)


class PriceFetcherFactory(Protocol):
    """Protocol describing a callable returning a price fetcher for testing."""

    def __call__(self) -> PriceFetcherProtocol: ...


@dataclass(frozen=True)
class SectorFetcherConfig:
    """Configuration for sector return fetching."""

    auto_adjust: bool = True
    threads: bool | int | None = True
    progress: bool = False
    cache_path: Path | None = None
    ticker_overrides: Mapping[str, tuple[str, ...]] | None = None
    min_coverage_ratio: float = 0.85


class YFinanceSectorFetcher(SectorReturnFetcher):
    """Fetch sector returns via the shared yfinance adapter."""

    def __init__(
        self,
        *,
        price_fetcher_factory: PriceFetcherFactory | None = None,
        config: SectorFetcherConfig | None = None,
    ) -> None:
        self._price_fetcher_factory = price_fetcher_factory
        self._config = config or SectorFetcherConfig()

    def __call__(self, request: SectorDataRequest) -> pl.DataFrame:
        fetcher: PriceFetcherProtocol | None = None
        if self._price_fetcher_factory is not None:
            fetcher = self._price_fetcher_factory()

        ticker_map = _build_ticker_map(request.sectors, self._config.ticker_overrides)
        ingest_config = YFinanceIngestConfig(
            symbols=tuple(sorted({ticker for tickers in ticker_map.values() for ticker in tickers})),
            start=request.start,
            end=request.end,
            interval=request.frequency,
            auto_adjust=self._config.auto_adjust,
            threads=self._config.threads,
            progress=self._config.progress,
        )
        try:
            prices = fetch_asset_history(ingest_config, fetcher=fetcher)
        except ImportError as exc:
            message = (
                "yfinance is required to fetch sector prices. Install with `pip install yfinance` "
                "or provide a custom price fetcher."
            )
            raise RuntimeError(message) from exc
        metrics = MetricsManager.default()
        expected_days = max(
            1,
            int(pd.date_range(start=request.start.date(), end=request.end.date(), freq="B").size),
        )
        price_column = request.price_column
        if price_column not in prices.columns:
            fallback = None
            if "adj_close" in prices.columns:
                fallback = "adj_close"
            elif "close" in prices.columns:
                fallback = "close"
            if fallback is None:
                available = ", ".join(sorted(prices.columns))
                msg = (
                    f"Requested price column '{price_column}' not available in fetched data. "
                    f"Available columns: {available}"
                )
                raise ValueError(msg)
            LOGGER.warning(
                "price_column_missing_falling_back",
                requested=price_column,
                fallback=fallback,
            )
            price_column = fallback

        coverage_counts = _coverage_by_ticker(prices)

        selected = _select_tickers(
            ticker_map,
            coverage_counts,
            expected_days=expected_days,
            min_ratio=self._config.min_coverage_ratio,
        )
        for sector, selection in selected.items():
            metrics.observe(
                "playground_sector_proxy_coverage",
                "Coverage ratio for chosen sector proxy ticker",
                selection.coverage_ratio,
                labels={"sector": sector, "ticker": selection.ticker},
            )
            if selection.coverage_ratio < self._config.min_coverage_ratio:
                LOGGER.warning(
                    "Sector coverage below threshold",
                    sector=sector,
                    ticker=selection.ticker,
                    ratio=selection.coverage_ratio,
                )

        returns = compute_returns(prices, price_column=price_column)
        selected_tickers = tuple(choice.ticker for choice in selected.values())
        mapping = {choice.ticker: sector for sector, choice in selected.items()}
        frame = (
            returns
            .filter(pl.col("symbol").is_in(selected_tickers))
            .with_columns(
                pl.col("symbol").replace(mapping).alias("sector_symbol"),
            )
            .drop_nulls(subset=["sector_symbol", "return"])
            .with_columns(pl.col("sector_symbol").alias("symbol"))
            .drop("sector_symbol")
            .select("timestamp", "symbol", "return")
            .sort(["timestamp", "symbol"])
        )

        if self._config.cache_path is not None:
            self._config.cache_path.parent.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(self._config.cache_path)

        return frame


class _SeriesBuilder(Protocol):
    def build(self, fred: _fredapi.Fred, index: pd.DatetimeIndex) -> pd.Series: ...


@dataclass(frozen=True)
class _SingleSeriesBuilder(_SeriesBuilder):
    series_id: str

    def build(self, fred: _fredapi.Fred, index: pd.DatetimeIndex) -> pd.Series:
        raw = fred.get_series(series_id=self.series_id)
        series = pd.Series(raw)
        series.index = pd.to_datetime(series.index).tz_localize(UTC)
        aligned = series.reindex(index).ffill()
        return cast(pd.Series[Any], aligned)


@dataclass(frozen=True)
class _SpreadSeriesBuilder(_SeriesBuilder):
    minuend_series: str
    subtrahend_series: str

    def build(self, fred: _fredapi.Fred, index: pd.DatetimeIndex) -> pd.Series:
        minuend = _SingleSeriesBuilder(self.minuend_series).build(fred, index)
        subtrahend = _SingleSeriesBuilder(self.subtrahend_series).build(fred, index)
        spread = minuend - subtrahend
        return spread


@dataclass(slots=True)
class FactorFetcherConfig:
    """Configuration describing FRED-backed factor series."""

    fred_api_key: str | None = None
    factor_series: Mapping[str, _SeriesBuilder] | None = None

    def resolve_series(self, factor_columns: tuple[str, ...]) -> Mapping[str, _SeriesBuilder]:
        mapping = self.factor_series
        if mapping is None:
            mapping = {
                "factor_duration": _SpreadSeriesBuilder("DGS10", "DGS2"),
                "factor_credit": _SingleSeriesBuilder("BAMLC0A0CM"),
                "factor_liquidity": _SingleSeriesBuilder("STLFSI2"),
            }
        missing = [name for name in factor_columns if name not in mapping]
        if missing:
            msg = f"Factor mapping missing definitions for: {missing}"
            raise ValueError(msg)
        return {name: mapping[name] for name in factor_columns}


class FREDFactorFetcher(FactorReturnFetcher):
    """Fetch factor level time series using FRED API."""

    def __init__(self, config: FactorFetcherConfig | None = None) -> None:
        self._config = config or FactorFetcherConfig()

    def __call__(self, request: FactorDataRequest) -> pl.DataFrame:
        if _fredapi is None:
            raise RuntimeError("fredapi dependency is not available")

        api_key = self._config.fred_api_key or os.getenv("FRED_API_KEY")
        if not api_key:
            raise ValueError("FRED_API_KEY environment variable must be set for factor fetching")

        start = request.start.astimezone(UTC)
        end = request.end.astimezone(UTC)
        index = pd.date_range(start=start, end=end, freq="D", tz=UTC)

        fred = _fredapi.Fred(api_key=api_key)
        series_map = self._config.resolve_series(request.factor_columns)

        data = pd.DataFrame(index=index)
        for column, builder in series_map.items():
            try:
                series = builder.build(fred, index)
            except Exception:  # pragma: no cover - dependency failure path
                LOGGER.exception("Failed to fetch FRED series", column=column)
                raise
            data[column] = series

        data = data.dropna(how="all").reset_index(names="timestamp")
        frame = pl.from_pandas(
            data,
            schema_overrides={"timestamp": pl.Datetime("us", "UTC")},
        )

        metrics = MetricsManager.default()
        expected_days = max(
            1,
            int(pd.date_range(start=request.start.date(), end=request.end.date(), freq="B").size),
        )
        for column in series_map:
            observed = int(
                frame.select(
                    pl.col(column).is_not_null().sum().alias("observed"),
                ).get_column("observed")[0]
            )
            ratio = float(observed) / float(expected_days)
            ratio = max(0.0, min(1.0, ratio))
            metrics.observe(
                "playground_factor_fetch_coverage",
                "Coverage ratio for raw factor levels",
                ratio,
                labels={"factor": column},
            )
            if ratio < 0.8:
                LOGGER.warning(
                    "Factor coverage below threshold",
                    factor=column,
                    ratio=ratio,
                )
        return frame


class CachedFactorFetcher(FactorReturnFetcher):
    """Wrap another factor fetcher with parquet caching on disk."""

    def __init__(self, inner: FactorReturnFetcher, cache_path: Path) -> None:
        self._inner = inner
        self._cache_path = cache_path

    def __call__(self, request: FactorDataRequest) -> pl.DataFrame:
        cached: pl.DataFrame | None = None
        if self._cache_path.exists():
            cached = pl.read_parquet(self._cache_path)
            if _covers_range(cached, request.start, request.end):
                cached_start, cached_end = _timestamp_bounds(cached)
                LOGGER.debug(
                    "Using cached factor data",
                    path=str(self._cache_path),
                    cached_start=str(cached_start),
                    cached_end=str(cached_end),
                )
                return cached.filter(
                    (pl.col("timestamp") >= request.start) & (pl.col("timestamp") <= request.end),
                )
            LOGGER.info(
                "Extending factor cache for broader request range",
                path=str(self._cache_path),
                request_start=str(request.start),
                request_end=str(request.end),
            )

        fresh = self._inner(request)

        combined = _merge_factor_frames(cached, fresh)
        filtered = combined.filter(
            (pl.col("timestamp") >= request.start) & (pl.col("timestamp") <= request.end),
        )

        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        combined.write_parquet(self._cache_path)
        return filtered


def build_sector_fetcher(config: SectorFetcherConfig | None = None) -> SectorReturnFetcher:
    """Create a yfinance-backed sector fetcher."""
    return YFinanceSectorFetcher(config=config)


def build_factor_fetcher(config: FactorFetcherConfig | None = None, *, cache_path: Path | None = None) -> FactorReturnFetcher:
    """Create a FRED-backed factor fetcher with optional caching."""
    base = FREDFactorFetcher(config=config)
    if cache_path is not None:
        return CachedFactorFetcher(base, cache_path)
    return base


__all__ = [
    "FREDFactorFetcher",
    "FactorFetcherConfig",
    "ProxySelection",
    "SectorFetcherConfig",
    "YFinanceSectorFetcher",
    "build_factor_fetcher",
    "build_sector_fetcher",
]


@dataclass(slots=True)
class ProxySelection:
    """Chosen proxy ticker and its coverage ratio."""

    ticker: str
    coverage_ratio: float


def _build_ticker_map(
    sectors: tuple[str, ...],
    overrides: Mapping[str, tuple[str, ...]] | None,
) -> dict[str, tuple[str, ...]]:
    mapping: dict[str, tuple[str, ...]] = {}
    for sector in sectors:
        candidates = overrides.get(sector) if overrides is not None else None
        mapping[sector] = candidates or (sector,)
    return mapping


def _coverage_by_ticker(prices: pl.DataFrame) -> dict[str, int]:
    counts = (
        prices
        .group_by("symbol")
        .agg(pl.count().alias("count"))
    )
    return {str(row["symbol"]): int(row["count"]) for row in counts.iter_rows(named=True)}


def _select_tickers(
    ticker_map: Mapping[str, tuple[str, ...]],
    coverage_counts: Mapping[str, int],
    *,
    expected_days: int,
    min_ratio: float,
) -> dict[str, ProxySelection]:
    selections: dict[str, ProxySelection] = {}
    divisor = max(expected_days, 1)
    for sector, candidates in ticker_map.items():
        if not candidates:
            raise ValueError(f"No ticker candidates provided for sector '{sector}'")

        best_ticker = candidates[0]
        best_ratio = float(coverage_counts.get(best_ticker, 0)) / float(divisor)

        for ticker in candidates:
            observed = coverage_counts.get(ticker, 0)
            ratio = float(observed) / float(divisor)
            ratio = max(0.0, min(1.0, ratio))
            if ratio >= min_ratio:
                best_ticker = ticker
                best_ratio = ratio
                break
            if ratio > best_ratio:
                best_ticker = ticker
                best_ratio = ratio

        selections[sector] = ProxySelection(ticker=best_ticker, coverage_ratio=best_ratio)

    return selections


def _covers_range(frame: pl.DataFrame | None, start: datetime, end: datetime) -> bool:
    if frame is None or frame.is_empty():
        return False
    min_ts, max_ts = _timestamp_bounds(frame)
    return min_ts <= start and max_ts >= end


def _timestamp_bounds(frame: pl.DataFrame) -> tuple[datetime, datetime]:
    start = frame.select(pl.col("timestamp").min()).item()
    end = frame.select(pl.col("timestamp").max()).item()
    return start, end


def _merge_factor_frames(
    cached: pl.DataFrame | None,
    fresh: pl.DataFrame,
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    if cached is not None and not cached.is_empty():
        frames.append(cached)
    if not fresh.is_empty():
        frames.append(fresh)
    if not frames:
        return pl.DataFrame()
    combined = pl.concat(frames, how="vertical")
    combined = (
        combined
        .unique(subset=["timestamp"], keep="last")
        .sort("timestamp")
    )
    return combined
