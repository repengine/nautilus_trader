"""Tests for the yfinance ingestion adapter."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import pytest

from ml.data.ingest.yfinance_adapter import (
    YFinanceIngestConfig,
    YFinanceIngestError,
    fetch_asset_history,
    compute_returns,
    ingest_asset_returns,
)


class _StubFetcher:
    def __init__(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="D", tz=UTC)
        data = {
            ("Adj Close", "SPY"): [100.0, 101.0, 102.0],
            ("Adj Close", "TLT"): [200.0, 198.0, 202.0],
            ("Close", "SPY"): [100.0, 101.0, 102.0],
            ("Close", "TLT"): [200.0, 198.0, 202.0],
            ("High", "SPY"): [101.0, 102.0, 103.0],
            ("High", "TLT"): [201.0, 199.0, 203.0],
            ("Low", "SPY"): [99.0, 100.0, 101.0],
            ("Low", "TLT"): [199.0, 197.0, 201.0],
            ("Open", "SPY"): [100.0, 101.0, 102.0],
            ("Open", "TLT"): [200.0, 198.0, 202.0],
            ("Volume", "SPY"): [1_000_000.0, 1_100_000.0, 1_050_000.0],
            ("Volume", "TLT"): [2_000_000.0, 1_900_000.0, 2_050_000.0],
        }
        self.frame = pd.DataFrame(data, index=dates)
        self.calls: list[YFinanceIngestConfig] = []

    def fetch(self, config: YFinanceIngestConfig) -> pd.DataFrame:
        self.calls.append(config)
        return self.frame


@pytest.fixture()
def stub_fetcher() -> _StubFetcher:
    return _StubFetcher()


@pytest.fixture()
def config() -> YFinanceIngestConfig:
    return YFinanceIngestConfig(
        symbols=("SPY", "TLT"),
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 4, tzinfo=UTC),
    )


def test_fetch_asset_history_returns_tidy_frame(config: YFinanceIngestConfig, stub_fetcher: _StubFetcher) -> None:
    frame = fetch_asset_history(config, fetcher=stub_fetcher)

    assert frame.height == 6
    assert frame.columns == ["timestamp", "symbol", "open", "high", "low", "close", "adj_close", "volume"]

    spy_first = frame.filter(pl.col("symbol") == "SPY").row(0)
    assert spy_first[2] == pytest.approx(100.0)
    assert isinstance(spy_first[0], datetime)
    assert spy_first[0].tzinfo is not None
    assert spy_first[0].tzinfo.tzname(spy_first[0]) == "UTC"


def test_compute_returns_produces_expected_values(config: YFinanceIngestConfig, stub_fetcher: _StubFetcher) -> None:
    prices = fetch_asset_history(config, fetcher=stub_fetcher)
    enriched = compute_returns(prices)

    spy_returns = enriched.filter(pl.col("symbol") == "SPY").select("return").to_series().to_list()
    assert spy_returns[0] is None
    assert spy_returns[1] == pytest.approx(0.01)
    assert spy_returns[2] == pytest.approx(0.00990099, rel=1e-6)


def test_ingest_asset_returns_writes_parquet(
    tmp_path: Path,
    config: YFinanceIngestConfig,
    stub_fetcher: _StubFetcher,
) -> None:
    output_file = tmp_path / "yfinance" / "asset_returns.parquet"
    returns = ingest_asset_returns(config, output_file, fetcher=stub_fetcher)

    assert output_file.exists()
    reloaded = pl.read_parquet(output_file)
    assert reloaded.height == returns.height


def test_compute_returns_missing_column_raises(config: YFinanceIngestConfig, stub_fetcher: _StubFetcher) -> None:
    prices = fetch_asset_history(config, fetcher=stub_fetcher)
    with pytest.raises(YFinanceIngestError):
        compute_returns(prices, price_column="missing")


def test_single_symbol_frame_normalizes(config: YFinanceIngestConfig) -> None:
    dates = pd.date_range("2024-01-01", periods=2, freq="D", tz=UTC)
    frame = pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "High": [10.5, 11.5],
            "Low": [9.5, 10.5],
            "Close": [10.2, 11.1],
            "Adj Close": [10.2, 11.1],
            "Volume": [1_000.0, 1_100.0],
        },
        index=dates,
    )

    class _SingleFetcher:
        def fetch(self, _config: YFinanceIngestConfig) -> pd.DataFrame:
            return frame

    single_config = YFinanceIngestConfig(symbols=("GLD",), start=config.start, end=config.end)
    prices = fetch_asset_history(single_config, fetcher=_SingleFetcher())
    assert prices.filter(pl.col("symbol") == "GLD").height == 2
    assert prices.columns == ["timestamp", "symbol", "open", "high", "low", "close", "adj_close", "volume"]
