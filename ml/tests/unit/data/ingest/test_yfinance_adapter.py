"""Tests for the yfinance ingestion adapter."""

from __future__ import annotations

import importlib
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import pandas as pd
import polars as pl
import pytest

from ml.data.ingest import yfinance_adapter as adapter
from ml.data.ingest.yfinance_adapter import (
    YFinancePriceFetcher,
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


def test_config_validation_rejects_empty_symbols_and_invalid_end() -> None:
    with pytest.raises(ValueError, match="At least one symbol"):
        YFinanceIngestConfig(symbols=(), start=datetime(2024, 1, 1, tzinfo=UTC))

    with pytest.raises(ValueError, match="End timestamp must be after start timestamp"):
        YFinanceIngestConfig(
            symbols=("SPY",),
            start=datetime(2024, 1, 2, tzinfo=UTC),
            end=datetime(2024, 1, 1, tzinfo=UTC),
        )


def test_ensure_yfinance_uses_cache_and_raises_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel_module = SimpleNamespace(download=lambda **_: pd.DataFrame({"Close": [1.0]}))
    monkeypatch.setattr(adapter, "_YFINANCE_MODULE", sentinel_module)
    assert adapter._ensure_yfinance() is sentinel_module

    monkeypatch.setattr(adapter, "_YFINANCE_MODULE", None)

    def _raise(_name: str) -> Any:
        raise ImportError("missing")

    monkeypatch.setattr(importlib, "import_module", _raise)

    with pytest.raises(ImportError, match="yfinance is required"):
        adapter._ensure_yfinance()


def test_ensure_yfinance_import_success_sets_module_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    imported = SimpleNamespace(download=lambda **_: pd.DataFrame({"Close": [1.0]}))

    monkeypatch.setattr(adapter, "_YFINANCE_MODULE", None)
    monkeypatch.setattr(importlib, "import_module", lambda _name: imported)

    module = adapter._ensure_yfinance()

    assert module is imported
    assert adapter._YFINANCE_MODULE is imported


def test_price_fetcher_wraps_download_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    broken_module = SimpleNamespace(download=lambda **_: (_ for _ in ()).throw(RuntimeError("network")))
    monkeypatch.setattr(adapter, "_YFINANCE_MODULE", broken_module)
    fetcher = YFinancePriceFetcher()

    with pytest.raises(YFinanceIngestError, match="download failed"):
        fetcher.fetch(
            YFinanceIngestConfig(
                symbols=("SPY",),
                start=datetime(2024, 1, 1, tzinfo=UTC),
            ),
        )


def test_price_fetcher_rejects_empty_download(monkeypatch: pytest.MonkeyPatch) -> None:
    empty_module = SimpleNamespace(download=lambda **_: pd.DataFrame())
    monkeypatch.setattr(adapter, "_YFINANCE_MODULE", empty_module)
    fetcher = YFinancePriceFetcher()

    with pytest.raises(YFinanceIngestError, match="returned no data"):
        fetcher.fetch(
            YFinanceIngestConfig(
                symbols=("SPY",),
                start=datetime(2024, 1, 1, tzinfo=UTC),
            ),
        )


def test_price_fetcher_returns_dataframe_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = pd.DataFrame({"Close": [100.0, 101.0]})
    module = SimpleNamespace(download=lambda **_: expected)
    monkeypatch.setattr(adapter, "_YFINANCE_MODULE", module)

    fetcher = YFinancePriceFetcher()
    frame = fetcher.fetch(
        YFinanceIngestConfig(
            symbols=("SPY",),
            start=datetime(2024, 1, 1, tzinfo=UTC),
        ),
    )

    assert frame.equals(expected)


def test_normalize_prices_raises_for_single_level_multi_symbol_frame() -> None:
    frame = pd.DataFrame({"Close": [10.0]})

    with pytest.raises(YFinanceIngestError, match="Expected single symbol"):
        adapter._normalize_prices(frame, symbols=("SPY", "QQQ"))


def test_normalize_prices_handles_existing_timestamp_column() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)],
            "Open": [10.0, 11.0],
            "High": [10.5, 11.5],
            "Low": [9.5, 10.5],
            "Close": [10.2, 11.1],
            "Adj Close": [10.1, 11.0],
            "Volume": [1000.0, 1200.0],
        },
    )

    normalized = adapter._normalize_prices(frame, symbols=("SPY",))

    assert normalized.columns.tolist() == [
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]
    assert normalized["symbol"].tolist() == ["SPY", "SPY"]


def test_normalize_prices_coerces_non_series_timestamp_and_fills_missing_columns() -> None:
    class _ListTimestampFrame(pd.DataFrame):
        @property
        def _constructor(self) -> type[_ListTimestampFrame]:
            return _ListTimestampFrame

        def pop(self, item: str) -> Any:
            value = super().pop(item)
            if item == "timestamp":
                return value.to_list()
            return value

    frame = _ListTimestampFrame(
        {
            "timestamp": [datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)],
            "Adj Close": [10.0, 11.0],
            "Volume": [100.0, 101.0],
        },
    )

    normalized = adapter._normalize_prices(frame, symbols=("SPY",))

    assert normalized["symbol"].tolist() == ["SPY", "SPY"]
    assert normalized.columns.tolist() == [
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]
    assert normalized["adj_close"].tolist() == [10.0, 11.0]
    assert normalized["volume"].tolist() == [100.0, 101.0]
    assert normalized["open"].isna().all()


def test_localize_timestamp_handles_naive_and_aware_values() -> None:
    naive = adapter._localize_timestamp([datetime(2024, 1, 1), datetime(2024, 1, 2)])
    aware = adapter._localize_timestamp(
        [datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)],
    )

    assert str(naive.dt.tz) == "UTC"
    assert str(aware.dt.tz) == "UTC"


def test_ingest_asset_returns_without_output_path(
    config: YFinanceIngestConfig,
    stub_fetcher: _StubFetcher,
) -> None:
    returns = ingest_asset_returns(config, output_path=None, fetcher=stub_fetcher)

    assert "return" in returns.columns
    assert "log_return" in returns.columns
