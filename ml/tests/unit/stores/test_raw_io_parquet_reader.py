#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass

from ml._imports import HAS_POLARS, pl
from ml.registry.dataclasses import DatasetType
from ml.stores.io_raw import ParquetCatalogRawReader


@dataclass
class _FakeBarType:
    instrument_id: str


@dataclass
class _FakeBar:
    bar_type: _FakeBarType
    open: float
    high: float
    low: float
    close: float
    volume: float
    ts_event: int


@dataclass
class _FakeQuote:
    instrument_id: str
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    ts_event: int


@dataclass
class _FakeTrade:
    instrument_id: str
    price: float
    size: float
    aggressor_side: str
    ts_event: int


class _FakeCatalog:
    def __init__(self) -> None:
        self._bars: list[_FakeBar] = []
        self._quotes: list[_FakeQuote] = []
        self._trades: list[_FakeTrade] = []

    # Methods with the same names as ParquetDataCatalog
    def bars(self, instrument_ids: list[str], start: int | None, end: int | None) -> list[_FakeBar]:
        return self._bars

    def quote_ticks(
        self,
        instrument_ids: list[str],
        start: int | None,
        end: int | None,
    ) -> list[_FakeQuote]:
        return self._quotes

    def trade_ticks(
        self,
        instrument_ids: list[str],
        start: int | None,
        end: int | None,
    ) -> list[_FakeTrade]:
        return self._trades


def test_parquet_reader_returns_dataframe_for_bars() -> None:
    cat = _FakeCatalog()
    cat._bars.append(
        _FakeBar(
            bar_type=_FakeBarType("SPY.NYSE"),
            open=100.0,
            high=101.0,
            low=99.5,
            close=100.5,
            volume=1_000_000.0,
            ts_event=123,
        ),
    )
    reader = ParquetCatalogRawReader(cat)
    df = reader.read_range(
        dataset_type=DatasetType.BARS,
        instrument_id="SPY.NYSE",
        start_ns=100,
        end_ns=200,
    )
    if HAS_POLARS:
        assert hasattr(df, "columns")
        assert "instrument_id" in df.columns
        assert df.height == 1
    else:
        assert isinstance(df, list)


def test_parquet_reader_returns_dataframe_for_quotes_and_trades() -> None:
    cat = _FakeCatalog()
    cat._quotes.append(
        _FakeQuote(
            instrument_id="SPY.NYSE",
            bid_price=100.0,
            ask_price=100.1,
            bid_size=10.0,
            ask_size=10.0,
            ts_event=123,
        ),
    )
    cat._trades.append(
        _FakeTrade(
            instrument_id="SPY.NYSE",
            price=100.05,
            size=5.0,
            aggressor_side="BUY",
            ts_event=124,
        ),
    )

    reader = ParquetCatalogRawReader(cat)
    qdf = reader.read_range(
        dataset_type=DatasetType.QUOTES,
        instrument_id="SPY.NYSE",
        start_ns=100,
        end_ns=200,
    )
    tdf = reader.read_range(
        dataset_type=DatasetType.TRADES,
        instrument_id="SPY.NYSE",
        start_ns=100,
        end_ns=200,
    )
    if HAS_POLARS:
        assert hasattr(qdf, "columns") and qdf.height == 1
        assert hasattr(tdf, "columns") and tdf.height == 1
    else:
        assert isinstance(qdf, list)
        assert isinstance(tdf, list)
