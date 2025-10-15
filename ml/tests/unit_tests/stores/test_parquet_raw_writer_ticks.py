from typing import Any

from ml.registry.dataclasses import DatasetType
from ml.stores.io_raw import ParquetCatalogRawWriter


class _FakeCatalog:
    def __init__(self) -> None:
        self.items: list[Any] | None = None

    def write_data(self, items: list[Any]) -> None:
        self.items = items


def test_parquet_raw_writer_converts_quotes_to_domain() -> None:
    catalog = _FakeCatalog()
    writer = ParquetCatalogRawWriter(catalog)

    quote_rows = [
        {
            "instrument_id": "SPY.EQUS",
            "ts_event": 1_000_000,
            "ts_init": 1_000_000,
            "bid": 1.0,
            "ask": 1.1,
            "bid_size": 100.0,
            "ask_size": 200.0,
        },
    ]
    count = writer.write(dataset_type=DatasetType.QUOTES, data=quote_rows)
    assert count == 1
    assert catalog.items is not None
    from nautilus_trader.model.data import QuoteTick as _QuoteTick

    assert isinstance(catalog.items[0], _QuoteTick)


def test_parquet_raw_writer_converts_trades_to_domain() -> None:
    catalog = _FakeCatalog()
    writer = ParquetCatalogRawWriter(catalog)

    trade_rows = [
        {
            "instrument_id": "SPY.EQUS",
            "ts_event": 2_000_000,
            "ts_init": 2_000_000,
            "price": 2.0,
            "size": 5.0,
            "aggressor_side": "UNKNOWN",
            "trade_id": "T1",
        },
    ]
    count = writer.write(dataset_type=DatasetType.TRADES, data=trade_rows)
    assert count == 1
    assert catalog.items is not None
    from nautilus_trader.model.data import TradeTick as _TradeTick

    assert isinstance(catalog.items[0], _TradeTick)
