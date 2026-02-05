from pathlib import Path
from typing import Any

import pytest
from nautilus_trader.model.data import QuoteTick as _QuoteTick
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
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


def test_parquet_raw_writer_replaces_overlaps(tmp_path: Path) -> None:
    catalog_root = tmp_path / "catalog"
    catalog = ParquetDataCatalog(str(catalog_root))
    writer = ParquetCatalogRawWriter(catalog, replace_on_overlap=True)

    first = [
        {
            "instrument_id": "SPY.EQUS",
            "ts_event": 1,
            "ts_init": 1,
            "bid": 1.0,
            "ask": 1.1,
            "bid_size": 10.0,
            "ask_size": 20.0,
        },
        {
            "instrument_id": "SPY.EQUS",
            "ts_event": 2,
            "ts_init": 2,
            "bid": 1.2,
            "ask": 1.3,
            "bid_size": 11.0,
            "ask_size": 21.0,
        },
    ]
    second = [
        {
            "instrument_id": "SPY.EQUS",
            "ts_event": 2,
            "ts_init": 2,
            "bid": 1.25,
            "ask": 1.35,
            "bid_size": 12.0,
            "ask_size": 22.0,
        },
        {
            "instrument_id": "SPY.EQUS",
            "ts_event": 3,
            "ts_init": 3,
            "bid": 1.26,
            "ask": 1.36,
            "bid_size": 13.0,
            "ask_size": 23.0,
        },
    ]

    writer.write(dataset_type=DatasetType.QUOTES, data=first)
    writer.write(dataset_type=DatasetType.QUOTES, data=second)

    intervals = catalog.get_intervals(_QuoteTick, "SPY.EQUS")
    assert intervals == [(2, 3)]

    files = list((catalog_root / "data" / "quote_tick" / "SPY.EQUS").glob("*.parquet"))
    assert len(files) == 1


def test_parquet_raw_writer_uses_dataset_type_identifier_template(tmp_path: Path) -> None:
    catalog_root = tmp_path / "catalog"
    catalog = ParquetDataCatalog(str(catalog_root))
    writer = ParquetCatalogRawWriter(
        catalog,
        dataset_type_identifier_templates={DatasetType.MBP1: "{instrument_id}-MBP1"},
    )

    quote_rows = [
        {
            "instrument_id": "SPY.EQUS",
            "ts_event": 1,
            "ts_init": 1,
            "bid": 1.0,
            "ask": 1.1,
            "bid_size": 10.0,
            "ask_size": 20.0,
        },
    ]

    count = writer.write(dataset_type=DatasetType.MBP1, data=quote_rows)
    assert count == 1

    intervals = catalog.get_intervals(_QuoteTick, "SPY.EQUS-MBP1")
    assert intervals
    default_dir = catalog_root / "data" / "quote_tick" / "SPY.EQUS"
    assert not list(default_dir.glob("*.parquet"))


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


def test_parquet_raw_writer_rejects_tabular_mbp10() -> None:
    catalog = _FakeCatalog()
    writer = ParquetCatalogRawWriter(catalog)

    with pytest.raises(ValueError, match="mbp10"):
        writer.write(
            dataset_type=DatasetType.MBP10,
            data=[{"instrument_id": "SPY.EQUS", "ts_event": 1, "ts_init": 1}],
        )


def test_parquet_raw_writer_rejects_tabular_mbo() -> None:
    catalog = _FakeCatalog()
    writer = ParquetCatalogRawWriter(catalog)

    with pytest.raises(ValueError, match="mbo"):
        writer.write(
            dataset_type=DatasetType.MBO,
            data=[{"instrument_id": "SPY.EQUS", "ts_event": 1, "ts_init": 1}],
        )
