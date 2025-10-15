from typing import Any

from ml.stores.io_raw import ParquetCatalogRawWriter
from ml.registry.dataclasses import DatasetType


class _FakeCatalog:
    def __init__(self) -> None:
        self.items: list[Any] | None = None

    def write_data(self, items: list[Any]) -> None:
        self.items = items


def test_parquet_raw_writer_converts_rows_to_bars() -> None:
    catalog = _FakeCatalog()
    writer = ParquetCatalogRawWriter(catalog)

    records = [
        {
            "instrument_id": "SPY.EQUS",
            "ts_event": 1_000_000_000,
            "ts_init": 1_000_000_000,
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": 1.5,
            "volume": 100.0,
        },
        {
            "instrument_id": "SPY.EQUS",
            "ts_event": 2_000_000_000,
            "ts_init": 2_000_000_000,
            "open": 1.5,
            "high": 2.5,
            "low": 1.0,
            "close": 2.0,
            "volume": 200.0,
        },
    ]

    count = writer.write(dataset_type=DatasetType.BARS, data=records)
    assert count == 2
    assert catalog.items is not None
    # Domain objects should be Bar instances
    from nautilus_trader.model.data import Bar as _Bar

    written = catalog.items
    assert isinstance(written, list)
    assert isinstance(written[0], _Bar)
    assert isinstance(written[1], _Bar)
