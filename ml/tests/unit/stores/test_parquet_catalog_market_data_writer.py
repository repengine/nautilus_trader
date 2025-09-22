#!/usr/bin/env python3

from __future__ import annotations

import pandas as pd

from ml.stores.writers import ParquetCatalogMarketDataWriter


class _FakeCatalog:
    def __init__(self) -> None:
        self.items: list[object] = []

    def write_data(self, items: list[object]) -> None:
        self.items.extend(items)


def test_parquet_catalog_market_data_writer_maps_rows_to_bars() -> None:
    cat = _FakeCatalog()
    writer = ParquetCatalogMarketDataWriter(cat)

    df = pd.DataFrame(
        {
            "instrument_id": ["SPY.NYSE", "SPY.NYSE"],
            "ts_event": [1, 2],
            "ts_init": [11, 12],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.5, 100.5],
            "close": [100.5, 101.5],
            "volume": [1000.0, 1100.0],
        },
    )
    n = writer.write(dataset_id="bars_ds", schema="bars", instrument_id="SPY.NYSE", df=df)
    assert n == 2
    assert len(cat.items) == 2


def test_parquet_catalog_market_data_writer_uses_manifest_template() -> None:
    cat = _FakeCatalog()

    class _Manifest:
        def __init__(self) -> None:
            self.metadata = {"bar_type_template": "{instrument_id}-5-MINUTE-LAST-EXTERNAL"}

    manifest = _Manifest()
    writer = ParquetCatalogMarketDataWriter(cat, manifest_resolver=lambda _: manifest)

    df = pd.DataFrame(
        {
            "instrument_id": ["SPY.XNYS"],
            "ts_event": [1],
            "ts_init": [1],
            "open": [100.0],
            "high": [101.0],
            "low": [99.5],
            "close": [100.5],
            "volume": [1000.0],
        },
    )
    writer.write(dataset_id="bars_ds", schema="bars", instrument_id="SPY.XNYS", df=df)
    assert cat.items
    assert str(cat.items[0].bar_type) == "SPY.XNYS-5-MINUTE-LAST-EXTERNAL"
