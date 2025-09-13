#!/usr/bin/env python3

from __future__ import annotations

import pandas as pd

from ml.stores.market_data_writer import ParquetCatalogMarketDataWriter


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
