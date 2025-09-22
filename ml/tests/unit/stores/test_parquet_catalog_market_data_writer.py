#!/usr/bin/env python3

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from ml.stores.writers import FanoutMarketDataWriter
from ml.stores.writers import ParquetCatalogMarketDataWriter
from ml.registry.dataclasses import DatasetManifest, DatasetType, StorageKind


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
    manifest = DatasetManifest(
        dataset_id="bars_ds",
        dataset_type=DatasetType.BARS,
        storage_kind=StorageKind.PARQUET,
        location="/tmp",
        partitioning={"by": "ts_event", "interval": "daily"},
        retention_days=90,
        schema={"instrument_id": "str", "ts_event": "int64", "ts_init": "int64"},
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["instrument_id", "ts_event"],
        schema_hash="hash",
        constraints={},
        lineage=[],
        pipeline_signature="sig",
        version="1.0.0",
        created_at=0,
        last_modified=0,
        metadata={"bar_type_template": "{instrument_id}-5-MINUTE-LAST-EXTERNAL"},
    )
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
    bar = cast(Any, cat.items[0])
    assert str(bar.bar_type) == "SPY.XNYS-5-MINUTE-LAST-EXTERNAL"


def test_fanout_market_data_writer_calls_primary_and_mirror() -> None:
    calls: list[str] = []

    class _Primary:
        def write(
            self,
            *,
            dataset_id: str,
            schema: str,
            instrument_id: str,
            df: pd.DataFrame,
        ) -> int:
            calls.append(f"primary:{dataset_id}:{schema}:{instrument_id}")
            return len(df.index)

    class _Mirror:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def write(
            self,
            *,
            dataset_id: str,
            schema: str,
            instrument_id: str,
            df: pd.DataFrame,
        ) -> int:
            self.calls.append(f"mirror:{dataset_id}:{schema}:{instrument_id}")
            return len(df.index)

    primary = _Primary()
    mirror = _Mirror()
    writer = FanoutMarketDataWriter(primary=primary, mirrors=(mirror,))
    frame = pd.DataFrame({"ts_event": [1], "ts_init": [1]})

    written = writer.write(
        dataset_id="bars_ds",
        schema="bars",
        instrument_id="SPY.NYSE",
        df=frame,
    )

    assert written == 1
    assert calls == ["primary:bars_ds:bars:SPY.NYSE"]
    assert mirror.calls == ["mirror:bars_ds:bars:SPY.NYSE"]


def test_fanout_market_data_writer_swallows_mirror_failure(caplog: Any) -> None:
    class _Primary:
        def write(
            self,
            *,
            dataset_id: str,
            schema: str,
            instrument_id: str,
            df: pd.DataFrame,
        ) -> int:
            return len(df.index)

    class _Mirror:
        def write(
            self,
            *,
            dataset_id: str,
            schema: str,
            instrument_id: str,
            df: pd.DataFrame,
        ) -> int:
            raise RuntimeError("mirror failure")

    writer = FanoutMarketDataWriter(primary=_Primary(), mirrors=(_Mirror(),))
    frame = pd.DataFrame({"ts_event": [1], "ts_init": [1]})

    with caplog.at_level("WARNING"):
        written = writer.write(
            dataset_id="bars_ds",
            schema="bars",
            instrument_id="SPY.NYSE",
            df=frame,
        )

    assert written == 1
    assert any("Mirror market data write failed" in rec.message for rec in caplog.records)
