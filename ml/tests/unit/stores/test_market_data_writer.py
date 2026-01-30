#!/usr/bin/env python3

from __future__ import annotations

import time
from pathlib import Path

from typing import Any, cast

import pandas as pd
import pytest
from sqlalchemy import BigInteger
from sqlalchemy import Column
from sqlalchemy import Float
from sqlalchemy import MetaData
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import create_engine
from sqlalchemy import text

from ml.config.events import Source
from ml.registry.dataclasses import (
    DataContract,
    DatasetManifest,
    DatasetType,
    QualityFlag,
    StorageKind,
    ValidationRule,
    ValidationRuleType,
)
from ml.features.earnings.store import DummyEarningsStore
from ml.config.market_data import MarketDataTableConfig
from ml.stores.providers import SqlMarketDataWriter
from ml.stores.writers import DataStoreMarketDataWriter
from ml.stores.io_raw import RawIngestionWriterProtocol


from ml.config.events import EventStatus, Stage
from ml.registry.protocols import RegistryProtocol


class _StubRegistry:
    def __init__(self, manifest: DatasetManifest, contract: DataContract) -> None:
        self.manifest = manifest
        self.contract = contract
        self.events: list[tuple[str, str]] = []
        self.watermarks: list[int] = []

    def emit_event(
        self,
        dataset_id: str,
        instrument_id: str,
        stage: Stage,
        source: Source,
        run_id: str,
        ts_min: int,
        ts_max: int,
        count: int,
        status: EventStatus,
        error: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.events.append((dataset_id, status.value))

    def update_watermark(
        self,
        dataset_id: str,
        instrument_id: str,
        source: Source,
        last_success_ns: int,
        count: int,
        completeness_pct: float,
    ) -> None:
        self.watermarks.append(last_success_ns)

    def get_manifest(self, dataset_id: str) -> DatasetManifest:
        return self.manifest

    def get_contract(self, dataset_id: str) -> DataContract:
        return self.contract

    def register_dataset(self, manifest: DatasetManifest) -> str:  # pragma: no cover - unused
        return manifest.dataset_id


class _FakeRawWriter(RawIngestionWriterProtocol):
    def write(self, *, dataset_type, data):
        return len(data) if hasattr(data, "__len__") else 0


def _manifest(ds: str) -> DatasetManifest:
    return DatasetManifest(
        dataset_id=ds,
        dataset_type=DatasetType.BARS,
        storage_kind=StorageKind.PARQUET,
        location=str(Path("/tmp")),
        partitioning={"by": ["date"]},
        retention_days=30,
        schema={
            "instrument_id": "str",
            "ts_event": "int64",
            "ts_init": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        },
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["instrument_id", "ts_event"],
        schema_hash="",
        constraints={},
        lineage=[],
        pipeline_signature="test",
        version="1.0.0",
        created_at=time.time_ns(),
        last_modified=time.time_ns(),
        metadata={},
    )


def _contract(ds: str) -> DataContract:
    return DataContract(
        contract_id=f"{ds}_contract",
        dataset_id=ds,
        version="1.0.0",
        validation_rules=[
            ValidationRule(
                rule_type=ValidationRuleType.MONOTONICITY,
                field_name="ts_event",
                parameters={"direction": "increasing"},
                severity=QualityFlag.WARN,
                description="ts increasing",
            ),
        ],
        enforcement_mode="lenient",
    )


def test_market_data_writer_uses_datastore_and_emits_success(
    mock_feature_store: Any,
    mock_model_store: Any,
    mock_strategy_store: Any,
    datastore_class: type[Any],
) -> None:
    # Setup DataStore with stub registry and mock stores, plus a fake raw writer
    reg = _StubRegistry(_manifest("bars_ds"), _contract("bars_ds"))
    ds = cast(Any, datastore_class)(
        connection_string="sqlite:///:memory:",
        registry=reg,
        feature_store=mock_feature_store,
        model_store=mock_model_store,
        strategy_store=mock_strategy_store,
        earnings_store=DummyEarningsStore(),
        raw_writer=_FakeRawWriter(),
        fail_on_validation_error=False,
    )
    writer = DataStoreMarketDataWriter(ds)

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
    n = writer.write(
        dataset_id="bars_ds",
        schema="bars",
        instrument_id="SPY.NYSE",
        df=df,
    )

    assert n == 2
    # SUCCESS event should appear
    assert any(status == "success" for _ds, status in reg.events)
    assert len(reg.watermarks) == 1


def test_sql_market_data_writer_maps_mbp_columns(patch_engine_manager) -> None:
    engine = create_engine("sqlite://")
    metadata = MetaData()
    Table(
        "market_data",
        metadata,
        Column("instrument_id", String(100), nullable=False),
        Column("ts_event", BigInteger, nullable=False),
        Column("ts_init", BigInteger, nullable=False),
        Column("bid", Float),
        Column("ask", Float),
        Column("bid_size", Float),
        Column("ask_size", Float),
    )
    metadata.create_all(engine)

    with patch_engine_manager(engine=engine):
        writer = SqlMarketDataWriter(connection_string="sqlite://", table_name="market_data")
        frame = pd.DataFrame(
            {
                "ts_event": [1, 2],
                "ts_init": [1, 2],
                "bid_px": [100.0, None],
                "ask_px": [101.0, None],
                "bid_sz": [10.0, None],
                "ask_sz": [12.0, None],
                "bid_px_00": [None, 200.0],
                "ask_px_00": [None, 201.0],
                "bid_sz_00": [None, 20.0],
                "ask_sz_00": [None, 22.0],
            },
        )
        writer.write(
            dataset_id="EQUS.MINI_MBP1",
            schema="mbp-1",
            instrument_id="SPY.XNAS",
            df=frame,
        )

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT ts_event, bid, ask, bid_size, ask_size "
                "FROM market_data ORDER BY ts_event"
            ),
        ).fetchall()

    assert rows == [
        (1, 100.0, 101.0, 10.0, 12.0),
        (2, 200.0, 201.0, 20.0, 22.0),
    ]
    engine.dispose()


def test_sql_market_data_writer_filters_quote_sentinel_price(patch_engine_manager) -> None:
    engine = create_engine("sqlite://")
    metadata = MetaData()
    Table(
        "market_data",
        metadata,
        Column("instrument_id", String(100), nullable=False),
        Column("ts_event", BigInteger, nullable=False),
        Column("ts_init", BigInteger, nullable=False),
        Column("bid", Float),
        Column("ask", Float),
        Column("bid_size", Float),
        Column("ask_size", Float),
    )
    metadata.create_all(engine)

    sentinel = MarketDataTableConfig().quote_sentinel_price
    assert sentinel is not None

    with patch_engine_manager(engine=engine):
        writer = SqlMarketDataWriter(connection_string="sqlite://", table_name="market_data")
        frame = pd.DataFrame(
            {
                "ts_event": [1, 2],
                "ts_init": [1, 2],
                "bid": [sentinel, 100.0],
                "ask": [101.0, sentinel],
                "bid_size": [10.0, 11.0],
                "ask_size": [12.0, 13.0],
            },
        )
        writer.write(
            dataset_id="EQUS.MINI_QUOTES",
            schema="quotes",
            instrument_id="SPY.XNAS",
            df=frame,
        )

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT ts_event, bid, ask "
                "FROM market_data ORDER BY ts_event"
            ),
        ).fetchall()

    assert rows == [
        (1, None, 101.0),
        (2, 100.0, None),
    ]
    engine.dispose()
