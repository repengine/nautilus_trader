#!/usr/bin/env python3

from __future__ import annotations

import time
from pathlib import Path

from typing import Any, cast

import pandas as pd
import pytest

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
from ml.stores.data_store import DataStore
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
) -> None:
    # Setup DataStore with stub registry and mock stores, plus a fake raw writer
    reg = _StubRegistry(_manifest("bars_ds"), _contract("bars_ds"))
    ds = cast(Any, DataStore)(
        connection_string="sqlite:///:memory:",
        registry=reg,
        feature_store=mock_feature_store,
        model_store=mock_model_store,
        strategy_store=mock_strategy_store,
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
