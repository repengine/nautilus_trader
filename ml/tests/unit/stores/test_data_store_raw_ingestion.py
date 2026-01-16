#!/usr/bin/env python3

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from ml._imports import HAS_POLARS, pl, pd
from ml.config.events import EventStatus, Source, Stage
from ml.ml_types import DataFrameLike
from ml.registry.dataclasses import (
    DataContract,
    DatasetManifest,
    DatasetType,
    StorageKind,
    ValidationRule,
    ValidationRuleType,
    QualityFlag,
)
from ml.registry.protocols import RegistryProtocol
from ml.features.earnings.store import DummyEarningsStore
from ml.stores.data_store_facade import DataStore
from ml.stores.io_raw import RawIngestionWriterProtocol, RawReaderProtocol


@dataclass
class _RecordedEvent:
    dataset_id: str
    status: EventStatus
    stage: Stage
    source: Source
    ts_max: int
    count: int


class _TestRegistry(RegistryProtocol):
    """
    Lightweight stub for RegistryProtocol used in unit tests.
    """

    def __init__(self, manifest: DatasetManifest, contract: DataContract) -> None:
        self._manifest = manifest
        self._contract = contract
        self.emitted: list[_RecordedEvent] = []
        self.watermarks: list[tuple[str, int]] = []  # (instrument_id, last_success_ns)

    # Protocol methods
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
        self.emitted.append(
            _RecordedEvent(
                dataset_id=dataset_id,
                status=status,
                stage=stage,
                source=source,
                ts_max=ts_max,
                count=count,
            ),
        )

    def update_watermark(
        self,
        dataset_id: str,
        instrument_id: str,
        source: Source,
        last_success_ns: int,
        count: int,
        completeness_pct: float,
    ) -> None:
        self.watermarks.append((instrument_id, last_success_ns))

    # Data access APIs
    def get_manifest(self, dataset_id: str) -> DatasetManifest:
        return self._manifest

    def get_contract(self, dataset_id: str) -> DataContract:
        return self._contract

    def register_dataset(self, manifest: DatasetManifest) -> str:
        return manifest.dataset_id

    def update_manifest(self, dataset_id: str, changes: dict[str, object]) -> None:
        return None


class _FakeRawWriter(RawIngestionWriterProtocol):
    def __init__(self, fail: bool = False, zero: bool = False) -> None:
        self.fail = fail
        self.zero = zero
        self.last_args: tuple[DatasetType, int] | None = None

    def write(
        self,
        *,
        dataset_type: DatasetType,
        data: DataFrameLike | list[dict[str, object]],
    ) -> int:
        # Record call
        n = len(data) if hasattr(data, "__len__") else 0
        self.last_args = (dataset_type, n)
        if self.fail:
            raise RuntimeError("writer failed")
        if self.zero:
            return 0
        return n


class _FakeRawReader(RawReaderProtocol):
    def read_range(
        self,
        *,
        dataset_type: DatasetType,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> DataFrameLike:
        if HAS_POLARS and pl is not None:
            return cast(
                DataFrameLike,
                pl.DataFrame(
                    {
                        "instrument_id": [instrument_id],
                        "ts_event": [start_ns],
                        "ts_init": [start_ns + 1],
                    },
                ),
            )
        if pd is not None:
            return cast(
                DataFrameLike,
                pd.DataFrame(
                    {
                        "instrument_id": [instrument_id],
                        "ts_event": [start_ns],
                        "ts_init": [start_ns + 1],
                    },
                ),
            )
        raise RuntimeError("No dataframe library available for raw reader")


def _make_manifest(dataset_id: str, dataset_type: DatasetType) -> DatasetManifest:
    return DatasetManifest(
        dataset_id=dataset_id,
        dataset_type=dataset_type,
        storage_kind=StorageKind.PARQUET,
        location=str(Path("/tmp")),
        partitioning={"by": ["date"]},
        retention_days=30,
        schema={
            "instrument_id": "str",
            "ts_event": "int64",
            "ts_init": "int64",
            "value": "float64",
        },
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["instrument_id", "ts_event"],
        schema_hash="",  # computed in __post_init__
        constraints={},
        lineage=[],
        pipeline_signature="test",
        version="1.0.0",
        created_at=time.time_ns(),
        last_modified=time.time_ns(),
        metadata={},
    )


def _make_contract(dataset_id: str) -> DataContract:
    return DataContract(
        contract_id=f"{dataset_id}_contract",
        dataset_id=dataset_id,
        version="1.0.0",
        validation_rules=[
            ValidationRule(
                rule_type=ValidationRuleType.MONOTONICITY,
                field_name="ts_event",
                parameters={"direction": "increasing"},
                severity=QualityFlag.FAIL,
                description="Timestamps must be increasing",
            ),
        ],
        enforcement_mode="lenient",
    )


def _make_df(instrument: str, n: int = 3) -> DataFrameLike:
    rows = [
        {
            "instrument_id": instrument,
            "ts_event": 1_000 + i,
            "ts_init": 2_000 + i,
            "value": float(i),
        }
        for i in range(n)
    ]
    if HAS_POLARS and pl is not None:
        return cast(DataFrameLike, pl.DataFrame(rows))
    if pd is not None:
        return cast(DataFrameLike, pd.DataFrame(rows))
    raise RuntimeError("No dataframe library available for test")


@pytest.mark.unit
def test_raw_write_without_writer_emits_partial_and_no_watermark(
    mock_feature_store: Any,
    mock_model_store: Any,
    mock_strategy_store: Any,
) -> None:
    manifest = _make_manifest("bars_X", DatasetType.BARS)
    contract = _make_contract("bars_X")
    reg = _TestRegistry(manifest, contract)

    store = DataStore(
        connection_string="sqlite:///:memory:",
        registry=reg,  # use stub
        feature_store=mock_feature_store,
        model_store=mock_model_store,
        strategy_store=mock_strategy_store,
        earnings_store=DummyEarningsStore(),
        fail_on_validation_error=False,
    )

    df = _make_df("EUR/USD", n=5)
    event = store.write_ingestion(
        dataset_id="bars_X",
        records=df,
        source=Source.BACKFILL.value,
        run_id="run1",
        instrument_id="EUR/USD",
    )

    # Event object reports partial and no watermark updates
    assert event.status == EventStatus.PARTIAL.value
    # Registry recorded PARTIAL event and zero watermarks
    assert any(e.status.value == EventStatus.PARTIAL.value for e in reg.emitted)
    assert reg.watermarks == []


@pytest.mark.unit
def test_raw_write_with_writer_emits_success_and_watermark(
    mock_feature_store: Any,
    mock_model_store: Any,
    mock_strategy_store: Any,
) -> None:
    manifest = _make_manifest("bars_Y", DatasetType.BARS)
    contract = _make_contract("bars_Y")
    reg = _TestRegistry(manifest, contract)

    writer = _FakeRawWriter()
    store = DataStore(
        connection_string="sqlite:///:memory:",
        registry=reg,
        raw_writer=writer,
        feature_store=mock_feature_store,
        model_store=mock_model_store,
        strategy_store=mock_strategy_store,
        earnings_store=DummyEarningsStore(),
        fail_on_validation_error=False,
    )

    df = _make_df("EUR/USD", n=4)
    event = store.write_ingestion(
        dataset_id="bars_Y",
        records=df,
        source=Source.HISTORICAL.value,
        run_id="run2",
        instrument_id="EUR/USD",
    )

    assert event.status == EventStatus.SUCCESS.value
    assert any(e.status.value == EventStatus.SUCCESS.value for e in reg.emitted)
    assert len(reg.watermarks) == 1


@pytest.mark.unit
def test_raw_read_with_reader_respects_range(
    mock_feature_store: Any,
    mock_model_store: Any,
    mock_strategy_store: Any,
) -> None:
    manifest = _make_manifest("bars_Z", DatasetType.BARS)
    contract = _make_contract("bars_Z")
    reg = _TestRegistry(manifest, contract)
    reader = _FakeRawReader()
    store = DataStore(
        connection_string="sqlite:///:memory:",
        registry=reg,
        raw_reader=reader,
        feature_store=mock_feature_store,
        model_store=mock_model_store,
        strategy_store=mock_strategy_store,
        earnings_store=DummyEarningsStore(),
        fail_on_validation_error=False,
    )

    df_any = store.read_range(
        dataset_id="bars_Z",
        instrument_id="SPY.NYSE",
        start_ns=111,
        end_ns=222,
    )

    # Verify returned structure contains required fields
    if HAS_POLARS:
        assert hasattr(df_any, "columns") and set(["instrument_id", "ts_event"]).issubset(
            set(df_any.columns),
        )
    else:
        assert isinstance(df_any, list) and df_any and "instrument_id" in df_any[0]
